"""
Paper Method 到 SVG 图标替换完整流程 (Label 模式增强版 + Box合并 + 多Prompt支持)

支持的 API Provider：
- openrouter: OpenRouter API (https://openrouter.ai/api/v1)
- bianxie: Bianxie API (https://api.bianxie.ai/v1) - 使用 OpenAI SDK

占位符模式 (--placeholder_mode):
- none: 无特殊样式（默认黑色边框）
- box: 传入 boxlib 坐标给 LLM
- label: 灰色填充+黑色边框+序号标签 <AF>01, <AF>02...（推荐）

SAM3 多Prompt支持 (--sam_prompt):
- 支持逗号分隔的多个text prompt
- 例如: "icon,diagram,arrow,chart"
- 对每个prompt分别检测，然后合并去重结果
- boxlib.json 会记录每个box的来源prompt

Box合并功能 (--merge_threshold):
- 对SAM3检测到的重叠box进行合并去重
- 重叠比例 = 交集面积 / 较小box面积
- 默认阈值0.9，设为0表示不合并
- 跨prompt检测结果也会自动去重

流程：
1. 输入 paper method 文本，调用 Gemini 生成学术风格图片 -> figure.png
2. SAM3 分割图片，用灰色填充+黑色边框+序号标记 -> samed.png + boxlib.json
   2.1 支持多个text prompts分别检测
   2.2 合并重叠的boxes（可选，通过 --merge_threshold 控制）
3. 裁切分割区域 + RMBG2 去背景 -> icons/icon_AF01_nobg.png, icon_AF02_nobg.png...
4. 多模态调用 Gemini 生成 SVG（占位符样式与 samed.png 一致）-> template.svg
4.5. SVG 语法验证（lxml）+ LLM 修复
4.6. LLM 优化 SVG 模板（位置和样式对齐）-> optimized_template.svg
     可通过 --optimize_iterations 参数控制迭代次数（0 表示跳过优化）
4.7. 坐标系对齐：比较 figure.png 与 SVG 尺寸，计算缩放因子
5. 根据序号匹配，将透明图标替换到 SVG 占位符中 -> final.svg

使用方法：
    # 使用 Bianxie + label 模式（默认）
    python iou_autofigure.py --method_file paper_method.txt --output_dir ./output --api_key "your-key"

    # 使用 OpenRouter
    python iou_autofigure.py --method_file paper_method.txt --output_dir ./output --api_key "sk-or-v1-xxx" --provider openrouter

    # 使用 box 模式（传入坐标）
    python iou_autofigure.py --method_file paper_method.txt --output_dir ./output --placeholder_mode box

    # 使用多个 SAM3 prompts 检测
    python iou_autofigure.py --method_file paper_method.txt --output_dir ./output --sam_prompt "icon,diagram,arrow"

    # 跳过步骤 4.6 优化（设置迭代次数为 0）
    python iou_autofigure.py --method_file paper_method.txt --output_dir ./output --optimize_iterations 0

    # 设置步骤 4.6 优化迭代 3 次
    python iou_autofigure.py --method_file paper_method.txt --output_dir ./output --optimize_iterations 3

    # 自定义 box 合并阈值（0.8）
    python iou_autofigure.py --method_file paper_method.txt --output_dir ./output --merge_threshold 0.8

    # 禁用 box 合并
    python iou_autofigure.py --method_file paper_method.txt --output_dir ./output --merge_threshold 0
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal

import requests
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


# ============================================================================
# Provider 配置
# ============================================================================

PROVIDER_CONFIGS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_image_model": "google/gemini-3-pro-image-preview",
        "default_svg_model": "google/gemini-3-pro-preview",
    },
    "bianxie": {
        "base_url": "https://api.bianxie.ai/v1",
        "default_image_model": "gemini-3-pro-image-preview",
        "default_svg_model": "gemini-3-pro-preview",
    },
}

ProviderType = Literal["openrouter", "bianxie"]
PlaceholderMode = Literal["none", "box", "label"]

# SAM3 API config
SAM3_FAL_API_URL = "https://fal.run/fal-ai/sam-3/image"
SAM3_ROBOFLOW_API_URL = "https://serverless.roboflow.com/sam3/concept_segment"
SAM3_API_TIMEOUT = 300

# Step 1 reference image settings (overridden by CLI)
USE_REFERENCE_IMAGE = False
REFERENCE_IMAGE_PATH: Optional[str] = None


# ============================================================================
# 统一的 LLM 调用接口
# ============================================================================

def call_llm_text(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """
    统一的文本 LLM 调用接口

    Args:
        prompt: 文本提示
        api_key: API Key
        model: 模型名称
        base_url: API base URL
        provider: API 提供商
        reference_image: 参考图片（可选）
        max_tokens: 最大输出 token 数
        temperature: 温度参数

    Returns:
        LLM 响应文本
    """
    if provider == "bianxie":
        return _call_bianxie_text(prompt, api_key, model, base_url, max_tokens, temperature)
    else:  # openrouter
        return _call_openrouter_text(prompt, api_key, model, base_url, max_tokens, temperature)


def call_llm_multimodal(
    contents: List[Any],
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """
    统一的多模态 LLM 调用接口

    Args:
        contents: 内容列表（字符串或 PIL Image）
        api_key: API Key
        model: 模型名称
        base_url: API base URL
        provider: API 提供商
        max_tokens: 最大输出 token 数
        temperature: 温度参数

    Returns:
        LLM 响应文本
    """
    if provider == "bianxie":
        return _call_bianxie_multimodal(contents, api_key, model, base_url, max_tokens, temperature)
    else:  # openrouter
        return _call_openrouter_multimodal(contents, api_key, model, base_url, max_tokens, temperature)


def call_llm_image_generation(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    reference_image: Optional[Image.Image] = None,
) -> Optional[Image.Image]:
    """
    统一的图像生成 LLM 调用接口

    Args:
        prompt: 文本提示
        api_key: API Key
        model: 模型名称
        base_url: API base URL
        provider: API 提供商

    Returns:
        生成的 PIL Image，失败返回 None
    """
    if provider == "bianxie":
        return _call_bianxie_image_generation(prompt, api_key, model, base_url, reference_image)
    else:  # openrouter
        return _call_openrouter_image_generation(prompt, api_key, model, base_url, reference_image)


# ============================================================================
# Bianxie Provider 实现 (使用 OpenAI SDK)
# ============================================================================

def _call_bianxie_text(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """使用 OpenAI SDK 调用 Bianxie 文本接口"""
    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)

        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return completion.choices[0].message.content if completion and completion.choices else None
    except Exception as e:
        print(f"[Bianxie] API 调用失败: {e}")
        raise


def _call_bianxie_multimodal(
    contents: List[Any],
    api_key: str,
    model: str,
    base_url: str,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """使用 OpenAI SDK 调用 Bianxie 多模态接口"""
    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)

        message_content: List[Dict[str, Any]] = []
        for part in contents:
            if isinstance(part, str):
                message_content.append({"type": "text", "text": part})
            elif isinstance(part, Image.Image):
                buf = io.BytesIO()
                part.save(buf, format='PNG')
                image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                })

        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message_content}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return completion.choices[0].message.content if completion and completion.choices else None
    except Exception as e:
        print(f"[Bianxie] 多模态 API 调用失败: {e}")
        raise


def _call_bianxie_image_generation(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    reference_image: Optional[Image.Image] = None,
) -> Optional[Image.Image]:
    """使用 OpenAI SDK 调用 Bianxie 图像生成接口"""
    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)

        if reference_image is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            buf = io.BytesIO()
            reference_image.save(buf, format='PNG')
            image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            message_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ]
            messages = [{"role": "user", "content": message_content}]

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
        )

        content = completion.choices[0].message.content if completion and completion.choices else None

        if not content:
            return None

        # Bianxie 返回 Markdown 格式的图片: ![text](data:image/png;base64,...)
        pattern = r'data:image/(png|jpeg|jpg|webp);base64,([A-Za-z0-9+/=]+)'
        match = re.search(pattern, content)

        if match:
            image_base64 = match.group(2)
            image_data = base64.b64decode(image_base64)
            return Image.open(io.BytesIO(image_data))

        return None
    except Exception as e:
        print(f"[Bianxie] 图像生成 API 调用失败: {e}")
        raise


# ============================================================================
# OpenRouter Provider 实现 (使用 requests)
# ============================================================================

def _get_openrouter_headers(api_key: str) -> dict:
    """获取 OpenRouter 请求头"""
    return {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'HTTP-Referer': 'https://localhost',
        'X-Title': 'MethodToSVG'
    }


def _get_openrouter_api_url(base_url: str) -> str:
    """获取 OpenRouter API URL"""
    if not base_url.endswith('/chat/completions'):
        if base_url.endswith('/'):
            return base_url + 'chat/completions'
        else:
            return base_url + '/chat/completions'
    return base_url


def _call_openrouter_text(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """使用 requests 调用 OpenRouter 文本接口"""
    api_url = _get_openrouter_api_url(base_url)
    headers = _get_openrouter_headers(api_key)

    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'temperature': temperature,
        'stream': False
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=300)

    if response.status_code != 200:
        raise Exception(f'OpenRouter API 错误: {response.status_code} - {response.text[:500]}')

    result = response.json()

    if 'error' in result:
        error_msg = result.get('error', {})
        if isinstance(error_msg, dict):
            error_msg = error_msg.get('message', str(error_msg))
        raise Exception(f'OpenRouter API 错误: {error_msg}')

    choices = result.get('choices', [])
    if not choices:
        return None

    return choices[0].get('message', {}).get('content', '')


def _call_openrouter_multimodal(
    contents: List[Any],
    api_key: str,
    model: str,
    base_url: str,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """使用 requests 调用 OpenRouter 多模态接口"""
    api_url = _get_openrouter_api_url(base_url)
    headers = _get_openrouter_headers(api_key)

    message_content: List[Dict[str, Any]] = []
    for part in contents:
        if isinstance(part, str):
            message_content.append({"type": "text", "text": part})
        elif isinstance(part, Image.Image):
            buf = io.BytesIO()
            part.save(buf, format='PNG')
            image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"}
            })

    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': message_content}],
        'max_tokens': max_tokens,
        'temperature': temperature,
        'stream': False
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=300)

    if response.status_code != 200:
        raise Exception(f'OpenRouter API 错误: {response.status_code} - {response.text[:500]}')

    result = response.json()

    if 'error' in result:
        error_msg = result.get('error', {})
        if isinstance(error_msg, dict):
            error_msg = error_msg.get('message', str(error_msg))
        raise Exception(f'OpenRouter API 错误: {error_msg}')

    choices = result.get('choices', [])
    if not choices:
        return None

    return choices[0].get('message', {}).get('content', '')


def _call_openrouter_image_generation(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    reference_image: Optional[Image.Image] = None,
) -> Optional[Image.Image]:
    """使用 requests 调用 OpenRouter 图像生成接口"""
    api_url = _get_openrouter_api_url(base_url)
    headers = _get_openrouter_headers(api_key)

    if reference_image is None:
        messages = [{'role': 'user', 'content': prompt}]
    else:
        buf = io.BytesIO()
        reference_image.save(buf, format='PNG')
        image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        message_content: List[Dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ]
        messages = [{'role': 'user', 'content': message_content}]

    payload = {
        'model': model,
        'messages': messages,
        'modalities': ['image', 'text'],
        'stream': False
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=300)

    if response.status_code != 200:
        raise Exception(f'OpenRouter API 错误: {response.status_code} - {response.text[:500]}')

    result = response.json()

    if 'error' in result:
        error_msg = result.get('error', {})
        if isinstance(error_msg, dict):
            error_msg = error_msg.get('message', str(error_msg))
        raise Exception(f'OpenRouter API 错误: {error_msg}')

    # OpenRouter 返回图片在 message.images[] 数组中
    choices = result.get('choices', [])
    if not choices:
        return None

    message = choices[0].get('message', {})
    images = message.get('images', [])

    if images and len(images) > 0:
        first_image = images[0]

        if isinstance(first_image, dict):
            image_url_obj = first_image.get('image_url', {})
            if isinstance(image_url_obj, dict):
                image_url = image_url_obj.get('url', '')
            else:
                image_url = str(image_url_obj)
        else:
            image_url = str(first_image)

        if image_url.startswith('data:image/'):
            pattern = r'data:image/(png|jpeg|jpg|webp);base64,(.+)'
            match = re.match(pattern, image_url)
            if match:
                image_base64 = match.group(2)
                image_data = base64.b64decode(image_base64)
                return Image.open(io.BytesIO(image_data))

    return None


# ============================================================================
# 步骤一：调用 LLM 生成图片
# ============================================================================

def generate_figure_from_method(
    method_text: str,
    output_path: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    use_reference_image: Optional[bool] = None,
    reference_image_path: Optional[str] = None,
) -> str:
    """
    使用 LLM 生成学术风格图片

    Args:
        method_text: Paper method 文本内容
        output_path: 输出图片路径
        api_key: API Key
        model: 生图模型名称
        base_url: API base URL
        provider: API 提供商
        use_reference_image: 是否使用参考图片（None 则使用全局设置）
        reference_image_path: 参考图片路径（None 则使用全局设置）

    Returns:
        生成的图片路径
    """
    print("=" * 60)
    print("步骤一：使用 LLM 生成学术风格图片")
    print("=" * 60)
    print(f"Provider: {provider}")
    print(f"模型: {model}")

    if use_reference_image is None:
        use_reference_image = USE_REFERENCE_IMAGE
    if reference_image_path is None:
        reference_image_path = REFERENCE_IMAGE_PATH
    if reference_image_path:
        use_reference_image = True

    reference_image = None
    if use_reference_image:
        if not reference_image_path:
            raise ValueError("启用参考图模式但未提供 reference_image_path")
        reference_image = Image.open(reference_image_path)
        print(f"参考图片: {reference_image_path}")

    if use_reference_image:
        prompt = f"""Generate a figure to visualize the method described below.

You should closely imitate the visual (artistic) style of the reference figure I provide, focusing only on aesthetic aspects, NOT on layout or structure.

Specifically, match:
- overall visual tone and mood
- illustration abstraction level
- line style
- color usage
- shading style
- icon and shape style
- arrow and connector aesthetics
- typography feel

The content structure, number of components, and layout may differ freely.
Only the visual style should be consistent.

The goal is that the figure looks like it was drawn by the same illustrator using the same visual design language as the reference figure.

Below is the method section of the paper:
\"\"\"
{method_text}
\"\"\""""
    else:
        prompt = f"""Generate a professional academic journal style figure for the paper below so as to visualize the method it proposes, below is the method section of this paper:

{method_text}

The figure should be engaging and using academic journal style with cute characters."""

    print(f"发送请求到: {base_url}")

    img = call_llm_image_generation(
        prompt=prompt,
        api_key=api_key,
        model=model,
        base_url=base_url,
        provider=provider,
        reference_image=reference_image,
    )

    if img is None:
        raise Exception('API 响应中没有找到图片')

    # 确保输出目录存在
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 转换为 PNG 保存
    img.save(str(output_path), format='PNG')
    print(f"图片已保存: {output_path}")
    return str(output_path)


# ============================================================================
# 步骤二：SAM3 分割 + Box合并 + 灰色填充+黑色边框+序号标记
# ============================================================================

def get_label_font(box_width: int, box_height: int) -> ImageFont.FreeTypeFont:
    """
    根据 box 尺寸动态计算合适的字体大小

    Args:
        box_width: 矩形宽度
        box_height: 矩形高度

    Returns:
        PIL ImageFont 对象
    """
    # 字体大小为 box 短边的 1/4，最小 12，最大 48
    min_dim = min(box_width, box_height)
    font_size = max(12, min(48, min_dim // 4))

    # 尝试加载字体
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "C:/Windows/Fonts/arial.ttf",  # Windows
    ]

    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, font_size)
        except (IOError, OSError):
            continue

    # 回退到默认字体
    try:
        return ImageFont.load_default()
    except:
        return None


# ============================================================================
# Box 合并辅助函数
# ============================================================================

def calculate_overlap_ratio(box1: dict, box2: dict) -> float:
    """
    计算两个box的重叠比例

    Args:
        box1: 第一个box，包含 x1, y1, x2, y2
        box2: 第二个box，包含 x1, y1, x2, y2

    Returns:
        重叠比例 = 交集面积 / 较小box面积
    """
    # 计算交集区域
    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])

    # 无交集
    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # 计算各自面积
    area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])

    if area1 == 0 or area2 == 0:
        return 0.0

    # 返回交集占较小box的比例
    return intersection / min(area1, area2)


def merge_two_boxes(box1: dict, box2: dict) -> dict:
    """
    合并两个box为最小包围矩形

    Args:
        box1: 第一个box
        box2: 第二个box

    Returns:
        合并后的box（最小包围矩形）
    """
    merged = {
        "x1": min(box1["x1"], box2["x1"]),
        "y1": min(box1["y1"], box2["y1"]),
        "x2": max(box1["x2"], box2["x2"]),
        "y2": max(box1["y2"], box2["y2"]),
        "score": max(box1.get("score", 0), box2.get("score", 0)),  # 保留较高置信度
    }
    # 合并 prompt 字段（如果存在）
    prompt1 = box1.get("prompt", "")
    prompt2 = box2.get("prompt", "")
    if prompt1 and prompt2:
        if prompt1 == prompt2:
            merged["prompt"] = prompt1
        else:
            # 合并不同的 prompts，保留置信度更高的那个
            if box1.get("score", 0) >= box2.get("score", 0):
                merged["prompt"] = prompt1
            else:
                merged["prompt"] = prompt2
    elif prompt1:
        merged["prompt"] = prompt1
    elif prompt2:
        merged["prompt"] = prompt2
    return merged


def merge_overlapping_boxes(boxes: list, overlap_threshold: float = 0.9) -> list:
    """
    迭代合并重叠的boxes

    Args:
        boxes: box列表，每个box包含 x1, y1, x2, y2, score
        overlap_threshold: 重叠阈值，超过此值则合并（默认0.9）

    Returns:
        合并后的box列表，重新编号
    """
    if overlap_threshold <= 0 or len(boxes) <= 1:
        return boxes

    # 复制列表避免修改原数据
    working_boxes = [box.copy() for box in boxes]

    merged = True
    iteration = 0
    while merged:
        merged = False
        iteration += 1
        n = len(working_boxes)

        for i in range(n):
            if merged:
                break
            for j in range(i + 1, n):
                ratio = calculate_overlap_ratio(working_boxes[i], working_boxes[j])
                if ratio >= overlap_threshold:
                    # 合并 box_i 和 box_j
                    new_box = merge_two_boxes(working_boxes[i], working_boxes[j])
                    # 移除原有两个box，添加合并后的box
                    working_boxes = [
                        working_boxes[k] for k in range(n) if k != i and k != j
                    ]
                    working_boxes.append(new_box)
                    merged = True
                    print(f"    迭代 {iteration}: 合并 box {i} 和 box {j} (重叠比例: {ratio:.2f})")
                    break

    # 重新编号
    result = []
    for idx, box in enumerate(working_boxes):
        result_box = {
            "id": idx,
            "label": f"<AF>{idx + 1:02d}",
            "x1": box["x1"],
            "y1": box["y1"],
            "x2": box["x2"],
            "y2": box["y2"],
            "score": box.get("score", 0),
        }
        # 保留 prompt 字段（如果存在）
        if "prompt" in box:
            result_box["prompt"] = box["prompt"]
        result.append(result_box)

    return result


def _get_fal_api_key(sam_api_key: Optional[str]) -> str:
    key = sam_api_key or os.environ.get("FAL_KEY")
    if not key:
        raise ValueError("SAM3 fal.ai API key missing: set --sam_api_key or FAL_KEY environment variable")
    return key


def _get_roboflow_api_key(sam_api_key: Optional[str]) -> str:
    key = sam_api_key or os.environ.get("ROBOFLOW_API_KEY") or os.environ.get("API_KEY")
    if not key:
        raise ValueError(
            "SAM3 Roboflow API key missing: set --sam_api_key or ROBOFLOW_API_KEY/API_KEY environment variable"
        )
    return key


def _image_to_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_b64}"


def _image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _cxcywh_norm_to_xyxy(box: list | tuple, width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    if not box or len(box) < 4:
        return None
    try:
        cx, cy, bw, bh = [float(v) for v in box[:4]]
    except (TypeError, ValueError):
        return None

    cx *= width
    cy *= height
    bw *= width
    bh *= height

    x1 = int(round(cx - bw / 2.0))
    y1 = int(round(cy - bh / 2.0))
    x2 = int(round(cx + bw / 2.0))
    y2 = int(round(cy + bh / 2.0))

    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _polygon_to_bbox(points: list, width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    xs: list[float] = []
    ys: list[float] = []

    for pt in points:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        try:
            x = float(pt[0])
            y = float(pt[1])
        except (TypeError, ValueError):
            continue
        xs.append(x)
        ys.append(y)

    if not xs or not ys:
        return None

    x1 = int(round(min(xs)))
    y1 = int(round(min(ys)))
    x2 = int(round(max(xs)))
    y2 = int(round(max(ys)))

    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _extract_sam3_api_detections(response_json: dict, image_size: tuple[int, int]) -> list[dict]:
    width, height = image_size
    detections: list[dict] = []

    metadata = response_json.get("metadata") if isinstance(response_json, dict) else None
    if isinstance(metadata, list) and metadata:
        for item in metadata:
            if not isinstance(item, dict):
                continue
            box = item.get("box")
            xyxy = _cxcywh_norm_to_xyxy(box, width, height)
            if not xyxy:
                continue
            score = item.get("score")
            detections.append(
                {"x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3], "score": score}
            )
        return detections

    boxes = response_json.get("boxes") if isinstance(response_json, dict) else None
    scores = response_json.get("scores") if isinstance(response_json, dict) else None
    if isinstance(boxes, list) and boxes:
        scores_list = scores if isinstance(scores, list) else []
        for idx, box in enumerate(boxes):
            xyxy = _cxcywh_norm_to_xyxy(box, width, height)
            if not xyxy:
                continue
            score = scores_list[idx] if idx < len(scores_list) else None
            detections.append(
                {"x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3], "score": score}
            )

    return detections


def _extract_roboflow_detections(response_json: dict, image_size: tuple[int, int]) -> list[dict]:
    width, height = image_size
    detections: list[dict] = []

    prompt_results = response_json.get("prompt_results") if isinstance(response_json, dict) else None
    if not isinstance(prompt_results, list):
        return detections

    for prompt_result in prompt_results:
        if not isinstance(prompt_result, dict):
            continue
        predictions = prompt_result.get("predictions", [])
        if not isinstance(predictions, list):
            continue
        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue
            confidence = prediction.get("confidence")
            masks = prediction.get("masks", [])
            if not isinstance(masks, list):
                continue
            for mask in masks:
                points = []
                if isinstance(mask, list) and mask:
                    if isinstance(mask[0], (list, tuple)) and len(mask[0]) >= 2 and isinstance(
                        mask[0][0], (int, float)
                    ):
                        points = mask
                    elif isinstance(mask[0], (list, tuple)):
                        for sub in mask:
                            if isinstance(sub, (list, tuple)) and len(sub) >= 2 and isinstance(
                                sub[0], (int, float)
                            ):
                                points.append(sub)
                            elif isinstance(sub, (list, tuple)) and sub and isinstance(
                                sub[0], (list, tuple)
                            ):
                                for pt in sub:
                                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                                        points.append(pt)
                if not points:
                    continue
                xyxy = _polygon_to_bbox(points, width, height)
                if not xyxy:
                    continue
                detections.append(
                    {
                        "x1": xyxy[0],
                        "y1": xyxy[1],
                        "x2": xyxy[2],
                        "y2": xyxy[3],
                        "score": confidence,
                    }
                )

    return detections


def _call_sam3_api(
    image_data_uri: str,
    prompt: str,
    api_key: str,
    max_masks: int,
) -> dict:
    headers = {
        "Authorization": f"Key {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "image_url": image_data_uri,
        "prompt": prompt,
        "apply_mask": False,
        "return_multiple_masks": True,
        "max_masks": max_masks,
        "include_scores": True,
        "include_boxes": True,
    }
    response = requests.post(SAM3_FAL_API_URL, headers=headers, json=payload, timeout=SAM3_API_TIMEOUT)
    if response.status_code != 200:
        raise Exception(f"SAM3 API 错误: {response.status_code} - {response.text[:500]}")
    result = response.json()
    if isinstance(result, dict) and "error" in result:
        raise Exception(f"SAM3 API 错误: {result.get('error')}")
    return result


def _call_sam3_roboflow_api(
    image_base64: str,
    prompt: str,
    api_key: str,
    min_score: float,
) -> dict:
    payload = {
        "image": {"type": "base64", "value": image_base64},
        "prompts": [{"type": "text", "text": prompt}],
        "format": "polygon",
        "output_prob_thresh": min_score,
    }
    url = f"{SAM3_ROBOFLOW_API_URL}?api_key={api_key}"
    response = requests.post(url, json=payload, timeout=SAM3_API_TIMEOUT)
    if response.status_code != 200:
        raise Exception(f"SAM3 Roboflow API 错误: {response.status_code} - {response.text[:500]}")
    result = response.json()
    if isinstance(result, dict) and "error" in result:
        raise Exception(f"SAM3 Roboflow API 错误: {result.get('error')}")
    return result


def segment_with_sam3(
    image_path: str,
    output_dir: str,
    text_prompts: str = "icon",
    min_score: float = 0.5,
    merge_threshold: float = 0.9,
    sam_backend: Literal["local", "fal", "roboflow", "api"] = "local",
    sam_api_key: Optional[str] = None,
    sam_max_masks: int = 32,
) -> tuple[str, str, list]:
    """
    使用 SAM3 分割图片，用灰色填充+黑色边框+序号标记，生成 boxlib.json

    占位符样式：
    - 灰色填充 (#808080)
    - 黑色边框 (width=3)
    - 白色居中序号标签 (<AF>01, <AF>02, ...)

    Args:
        image_path: 输入图片路径
        output_dir: 输出目录
        text_prompts: SAM3 文本提示，支持逗号分隔的多个prompt（如 "icon,diagram,arrow"）
        min_score: 最低置信度阈值
        merge_threshold: Box合并阈值，重叠比例超过此值则合并（0表示不合并，默认0.9）

    Returns:
        (samed_path, boxlib_path, valid_boxes)
    """
    print("\n" + "=" * 60)
    print("步骤二：SAM3 分割 + 灰色填充+黑色边框+序号标记")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path)
    original_size = image.size
    print(f"原图尺寸: {original_size[0]} x {original_size[1]}")

    # 解析多个 prompts（支持逗号分隔）
    prompt_list = [p.strip() for p in text_prompts.split(",") if p.strip()]
    print(f"使用的 prompts: {prompt_list}")

    # 对每个 prompt 分别检测并收集结果
    all_detected_boxes = []
    total_detected = 0

    backend = sam_backend
    if backend == "api":
        backend = "fal"

    if backend == "local":
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        import sam3

        sam3_dir = Path(sam3.__path__[0]) if hasattr(sam3, '__path__') else Path(sam3.__file__).parent
        bpe_path = sam3_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        if not bpe_path.exists():
            bpe_path = None
            print("警告: 未找到 bpe 文件，使用默认路径")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        model = build_sam3_image_model(device=device, bpe_path=str(bpe_path) if bpe_path else None)
        processor = Sam3Processor(model, device=device)
        inference_state = processor.set_image(image)

        for prompt in prompt_list:
            print(f"\n  正在检测: '{prompt}'")
            output = processor.set_text_prompt(state=inference_state, prompt=prompt)

            boxes = output["boxes"]
            scores = output["scores"]

            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()

            prompt_count = 0
            for box, score in zip(boxes, scores):
                if score >= min_score:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    all_detected_boxes.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "score": float(score),
                        "prompt": prompt  # 记录来源 prompt
                    })
                    prompt_count += 1
                    print(f"    对象 {prompt_count}: ({x1}, {y1}, {x2}, {y2}), score={score:.3f}")
                else:
                    print(f"    跳过: score={score:.3f} < {min_score}")

            print(f"  '{prompt}' 检测到 {prompt_count} 个有效对象")
            total_detected += prompt_count

        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elif backend == "fal":
        api_key = _get_fal_api_key(sam_api_key)
        max_masks = max(1, min(32, int(sam_max_masks)))
        image_data_uri = _image_to_data_uri(image)
        print(f"SAM3 fal.ai API 模式: max_masks={max_masks}")

        for prompt in prompt_list:
            print(f"\n  正在检测: '{prompt}'")
            response_json = _call_sam3_api(
                image_data_uri=image_data_uri,
                prompt=prompt,
                api_key=api_key,
                max_masks=max_masks,
            )
            detections = _extract_sam3_api_detections(response_json, original_size)
            prompt_count = 0
            for det in detections:
                score = det.get("score")
                score_val = float(score) if score is not None else 0.0
                if score_val >= min_score:
                    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                    all_detected_boxes.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "score": score_val,
                        "prompt": prompt  # 记录来源 prompt
                    })
                    prompt_count += 1
                    print(f"    对象 {prompt_count}: ({x1}, {y1}, {x2}, {y2}), score={score_val:.3f}")
                else:
                    print(f"    跳过: score={score_val:.3f} < {min_score}")

            print(f"  '{prompt}' 检测到 {prompt_count} 个有效对象")
            total_detected += prompt_count
    elif backend == "roboflow":
        api_key = _get_roboflow_api_key(sam_api_key)
        image_base64 = _image_to_base64(image)
        print("SAM3 Roboflow API 模式: format=polygon")

        for prompt in prompt_list:
            print(f"\n  正在检测: '{prompt}'")
            response_json = _call_sam3_roboflow_api(
                image_base64=image_base64,
                prompt=prompt,
                api_key=api_key,
                min_score=min_score,
            )
            detections = _extract_roboflow_detections(response_json, original_size)
            prompt_count = 0
            for det in detections:
                score = det.get("score")
                score_val = float(score) if score is not None else 0.0
                if score_val >= min_score:
                    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                    all_detected_boxes.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "score": score_val,
                        "prompt": prompt
                    })
                    prompt_count += 1
                    print(f"    对象 {prompt_count}: ({x1}, {y1}, {x2}, {y2}), score={score_val:.3f}")
                else:
                    print(f"    跳过: score={score_val:.3f} < {min_score}")

            print(f"  '{prompt}' 检测到 {prompt_count} 个有效对象")
            total_detected += prompt_count
    else:
        raise ValueError(f"未知 SAM3 后端: {sam_backend}")

    print(f"\n总计检测: {total_detected} 个对象 (来自 {len(prompt_list)} 个 prompts)")

    # 为所有检测到的 boxes 分配临时 id 和 label（用于合并）
    valid_boxes = []
    for i, box_data in enumerate(all_detected_boxes):
        valid_boxes.append({
            "id": i,
            "label": f"<AF>{i + 1:02d}",
            "x1": box_data["x1"],
            "y1": box_data["y1"],
            "x2": box_data["x2"],
            "y2": box_data["y2"],
            "score": box_data["score"],
            "prompt": box_data["prompt"]
        })

    # === 新增：合并重叠的boxes ===
    if merge_threshold > 0 and len(valid_boxes) > 1:
        print(f"\n  合并重叠的boxes (阈值: {merge_threshold})...")
        original_count = len(valid_boxes)
        valid_boxes = merge_overlapping_boxes(valid_boxes, merge_threshold)
        merged_count = original_count - len(valid_boxes)
        if merged_count > 0:
            print(f"  合并完成: {original_count} -> {len(valid_boxes)} (合并了 {merged_count} 个)")
            # 打印合并后的box信息
            print(f"\n  合并后的boxes:")
            for box_info in valid_boxes:
                print(f"    {box_info['label']}: ({box_info['x1']}, {box_info['y1']}, {box_info['x2']}, {box_info['y2']})")
        else:
            print(f"  无需合并，所有boxes重叠比例均低于阈值")

    # 使用合并后的 valid_boxes 创建标记图片
    print(f"\n  绘制 samed.png (使用 {len(valid_boxes)} 个boxes)...")
    samed_image = image.copy()
    draw = ImageDraw.Draw(samed_image)

    for box_info in valid_boxes:
        x1, y1, x2, y2 = box_info["x1"], box_info["y1"], box_info["x2"], box_info["y2"]
        label = box_info["label"]

        # 灰色填充 + 黑色边框
        draw.rectangle([x1, y1, x2, y2], fill="#808080", outline="black", width=3)

        # 计算中心点
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # 获取合适大小的字体
        box_width = x2 - x1
        box_height = y2 - y1
        font = get_label_font(box_width, box_height)

        # 绘制白色居中序号标签
        if font:
            # 使用 anchor="mm" 居中绘制（如果支持）
            try:
                draw.text((cx, cy), label, fill="white", anchor="mm", font=font)
            except TypeError:
                # 旧版本 PIL 不支持 anchor，手动计算位置
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                text_x = cx - text_width // 2
                text_y = cy - text_height // 2
                draw.text((text_x, text_y), label, fill="white", font=font)
        else:
            # 无字体时使用默认
            draw.text((cx, cy), label, fill="white")

    samed_path = output_dir / "samed.png"
    samed_image.save(str(samed_path))
    print(f"标记图片已保存: {samed_path}")

    boxlib_data = {
        "image_size": {"width": original_size[0], "height": original_size[1]},
        "prompts_used": prompt_list,
        "boxes": valid_boxes
    }

    boxlib_path = output_dir / "boxlib.json"
    with open(boxlib_path, 'w', encoding='utf-8') as f:
        json.dump(boxlib_data, f, indent=2, ensure_ascii=False)
    print(f"Box 信息已保存: {boxlib_path}")

    return str(samed_path), str(boxlib_path), valid_boxes


# ============================================================================
# 步骤三：裁切 + RMBG2 去背景
# ============================================================================

class BriaRMBG2Remover:
    """使用 BRIA-RMBG 2.0 模型进行高质量背景抠图"""

    def __init__(self, model_path: Path | str | None = None, output_dir: Path | str | None = None):
        self.model_path = Path(model_path) if model_path else None
        self.output_dir = Path(output_dir) if output_dir else Path("./output/icons")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if self.model_path and self.model_path.exists():
            print(f"加载本地 RMBG 权重: {self.model_path}")
            self.model = AutoModelForImageSegmentation.from_pretrained(
                str(self.model_path), trust_remote_code=True,
            ).eval().to(device)
        else:
            print("从 HuggingFace 加载 RMBG-2.0 模型...")
            self.model = AutoModelForImageSegmentation.from_pretrained(
                "briaai/RMBG-2.0", trust_remote_code=True,
            ).eval().to(device)

        self.image_size = (1024, 1024)
        self.transform_image = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def remove_background(self, image: Image.Image, output_name: str) -> str:
        image_rgb = image.convert("RGB")
        input_tensor = self.transform_image(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid().cpu()

        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_rgb.size)

        out = image_rgb.copy()
        out.putalpha(mask)

        out_path = self.output_dir / f"{output_name}_nobg.png"
        out.save(out_path)
        return str(out_path)


def crop_and_remove_background(
    image_path: str,
    boxlib_path: str,
    output_dir: str,
    rmbg_model_path: Optional[str] = None,
) -> list[dict]:
    """
    根据 boxlib.json 裁切图片并使用 RMBG2 去背景

    文件命名使用 label: icon_AF01.png, icon_AF01_nobg.png
    """
    print("\n" + "=" * 60)
    print("步骤三：裁切 + RMBG2 去背景")
    print("=" * 60)

    output_dir = Path(output_dir)
    icons_dir = output_dir / "icons"
    icons_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path)
    with open(boxlib_path, 'r', encoding='utf-8') as f:
        boxlib_data = json.load(f)

    boxes = boxlib_data["boxes"]

    if len(boxes) == 0:
        print("警告: 没有检测到有效的 box")
        return []

    remover = BriaRMBG2Remover(model_path=rmbg_model_path, output_dir=icons_dir)

    icon_infos = []
    for box_info in boxes:
        box_id = box_info["id"]
        label = box_info.get("label", f"<AF>{box_id + 1:02d}")
        # 将 <AF>01 转换为 AF01 用于文件名
        label_clean = label.replace("<", "").replace(">", "")

        x1, y1, x2, y2 = box_info["x1"], box_info["y1"], box_info["x2"], box_info["y2"]

        cropped = image.crop((x1, y1, x2, y2))
        crop_path = icons_dir / f"icon_{label_clean}.png"
        cropped.save(crop_path)

        nobg_path = remover.remove_background(cropped, f"icon_{label_clean}")

        icon_infos.append({
            "id": box_id,
            "label": label,
            "label_clean": label_clean,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "width": x2 - x1, "height": y2 - y1,
            "crop_path": str(crop_path),
            "nobg_path": nobg_path,
        })

        print(f"  {label}: 裁切并去背景完成 -> {nobg_path}")

    del remover
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return icon_infos


# ============================================================================
# 步骤四：多模态调用生成 SVG
# ============================================================================

def generate_svg_template(
    figure_path: str,
    samed_path: str,
    boxlib_path: str,
    output_path: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    placeholder_mode: PlaceholderMode = "label",
) -> str:
    """
    使用多模态 LLM 生成 SVG 代码

    Args:
        placeholder_mode: 占位符模式
            - "none": 无特殊样式
            - "box": 传入 boxlib 坐标
            - "label": 灰色填充+黑色边框+序号标签（推荐）
    """
    print("\n" + "=" * 60)
    print("步骤四：多模态调用生成 SVG")
    print("=" * 60)
    print(f"Provider: {provider}")
    print(f"模型: {model}")
    print(f"占位符模式: {placeholder_mode}")

    figure_img = Image.open(figure_path)
    samed_img = Image.open(samed_path)

    figure_width, figure_height = figure_img.size
    print(f"原图尺寸: {figure_width} x {figure_height}")

    # 基础 prompt
    base_prompt = f"""编写svg代码来实现像素级别的复现这张图片（除了图标用相同大小的矩形占位符填充之外其他文字和组件(尤其是箭头样式)都要保持一致（即灰色矩形覆盖的内容就是图标））

CRITICAL DIMENSION REQUIREMENT:
- The original image has dimensions: {figure_width} x {figure_height} pixels
- Your SVG MUST use these EXACT dimensions to ensure accurate icon placement:
  - Set viewBox="0 0 {figure_width} {figure_height}"
  - Set width="{figure_width}" height="{figure_height}"
- DO NOT scale or resize the SVG
"""

    if placeholder_mode == "box":
        # box 模式：传入 boxlib 坐标
        with open(boxlib_path, 'r', encoding='utf-8') as f:
            boxlib_content = f.read()

        prompt_text = base_prompt + f"""
ICON COORDINATES FROM boxlib.json:
The following JSON contains precise icon coordinates detected by SAM3:
{boxlib_content}
Use these coordinates to accurately position your icon placeholders in the SVG.

Please output ONLY the SVG code, starting with <svg and ending with </svg>. Do not include any explanation or markdown formatting."""

    elif placeholder_mode == "label":
        # label 模式：要求占位符样式与 samed.png 一致
        prompt_text = base_prompt + """
PLACEHOLDER STYLE REQUIREMENT:
Look at the second image (samed.png) - each icon area is marked with a gray rectangle (#808080), black border, and a centered label like <AF>01, <AF>02, etc.

Your SVG placeholders MUST match this exact style:
- Rectangle with fill="#808080" and stroke="black" stroke-width="2"
- Centered white text showing the same label (<AF>01, <AF>02, etc.)
- Wrap each placeholder in a <g> element with id matching the label (e.g., id="AF01")

Example placeholder structure:
<g id="AF01">
  <rect x="100" y="50" width="80" height="80" fill="#808080" stroke="black" stroke-width="2"/>
  <text x="140" y="90" text-anchor="middle" dominant-baseline="middle" fill="white" font-size="14">&lt;AF&gt;01</text>
</g>

Please output ONLY the SVG code, starting with <svg and ending with </svg>. Do not include any explanation or markdown formatting."""

    else:  # none 模式
        prompt_text = base_prompt + """
Please output ONLY the SVG code, starting with <svg and ending with </svg>. Do not include any explanation or markdown formatting."""

    contents = [prompt_text, figure_img, samed_img]

    print(f"发送多模态请求到: {base_url}")

    content = call_llm_multimodal(
        contents=contents,
        api_key=api_key,
        model=model,
        base_url=base_url,
        provider=provider,
        max_tokens=50000,
    )

    if not content:
        raise Exception('API 响应中没有内容')

    svg_code = extract_svg_code(content)

    if not svg_code:
        raise Exception('无法从响应中提取 SVG 代码')

    # 步骤 4.5：SVG 语法验证和修复
    svg_code = check_and_fix_svg(
        svg_code=svg_code,
        api_key=api_key,
        model=model,
        base_url=base_url,
        provider=provider,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_code)

    print(f"SVG 模板已保存: {output_path}")
    return str(output_path)


def extract_svg_code(content: str) -> Optional[str]:
    """从响应内容中提取 SVG 代码"""
    pattern = r'(<svg[\s\S]*?</svg>)'
    match = re.search(pattern, content, re.IGNORECASE)
    if match:
        return match.group(1)

    pattern = r'```(?:svg|xml)?\s*([\s\S]*?)```'
    match = re.search(pattern, content)
    if match:
        code = match.group(1).strip()
        if code.startswith('<svg'):
            return code

    if content.strip().startswith('<svg'):
        return content.strip()

    return None


# ============================================================================
# 步骤 4.5：SVG 语法验证和修复
# ============================================================================

def validate_svg_syntax(svg_code: str) -> tuple[bool, list[str]]:
    """使用 lxml 解析验证 SVG 语法"""
    try:
        from lxml import etree
        etree.fromstring(svg_code.encode('utf-8'))
        return True, []
    except ImportError:
        print("  警告: lxml 未安装，使用内置 xml.etree 进行验证")
        try:
            import xml.etree.ElementTree as ET
            ET.fromstring(svg_code)
            return True, []
        except ET.ParseError as e:
            return False, [f"XML 解析错误: {str(e)}"]
    except Exception as e:
        from lxml import etree
        if isinstance(e, etree.XMLSyntaxError):
            errors = []
            error_log = e.error_log
            for error in error_log:
                errors.append(f"行 {error.line}, 列 {error.column}: {error.message}")
            if not errors:
                errors.append(f"行 {e.lineno}, 列 {e.offset}: {e.msg}")
            return False, errors
        else:
            return False, [f"解析错误: {str(e)}"]


def fix_svg_with_llm(
    svg_code: str,
    errors: list[str],
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    max_retries: int = 3,
) -> str:
    """使用 LLM 修复 SVG 语法错误"""
    print("\n  " + "-" * 50)
    print("  检测到 SVG 语法错误，调用 LLM 修复...")
    print("  " + "-" * 50)
    for err in errors:
        print(f"    {err}")

    current_svg = svg_code
    current_errors = errors

    for attempt in range(max_retries):
        print(f"\n  修复尝试 {attempt + 1}/{max_retries}...")

        error_list = "\n".join([f"  - {err}" for err in current_errors])
        prompt = f"""The following SVG code has XML syntax errors detected by an XML parser. Please fix ALL the errors and return valid SVG code.

SYNTAX ERRORS DETECTED:
{error_list}

ORIGINAL SVG CODE:
```xml
{current_svg}
```

IMPORTANT INSTRUCTIONS:
1. Fix all XML syntax errors (unclosed tags, invalid attributes, unescaped characters, etc.)
2. Ensure the output is valid XML that can be parsed by lxml
3. Keep all the visual elements and structure intact
4. Return ONLY the fixed SVG code, starting with <svg and ending with </svg>
5. Do NOT include any markdown formatting, explanation, or code blocks - just the raw SVG code"""

        try:
            content = call_llm_text(
                prompt=prompt,
                api_key=api_key,
                model=model,
                base_url=base_url,
                provider=provider,
                max_tokens=16000,
                temperature=0.3,
            )

            if not content:
                print("    响应为空")
                continue

            fixed_svg = extract_svg_code(content)

            if not fixed_svg:
                print("    无法从响应中提取 SVG 代码")
                continue

            is_valid, new_errors = validate_svg_syntax(fixed_svg)

            if is_valid:
                print("    修复成功！SVG 语法验证通过")
                return fixed_svg
            else:
                print(f"    修复后仍有 {len(new_errors)} 个错误:")
                for err in new_errors[:3]:
                    print(f"      {err}")
                if len(new_errors) > 3:
                    print(f"      ... 还有 {len(new_errors) - 3} 个错误")
                current_svg = fixed_svg
                current_errors = new_errors

        except Exception as e:
            print(f"    修复过程出错: {e}")
            continue

    print(f"  警告: 达到最大重试次数 ({max_retries})，返回最后一次的 SVG 代码")
    return current_svg


def check_and_fix_svg(
    svg_code: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
) -> str:
    """检查 SVG 语法并在需要时调用 LLM 修复"""
    print("\n" + "-" * 50)
    print("步骤 4.5：SVG 语法验证（使用 lxml XML 解析器）")
    print("-" * 50)

    is_valid, errors = validate_svg_syntax(svg_code)

    if is_valid:
        print("  SVG 语法验证通过！")
        return svg_code
    else:
        print(f"  发现 {len(errors)} 个语法错误")
        fixed_svg = fix_svg_with_llm(
            svg_code=svg_code,
            errors=errors,
            api_key=api_key,
            model=model,
            base_url=base_url,
            provider=provider,
        )
        return fixed_svg


# ============================================================================
# 步骤 4.7：坐标系对齐
# ============================================================================

def get_svg_dimensions(svg_code: str) -> tuple[Optional[float], Optional[float]]:
    """从 SVG 代码中提取坐标系尺寸"""
    viewbox_pattern = r'viewBox=["\']([^"\']+)["\']'
    viewbox_match = re.search(viewbox_pattern, svg_code, re.IGNORECASE)

    if viewbox_match:
        viewbox_value = viewbox_match.group(1).strip()
        parts = viewbox_value.split()
        if len(parts) >= 4:
            try:
                vb_width = float(parts[2])
                vb_height = float(parts[3])
                return vb_width, vb_height
            except ValueError:
                pass

    def parse_dimension(attr_name: str) -> Optional[float]:
        pattern = rf'{attr_name}=["\']([^"\']+)["\']'
        match = re.search(pattern, svg_code, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            numeric_match = re.match(r'([\d.]+)', value)
            if numeric_match:
                try:
                    return float(numeric_match.group(1))
                except ValueError:
                    pass
        return None

    width = parse_dimension('width')
    height = parse_dimension('height')

    if width and height:
        return width, height

    return None, None


def calculate_scale_factors(
    figure_width: int,
    figure_height: int,
    svg_width: float,
    svg_height: float,
) -> tuple[float, float]:
    """计算从 figure.png 像素坐标到 SVG 坐标的缩放因子"""
    scale_x = svg_width / figure_width
    scale_y = svg_height / figure_height
    return scale_x, scale_y


# ============================================================================
# 步骤五：图标替换到 SVG（支持序号匹配）
# ============================================================================

def replace_icons_in_svg(
    template_svg_path: str,
    icon_infos: list[dict],
    output_path: str,
    scale_factors: tuple[float, float] = (1.0, 1.0),
    match_by_label: bool = True,
) -> str:
    """
    将透明背景图标替换到 SVG 中的占位符

    Args:
        template_svg_path: SVG 模板路径
        icon_infos: 图标信息列表
        output_path: 输出路径
        scale_factors: 坐标缩放因子
        match_by_label: 是否使用序号匹配（label 模式）
    """
    print("\n" + "=" * 60)
    print("步骤五：图标替换到 SVG")
    print("=" * 60)
    print(f"匹配模式: {'序号匹配' if match_by_label else '坐标匹配'}")

    scale_x, scale_y = scale_factors
    if scale_x != 1.0 or scale_y != 1.0:
        print(f"应用坐标缩放: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")

    with open(template_svg_path, 'r', encoding='utf-8') as f:
        svg_content = f.read()

    for icon_info in icon_infos:
        label = icon_info.get("label", "")
        label_clean = icon_info.get("label_clean", label.replace("<", "").replace(">", ""))
        nobg_path = icon_info["nobg_path"]

        # 读取图标并转为 base64
        icon_img = Image.open(nobg_path)
        buf = io.BytesIO()
        icon_img.save(buf, format="PNG")
        icon_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        replaced = False

        if match_by_label and label:
            # 方式1：查找 id="AF01" 的 <g> 元素
            g_pattern = rf'<g[^>]*\bid=["\']?{re.escape(label_clean)}["\']?[^>]*>[\s\S]*?</g>'
            g_match = re.search(g_pattern, svg_content, re.IGNORECASE)

            if g_match:
                g_content = g_match.group(0)

                # 提取 <g> 元素的 transform="translate(x, y)" （如果存在）
                # 这处理 LLM 生成 <g id="AF01" transform="translate(100, 50)"><rect x="0" y="0" ...> 的情况
                g_tag_match = re.match(r'<g[^>]*>', g_content, re.IGNORECASE)
                translate_x, translate_y = 0.0, 0.0
                if g_tag_match:
                    g_tag = g_tag_match.group(0)
                    # 匹配 transform="translate(100, 50)" 或 transform="translate(100 50)"
                    transform_pattern = r'transform=["\'][^"\']*translate\s*\(\s*([\d.-]+)[\s,]+([\d.-]+)\s*\)'
                    transform_match = re.search(transform_pattern, g_tag, re.IGNORECASE)
                    if transform_match:
                        translate_x = float(transform_match.group(1))
                        translate_y = float(transform_match.group(2))

                # 从 <g> 中提取 <rect> 的尺寸
                rect_patterns = [
                    # x="100" y="50" width="80" height="80"
                    r'<rect[^>]*\bx=["\']?([\d.]+)["\']?[^>]*\by=["\']?([\d.]+)["\']?[^>]*\bwidth=["\']?([\d.]+)["\']?[^>]*\bheight=["\']?([\d.]+)["\']?',
                    # width="80" height="80" x="100" y="50" (属性顺序不同)
                    r'<rect[^>]*\bwidth=["\']?([\d.]+)["\']?[^>]*\bheight=["\']?([\d.]+)["\']?[^>]*\bx=["\']?([\d.]+)["\']?[^>]*\by=["\']?([\d.]+)["\']?',
                ]

                rect_info = None
                for rp in rect_patterns:
                    rect_match = re.search(rp, g_content, re.IGNORECASE)
                    if rect_match:
                        groups = rect_match.groups()
                        if len(groups) == 4:
                            if 'width' in rp[:50]:  # 第二种模式
                                width, height, x, y = groups
                            else:
                                x, y, width, height = groups
                            rect_info = {
                                'x': float(x),
                                'y': float(y),
                                'width': float(width),
                                'height': float(height)
                            }
                            break

                if rect_info:
                    # 将 <g> 的 transform translate 值加到 rect 坐标上
                    x = rect_info['x'] + translate_x
                    y = rect_info['y'] + translate_y
                    width, height = rect_info['width'], rect_info['height']

                    # 如果应用了 transform，输出提示
                    if translate_x != 0 or translate_y != 0:
                        print(f"  {label}: 检测到 <g> transform: translate({translate_x}, {translate_y})")

                    # 创建 image 标签替换整个 <g>
                    image_tag = f'<image id="icon_{label_clean}" x="{x}" y="{y}" width="{width}" height="{height}" href="data:image/png;base64,{icon_b64}" preserveAspectRatio="xMidYMid meet"/>'
                    svg_content = svg_content.replace(g_content, image_tag)
                    print(f"  {label}: 替换成功 (序号匹配 <g>) at ({x}, {y}) size {width}x{height}")
                    replaced = True

            # 方式2：查找包含 label 文本的 <text> 元素附近的 <rect>
            if not replaced:
                # 查找包含 <AF>01 或 &lt;AF&gt;01 的文本
                text_patterns = [
                    rf'<text[^>]*>[^<]*{re.escape(label)}[^<]*</text>',
                    rf'<text[^>]*>[^<]*&lt;AF&gt;{label_clean[2:]}[^<]*</text>',
                ]

                for tp in text_patterns:
                    text_match = re.search(tp, svg_content, re.IGNORECASE)
                    if text_match:
                        # 找到文本，向前查找最近的 <rect>
                        text_pos = text_match.start()
                        preceding_svg = svg_content[:text_pos]

                        # 查找最后一个 <rect>
                        rect_matches = list(re.finditer(r'<rect[^>]*/?\s*>', preceding_svg, re.IGNORECASE))
                        if rect_matches:
                            last_rect = rect_matches[-1]
                            rect_content = last_rect.group(0)

                            # 提取 rect 的属性
                            x_match = re.search(r'\bx=["\']?([\d.]+)', rect_content)
                            y_match = re.search(r'\by=["\']?([\d.]+)', rect_content)
                            w_match = re.search(r'\bwidth=["\']?([\d.]+)', rect_content)
                            h_match = re.search(r'\bheight=["\']?([\d.]+)', rect_content)

                            if all([x_match, y_match, w_match, h_match]):
                                x = float(x_match.group(1))
                                y = float(y_match.group(1))
                                width = float(w_match.group(1))
                                height = float(h_match.group(1))

                                # 替换 rect 和 text
                                image_tag = f'<image id="icon_{label_clean}" x="{x}" y="{y}" width="{width}" height="{height}" href="data:image/png;base64,{icon_b64}" preserveAspectRatio="xMidYMid meet"/>'

                                # 删除 text
                                svg_content = svg_content.replace(text_match.group(0), '')
                                # 替换 rect
                                svg_content = svg_content.replace(rect_content, image_tag, 1)

                                print(f"  {label}: 替换成功 (序号匹配 <text>) at ({x}, {y}) size {width}x{height}")
                                replaced = True
                                break

        # 回退：使用坐标匹配
        if not replaced:
            orig_x1, orig_y1 = icon_info["x1"], icon_info["y1"]
            orig_width, orig_height = icon_info["width"], icon_info["height"]

            x1 = orig_x1 * scale_x
            y1 = orig_y1 * scale_y
            width = orig_width * scale_x
            height = orig_height * scale_y

            image_tag = f'<image id="icon_{label_clean}" x="{x1:.1f}" y="{y1:.1f}" width="{width:.1f}" height="{height:.1f}" href="data:image/png;base64,{icon_b64}" preserveAspectRatio="xMidYMid meet"/>'

            x1_int, y1_int = int(round(x1)), int(round(y1))

            # 精确匹配
            rect_pattern = rf'<rect[^>]*x=["\']?{x1_int}(?:\.0)?["\']?[^>]*y=["\']?{y1_int}(?:\.0)?["\']?[^>]*/?\s*>'
            if re.search(rect_pattern, svg_content):
                svg_content = re.sub(rect_pattern, image_tag, svg_content, count=1)
                print(f"  {label}: 替换成功 (坐标精确匹配) at ({x1:.1f}, {y1:.1f})")
                replaced = True
            else:
                # 近似匹配
                tolerance = 10
                found = False
                for dx in range(-tolerance, tolerance+1, 2):
                    for dy in range(-tolerance, tolerance+1, 2):
                        search_x = x1_int + dx
                        search_y = y1_int + dy
                        rect_pattern = rf'<rect[^>]*x=["\']?{search_x}(?:\.0)?["\']?[^>]*y=["\']?{search_y}(?:\.0)?["\']?[^>]*(?:fill=["\']?(?:#[0-9A-Fa-f]{{3,6}}|gray|grey)["\']?|stroke=["\']?(?:black|#000|#000000)["\']?)[^>]*/?\s*>'
                        if re.search(rect_pattern, svg_content, re.IGNORECASE):
                            svg_content = re.sub(rect_pattern, image_tag, svg_content, count=1, flags=re.IGNORECASE)
                            print(f"  {label}: 替换成功 (坐标近似匹配) at ({x1:.1f}, {y1:.1f})")
                            found = True
                            replaced = True
                            break
                    if found:
                        break

        if not replaced:
            # 追加到 SVG 末尾
            orig_x1, orig_y1 = icon_info["x1"], icon_info["y1"]
            orig_width, orig_height = icon_info["width"], icon_info["height"]
            x1 = orig_x1 * scale_x
            y1 = orig_y1 * scale_y
            width = orig_width * scale_x
            height = orig_height * scale_y

            image_tag = f'<image id="icon_{label_clean}" x="{x1:.1f}" y="{y1:.1f}" width="{width:.1f}" height="{height:.1f}" href="data:image/png;base64,{icon_b64}" preserveAspectRatio="xMidYMid meet"/>'
            svg_content = svg_content.replace('</svg>', f'  {image_tag}\n</svg>')
            print(f"  {label}: 追加到 SVG at ({x1:.1f}, {y1:.1f}) (未找到匹配的占位符)")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)

    print(f"最终 SVG 已保存: {output_path}")
    return str(output_path)


# ============================================================================
# 步骤 4.6：LLM 优化 SVG
# ============================================================================

def count_base64_images(svg_code: str) -> int:
    """统计 SVG 中嵌入的 base64 图片数量"""
    pattern = r'(?:href|xlink:href)=["\']data:image/[^;]+;base64,[A-Za-z0-9+/=]+'
    matches = re.findall(pattern, svg_code)
    return len(matches)


def validate_base64_images(svg_code: str, expected_count: int) -> tuple[bool, str]:
    """验证 SVG 中的 base64 图片是否完整"""
    actual_count = count_base64_images(svg_code)

    if actual_count < expected_count:
        return False, f"base64 图片数量不足: 期望 {expected_count}, 实际 {actual_count}"

    pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
    for match in re.finditer(pattern, svg_code):
        b64_data = match.group(1)
        if len(b64_data) % 4 != 0:
            return False, f"发现截断的 base64 数据（长度 {len(b64_data)} 不是 4 的倍数）"
        if len(b64_data) < 100:
            return False, f"发现过短的 base64 数据（长度 {len(b64_data)}），可能被截断"

    return True, f"base64 图片验证通过: {actual_count} 张图片"


def svg_to_png(svg_path: str, output_path: str, scale: float = 1.0) -> Optional[str]:
    """将 SVG 转换为 PNG"""
    try:
        import cairosvg
        cairosvg.svg2png(url=svg_path, write_to=output_path, scale=scale)
        return output_path
    except ImportError:
        print("  警告: cairosvg 未安装，尝试使用其他方法")
        try:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPM
            drawing = svg2rlg(svg_path)
            renderPM.drawToFile(drawing, output_path, fmt="PNG")
            return output_path
        except ImportError:
            print("  警告: svglib 也未安装，无法转换 SVG 到 PNG")
            return None
        except Exception as e:
            print(f"  警告: svglib 转换失败: {e}")
            return None
    except Exception as e:
        print(f"  警告: cairosvg 转换失败: {e}")
        return None


def optimize_svg_with_llm(
    figure_path: str,
    samed_path: str,
    final_svg_path: str,
    output_path: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    max_iterations: int = 2,
    skip_base64_validation: bool = False,
) -> str:
    """
    使用 LLM 优化 SVG，使其与原图更加对齐

    Args:
        figure_path: 原图路径
        samed_path: 标记图路径
        final_svg_path: 输入 SVG 路径
        output_path: 输出 SVG 路径
        api_key: API Key
        model: 模型名称
        base_url: API base URL
        provider: API 提供商
        max_iterations: 最大迭代次数（0 表示跳过优化）
        skip_base64_validation: 是否跳过 base64 图片验证

    Returns:
        优化后的 SVG 路径
    """
    print("\n" + "=" * 60)
    print("步骤 4.6：LLM 优化 SVG（位置和样式对齐）")
    print("=" * 60)
    print(f"Provider: {provider}")
    print(f"模型: {model}")
    print(f"最大迭代次数: {max_iterations}")

    # 如果迭代次数为 0，直接复制文件并跳过优化
    if max_iterations == 0:
        print("  迭代次数为 0，跳过 LLM 优化")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(final_svg_path, output_path)
        print(f"  直接复制模板: {final_svg_path} -> {output_path}")
        return str(output_path)

    with open(final_svg_path, 'r', encoding='utf-8') as f:
        current_svg = f.read()

    output_dir = Path(final_svg_path).parent

    original_image_count = 0
    if not skip_base64_validation:
        original_image_count = count_base64_images(current_svg)
        print(f"原始 SVG 包含 {original_image_count} 张嵌入图片")
    else:
        print("跳过 base64 图片验证（模板 SVG）")

    for iteration in range(max_iterations):
        print(f"\n  优化迭代 {iteration + 1}/{max_iterations}")
        print("  " + "-" * 50)

        current_svg_path = output_dir / f"temp_svg_iter_{iteration}.svg"
        current_png_path = output_dir / f"temp_png_iter_{iteration}.png"

        with open(current_svg_path, 'w', encoding='utf-8') as f:
            f.write(current_svg)

        png_result = svg_to_png(str(current_svg_path), str(current_png_path))

        if png_result is None:
            print("  无法将 SVG 转换为 PNG，跳过优化")
            break

        figure_img = Image.open(figure_path)
        samed_img = Image.open(samed_path)
        current_png_img = Image.open(str(current_png_path))

        prompt = f"""You are an expert SVG optimizer. Compare the current SVG rendering with the original figure and optimize the SVG code to better match the original.

I'm providing you with 4 inputs:
1. **Image 1 (figure.png)**: The original target figure that we want to replicate
2. **Image 2 (samed.png)**: The same figure with icon positions marked as gray rectangles with labels (<AF>01, <AF>02, etc.)
3. **Image 3 (current SVG rendered as PNG)**: The current state of our SVG
4. **Current SVG code**: The SVG code that needs optimization

Please carefully compare and check the following **TWO MAJOR ASPECTS with EIGHT KEY POINTS**:

## ASPECT 1: POSITION (位置)
1. **Icons (图标)**: Are icon placeholder positions matching the original?
2. **Text (文字)**: Are text elements positioned correctly?
3. **Arrows (箭头)**: Are arrows starting/ending at correct positions?
4. **Lines/Borders (线条)**: Are lines and borders aligned properly?

## ASPECT 2: STYLE (样式)
5. **Icons (图标)**: Icon placeholder sizes, proportions (must have gray fill #808080, black border, and centered label)
6. **Text (文字)**: Font sizes, colors, weights
7. **Arrows (箭头)**: Arrow styles, thicknesses, colors
8. **Lines/Borders (线条)**: Line styles, colors, stroke widths

**CURRENT SVG CODE:**
```xml
{current_svg}
```

**IMPORTANT:**
- Output ONLY the optimized SVG code
- Start with <svg and end with </svg>
- Do NOT include markdown formatting or explanations
- Keep all icon placeholder structures intact (the <g> elements with id like "AF01")
- Focus on position and style corrections"""

        contents = [prompt, figure_img, samed_img, current_png_img]

        try:
            print("  发送优化请求...")
            content = call_llm_multimodal(
                contents=contents,
                api_key=api_key,
                model=model,
                base_url=base_url,
                provider=provider,
                max_tokens=50000,
                temperature=0.3,
            )

            if not content:
                print("  响应为空")
                continue

            optimized_svg = extract_svg_code(content)

            if not optimized_svg:
                print("  无法从响应中提取 SVG 代码")
                continue

            is_valid, errors = validate_svg_syntax(optimized_svg)

            if not is_valid:
                print(f"  优化后的 SVG 有语法错误，尝试修复...")
                optimized_svg = fix_svg_with_llm(
                    svg_code=optimized_svg,
                    errors=errors,
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                    provider=provider,
                )

            if not skip_base64_validation:
                images_valid, images_msg = validate_base64_images(optimized_svg, original_image_count)
                if not images_valid:
                    print(f"  警告: {images_msg}")
                    print("  拒绝此次优化，保留上一版本 SVG")
                    continue
                print(f"  {images_msg}")

            current_svg = optimized_svg
            print("  优化迭代完成")

        except Exception as e:
            print(f"  优化过程出错: {e}")
            continue

        try:
            current_svg_path.unlink(missing_ok=True)
            current_png_path.unlink(missing_ok=True)
        except:
            pass

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(current_svg)

    final_png_path = output_path.with_suffix('.png')
    svg_to_png(str(output_path), str(final_png_path))
    print(f"\n  优化后的 SVG 已保存: {output_path}")
    print(f"  PNG 预览已保存: {final_png_path}")

    return str(output_path)


# ============================================================================
# 主函数：完整流程
# ============================================================================

def method_to_svg(
    method_text: str,
    output_dir: str = "./output",
    api_key: str = None,
    base_url: str = None,
    provider: ProviderType = "bianxie",
    image_gen_model: str = None,
    svg_gen_model: str = None,
    sam_prompts: str = "icon",
    min_score: float = 0.5,
    sam_backend: Literal["local", "fal", "roboflow", "api"] = "local",
    sam_api_key: Optional[str] = None,
    sam_max_masks: int = 32,
    rmbg_model_path: Optional[str] = None,
    stop_after: int = 5,
    placeholder_mode: PlaceholderMode = "label",
    optimize_iterations: int = 2,
    merge_threshold: float = 0.9,
) -> dict:
    """
    完整流程：Paper Method → SVG with Icons

    Args:
        method_text: Paper method 文本内容
        output_dir: 输出目录
        api_key: API Key
        base_url: API base URL
        provider: API 提供商
        image_gen_model: 生图模型
        svg_gen_model: SVG 生成模型
        sam_prompts: SAM3 文本提示，支持逗号分隔的多个prompt（如 "icon,diagram,arrow"）
        min_score: SAM3 最低置信度
        sam_backend: SAM3 后端（local/fal/roboflow/api）
        sam_api_key: SAM3 API Key（api 模式使用）
        sam_max_masks: SAM3 API 最大 masks 数（api 模式使用）
        rmbg_model_path: RMBG 模型路径
        stop_after: 执行到指定步骤后停止
        placeholder_mode: 占位符模式
            - "none": 无特殊样式
            - "box": 传入 boxlib 坐标
            - "label": 灰色填充+黑色边框+序号标签（推荐）
        optimize_iterations: 步骤 4.6 优化迭代次数（0 表示跳过优化）
        merge_threshold: Box合并阈值，重叠比例超过此值则合并（0表示不合并，默认0.9）

    Returns:
        结果字典
    """
    if not api_key:
        raise ValueError("必须提供 api_key")

    # 获取默认配置
    config = PROVIDER_CONFIGS[provider]
    if base_url is None:
        base_url = config["base_url"]
    if image_gen_model is None:
        image_gen_model = config["default_image_model"]
    if svg_gen_model is None:
        svg_gen_model = config["default_svg_model"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Paper Method 到 SVG 图标替换流程 (Label 模式增强版 + Box合并)")
    print("=" * 60)
    print(f"Provider: {provider}")
    print(f"输出目录: {output_dir}")
    print(f"生图模型: {image_gen_model}")
    print(f"SVG模型: {svg_gen_model}")
    print(f"SAM提示词: {sam_prompts}")
    print(f"最低置信度: {min_score}")
    sam_backend_value = "fal" if sam_backend == "api" else sam_backend
    print(f"SAM后端: {sam_backend_value}")
    if sam_backend_value == "fal":
        print(f"SAM3 API max_masks: {sam_max_masks}")
    print(f"执行到步骤: {stop_after}")
    print(f"占位符模式: {placeholder_mode}")
    print(f"优化迭代次数: {optimize_iterations}")
    print(f"Box合并阈值: {merge_threshold}")
    print("=" * 60)

    # 步骤一：生成图片
    figure_path = output_dir / "figure.png"
    generate_figure_from_method(
        method_text=method_text,
        output_path=str(figure_path),
        api_key=api_key,
        model=image_gen_model,
        base_url=base_url,
        provider=provider,
    )

    if stop_after == 1:
        print("\n" + "=" * 60)
        print("已在步骤 1 后停止")
        print("=" * 60)
        return {
            "figure_path": str(figure_path),
            "samed_path": None,
            "boxlib_path": None,
            "icon_infos": [],
            "template_svg_path": None,
            "optimized_template_path": None,
            "final_svg_path": None,
        }

    # 步骤二：SAM3 分割（包含Box合并）
    samed_path, boxlib_path, valid_boxes = segment_with_sam3(
        image_path=str(figure_path),
        output_dir=str(output_dir),
        text_prompts=sam_prompts,
        min_score=min_score,
        merge_threshold=merge_threshold,
        sam_backend=sam_backend_value,
        sam_api_key=sam_api_key,
        sam_max_masks=sam_max_masks,
    )

    if len(valid_boxes) == 0:
        print("\n警告: 没有检测到有效的图标，流程终止")
        return {
            "figure_path": str(figure_path),
            "samed_path": samed_path,
            "boxlib_path": boxlib_path,
            "icon_infos": [],
            "template_svg_path": None,
            "final_svg_path": None,
        }

    print(f"\n检测到 {len(valid_boxes)} 个图标")

    if stop_after == 2:
        print("\n" + "=" * 60)
        print("已在步骤 2 后停止")
        print("=" * 60)
        return {
            "figure_path": str(figure_path),
            "samed_path": samed_path,
            "boxlib_path": boxlib_path,
            "icon_infos": [],
            "template_svg_path": None,
            "optimized_template_path": None,
            "final_svg_path": None,
        }

    # 步骤三：裁切 + 去背景
    icon_infos = crop_and_remove_background(
        image_path=str(figure_path),
        boxlib_path=boxlib_path,
        output_dir=str(output_dir),
        rmbg_model_path=rmbg_model_path,
    )

    if stop_after == 3:
        print("\n" + "=" * 60)
        print("已在步骤 3 后停止")
        print("=" * 60)
        return {
            "figure_path": str(figure_path),
            "samed_path": samed_path,
            "boxlib_path": boxlib_path,
            "icon_infos": icon_infos,
            "template_svg_path": None,
            "optimized_template_path": None,
            "final_svg_path": None,
        }

    # 步骤四：生成 SVG 模板
    template_svg_path = output_dir / "template.svg"
    generate_svg_template(
        figure_path=str(figure_path),
        samed_path=samed_path,
        boxlib_path=boxlib_path,
        output_path=str(template_svg_path),
        api_key=api_key,
        model=svg_gen_model,
        base_url=base_url,
        provider=provider,
        placeholder_mode=placeholder_mode,
    )

    # 步骤 4.6：LLM 优化 SVG 模板（可配置迭代次数，0 表示跳过）
    optimized_template_path = output_dir / "optimized_template.svg"
    optimize_svg_with_llm(
        figure_path=str(figure_path),
        samed_path=samed_path,
        final_svg_path=str(template_svg_path),
        output_path=str(optimized_template_path),
        api_key=api_key,
        model=svg_gen_model,
        base_url=base_url,
        provider=provider,
        max_iterations=optimize_iterations,
        skip_base64_validation=True,
    )

    if stop_after == 4:
        print("\n" + "=" * 60)
        print("已在步骤 4 后停止")
        print("=" * 60)
        return {
            "figure_path": str(figure_path),
            "samed_path": samed_path,
            "boxlib_path": boxlib_path,
            "icon_infos": icon_infos,
            "template_svg_path": str(template_svg_path),
            "optimized_template_path": str(optimized_template_path),
            "final_svg_path": None,
        }

    # 步骤 4.7：坐标系对齐
    print("\n" + "-" * 50)
    print("步骤 4.7：坐标系对齐")
    print("-" * 50)

    figure_img = Image.open(figure_path)
    figure_width, figure_height = figure_img.size
    print(f"原图尺寸: {figure_width} x {figure_height}")

    with open(optimized_template_path, 'r', encoding='utf-8') as f:
        svg_code = f.read()

    svg_width, svg_height = get_svg_dimensions(svg_code)

    if svg_width and svg_height:
        print(f"SVG 尺寸: {svg_width} x {svg_height}")

        if abs(svg_width - figure_width) < 1 and abs(svg_height - figure_height) < 1:
            print("尺寸匹配，使用 1:1 坐标映射")
            scale_factors = (1.0, 1.0)
        else:
            scale_x, scale_y = calculate_scale_factors(
                figure_width, figure_height, svg_width, svg_height
            )
            scale_factors = (scale_x, scale_y)
            print(f"尺寸不匹配，计算缩放因子: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")
    else:
        print("警告: 无法提取 SVG 尺寸，使用 1:1 坐标映射")
        scale_factors = (1.0, 1.0)

    # 步骤五：图标替换
    final_svg_path = output_dir / "final.svg"
    replace_icons_in_svg(
        template_svg_path=str(optimized_template_path),
        icon_infos=icon_infos,
        output_path=str(final_svg_path),
        scale_factors=scale_factors,
        match_by_label=(placeholder_mode == "label"),
    )

    print("\n" + "=" * 60)
    print("流程完成！")
    print("=" * 60)
    print(f"原始图片: {figure_path}")
    print(f"标记图片: {samed_path}")
    print(f"Box信息: {boxlib_path}")
    print(f"图标数量: {len(icon_infos)}")
    print(f"SVG模板: {template_svg_path}")
    print(f"优化后模板: {optimized_template_path}")
    print(f"最终SVG: {final_svg_path}")

    return {
        "figure_path": str(figure_path),
        "samed_path": samed_path,
        "boxlib_path": boxlib_path,
        "icon_infos": icon_infos,
        "template_svg_path": str(template_svg_path),
        "optimized_template_path": str(optimized_template_path),
        "final_svg_path": str(final_svg_path),
    }


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paper Method 到 SVG 图标替换工具 (Label 模式增强版 + Box合并)"
    )

    # 输入参数
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--method_text", help="Paper method 文本内容")
    input_group.add_argument("--method_file", default="./paper.txt", help="包含 paper method 的文本文件路径")

    # 输出参数
    parser.add_argument("--output_dir", default="./output", help="输出目录（默认: ./output）")

    # Provider 参数
    parser.add_argument(
        "--provider",
        choices=["openrouter", "bianxie"],
        default="bianxie",
        help="API 提供商（默认: bianxie）"
    )

    # API 参数
    parser.add_argument("--api_key", default=None, help="API Key")
    parser.add_argument("--base_url", default=None, help="API base URL（默认根据 provider 自动设置）")

    # 模型参数
    parser.add_argument("--image_model", default=None, help="生图模型（默认根据 provider 自动设置）")
    parser.add_argument("--svg_model", default=None, help="SVG生成模型（默认根据 provider 自动设置）")

    # Step 1 参考图片参数
    parser.add_argument(
        "--use_reference_image",
        action="store_true",
        help="步骤一使用参考图片风格（需要同时提供 --reference_image_path）"
    )
    parser.add_argument("--reference_image_path", default=None, help="参考图片路径（可选）")

    # SAM3 参数
    parser.add_argument("--sam_prompt", default="icon,robot,animal,person", help="SAM3 文本提示，支持逗号分隔多个prompt（如 'icon,diagram,arrow'，默认: icon）")
    parser.add_argument("--min_score", type=float, default=0.0, help="SAM3 最低置信度阈值（默认: 0.0）")
    parser.add_argument(
        "--sam_backend",
        choices=["local", "fal", "roboflow", "api"],
        default="local",
        help="SAM3 后端：local(本地部署)/fal(fal.ai)/roboflow(Roboflow)/api(旧别名=fal)",
    )
    parser.add_argument("--sam_api_key", default=None, help="SAM3 API Key（默认使用 FAL_KEY）")
    parser.add_argument(
        "--sam_max_masks",
        type=int,
        default=32,
        help="SAM3 API 最大 masks 数（仅 api 后端，默认: 32）",
    )

    # RMBG 参数
    parser.add_argument("--rmbg_model_path", default=None, help="RMBG 模型本地路径（可选）")

    # 流程控制参数
    parser.add_argument(
        "--stop_after",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=5,
        help="执行到指定步骤后停止（1-5，默认: 5 完整流程）"
    )

    # 占位符模式参数
    parser.add_argument(
        "--placeholder_mode",
        choices=["none", "box", "label"],
        default="label",
        help="占位符模式：none(无样式)/box(传坐标)/label(序号匹配)（默认: label）"
    )

    # 步骤 4.6 优化迭代次数参数
    parser.add_argument(
        "--optimize_iterations",
        type=int,
        default=0,
        help="步骤 4.6 LLM 优化迭代次数（0 表示跳过优化，默认: 0）"
    )

    # Box 合并阈值参数
    parser.add_argument(
        "--merge_threshold",
        type=float,
        default=0.001,
        help="Box合并阈值，重叠比例超过此值则合并（0表示不合并，默认: 0.9）"
    )

    args = parser.parse_args()

    if args.use_reference_image and not args.reference_image_path:
        parser.error("--use_reference_image 需要 --reference_image_path")
    if args.reference_image_path and not Path(args.reference_image_path).is_file():
        parser.error(f"参考图片不存在: {args.reference_image_path}")

    USE_REFERENCE_IMAGE = bool(args.use_reference_image)
    REFERENCE_IMAGE_PATH = args.reference_image_path
    if REFERENCE_IMAGE_PATH:
        USE_REFERENCE_IMAGE = True

    # 获取 method 文本：优先使用 --method_text
    method_text = args.method_text
    if method_text is None:
        with open(args.method_file, 'r', encoding='utf-8') as f:
            method_text = f.read()

    # 运行完整流程
    result = method_to_svg(
        method_text=method_text,
        output_dir=args.output_dir,
        api_key=args.api_key,
        base_url=args.base_url,
        provider=args.provider,
        image_gen_model=args.image_model,
        svg_gen_model=args.svg_model,
        sam_prompts=args.sam_prompt,
        min_score=args.min_score,
        sam_backend=args.sam_backend,
        sam_api_key=args.sam_api_key,
        sam_max_masks=args.sam_max_masks,
        rmbg_model_path=args.rmbg_model_path,
        stop_after=args.stop_after,
        placeholder_mode=args.placeholder_mode,
        optimize_iterations=args.optimize_iterations,
        merge_threshold=args.merge_threshold,
    )

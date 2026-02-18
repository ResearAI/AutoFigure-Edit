"""Local/Kimi Provider 实现"""

import base64
import io
import os
from typing import Optional, List, Any, Dict
from PIL import Image


def call_local_image_generation(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    reference_image: Optional[Image.Image] = None,
    num_inference_steps: int = 50,
    seed: int = 42,
    cfg: float = 7.5,
    local_img_path: str = "",
) -> Optional[Image.Image]:
    """使用 local 调用图像生成接口"""
    if not os.path.exists(local_img_path):
        raise FileNotFoundError(f"本地图片不存在: {local_img_path}")
    image = Image.open(local_img_path).convert("RGB")
    return image


def call_kimi_multimodal(
    contents: List[Any],
    api_key: str,
    model: str,
    base_url: str,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """使用 OpenAI SDK 调用 Kimi 多模态接口"""
    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)
        message_content: List[Dict[str, Any]] = []
        for part in contents:
            if isinstance(part, str):
                message_content.append({"type": "text", "text": part})
            elif isinstance(part, Image.Image):
                # 压缩图片以减少 token 消耗
                img = part.copy()
                # 如果图片太大，缩小到合理尺寸
                max_size = 5000
                if img.width > max_size or img.height > max_size:
                    ratio = min(max_size / img.width, max_size / img.height)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    print(f"[Kimi] 图片已压缩: {part.size} -> {img.size}")
                
                buf = io.BytesIO()
                # 使用 JPEG 格式并降低质量以减少大小
                img.convert('RGB').save(buf, format='JPEG', quality=85)
                image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                print(f"[Kimi] 图片 Base64 大小: {len(image_b64)} 字符")
                
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                })

        print(f"[Kimi] 发送请求: model={model}, max_tokens={max_tokens}")
        print(f"[Kimi] 消息内容包含: {len([c for c in contents if isinstance(c, str)])} 个文本, {len([c for c in contents if isinstance(c, Image.Image)])} 张图片")
        
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message_content}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        print(f"[Kimi] API 响应状态: completion={completion is not None}, choices={len(completion.choices) if completion and completion.choices else 0}")
        
        if completion and completion.choices:
            content = completion.choices[0].message.content
            finish_reason = completion.choices[0].finish_reason if completion.choices else None
            print(f"[Kimi] 返回内容长度: {len(content) if content else 0}")
            print(f"[Kimi] finish_reason: {finish_reason}")
            
            if not content:
                print(f"[Kimi] 警告: API 返回了空内容")
                print(f"[Kimi] 完整响应: {completion}")
                
                # 如果是因为长度限制，给出提示
                if finish_reason == "length":
                    print(f"[Kimi] 提示: 响应被截断，可能需要增加 max_tokens")
                elif finish_reason == "content_filter":
                    print(f"[Kimi] 提示: 内容被过滤，可能触发了安全策略")
                    
            return content
        else:
            print(f"[Kimi] 错误: API 没有返回 choices")
            print(f"[Kimi] 完整响应: {completion}")
            return None
            
    except Exception as e:
        print(f"[Kimi] 多模态 API 调用失败: {e}")
        import traceback
        traceback.print_exc()
        raise

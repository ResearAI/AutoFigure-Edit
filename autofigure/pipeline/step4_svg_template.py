"""主流程模块 - 步骤四：生成 SVG 模板"""

from pathlib import Path
from typing import Optional
from PIL import Image

from ..config import ProviderType, PlaceholderMode
from ..providers import call_llm_multimodal
from ..utils.svg_utils import extract_svg_code
from ..utils.validation import check_and_fix_svg


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
    """使用多模态 LLM 生成 SVG 代码"""
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

    base_prompt = f"""编写svg代码来实现像素级别的复现这张图片（除了图标用相同大小的矩形占位符填充之外其他文字和组件(尤其是箭头样式)都要保持一致（即灰色矩形覆盖的内容就是图标））

CRITICAL DIMENSION REQUIREMENT:
- The original image has dimensions: {figure_width} x {figure_height} pixels
- Your SVG MUST use these EXACT dimensions to ensure accurate icon placement:
  - Set viewBox="0 0 {figure_width} {figure_height}"
  - Set width="{figure_width}" height="{figure_height}"
- DO NOT scale or resize the SVG
"""

    if placeholder_mode == "box":
        with open(boxlib_path, 'r', encoding='utf-8') as f:
            boxlib_content = f.read()

        prompt_text = base_prompt + f"""
ICON COORDINATES FROM boxlib.json:
The following JSON contains precise icon coordinates detected by SAM3:
{boxlib_content}
Use these coordinates to accurately position your icon placeholders in the SVG.

Please output ONLY the SVG code, starting with <svg and ending with </svg>. Do not include any explanation or markdown formatting."""

    elif placeholder_mode == "label":
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

    else:
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
        #max_tokens=50000,
    )

    if not content:
        raise Exception('API 响应中没有内容')

    svg_code = extract_svg_code(content)

    if not svg_code:
        raise Exception('无法从响应中提取 SVG 代码')

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


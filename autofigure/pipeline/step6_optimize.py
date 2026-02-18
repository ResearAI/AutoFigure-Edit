"""主流程模块 - 步骤六：优化 SVG"""

import re
import shutil
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from ..config import ProviderType
from ..providers import call_llm_multimodal
from ..utils.svg_utils import extract_svg_code
from ..utils.validation import validate_svg_syntax, fix_svg_with_llm


def count_base64_images(svg_code: str) -> int:
    """统计 SVG 中嵌入的 base64 图片数量"""
    pattern = r'(?:href|xlink:href)=["\']data:image/[^;]+;base64,[A-Za-z0-9+/=]+'
    matches = re.findall(pattern, svg_code)
    return len(matches)


def validate_base64_images(svg_code: str, expected_count: int) -> Tuple[bool, str]:
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
        try:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPM
            drawing = svg2rlg(svg_path)
            renderPM.drawToFile(drawing, output_path, fmt="PNG")
            return output_path
        except:
            return None
    except:
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
    """使用 LLM 优化 SVG，使其与原图更加对齐"""
    print("\n" + "=" * 60)
    print("步骤 4.6：LLM 优化 SVG（位置和样式对齐）")
    print("=" * 60)
    print(f"Provider: {provider}")
    print(f"模型: {model}")
    print(f"最大迭代次数: {max_iterations}")

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

    for iteration in range(max_iterations):
        print(f"\n  优化迭代 {iteration + 1}/{max_iterations}")

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
1. **Image 1 (figure.png)**: The original target figure
2. **Image 2 (samed.png)**: The same figure with icon positions marked
3. **Image 3 (current SVG rendered as PNG)**: The current state of our SVG
4. **Current SVG code**: The SVG code that needs optimization

Please carefully compare and optimize:
- Icon positions and sizes
- Text positions, fonts, colors
- Arrow styles and positions
- Line styles and colors

**CURRENT SVG CODE:**
```xml
{current_svg}
```

**IMPORTANT:**
- Output ONLY the optimized SVG code
- Start with <svg and end with </svg>
- Do NOT include markdown formatting or explanations
- Keep all icon placeholder structures intact"""

        contents = [prompt, figure_img, samed_img, current_png_img]

        try:
            content = call_llm_multimodal(
                contents=contents,
                api_key=api_key,
                model=model,
                base_url=base_url,
                provider=provider,
                #smax_tokens=50000,
                temperature=0.3,
            )

            if not content:
                continue

            optimized_svg = extract_svg_code(content)

            if not optimized_svg:
                continue

            is_valid, errors = validate_svg_syntax(optimized_svg)

            if not is_valid:
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
                    continue

            current_svg = optimized_svg

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

    return str(output_path)


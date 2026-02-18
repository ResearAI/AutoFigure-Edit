"""主流程模块 - 步骤一：生成图片"""

from pathlib import Path
from typing import Optional
from PIL import Image

from ..config import ProviderType, USE_REFERENCE_IMAGE, REFERENCE_IMAGE_PATH
from ..providers import call_llm_image_generation


def generate_figure_from_method(
    method_text: str,
    output_path: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    use_reference_image: Optional[bool] = None,
    reference_image_path: Optional[str] = None,
    local_img_path: Optional[str] = None,
) -> str:
    """使用 LLM 生成学术风格图片"""
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
        local_img_path=local_img_path,
    )

    if img is None:
        raise Exception('API 响应中没有找到图片')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img.save(str(output_path), format='PNG')
    print(f"图片已保存: {output_path}")
    return str(output_path)


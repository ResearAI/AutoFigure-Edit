"""主流程 - 完整的 Pipeline"""

from pathlib import Path
from typing import Optional, Literal

from PIL import Image

from ..config import PROVIDER_CONFIGS, ProviderType, PlaceholderMode
from ..processors import segment_with_sam3, crop_and_remove_background
from ..utils.svg_utils import get_svg_dimensions, calculate_scale_factors
from .step1_generate import generate_figure_from_method
from .step4_svg_template import generate_svg_template
from .step5_replace_icons import replace_icons_in_svg
from .step6_optimize import optimize_svg_with_llm
from .step4_chart_code import (
    generate_chart_code,
    run_chart_code,
    optimize_chart_code_with_llm,
)


def method_to_svg(
    method_text: str,
    output_dir: str = "./output",
    api_key: str = None,
    base_url: str = None,
    provider: ProviderType = "bianxie",
    image_gen_model: str = None,
    local_img_path: str = None,
    svg_gen_model: str = None,
    sam_prompts: str = "icon",
    min_score: float = 0.5,
    sam_backend: Literal["local", "fal", "roboflow", "api"] = "local",
    sam_api_key: Optional[str] = None,
    sam_checkpoint_path: Optional[str] = None,
    sam_bpe_path: Optional[str] = None,
    sam_max_masks: int = 32,
    rmbg_model_path: Optional[str] = None,
    stop_after: int = 5,
    placeholder_mode: PlaceholderMode = "label",
    optimize_iterations: int = 2,
    merge_threshold: float = 0.9,
    task_type: Literal["icon_svg", "chart_code"] = "icon_svg",
    # 学术图代码复现是否使用 SAM3/boxlib 作为额外参考（仅 chart_code 模式有效）
    chart_use_sam: bool = True,
) -> dict:
    """完整流程：Paper Method → SVG with Icons"""
    if not api_key:
        raise ValueError("必须提供 api_key")

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
    print("Paper Method 到 SVG 图标替换流程")
    print("=" * 60)
    print(f"Provider: {provider}")
    print(f"输出目录: {output_dir}")
    print(f"生图模型: {image_gen_model}")
    print(f"SVG模型: {svg_gen_model}")
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
        local_img_path=local_img_path,
    )

    if stop_after == 1:
        return {"figure_path": str(figure_path)}

    # 如果是学术图代码复现模式且不使用 SAM3，则直接基于 figure.png 生成代码
    if task_type == "chart_code" and not chart_use_sam:
        print("\n" + "=" * 60)
        print("学术图代码复现模式（不使用 SAM3）：仅根据原图生成 Python 画图代码")
        print("=" * 60)
        chart_code_path = output_dir / "chart_code.py"
        generate_chart_code(
            figure_path=str(figure_path),
            samed_path=None,
            boxlib_path=None,
            output_path=str(chart_code_path),
            api_key=api_key,
            model=svg_gen_model,
            base_url=base_url,
            provider=provider,
        )

        # 根据 optimize_iterations 决定是否进行多轮优化
        reconstructed_chart_path: Optional[str] = None
        if optimize_iterations and optimize_iterations > 0:
            _, reconstructed_chart_path = optimize_chart_code_with_llm(
                figure_path=str(figure_path),
                chart_code_path=str(chart_code_path),
                api_key=api_key,
                model=svg_gen_model,
                base_url=base_url,
                provider=provider,
                max_iterations=optimize_iterations,
            )
        else:
            # 至少执行一次脚本，得到初始的重建结果
            reconstructed_path_obj, _ = run_chart_code(
                chart_code_path=str(chart_code_path),
                log_dir=output_dir,
                run_name="initial",
            )
            reconstructed_chart_path = str(reconstructed_path_obj)

        return {
            "figure_path": str(figure_path),
            "chart_code_path": str(chart_code_path),
            "reconstructed_chart_path": reconstructed_chart_path,
        }

    # 步骤二：SAM3 分割
    samed_path, boxlib_path, valid_boxes = segment_with_sam3(
        image_path=str(figure_path),
        output_dir=str(output_dir),
        text_prompts=sam_prompts,
        min_score=min_score,
        merge_threshold=merge_threshold,
        sam_backend=sam_backend if sam_backend != "api" else "fal",
        sam_api_key=sam_api_key,
        sam_checkpoint_path=sam_checkpoint_path,
        sam_bpe_path=sam_bpe_path,
        sam_max_masks=sam_max_masks,
    )

    if len(valid_boxes) == 0:
        print("\n警告: 没有检测到有效的图标，流程终止")
        return {"figure_path": str(figure_path), "samed_path": samed_path}

    if stop_after == 2:
        return {"figure_path": str(figure_path), "samed_path": samed_path, "boxlib_path": boxlib_path}

    # 步骤三：裁切 + 去背景（图标替换模式使用）
    icon_infos = crop_and_remove_background(
        image_path=str(figure_path),
        boxlib_path=boxlib_path,
        output_dir=str(output_dir),
        rmbg_model_path=rmbg_model_path,
    )

    if stop_after == 3:
        return {
            "figure_path": str(figure_path),
            "samed_path": samed_path,
            "boxlib_path": boxlib_path,
            "icon_infos": icon_infos,
        }

    # 学术图代码复现模式：使用 SAM3 + boxlib 作为参考生成 Python 画图代码后直接返回
    if task_type == "chart_code":
        print("\n" + "=" * 60)
        print("学术图代码复现模式（使用 SAM3）：根据原图 + SAM3 结果生成 Python 画图代码")
        print("=" * 60)
        chart_code_path = output_dir / "chart_code.py"
        generate_chart_code(
            figure_path=str(figure_path),
            samed_path=samed_path,
            boxlib_path=boxlib_path,
            output_path=str(chart_code_path),
            api_key=api_key,
            model=svg_gen_model,
            base_url=base_url,
            provider=provider,
        )

        reconstructed_chart_path: Optional[str] = None
        if optimize_iterations and optimize_iterations > 0:
            _, reconstructed_chart_path = optimize_chart_code_with_llm(
                figure_path=str(figure_path),
                chart_code_path=str(chart_code_path),
                api_key=api_key,
                model=svg_gen_model,
                base_url=base_url,
                provider=provider,
                max_iterations=optimize_iterations,
            )
        else:
            reconstructed_path_obj, _ = run_chart_code(
                chart_code_path=str(chart_code_path),
                log_dir=output_dir,
                run_name="initial",
            )
            reconstructed_chart_path = str(reconstructed_path_obj)

        return {
            "figure_path": str(figure_path),
            "samed_path": samed_path,
            "boxlib_path": boxlib_path,
            "chart_code_path": str(chart_code_path),
            "reconstructed_chart_path": reconstructed_chart_path,
        }

    # 步骤四：生成 SVG 模板（图标替换模式）
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

    # 步骤 4.6：LLM 优化 SVG 模板
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
        return {
            "figure_path": str(figure_path),
            "samed_path": samed_path,
            "template_svg_path": str(template_svg_path),
            "optimized_template_path": str(optimized_template_path),
        }

    # 步骤 4.7：坐标系对齐
    print("\n" + "-" * 50)
    print("步骤 4.7：坐标系对齐")
    print("-" * 50)

    figure_img = Image.open(figure_path)
    figure_width, figure_height = figure_img.size

    with open(optimized_template_path, 'r', encoding='utf-8') as f:
        svg_code = f.read()

    svg_width, svg_height = get_svg_dimensions(svg_code)

    if svg_width and svg_height:
        if abs(svg_width - figure_width) < 1 and abs(svg_height - figure_height) < 1:
            scale_factors = (1.0, 1.0)
        else:
            scale_factors = calculate_scale_factors(
                figure_width, figure_height, svg_width, svg_height
            )
    else:
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

    return {
        "figure_path": str(figure_path),
        "samed_path": samed_path,
        "boxlib_path": boxlib_path,
        "icon_infos": icon_infos,
        "template_svg_path": str(template_svg_path),
        "optimized_template_path": str(optimized_template_path),
        "final_svg_path": str(final_svg_path),
    }


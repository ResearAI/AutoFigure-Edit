#!/usr/bin/env python3
"""
AutoFigure - Paper Method 到 SVG 图标替换工具

重构后的主入口文件，使用模块化架构
"""

import argparse
from pathlib import Path

from autofigure.pipeline import method_to_svg, evaluate_chart_code
from autofigure.config import PROVIDER_CONFIGS
from autofigure.converters import svg_to_ppt


def main():
    parser = argparse.ArgumentParser(
        description="Paper Method 到 SVG 图标替换工具 (模块化版本)"
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
        choices=["openrouter", "bianxie", "local"],
        default="local",
        help="API 提供商（默认: local）"
    )

    # API 参数
    parser.add_argument("--api_key", default=None, help="API Key")
    parser.add_argument("--base_url", default=None, help="API base URL（默认根据 provider 自动设置）")

    # 模型参数
    parser.add_argument("--image_model", default=None, help="生图模型（默认根据 provider 自动设置）")
    parser.add_argument("--svg_model", default=None, help="SVG生成模型（默认根据 provider 自动设置）")

    # local_img_path
    parser.add_argument("--local_img_path", default=None, help="本地图片路径（可选）")

    # 任务类型参数：图标替换 / 学术图代码复现
    parser.add_argument(
        "--task_type",
        choices=["icon_svg", "chart_code"],
        default="icon_svg",
        help="任务类型：icon_svg=图标替换（默认），chart_code=学术图(折线/柱状)代码复现",
    )

    # 学术图代码复现时是否使用 SAM3/boxlib 作为额外参考
    parser.add_argument(
        "--chart_use_sam",
        action="store_true",
        help="学术图代码复现 (chart_code) 模式下，使用 SAM3 分割结果和 boxlib.json 作为额外结构参考；"
             "默认不加此参数时，仅根据原图生成 Python 画图代码，便于对比两种效果。",
    )

    # 评分参数
    parser.add_argument(
        "--enable_evaluation",
        action="store_true",
        help="启用代码评分功能（仅在 chart_code 模式下有效）",
    )
    parser.add_argument(
        "--reference_code_path",
        default=None,
        help="参考代码路径（用于评分，默认: /data/code_yjh/AutoFigure-Edit/AutoFigure-Edit/inputs/test1.py）",
    )
    parser.add_argument(
        "--reference_image_path_for_eval",
        default=None,
        help="参考图像路径（用于评分，可选）",
    )

    # Step 1 参考图片参数
    parser.add_argument("--use_reference_image", action="store_true", help="步骤一使用参考图片风格")
    parser.add_argument("--reference_image_path", default=None, help="参考图片路径（可选）")

    # SAM3 参数
    parser.add_argument("--sam_prompt", default="icon,robot,animal,person", help="SAM3 文本提示")
    parser.add_argument("--min_score", type=float, default=0.0, help="SAM3 最低置信度阈值（默认: 0.0）")
    parser.add_argument(
        "--sam_backend",
        choices=["local", "fal", "roboflow", "api"],
        default="local",
        help="SAM3 后端",
    )
    parser.add_argument("--sam_api_key", default=None, help="SAM3 API Key")
    parser.add_argument("--sam_checkpoint_path", default=None, help="SAM3 模型 checkpoint 路径（默认: /root/models/sam3/sam3.pt）")
    parser.add_argument("--sam_bpe_path", default=None, help="SAM3 BPE 词汇表路径（可选，默认使用 sam3 包内路径）")
    parser.add_argument("--sam_max_masks", type=int, default=32, help="SAM3 API 最大 masks 数")

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
        help="占位符模式（默认: label）"
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
        help="Box合并阈值（0表示不合并，默认: 0.001）"
    )

    # SVG转PPT参数
    parser.add_argument(
        "--convert_to_ppt",
        action="store_true",
        help="将生成的SVG自动转换为PPT格式"
    )
    parser.add_argument(
        "--ppt_output_path",
        default=None,
        help="PPT输出路径（默认: output_dir/output.pptx）"
    )

    args = parser.parse_args()

    # 获取 method 文本
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
        local_img_path=args.local_img_path,
        svg_gen_model=args.svg_model,
        sam_prompts=args.sam_prompt,
        min_score=args.min_score,
        sam_backend=args.sam_backend,
        sam_api_key=args.sam_api_key,
        sam_checkpoint_path=args.sam_checkpoint_path,
        sam_bpe_path=args.sam_bpe_path,
        sam_max_masks=args.sam_max_masks,
        rmbg_model_path=args.rmbg_model_path,
        stop_after=args.stop_after,
        placeholder_mode=args.placeholder_mode,
        optimize_iterations=args.optimize_iterations,
        merge_threshold=args.merge_threshold,
        task_type=args.task_type,
        chart_use_sam=args.chart_use_sam,
    )

    print("\n执行结果:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    # SVG转PPT功能
    if args.convert_to_ppt:
        # 查找生成的SVG文件
        svg_path = None
        if "final_svg_path" in result:
            svg_path = result["final_svg_path"]
        elif "svg_path" in result:
            svg_path = result["svg_path"]
        
        if svg_path and Path(svg_path).exists():
            # 设置PPT输出路径
            if args.ppt_output_path:
                ppt_path = args.ppt_output_path
            else:
                ppt_path = str(Path(args.output_dir) / "output.pptx")
            
            try:
                print(f"\n正在将SVG转换为PPT...")
                ppt_output = svg_to_ppt(svg_path, ppt_path)
                result["ppt_path"] = ppt_output
                print(f"PPT已生成: {ppt_output}")
            except Exception as e:
                print(f"\nSVG转PPT失败: {e}")
        else:
            print("\n警告: 未找到SVG文件，跳过PPT转换")

    # 如果启用评分且任务类型为 chart_code
    if args.enable_evaluation and args.task_type == "chart_code":
        if "chart_code_path" not in result:
            print("\n警告: 未生成 chart_code，跳过评分")
        else:
            # 设置默认参考代码路径
            reference_code_path = args.reference_code_path
            if reference_code_path is None:
                reference_code_path = "/data/code_yjh/AutoFigure-Edit/AutoFigure-Edit/inputs/test1.py"
            
            # 检查参考代码是否存在
            if not Path(reference_code_path).exists():
                print(f"\n错误: 参考代码文件不存在: {reference_code_path}")
            else:
                # 执行评分
                evaluation_output_path = Path(args.output_dir) / "evaluation_scores.json"
                
                try:
                    scores = evaluate_chart_code(
                        generated_code_path=result["chart_code_path"],
                        reference_code_path=reference_code_path,
                        generated_image_path=result.get("reconstructed_chart_path"),
                        reference_image_path=args.reference_image_path_for_eval,
                        output_path=str(evaluation_output_path),
                        api_key=args.api_key,
                        model=args.svg_model or PROVIDER_CONFIGS[args.provider]["default_svg_model"],
                        base_url=args.base_url or PROVIDER_CONFIGS[args.provider]["base_url"],
                        provider=args.provider,
                    )
                    
                    result["evaluation_scores"] = scores
                    result["evaluation_output_path"] = str(evaluation_output_path)
                    
                except Exception as e:
                    print(f"\n评分过程出错: {e}")


if __name__ == "__main__":
    main()

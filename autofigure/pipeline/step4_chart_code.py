"""主流程模块 - 学术图代码复现（折线图 / 柱状图 → Python 代码）"""

from pathlib import Path
from typing import Optional, Tuple
import subprocess

from PIL import Image

from ..config import ProviderType
from ..providers import call_llm_multimodal


def _extract_code_block(content: str) -> str:
    """从可能包含 Markdown 代码块的响应中提取纯 Python 代码。"""
    if not content:
        return content

    if "```" not in content:
        return content.strip()

    parts = content.split("```")
    # 典型结构：前缀 / ```python\ncode\n``` / 后缀
    if len(parts) < 2:
        return content.strip()

    code_block = parts[1]
    # 去掉可能的语言标记行
    if code_block.lstrip().startswith("python"):
        code_block = code_block.lstrip()[len("python") :]

    return code_block.strip()


def generate_chart_code(
    figure_path: str,
    samed_path: Optional[str],
    boxlib_path: Optional[str],
    output_path: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
) -> str:
    """使用多模态 LLM 生成用于复现学术图的 Python 画图代码，并在 outputs 中记录调用日志。"""
    print("\n" + "=" * 60)
    print("步骤四（学术图）：多模态调用生成 Python 画图代码")
    print("=" * 60)
    print(f"Provider: {provider}")
    print(f"模型: {model}")

    figure_img = Image.open(figure_path)

    # 是否有 SAM3 / boxlib 额外结构信息
    sam_available = (
        samed_path is not None
        and boxlib_path is not None
        and Path(samed_path).exists()
        and Path(boxlib_path).exists()
    )

    boxlib_for_prompt: Optional[str] = None
    if sam_available:
        samed_img = Image.open(samed_path)
        with open(boxlib_path, "r", encoding="utf-8") as f:
            boxlib_content = f.read()

            # 为了避免上下文超长，这里对 boxlib.json 做一个简单截断（只保留前一部分）
            MAX_BOXLIB_CHARS = 10000
            if len(boxlib_content) > MAX_BOXLIB_CHARS:
                truncated_note = "\n\n[注意] boxlib.json 过长，已在提示词中截断，仅保留前部分内容用于参考。\n"
                boxlib_for_prompt = boxlib_content[:MAX_BOXLIB_CHARS] + "\n...\n" + truncated_note
            else:
                boxlib_for_prompt = boxlib_content

    if sam_available and boxlib_for_prompt is not None:
        prompt_text = f"""你是一个资深的学术绘图助手。
现在给你一张学术图（可能是折线图、柱状图或它们的组合），你的任务是生成一段可运行的 Python 代码（使用 matplotlib 或 seaborn），尽可能复现这张图的视觉效果。

我会提供：
1. 原始图像 figure.png —— 这张图中的颜色、线型、布局才是真实目标，请严格以它为准。
2. 叠加了灰色矩形和标签的分割标注图 samed.png —— 注意：灰色矩形只是掩膜标注，用于提示哪些区域是重要元素，它的灰色填充和黑色边框不代表图表真实配色，请在配色上完全忽略这些灰块的颜色，只用它们的“位置和大小”来辅助理解结构。
3. 一个 JSON 文件 boxlib.json，里面是所有矩形的坐标信息：
{boxlib_for_prompt}

请你结合这些信息，推断图的结构和大致数据，并输出完整可运行的 Python 代码，要求：
- 识别图的类型：折线图 / 柱状图 / 多组对比 / 组合图 等
- 尽可能根据坐标轴刻度、网格线、数据点位置等，估计出每组数据的大致数值（无需完全精确，但视觉上要合理）
- 使用 matplotlib（优先）或 seaborn 来绘图
- 配色、线型、marker 等请以 figure.png 为准，不要继承 samed.png 里灰色矩形的颜色。
- 包含：
  - 必要的 import（如 matplotlib.pyplot、numpy / pandas 等）
  - 数据数组或 DataFrame 的定义
  - 画图代码（包括颜色、线型、marker、legend、轴标签、标题、网格线等）
  - 保存语句，例如：plt.savefig('reconstructed_chart.png', dpi=300, bbox_inches='tight')

重要要求：
- 只输出「纯 Python 代码」，不要任何自然语言解释，不要 Markdown 代码块标记。
- 不要调用任何大模型 API，只能使用本地库（matplotlib / seaborn / numpy / pandas 等）。
- 如果有些具体数值从图中无法精确读出，可以做合理近似，但要尽量保持曲线形状和柱子高度比例接近原图。
"""
        contents = [prompt_text, figure_img, samed_img]
    else:
        # 无 SAM3/boxlib 时的简化提示词：仅根据原始图片推断图表结构和大致数据
        prompt_text = """你是一个资深的学术绘图助手。
现在给你一张学术图（可能是折线图、柱状图或它们的组合），你的任务是生成一段可运行的 Python 代码（使用 matplotlib 或 seaborn），尽可能复现这张图的视觉效果。

我只会提供原始图像 figure.png（不包含额外的分割标注或坐标信息），请你：
- 识别图的类型：折线图 / 柱状图 / 多组对比 / 组合图 等
- 根据坐标轴刻度、网格线、数据点位置等，估计出每组数据的大致数值（无需完全精确，但视觉上要合理）
- 使用 matplotlib（优先）或 seaborn 来绘图
- 包含：
  - 必要的 import（如 matplotlib.pyplot、numpy / pandas 等）
  - 数据数组或 DataFrame 的定义
  - 画图代码（包括颜色、线型、marker、legend、轴标签、标题、网格线等）
  - 保存语句，例如：plt.savefig('reconstructed_chart.png', dpi=300, bbox_inches='tight')

重要要求：
- 只输出「纯 Python 代码」，不要任何自然语言解释，不要 Markdown 代码块标记。
- 不要调用任何大模型 API，只能使用本地库（matplotlib / seaborn / numpy / pandas 等）。
- 如果有些具体数值从图中无法精确读出，可以做合理近似，但要尽量保持曲线形状和柱子高度比例接近原图。
"""
        contents = [prompt_text, figure_img]

    # 日志目录：与 output_path 同目录
    output_path = Path(output_path)
    log_dir = output_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # 记录 prompt 到日志文件
    prompt_log_path = log_dir / "chart_code_prompt.txt"
    with open(prompt_log_path, "w", encoding="utf-8") as f:
        f.write(
            "=== Chart Code Prompt ===\n"
            f"provider: {provider}\n"
            f"model: {model}\n"
            f"figure_path: {figure_path}\n"
            f"samed_path: {samed_path}\n"
            f"boxlib_path: {boxlib_path}\n"
            "-------------------------\n\n"
        )
        f.write(prompt_text)

    print(f"发送多模态请求到: {base_url}")

    try:
        content = call_llm_multimodal(
            contents=contents,
            api_key=api_key,
            model=model,
            base_url=base_url,
            provider=provider,
            max_tokens=50000,
            )
    except Exception as e:
        # 记录异常信息，方便排查
        error_log_path = log_dir / "chart_code_error.txt"
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write("=== Chart Code API 调用异常 ===\n")
            f.write(repr(e))
            f.write("\n")
        raise

    # 记录原始响应（无论是否为空）
    response_log_path = log_dir / "chart_code_raw_response.txt"
    with open(response_log_path, "w", encoding="utf-8") as f:
        f.write("=== Chart Code Raw Response ===\n")
        if content is None:
            f.write("[content is None]\n")
        else:
            try:
                f.write(str(content))
            except Exception as e:  # 防御性写入
                f.write(f"[无法直接写入 content，错误: {e!r}]\n")

    if not content or (isinstance(content, str) and not content.strip()):
        # 保留日志文件，抛出更明确的异常，提示去 outputs 查看
        raise Exception(
            "API 响应中没有内容，已将请求和原始响应分别保存到 "
            f"'{prompt_log_path.name}' 和 '{response_log_path.name}' 方便排查。"
        )

    # 如果模型返回的是 Markdown 代码块，提取出纯 Python 代码
    cleaned = _extract_code_block(content) if isinstance(content, str) else str(content)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"学术图 Python 代码已保存: {output_path}")
    return str(output_path)


def run_chart_code(
    chart_code_path: str,
    log_dir: Optional[Path] = None,
    run_name: str = "final",
) -> Tuple[Path, int]:
    """执行当前的 chart_code.py，返回重建图路径和退出码，并记录运行日志。"""
    path_obj = Path(chart_code_path)
    work_dir = path_obj.parent
    if log_dir is None:
        log_dir = work_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"chart_code_run_{run_name}.log"

    print(f"\n执行学术图代码脚本: {path_obj.name} (run_name={run_name})")
    try:
        proc = subprocess.run(
            ["python3", path_obj.name],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
        )
    except Exception as e:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=== Chart Code Run Exception ===\n")
            f.write(repr(e))
            f.write("\n")
        print(f"运行 chart_code.py 发生异常，详情见日志: {log_path}")
        # 即使异常，也返回一个理论上的输出路径，方便上层逻辑继续处理
        return work_dir / "reconstructed_chart.png", -1

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== Chart Code Run Log ===\n")
        f.write(f"returncode: {proc.returncode}\n")
        f.write("\n--- STDOUT ---\n")
        f.write(proc.stdout or "")
        f.write("\n--- STDERR ---\n")
        f.write(proc.stderr or "")

    if proc.returncode != 0:
        print(f"运行 chart_code.py 退出码非零 ({proc.returncode})，请查看日志: {log_path}")
    else:
        print(f"chart_code.py 运行完成，日志: {log_path}")

    reconstructed_path = work_dir / "reconstructed_chart.png"
    if not reconstructed_path.exists():
        print(f"警告: 未找到重建图文件: {reconstructed_path}")

    return reconstructed_path, proc.returncode


def optimize_chart_code_with_llm(
    figure_path: str,
    chart_code_path: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    max_iterations: int = 1,
) -> Tuple[str, Optional[str]]:
    """多轮执行 + 优化学术图 Python 代码。

    - 每一轮：
      1）执行当前 chart_code.py 得到 reconstructed_chart.png（如有）
      2）把原图 + 当前重建图 + 现有代码发给 LLM，请它给出更接近原图的新版代码
    - 最终返回：最终代码路径 & 最后一轮重建图路径（如有）
    """
    if max_iterations <= 0:
        return chart_code_path, None

    output_dir = Path(chart_code_path).parent
    figure_img = Image.open(figure_path)

    latest_reconstructed: Optional[Path] = None

    for i in range(max_iterations):
        iter_idx = i + 1
        print("\n" + "=" * 60)
        print(f"学术图代码优化迭代 第 {iter_idx}/{max_iterations} 轮")
        print("=" * 60)

        # 1) 先执行当前代码，得到重建图
        reconstructed_path, returncode = run_chart_code(
            chart_code_path,
            log_dir=output_dir,
            run_name=f"iter{iter_idx}",
        )
        latest_reconstructed = reconstructed_path

        # 尝试加载当前重建图（可能不存在）
        reconstructed_img: Optional[Image.Image] = None
        if reconstructed_path.exists():
            try:
                reconstructed_img = Image.open(reconstructed_path)
            except Exception as e:  # noqa: BLE001
                print(f"警告: 无法打开重建图 {reconstructed_path}: {e!r}")

        # 2) 读入当前代码
        with open(chart_code_path, "r", encoding="utf-8") as f:
            current_code = f.read()

        # 3) 构造优化提示词
        prompt_text = f"""你是一个资深的学术绘图代码优化助手。
我有一张目标学术图（figure.png）以及当前版本的 Python 画图代码（使用 matplotlib / seaborn），
并根据这段代码生成了一张重建图（reconstructed_chart.png，如果存在）。

你的任务是：在保持代码结构清晰、可运行的前提下，改写和优化这段 Python 代码，使生成的图在视觉上尽量接近原始目标图。

当前代码如下（请完整理解后再改写）：
```python
{current_code}
```

如果上一轮运行代码时有报错或未生成重建图，请尽量根据错误信息推断问题并修复，使代码可以顺利运行并生成图像。

重要要求：
- 输出必须是「完整的、可直接运行的」Python 代码，不要任何解释文字，也不要 Markdown 代码块标记。
- 可以调整数据（在合理范围内近似推断）、配色、线型、坐标轴范围、刻度、网格、图例等，使结果更接近原图。
- 代码中必须包含保存图片的语句，例如：plt.savefig('reconstructed_chart.png', dpi=300, bbox_inches='tight')。
- 请确保导入依赖完整（如 import numpy as np, import matplotlib.pyplot as plt 等），代码不依赖外部输入。
"""

        contents = [prompt_text, figure_img]
        if reconstructed_img is not None:
            contents.append(reconstructed_img)

        # 日志文件
        prompt_log_path = output_dir / f"chart_code_opt_prompt_iter{iter_idx}.txt"
        raw_log_path = output_dir / f"chart_code_opt_raw_response_iter{iter_idx}.txt"

        with open(prompt_log_path, "w", encoding="utf-8") as f:
            f.write("=== Chart Code Optimize Prompt ===\n")
            f.write(f"iteration: {iter_idx}/{max_iterations}\n")
            f.write(f"figure_path: {figure_path}\n")
            f.write(f"chart_code_path: {chart_code_path}\n")
            f.write(f"reconstructed_path: {reconstructed_path}\n")
            f.write("-------------------------\n\n")
            f.write(prompt_text)

        print(f"发送优化多模态请求到: {base_url} (iteration={iter_idx})")

        try:
            content = call_llm_multimodal(
                contents=contents,
                api_key=api_key,
                model=model,
                base_url=base_url,
                provider=provider,
                max_tokens=50000,
            )
        except Exception as e:  # noqa: BLE001
            with open(raw_log_path, "w", encoding="utf-8") as f:
                f.write("=== Chart Code Optimize Exception ===\n")
                f.write(repr(e))
                f.write("\n")
            print(f"优化第 {iter_idx} 轮调用失败，详情见日志: {raw_log_path}，将提前结束优化。")
            break

        with open(raw_log_path, "w", encoding="utf-8") as f:
            f.write("=== Chart Code Optimize Raw Response ===\n")
            if content is None:
                f.write("[content is None]\n")
            else:
                try:
                    f.write(str(content))
                except Exception as e:  # noqa: BLE001
                    f.write(f"[无法直接写入 content，错误: {e!r}]\n")

        if not content or (isinstance(content, str) and not content.strip()):
            print(f"优化第 {iter_idx} 轮响应为空，将提前结束优化。")
            break

        # 提取纯代码
        cleaned = _extract_code_block(content) if isinstance(content, str) else str(content)
        if not cleaned.strip():
            print(f"优化第 {iter_idx} 轮未能提取到有效代码，将提前结束优化。")
            break

        # 保存本轮前后的代码快照
        before_path = output_dir / f"chart_code_iter{iter_idx}_before.py"
        after_path = output_dir / f"chart_code_iter{iter_idx}_after.py"
        with open(before_path, "w", encoding="utf-8") as f:
            f.write(current_code)
        with open(after_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        # 覆盖主代码文件
        with open(chart_code_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        print(f"第 {iter_idx} 轮优化完成，更新后的代码已写入: {chart_code_path}")

    # 返回最终代码路径及最后一轮的重建图路径（字符串形式，便于序列化）
    return str(chart_code_path), str(latest_reconstructed) if latest_reconstructed else None

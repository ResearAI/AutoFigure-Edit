"""评分模块 - 评估生成的 Python 代码质量"""

from pathlib import Path
from typing import Optional, Dict
import json

from PIL import Image

from ..config import ProviderType
from ..providers import call_llm_text


def evaluate_chart_code(
    generated_code_path: str,
    reference_code_path: str,
    generated_image_path: Optional[str],
    reference_image_path: Optional[str],
    output_path: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
) -> Dict[str, float]:
    """使用 LLM 评估生成的 Python 代码质量（仅基于代码，不使用图片）。
    
    评估维度：
    1. 参数准确度 (Parameter Accuracy): 数据、坐标轴、刻度等参数的准确性
    2. 视觉相似度 (Visual Similarity): 通过代码推断视觉效果的相似程度
    3. 代码可执行性 (Executability): 代码是否能正常运行
    
    返回：包含各维度分数和总分的字典
    """
    print("\n" + "=" * 60)
    print("步骤七：评估生成的 Python 代码")
    print("=" * 60)
    print(f"Provider: {provider}")
    print(f"模型: {model}")
    print(f"生成代码: {generated_code_path}")
    print(f"参考代码: {reference_code_path}")
    print("注意: 图片对比已禁用，仅基于代码进行评估")
    
    # 读取代码文件
    with open(generated_code_path, "r", encoding="utf-8") as f:
        generated_code = f.read()
    
    with open(reference_code_path, "r", encoding="utf-8") as f:
        reference_code = f.read()
    
    # 构建评估提示词（仅基于代码，3个维度）
    prompt_text = f"""你是一个专业的代码评审专家，现在需要你对生成的 Python 画图代码进行全面评分。

我会提供：
1. 生成的代码（待评估）
2. 参考代码（标准答案）

请仅通过阅读和分析代码，从以下三个维度进行评分（每项满分25分，总分75分）：

**1. 参数准确度 (Parameter Accuracy, 25分)**
- 数据数组/DataFrame 的数值是否与参考代码接近
- 坐标轴范围、刻度设置是否准确
- 图表类型（折线图/柱状图等）是否正确
- 数据点数量和分布是否合理

**2. 视觉相似度 (Visual Similarity, 25分)**
- 配色方案是否接近参考代码
- 线型、marker 样式是否匹配
- 图例、标签、标题等元素是否完整
- 整体布局和视觉效果（通过代码推断）是否相似
- 字体大小、网格线等细节是否合理

**3. 代码可执行性 (Executability, 25分)**
- 代码是否包含所有必要的 import 语句
- 是否有语法错误
- 是否能独立运行（不依赖外部输入）
- 是否正确保存输出图像

---

**生成的代码：**
```python
{generated_code}
```

**参考代码：**
```python
{reference_code}
```

---

请严格按照以下 JSON 格式输出评分结果（不要有任何其他文字）：

{{
  "parameter_accuracy": {{
    "score": <0-25的数字>,
    "comment": "<简短评价，50字以内>"
  }},
  "visual_similarity": {{
    "score": <0-25的数字>,
    "comment": "<简短评价，50字以内>"
  }},
  "executability": {{
    "score": <0-25的数字>,
    "comment": "<简短评价，50字以内>"
  }},
  "total_score": <总分，0-75>,
  "overall_comment": "<总体评价，100字以内>"
}}
"""
    
    # 日志目录
    output_path = Path(output_path)
    log_dir = output_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 记录 prompt
    prompt_log_path = log_dir / "evaluation_prompt.txt"
    with open(prompt_log_path, "w", encoding="utf-8") as f:
        f.write("=== Evaluation Prompt ===\n")
        f.write(f"provider: {provider}\n")
        f.write(f"model: {model}\n")
        f.write(f"generated_code_path: {generated_code_path}\n")
        f.write(f"reference_code_path: {reference_code_path}\n")
        f.write("mode: text-only (no images)\n")
        f.write("dimensions: 3 (no code style)\n")
        f.write("-------------------------\n\n")
        f.write(prompt_text)
    
    print(f"发送评估请求到: {base_url}")
    
    try:
        content = call_llm_text(
            prompt=prompt_text,
            api_key=api_key,
            model=model,
            base_url=base_url,
            provider=provider,
            
        )
    except Exception as e:
        error_log_path = log_dir / "evaluation_error.txt"
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write("=== Evaluation API 调用异常 ===\n")
            f.write(repr(e))
            f.write("\n")
        raise
    
    # 记录原始响应
    response_log_path = log_dir / "evaluation_raw_response.txt"
    with open(response_log_path, "w", encoding="utf-8") as f:
        f.write("=== Evaluation Raw Response ===\n")
        if content is None:
            f.write("[content is None]\n")
        else:
            f.write(str(content))
    
    if not content or (isinstance(content, str) and not content.strip()):
        raise Exception(
            f"API 响应为空，详情见日志: {response_log_path}"
        )
    
    # 解析 JSON 响应
    try:
        # 提取 JSON（可能包含在 Markdown 代码块中）
        content_str = str(content).strip()
        if "```json" in content_str:
            start = content_str.find("```json") + 7
            end = content_str.find("```", start)
            content_str = content_str[start:end].strip()
        elif "```" in content_str:
            start = content_str.find("```") + 3
            end = content_str.find("```", start)
            content_str = content_str[start:end].strip()
        
        scores = json.loads(content_str)
    except json.JSONDecodeError as e:
        print(f"警告: 无法解析 JSON 响应: {e}")
        print(f"原始响应: {content}")
        # 返回默认分数
        scores = {
            "parameter_accuracy": {"score": 0, "comment": "解析失败"},
            "visual_similarity": {"score": 0, "comment": "解析失败"},
            "executability": {"score": 0, "comment": "解析失败"},
            "total_score": 0,
            "overall_comment": "评分响应解析失败"
        }
    
    # 保存评分结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    
    print(f"\n评分结果已保存: {output_path}")
    print("\n" + "=" * 60)
    print("评分结果：")
    print("=" * 60)
    print(f"参数准确度: {scores['parameter_accuracy']['score']}/25")
    print(f"  评价: {scores['parameter_accuracy']['comment']}")
    print(f"\n视觉相似度: {scores['visual_similarity']['score']}/25")
    print(f"  评价: {scores['visual_similarity']['comment']}")
    print(f"\n代码可执行性: {scores['executability']['score']}/25")
    print(f"  评价: {scores['executability']['comment']}")
    print(f"\n总分: {scores['total_score']}/75")
    print(f"总体评价: {scores['overall_comment']}")
    print("=" * 60)
    
    return scores

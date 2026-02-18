"""SVG 验证和修复工具"""

from typing import Optional, Tuple, List

from ..config import ProviderType
from ..providers import call_llm_text
from .svg_utils import extract_svg_code


def validate_svg_syntax(svg_code: str) -> Tuple[bool, List[str]]:
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
    errors: List[str],
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
                print(f"    修复后仍有 {len(new_errors)} 个错误")
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


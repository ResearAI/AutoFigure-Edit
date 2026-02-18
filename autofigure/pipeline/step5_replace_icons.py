"""主流程模块 - 步骤五：图标替换"""

import base64
import io
import re
from pathlib import Path
from typing import List, Tuple

from PIL import Image


def replace_icons_in_svg(
    template_svg_path: str,
    icon_infos: List[dict],
    output_path: str,
    scale_factors: Tuple[float, float] = (1.0, 1.0),
    match_by_label: bool = True,
) -> str:
    """将透明背景图标替换到 SVG 中的占位符"""
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

                g_tag_match = re.match(r'<g[^>]*>', g_content, re.IGNORECASE)
                translate_x, translate_y = 0.0, 0.0
                if g_tag_match:
                    g_tag = g_tag_match.group(0)
                    transform_pattern = r'transform=["\'][^"\']*translate\s*\(\s*([\d.-]+)[\s,]+([\d.-]+)\s*\)'
                    transform_match = re.search(transform_pattern, g_tag, re.IGNORECASE)
                    if transform_match:
                        translate_x = float(transform_match.group(1))
                        translate_y = float(transform_match.group(2))

                rect_patterns = [
                    r'<rect[^>]*\bx=["\']?([\d.]+)["\']?[^>]*\by=["\']?([\d.]+)["\']?[^>]*\bwidth=["\']?([\d.]+)["\']?[^>]*\bheight=["\']?([\d.]+)["\']?',
                    r'<rect[^>]*\bwidth=["\']?([\d.]+)["\']?[^>]*\bheight=["\']?([\d.]+)["\']?[^>]*\bx=["\']?([\d.]+)["\']?[^>]*\by=["\']?([\d.]+)["\']?',
                ]

                rect_info = None
                for rp in rect_patterns:
                    rect_match = re.search(rp, g_content, re.IGNORECASE)
                    if rect_match:
                        groups = rect_match.groups()
                        if len(groups) == 4:
                            if 'width' in rp[:50]:
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
                    x = rect_info['x'] + translate_x
                    y = rect_info['y'] + translate_y
                    width, height = rect_info['width'], rect_info['height']

                    image_tag = f'<image id="icon_{label_clean}" x="{x}" y="{y}" width="{width}" height="{height}" href="data:image/png;base64,{icon_b64}" preserveAspectRatio="xMidYMid meet"/>'
                    svg_content = svg_content.replace(g_content, image_tag)
                    print(f"  {label}: 替换成功 (序号匹配 <g>) at ({x}, {y}) size {width}x{height}")
                    replaced = True

            # 方式2：查找包含 label 文本的 <text> 元素
            if not replaced:
                text_patterns = [
                    rf'<text[^>]*>[^<]*{re.escape(label)}[^<]*</text>',
                    rf'<text[^>]*>[^<]*&lt;AF&gt;{label_clean[2:]}[^<]*</text>',
                ]

                for tp in text_patterns:
                    text_match = re.search(tp, svg_content, re.IGNORECASE)
                    if text_match:
                        text_pos = text_match.start()
                        preceding_svg = svg_content[:text_pos]

                        rect_matches = list(re.finditer(r'<rect[^>]*/?\s*>', preceding_svg, re.IGNORECASE))
                        if rect_matches:
                            last_rect = rect_matches[-1]
                            rect_content = last_rect.group(0)

                            x_match = re.search(r'\bx=["\']?([\d.]+)', rect_content)
                            y_match = re.search(r'\by=["\']?([\d.]+)', rect_content)
                            w_match = re.search(r'\bwidth=["\']?([\d.]+)', rect_content)
                            h_match = re.search(r'\bheight=["\']?([\d.]+)', rect_content)

                            if all([x_match, y_match, w_match, h_match]):
                                x = float(x_match.group(1))
                                y = float(y_match.group(1))
                                width = float(w_match.group(1))
                                height = float(h_match.group(1))

                                image_tag = f'<image id="icon_{label_clean}" x="{x}" y="{y}" width="{width}" height="{height}" href="data:image/png;base64,{icon_b64}" preserveAspectRatio="xMidYMid meet"/>'

                                svg_content = svg_content.replace(text_match.group(0), '')
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

            svg_content = svg_content.replace('</svg>', f'  {image_tag}\n</svg>')
            print(f"  {label}: 追加到 SVG at ({x1:.1f}, {y1:.1f}) (未找到匹配的占位符)")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)

    print(f"最终 SVG 已保存: {output_path}")
    return str(output_path)


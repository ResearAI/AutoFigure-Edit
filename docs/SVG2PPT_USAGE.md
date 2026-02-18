# SVG转PPT功能说明

## 功能概述

AutoFigure现在支持将生成的SVG文件自动转换为PowerPoint (PPT) 格式，方便在演示文稿中使用。

## 使用方法

### 方法1: 在主流程中自动转换

在运行`autofigure_main.py`时添加`--convert_to_ppt`参数：

```bash
python autofigure_main.py \
    --method_file ./paper.txt \
    --output_dir ./output \
    --convert_to_ppt \
    --ppt_output_path ./output/my_figure.pptx
```

参数说明：
- `--convert_to_ppt`: 启用SVG转PPT功能
- `--ppt_output_path`: (可选) 指定PPT输出路径，默认为`output_dir/output.pptx`

### 方法2: 单独转换已有的SVG文件

使用Python代码直接调用转换函数：

```python
from autofigure.converters import svg_to_ppt

# 转换SVG为PPT
svg_path = "./outputs/demo/final.svg"
ppt_path = "./outputs/demo/output.pptx"

result = svg_to_ppt(svg_path, ppt_path)
print(f"PPT已生成: {result}")
```

或使用测试脚本：

```bash
bash examples/test_svg2ppt.sh
```

## 支持的SVG元素

转换器支持以下SVG元素：

- **基本形状**: 矩形(rect)、圆形(circle)、椭圆(ellipse)
- **线条**: 直线(line)、折线(polyline)、多边形(polygon)
- **路径**: path元素（支持M、L、Z命令）
- **文本**: text元素（支持字体大小、颜色、对齐方式、粗体）
- **图片**: image元素（支持base64编码的图片）
- **箭头**: 支持带箭头标记的线条

## 示例

### 完整流程示例

```bash
# 从method文本生成SVG并自动转换为PPT
python autofigure_main.py \
    --method_text "Our method uses a neural network to process images" \
    --output_dir ./output \
    --provider local \
    --convert_to_ppt
```

### 仅转换示例

```bash
# 使用测试脚本转换demo中的SVG
bash examples/test_svg2ppt.sh
```

## 技术细节

- SVG坐标单位(px)会自动转换为PPT的EMU单位
- 颜色支持十六进制格式和常见颜色名称
- 文本定位会根据`text-anchor`属性自动调整
- 箭头通过XML直接添加到连接线

## 依赖

转换功能需要以下Python包：
- `python-pptx`: PPT文件操作
- `lxml`: XML/SVG解析

## 注意事项

1. 复杂的SVG效果（如渐变、滤镜等）可能无法完全转换
2. 转换后的PPT中，所有元素都在同一张幻灯片上
3. 建议在转换前检查SVG文件是否正确生成


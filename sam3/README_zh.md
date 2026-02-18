# AutoFigure-Edit: 学术图表自动生成与编辑系统

## 项目简介

AutoFigure-Edit 是一个基于多模态大语言模型的学术图表自动生成系统。该系统能够根据论文方法描述文本，自动生成高质量的学术风格图表，并通过智能分割、图标提取和 SVG 重构技术，生成可编辑的矢量图形。

### 核心特性

- **智能图表生成**：基于论文方法描述自动生成学术风格图表
- **精准图标分割**：使用 SAM3 模型进行多提示词图标检测
- **智能去背景**：采用 BRIA-RMBG 2.0 模型进行高质量背景移除
- **SVG 矢量化**：将位图转换为可编辑的 SVG 矢量图形
- **自动优化对齐**：通过 LLM 迭代优化图表布局和样式

---

## 执行流程

### 完整流程图

```
论文方法文本
    ↓
[步骤1] LLM 生成学术图表 (figure.png)
    ↓
[步骤2] SAM3 多提示词分割 + Box 合并 (samed.png + boxlib.json)
    ↓
[步骤3] 图标裁切 + RMBG2 去背景 (icons/*.png)
    ↓
[步骤4] 多模态 LLM 生成 SVG 模板 (template.svg)
    ↓
[步骤4.5] XML 语法验证 + LLM 修复
    ↓
[步骤4.6] LLM 迭代优化对齐 (optimized_template.svg)
    ↓
[步骤4.7] 坐标系对齐计算
    ↓
[步骤5] 图标替换到 SVG (final.svg)
```

### 详细步骤说明

#### 步骤 1：LLM 生成学术图表

**输入**：论文方法描述文本  
**输出**：`figure.png` (1024×1024 像素)

使用多模态大语言模型（如 Gemini）根据论文方法描述生成学术风格的图表。系统支持参考图片风格迁移，可以模仿特定的视觉风格。

**提示词策略**：
- 强调学术期刊风格
- 要求清晰的方法流程可视化
- 支持可爱角色和图标元素

#### 步骤 2：SAM3 多提示词分割 + Box 合并

**输入**：`figure.png`  
**输出**：`samed.png`（标记图）、`boxlib.json`（坐标信息）

使用 SAM3（Segment Anything Model 3）进行智能图标检测：

**多提示词检测**：
- 支持逗号分隔的多个检测提示词（如 `icon,robot,animal,person`）
- 对每个提示词分别检测，然后合并结果
- 自动记录每个 box 的来源提示词

**Box 合并算法**：
- 计算重叠比例 = 交集面积 / 较小 box 面积
- 默认阈值 0.9，超过阈值则合并为最小包围矩形
- 跨提示词检测结果自动去重

**标记样式**：
- 灰色填充 (#808080)
- 黑色边框 (width=3)
- 白色居中序号标签 (`<AF>01`, `<AF>02`, ...)

**本次执行结果**：
- 使用提示词：`icon, robot, animal, person`
- 检测到 7 个有效图标区域
- 置信度范围：0.728 - 0.895

#### 步骤 3：图标裁切 + RMBG2 去背景

**输入**：`figure.png` + `boxlib.json`  
**输出**：`icons/icon_AF01_nobg.png` ~ `icon_AF07_nobg.png`

根据检测到的坐标信息裁切图标区域，并使用 BRIA-RMBG 2.0 模型进行高质量背景移除：

- 裁切原图对应区域
- 使用深度学习模型精准抠图
- 保留透明通道，生成 PNG 格式
- 自动适配 GPU/CPU 运行环境

**本次执行结果**：
- 生成 7 个透明背景图标
- 每个图标保留原始尺寸和细节

#### 步骤 4：多模态 LLM 生成 SVG 模板

**输入**：`figure.png` + `samed.png` + `boxlib.json`  
**输出**：`template.svg`

使用多模态 LLM 分析原图和标记图，生成 SVG 代码：

**占位符模式**（本次使用 `label` 模式）：
- `none`：无特殊样式
- `box`：传入 boxlib 坐标给 LLM
- `label`：要求 SVG 占位符与 samed.png 样式一致（推荐）

**提示词要求**：
- 像素级复现原图（除图标外）
- 保持精确尺寸（viewBox="0 0 1024 1024"）
- 占位符使用灰色矩形 + 黑色边框 + 序号标签
- 每个占位符包裹在 `<g id="AF01">` 元素中

#### 步骤 4.5：XML 语法验证 + LLM 修复

**输入**：`template.svg`  
**输出**：语法正确的 SVG 代码

使用 lxml 解析器验证 SVG 语法：
- 检测未闭合标签、非法属性、未转义字符等错误
- 如发现错误，自动调用 LLM 修复
- 最多重试 3 次，确保输出有效的 XML

#### 步骤 4.6：LLM 迭代优化对齐

**输入**：`figure.png` + `samed.png` + `template.svg`  
**输出**：`optimized_template.svg`

通过多轮迭代优化 SVG 与原图的对齐：

**优化维度**：
1. **位置对齐**：图标、文字、箭头、线条的位置
2. **样式对齐**：字体大小、颜色、线条粗细、箭头样式

**优化流程**：
- 将当前 SVG 渲染为 PNG
- 与原图对比，生成优化提示
- LLM 输出改进的 SVG 代码
- 验证语法和 base64 图片完整性
- 重复迭代（可配置次数，默认 2 次）

**本次执行**：迭代次数设为 0，跳过优化（直接使用模板）

#### 步骤 4.7：坐标系对齐

**输入**：`figure.png` 尺寸 + SVG viewBox  
**输出**：缩放因子 (scale_x, scale_y)

计算从位图像素坐标到 SVG 坐标的映射关系：
- 提取 SVG 的 viewBox 或 width/height 属性
- 计算缩放因子：`scale_x = svg_width / figure_width`
- 如果尺寸匹配，使用 1:1 映射

#### 步骤 5：图标替换到 SVG

**输入**：`optimized_template.svg` + 透明图标  
**输出**：`final.svg`

将透明背景图标嵌入到 SVG 占位符位置：

**匹配策略**（label 模式）：
1. 查找 `<g id="AF01">` 元素，提取 rect 尺寸和位置
2. 处理 transform="translate(x, y)" 变换
3. 查找包含 `<AF>01` 文本的 text 元素附近的 rect
4. 回退到坐标匹配（精确匹配 → 近似匹配）
5. 无匹配则追加到 SVG 末尾

**图标嵌入**：
- 将 PNG 转换为 base64 编码
- 使用 `<image>` 标签替换占位符
- 设置 `preserveAspectRatio="xMidYMid meet"` 保持比例

---

## 使用的模型和提示词

### 模型配置

| 步骤 | 模型 | 用途 |
|------|------|------|
| 步骤 1 | `gemini-2.5-banana-web` | 图表生成 |
| 步骤 2 | SAM3 (本地部署) | 图标分割 |
| 步骤 3 | BRIA-RMBG 2.0 | 背景移除 |
| 步骤 4 | `kimi-k2.5` | SVG 生成 |
| 步骤 4.5/4.6 | `kimi-k2.5` | SVG 修复和优化 |

### 关键提示词

#### 图表生成提示词（步骤 1）

```
Generate a professional academic journal style figure for the paper below 
so as to visualize the method it proposes, below is the method section of 
this paper:

{method_text}

The figure should be engaging and using academic journal style with cute 
characters.
```

#### SAM3 检测提示词（步骤 2）

```
icon,robot,animal,person
```

多提示词策略可以提高检测召回率，覆盖不同类型的视觉元素。

#### SVG 生成提示词（步骤 4）

```
编写svg代码来实现像素级别的复现这张图片（除了图标用相同大小的矩形占位符
填充之外其他文字和组件(尤其是箭头样式)都要保持一致（即灰色矩形覆盖的内容
就是图标））

CRITICAL DIMENSION REQUIREMENT:
- The original image has dimensions: 1024 x 1024 pixels
- Your SVG MUST use these EXACT dimensions to ensure accurate icon placement:
  - Set viewBox="0 0 1024 1024"
  - Set width="1024" height="1024"
- DO NOT scale or resize the SVG

PLACEHOLDER STYLE REQUIREMENT:
Look at the second image (samed.png) - each icon area is marked with a gray 
rectangle (#808080), black border, and a centered label like <AF>01, <AF>02, etc.

Your SVG placeholders MUST match this exact style:
- Rectangle with fill="#808080" and stroke="black" stroke-width="2"
- Centered white text showing the same label (<AF>01, <AF>02, etc.)
- Wrap each placeholder in a <g> element with id matching the label (e.g., id="AF01")
```

#### SVG 优化提示词（步骤 4.6）

```
You are an expert SVG optimizer. Compare the current SVG rendering with the 
original figure and optimize the SVG code to better match the original.

Please carefully compare and check the following TWO MAJOR ASPECTS with EIGHT 
KEY POINTS:

## ASPECT 1: POSITION (位置)
1. Icons (图标): Are icon placeholder positions matching the original?
2. Text (文字): Are text elements positioned correctly?
3. Arrows (箭头): Are arrows starting/ending at correct positions?
4. Lines/Borders (线条): Are lines and borders aligned properly?

## ASPECT 2: STYLE (样式)
5. Icons (图标): Icon placeholder sizes, proportions
6. Text (文字): Font sizes, colors, weights
7. Arrows (箭头): Arrow styles, thicknesses, colors
8. Lines/Borders (线条): Line styles, colors, stroke widths
```

---

## 执行结果展示

### 输出文件结构

```
outputs/demo/
├── figure.png              # 步骤1：原始生成图表
├── samed.png               # 步骤2：标记图（灰色占位符+序号）
├── boxlib.json             # 步骤2：检测坐标信息
├── icons/                  # 步骤3：提取的图标
│   ├── icon_AF01.png       # 裁切的原始图标
│   ├── icon_AF01_nobg.png  # 去背景后的透明图标
│   ├── icon_AF02.png
│   ├── icon_AF02_nobg.png
│   └── ...                 # 共7个图标
├── template.svg            # 步骤4：初始SVG模板
├── optimized_template.svg  # 步骤4.6：优化后的SVG模板
└── final.svg               # 步骤5：最终SVG（嵌入图标）
```

### 检测结果统计

| 指标 | 数值 |
|------|------|
| 图表尺寸 | 1024 × 1024 px |
| 检测提示词 | 4 个 (icon, robot, animal, person) |
| 检测到的图标 | 7 个 |
| 平均置信度 | 0.855 |
| 最高置信度 | 0.895 (AF04) |
| 最低置信度 | 0.728 (AF01) |

### 图标坐标信息

| 标签 | 位置 (x1, y1, x2, y2) | 尺寸 | 置信度 | 来源提示词 |
|------|----------------------|------|--------|-----------|
| AF01 | (861, 860, 938, 924) | 77×64 | 0.728 | icon |
| AF02 | (76, 206, 125, 248) | 49×42 | 0.895 | icon |
| AF03 | (78, 323, 126, 361) | 48×38 | 0.885 | icon |
| AF04 | (622, 205, 672, 248) | 50×43 | 0.895 | icon |
| AF05 | (627, 460, 659, 488) | 32×28 | 0.794 | icon |
| AF06 | (362, 323, 411, 361) | 49×38 | 0.895 | icon |
| AF07 | (221, 324, 270, 362) | 49×38 | 0.890 | icon |

### 可视化结果

#### 原始生成图表 (figure.png)
![原始图表](outputs/demo/figure.png)

#### 标记图 (samed.png)
![标记图](outputs/demo/samed.png)

#### 最终 SVG (final.svg)
![最终SVG](outputs/demo/final.svg)

#### 提取的透明图标示例
| AF01 | AF02 | AF03 | AF04 |
|------|------|------|------|
| ![](outputs/demo/icons/icon_AF01_nobg.png) | ![](outputs/demo/icons/icon_AF02_nobg.png) | ![](outputs/demo/icons/icon_AF03_nobg.png) | ![](outputs/demo/icons/icon_AF04_nobg.png) |

---

## 相比直接使用 Banana 生图的优势

### 1. **可编辑性**

| 特性 | Banana 直接生图 | AutoFigure-Edit |
|------|----------------|-----------------|
| 输出格式 | PNG/JPG (位图) | SVG (矢量图) |
| 文字修改 | ❌ 需要重新生成 | ✅ 直接编辑 SVG 文本 |
| 颜色调整 | ❌ 需要图像处理 | ✅ 修改 fill/stroke 属性 |
| 布局调整 | ❌ 无法调整 | ✅ 移动 SVG 元素 |
| 图标替换 | ❌ 无法替换 | ✅ 替换 base64 或外部链接 |
| 缩放质量 | ❌ 放大模糊 | ✅ 矢量无损缩放 |

### 2. **精准控制**

- **尺寸精确**：SVG 坐标精确到像素，位图生成存在随机性
- **样式一致**：可以统一修改所有元素的样式（字体、颜色、线宽）
- **版本管理**：SVG 是文本格式，便于 Git 版本控制和 diff 对比
- **自动化集成**：可以编程方式批量修改 SVG 元素

### 3. **图标复用**

- **提取图标库**：自动提取所有图标，建立可复用的图标库
- **跨图表复用**：提取的透明图标可用于其他图表
- **风格统一**：确保多个图表使用相同的图标风格
- **独立优化**：可以单独优化每个图标的质量

### 4. **迭代优化**

- **渐进式改进**：通过 LLM 多轮迭代优化布局和样式
- **对比验证**：自动对比原图和 SVG 渲染结果
- **语法保证**：XML 解析器确保输出有效的 SVG
- **人工介入**：可以在任何步骤手动调整后继续流程

### 5. **学术出版友好**

- **期刊要求**：多数学术期刊要求提交矢量图（PDF/SVG/EPS）
- **高分辨率**：矢量图可以无损缩放到任意分辨率
- **文件大小**：SVG 通常比高分辨率位图小
- **打印质量**：矢量图打印效果远优于位图

---

## 现存问题和可优化点

### 当前问题

#### 1. **图标检测不完整**

**问题描述**：
- 本次执行只检测到 7 个图标，但原图可能包含更多视觉元素
- 某些小图标或特殊形状可能被遗漏
- 置信度阈值设置影响检测召回率

**影响**：
- 部分图标未被提取和矢量化
- 最终 SVG 中这些元素仍为位图

**可能原因**：
- SAM3 提示词不够全面
- 置信度阈值过高（当前 0.0）
- 图标与背景对比度低

#### 2. **SVG 优化被跳过**

**问题描述**：
- 本次执行将 `optimize_iterations` 设为 0，跳过了步骤 4.6
- 未进行 LLM 迭代优化对齐

**影响**：
- SVG 布局和样式可能与原图存在偏差
- 文字位置、箭头样式可能不够精确
- 整体视觉效果可能不如优化后的版本

**建议**：
- 设置 `--optimize_iterations 2` 进行 2 轮优化
- 对比优化前后的效果差异

#### 3. **坐标匹配可能失败**

**问题描述**：
- 如果 LLM 生成的 SVG 占位符位置与 boxlib 坐标偏差较大
- 序号匹配可能失败，导致图标无法正确替换

**影响**：
- 图标被追加到 SVG 末尾，而非替换占位符
- 可能出现重叠或位置错误

**解决方案**：
- 使用 `label` 模式（当前已使用）提高匹配成功率
- 增加坐标近似匹配的容差范围

#### 4. **Box 合并可能过度**

**问题描述**：
- 当前合并阈值 0.001 非常低，几乎不合并
- 如果设置过高（如 0.9），可能将相邻但独立的图标合并

**影响**：
- 过度合并：丢失独立图标
- 合并不足：产生重复的 box

**建议**：
- 根据具体图表调整阈值（推荐 0.7-0.9）
- 可视化检查 `samed.png` 确认合并效果

### 可优化方向

#### 1. **增强图标检测**

**优化方案**：
- **扩展提示词库**：添加更多类型（`diagram, chart, arrow, shape, symbol`）
- **多尺度检测**：对图像进行多尺度处理，检测不同大小的图标
- **后处理过滤**：根据面积、长宽比过滤异常 box
- **人工标注辅助**：支持手动添加遗漏的图标区域

**预期效果**：
- 提高检测召回率 10-20%
- 减少漏检的小图标

#### 2. **改进 SVG 生成质量**

**优化方案**：
- **分步生成**：先生成布局结构，再填充细节
- **模板库**：建立常见学术图表的 SVG 模板库
- **样式提取**：自动提取原图的颜色、字体、线条样式
- **约束生成**：添加更多约束条件（如网格对齐、等间距）

**预期效果**：
- SVG 与原图相似度提升 15-25%
- 减少手动调整的工作量

#### 3. **优化迭代策略**

**优化方案**：
- **差异热图**：生成原图与 SVG 的像素差异热图
- **分区优化**：将图表分为多个区域，分别优化
- **评分机制**：自动评估每次迭代的改进程度
- **早停策略**：当改进幅度小于阈值时提前停止

**预期效果**：
- 减少不必要的迭代次数
- 提高优化效率 30-40%

#### 4. **支持更多输出格式**

**优化方案**：
- **PDF 导出**：将 SVG 转换为高质量 PDF
- **EPS 导出**：支持传统学术出版格式
- **LaTeX 集成**：生成可直接嵌入 LaTeX 的代码
- **交互式 SVG**：添加鼠标悬停、点击等交互效果

**预期效果**：
- 满足不同出版渠道的需求
- 提升用户体验

#### 5. **性能优化**

**优化方案**：
- **模型量化**：使用 INT8 量化加速 SAM3 和 RMBG
- **批处理**：支持批量处理多个图表
- **缓存机制**：缓存中间结果，支持断点续传
- **并行处理**：图标去背景等步骤并行执行

**预期效果**：
- 单图处理时间减少 40-50%
- 支持大规模批量处理

#### 6. **用户交互增强**

**优化方案**：
- **Web UI**：开发可视化界面，支持拖拽调整
- **实时预览**：每个步骤完成后实时显示结果
- **手动修正**：支持手动调整 box 位置和 SVG 元素
- **版本对比**：并排对比不同参数的生成结果

**预期效果**：
- 降低使用门槛
- 提高结果可控性

#### 7. **智能提示词优化**

**优化方案**：
- **自适应提示词**：根据论文领域自动选择提示词
- **提示词学习**：从用户反馈中学习最优提示词组合
- **多语言支持**：支持中文论文方法描述
- **风格迁移**：支持指定参考图片的视觉风格

**预期效果**：
- 生成图表质量提升 20-30%
- 减少参数调优时间

---

## 使用示例

### 基础用法

```bash
python autofigure2_local.py \
  --method_file paper.txt \
  --output_dir ./outputs/demo \
  --provider local \
  --api_key "your-api-key"
```

### 高级用法

```bash
# 使用多个 SAM3 提示词 + 2 轮优化
python autofigure2_local.py \
  --method_file paper.txt \
  --output_dir ./outputs/demo \
  --provider local \
  --api_key "your-api-key" \
  --sam_prompt "icon,robot,animal,person,diagram,arrow" \
  --optimize_iterations 2 \
  --merge_threshold 0.8
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--method_file` | - | 论文方法描述文本文件 |
| `--output_dir` | `./output` | 输出目录 |
| `--provider` | `local` | API 提供商 (local/bianxie/openrouter) |
| `--sam_prompt` | `icon,robot,animal,person` | SAM3 检测提示词 |
| `--min_score` | `0.0` | 最低置信度阈值 |
| `--sam_backend` | `local` | SAM3 后端 (local/fal/roboflow) |
| `--optimize_iterations` | `0` | 优化迭代次数 |
| `--merge_threshold` | `0.001` | Box 合并阈值 |
| `--placeholder_mode` | `label` | 占位符模式 (none/box/label) |
| `--stop_after` | `5` | 执行到指定步骤后停止 |

---

## 技术栈

- **深度学习框架**：PyTorch, Transformers
- **图像处理**：PIL, OpenCV
- **模型**：SAM3, BRIA-RMBG 2.0
- **LLM API**：OpenAI SDK, Gemini, Kimi
- **XML 处理**：lxml
- **SVG 渲染**：cairosvg, svglib

---

## 总结

AutoFigure-Edit 系统通过结合多模态大语言模型、图像分割模型和背景移除模型，实现了从论文文本到可编辑矢量图的自动化流程。相比直接使用 Banana 等工具生成位图，本系统具有以下核心优势：

1. **输出可编辑的 SVG 矢量图**，满足学术出版要求
2. **自动提取和管理图标库**，支持跨图表复用
3. **支持迭代优化和人工介入**，确保高质量输出
4. **完整的中间结果保存**，便于调试和改进

当前系统仍存在图标检测不完整、优化流程可改进等问题，但通过扩展提示词、改进生成策略、增强用户交互等优化方向，可以进一步提升系统的实用性和鲁棒性。

---

## 许可证

本项目仅供学术研究使用。

## 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。


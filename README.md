<div align="center">

<img src="img/logo.png" alt="AutoFigure-edit Logo" width="100%"/>

# AutoFigure: Generating and Refining Publication-Ready Scientific Illustrations [ICLR 2026]

<p align="center">
  <a href="README.md">English</a> | <a href="README_ZH.md">中文</a>
</p>

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue?style=for-the-badge&logo=openreview)](https://openreview.net/forum?id=5N3z9JQJKq)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-FigureBench-orange?style=for-the-badge)](https://huggingface.co/datasets/WestlakeNLP/FigureBench)

<p align="center">
  <strong>From Method Text to Editable SVG</strong><br>
  AutoFigure-edit is the next version of AutoFigure. It turns paper method sections into fully editable SVG figures and lets you refine them in an embedded SVG editor.
</p>

[Quick Start](#-quick-start) • [Web Interface](#-web-interface) • [How It Works](#-how-it-works) • [Configuration](#-configuration) • [Citation](#-citation--license)

[[`Paper`](https://openreview.net/forum?id=5N3z9JQJKq)]
[[`Project`](https://github.com/ResearAI/AutoFigure)]
[[`BibTeX`](#-citation--license)]

</div>

---

## ✨ Features

| Feature | Description |
| :--- | :--- |
| 📝 **Text-to-Figure** | Generate a draft figure directly from method text. |
| 🧠 **SAM3 Icon Detection** | Detect icon regions from multiple prompts and merge overlaps. |
| 🎯 **Labeled Placeholders** | Insert consistent AF-style placeholders for reliable SVG mapping. |
| 🧩 **SVG Generation** | Produce an editable SVG template aligned to the figure. |
| 🖥️ **Embedded Editor** | Edit the SVG in-browser using the bundled svg-edit. |
| 📦 **Artifact Outputs** | Save PNG/SVG outputs and icon crops per run. |
| 📊 **Chart-to-Code** | Convert charts to Python code using SAM3 segmentation (optional) with code evaluation. |
| 📑 **SVG-to-PPT** | Export generated SVG figures directly to PowerPoint presentations. |

---

## 🎨 Gallery: Editable Vectorization & Style Transfer

AutoFigure-edit introduces two breakthrough capabilities:

1.  **Fully Editable SVGs (Pure Code Implementation):** Unlike raster images, our outputs are structured Vector Graphics (SVG). Every component is editable—text, shapes, and layout can be modified losslessly.
2.  **Style Transfer:** The system can mimic the artistic style of reference images provided by the user.

Below are **9 examples** covering 3 different papers. Each paper is generated using 3 different reference styles.
*(Each image shows: **Left** = AutoFigure Generation | **Right** = Vectorized Editable SVG)*

| Paper & Style Transfer Demonstration |
| :---: |
| **[CycleResearcher](https://github.com/zhu-minjun/Researcher) / [Style 1](https://arxiv.org/pdf/2510.09558)**<br><img src="img/case/4.png" width="100%" alt="Paper 1 Style 1"/> |
| **[CycleResearcher](https://github.com/zhu-minjun/Researcher) / [Style 2](https://arxiv.org/pdf/2503.18102)**<br><img src="img/case/5.png" width="100%" alt="Paper 1 Style 2"/> |
| **[CycleResearcher](https://github.com/zhu-minjun/Researcher) / [Style 3](https://arxiv.org/pdf/2510.14512)**<br><img src="img/case/6.png" width="100%" alt="Paper 1 Style 3"/> |
| **[DeepReviewer](https://github.com/zhu-minjun/Researcher) / [Style 1](https://arxiv.org/pdf/2510.09558)**<br><img src="img/case/7.png" width="100%" alt="Paper 2 Style 1"/> |
| **[DeepReviewer](https://github.com/zhu-minjun/Researcher) / [Style 2](https://arxiv.org/pdf/2503.18102)**<br><img src="img/case/8.png" width="100%" alt="Paper 2 Style 2"/> |
| **[DeepReviewer](https://github.com/zhu-minjun/Researcher) / [Style 3](https://arxiv.org/pdf/2510.14512)**<br><img src="img/case/9.png" width="100%" alt="Paper 2 Style 3"/> |
| **[DeepScientist](https://github.com/ResearAI/DeepScientist) / [Style 1](https://arxiv.org/pdf/2510.09558)**<br><img src="img/case/10.png" width="100%" alt="Paper 3 Style 1"/> |
| **[DeepScientist](https://github.com/ResearAI/DeepScientist) / [Style 2](https://arxiv.org/pdf/2503.18102)**<br><img src="img/case/11.png" width="100%" alt="Paper 3 Style 2"/> |
| **[DeepScientist](https://github.com/ResearAI/DeepScientist) / [Style 3](https://arxiv.org/pdf/2510.14512)**<br><img src="img/case/12.png" width="100%" alt="Paper 3 Style 3"/> |

---
## 🚀 How It Works

The AutoFigure-edit pipeline transforms a raw generation into an editable SVG in four distinct stages:

<div align="center">
  <img src="img/pipeline.png" width="100%" alt="Pipeline Visualization: Figure -> SAM -> Template -> Final"/>
  <br>
  <em>(1) Raw Generation &rarr; (2) SAM3 Segmentation &rarr; (3) SVG Layout Template &rarr; (4) Final Assembled Vector</em>
</div>

<br>

1.  **Generation (`figure.png`):** The LLM generates a raster draft based on the method text.
2.  **Segmentation (`sam.png`):** SAM3 detects and segments distinct icons and text regions.
3.  **Templating (`template.svg`):** The system constructs a structural SVG wireframe using placeholders.
4.  **Assembly (`final.svg`):** High-quality cropped icons and vectorized text are injected into the template.

<details>
<summary><strong>View Detailed Technical Pipeline</strong></summary>

<br>
<div align="center">
  <img src="img/edit_method.png" width="100%" alt="AutoFigure-edit Technical Pipeline"/>
</div>

AutoFigure2’s pipeline starts from the paper’s method text and first calls a **text‑to‑image LLM** to render a journal‑style schematic, saved as `figure.png`. The system then runs **SAM3 segmentation** on that image using one or more text prompts (e.g., “icon, diagram, arrow”), merges overlapping detections by an IoU‑like threshold, and draws gray‑filled, black‑outlined labeled boxes on the original; this produces both `samed.png` (the labeled mask overlay) and a structured `boxlib.json` with coordinates, scores, and prompt sources.

Next, each box is cropped from the original figure and passed through **RMBG‑2.0** for background removal, yielding transparent icon assets under `icons/*.png` and `*_nobg.png`. With `figure.png`, `samed.png`, and `boxlib.json` as multimodal inputs, the LLM generates a **placeholder‑style SVG** (`template.svg`) whose boxes match the labeled regions.

Optionally, the SVG is iteratively refined by an **LLM optimizer** to better align strokes, layouts, and styles, resulting in `optimized_template.svg` (or the original template if optimization is skipped). The system then compares the SVG dimensions with the original figure to compute scale factors and aligns coordinate systems. Finally, it replaces each placeholder in the SVG with the corresponding transparent icon (matched by label/ID), producing the assembled `final.svg`.

**Key configuration details:**
- **Placeholder Mode:** Controls how icon boxes are encoded in the prompt (`label`, `box`, or `none`).
- **Optimization:** `optimize_iterations=0` allows skipping the refinement step to use the raw structure directly.
</details>

---

## ⚡ Quick Start

### Option 1: Conda Environment (Recommended)

```bash
# 1) Create and activate conda environment
conda create -n autofigure python=3.10
conda activate autofigure

# 2) Install dependencies
pip install -r requirements.txt

# 3) Install SAM3 separately (not vendored in this repo)

pip install -e sam3
```

### Option 2: Docker Deployment

```bash
# Build Docker image
docker build -f docker/Dockerfile -t autofigure:latest .

# Run container with GPU support
docker run --name autofigure \
  --gpus all \
  --shm-size 32g \
  -p 30001:30000 \
  --ipc=host \
  -v /path/to/models:/root/models \
  -v /path/to/code:/app/ \
  -it autofigure:latest /bin/bash
```
```

### Option 3: CLI

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Install SAM3 separately 

pip install -e sam3
```


**Run:**

```bash
# Basic usage with text-to-image generation
python autofigure_main.py \
  --method_file paper.txt \
  --output_dir outputs/demo \
  --provider bianxie \
  --api_key YOUR_KEY

# Using local image (skip text-to-image generation)
python autofigure_main.py \
  --method_file paper.txt \
  --output_dir outputs/demo \
  --provider local \
  --local_img_path path/to/your/image.png \
  --sam_checkpoint_path /path/to/sam3.pt

# Convert chart to Python code (with SAM3 segmentation)
python autofigure_main.py \
  --method_file paper.txt \
  --output_dir outputs/chart_demo \
  --provider local \
  --local_img_path path/to/chart.png \
  --task_type chart_code \
  --chart_use_sam \
  --sam_checkpoint_path /path/to/sam3.pt \
  --sam_prompt "axis,line,curve,bar,marker,legend,grid" \
  --enable_evaluation \
  --reference_code_path path/to/reference.py

# Generate SVG and convert to PowerPoint
python autofigure_main.py \
  --method_file paper.txt \
  --output_dir outputs/demo \
  --provider local \
  --local_img_path path/to/image.png \
  --sam_checkpoint_path /path/to/sam3.pt \
  --convert_to_ppt \
  --ppt_output_path outputs/demo/result.pptx
```

### Option 4: Web Interface

```bash
python server.py
```

Then open `http://localhost:8000`.

---

## 🖥️ Web Interface Demo

AutoFigure-edit provides a visual web interface designed for seamless generation and editing.

### 1. Configuration Page
<img src="img/demo_start.png" width="100%" alt="Configuration Page" style="border: 1px solid #ddd; border-radius: 8px; margin-bottom: 10px;"/>

On the start page, paste your paper's method text on the left. On the right, configure your generation settings:
*   **Provider:** Select your LLM provider (OpenRouter or Bianxie).
*   **Optimize:** Set SVG template refinement iterations (recommend `0` for standard use).
*   **Reference Image:** Upload a target image to enable style transfer.
*   **SAM3 Backend:** Choose local SAM3 or the fal.ai API (API key optional).

### 2. Canvas & Editor
<img src="img/demo_canvas.png" width="100%" alt="Canvas Page" style="border: 1px solid #ddd; border-radius: 8px; margin-bottom: 10px;"/>

The generation result loads directly into an integrated [SVG-Edit](https://github.com/SVG-Edit/svgedit) canvas, allowing for full vector editing.
*   **Status & Logs:** Check real-time progress (top-left) and view detailed execution logs (top-right button).
*   **Artifacts Drawer:** Click the floating button (bottom-right) to expand the **Artifacts Panel**. This contains all intermediate outputs (icons, SVG templates, etc.). You can **drag and drop** any artifact directly onto the canvas for custom composition.

---

## 🧩 SAM3 Installation Notes

AutoFigure-edit depends on SAM3 but does **not** vendor it. Please follow the
official SAM3 installation guide and prerequisites. The upstream repo currently
targets Python 3.12+, PyTorch 2.7+, and CUDA 12.6 for GPU builds.

SAM3 checkpoints are hosted on Hugging Face and may require you to request
access and authenticate (e.g., `huggingface-cli login`) before download.

- SAM3 repo: https://github.com/facebookresearch/sam3
- SAM3 Hugging Face: https://huggingface.co/facebook/sam3

### SAM3 API Mode (No Local Install)

If you prefer not to install SAM3 locally, you can use an API backend (also supported in the Web demo). **We recommend using [Roboflow](https://roboflow.com/) as it is free to use.**

**Option A: fal.ai**

```bash
export FAL_KEY="your-fal-key"
python autofigure_main.py \
  --method_file paper.txt \
  --output_dir outputs/demo \
  --provider bianxie \
  --api_key YOUR_KEY \
  --sam_backend fal
```

**Option B: Roboflow**

```bash
export ROBOFLOW_API_KEY="your-roboflow-key"
python autofigure_main.py \
  --method_file paper.txt \
  --output_dir outputs/demo \
  --provider bianxie \
  --api_key YOUR_KEY \
  --sam_backend roboflow
```

Optional CLI flags (API):
- `--sam_api_key` (overrides `FAL_KEY`/`ROBOFLOW_API_KEY`)
- `--sam_max_masks` (default: 32, fal.ai only)

## ⚙️ Configuration


### Supported LLM Providers

| Provider | Base URL | Notes |
|----------|----------|------|
| **OpenRouter** | `openrouter.ai/api/v1` | Supports Gemini/Claude/others |
| **Bianxie** | `api.bianxie.ai/v1` | OpenAI-compatible API |
| **Local** | N/A | Use local images without text-to-image generation |

Common CLI flags:

- `--provider` (openrouter | bianxie | local)
- `--image_model`, `--svg_model`
- `--local_img_path` (path to local image when using local provider)
- `--task_type` (icon_svg | chart_code, default: icon_svg)
- `--chart_use_sam` (use SAM3 for chart code generation)
- `--enable_evaluation` (enable code evaluation for chart_code mode)
- `--sam_prompt` (comma-separated prompts)
- `--sam_backend` (local | fal | roboflow | api)
- `--sam_checkpoint_path` (path to SAM3 checkpoint)
- `--sam_api_key` (API key override; falls back to `FAL_KEY` or `ROBOFLOW_API_KEY`)
- `--sam_max_masks` (fal.ai max masks, default 32)
- `--merge_threshold` (0 disables merging)
- `--optimize_iterations` (0 disables optimization)
- `--reference_image_path` (optional, for style transfer)
- `--convert_to_ppt` (convert SVG to PowerPoint)
- `--ppt_output_path` (PPT output path)

---

## 📁 Project Structure

<details>
<summary>Click to expand directory tree</summary>

```
AutoFigure-edit/
├── autofigure_main.py         # Main entry point
├── server.py                  # FastAPI backend for web interface
├── requirements.txt           # Python dependencies
├── autofigure/                # Core package
│   ├── config.py              # Configuration and provider settings
│   ├── pipeline/              # Pipeline modules
│   │   ├── step1_generate.py  # Text-to-image generation
│   │   ├── step2_sam.py       # SAM3 segmentation
│   │   ├── step3_rmbg.py      # Background removal
│   │   ├── step4_svg_template.py  # SVG template generation
│   │   ├── step4_chart_code.py    # Chart-to-code generation
│   │   └── step5_assemble.py  # Final SVG assembly
│   ├── providers/             # LLM provider implementations
│   │   ├── openrouter.py
│   │   ├── bianxie.py
│   │   └── local.py           # Local image mode
│   ├── processors/            # Image processing utilities
│   ├── converters/            # Format converters (SVG to PPT)
│   └── utils/                 # Helper functions
├── docker/                    # Docker deployment files
│   ├── Dockerfile
│   └── README.md
├── examples/                  # Example scripts and inputs
│   ├── testfigure.sh
│   └── testchart_local.sh
├── web/                       # Web interface frontend
│   ├── index.html
│   ├── canvas.html
│   ├── styles.css
│   ├── app.js
│   └── vendor/svg-edit/       # Embedded SVG editor
└── img/                       # README assets
```
</details>

---
└── img/                   # README assets
```
</details>

---

## 🤝 Community & Support

**WeChat Discussion Group**  
Scan the QR code to join our community. If the code is expired, please add WeChat ID `nauhcutnil` or contact `tuchuan@mail.hfut.edu.cn`.

<table>
  <tr>
    <td><img src="img/wechat3.jpg" width="200" alt="WeChat 1"/></td>
    <td><img src="img/wechat4.jpg" width="200" alt="WeChat 2"/></td>
  </tr>
</table>

---

## 📜 Citation & License

If you find **AutoFigure** or **FigureBench** helpful, please cite:

```bibtex
@inproceedings{
zhu2026autofigure,
title={AutoFigure: Generating and Refining Publication-Ready Scientific Illustrations},
author={Minjun Zhu and Zhen Lin and Yixuan Weng and Panzhong Lu and Qiujie Xie and Yifan Wei and Yifan_Wei and Sifan Liu and QiYao Sun and Yue Zhang},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=5N3z9JQJKq}
}

@dataset{figurebench2025,
  title = {FigureBench: A Benchmark for Automated Scientific Illustration Generation},
  author = {WestlakeNLP},
  year = {2025},
  url = {https://huggingface.co/datasets/WestlakeNLP/FigureBench}
}
```

This project is licensed under the MIT License - see `LICENSE` for details.

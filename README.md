<div align="center">

<img src="img/logo.png" alt="AutoFigure-Edit Logo" width="100%"/>

# AutoFigure-Edit: Generating Editable Scientific Illustration

<p align="center">
  <a href="README.md">English</a> | <a href="README_ZH.md">中文</a>
</p>

[![arXiv 2603.06674](https://img.shields.io/badge/arXiv-2603.06674-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.06674)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-FigureBench-orange?style=for-the-badge)](https://huggingface.co/datasets/WestlakeNLP/FigureBench)
[![Website](https://img.shields.io/badge/Website-deepscientist.cc-brightgreen?style=for-the-badge&logo=googlechrome&logoColor=white)](https://deepscientist.cc/)

<p align="center">
  <strong>From Method Text to Editable SVG</strong><br>
  AutoFigure-Edit is the next version of AutoFigure. It turns paper method sections into fully editable SVG figures and lets you refine them in an embedded SVG editor.
</p>

[Quick Start](#-quick-start) • [Web Interface](#-web-interface) • [How It Works](#-how-it-works) • [Configuration](#-configuration) • [Citation](#-citation--license)

[[`Paper`](https://arxiv.org/abs/2603.06674)]
[[`AutoFigure`](https://github.com/ResearAI/AutoFigure)]
[[`BibTeX`](#-citation--license)]

</div>

---




https://github.com/user-attachments/assets/6f93deb4-9854-4f1e-8097-53b0c3378a0d





## 🔥 News

- **[2026.04.23]** 🚀 **AutoFigure-Edit v1.1** is now available. This release primarily adds user-supplied stage-1 figure import, official OpenAI model support including `gpt-image-2` and `gpt-5.5`, Bianxie AI / `custom` OpenAI-compatible routing, and a bilingual configuration workflow. See the full [release notes](releases/v1.1.md).
- **[2026.03.24]** 🧠 Our sister project **DeepScientist v1.5** is now officially released. It is a local-first open-source autonomous research system for end-to-end scientific discovery. Explore it on [GitHub](https://github.com/ResearAI/DeepScientist) or read the [ICLR 2026 paper](https://openreview.net/forum?id=cZFgsLq8Gs).
- **[2026.03.11]** 📄 Our **AutoFigure-Edit** paper is now available on [arXiv](https://arxiv.org/abs/2603.06674) and featured in 🤗[Hugging Face Daily Papers](https://huggingface.co/papers/2603.06674)! If you find our work helpful, please consider giving us an **upvote** on Hugging Face and **citing** our paper. Thank you! ❤️
- **[2026.02.17]** 🚀 The **AutoFigure-Edit online platform** is now live! It is free for all scholars to use. Try it out at [deepscientist.cc](https://deepscientist.cc).
- **[2026.01.26]** 🎉 AutoFigure has been accepted to **ICLR 2026**! You can read the paper on [arXiv](https://arxiv.org/abs/2602.03828).

---

## 🆕 V1.1 (2026.04.23)

AutoFigure-Edit v1.1 is published as tag `v1.1`. This release focuses on two practical workflows that were still awkward in earlier public builds: starting from a user-supplied stage-1 academic figure, and running the pipeline cleanly with official OpenAI models or OpenAI-compatible gateways.

- **User-supplied stage-1 figure import:** You can now upload an existing academic raster figure, skip step 1 image generation, and continue directly from SAM + SVG reconstruction in both the web UI and CLI workflow.
- **Official OpenAI model support:** Step 1 can now use the OpenAI Images API with `gpt-image-2`, while the OpenAI Responses path is documented and exposed for text plus multimodal SVG reconstruction with `gpt-5.5` as the default SVG model.
- **Bianxie AI and `custom` OpenAI-compatible routing:** The CLI and web UI expose `bianxie` as a built-in compatible route with `https://api.bianxie.ai/v1`, while `custom` remains available for user-supplied OpenAI-compatible `/v1` base URLs. The `openai_response` route can inherit the same compatible `base_url` and `api_key` by default.
- **Bilingual setup and onboarding:** The main page, import page, canvas, and guide now support in-page Chinese / English switching, and the built-in guide explains workflow choices, fields, SAM backends, and recommended presets.

Full release notes: [releases/v1.1.md](releases/v1.1.md)

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

### Option 0: Docker Deployment Guide (Recommended)

Use Docker for a reproducible one-command setup without local Python/SAM3 installation.

#### 0) Prerequisites

- Docker Desktop (with Docker Compose v2)
- Port `8000` available on host
- HuggingFace access to `briaai/RMBG-2.0`: https://huggingface.co/briaai/RMBG-2.0

#### 1) Prepare `.env`

```bash
# Linux/macOS
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

At minimum, set this in `.env`:

```bash
HF_TOKEN=hf_xxx
```

Optional but recommended:

```bash
# SAM3 API backend (Docker default in UI is Roboflow)
ROBOFLOW_API_KEY=your_roboflow_key

# Step-4 multimodal retry tuning (OpenRouter)
OPENROUTER_MULTIMODAL_RETRIES=3
OPENROUTER_MULTIMODAL_RETRY_DELAY=1.5

# DNS override for Roboflow name-resolution issues
DOCKER_DNS_1=223.5.5.5
DOCKER_DNS_2=119.29.29.29
```

For restricted networks, you can also set build mirrors:

```bash
BASE_IMAGE=docker.m.daocloud.io/library/python:3.11-slim
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
PIP_EXTRA_INDEX_URL=
```

#### 2) Build and start

```bash
docker compose up -d --build
```

Open `http://localhost:8000`.

#### 3) Verify service health

```bash
docker compose ps
curl http://localhost:8000/healthz
```

Expected health response: `{"status":"ok"}`.

#### 4) Daily operations

```bash
# Stream logs
docker compose logs -f autofigure-edit

# Restart service
docker compose restart autofigure-edit

# Rebuild from scratch (no cache)
docker compose build --no-cache
docker compose up -d

# Stop and remove container
docker compose down
```

#### 5) Persistence and defaults

- Persistent outputs: `./outputs`, `./uploads`
- Persistent HuggingFace cache: Docker volume `hf_cache` (`/app/.cache/huggingface`)
- Docker/Web default SAM backend: `roboflow`
- Default SAM prompt: `icon,person,robot,animal`
- Current default models:
  - `openrouter`: image `google/gemini-3.1-flash-image-preview`, svg `google/gemini-3.1-pro-preview`
  - `bianxie`: image `gpt-image-2`, svg `gemini-3.1-pro-preview` (built-in base URL `https://api.bianxie.ai/v1`)
  - `custom`: image `gemini-3.1-flash-image-preview`, svg `gemini-3.1-pro-preview` (requires your own OpenAI-compatible `/v1` base URL)
  - `gemini`: image `gemini-3.1-flash-image-preview`, svg `gemini-3.1-pro-preview`
  - `openai_response`: image `gpt-image-2` (step 1 fallback), svg `gpt-5.5` via Responses API
- Optional step-1 override:
  - `--image_provider openai`: image `gpt-image-2` via the official OpenAI Images API

#### 6) Common Docker networking issues

- `Temporary failure in name resolution` (Roboflow): set `DOCKER_DNS_1/2` in `.env`, then `docker compose up -d --build`.
- Cannot reach Docker Hub auth (`auth.docker.io`): set `BASE_IMAGE` and `PIP_INDEX_URL` mirrors in `.env`.
- Optional Roboflow endpoint override:
  - `ROBOFLOW_API_URL=<your_reachable_roboflow_endpoint>`
  - `ROBOFLOW_API_FALLBACK_URLS=<comma_separated_backup_endpoints>`

### Option 1: CLI

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Install SAM3 separately (not vendored in this repo)
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

**Run:**

```bash
python autofigure2.py \
  --method_file paper.txt \
  --output_dir outputs/demo \
  --provider bianxie \
  --api_key YOUR_KEY
```

Use OpenAI only for step 1 image generation while keeping SVG reconstruction on the original provider:

```bash
python autofigure2.py \
  --method_file paper.txt \
  --output_dir outputs/demo \
  --provider gemini \
  --api_key GEMINI_KEY \
  --image_provider openai \
  --image_api_key OPENAI_KEY \
  --image_model gpt-image-2
```

Use the OpenAI Responses API for text + multimodal SVG reconstruction:

```bash
python autofigure2.py \
  --method_file paper.txt \
  --output_dir outputs/demo \
  --provider openai_response \
  --api_key OPENAI_KEY
```

Continue from an existing stage-1 figure and skip image generation:

```bash
python autofigure2.py \
  --input_figure_path ./my_stage1_figure.png \
  --output_dir outputs/import_demo \
  --provider openai_response \
  --api_key OPENAI_KEY \
  --svg_model gpt-5.5
```

### Option 2: Web Interface

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
*   **Provider:** Select your LLM provider (Bianxie AI, OpenRouter, Custom, Gemini, or OpenAI Responses).
*   **Image Provider:** Optionally override **step 1 only** to use OpenAI GPT-Image.
*   **Optimize:** Set SVG template refinement iterations (recommend `0` for standard use).
*   **Image Size:** Available when the effective step-1 image provider is **Gemini**. Choose `1K`, `2K`, or `4K`.
*   **Auto Upscale:** Enabled by default. Upscales `figure.png` to a 4K long edge (`3840px`) while preserving aspect ratio.
*   **Reference Image:** Upload a target image to enable style transfer.
*   **SAM3 Backend:** Choose local SAM3 or the fal.ai API (API key optional).

If you already have the first-stage raster figure, use the black button in the top-right corner:

*   **I already have the stage-1 figure:** Opens a dedicated import page where you upload an existing academic figure and continue directly from SAM + SVG reconstruction.

### 2. Canvas & Editor
<img src="img/demo_canvas.png" width="100%" alt="Canvas Page" style="border: 1px solid #ddd; border-radius: 8px; margin-bottom: 10px;"/>

The generation result loads directly into an integrated [SVG-Edit](https://github.com/SVG-Edit/svgedit) canvas, allowing for full vector editing.
*   **Status & Logs:** Check real-time progress (top-left) and view detailed execution logs (top-right button).
*   **Artifacts Drawer:** Click the floating button (bottom-right) to expand the **Artifacts Panel**. This contains all intermediate outputs (icons, SVG templates, etc.). You can **drag and drop** any artifact directly onto the canvas for custom composition.
*   **History:** Open the **History** button to browse saved runs from `outputs/` and reopen an older result in the canvas.

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
python autofigure2.py \
  --method_file paper.txt \
  --output_dir outputs/demo \
  --provider bianxie \
  --api_key YOUR_KEY \
  --sam_backend fal
```

**Option B: Roboflow**

```bash
export ROBOFLOW_API_KEY="your-roboflow-key"
python autofigure2.py \
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
| **Bianxie AI** | `api.bianxie.ai/v1` | Built-in OpenAI-compatible aggregate API; supports GPT-image-2 and Gemini-3.1-Pro access for mainland China users without a foreign credit card |
| **Custom** | `<your-compatible-endpoint>/v1` (required) | Vendor-neutral OpenAI-compatible API |
| **Gemini (Google)** | `generativelanguage.googleapis.com/v1beta` | Official Google Gemini API (`google-genai`) |
| **OpenAI Responses** | `api.openai.com/v1` | Uses the official OpenAI Responses API for text + multimodal |

Common CLI flags:

- `--method_text`, `--method_file`, or `--input_figure_path`
- `--provider` (openrouter | bianxie | custom | gemini | openai_response)
- `--image_provider` (openrouter | bianxie | custom | gemini | openai, optional step-1 override)
- `--image_api_key`, `--image_base_url`
- `--image_model`, `--svg_model`
- `--image_size` (1K | 2K | 4K, Gemini only)
- `--disable_auto_upscale` (disable the default 4K aspect-ratio-preserving upscale after step 1)
- `--sam_prompt` (comma-separated prompts)
- `--sam_backend` (local | fal | roboflow | api)
- `--sam_api_key` (API key override; falls back to `FAL_KEY` or `ROBOFLOW_API_KEY`)
- `--sam_max_masks` (fal.ai max masks, default 32)
- `--merge_threshold` (0 disables merging)
- `--optimize_iterations` (0 disables optimization)
- `--reference_image_path` (optional)

### Custom Provider / Custom Base URL

If you want to use a self-hosted or third-party OpenAI-compatible endpoint, use:

- `--provider custom`
- `--base_url <your_openai_compatible_v1_root>`
- `--image_model <image_model_id>`
- `--svg_model <svg_model_id>`

You can also set `AUTOFIGURE_CUSTOM_BASE_URL` instead of passing `--base_url` every time.

`base_url` must be the OpenAI-compatible `/v1` root:

```text
https://your-provider.example/v1
```

Do not pass a concrete endpoint path such as:

```text
https://your-provider.example/v1/chat/completions
```

For text reasoning and SVG reconstruction, the Custom route calls:

```http
POST /chat/completions
Authorization: Bearer <api_key>
```

Text-only requests use the normal Chat Completions message shape:

```json
{
  "model": "your-text-or-svg-model",
  "messages": [{ "role": "user", "content": "..." }],
  "max_tokens": 16000,
  "temperature": 0.7
}
```

Multimodal SVG reconstruction must support OpenAI-style `image_url` data URIs:

```json
{
  "role": "user",
  "content": [
    { "type": "text", "text": "..." },
    {
      "type": "image_url",
      "image_url": { "url": "data:image/png;base64,..." }
    }
  ]
}
```

The response must return content in the standard shape:

```json
{
  "choices": [
    { "message": { "content": "<svg ...>...</svg>" } }
  ]
}
```

The SVG may be returned as raw `<svg>...</svg>` or inside a markdown code block.

The built-in `bianxie` route uses the same OpenAI-compatible chat shape for text/SVG reconstruction, and uses an OpenAI Images-compatible path for `gpt-image-2` step-1 image generation.

For step-1 image generation with `--image_provider custom` (or when `--provider custom` is linked to step 1), this repo currently calls `/chat/completions` and expects the returned message content to contain a base64 image data URI:

```text
![image](data:image/png;base64,...)
```

or:

```text
data:image/png;base64,...
```

If your provider only exposes an OpenAI Images-compatible `/images/generations` route, use `--image_provider openai` for the official OpenAI Images API, or keep image generation on another supported route.

### OpenAI GPT-Image for Step 1

As of April 23, 2026, OpenAI's official Images API supports `images.generate` and `images.edit` for GPT-Image models. In this repo, `--image_provider openai` uses the OpenAI Images API for step 1 only:

- no reference image: `images.generate`
- with reference image: `images.edit`
- default model: `gpt-image-2` (override with `--image_model`)
- API key precedence: `--image_api_key` -> `OPENAI_API_KEY` -> `--api_key`

### Default 4K Upscale

After step 1, the generated `figure.png` is upscaled by default so its long edge reaches `3840px` while preserving the original aspect ratio. If the generated image is already at or above a 4K long edge, the upscale step is skipped automatically.

Disable it with:

```bash
--disable_auto_upscale
```

### OpenAI Responses Provider

As of April 23, 2026, OpenAI's official Responses API supports text output plus multimodal input with `input_text` and `input_image`. In this repo, `--provider openai_response` means:

- text calls use `client.responses.create(...)`
- multimodal SVG reconstruction also uses `client.responses.create(...)`
- step 1 image generation falls back to the official OpenAI Images API unless `--image_provider` is explicitly set
- default SVG model: `gpt-5.5` (override with `--svg_model`)

### Importing an Existing Stage-1 Figure

If you already have the academic raster figure from step 1, use `--input_figure_path` to skip image generation entirely. The pipeline will normalize the imported image into `figure.png`, optionally apply the default 4K aspect-ratio-preserving upscale, and then continue from SAM segmentation and SVG reconstruction.

---

## 📁 Project Structure

<details>
<summary>Click to expand directory tree</summary>

```
AutoFigure-edit/
├── autofigure2.py         # Main pipeline
├── server.py              # FastAPI backend
├── requirements.txt
├── web/                   # Static frontend
│   ├── index.html
│   ├── canvas.html
│   ├── history.html
│   ├── styles.css
│   ├── app.js
│   └── vendor/svg-edit/   # Embedded SVG editor
└── img/                   # README assets
```
</details>

---

## 🤝 Community & Support

**WeChat Discussion Group**  
Scan the QR code to join our community. If the QR code has expired, please add WeChat ID `nauhcutnil` or contact `tuchuan@mail.hfut.edu.cn`.

**When sending a friend request, please identify yourself in the format:**  
`University/Company - Degree/Position - Name`  
(e.g., `Westlake University - PhD Student - Zhang San`).

<table>
  <tr>
    <td><img src="img/wechat11.jpg" width="200" alt="WeChat 1"/></td>
    <td><img src="img/lzwechat.jpg" width="200" alt="WeChat 2"/></td>
  </tr>
</table>

---

## 📜 Citation & License

If you find **AutoFigure**, **AutoFigure-Edit**, or **FigureBench** helpful, please cite:

```bibtex
@inproceedings{
zhu2026autofigure,
title={AutoFigure: Generating and Refining Publication-Ready Scientific Illustrations},
author={Minjun Zhu and Zhen Lin and Yixuan Weng and Panzhong Lu and Qiujie Xie and Yifan Wei and Sifan Liu and Qiyao Sun and Yue Zhang},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=5N3z9JQJKq}
}

@inproceedings{lin-etal-2026-autofigure,
    title = "{A}uto{F}igure-Edit: Generating Editable Scientific Illustrations via Reference-Guided Styling",
    author = "Lin, Zhen  and
      Xie, Qiujie  and
      Zhu, Minjun  and
      Li, Shichen  and
      Sun, QiYao  and
      Gu, Enhao  and
      Ding, Yiran  and
      Sun, Ke  and
      Guo, Fang  and
      Lu, Panzhong  and
      Ning, Zhiyuan  and
      Weng, Yixuan  and
      Zhang, Yue",
    editor = "Durrett, Greg  and
      Jian, Ping",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 3: System Demonstrations)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-demo.6/",
    doi = "10.18653/v1/2026.acl-demo.6",
    pages = "57--67",
    ISBN = "979-8-89176-392-0",
    abstract = "High-quality scientific illustrations are essential for communicating complex scientific and technical concepts, yet existing automated systems remain limited in editability, stylistic controllability, and efficiency. We present AutoFigure-Edit, an end-to-end system that generates fully editable scientific illustrations from long-form scientific text while enabling flexible style adaptation through user-provided reference images. By combining long-context understanding, reference-guided styling, and native SVG editing, it enables efficient creation and refinement of high-quality scientific illustrations. To facilitate further progress in this field, we release the video at https://youtu.be/10IH8SyJjAQ, the full codebase at https://github.com/ResearAI/AutoFigure-Edit and provide a live demo for easy access and interactive use at https://autofigure.cc/."
}

@dataset{figurebench2025,
  title = {FigureBench: A Benchmark for Automated Scientific Illustration Generation},
  author = {WestlakeNLP},
  year = {2025},
  url = {https://huggingface.co/datasets/WestlakeNLP/FigureBench}
}
```

Repository metadata and usage guidance:

- [CITATION.cff](./CITATION.cff)
- [Citation and attribution guidance](./CITATION_AND_ATTRIBUTION.md)
- [Name and logo usage](./TRADEMARK.md)

## 🙏 Acknowledgments

We would like to thank the [Linux.do](https://linux.do/) community for their support.

We also thank the following sponsor for supporting this project:

| Sponsor | Link | Support | Note |
|---|---|---|---|
| Bianxie AI Aggregate API | [https://api.bianxie.ai](https://bianxieai.com/autofigure) | Provides mainland China-friendly access to GPT-image-2 and Gemini-3.1-Pro for AutoFigure-Edit users | No foreign credit card required |

This project is licensed under the MIT License - see `LICENSE` for details.
Name and logo usage are covered separately in `TRADEMARK.md`.

---

## More From ResearAI

Explore more open-source research tools from ResearAI:

| Project | What it does |
|---|---|
| [DeepScientist](https://github.com/ResearAI/DeepScientist) | autonomous scientific discovery system |
| [AutoFigure](https://github.com/ResearAI/AutoFigure) | generate paper-ready figures |
| [DeepReviewer-v2](https://github.com/ResearAI/DeepReviewer-v2) | review papers and drafts |
| [Awesome-AI-Scientist](https://github.com/ResearAI/Awesome-AI-Scientist) | curated AI scientist landscape |



---

## Q&A

The optimal configuration for this project uses `gemini-3.1-flash-image-preview` from Google AI Studio [[https://aistudio.google.com/](https://aistudio.google.com/)] as the image generation model and `gemini-3.1-pro-preview` as the SVG conversion model. Each run costs approximately $0.50, consumes about 30,000 tokens, and takes around 20 minutes. **It is strongly recommended to use the 4K option for optimal performance**, as using 1K or 2K resolutions will result in the final generated SVG being unusually blurry.

[Mainland China Notice] Gemini's Terms of Service do not permit access or usage by users in mainland China. If OpenRouter throws an error, it is often because an account registered in mainland China lacks the necessary permissions to use Gemini. **It is recommended to use an OpenRouter account registered in the United States or Europe and to ensure compliant usage.**

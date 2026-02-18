"""SAM3 图像分割处理器"""

import base64
import io
import json
import os
from pathlib import Path
from typing import Optional, Literal, List, Dict, Any, Tuple

import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFont

from ..config import SAM3_FAL_API_URL, SAM3_ROBOFLOW_API_URL, SAM3_API_TIMEOUT
from ..utils.box_utils import merge_overlapping_boxes


def get_label_font(box_width: int, box_height: int) -> ImageFont.FreeTypeFont:
    """根据 box 尺寸动态计算合适的字体大小"""
    min_dim = min(box_width, box_height)
    font_size = max(12, min(48, min_dim // 4))

    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]

    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, font_size)
        except (IOError, OSError):
            continue

    try:
        return ImageFont.load_default()
    except:
        return None


def _get_fal_api_key(sam_api_key: Optional[str]) -> str:
    key = sam_api_key or os.environ.get("FAL_KEY")
    if not key:
        raise ValueError("SAM3 fal.ai API key missing")
    return key


def _get_roboflow_api_key(sam_api_key: Optional[str]) -> str:
    key = sam_api_key or os.environ.get("ROBOFLOW_API_KEY") or os.environ.get("API_KEY")
    if not key:
        raise ValueError("SAM3 Roboflow API key missing")
    return key


def _image_to_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_b64}"


def _image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _cxcywh_norm_to_xyxy(box: list, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if not box or len(box) < 4:
        return None
    try:
        cx, cy, bw, bh = [float(v) for v in box[:4]]
    except (TypeError, ValueError):
        return None

    cx *= width
    cy *= height
    bw *= width
    bh *= height

    x1 = int(round(cx - bw / 2.0))
    y1 = int(round(cy - bh / 2.0))
    x2 = int(round(cx + bw / 2.0))
    y2 = int(round(cy + bh / 2.0))

    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _polygon_to_bbox(points: list, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    xs: List[float] = []
    ys: List[float] = []

    for pt in points:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        try:
            x = float(pt[0])
            y = float(pt[1])
        except (TypeError, ValueError):
            continue
        xs.append(x)
        ys.append(y)

    if not xs or not ys:
        return None

    x1 = int(round(min(xs)))
    y1 = int(round(min(ys)))
    x2 = int(round(max(xs)))
    y2 = int(round(max(ys)))

    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _extract_sam3_api_detections(response_json: dict, image_size: Tuple[int, int]) -> List[dict]:
    width, height = image_size
    detections: List[dict] = []

    metadata = response_json.get("metadata") if isinstance(response_json, dict) else None
    if isinstance(metadata, list) and metadata:
        for item in metadata:
            if not isinstance(item, dict):
                continue
            box = item.get("box")
            xyxy = _cxcywh_norm_to_xyxy(box, width, height)
            if not xyxy:
                continue
            score = item.get("score")
            detections.append(
                {"x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3], "score": score}
            )
        return detections

    boxes = response_json.get("boxes") if isinstance(response_json, dict) else None
    scores = response_json.get("scores") if isinstance(response_json, dict) else None
    if isinstance(boxes, list) and boxes:
        scores_list = scores if isinstance(scores, list) else []
        for idx, box in enumerate(boxes):
            xyxy = _cxcywh_norm_to_xyxy(box, width, height)
            if not xyxy:
                continue
            score = scores_list[idx] if idx < len(scores_list) else None
            detections.append(
                {"x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3], "score": score}
            )

    return detections


def _extract_roboflow_detections(response_json: dict, image_size: Tuple[int, int]) -> List[dict]:
    width, height = image_size
    detections: List[dict] = []

    prompt_results = response_json.get("prompt_results") if isinstance(response_json, dict) else None
    if not isinstance(prompt_results, list):
        return detections

    for prompt_result in prompt_results:
        if not isinstance(prompt_result, dict):
            continue
        predictions = prompt_result.get("predictions", [])
        if not isinstance(predictions, list):
            continue
        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue
            confidence = prediction.get("confidence")
            masks = prediction.get("masks", [])
            if not isinstance(masks, list):
                continue
            for mask in masks:
                points = []
                if isinstance(mask, list) and mask:
                    if isinstance(mask[0], (list, tuple)) and len(mask[0]) >= 2 and isinstance(
                        mask[0][0], (int, float)
                    ):
                        points = mask
                    elif isinstance(mask[0], (list, tuple)):
                        for sub in mask:
                            if isinstance(sub, (list, tuple)) and len(sub) >= 2 and isinstance(
                                sub[0], (int, float)
                            ):
                                points.append(sub)
                            elif isinstance(sub, (list, tuple)) and sub and isinstance(
                                sub[0], (list, tuple)
                            ):
                                for pt in sub:
                                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                                        points.append(pt)
                if not points:
                    continue
                xyxy = _polygon_to_bbox(points, width, height)
                if not xyxy:
                    continue
                detections.append(
                    {
                        "x1": xyxy[0],
                        "y1": xyxy[1],
                        "x2": xyxy[2],
                        "y2": xyxy[3],
                        "score": confidence,
                    }
                )

    return detections


def _call_sam3_api(image_data_uri: str, prompt: str, api_key: str, max_masks: int) -> dict:
    headers = {
        "Authorization": f"Key {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "image_url": image_data_uri,
        "prompt": prompt,
        "apply_mask": False,
        "return_multiple_masks": True,
        "max_masks": max_masks,
        "include_scores": True,
        "include_boxes": True,
    }
    response = requests.post(SAM3_FAL_API_URL, headers=headers, json=payload, timeout=SAM3_API_TIMEOUT)
    if response.status_code != 200:
        raise Exception(f"SAM3 API 错误: {response.status_code} - {response.text[:500]}")
    result = response.json()
    if isinstance(result, dict) and "error" in result:
        raise Exception(f"SAM3 API 错误: {result.get('error')}")
    return result


def _call_sam3_roboflow_api(image_base64: str, prompt: str, api_key: str, min_score: float) -> dict:
    payload = {
        "image": {"type": "base64", "value": image_base64},
        "prompts": [{"type": "text", "text": prompt}],
        "format": "polygon",
        "output_prob_thresh": min_score,
    }
    url = f"{SAM3_ROBOFLOW_API_URL}?api_key={api_key}"
    response = requests.post(url, json=payload, timeout=SAM3_API_TIMEOUT)
    if response.status_code != 200:
        raise Exception(f"SAM3 Roboflow API 错误: {response.status_code} - {response.text[:500]}")
    result = response.json()
    if isinstance(result, dict) and "error" in result:
        raise Exception(f"SAM3 Roboflow API 错误: {result.get('error')}")
    return result


def segment_with_sam3(
    image_path: str,
    output_dir: str,
    text_prompts: str = "icon",
    min_score: float = 0.5,
    merge_threshold: float = 0.9,
    sam_backend: Literal["local", "fal", "roboflow", "api"] = "local",
    sam_api_key: Optional[str] = None,
    sam_checkpoint_path: Optional[str] = None,
    sam_bpe_path: Optional[str] = None,
    sam_max_masks: int = 32,
) -> Tuple[str, str, list]:
    """使用 SAM3 分割图片"""
    print("\n" + "=" * 60)
    print("步骤二：SAM3 分割 + 灰色填充+黑色边框+序号标记")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path)
    original_size = image.size
    print(f"原图尺寸: {original_size[0]} x {original_size[1]}")

    prompt_list = [p.strip() for p in text_prompts.split(",") if p.strip()]
    print(f"使用的 prompts: {prompt_list}")

    all_detected_boxes = []
    total_detected = 0

    backend = sam_backend
    if backend == "api":
        backend = "fal"

    if backend == "local":
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        import sam3
        
        checkpoint_path = sam_checkpoint_path or '/root/models/sam3/sam3.pt'
        
        if sam_bpe_path:
            bpe_path = Path(sam_bpe_path)
            if not bpe_path.exists():
                print(f"警告: 指定的 BPE 路径不存在: {sam_bpe_path}，将尝试使用默认路径")
                bpe_path = None
        else:
            bpe_path = None
        
        if bpe_path is None:
            sam3_dir = Path(sam3.__path__[0]) if hasattr(sam3, '__path__') else Path(sam3.__file__).parent
            bpe_path = sam3_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"
            if not bpe_path.exists():
                bpe_path = None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        model = build_sam3_image_model(device=device, bpe_path=str(bpe_path) if bpe_path else None, checkpoint_path=checkpoint_path)
        processor = Sam3Processor(model, device=device)
        inference_state = processor.set_image(image)

        for prompt in prompt_list:
            print(f"\n  正在检测: '{prompt}'")
            output = processor.set_text_prompt(state=inference_state, prompt=prompt)

            boxes = output["boxes"]
            scores = output["scores"]

            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()

            prompt_count = 0
            for box, score in zip(boxes, scores):
                if score >= min_score:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    all_detected_boxes.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "score": float(score),
                        "prompt": prompt
                    })
                    prompt_count += 1

            print(f"  '{prompt}' 检测到 {prompt_count} 个有效对象")
            total_detected += prompt_count

        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elif backend == "fal":
        api_key = _get_fal_api_key(sam_api_key)
        max_masks = max(1, min(32, int(sam_max_masks)))
        image_data_uri = _image_to_data_uri(image)

        for prompt in prompt_list:
            print(f"\n  正在检测: '{prompt}'")
            response_json = _call_sam3_api(image_data_uri, prompt, api_key, max_masks)
            detections = _extract_sam3_api_detections(response_json, original_size)
            prompt_count = 0
            for det in detections:
                score = det.get("score")
                score_val = float(score) if score is not None else 0.0
                if score_val >= min_score:
                    all_detected_boxes.append({**det, "score": score_val, "prompt": prompt})
                    prompt_count += 1
            print(f"  '{prompt}' 检测到 {prompt_count} 个有效对象")
            total_detected += prompt_count

    elif backend == "roboflow":
        api_key = _get_roboflow_api_key(sam_api_key)
        image_base64 = _image_to_base64(image)

        for prompt in prompt_list:
            print(f"\n  正在检测: '{prompt}'")
            response_json = _call_sam3_roboflow_api(image_base64, prompt, api_key, min_score)
            detections = _extract_roboflow_detections(response_json, original_size)
            prompt_count = 0
            for det in detections:
                score_val = float(det.get("score", 0.0))
                if score_val >= min_score:
                    all_detected_boxes.append({**det, "score": score_val, "prompt": prompt})
                    prompt_count += 1
            print(f"  '{prompt}' 检测到 {prompt_count} 个有效对象")
            total_detected += prompt_count

    print(f"\n总计检测: {total_detected} 个对象")

    valid_boxes = []
    for i, box_data in enumerate(all_detected_boxes):
        valid_boxes.append({
            "id": i,
            "label": f"<AF>{i + 1:02d}",
            "x1": box_data["x1"],
            "y1": box_data["y1"],
            "x2": box_data["x2"],
            "y2": box_data["y2"],
            "score": box_data["score"],
            "prompt": box_data["prompt"]
        })

    if merge_threshold > 0 and len(valid_boxes) > 1:
        print(f"\n  合并重叠的boxes (阈值: {merge_threshold})...")
        original_count = len(valid_boxes)
        valid_boxes = merge_overlapping_boxes(valid_boxes, merge_threshold)
        merged_count = original_count - len(valid_boxes)
        if merged_count > 0:
            print(f"  合并完成: {original_count} -> {len(valid_boxes)}")

    print(f"\n  绘制 samed.png (使用 {len(valid_boxes)} 个boxes)...")
    samed_image = image.copy()
    draw = ImageDraw.Draw(samed_image)

    for box_info in valid_boxes:
        x1, y1, x2, y2 = box_info["x1"], box_info["y1"], box_info["x2"], box_info["y2"]
        label = box_info["label"]

        draw.rectangle([x1, y1, x2, y2], fill="#808080", outline="black", width=3)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        box_width = x2 - x1
        box_height = y2 - y1
        font = get_label_font(box_width, box_height)

        if font:
            try:
                draw.text((cx, cy), label, fill="white", anchor="mm", font=font)
            except TypeError:
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                text_x = cx - text_width // 2
                text_y = cy - text_height // 2
                draw.text((text_x, text_y), label, fill="white", font=font)
        else:
            draw.text((cx, cy), label, fill="white")

    samed_path = output_dir / "samed.png"
    samed_image.save(str(samed_path))
    print(f"标记图片已保存: {samed_path}")

    boxlib_data = {
        "image_size": {"width": original_size[0], "height": original_size[1]},
        "prompts_used": prompt_list,
        "boxes": valid_boxes
    }

    boxlib_path = output_dir / "boxlib.json"
    with open(boxlib_path, 'w', encoding='utf-8') as f:
        json.dump(boxlib_data, f, indent=2, ensure_ascii=False)
    print(f"Box 信息已保存: {boxlib_path}")

    return str(samed_path), str(boxlib_path), valid_boxes


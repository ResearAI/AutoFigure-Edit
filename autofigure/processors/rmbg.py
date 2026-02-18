"""RMBG 背景去除处理器"""

import json
from pathlib import Path
from typing import Optional, List

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


class BriaRMBG2Remover:
    """使用 BRIA-RMBG 2.0 模型进行高质量背景抠图"""

    def __init__(self, model_path: Optional[Path] = None, output_dir: Optional[Path] = None):
        self.model_path = Path(model_path) if model_path else None
        self.output_dir = Path(output_dir) if output_dir else Path("./output/icons")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if self.model_path and self.model_path.exists():
            print(f"加载本地 RMBG 权重: {self.model_path}")
            self.model = AutoModelForImageSegmentation.from_pretrained(
                str(self.model_path), trust_remote_code=True,
            ).eval().to(device)
        else:
            print("从 HuggingFace 加载 RMBG-2.0 模型...")
            self.model = AutoModelForImageSegmentation.from_pretrained(
                "briaai/RMBG-2.0", trust_remote_code=True,
            ).eval().to(device)

        self.image_size = (1024, 1024)
        self.transform_image = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def remove_background(self, image: Image.Image, output_name: str) -> str:
        image_rgb = image.convert("RGB")
        input_tensor = self.transform_image(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid().cpu()

        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_rgb.size)

        out = image_rgb.copy()
        out.putalpha(mask)

        out_path = self.output_dir / f"{output_name}_nobg.png"
        out.save(out_path)
        return str(out_path)


def crop_and_remove_background(
    image_path: str,
    boxlib_path: str,
    output_dir: str,
    rmbg_model_path: Optional[str] = None,
) -> List[dict]:
    """根据 boxlib.json 裁切图片并使用 RMBG2 去背景"""
    print("\n" + "=" * 60)
    print("步骤三：裁切 + RMBG2 去背景")
    print("=" * 60)

    output_dir = Path(output_dir)
    icons_dir = output_dir / "icons"
    icons_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path)
    with open(boxlib_path, 'r', encoding='utf-8') as f:
        boxlib_data = json.load(f)

    boxes = boxlib_data["boxes"]

    if len(boxes) == 0:
        print("警告: 没有检测到有效的 box")
        return []

    remover = BriaRMBG2Remover(model_path=rmbg_model_path, output_dir=icons_dir)

    icon_infos = []
    for box_info in boxes:
        box_id = box_info["id"]
        label = box_info.get("label", f"<AF>{box_id + 1:02d}")
        label_clean = label.replace("<", "").replace(">", "")

        x1, y1, x2, y2 = box_info["x1"], box_info["y1"], box_info["x2"], box_info["y2"]

        cropped = image.crop((x1, y1, x2, y2))
        crop_path = icons_dir / f"icon_{label_clean}.png"
        cropped.save(crop_path)

        nobg_path = remover.remove_background(cropped, f"icon_{label_clean}")

        icon_infos.append({
            "id": box_id,
            "label": label,
            "label_clean": label_clean,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "width": x2 - x1, "height": y2 - y1,
            "crop_path": str(crop_path),
            "nobg_path": nobg_path,
        })

        print(f"  {label}: 裁切并去背景完成 -> {nobg_path}")

    del remover
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return icon_infos


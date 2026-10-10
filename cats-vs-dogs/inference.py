"""任务 8：独立的推理程序。

输入一张本地图片，自动完成与训练时一致的预处理，输出“猫/狗”、预测概率，
并保存处理后的图片（带预测标注）。

用法：
    python inference.py --image path/to/photo.jpg
    python inference.py --image photo.jpg --ckpt outputs/cnn_aug_best.pt --out result.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import models as M  # noqa: E402

# 与训练/验证时完全一致的预处理参数（见 dataset.py）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_CKPT = HERE / "outputs" / "cnn_aug_best.pt"
CLASS_NAMES = {0: "猫 Cat", 1: "狗 Dog"}


def build_transform(size: int):
    return T.Compose([
        T.Resize(int(size * 1.14)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def to_displayable(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    x = (tensor.cpu() * std + mean).clamp(0, 1)
    return (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def load_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("arch", "cnn") == "mlp":
        model = M.MLP(size=ckpt.get("size", 64))
    else:
        model = M.SmallCNN(width=ckpt.get("width", 32))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def _font(size: int = 26):
    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def predict(image_path: Path, ckpt_path: Path = DEFAULT_CKPT,
            out_path: Path | None = None) -> dict:
    model, ckpt = load_model(ckpt_path)
    size = ckpt.get("size", 128)

    img = Image.open(image_path).convert("RGB")
    tensor = build_transform(size)(img)
    with torch.no_grad():
        prob = F.softmax(model(tensor.unsqueeze(0)), dim=1)[0]
    pred = int(prob.argmax())

    processed = to_displayable(tensor)
    result = {
        "image": str(image_path),
        "prediction": CLASS_NAMES[pred],
        "label": pred,
        "probability": float(prob[pred]),
        "prob_cat": float(prob[0]),
        "prob_dog": float(prob[1]),
        "processed_size": [size, size],
    }

    if out_path is None:
        out_dir = HERE / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"inference_{image_path.stem}.png"
    out_path = Path(out_path)
    canvas = Image.fromarray(processed)
    draw = ImageDraw.Draw(canvas)
    text = f"{CLASS_NAMES[pred]}  p={float(prob[pred]):.2f}"
    draw.rectangle([0, 0, canvas.width, 38], fill=(0, 0, 0))
    draw.text((8, 6), text, fill=(255, 255, 255), font=_font(24))
    canvas.save(out_path)
    result["output_image"] = str(out_path)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="猫狗分类推理程序")
    ap.add_argument("--image", required=True, help="本地图片路径")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--out", default=None, help="处理后图片保存路径")
    args = ap.parse_args(argv)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"找不到图片: {image_path}")
        return 1
    res = predict(image_path, Path(args.ckpt), args.out)
    print(f"图片      : {res['image']}")
    print(f"预测结果  : {res['prediction']}")
    print(f"预测概率  : {res['probability']:.4f}  "
          f"(猫 {res['prob_cat']:.4f} / 狗 {res['prob_dog']:.4f})")
    print(f"处理后图片: {res['output_image']}  (尺寸 {res['processed_size']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

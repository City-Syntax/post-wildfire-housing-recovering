"""Measure per-image and pre/post streetscape vegetation with SegFormer or HSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


ADE20K_VEGETATION = {4, 9, 17, 66, 72, 96, 109, 126}


def hsv_fraction(path: Path) -> float:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    difference = maximum - minimum
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hue = np.zeros_like(maximum)
    valid = difference > 0
    red_max = (maximum == red) & valid
    green_max = (maximum == green) & valid
    blue_max = (maximum == blue) & valid
    hue[red_max] = (60 * (green[red_max] - blue[red_max]) / difference[red_max]) % 360
    hue[green_max] = 60 * (blue[green_max] - red[green_max]) / difference[green_max] + 120
    hue[blue_max] = 60 * (red[blue_max] - green[blue_max]) / difference[blue_max] + 240
    saturation = np.divide(
        difference, maximum, out=np.zeros_like(difference), where=maximum != 0
    )
    vegetation = (
        (hue >= 60) & (hue <= 160) & (saturation > 0.15) & (maximum > 0.15)
    )
    return float(vegetation.mean())


def load_segformer(model_name: str):
    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return torch, processor, model, device


def segformer_fraction(path: Path, model_name: str, components=None) -> float:
    torch, processor, model, device = components or load_segformer(model_name)
    image = Image.open(path).convert("RGB")
    inputs = {
        key: value.to(device)
        for key, value in processor(images=image, return_tensors="pt").items()
    }
    with torch.no_grad():
        logits = model(**inputs).logits
    logits = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    labels = logits.argmax(dim=1)[0].cpu().numpy()
    return float(np.isin(labels, list(ADE20K_VEGETATION)).mean())


def measure_images(paths: list[Path], method: str, model_name: str) -> list[float]:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing image files: " + ", ".join(missing))
    if method == "hsv":
        return [hsv_fraction(path) for path in paths]
    components = load_segformer(model_name)
    return [segformer_fraction(path, model_name, components) for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path)
    parser.add_argument("--pre-images", nargs="+", type=Path)
    parser.add_argument("--post-images", nargs="+", type=Path)
    parser.add_argument("--method", choices=["segformer", "hsv"], default="segformer")
    parser.add_argument(
        "--model",
        default="nvidia/segformer-b0-finetuned-ade-512-512",
    )
    args = parser.parse_args()
    sequence_mode = args.pre_images is not None or args.post_images is not None
    if args.image and sequence_mode:
        raise SystemExit("Use either one positional image or --pre-images/--post-images")
    if not args.image and not (args.pre_images and args.post_images):
        raise SystemExit("Provide one image, or both --pre-images and --post-images")

    if args.image:
        fraction = measure_images([args.image], args.method, args.model)[0]
        result = {"method": args.method, "vegetation_fraction": fraction}
    else:
        values = measure_images(
            args.pre_images + args.post_images, args.method, args.model
        )
        pre = values[: len(args.pre_images)]
        post = values[len(args.pre_images) :]
        pre_median = float(np.median(pre))
        post_median = float(np.median(post))
        result = {
            "method": args.method,
            "pre_image_fractions": pre,
            "post_image_fractions": post,
            "pre_median": pre_median,
            "post_median": post_median,
            "post_minus_pre": post_median - pre_median,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

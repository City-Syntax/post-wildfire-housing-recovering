"""Classify one chronological pre/post Street View sequence with Claude."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path


LABELS = {
    "empty_lot",
    "rebuilt_equal",
    "rebuilt_improved",
    "rebuilt_degraded",
    "obscured",
}

PROMPT = """Classify the visible recovery trajectory of one building destroyed
by wildfire. The attached images show the same parcel in chronological order.
Use only visible evidence in the images and judge the latest recovery state.

Return one of five labels using these operational definitions:
- empty_lot: bare ground, cleared rubble, or a foundation pad is visible, with
  no replacement superstructure;
- rebuilt_equal: the replacement preserves the original footprint, height, and
  apparent material grade within an approximate 10% scale tolerance;
- rebuilt_improved: the replacement has a visibly larger footprint, additional
  stories, higher-grade exterior materials, greater architectural complexity,
  or substantial high-end landscaping;
- rebuilt_degraded: the replacement has a smaller footprint, fewer stories,
  lower-grade materials, or is a manufactured/mobile-home replacement for a
  previously site-built structure;
- obscured: canopy, renderer artifacts, camera angle, or image quality prevents
  a reliable judgment. Default to obscured rather than guess.

Return JSON only in this form:
{"trajectory":"one_label","confidence":0.0,"evidence":"brief visual rationale"}
Confidence must be a number from 0 to 1.
"""


def image_block(path: Path) -> dict:
    suffix = path.suffix.lower()
    media_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    if suffix not in media_types:
        raise ValueError(f"Unsupported image type: {path}")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_types[suffix],
            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
        },
    }


def parse_result(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("The model did not return a JSON object")
    raw = json.loads(match.group(0))
    trajectory = str(raw.get("trajectory", raw.get("label", ""))).strip().lower()
    if trajectory not in LABELS:
        raise ValueError(f"Unexpected trajectory label: {trajectory}")
    try:
        confidence = float(raw["confidence"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("The model did not return a numeric confidence") from error
    if not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1")
    evidence = str(raw.get("evidence", "")).strip()
    if not evidence:
        raise ValueError("The model did not return visual evidence")
    return {
        "trajectory": trajectory,
        "confidence": confidence,
        "evidence": evidence,
    }


def classify(client, model: str, images: list[Path]) -> dict:
    content = [{"type": "text", "text": PROMPT}]
    content.extend(image_block(path) for path in images)
    response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": content}],
    )
    response_text = "\n".join(
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    )
    return parse_result(response_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, nargs="+", type=Path)
    parser.add_argument("--model", required=True, help="exact API model identifier")
    args = parser.parse_args()

    if not 2 <= len(args.images) <= 8:
        raise SystemExit("Provide between two and eight chronological images")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY before running classification")
    missing = [str(path) for path in args.images if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing image files: " + ", ".join(missing))

    try:
        from anthropic import Anthropic
    except ImportError as error:
        raise SystemExit(
            "Install the repository environment before classification: "
            "conda env create -f environment.yml"
        ) from error
    result = classify(Anthropic(), args.model, args.images)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

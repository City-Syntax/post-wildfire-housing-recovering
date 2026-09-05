"""Render one authorized panorama reference with the Street View Static API."""

from __future__ import annotations

import argparse
import io
import os
import re
from pathlib import Path

import requests
from PIL import Image


API_URL = "https://maps.googleapis.com/maps/api/streetview"


def render_panorama(
    *,
    panoid: str,
    heading: float,
    output: Path,
    api_key: str,
    fov: int = 90,
    pitch: int = 0,
    size: str = "640x640",
    http=requests,
) -> Path:
    if not panoid.strip():
        raise ValueError("panoid cannot be empty")
    if not 0 <= heading < 360:
        raise ValueError("heading must be in [0, 360)")
    if not 10 <= fov <= 120:
        raise ValueError("fov must be between 10 and 120 degrees")
    if not -90 <= pitch <= 90:
        raise ValueError("pitch must be between -90 and 90 degrees")
    if not re.fullmatch(r"[1-9]\d*x[1-9]\d*", size):
        raise ValueError("size must use WIDTHxHEIGHT format, for example 640x640")

    response = http.get(
        API_URL,
        params={
            "pano": panoid,
            "heading": heading,
            "fov": fov,
            "pitch": pitch,
            "size": size,
            "key": api_key,
        },
        timeout=30,
    )
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    image.load()
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=95)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panoid", required=True)
    parser.add_argument("--heading", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fov", type=int, default=90)
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--size", default="640x640")
    args = parser.parse_args()

    key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not key:
        raise SystemExit("Set GOOGLE_MAPS_API_KEY before requesting an image")

    output = render_panorama(
        panoid=args.panoid,
        heading=args.heading,
        output=args.output,
        api_key=key,
        fov=args.fov,
        pitch=args.pitch,
        size=args.size,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

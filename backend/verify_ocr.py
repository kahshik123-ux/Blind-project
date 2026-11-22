from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from backend.main import app


def main() -> None:
    image = Image.new("RGB", (1200, 400), "white")
    font = ImageFont.truetype(
        str(Path(os.environ["WINDIR"], "Fonts", "arial.ttf")),
        190,
    )
    draw = ImageDraw.Draw(image)
    draw.text((55, 75), "HELLO", fill="black", font=font)
    image_buffer = BytesIO()
    image.save(image_buffer, format="PNG")

    with TestClient(app) as client:
        response = client.post(
            "/api/ocr/combined",
            files={"image": ("hello.png", image_buffer.getvalue(), "image/png")},
        )
    response.raise_for_status()
    result = response.json()

    print(f"Primary result ({result['primary_engine']}): {result['text']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Uncertain regions: {len(result['uncertain_regions'])}")
    for engine_name in ("paddle", "easy"):
        engine = result["engines"][engine_name]
        print(
            f"{engine_name}: {engine['text'] if engine else 'unavailable'} "
            f"(confidence {engine['confidence'] if engine else 0}%)"
        )
        if not engine or "hello" not in engine["text"].casefold():
            raise SystemExit(f"{engine_name} failed to recognize HELLO.")


if __name__ == "__main__":
    main()

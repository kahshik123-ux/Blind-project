from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


MAX_IMAGE_PIXELS = 36_000_000
MAX_IMAGE_SIDE = 1600
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


@dataclass(frozen=True)
class ImageDiagnostics:
    width: int
    height: int
    brightness: float
    sharpness: float
    is_blurry: bool

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "width": self.width,
            "height": self.height,
            "brightness": self.brightness,
            "sharpness": self.sharpness,
            "is_blurry": self.is_blurry,
        }


def decode_and_preprocess(image_bytes: bytes) -> tuple[np.ndarray, ImageDiagnostics]:
    if not image_bytes:
        raise ValueError("The uploaded image is empty.")

    try:
        with Image.open(BytesIO(image_bytes)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("The image is too large to process.")
            oriented = ImageOps.exif_transpose(source)
            rgb = np.asarray(oriented.convert("RGB"))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The uploaded file is not a supported image.") from exc

    scale = min(1.0, MAX_IMAGE_SIDE / max(rgb.shape[1], rgb.shape[0]))
    if scale < 1:
        rgb = cv2.resize(
            rgb,
            (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))

    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=5, templateWindowSize=7, searchWindowSize=21)
    sharpened = cv2.addWeighted(denoised, 1.25, cv2.GaussianBlur(denoised, (0, 0), 1.2), -0.25, 0)
    processed = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    diagnostics = ImageDiagnostics(
        width=int(rgb.shape[1]),
        height=int(rgb.shape[0]),
        brightness=round(brightness, 1),
        sharpness=round(sharpness, 1),
        is_blurry=sharpness < 45,
    )
    return processed, diagnostics

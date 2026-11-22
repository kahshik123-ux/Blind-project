from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_DIR / "models"
PADDLE_MODELS = MODELS_DIR / "paddle"
EASYOCR_MODELS = MODELS_DIR / "easyocr"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PADDLE_MODELS.mkdir(parents=True, exist_ok=True)
    EASYOCR_MODELS.mkdir(parents=True, exist_ok=True)
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(PADDLE_MODELS)
    os.environ["PADDLE_PDX_MODEL_SOURCE"] = "bos"
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"

    print("Downloading PaddleOCR English CPU models from Paddle's official model host ...", flush=True)
    from backend.ocr_service import _get_paddle_engine

    paddle_engine = _get_paddle_engine()
    list(paddle_engine.predict(np.full((128, 512, 3), 255, dtype=np.uint8)))
    print("PaddleOCR detector, recognizer, and orientation models are ready.", flush=True)

    print("Downloading EasyOCR English CPU models into backend/models/easyocr ...", flush=True)
    from backend.ocr_service import _get_easy_reader

    _get_easy_reader()
    print("EasyOCR English models are ready.", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import base64
import importlib
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from backend.image_processing import decode_and_preprocess
from backend.ocr_service import (
    EASYOCR_MODELS,
    PADDLE_MODELS,
    compare_results,
    recognize_easy,
    recognize_paddle,
)


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
PADDLE_MODEL_NAMES = (
    "PP-OCRv6_small_det",
    "PP-OCRv6_small_rec",
    "PP-LCNet_x1_0_doc_ori",
    "PP-LCNet_x0_25_textline_ori",
)

app = FastAPI(title="VisionVoice Local OCR API", version="1.0.0")
LOGGER = logging.getLogger(__name__)
_PADDLE_RUNTIME_ERROR: str | None = None
_PADDLE_RUNTIME_VERIFIED = False


def _paddle_runtime_status() -> tuple[bool, str | None]:
    try:
        importlib.import_module("paddleocr")
    except (ImportError, OSError) as exc:
        LOGGER.warning("PaddleOCR runtime could not be imported: %s", exc)
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


class AIAnalysisRequest(BaseModel):
    image_base64: str
    mime_type: str = "image/jpeg"
    paddle_text: str = ""
    easy_text: str = ""


async def _read_image(file: UploadFile) -> tuple[Any, dict[str, Any]]:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, WebP, or BMP image.")
    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The image exceeds the 15 MiB upload limit.")
    try:
        image, diagnostics = await run_in_threadpool(decode_and_preprocess, image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return image, diagnostics.as_dict()


def _engine_call(engine_name: str, image: Any) -> tuple[dict[str, Any] | None, str | None]:
    global _PADDLE_RUNTIME_ERROR, _PADDLE_RUNTIME_VERIFIED
    try:
        if engine_name == "paddle":
            result = recognize_paddle(image)
            _PADDLE_RUNTIME_ERROR = None
            _PADDLE_RUNTIME_VERIFIED = True
            return result, None
        return recognize_easy(image), None
    except Exception as exc:
        if engine_name == "paddle":
            _PADDLE_RUNTIME_ERROR = f"{type(exc).__name__}: {exc}"
            _PADDLE_RUNTIME_VERIFIED = False
        LOGGER.exception("%s failed to recognize the uploaded image", engine_name)
        return None, f"{type(exc).__name__}: {exc}"


async def _combined(image: Any, diagnostics: dict[str, Any]) -> dict[str, Any]:
    paddle_result, paddle_error = await run_in_threadpool(_engine_call, "paddle", image)
    easy_result, easy_error = await run_in_threadpool(_engine_call, "easy", image)
    if paddle_result is None and easy_result is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Both local OCR engines failed to process the image.",
                "paddle_error": paddle_error,
                "easy_error": easy_error,
            },
        )
    try:
        comparison = compare_results(paddle_result, easy_result)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        **comparison,
        "image_diagnostics": diagnostics,
        "engine_errors": {
            "paddle": paddle_error,
            "easy": easy_error,
        },
        "offline_local": True,
    }


@app.get("/api/health")
async def health() -> dict[str, Any]:
    paddle_models_ready = all(
        (PADDLE_MODELS / "official_models" / name / "inference.yml").is_file()
        for name in PADDLE_MODEL_NAMES
    )
    paddle_runtime_ready, paddle_error = _paddle_runtime_status()
    if _PADDLE_RUNTIME_ERROR:
        paddle_runtime_ready = False
        paddle_error = _PADDLE_RUNTIME_ERROR
    paddle_runtime_verified = (
        paddle_models_ready
        and paddle_runtime_ready
        and _PADDLE_RUNTIME_VERIFIED
    )
    return {
        "status": "ok",
        "paddle_available": paddle_runtime_verified,
        "paddle_runtime_verified": paddle_runtime_verified,
        "paddle_models_ready": paddle_models_ready,
        "paddle_error": paddle_error,
        "easyocr_available": (
            (EASYOCR_MODELS / "craft_mlt_25k.pth").is_file()
            and (EASYOCR_MODELS / "english_g2.pth").is_file()
        ),
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
    }


@app.post("/api/ocr/paddle")
async def paddle_ocr(image: UploadFile = File(...)) -> dict[str, Any]:
    processed, diagnostics = await _read_image(image)
    result, error = await run_in_threadpool(_engine_call, "paddle", processed)
    if result is None:
        raise HTTPException(status_code=503, detail=f"PaddleOCR failed: {error}")
    return {**result, "image_diagnostics": diagnostics, "offline_local": True}


@app.post("/api/ocr/easy")
async def easy_ocr(image: UploadFile = File(...)) -> dict[str, Any]:
    processed, diagnostics = await _read_image(image)
    result, error = await run_in_threadpool(_engine_call, "easy", processed)
    if result is None:
        raise HTTPException(status_code=503, detail=f"EasyOCR failed: {error}")
    return {**result, "image_diagnostics": diagnostics, "offline_local": True}


@app.post("/api/ocr/combined")
async def combined_ocr(image: UploadFile = File(...)) -> dict[str, Any]:
    processed, diagnostics = await _read_image(image)
    return await _combined(processed, diagnostics)


@app.post("/api/ai/analyze")
async def analyze_with_gemini(request: AIAnalysisRequest) -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Gemini is not configured. Local PaddleOCR and EasyOCR continue to work.",
        )
    if request.mime_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported image type.")
    if len(request.image_base64) > (MAX_UPLOAD_BYTES * 4 // 3) + 8:
        raise HTTPException(status_code=413, detail="The image exceeds the 15 MiB request limit.")
    try:
        image_bytes = base64.b64decode(request.image_base64, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise HTTPException(status_code=400, detail="The image data is not valid base64.") from exc
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The image exceeds the 15 MiB request limit.")

    prompt = (
        "Read only text that is visibly present in this document image. Do not infer, "
        "correct from context, or invent missing words. Preserve reading order and line "
        "breaks. Compare this transcription with the two local OCR outputs below; identify "
        "only clearly visible corrections, and mark anything unreadable as UNCLEAR. "
        "Return the exact visible transcription and a short list of uncertain regions. "
        "Do not replace either OCR result silently.\n\n"
        f"PaddleOCR:\n{request.paddle_text[:12000]}\n\n"
        f"EasyOCR:\n{request.easy_text[:12000]}"
    )
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        try:
            async with client.aio as async_client:
                response = await async_client.interactions.create(
                    model=os.getenv("GEMINI_MODEL", "gemini-4-hydrogen"),
                    input=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "data": base64.b64encode(image_bytes).decode("ascii"),
                            "mime_type": request.mime_type,
                        },
                    ],
                    store=False,
                )
        finally:
            client.close()
        return {
            "analysis": response.output_text or "",
            "model": os.getenv("GEMINI_MODEL", "gemini-4-hydrogen"),
            "advisory_only": True,
        }
    except Exception as exc:
        LOGGER.exception("Gemini advisory image analysis failed")
        raise HTTPException(
            status_code=502,
            detail=f"Gemini image analysis failed. Local OCR is still available ({type(exc).__name__}).",
        ) from exc

from __future__ import annotations

import os
import re
import threading
from collections.abc import Mapping
from difflib import SequenceMatcher
import logging
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
PADDLE_MODELS = MODELS_DIR / "paddle"
EASYOCR_MODELS = MODELS_DIR / "easyocr"
os.environ["PADDLE_PDX_CACHE_HOME"] = str(PADDLE_MODELS)
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ["PADDLE_PDX_MODEL_SOURCE"] = "bos"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"

LOGGER = logging.getLogger(__name__)
_paddle_lock = threading.Lock()
_easy_lock = threading.Lock()
_paddle_engine: Any = None
_easy_reader: Any = None


def _new_paddle_engine() -> Any:
    from paddleocr import PaddleOCR

    return PaddleOCR(
        text_detection_model_name="PP-OCRv6_small_det",
        text_recognition_model_name="PP-OCRv6_small_rec",
        doc_orientation_classify_model_name="PP-LCNet_x1_0_doc_ori",
        textline_orientation_model_name="PP-LCNet_x0_25_textline_ori",
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        text_det_limit_side_len=1600,
        text_det_limit_type="max",
        device="cpu",
    )


def _get_paddle_engine() -> Any:
    global _paddle_engine
    if _paddle_engine is None:
        with _paddle_lock:
            if _paddle_engine is None:
                _paddle_engine = _new_paddle_engine()
    return _paddle_engine


def _get_easy_reader() -> Any:
    global _easy_reader
    if _easy_reader is None:
        with _easy_lock:
            if _easy_reader is None:
                import easyocr

                _easy_reader = easyocr.Reader(
                    ["en"],
                    gpu=False,
                    model_storage_directory=str(EASYOCR_MODELS),
                    download_enabled=True,
                    verbose=True,
                )
    return _easy_reader


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _paddle_result_mapping(result: Any) -> dict[str, Any]:
    result = _plain(result)
    if not isinstance(result, Mapping):
        return {}
    payload = result.get("res", result)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _sort_reading_order(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not lines:
        return []
    heights = [
        max(1.0, float(line["bbox"][3]) - float(line["bbox"][1]))
        for line in lines
    ]
    tolerance = max(8.0, float(np.median(heights)) * 0.6)
    pending = sorted(lines, key=lambda line: (line["center_y"], line["bbox"][0]))
    rows: list[list[dict[str, Any]]] = []
    for line in pending:
        row = next(
            (row for row in rows if abs(float(np.mean([item["center_y"] for item in row])) - line["center_y"]) <= tolerance),
            None,
        )
        if row is None:
            rows.append([line])
        else:
            row.append(line)
    return [
        line
        for row in sorted(rows, key=lambda group: min(item["center_y"] for item in group))
        for line in sorted(row, key=lambda item: item["bbox"][0])
    ]


def _line(text: str, confidence: float, bbox: Any, engine: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        points = np.asarray(bbox, dtype=np.float64).reshape(-1, 2)
        if points.size == 0:
            return None
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
    except (TypeError, ValueError):
        return None
    return {
        "text": text,
        "confidence": round(float(confidence), 4),
        "bbox": [round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1)],
        "center_y": float((y1 + y2) / 2),
        "engine": engine,
    }


def _paddle_lines(image: np.ndarray) -> list[dict[str, Any]]:
    outputs = _get_paddle_engine().predict(image)
    lines: list[dict[str, Any]] = []
    for output in outputs:
        payload = _paddle_result_mapping(output)
        texts = payload.get("rec_texts", [])
        scores = payload.get("rec_scores", [])
        boxes = payload.get("rec_polys", payload.get("rec_boxes", []))
        for index, text in enumerate(texts):
            if index >= len(boxes):
                continue
            score = float(scores[index]) if index < len(scores) else 0.0
            item = _line(str(text), score, boxes[index], "paddle")
            if item:
                lines.append(item)
    return _sort_reading_order(lines)


def _easy_lines(image: np.ndarray) -> list[dict[str, Any]]:
    results = _get_easy_reader().readtext(
        image,
        detail=1,
        paragraph=False,
        canvas_size=1280,
        min_size=8,
        contrast_ths=0.1,
        adjust_contrast=0.6,
    )
    lines = []
    for box, text, confidence in results:
        item = _line(str(text), float(confidence), box, "easyocr")
        if item:
            lines.append(item)
    return _sort_reading_order(lines)


def recognize_paddle(image: np.ndarray) -> dict[str, Any]:
    lines = _paddle_lines(image)
    return {
        "engine": "PaddleOCR PP-OCRv6 small",
        "text": "\n".join(line["text"] for line in lines),
        "confidence": _mean_confidence(lines),
        "lines": _public_lines(lines),
    }


def recognize_easy(image: np.ndarray) -> dict[str, Any]:
    lines = _easy_lines(image)
    return {
        "engine": "EasyOCR English",
        "text": "\n".join(line["text"] for line in lines),
        "confidence": _mean_confidence(lines),
        "lines": _public_lines(lines),
    }


def _public_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "text": line["text"],
            "confidence": line["confidence"],
            "bbox": line["bbox"],
        }
        for line in lines
    ]


def _mean_confidence(lines: list[dict[str, Any]]) -> float:
    if not lines:
        return 0.0
    return round(float(np.mean([line["confidence"] for line in lines])) * 100, 1)


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def _intersection_over_union(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union else 0.0


def compare_results(
    paddle_result: dict[str, Any] | None,
    easy_result: dict[str, Any] | None,
) -> dict[str, Any]:
    paddle_lines = paddle_result["lines"] if paddle_result else []
    easy_lines = easy_result["lines"] if easy_result else []
    uncertain: list[dict[str, Any]] = []
    matched_easy: set[int] = set()

    for paddle_line in paddle_lines:
        best_index = -1
        best_overlap = 0.0
        for index, easy_line in enumerate(easy_lines):
            if index in matched_easy:
                continue
            overlap = _intersection_over_union(paddle_line["bbox"], easy_line["bbox"])
            if overlap > best_overlap:
                best_index = index
                best_overlap = overlap

        if best_index < 0 or best_overlap < 0.12:
            uncertain.append({
                "bbox": paddle_line["bbox"],
                "paddle_text": paddle_line["text"],
                "easy_text": None,
                "reason": "no matching EasyOCR line",
            })
            continue

        matched_easy.add(best_index)
        easy_line = easy_lines[best_index]
        agreement = SequenceMatcher(
            None,
            _normalized(paddle_line["text"]),
            _normalized(easy_line["text"]),
        ).ratio()
        if agreement < 0.85 or min(paddle_line["confidence"], easy_line["confidence"]) < 0.45:
            uncertain.append({
                "bbox": paddle_line["bbox"],
                "paddle_text": paddle_line["text"],
                "easy_text": easy_line["text"],
                "reason": "OCR engines disagree or have low confidence",
            })

    for index, easy_line in enumerate(easy_lines):
        if index not in matched_easy:
            uncertain.append({
                "bbox": easy_line["bbox"],
                "paddle_text": None,
                "easy_text": easy_line["text"],
                "reason": "no matching PaddleOCR line",
            })

    primary = paddle_result or easy_result
    if primary is None:
        raise RuntimeError("Both local OCR engines failed to return a result.")

    return {
        "text": primary["text"],
        "confidence": primary["confidence"],
        "primary_engine": primary["engine"],
        "engines": {
            "paddle": paddle_result,
            "easy": easy_result,
        },
        "uncertain_regions": uncertain,
        "needs_review": bool(uncertain) or primary["confidence"] < 60,
    }

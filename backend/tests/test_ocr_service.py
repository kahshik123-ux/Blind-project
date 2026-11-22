import unittest

from backend.ocr_service import _normalized, _paddle_result_mapping, compare_results


def line(text: str, confidence: float, bbox: list[float]) -> dict[str, object]:
    return {"text": text, "confidence": confidence, "bbox": bbox}


class OCRComparisonTests(unittest.TestCase):
    def test_paddle_payload_unwraps_result_object(self) -> None:
        result = _paddle_result_mapping({
            "res": {"rec_texts": ["Hello"], "rec_scores": [0.98]},
        })
        self.assertEqual(result["rec_texts"], ["Hello"])

    def test_normalization_ignores_case_and_punctuation(self) -> None:
        self.assertEqual(_normalized("Hello, WORLD!"), "hello world")

    def test_agreeing_engines_do_not_create_uncertain_regions(self) -> None:
        paddle = {
            "engine": "PaddleOCR",
            "text": "Hello",
            "confidence": 96.0,
            "lines": [line("Hello", 0.96, [0, 0, 100, 30])],
        }
        easy = {
            "engine": "EasyOCR",
            "text": "hello",
            "confidence": 92.0,
            "lines": [line("hello", 0.92, [2, 0, 102, 30])],
        }
        result = compare_results(paddle, easy)
        self.assertEqual(result["text"], "Hello")
        self.assertFalse(result["needs_review"])
        self.assertEqual(result["uncertain_regions"], [])

    def test_disagreeing_engines_keep_primary_text_and_flag_region(self) -> None:
        paddle = {
            "engine": "PaddleOCR",
            "text": "Hello",
            "confidence": 90.0,
            "lines": [line("Hello", 0.9, [0, 0, 100, 30])],
        }
        easy = {
            "engine": "EasyOCR",
            "text": "Hella",
            "confidence": 88.0,
            "lines": [line("Hella", 0.88, [2, 0, 102, 30])],
        }
        result = compare_results(paddle, easy)
        self.assertEqual(result["text"], "Hello")
        self.assertTrue(result["needs_review"])
        self.assertEqual(result["uncertain_regions"][0]["easy_text"], "Hella")

    def test_easyocr_is_used_if_paddle_fails(self) -> None:
        easy = {
            "engine": "EasyOCR",
            "text": "Hello",
            "confidence": 91.0,
            "lines": [line("Hello", 0.91, [0, 0, 100, 30])],
        }
        result = compare_results(None, easy)
        self.assertEqual(result["text"], "Hello")
        self.assertEqual(result["primary_engine"], "EasyOCR")
        self.assertIsNone(result["engines"]["paddle"])


if __name__ == "__main__":
    unittest.main()

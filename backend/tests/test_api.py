import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app


class OCRApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health_does_not_load_models(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_ocr_rejects_non_image_upload(self) -> None:
        response = self.client.post(
            "/api/ocr/combined",
            files={"image": ("document.txt", b"not an image", "text/plain")},
        )
        self.assertEqual(response.status_code, 415)

    def test_gemini_requires_server_side_key(self) -> None:
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            response = self.client.post(
                "/api/ai/analyze",
                json={
                    "image_base64": "aGVsbG8=",
                    "mime_type": "image/jpeg",
                    "paddle_text": "Hello",
                    "easy_text": "Hello",
                },
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("not configured", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()

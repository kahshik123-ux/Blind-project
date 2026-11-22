import io
import unittest

import numpy as np
from PIL import Image

from backend.image_processing import decode_and_preprocess


def encoded_test_image() -> bytes:
    image = Image.fromarray(np.full((80, 120, 3), 255, dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class ImageProcessingTests(unittest.TestCase):
    def test_image_is_preprocessed_and_diagnostics_report_dimensions(self) -> None:
        processed, diagnostics = decode_and_preprocess(encoded_test_image())
        self.assertEqual(processed.shape, (80, 120, 3))
        self.assertEqual(diagnostics.width, 120)
        self.assertEqual(diagnostics.height, 80)
        self.assertTrue(diagnostics.is_blurry)

    def test_large_image_is_resized_before_preprocessing(self) -> None:
        image = Image.fromarray(np.full((80, 3000, 3), 255, dtype=np.uint8))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        processed, diagnostics = decode_and_preprocess(buffer.getvalue())

        self.assertEqual(processed.shape, (43, 1600, 3))
        self.assertEqual(diagnostics.width, 1600)
        self.assertEqual(diagnostics.height, 43)

    def test_invalid_image_is_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "supported image"):
            decode_and_preprocess(b"not an image")


if __name__ == "__main__":
    unittest.main()

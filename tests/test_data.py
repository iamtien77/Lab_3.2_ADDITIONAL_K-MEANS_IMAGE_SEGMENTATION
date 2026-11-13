"""Kiểm thử hợp đồng bàn giao dữ liệu ảnh của TASK-01."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.config import DEFAULT_IMAGE_PATH, TARGET_SIZE
from src.data.loader import load_image, validate_image
from src.data.preprocessing import convert_color_space, flatten_pixels, resize_image


class TestDataModule(unittest.TestCase):
    def test_default_image_metadata(self):
        image, metadata = load_image()
        self.assertTrue(validate_image(image))
        self.assertEqual(metadata["image_path"], str(DEFAULT_IMAGE_PATH.resolve()))
        self.assertEqual(metadata["filename"], DEFAULT_IMAGE_PATH.name)
        self.assertEqual(metadata["original_shape"], image.shape)
        self.assertEqual(metadata["total_pixels"], image.shape[0] * image.shape[1])
        self.assertEqual(metadata["file_size_bytes"], DEFAULT_IMAGE_PATH.stat().st_size)
        self.assertEqual(metadata["dtype"], "uint8")

    def test_custom_image_color_and_pillow_fallback(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "red.png"
            Image.new("RGB", (2, 1), (255, 0, 0)).save(path)
            image, metadata = load_image(path)
            expected = {"BGR": [0, 0, 255], "RGB": [255, 0, 0]}
            np.testing.assert_array_equal(image[0, 0], expected[metadata["color_space"]])
            with patch("src.data.loader.cv2", None):
                image, metadata = load_image(path)
            self.assertEqual(metadata["color_space"], "RGB")
            np.testing.assert_array_equal(image[0, 0], [255, 0, 0])

    def test_bgr_converts_and_rgb_stays_in_order(self):
        bgr = np.array([[[0, 0, 255], [10, 20, 30]]], dtype=np.uint8)
        expected = [[[255, 0, 0], [30, 20, 10]]]
        rgb = convert_color_space(bgr)
        np.testing.assert_array_equal(rgb, expected)
        np.testing.assert_array_equal(convert_color_space(rgb, "RGB"), expected)

    def test_resize_preserves_aspect_ratio_and_never_upscales(self):
        for (height, width), expected in (
            ((600, 800), (192, 256)),
            ((800, 600), (256, 192)),
            ((100, 200), (100, 200)),
            ((1, 1024), (1, 256)),
            ((1024, 1), (256, 1)),
            ((256, 128), (256, 128)),
        ):
            with self.subTest(size=(height, width)):
                image = np.zeros((height, width, 3), dtype=np.uint8)
                resized, shape = resize_image(image, TARGET_SIZE)
                self.assertEqual(shape, expected)
                self.assertEqual(resized.shape, (*expected, 3))
                self.assertEqual(resized.dtype, np.uint8)

    def test_resize_square_mode_and_invalid_target(self):
        image = np.zeros((6, 4, 3), dtype=np.uint8)
        resized, shape = resize_image(image, 3, keep_aspect_ratio=False)
        self.assertEqual(shape, (3, 3))
        self.assertEqual(resized.shape, (3, 3, 3))
        with self.assertRaisesRegex(ValueError, "target_size"):
            resize_image(image, 0)

    def test_flatten_pixel_order_dtype_range_and_round_trip(self):
        image = np.array([[[255, 0, 1], [2, 3, 4]], [[5, 6, 7], [8, 9, 10]]], dtype=np.uint8)
        expected = [[255, 0, 1], [2, 3, 4], [5, 6, 7], [8, 9, 10]]
        pixels = flatten_pixels(image)
        self.assertEqual(pixels.shape, (4, 3))
        self.assertEqual(pixels.dtype, np.float32)
        self.assertTrue(np.isfinite(pixels).all())
        self.assertEqual((pixels.min(), pixels.max()), (0, 255))
        np.testing.assert_array_equal(pixels, expected)
        np.testing.assert_array_equal(pixels.reshape(image.shape), image)
        self.assertEqual(flatten_pixels(image, as_float=False).dtype, np.uint8)

    def test_solid_color_and_original_preserved(self):
        source = np.full((300, 500, 3), [3, 100, 255], dtype=np.uint8)
        before = source.copy()
        original_rgb = convert_color_space(source)
        resized_rgb, (height, width) = resize_image(original_rgb)
        pixels = flatten_pixels(resized_rgb)
        expected = [255, 100, 3]
        np.testing.assert_array_equal(pixels, np.tile(expected, (height * width, 1)))
        np.testing.assert_array_equal(pixels.reshape(height, width, 3), resized_rgb)
        pixels[0] = 0
        resized_rgb[0, 0] = 0
        np.testing.assert_array_equal(original_rgb, before[..., ::-1])
        np.testing.assert_array_equal(source, before)

    def test_missing_corrupt_grayscale_and_alpha_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "Không tìm thấy"):
                load_image(root / "missing.png")
            invalid = root / "invalid.png"
            invalid.write_bytes(b"not an image")
            with self.assertRaisesRegex(ValueError, "Không đọc được"):
                load_image(invalid)
            for mode, message in (("L", "Ảnh xám|mode L"), ("RGBA", "alpha")):
                with self.subTest(mode=mode):
                    path = root / f"{mode}.png"
                    Image.new(mode, (3, 2)).save(path)
                    with self.assertRaisesRegex(ValueError, message):
                        load_image(path)

    def test_invalid_arrays(self):
        for image, error, message in (
            (None, ValueError, "rỗng"),
            (np.empty((0, 3, 3), dtype=np.uint8), ValueError, "rỗng"),
            (np.zeros((3, 2), dtype=np.uint8), ValueError, "xám"),
            (np.zeros((3, 2, 4), dtype=np.uint8), ValueError, "alpha"),
            (np.zeros((3, 2, 3), dtype=np.float32), TypeError, "uint8"),
            ([], TypeError, "NumPy"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(error, message):
                    validate_image(image)


if __name__ == "__main__":
    unittest.main()

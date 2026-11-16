"""
KIỂM THỬ ĐƠN VỊ: MODULE DỮ LIỆU & TIỀN XỬ LÝ (TEST DATA & PREPROCESSING)
"""

import unittest
from pathlib import Path
import numpy as np

from src.config import DEFAULT_IMAGE_PATH, TARGET_SIZE
from src.data.loader import load_image, validate_image
from src.data.preprocessing import convert_color_space, resize_image, flatten_pixels


class TestDataModule(unittest.TestCase):
    """
    Kiểm tra tính đúng đắn của việc nạp ảnh và tiền xử lý ảnh.
    """

    def test_default_image_exists(self):
        """Kiểm tra file ảnh mẫu data/raw/image-liquid.jpg có tồn tại trên đĩa."""
        self.assertTrue(DEFAULT_IMAGE_PATH.exists(), f"Không tìm thấy ảnh: {DEFAULT_IMAGE_PATH}")

    def test_load_image(self):
        """Kiểm tra hàm load_image đọc ra mảng 3 chiều uint8 hợp lệ."""
        img, metadata = load_image(DEFAULT_IMAGE_PATH)
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.ndim, 3)
        self.assertEqual(img.shape[2], 3)
        self.assertEqual(img.dtype, np.uint8)
        self.assertIn("original_shape", metadata)
        self.assertTrue(validate_image(img))

    def test_convert_color_space(self):
        """Kiểm tra chuyển đổi BGR sang RGB và RGB sang HSV giữ nguyên kích thước."""
        dummy_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        dummy_rgb[0, 0] = [255, 0, 0]
        hsv = convert_color_space(dummy_rgb, "RGB", "HSV")
        self.assertEqual(hsv.shape, (10, 10, 3))

    def test_resize_image(self):
        """Kiểm tra resize ảnh về kích thước target_size=256 chính xác."""
        dummy_img = np.zeros((500, 1000, 3), dtype=np.uint8)
        resized, (new_h, new_w) = resize_image(dummy_img, target_size=TARGET_SIZE, keep_aspect_ratio=True)
        self.assertEqual(max(new_h, new_w), TARGET_SIZE)
        self.assertEqual(new_w, TARGET_SIZE)
        self.assertEqual(new_h, 128)

    def test_flatten_pixels(self):
        """Kiểm tra ma trận pixel 2D có đúng kích thước (H*W, 3)."""
        dummy_img = np.zeros((16, 16, 3), dtype=np.uint8)
        pixel_matrix = flatten_pixels(dummy_img, as_float=True)
        self.assertEqual(pixel_matrix.shape, (256, 3))
        self.assertEqual(pixel_matrix.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()

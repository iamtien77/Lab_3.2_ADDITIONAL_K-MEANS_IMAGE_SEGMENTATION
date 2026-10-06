"""
KIỂM THỬ ĐƠN VỊ: MODULE DỮ LIỆU & TIỀN XỬ LÝ (TEST DATA & PREPROCESSING)
"""

import unittest
from pathlib import Path
import numpy as np
from src.config import DEFAULT_IMAGE_PATH


class TestDataModule(unittest.TestCase):
    """
    Ý LÀM:
    1. test_default_image_exists: Kiểm tra file ảnh mẫu data/raw/image-liquid.jpg có tồn tại trên đĩa.
    2. test_load_image: Kiểm tra hàm load_image đọc ra mảng 3 chiều uint8 hợp lệ.
    3. test_convert_color_space: Kiểm tra chuyển đổi BGR sang RGB giữ nguyên kích thước.
    4. test_resize_image: Kiểm tra resize ảnh về kích thước target_size=256 chính xác.
    5. test_flatten_pixels: Kiểm tra ma trận pixel 2D có đúng kích thước (H*W, 3).
    """
    pass


if __name__ == "__main__":
    unittest.main()

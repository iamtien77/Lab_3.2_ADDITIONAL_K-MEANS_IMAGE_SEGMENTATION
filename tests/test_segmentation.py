"""
KIỂM THỬ ĐƠN VỊ: TÁI TẠO VÀ PHÂN ĐOẠN ẢNH (TEST IMAGE SEGMENTATION)
"""

import unittest
import numpy as np


class TestImageSegmentation(unittest.TestCase):
    """
    Ý LÀM:
    1. test_reconstruct_shape: Kiểm tra ảnh phân đoạn tái tạo có đúng kích thước (H, W, 3) và kiểu uint8.
    2. test_reconstruct_colors: Kiểm tra các giá trị màu trong ảnh phân đoạn chỉ bao gồm tọa độ các tâm cụm.
    3. test_cluster_masks: Kiểm tra mặt nạ nhị phân của K cụm có tổng diện tích bằng 100% bức ảnh.
    """
    pass


if __name__ == "__main__":
    unittest.main()

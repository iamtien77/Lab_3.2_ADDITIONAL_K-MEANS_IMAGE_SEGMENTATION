"""
KIỂM THỬ ĐƠN VỊ: MODULE TÁI TẠO VÀ PHÂN ĐOẠN ẢNH (TEST SEGMENTATION)
"""

import unittest
import numpy as np

from src.segmentation.segmenter import ImageSegmenter


class TestSegmentationModule(unittest.TestCase):
    """
    Kiểm tra tái tạo ảnh phân đoạn và phục hồi kích thước.
    """

    def setUp(self):
        self.segmenter = ImageSegmenter()
        self.centroids = np.array([
            [255.0, 0.0, 0.0],
            [0.0, 255.0, 0.0]
        ], dtype=np.float32)
        # 4 điểm ảnh luân phiên cụm 0 và 1
        self.labels = np.array([0, 1, 1, 0], dtype=np.int32)

    def test_reconstruct_segmented_image(self):
        """Kiểm tra tái tạo ảnh phân đoạn 2x2x3."""
        seg_img = self.segmenter.reconstruct_segmented_image(
            labels=self.labels,
            centroids=self.centroids,
            target_shape=(2, 2)
        )
        self.assertEqual(seg_img.shape, (2, 2, 3))
        self.assertEqual(seg_img.dtype, np.uint8)
        # Pixel (0,0) phải là màu đỏ [255, 0, 0]
        np.testing.assert_array_equal(seg_img[0, 0], [255, 0, 0])
        # Pixel (0,1) phải là màu xanh lục [0, 255, 0]
        np.testing.assert_array_equal(seg_img[0, 1], [0, 255, 0])

    def test_reconstruct_original_size(self):
        """Kiểm tra phóng đại nhãn về kích thước gốc 4x4x3."""
        seg_orig = self.segmenter.reconstruct_original_size(
            labels_resized=self.labels,
            centroids=self.centroids,
            original_shape=(4, 4, 3),
            resized_shape=(2, 2)
        )
        self.assertEqual(seg_orig.shape, (4, 4, 3))
        self.assertEqual(seg_orig.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()

"""
=============================================================================
KIỂM THỬ ĐƠN VỊ: TÁI TẠO VÀ PHÂN ĐOẠN ẢNH (TEST IMAGE SEGMENTATION)
=============================================================================
"""

import ast
import unittest
from pathlib import Path

import numpy as np

from src.segmentation.segmenter import ImageSegmenter


class TestImageSegmentation(unittest.TestCase):
    """Bộ kiểm thử cho module tái tạo ảnh, phục hồi độ phân giải, mặt nạ và ranh giới cụm."""

    def test_reconstruct_shape_and_colors(self) -> None:
        """Kiểm tra ảnh tái tạo đúng shape (H, W, 3) uint8 và chỉ chứa màu tâm cụm."""
        segmenter = ImageSegmenter()
        centroids = np.array([[10.4, 20.6, 30.2], [200.0, 100.1, 50.9]], dtype=np.float32)
        labels = np.array([0, 1, 0, 1, 0, 0], dtype=np.int64)
        seg = segmenter.reconstruct_segmented_image(labels, centroids, target_shape=(2, 3))

        self.assertEqual(seg.shape, (2, 3, 3))
        self.assertEqual(seg.dtype, np.uint8)
        unique_colors = {tuple(row) for row in seg.reshape(-1, 3)}
        expected_palette = {(10, 21, 30), (200, 100, 51)}
        self.assertTrue(unique_colors.issubset(expected_palette))

    def test_reconstruct_original_size_preserves_discrete_palette(self) -> None:
        """Kiểm tra phóng đại kích thước gốc bằng Nearest Neighbor không sinh màu trung gian."""
        segmenter = ImageSegmenter()
        centroids = np.array([[0.0, 0.0, 255.0], [255.0, 0.0, 0.0]], dtype=np.float32)
        labels_small = np.array([0, 1, 1, 0], dtype=np.int64)
        seg_orig = segmenter.reconstruct_original_size(
            labels_small,
            centroids,
            original_shape=(8, 12, 3),
            resized_shape=(2, 2),
        )
        self.assertEqual(seg_orig.shape, (8, 12, 3))
        self.assertEqual(seg_orig.dtype, np.uint8)
        unique_colors = {tuple(row) for row in seg_orig.reshape(-1, 3)}
        self.assertEqual(unique_colors, {(0, 0, 255), (255, 0, 0)})

    def test_cluster_masks_coverage_and_boundaries(self) -> None:
        """Kiểm tra tổng diện tích mặt nạ K cụm đạt 100% và phát hiện đúng đường ranh giới."""
        segmenter = ImageSegmenter()
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        masks = segmenter.extract_cluster_masks(labels, shape=(2, 2), k=2)
        self.assertEqual(len(masks), 2)

        total_covered = int(np.sum((masks[0] > 0) | (masks[1] > 0)))
        overlap = int(np.sum((masks[0] > 0) & (masks[1] > 0)))
        self.assertEqual(total_covered, 4)
        self.assertEqual(overlap, 0)

        boundaries = segmenter.extract_cluster_boundaries(labels, shape=(2, 2))
        self.assertEqual(boundaries.shape, (2, 2))
        self.assertTrue(np.all(boundaries == 255))

    def test_phase3_segmentation_docstrings_are_vietnamese(self) -> None:
        """Kiểm tra hồi quy: toàn bộ docstring trong module phân đoạn phải viết bằng Tiếng Việt."""
        for file_path in [Path("src/segmentation/segmenter.py"), Path("tests/test_segmentation.py")]:
            tree = ast.parse(file_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                    doc = ast.get_docstring(node) or ""
                    self.assertTrue(
                        any(ord(ch) > 127 for ch in doc),
                        f"Docstring tại {file_path}:{getattr(node, 'name', '<module>')} chưa viết bằng Tiếng Việt.",
                    )


if __name__ == "__main__":
    unittest.main()

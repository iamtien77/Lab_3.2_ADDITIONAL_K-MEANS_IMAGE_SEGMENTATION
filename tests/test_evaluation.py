"""
KIỂM THỬ ĐƠN VỊ: MODULE ĐÁNH GIÁ CHẤT LƯỢNG (TEST EVALUATION)
"""

import unittest
import numpy as np

from src.evaluation.metrics import compute_wcss, compute_silhouette


class TestEvaluationModule(unittest.TestCase):
    """
    Kiểm tra tính toán WCSS và Silhouette Score.
    """

    def test_compute_wcss(self):
        """Kiểm tra tính đúng giá trị WCSS."""
        X = np.array([[1.0, 1.0], [5.0, 5.0]], dtype=np.float32)
        centroids = np.array([[1.0, 1.0], [5.0, 5.0]], dtype=np.float32)
        labels = np.array([0, 1], dtype=np.int32)
        wcss = compute_wcss(X, centroids, labels)
        self.assertAlmostEqual(wcss, 0.0)

    def test_compute_silhouette(self):
        """Kiểm tra Silhouette Score trên 2 cụm tách biệt."""
        cluster1 = np.ones((20, 2), dtype=np.float32) * 2.0
        cluster2 = np.ones((20, 2), dtype=np.float32) * 50.0
        X = np.vstack([cluster1, cluster2])
        labels = np.array([0] * 20 + [1] * 20, dtype=np.int32)

        score = compute_silhouette(X, labels)
        self.assertGreater(score, 0.8)


if __name__ == "__main__":
    unittest.main()

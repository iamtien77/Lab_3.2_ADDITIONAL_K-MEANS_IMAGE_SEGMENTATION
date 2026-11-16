"""
KIỂM THỬ ĐƠN VỊ: MODULE MÔ HÌNH THUẬT TOÁN K-MEANS (TEST MODELS)
"""

import unittest
import numpy as np

from src.models.kmeans_scratch import KMeansFromScratch
from src.models.kmeans_sklearn import SklearnKMeansTrainer


class TestModelsModule(unittest.TestCase):
    """
    Kiểm tra thuật toán K-Means thuần và mô hình Sklearn.
    """

    def setUp(self):
        # Tạo dữ liệu giả lập 2 cụm màu rất rõ ràng
        np.random.seed(42)
        cluster1 = np.random.normal(loc=10.0, scale=1.0, size=(50, 3))
        cluster2 = np.random.normal(loc=200.0, scale=1.0, size=(50, 3))
        self.X = np.vstack([cluster1, cluster2]).astype(np.float32)

    def test_kmeans_scratch_convergence(self):
        """Kiểm tra K-Means scratch hội tụ và tách biệt đúng 2 cụm."""
        model = KMeansFromScratch(n_clusters=2, max_iter=30, tol=1e-4, random_state=42)
        model.fit(self.X)
        self.assertTrue(model.converged_)
        self.assertEqual(len(model.centroids), 2)
        self.assertEqual(len(model.labels_), 100)
        self.assertLess(model.inertia_, 1000.0)

        # Kiểm tra nhãn phân tách
        labels = model.predict(self.X)
        self.assertEqual(len(np.unique(labels)), 2)

    def test_sklearn_trainer_elbow(self):
        """Kiểm tra SklearnKMeansTrainer quét K và tìm điểm Elbow."""
        trainer = SklearnKMeansTrainer(random_state=42)
        df_elbow, optimal_k = trainer.search_elbow(self.X, k_range=range(2, 6))
        self.assertEqual(len(df_elbow), 4)
        self.assertIn("K", df_elbow.columns)
        self.assertIn("Inertia", df_elbow.columns)
        self.assertTrue(2 <= optimal_k <= 5)


if __name__ == "__main__":
    unittest.main()

"""
=============================================================================
KIỂM THỬ ĐƠN VỊ: THUẬT TOÁN K-MEANS & LƯU TRỮ MÔ HÌNH (TEST MODELS)
=============================================================================
"""

import ast
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from src.models.kmeans_opencv import OpenCVKMeansTrainer
from src.models.kmeans_scratch import KMeansFromScratch
from src.models.kmeans_sklearn import SklearnKMeansTrainer
from src.models.minibatch_kmeans import MiniBatchKMeansTrainer
from src.models.persistence import load_model_pipeline, save_model_pipeline


class TestKMeansModels(unittest.TestCase):
    """Bộ kiểm thử cho KMeansFromScratch, Scikit-Learn, OpenCV, MiniBatch và Persistence."""

    def _build_synthetic_clusters(self, n_per_cluster: int = 100) -> np.ndarray:
        """Tạo tập điểm ảnh nhân tạo gồm 4 cụm màu tách biệt rõ ràng trong không gian RGB."""
        rng = np.random.default_rng(42)
        centers = np.array(
            [
                [20.0, 20.0, 220.0],
                [230.0, 30.0, 30.0],
                [20.0, 210.0, 40.0],
                [235.0, 225.0, 220.0],
            ],
            dtype=np.float32,
        )
        blocks = [
            rng.normal(loc=c, scale=4.0, size=(n_per_cluster, 3)) for c in centers
        ]
        X = np.clip(np.vstack(blocks), 0.0, 255.0).astype(np.float32)
        return X

    def test_kmeans_scratch_shape_labels_and_convergence(self) -> None:
        """Kiểm tra KMeansFromScratch trả về đúng shape, nhãn hợp lệ và WCSS không tăng."""
        X = self._build_synthetic_clusters(120)
        model = KMeansFromScratch(n_clusters=4, max_iter=50, tol=1e-4, random_state=42)
        labels = model.fit_predict(X)

        self.assertIsNotNone(model.centroids)
        assert model.centroids is not None
        self.assertEqual(model.centroids.shape, (4, 3))
        self.assertEqual(labels.shape, (480,))
        self.assertEqual(labels.dtype, np.int64)
        self.assertGreaterEqual(int(labels.min()), 0)
        self.assertLess(int(labels.max()), 4)
        self.assertEqual(model.stop_reason_, "converged")

        history_df = model.get_convergence_history()
        self.assertEqual(
            list(history_df.columns),
            ["iteration", "inertia", "centroid_shift", "label_change_ratio"],
        )
        inertias = history_df["inertia"].to_numpy()
        # WCSS phải không tăng qua các vòng lặp (cho phép sai số dấu phẩy động rất nhỏ)
        self.assertTrue(np.all(np.diff(inertias) <= 1e-3))

        # Kiểm tra hàm predict trên điểm dữ liệu mới
        preds = model.predict(X[:10])
        np.testing.assert_array_equal(preds, labels[:10])

    def test_kmeans_scratch_edge_cases_and_validation(self) -> None:
        """Kiểm tra biên: ảnh 1 màu (K=1), giới hạn max_iter, cụm rỗng và K không hợp lệ."""
        # Ảnh 1 màu thuần với K=1 -> WCSS phải bằng 0
        solid_X = np.full((64, 3), 150.0, dtype=np.float32)
        km_single = KMeansFromScratch(n_clusters=1, max_iter=10, random_state=42).fit(solid_X)
        self.assertAlmostEqual(float(km_single.inertia_), 0.0, places=5)
        self.assertTrue(np.allclose(km_single.centroids[0], [150.0, 150.0, 150.0]))

        # Kiểm tra dừng theo max_iter khi giới hạn 1 vòng và ngưỡng tol=0 trên dữ liệu phức tạp
        rng = np.random.default_rng(7)
        noisy_X = rng.uniform(0, 255, size=(300, 3)).astype(np.float32)
        km_capped = KMeansFromScratch(n_clusters=5, max_iter=1, tol=0.0, random_state=42).fit(noisy_X)
        self.assertEqual(km_capped.n_iter_, 1)
        self.assertEqual(km_capped.stop_reason_, "max_iter")

        # Kiểm tra tái khởi tạo cụm rỗng trong _update_centroids
        km_empty = KMeansFromScratch(n_clusters=3, random_state=42)
        forced_labels = np.array([0, 0, 1, 1], dtype=np.int64)
        sample_X = np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [100.0, 100.0, 100.0], [200.0, 200.0, 200.0]],
            dtype=np.float64,
        )
        updated = km_empty._update_centroids(sample_X, forced_labels, k=3)
        self.assertEqual(updated.shape, (3, 3))
        self.assertTrue(np.isfinite(updated).all())
        self.assertEqual(km_empty.empty_cluster_events_, 1)

        # Kiểm tra K vượt quá số màu phân biệt hoặc chứa NaN
        with self.assertRaises(ValueError):
            KMeansFromScratch(n_clusters=2).fit(solid_X)
        bad_X = noisy_X.copy()
        bad_X[0, 0] = np.nan
        with self.assertRaises(ValueError):
            KMeansFromScratch(n_clusters=3).fit(bad_X)

    def test_sklearn_elbow_and_final_model(self) -> None:
        """Kiểm tra SklearnKMeansTrainer tìm đúng điểm gập hình học và huấn luyện mô hình cuối."""
        k_vals = [2, 3, 4, 5, 6, 7, 8]
        inertias = [1000.0, 600.0, 250.0, 110.0, 95.0, 85.0, 78.0]
        trainer = SklearnKMeansTrainer(random_state=42)
        knee = trainer.find_knee_point(k_vals, inertias)
        self.assertIn(knee, [4, 5])

        X = self._build_synthetic_clusters(80)
        df_elbow, optimal_k = trainer.search_elbow(X, k_range=range(2, 7))
        self.assertIn("K", df_elbow.columns)
        self.assertIn("Inertia", df_elbow.columns)
        self.assertIn(optimal_k, range(2, 7))

        final_model = trainer.fit_final_model(X, optimal_k=4)
        self.assertEqual(final_model.cluster_centers_.shape, (4, 3))

    def test_scratch_final_labels_and_wcss_use_stored_centers(self) -> None:
        """Nhãn và WCSS cuối phải khớp tâm lưu trữ, kể cả pixel nằm sát ranh giới cụm."""
        values = np.array(
            [17, 15, 16, 10, 16, 6, 9, 15, 2, 6, 2, 9, 19, 2, 7, 8, 18, 4, 10, 5],
            dtype=np.float32,
        )
        X = np.repeat(values[:, None], 3, axis=1)
        model = KMeansFromScratch(n_clusters=3, max_iter=1, random_state=42).fit(X)
        np.testing.assert_array_equal(model.labels_, model.predict(X))
        differences = X.astype(np.float64) - model.centroids.astype(np.float64)[model.labels_]
        self.assertAlmostEqual(model.inertia_, float(np.sum(differences ** 2)), places=10)

    def test_minibatch_accepts_auto_initializations(self) -> None:
        """Benchmark phải nhận được baseline có n_init='auto' của Scikit-Learn."""
        X = self._build_synthetic_clusters(10)
        baseline = KMeans(n_clusters=4, n_init="auto", random_state=42).fit(X)
        report = MiniBatchKMeansTrainer().compare_batch_sizes(
            X, k=4, full_kmeans_model=baseline, batch_sizes=[16]
        )
        self.assertEqual(len(report), 2)
        self.assertTrue(np.isfinite(report["inertia"]).all())

    def test_opencv_single_rgb_pixel_keeps_feature_dimensions(self) -> None:
        """Ảnh một pixel phải tạo một nhãn và một tâm RGB, không tách kênh thành mẫu."""
        X = np.array([[120.0, 60.0, 30.0]], dtype=np.float32)
        compactness, labels, centers = OpenCVKMeansTrainer().train(X, k=1)
        self.assertEqual(labels.shape, (1,))
        np.testing.assert_array_equal(centers, X)
        self.assertEqual(compactness, 0.0)

    def test_minibatch_reports_requested_and_effective_batch_sizes(self) -> None:
        """Bảng benchmark phải phân biệt batch yêu cầu với batch thực dùng trên ảnh nhỏ."""
        X = self._build_synthetic_clusters(10)
        baseline = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X)
        report = MiniBatchKMeansTrainer().compare_batch_sizes(
            X, k=4, full_kmeans_model=baseline, batch_sizes=[16, 64, 128]
        )
        minibatch_rows = report.iloc[1:]
        self.assertIn("effective_batch_size", report.columns)
        self.assertEqual(minibatch_rows["batch_size"].tolist(), [16, 64, 128])
        self.assertEqual(minibatch_rows["effective_batch_size"].tolist(), [16, 40, 40])
        self.assertIn("b=40", minibatch_rows.iloc[1]["model"])

    def test_opencv_cross_validation_and_minibatch(self) -> None:
        """Kiểm tra OpenCVKMeansTrainer (kèm căn chỉnh hoán vị nhãn) và MiniBatchKMeansTrainer."""
        X = self._build_synthetic_clusters(150)
        sk_trainer = SklearnKMeansTrainer(random_state=42)
        sk_model = sk_trainer.fit_final_model(X, optimal_k=4)

        cv_trainer = OpenCVKMeansTrainer()
        compactness, cv_labels, cv_centers = cv_trainer.train(X, k=4, attempts=5)
        self.assertEqual(cv_centers.shape, (4, 3))
        self.assertEqual(cv_labels.shape, (X.shape[0],))
        self.assertGreaterEqual(compactness, 0.0)
        self.assertIsNotNone(cv_trainer.last_fit_time_sec_)
        self.assertGreater(cv_trainer.last_fit_time_sec_, 0.0)

        comparison = cv_trainer.compare_with_sklearn(
            sk_model.labels_,
            cv_labels,
            float(sk_model.inertia_),
            compactness,
            sklearn_time_sec=0.05,
        )
        self.assertGreater(comparison["label_agreement_pct"], 95.0)
        self.assertGreater(comparison["ari"], 0.95)
        self.assertIn("opencv_fit_time_sec", comparison)
        self.assertIn("sklearn_fit_time_sec", comparison)
        self.assertIn("opencv_vs_sklearn_speedup", comparison)

        # Kiểm tra bất biến hoán vị: nhãn đảo thứ tự hoàn toàn vẫn đạt 100% trùng khớp sau ghép
        permuted_labels = (sk_model.labels_ + 2) % 4
        perm_report = cv_trainer.compare_with_sklearn(
            sk_model.labels_, permuted_labels, 100.0, 100.0
        )
        self.assertEqual(perm_report["label_agreement_pct"], 100.0)

        mb_trainer = MiniBatchKMeansTrainer(random_state=42)
        df_mb = mb_trainer.compare_batch_sizes(X, k=4, full_kmeans_model=sk_model, batch_sizes=[64, 128])
        self.assertEqual(len(df_mb), 3)
        self.assertTrue((df_mb["ari"] <= 1.0001).all())

    def test_model_persistence_roundtrip(self) -> None:
        """Kiểm tra lưu và nạp lại mô hình qua joblib giữ nguyên tọa độ tâm cụm và dự đoán."""
        X = self._build_synthetic_clusters(60)
        model = SklearnKMeansTrainer(random_state=42).fit_final_model(X, optimal_k=4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "nested" / "kmeans_test.joblib"
            save_model_pipeline(model, model_path, metadata={"optimal_k": 4, "color_space": "RGB"})
            loaded_model, loaded_meta = load_model_pipeline(model_path)

            self.assertEqual(loaded_meta["optimal_k"], 4)
            np.testing.assert_array_almost_equal(model.cluster_centers_, loaded_model.cluster_centers_)
            np.testing.assert_array_equal(model.predict(X[:20]), loaded_model.predict(X[:20]))

    def test_phase2_docstrings_are_vietnamese(self) -> None:
        """Kiểm tra hồi quy: toàn bộ docstring của Phase 2 phải được viết bằng Tiếng Việt."""
        target_files = [
            Path("src/models/kmeans_scratch.py"),
            Path("src/models/kmeans_sklearn.py"),
            Path("src/models/kmeans_opencv.py"),
            Path("src/models/minibatch_kmeans.py"),
            Path("src/models/persistence.py"),
            Path("tests/test_models.py"),
        ]
        for file_path in target_files:
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

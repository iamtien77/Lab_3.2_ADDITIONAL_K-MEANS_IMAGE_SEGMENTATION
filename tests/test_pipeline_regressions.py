"""Kiểm tra các trường hợp biên từng làm sai hoặc dừng luồng bài lab."""

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np

import main as pipeline
from src.evaluation.metrics import compute_silhouette
from src.models.persistence import load_model_pipeline


class TestPipelineRegressions(unittest.TestCase):
    """Kiểm tra lỗi tại nơi các bước thật sự trao đổi dữ liệu."""

    def test_silhouette_nearly_full_sample(self):
        """Lấy gần hết dữ liệu không đòi phần bỏ lại phải đủ mọi cụm."""
        x = np.repeat([[0., 0., 0.], [255., 255., 255.]], [3, 4], axis=0)
        score = compute_silhouette(x, np.array([0] * 3 + [1] * 4), max_samples=6)
        self.assertAlmostEqual(score, 1.0)

    def test_silhouette_rare_cluster_is_retained(self):
        """Cụm chỉ có một điểm vẫn được giữ trong mẫu đánh giá."""
        x = np.zeros((100, 3), dtype=np.float32)
        x[-1] = 255
        labels = np.array([0] * 99 + [1])
        score = compute_silhouette(x, labels, max_samples=10)
        self.assertAlmostEqual(score, 0.9, places=5)

    def test_silhouette_insufficient_budget_and_single_pixel(self):
        """Trường hợp không định nghĩa được điểm số dùng giá trị quy ước an toàn."""
        x = np.repeat([[0., 0., 0.], [255., 255., 255.]], 3, axis=0)
        self.assertEqual(compute_silhouette(x, [0, 0, 0, 1, 1, 1], max_samples=2), 0.0)
        self.assertEqual(compute_silhouette(x[:1], [0]), 0.0)

    def test_preprocessing_statistics_stay_rgb(self):
        """Thống kê độ sáng và tên kênh không thay đổi theo hệ màu phân cụm."""
        image = np.full((4, 4, 3), [255, 0, 0], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.multiple(pipeline, METRICS_DIR=Path(tmp), PROCESSED_DATA_DIR=Path(tmp)):
                results = []
                for space in ("RGB", "HSV", "LAB"):
                    pipeline.step_2_preprocess_and_flatten(image, color_space=space)
                    results.append(json.loads((Path(tmp) / "data_quality_stats.json").read_text(encoding="utf-8")))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], results[2])
        self.assertAlmostEqual(results[0]["mean_brightness"], 76.24)

    def test_monochrome_pipeline_fixed_and_automatic_k(self):
        """Ảnh một màu đi hết pipeline với K=1 và cả chế độ tự chọn."""
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            source = root / "solid.png"
            Image.fromarray(np.full((4, 4, 3), [120, 60, 30], dtype=np.uint8)).save(source)
            for name in ("FIGURES_DIR", "METRICS_DIR", "PROCESSED_DATA_DIR", "SAVED_MODELS_DIR", "RAW_DATA_DIR"):
                folder = root / name.lower()
                folder.mkdir()
                stack.enter_context(mock.patch.object(pipeline, name, folder))
            stack.enter_context(mock.patch.object(pipeline, "DEFAULT_IMAGE_PATH", source))
            for name in dir(pipeline):
                if name.startswith("plot_"):
                    stack.enter_context(mock.patch.object(pipeline, name))
            stack.enter_context(mock.patch("src.deployment.predictor.plot_side_by_side"))
            for extra in (["--k", "1"], [], ["--k", "1", "--target-size", "1"]):
                pipeline.main(["--image", str(source), "--skip-color-space-benchmark", *extra])
                model, metadata = load_model_pipeline(pipeline.SAVED_MODELS_DIR / "kmeans_segmentation.joblib")
                self.assertEqual(model.n_clusters, 1)
                self.assertAlmostEqual(model.inertia_, 0.0)
                self.assertEqual(metadata["optimal_k"], 1)
                stats = json.loads((pipeline.METRICS_DIR / "cluster_stats.json").read_text(encoding="utf-8"))
                self.assertEqual(stats[0]["pixel_count"], 16)
                self.assertEqual(stats[0]["rgb_uint8"], [120, 60, 30])

    def test_preview_ks_respect_available_colors(self):
        """Ảnh hai màu không bị ép chạy các K xem trước lớn hơn dữ liệu."""
        x = np.repeat([[0., 0., 0.], [255., 255., 255.]], 4, axis=0).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(pipeline, "METRICS_DIR", Path(tmp)), \
                 mock.patch.object(pipeline, "plot_elbow_curve"), \
                 mock.patch.object(pipeline, "plot_k_evolution_grid") as grid:
                model, k, _ = pipeline.step_4_elbow_method_optimal_k(x, resized_shape=(2, 4), fixed_k=2)
                self.assertEqual(k, 2)
                self.assertEqual(set(grid.call_args.args[0]), {2})
                np.testing.assert_array_equal(model.predict(x), model.labels_)
                with self.assertRaisesRegex(ValueError, "K must lie"):
                    pipeline.step_4_elbow_method_optimal_k(x, fixed_k=3)

    def test_pandas_is_declared(self):
        """Dependency được import trực tiếp phải có trong hướng dẫn cài đặt."""
        requirements = Path("requirements.txt").read_text(encoding="utf-8").splitlines()
        self.assertTrue(any(line.lower().startswith("pandas") for line in requirements))

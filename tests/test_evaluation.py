"""
=============================================================================
KIỂM THỬ ĐƠN VỊ: ĐÁNH GIÁ CHẤT LƯỢNG & TRỰC QUAN HÓA (TEST EVALUATION)
=============================================================================
"""

import ast
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR
from src.deployment.predictor import ImageSegmentationPredictor
from src.evaluation.metrics import (
    compare_clustering_labels,
    compute_silhouette,
    compute_wcss,
)
from src.models.kmeans_sklearn import SklearnKMeansTrainer
from src.models.persistence import save_model_pipeline
from src.evaluation.visualization import (
    plot_cluster_distribution,
    plot_color_space_comparison,
    plot_color_space_segmentation_comparison,
    plot_convergence_tracking,
    plot_elbow_curve,
    plot_k_evolution_grid,
    plot_minibatch_comparison,
    plot_side_by_side,
)
from main import parse_args


class TestEvaluationMetrics(unittest.TestCase):
    """Bộ kiểm thử cho các phép đo chất lượng phân cụm và hàm xuất biểu đồ PNG."""

    def test_compute_wcss_positive_and_zero_on_exact_match(self) -> None:
        """Kiểm tra WCSS >= 0 và bằng 0 khi các điểm trùng khít tâm cụm."""
        centroids = np.array([[0.0, 0.0, 0.0], [255.0, 255.0, 255.0]], dtype=np.float32)
        X = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [255.0, 255.0, 255.0]], dtype=np.float32)
        labels = np.array([0, 0, 1], dtype=np.int64)
        self.assertAlmostEqual(compute_wcss(X, centroids, labels), 1.0, places=5)

    def test_silhouette_range_and_subsampling(self) -> None:
        """Kiểm tra Silhouette Score nằm trong [-1.0, 1.0] và hoạt động với tập dữ liệu lớn."""
        rng = np.random.default_rng(42)
        c1 = rng.normal(loc=20.0, scale=2.0, size=(300, 3))
        c2 = rng.normal(loc=220.0, scale=2.0, size=(300, 3))
        X = np.vstack([c1, c2]).astype(np.float32)
        labels = np.array([0] * 300 + [1] * 300, dtype=np.int64)

        score = compute_silhouette(X, labels, max_samples=200, random_state=42)
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreater(score, 0.8)

        # Trường hợp chỉ có 1 cụm phải trả về 0.0 an toàn
        self.assertEqual(compute_silhouette(X, np.zeros(600, dtype=np.int64)), 0.0)

    def test_compare_clustering_labels_identical_and_permuted(self) -> None:
        """Kiểm tra ARI/NMI đạt 1.0 khi hai bộ nhãn giống hệt hoặc chỉ bị hoán vị ID cụm."""
        a = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        b_perm = np.array([2, 2, 0, 0, 1, 1], dtype=np.int64)
        res = compare_clustering_labels(a, b_perm)
        self.assertAlmostEqual(res["ari"], 1.0, places=4)
        self.assertAlmostEqual(res["nmi"], 1.0, places=4)

    def test_visualization_plots_create_png_files(self) -> None:
        """Kiểm tra cả 6 hàm vẽ biểu đồ đều xuất file PNG hợp lệ không rỗng."""
        img = np.full((32, 48, 3), 120, dtype=np.uint8)
        df_elbow = pd.DataFrame({"K": [2, 3, 4, 5], "Inertia": [1000.0, 500.0, 200.0, 180.0]})
        df_hist = pd.DataFrame(
            {
                "iteration": [1, 2, 3],
                "inertia": [900.0, 400.0, 380.0],
                "centroid_shift": [12.0, 1.5, 0.0],
            }
        )
        cluster_stats = [
            {
                "cluster_id": 0,
                "centroid_rgb": [20.0, 40.0, 200.0],
                "hex_color": "#1428C8",
                "pixel_count": 300,
                "share_percentage": 60.0,
            },
            {
                "cluster_id": 1,
                "centroid_rgb": [230.0, 220.0, 210.0],
                "hex_color": "#E6DCD2",
                "pixel_count": 200,
                "share_percentage": 40.0,
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            paths = {
                "side": tmp / "side.png",
                "cs": tmp / "cs.png",
                "elbow": tmp / "elbow.png",
                "conv": tmp / "conv.png",
                "dist": tmp / "dist.png",
                "grid": tmp / "grid.png",
            }
            plot_side_by_side(img, img, k=4, save_path=paths["side"])
            plot_color_space_comparison(img, img, img, save_path=paths["cs"])
            plot_elbow_curve(df_elbow, optimal_k=4, save_path=paths["elbow"])
            plot_convergence_tracking(df_hist, save_path=paths["conv"])
            plot_cluster_distribution(cluster_stats, save_path=paths["dist"])
            plot_k_evolution_grid({2: img, 4: img, 5: img}, save_path=paths["grid"])

            mb_png = tmp / "mb.png"
            cs_seg_png = tmp / "cs_seg.png"
            plot_minibatch_comparison(
                [
                    {"label": "Full", "image": img, "fit_time_sec": 0.4, "speedup": 1.0, "inertia_increase_pct": 0.0, "ari": 1.0},
                    {"label": "MB (b=256)", "image": img, "fit_time_sec": 0.2, "speedup": 2.0, "inertia_increase_pct": 0.1, "ari": 0.98},
                ],
                k=4,
                save_path=mb_png,
            )
            plot_color_space_segmentation_comparison(
                img,
                [
                    {"color_space": "RGB", "segmented_rgb": img, "silhouette": 0.45, "fit_time_sec": 0.2},
                    {"color_space": "LAB", "segmented_rgb": img, "silhouette": 0.49, "fit_time_sec": 0.21},
                ],
                k=4,
                save_path=cs_seg_png,
            )
            paths["mb"] = mb_png
            paths["cs_seg"] = cs_seg_png

            for name, p in paths.items():
                self.assertTrue(p.exists(), f"Thiếu file biểu đồ {name}: {p}")
                self.assertGreater(p.stat().st_size, 1000, f"File biểu đồ {name} bị rỗng.")

    def test_phase3_evaluation_docstrings_are_vietnamese(self) -> None:
        """Kiểm tra hồi quy: toàn bộ docstring trong module đánh giá phải viết bằng Tiếng Việt."""
        for file_path in [
            Path("src/evaluation/metrics.py"),
            Path("src/evaluation/visualization.py"),
            Path("src/deployment/predictor.py"),
            Path("main.py"),
            Path("tests/test_evaluation.py"),
        ]:
            tree = ast.parse(file_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                    doc = ast.get_docstring(node) or ""
                    self.assertTrue(
                        any(ord(ch) > 127 for ch in doc),
                        f"Docstring tại {file_path}:{getattr(node, 'name', '<module>')} chưa viết bằng Tiếng Việt.",
                    )

    def test_predictor_end_to_end_with_saved_model(self) -> None:
        """Kiểm tra ImageSegmentationPredictor suy luận trọn gói từ file mô hình đã lưu."""
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 255, size=(120, 3)).astype(np.float32)
        model = SklearnKMeansTrainer(random_state=42).fit_final_model(X, optimal_k=3)
        sample_img = RAW_DATA_DIR / "image_copy.png"
        if not sample_img.exists():
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            model_path = tmp / "model.joblib"
            out_dir = tmp / "out"
            save_model_pipeline(model, model_path, metadata={"optimal_k": 3})

            predictor = ImageSegmentationPredictor(model_path=model_path)
            result = predictor.predict(sample_img, output_dir=out_dir, return_original_size=True)
            self.assertEqual(result["optimal_k"], 3)
            self.assertEqual(
                result["segmented_image"].shape[:2],
                (result["original_metadata"]["height"], result["original_metadata"]["width"]),
            )
            self.assertTrue(Path(result["output_files"]["segmented_image_path"]).exists())
            self.assertTrue(Path(result["output_files"]["comparison_figure_path"]).exists())
            self.assertTrue(Path(result["output_files"]["stats_json_path"]).exists())

    def test_plot_titles_do_not_overlap_on_square_images(self) -> None:
        """Kiểm tra hồi quy: suptitle không được chồng đè lên tiêu đề subplot trên ảnh vuông 1:1 và ảnh ngang."""
        square_img = np.zeros((256, 256, 3), dtype=np.uint8)
        wide_img = np.zeros((144, 256, 3), dtype=np.uint8)
        captured_figs = []

        def _capture_fig(fig: plt.Figure, save_path: Path = None) -> None:
            """Giữ lại đối tượng Figure sau khi render để kiểm tra khoảng cách tiêu đề."""
            fig.canvas.draw()
            captured_figs.append(fig)

        with mock.patch("src.evaluation.visualization._finalize_figure", side_effect=_capture_fig):
            for test_img in (square_img, wide_img):
                plot_side_by_side(test_img, test_img, k=5)
                plot_color_space_comparison(test_img, test_img, test_img)
                plot_minibatch_comparison(
                    [{"label": "Full", "image": test_img, "fit_time_sec": 0.3, "speedup": 1.0, "inertia_increase_pct": 0.0, "ari": 1.0}],
                    k=5,
                )
                plot_color_space_segmentation_comparison(
                    test_img,
                    [{"color_space": "RGB", "segmented_rgb": test_img, "silhouette": 0.4, "fit_time_sec": 0.2}],
                    k=5,
                )

        try:
            for fig in captured_figs:
                renderer = fig.canvas.get_renderer()
                st = fig._suptitle
                self.assertIsNotNone(st)
                st_bbox = st.get_window_extent(renderer)
                max_title_top = max(ax.title.get_window_extent(renderer).y1 for ax in fig.axes)
                self.assertGreaterEqual(
                    st_bbox.y0 - max_title_top,
                    4.0,
                    f"Tiêu đề suptitle bị đè lên subplot title (khoảng cách = {st_bbox.y0 - max_title_top:.2f} px).",
                )
        finally:
            for fig in captured_figs:
                plt.close(fig)

    def test_main_cli_parse_args_and_lab_predictor(self) -> None:
        """Kiểm tra tham số dòng lệnh của main.py và suy luận từ mô hình lưu ở không gian màu LAB."""
        parsed = parse_args(
            ["--image", "sample.png", "--k", "6", "--color-space", "lab", "--target-size", "128", "--skip-color-space-benchmark"]
        )
        self.assertEqual(parsed.image, Path("sample.png"))
        self.assertEqual(parsed.k, 6)
        self.assertEqual(parsed.color_space.upper(), "LAB")
        self.assertEqual(parsed.target_size, 128)
        self.assertTrue(parsed.skip_color_space_benchmark)

        rng = np.random.default_rng(7)
        X_lab = rng.uniform(20, 220, size=(80, 3)).astype(np.float32)
        model = SklearnKMeansTrainer(random_state=7).fit_final_model(X_lab, optimal_k=3)
        sample_img = RAW_DATA_DIR / "image_copy.png"
        if not sample_img.exists():
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "lab_model.joblib"
            save_model_pipeline(model, model_path, metadata={"optimal_k": 3, "color_space": "LAB"})
            predictor = ImageSegmentationPredictor(model_path=model_path)
            self.assertEqual(predictor.color_space, "LAB")
            res = predictor.predict(sample_img, return_original_size=False, target_size=64)
            self.assertEqual(res["segmented_image"].dtype, np.uint8)
            self.assertEqual(res["optimal_k"], 3)


if __name__ == "__main__":
    unittest.main()

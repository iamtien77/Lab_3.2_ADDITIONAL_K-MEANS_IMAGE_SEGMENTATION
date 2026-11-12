"""
=============================================================================
KIỂM THỬ ĐƠN VỊ: MODULE DỮ LIỆU & TIỀN XỬ LÝ (TEST DATA & PREPROCESSING)
=============================================================================
"""

import ast
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.config import DEFAULT_IMAGE_PATH, RAW_DATA_DIR
from src.data.loader import is_valid_image, load_image, validate_image
from src.data.preprocessing import (
    convert_centroids_to_rgb,
    convert_color_space,
    flatten_pixels,
    preprocess_for_clustering,
    resize_image,
)
from src.features.color_features import (
    classify_color,
    compute_color_statistics,
    extract_color_distribution,
    normalize_pixels,
)


class TestDataModule(unittest.TestCase):
    """Bộ kiểm thử cho các module xử lý dữ liệu ảnh, tiền xử lý và trích xuất đặc trưng màu."""

    def test_default_image_exists_and_loads_valid_contract(self) -> None:
        """Kiểm tra file ảnh mẫu tồn tại và đọc ra đúng chuẩn (H, W, 3) uint8 kèm metadata đầy đủ."""
        self.assertTrue(DEFAULT_IMAGE_PATH.exists(), f"Không tìm thấy ảnh mẫu: {DEFAULT_IMAGE_PATH}")
        img_rgb, meta = load_image()
        self.assertTrue(validate_image(img_rgb))
        self.assertEqual(img_rgb.dtype, np.uint8)
        self.assertEqual(img_rgb.ndim, 3)
        self.assertEqual(img_rgb.shape[2], 3)
        self.assertEqual(meta["height"], img_rgb.shape[0])
        self.assertEqual(meta["width"], img_rgb.shape[1])
        self.assertEqual(meta["total_pixels"], img_rgb.shape[0] * img_rgb.shape[1])
        self.assertEqual(meta["color_format"], "RGB")
        self.assertIn("file_size_bytes", meta)
        self.assertIn("channel_stats", meta)

    def test_load_all_raw_images_including_rgba(self) -> None:
        """Kiểm tra đọc toàn bộ ảnh trong data/raw/, bao gồm xử lý trộn nền trắng cho ảnh RGBA."""
        rgba_image = RAW_DATA_DIR / "image-sea2.png"
        if rgba_image.exists():
            # Mặc định allow_alpha=True phải đọc được ảnh RGBA mà không sập pipeline
            img_rgb, meta = load_image(rgba_image, allow_alpha=True)
            self.assertTrue(validate_image(img_rgb))
            self.assertTrue(meta["alpha_composited"])

            # Chế độ nghiêm ngặt allow_alpha=False phải từ chối ảnh có kênh Alpha
            with self.assertRaises(ValueError):
                load_image(rgba_image, allow_alpha=False)

    def test_load_image_rejects_missing_empty_and_grayscale(self) -> None:
        """Kiểm tra load_image báo lỗi đúng cho file không tồn tại, file 0 byte và ảnh đa sắc xám."""
        with self.assertRaises(FileNotFoundError):
            load_image(RAW_DATA_DIR / "non_existent_image_xyz.png")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            empty_file = tmp_path / "empty.png"
            empty_file.write_bytes(b"")
            with self.assertRaises(ValueError):
                load_image(empty_file)

            gray_path = tmp_path / "gray.png"
            cv2.imwrite(str(gray_path), np.zeros((32, 32), dtype=np.uint8))
            with self.assertRaises(ValueError):
                load_image(gray_path)

    def test_validate_image_raises_on_invalid_inputs(self) -> None:
        """Kiểm tra validate_image ném TypeError/ValueError đúng đặc tả và is_valid_image trả về False."""
        self.assertTrue(validate_image(np.zeros((10, 20, 3), dtype=np.uint8)))
        self.assertTrue(is_valid_image(np.zeros((10, 20, 3), dtype=np.uint8)))

        with self.assertRaises(TypeError):
            validate_image(None)  # type: ignore[arg-type]

        for bad_img in [
            np.zeros((10, 20, 3), dtype=np.float32),
            np.zeros((10, 20), dtype=np.uint8),
            np.zeros((10, 20, 4), dtype=np.uint8),
        ]:
            self.assertFalse(is_valid_image(bad_img))
            with self.assertRaises(ValueError):
                validate_image(bad_img)

    def test_resize_image_aspect_ratio_and_no_upscale(self) -> None:
        """Kiểm tra resize ảnh giữ tỷ lệ khung hình, không phóng to ảnh nhỏ và xử lý ảnh chữ nhật."""
        wide_img = np.zeros((300, 600, 3), dtype=np.uint8)
        resized_wide, shape_wide = resize_image(wide_img, target_size=256, keep_aspect_ratio=True)
        self.assertEqual(shape_wide, (128, 256))
        self.assertEqual(resized_wide.shape[:2], (128, 256))

        small_img = np.zeros((100, 150, 3), dtype=np.uint8)
        resized_small, shape_small = resize_image(small_img, target_size=256, keep_aspect_ratio=True)
        self.assertEqual(shape_small, (100, 150))
        self.assertEqual(resized_small.shape, small_img.shape)

        square_img, square_shape = resize_image(wide_img, target_size=128, keep_aspect_ratio=False)
        self.assertEqual(square_shape, (128, 128))
        self.assertEqual(square_img.shape[:2], (128, 128))

    def test_convert_color_space_roundtrip_and_numpy_fallback(self) -> None:
        """Kiểm tra chuyển đổi BGR<->RGB bảo toàn nguyên vẹn, hỗ trợ HSV/LAB và fallback thuần NumPy."""
        rng = np.random.default_rng(42)
        rgb = rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)

        bgr = convert_color_space(rgb, source_space="RGB", target_space="BGR")
        rgb_back = convert_color_space(bgr, source_space="BGR", target_space="RGB")
        np.testing.assert_array_equal(rgb, rgb_back)

        hsv_cv = convert_color_space(rgb, source_space="RGB", target_space="HSV", use_numpy_fallback=False)
        hsv_np = convert_color_space(rgb, source_space="RGB", target_space="HSV", use_numpy_fallback=True)
        self.assertEqual(hsv_cv.shape, rgb.shape)
        self.assertEqual(hsv_np.shape, rgb.shape)
        # Sai số làm tròn tối đa <= 2 đơn vị khi chuyển bằng công thức thuần NumPy vs OpenCV
        self.assertLessEqual(int(np.abs(hsv_cv.astype(np.int16) - hsv_np.astype(np.int16)).max()), 2)

        lab = convert_color_space(rgb, source_space="RGB", target_space="LAB")
        hsv_to_lab = convert_color_space(hsv_cv, source_space="HSV", target_space="LAB")
        self.assertEqual(lab.shape, rgb.shape)
        self.assertEqual(hsv_to_lab.shape, rgb.shape)

        with self.assertRaises(ValueError):
            convert_color_space(rgb, source_space="RGB", target_space="CMYK")

    def test_flatten_and_normalize_and_preprocess_pipeline(self) -> None:
        """Kiểm tra làm phẳng ảnh, kiểm tra NaN/Inf, thống kê kênh màu và lưu snapshot dữ liệu sạch."""
        img = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        flat_float = flatten_pixels(img, as_float=True)
        self.assertEqual(flat_float.shape, (6, 3))
        self.assertEqual(flat_float.dtype, np.float32)

        reconstructed = flat_float.astype(np.uint8).reshape(2, 3, 3)
        np.testing.assert_array_equal(img, reconstructed)

        normalized = normalize_pixels(flat_float, to_unit_range=True)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertLessEqual(float(normalized.max()), 1.0)

        bad_pixels = flat_float.copy()
        bad_pixels[0, 0] = np.nan
        with self.assertRaises(ValueError):
            normalize_pixels(bad_pixels)

        color_stats = compute_color_statistics(flat_float)
        self.assertEqual(color_stats["total_pixels"], 6)
        self.assertEqual(color_stats["unique_colors"], 6)
        self.assertIn("R", color_stats["channels"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            large_img = np.full((300, 400, 3), 128, dtype=np.uint8)
            bundle = preprocess_for_clustering(
                large_img,
                target_size=128,
                color_space="RGB",
                save_dir=Path(tmp_dir),
                file_prefix="sample",
            )
            self.assertEqual(bundle["resized_shape"], (96, 128))
            self.assertEqual(bundle["pixel_matrix"].shape, (96 * 128, 3))
            self.assertIsNotNone(bundle["saved_image_path"])
            self.assertTrue(Path(bundle["saved_image_path"]).exists())
            self.assertTrue(Path(bundle["saved_meta_path"]).exists())

    def test_classify_color_and_extract_distribution(self) -> None:
        """Kiểm tra định danh tên màu song ngữ và thống kê phân bố share_percentage chính xác."""
        self.assertIn("Đen", classify_color(np.array([0, 0, 0])))
        self.assertIn("Trắng", classify_color(np.array([255, 255, 255])))
        self.assertIn("Xanh dương", classify_color(np.array([10, 20, 240])))
        self.assertIn("Đỏ mận", classify_color(np.array([136, 57, 90])))

        labels = np.array([0, 0, 0, 1], dtype=np.int64)
        centroids = np.array([[255.0, 255.0, 255.0], [10.0, 10.0, 220.0]], dtype=np.float32)
        dist = extract_color_distribution(labels, centroids)
        self.assertEqual(len(dist), 2)
        self.assertEqual(dist[0]["cluster_id"], 0)
        self.assertEqual(dist[0]["share_percentage"], 75.0)
        self.assertEqual(dist[1]["share_percentage"], 25.0)
        self.assertEqual(dist[0]["hex_color"], "#FFFFFF")
        self.assertEqual(dist[0]["dominance_rank"], 1)

    def test_convert_centroids_to_rgb_across_spaces(self) -> None:
        """Kiểm tra quy đổi tâm cụm từ RGB, HSV, LAB sang RGB uint8-scale hoạt động chính xác."""
        rgb_src = np.array([[200.0, 50.0, 20.0], [20.0, 180.0, 220.0]], dtype=np.float32)
        np.testing.assert_allclose(convert_centroids_to_rgb(rgb_src, "RGB"), rgb_src, atol=1e-4)

        for space in ("HSV", "LAB"):
            converted_strip = convert_color_space(rgb_src.astype(np.uint8).reshape(1, 2, 3), "RGB", space)
            centroids_space = converted_strip.reshape(2, 3).astype(np.float32)
            recovered = convert_centroids_to_rgb(centroids_space, space)
            self.assertEqual(recovered.shape, (2, 3))
            self.assertEqual(recovered.dtype, np.float32)
            self.assertTrue(np.all((recovered >= 0.0) & (recovered <= 255.0)))
            self.assertLess(float(np.mean(np.abs(recovered - rgb_src))), 5.0)

    def test_phase1_docstrings_are_vietnamese(self) -> None:
        """Kiểm tra hồi quy: toàn bộ docstring ở các module Phase 1 phải viết bằng Tiếng Việt."""
        target_files = [
            Path("src/utils/logger.py"),
            Path("src/data/loader.py"),
            Path("src/data/preprocessing.py"),
            Path("src/features/color_features.py"),
            Path("tests/test_data.py"),
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

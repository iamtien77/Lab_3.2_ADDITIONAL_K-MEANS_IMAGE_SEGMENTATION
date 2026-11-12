"""
=============================================================================
MODULE: TRIỂN KHAI VÀ SUY LUẬN PHÂN ĐOẠN ẢNH (PREDICTOR / INFERENCE)
=============================================================================
Mục đích:
- Cung cấp pipeline suy luận trọn gói (End-to-End Inference) cho một bức ảnh mới bất kỳ.
- Tự động: nạp mô hình đã lưu -> đọc ảnh -> tiền xử lý/kiểm định -> gán cụm màu
  theo lô tối ưu bộ nhớ -> tái tạo ảnh phân đoạn -> thống kê phân bố và xuất file kết quả.
=============================================================================
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np

from src.config import SAVED_MODELS_DIR, TARGET_SIZE
from src.data.loader import load_image
from src.data.preprocessing import (
    convert_centroids_to_rgb,
    convert_color_space,
    flatten_pixels,
    resize_image,
)
from src.evaluation.visualization import plot_side_by_side
from src.features.color_features import extract_color_distribution
from src.models.persistence import load_model_pipeline
from src.segmentation.segmenter import ImageSegmenter
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ImageSegmentationPredictor:
    """
    Lớp điều phối suy luận phân đoạn ảnh trên ảnh đầu vào mới từ mô hình K-Means đã lưu.
    """

    def __init__(self, model_path: Optional[Path] = None):
        """
        Nạp mô hình K-Means và metadata từ đĩa, đồng thời khởi tạo ImageSegmenter.

        Tham số:
            model_path: Đường dẫn file .joblib đã huấn luyện. Mặc định dùng kmeans_segmentation.joblib.
        """
        resolved_path = Path(
            SAVED_MODELS_DIR / "kmeans_segmentation.joblib"
            if model_path is None
            else model_path
        )
        self.model, self.metadata = load_model_pipeline(resolved_path)
        self.centroids: Optional[np.ndarray] = getattr(
            self.model,
            "cluster_centers_",
            getattr(self.model, "centroids", None),
        )
        if self.centroids is None:
            raise ValueError(
                "Loaded model does not expose cluster_centers_ or centroids attribute."
            )

        self.color_space: str = str(self.metadata.get("color_space", "RGB")).strip().upper()
        self.rgb_centroids: np.ndarray = convert_centroids_to_rgb(
            self.centroids, source_space=self.color_space
        )
        self.segmenter = ImageSegmenter()
        logger.info(
            f"Đã khởi tạo ImageSegmentationPredictor với K={len(self.centroids)} tâm màu "
            f"({self.color_space}) từ '{resolved_path.name}'."
        )

    def _predict_in_chunks(self, pixel_matrix: np.ndarray, chunk_size: int = 250000) -> np.ndarray:
        """Gán nhãn theo từng lô để tránh tràn bộ nhớ RAM khi suy luận ảnh độ phân giải cao."""
        n_pixels = pixel_matrix.shape[0]
        if n_pixels <= chunk_size:
            return np.asarray(self.model.predict(pixel_matrix), dtype=np.int64)

        labels_out = np.empty(n_pixels, dtype=np.int64)
        for start in range(0, n_pixels, chunk_size):
            end = min(start + chunk_size, n_pixels)
            labels_out[start:end] = self.model.predict(pixel_matrix[start:end])
        return labels_out

    def predict(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[Path] = None,
        target_size: int = TARGET_SIZE,
        return_original_size: bool = True,
    ) -> Dict[str, Any]:
        """
        Chạy quy trình suy luận phân đoạn màu hoàn chỉnh cho một file ảnh bất kỳ.

        Tham số:
            image_path: Đường dẫn file ảnh cần phân đoạn.
            output_dir: Thư mục lưu ảnh phân đoạn, ảnh so sánh và file JSON thống kê (tùy chọn).
            target_size: Kích thước cạnh dài nhất dùng khi return_original_size=False.
            return_original_size: Nếu True, gán cụm toàn bộ pixel ảnh gốc để giữ nguyên chi tiết.

        Trả về:
            Từ điển chứa ảnh phân đoạn, metadata ảnh gốc, bảng phân bố màu và đường dẫn file đã xuất.
        """
        assert self.centroids is not None
        img_rgb, img_metadata = load_image(image_path)

        if return_original_size:
            display_orig = img_rgb
            target_shape = (int(img_rgb.shape[0]), int(img_rgb.shape[1]))
        else:
            display_orig, target_shape = resize_image(
                img_rgb, target_size=target_size, keep_aspect_ratio=True
            )

        feature_img = (
            convert_color_space(display_orig, source_space="RGB", target_space=self.color_space)
            if self.color_space != "RGB"
            else display_orig
        )
        pixels = flatten_pixels(feature_img, as_float=True)
        labels = self._predict_in_chunks(pixels)
        segmented_img = self.segmenter.reconstruct_segmented_image(
            labels, self.rgb_centroids, target_shape=target_shape
        )
        cluster_stats = extract_color_distribution(labels, self.rgb_centroids)

        output_files: Dict[str, str] = {}
        if output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(image_path).stem
            optimal_k = int(len(self.centroids))

            seg_path = out_dir / f"{stem}_segmented_k{optimal_k}.png"
            comp_path = out_dir / f"{stem}_comparison_k{optimal_k}.png"
            stats_path = out_dir / f"{stem}_stats_k{optimal_k}.json"

            # Xuất ảnh phân đoạn PNG bằng cv2.imencode + tofile để hỗ trợ đường dẫn Unicode
            ok, buf = cv2.imencode(".png", cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))
            if ok:
                buf.tofile(str(seg_path))
                output_files["segmented_image_path"] = str(seg_path)

            # Ánh xạ nhãn và ảnh gốc cùng kích thước về RGB để hiển thị so sánh nhất quán
            plot_side_by_side(
                display_orig,
                segmented_img,
                k=optimal_k,
                title=f"Inference Segmentation Result ({Path(image_path).name}, K={optimal_k})",
                save_path=comp_path,
            )
            output_files["comparison_figure_path"] = str(comp_path)

            stats_path.write_text(
                json.dumps(cluster_stats, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            output_files["stats_json_path"] = str(stats_path)
            logger.info(f"Đã xuất kết quả suy luận ảnh '{stem}' vào thư mục: {out_dir}")

        return {
            "segmented_image": segmented_img,
            "original_metadata": img_metadata,
            "cluster_stats": cluster_stats,
            "optimal_k": int(len(self.centroids)),
            "output_files": output_files,
        }

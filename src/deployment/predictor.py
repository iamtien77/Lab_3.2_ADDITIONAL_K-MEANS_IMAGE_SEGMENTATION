"""
=============================================================================
MODULE: TRIỂN KHAI VÀ SUY LUẬN PHÂN ĐOẠN ẢNH (PREDICTOR / INFERENCE)
=============================================================================
Mục đích:
- Cung cấp pipeline suy luận trọn gói (End-to-End Inference) cho ảnh mới.
- Tự động: nạp ảnh → tiền xử lý → phân đoạn → tái tạo kích thước gốc → lưu kết quả.
=============================================================================
"""

from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np

from src.utils.logger import get_logger
from src.data.loader import load_image
from src.data.preprocessing import convert_color_space, resize_image, flatten_pixels
from src.segmentation.segmenter import ImageSegmenter
from src.features.color_features import extract_color_distribution
from src.evaluation.visualization import plot_side_by_side
from src.models.persistence import load_model_pipeline
from src.config import TARGET_SIZE, FIGURES_DIR

logger = get_logger("ImageSegmentationPredictor")


class ImageSegmentationPredictor:
    """
    Lớp điều phối suy luận phân đoạn ảnh cho người dùng cuối.
    Cho phép áp dụng bộ tâm màu đã học vào bất kỳ ảnh mới nào.
    """

    def __init__(self, model_path: Optional[Path] = None):
        """
        Nạp mô hình K-Means và metadata từ model_path.
        Khởi tạo đối tượng ImageSegmenter.
        """
        self.model = None
        self.metadata: Dict[str, Any] = {}
        self.segmenter = ImageSegmenter()

        if model_path is not None:
            model_path = Path(model_path)
            self.model, self.metadata = load_model_pipeline(model_path)
            logger.info(
                f"Predictor sẵn sàng | model_path={model_path.name} | "
                f"metadata={self.metadata}"
            )
        else:
            logger.warning(
                "Khởi tạo Predictor không có model_path. "
                "Hãy cung cấp model sau bằng thuộc tính .model và .metadata."
            )

    def predict(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[Path] = None,
        target_size: int = TARGET_SIZE,
        return_original_size: bool = True,
    ) -> Dict[str, Any]:
        """
        Pipeline suy luận đầu cuối cho một ảnh mới.

        Returns dict:
            - segmented_img: ảnh phân đoạn (kích thước gốc hoặc resize tùy return_original_size)
            - segmented_resized: ảnh phân đoạn tại kích thước làm việc
            - labels: nhãn cụm (N,)
            - centroids: tâm cụm (K, 3)
            - color_distribution: thống kê phân bố màu
            - metadata: thông tin ảnh gốc
        """
        if self.model is None:
            raise RuntimeError(
                "Chưa có mô hình. Khởi tạo Predictor với model_path hợp lệ."
            )

        image_path = Path(image_path)

        # --- 1. Đọc ảnh ---
        img_bgr_or_rgb, img_meta = load_image(image_path)
        original_shape = img_meta["original_shape"]  # (H, W, 3)

        # --- 2. Chuyển màu sang RGB ---
        source_format = img_meta.get("color_format", "")
        if "BGR" in source_format.upper():
            img_rgb = convert_color_space(img_bgr_or_rgb, "BGR", "RGB")
        else:
            img_rgb = img_bgr_or_rgb

        # --- 3. Resize về kích thước làm việc ---
        img_resized, resized_shape = resize_image(img_rgb, target_size=target_size)

        # --- 4. Flatten ---
        pixel_matrix = flatten_pixels(img_resized, as_float=True)

        # --- 5. Dự đoán nhãn cụm ---
        # Hỗ trợ cả sklearn KMeans (predict) và KMeansFromScratch (predict)
        labels = self.model.predict(pixel_matrix)

        # Lấy centroids
        if hasattr(self.model, "cluster_centers_"):
            centroids = self.model.cluster_centers_
        elif hasattr(self.model, "centroids"):
            centroids = self.model.centroids
        else:
            raise AttributeError("Mô hình không có thuộc tính cluster_centers_ hoặc centroids.")

        # --- 6. Tái tạo ảnh phân đoạn ---
        segmented_resized = self.segmenter.reconstruct_segmented_image(
            labels, centroids, resized_shape
        )

        if return_original_size:
            segmented_output = self.segmenter.reconstruct_original_size(
                labels, centroids, original_shape, resized_shape
            )
        else:
            segmented_output = segmented_resized

        # --- 7. Thống kê phân bố màu ---
        color_dist = extract_color_distribution(labels, centroids)

        # --- 8. Lưu kết quả nếu có output_dir ---
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            stem = image_path.stem
            k = len(centroids)

            # Lưu ảnh phân đoạn
            seg_save = output_dir / f"{stem}_segmented_K{k}.png"
            try:
                import cv2
                seg_bgr = convert_color_space(segmented_output, "RGB", "BGR")
                cv2.imwrite(str(seg_save), seg_bgr)
            except ImportError:
                from PIL import Image as PILImage
                PILImage.fromarray(segmented_output).save(seg_save)

            # Lưu side-by-side
            sbs_save = output_dir / f"{stem}_side_by_side_K{k}.png"
            plot_side_by_side(
                img_rgb, segmented_output, k=k,
                title=f"K-Means Segmentation — {image_path.name}",
                save_path=sbs_save,
            )
            logger.info(f"Kết quả đã lưu tại: {output_dir}")

        return {
            "segmented_img": segmented_output,
            "segmented_resized": segmented_resized,
            "labels": labels,
            "centroids": centroids,
            "color_distribution": color_dist,
            "image_metadata": img_meta,
            "resized_shape": resized_shape,
        }

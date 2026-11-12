"""
=============================================================================
MODULE: TÁI TẠO VÀ PHÂN ĐOẠN ẢNH (IMAGE SEGMENTER)
=============================================================================
Mục đích (Đáp ứng đầy đủ Task 4 trong đề bài Practice Exercise 3.2):
- Ánh xạ từng điểm ảnh sang màu đại diện của tâm cụm tương ứng:
  centroids[labels].
- Khôi phục cấu trúc không gian 2D (H, W, 3) kiểu uint8 [0, 255].
- Hỗ trợ phục hồi kích thước gốc từ bản đồ nhãn bằng nội suy láng giềng gần nhất
  (Nearest Neighbor) và trích xuất mặt nạ vùng / đường ranh giới phân cụm.
=============================================================================
"""

from typing import Dict, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ImageSegmenter:
    """
    Lớp phụ trách chuyển đổi nhãn phân cụm và bảng màu tâm cụm thành ảnh phân đoạn.
    """

    def __init__(self):
        """
        Khởi tạo đối tượng phân đoạn và tái tạo ảnh.
        """
        pass

    def _validate_labels_and_centroids(
        self,
        labels: np.ndarray,
        centroids: np.ndarray,
        expected_len: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Kiểm định mảng nhãn 1D và ma trận tâm cụm 2D khớp kích thước kỳ vọng."""
        labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
        centroids_arr = np.asarray(centroids, dtype=np.float64)

        if labels_arr.size != expected_len:
            raise ValueError(
                f"Label count ({labels_arr.size}) does not match target spatial size ({expected_len})."
            )
        if centroids_arr.ndim != 2 or centroids_arr.shape[1] != 3 or centroids_arr.shape[0] == 0:
            raise ValueError("centroids must be a non-empty 2D array of shape (K, 3).")
        if not np.isfinite(centroids_arr).all():
            raise ValueError("centroids contains NaN or Inf values.")

        k = centroids_arr.shape[0]
        if int(labels_arr.min()) < 0 or int(labels_arr.max()) >= k:
            raise ValueError(
                f"Label values out of bounds [0, {k - 1}]: "
                f"got [{int(labels_arr.min())}, {int(labels_arr.max())}]."
            )

        palette_uint8 = np.clip(np.round(centroids_arr), 0, 255).astype(np.uint8)
        return labels_arr, palette_uint8

    def reconstruct_segmented_image(
        self,
        labels: np.ndarray,
        centroids: np.ndarray,
        target_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Thay thế mỗi pixel bằng màu tâm cụm và reshape thành ảnh RGB 3D (H, W, 3).

        Tham số:
            labels: Mảng 1D nhãn cụm có độ dài H * W.
            centroids: Ma trận tâm cụm (K, 3) trên thang RGB [0, 255].
            target_shape: Kích thước ảnh mục tiêu (height, width).

        Trả về:
            Mảng ảnh phân đoạn kiểu uint8 kích thước (height, width, 3).
        """
        height, width = int(target_shape[0]), int(target_shape[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"target_shape dimensions must be positive, got {target_shape}.")

        labels_arr, palette_uint8 = self._validate_labels_and_centroids(
            labels, centroids, height * width
        )

        # Ánh xạ nhãn rời rạc sang màu uint8 rồi khôi phục shape (H, W, 3) đúng thứ tự hàng
        segmented_pixels = palette_uint8[labels_arr]
        segmented_img = segmented_pixels.reshape((height, width, 3))

        logger.info(
            f"Đã tái tạo ảnh phân đoạn kích thước {width}x{height} với K={len(palette_uint8)} màu tâm cụm."
        )
        return segmented_img

    def reconstruct_original_size(
        self,
        labels_resized: np.ndarray,
        centroids: np.ndarray,
        original_shape: Tuple[int, int, int],
        resized_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Phóng đại bản đồ nhãn từ kích thước làm việc về độ phân giải gốc bằng Nearest Neighbor
        và ánh xạ màu tâm cụm để giữ biên màu sắc nét.

        Tham số:
            labels_resized: Mảng 1D nhãn cụm ở kích thước đã resize.
            centroids: Ma trận tâm cụm (K, 3).
            original_shape: Shape ảnh gốc (H_orig, W_orig, C).
            resized_shape: Shape ảnh sau resize (H_resized, W_resized).

        Trả về:
            Ảnh phân đoạn ở độ phân giải gốc (H_orig, W_orig, 3) kiểu uint8.
        """
        h_orig, w_orig = int(original_shape[0]), int(original_shape[1])
        h_res, w_res = int(resized_shape[0]), int(resized_shape[1])
        if h_orig <= 0 or w_orig <= 0 or h_res <= 0 or w_res <= 0:
            raise ValueError("Image shapes must have positive dimensions.")

        labels_arr, palette_uint8 = self._validate_labels_and_centroids(
            labels_resized, centroids, h_res * w_res
        )
        label_map_small = labels_arr.reshape((h_res, w_res)).astype(np.int32)

        if (h_res, w_res) == (h_orig, w_orig):
            label_map_orig = label_map_small
        else:
            # Dùng INTER_NEAREST trên nhãn rời rạc để tránh nội suy sinh màu giả ở biên cụm
            label_map_orig = cv2.resize(
                label_map_small,
                (w_orig, h_orig),
                interpolation=cv2.INTER_NEAREST,
            )

        segmented_orig = palette_uint8[label_map_orig]
        logger.info(
            f"Đã phục hồi ảnh phân đoạn từ {w_res}x{h_res} về kích thước gốc {w_orig}x{h_orig}."
        )
        return segmented_orig

    def extract_cluster_masks(
        self,
        labels: np.ndarray,
        shape: Tuple[int, int],
        k: int,
    ) -> Dict[int, np.ndarray]:
        """
        Tạo mặt nạ nhị phân (0 hoặc 255) cho từng cụm màu riêng biệt từ 0 đến k - 1.

        Tham số:
            labels: Mảng 1D nhãn cụm có độ dài H * W.
            shape: Kích thước không gian (height, width).
            k: Tổng số cụm K.

        Trả về:
            Từ điển {cluster_id: mask_2d_uint8}.
        """
        height, width = int(shape[0]), int(shape[1])
        labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
        if labels_arr.size != height * width:
            raise ValueError(
                f"Label length ({labels_arr.size}) does not match shape ({height}x{width})."
            )
        if k < 1:
            raise ValueError(f"k must be positive, got {k}.")

        label_map = labels_arr.reshape((height, width))
        masks: Dict[int, np.ndarray] = {}
        for cluster_id in range(int(k)):
            masks[cluster_id] = np.where(label_map == cluster_id, 255, 0).astype(np.uint8)
        return masks

    def extract_cluster_boundaries(
        self,
        labels: np.ndarray,
        shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Phát hiện các điểm ảnh nằm trên đường ranh giới chuyển tiếp giữa các cụm màu khác nhau.

        Tham số:
            labels: Mảng 1D nhãn cụm có độ dài H * W.
            shape: Kích thước không gian (height, width).

        Trả về:
            Mảng nhị phân 2D (height, width) kiểu uint8 với điểm biên mang giá trị 255.
        """
        height, width = int(shape[0]), int(shape[1])
        labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
        if labels_arr.size != height * width:
            raise ValueError(
                f"Label length ({labels_arr.size}) does not match shape ({height}x{width})."
            )

        label_map = labels_arr.reshape((height, width))
        boundary = np.zeros((height, width), dtype=np.uint8)

        # Kiểm tra sự khác nhãn với pixel liền kề theo chiều ngang và chiều dọc
        if width > 1:
            h_diff = label_map[:, 1:] != label_map[:, :-1]
            boundary[:, 1:][h_diff] = 255
            boundary[:, :-1][h_diff] = 255
        if height > 1:
            v_diff = label_map[1:, :] != label_map[:-1, :]
            boundary[1:, :][v_diff] = 255
            boundary[:-1, :][v_diff] = 255

        return boundary

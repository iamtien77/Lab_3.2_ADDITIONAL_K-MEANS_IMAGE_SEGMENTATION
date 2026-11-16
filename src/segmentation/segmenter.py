"""
=============================================================================
MODULE: TÁI TẠO VÀ PHÂN ĐOẠN ẢNH (IMAGE SEGMENTER)
=============================================================================
Mục đích (Đáp ứng đầy đủ Task 4 trong đề bài Practice Exercise 3.2):
- Thay thế mỗi pixel bằng màu của tâm cụm tương ứng.
- Reshape ảnh phân đoạn trở lại kích thước không gian 2D.
- Trích xuất bản đồ nhãn (Label Map), mặt nạ phân vùng (Masks) và đường ranh giới (Boundaries).
=============================================================================
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("ImageSegmenter")


class ImageSegmenter:
    """
    Lớp phụ trách chuyển đổi nhãn cụm và tâm màu thành ảnh phân đoạn hoàn chỉnh.
    """

    def __init__(self):
        """Khởi tạo đối tượng phân đoạn ảnh."""
        logger.info("ImageSegmenter khởi tạo.")

    def reconstruct_segmented_image(
        self,
        labels: np.ndarray,
        centroids: np.ndarray,
        target_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Tái tạo ảnh phân đoạn tại kích thước làm việc (ảnh đã resize).
        Mỗi pixel được thay thế bằng màu tâm cụm của nó.

        Args:
            labels:       (N,) nhãn cụm cho từng pixel.
            centroids:    (K, 3) tâm cụm màu.
            target_shape: (h, w) kích thước ảnh làm việc.
        Returns:
            ảnh phân đoạn (h, w, 3) kiểu uint8.
        """
        labels = np.asarray(labels, dtype=np.int32)
        centroids = np.asarray(centroids, dtype=np.float32)

        h, w = target_shape
        n_pixels = h * w

        if len(labels) != n_pixels:
            raise ValueError(
                f"Số nhãn ({len(labels)}) không khớp với h×w={n_pixels} (target_shape={target_shape})."
            )

        # Ánh xạ nhãn → màu tâm
        segmented_pixels = centroids[labels]  # (N, 3)

        # Clip về [0, 255] và chuyển sang uint8
        segmented_pixels = np.clip(segmented_pixels, 0, 255).astype(np.uint8)

        # Reshape → (h, w, 3)
        segmented_img = segmented_pixels.reshape(h, w, 3)

        logger.info(f"Tái tạo ảnh phân đoạn: shape={segmented_img.shape}")
        return segmented_img

    def reconstruct_original_size(
        self,
        labels_resized: np.ndarray,
        centroids: np.ndarray,
        original_shape: Tuple[int, int, int],
        resized_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Phục hồi ảnh phân đoạn về độ phân giải gốc (H_orig, W_orig).
        Phương pháp: Nearest-neighbor upscale nhãn → ánh xạ màu.
        Không học lại tâm; không reshape nhãn resized thành nhãn gốc trực tiếp.

        Args:
            labels_resized: (N_resized,) nhãn của ảnh làm việc.
            centroids:      (K, 3) tâm cụm màu.
            original_shape: (H_orig, W_orig, 3) shape ảnh gốc.
            resized_shape:  (h_resized, w_resized) shape ảnh làm việc.
        Returns:
            ảnh phân đoạn (H_orig, W_orig, 3) uint8.
        """
        h_r, w_r = resized_shape
        H_orig, W_orig = original_shape[0], original_shape[1]

        # Reshape nhãn resized thành 2D label map
        label_map_resized = labels_resized.reshape(h_r, w_r).astype(np.int32)

        # Upscale label map bằng nearest neighbor (giữ nguyên nhãn rời rạc)
        try:
            import cv2
            label_map_orig = cv2.resize(
                label_map_resized.astype(np.float32),
                (W_orig, H_orig),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)
        except ImportError:
            # Fallback: lặp nhãn bằng numpy repeat
            scale_h = H_orig / h_r
            scale_w = W_orig / w_r
            row_idx = np.minimum(
                (np.arange(H_orig) / scale_h).astype(np.int32), h_r - 1
            )
            col_idx = np.minimum(
                (np.arange(W_orig) / scale_w).astype(np.int32), w_r - 1
            )
            label_map_orig = label_map_resized[np.ix_(row_idx, col_idx)]

        # Ánh xạ nhãn → màu tâm
        centroids = np.asarray(centroids, dtype=np.float32)
        labels_flat = label_map_orig.flatten()
        segmented_pixels = centroids[labels_flat]
        segmented_img = np.clip(segmented_pixels, 0, 255).astype(np.uint8)
        segmented_img = segmented_img.reshape(H_orig, W_orig, 3)

        logger.info(
            f"Phục hồi ảnh về kích thước gốc: ({h_r}×{w_r}) → ({H_orig}×{W_orig})"
        )
        return segmented_img

    def extract_cluster_masks(
        self,
        labels: np.ndarray,
        shape: Tuple[int, int],
        k: int,
    ) -> Dict[int, np.ndarray]:
        """
        Tạo mặt nạ nhị phân cho từng cụm.
        Returns: {cluster_id: mask_uint8 (H, W) với 255=thuộc cụm, 0=không}
        """
        label_map = labels.reshape(shape[0], shape[1]).astype(np.int32)
        masks: Dict[int, np.ndarray] = {}

        for c in range(k):
            mask = ((label_map == c) * 255).astype(np.uint8)
            masks[c] = mask

        return masks

    def extract_cluster_boundaries(
        self,
        labels: np.ndarray,
        shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Trích xuất đường ranh giới giữa các cụm.
        Returns: mặt nạ nhị phân (H, W) uint8 — 255 = biên giới cụm.
        """
        label_map = labels.reshape(shape[0], shape[1]).astype(np.int32)

        # Phát hiện thay đổi nhãn theo chiều ngang và dọc
        diff_h = np.abs(np.diff(label_map, axis=0))  # (H-1, W)
        diff_w = np.abs(np.diff(label_map, axis=1))  # (H, W-1)

        boundary = np.zeros(shape, dtype=np.uint8)

        # Đánh dấu pixel biên giới (theo cả hai chiều)
        boundary[:-1, :] |= (diff_h > 0).astype(np.uint8) * 255
        boundary[:, :-1] |= (diff_w > 0).astype(np.uint8) * 255

        return boundary

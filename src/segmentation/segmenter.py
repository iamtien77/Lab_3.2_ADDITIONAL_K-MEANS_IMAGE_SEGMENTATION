"""
=============================================================================
MODULE: TÁI TẠO VÀ PHÂN ĐOẠN ẢNH (IMAGE SEGMENTER)
=============================================================================
Mục đích (Đáp ứng đầy đủ Task 4 trong đề bài Practice Exercise 3.2):
- Thay thế mỗi pixel bằng màu của tâm cụm tương ứng:
  "Replace each pixel with the color of its assigned cluster centroid."
- Reshape ảnh phân đoạn trở lại kích thước không gian 2D:
  "Reshape the segmented image back to its original dimensions."
- Trích xuất bản đồ nhãn (Label Map), mặt nạ phân vùng (Masks) và đường ranh giới cụm (Boundaries).
=============================================================================
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


class ImageSegmenter:
    """
    Lớp phụ trách chuyển đổi nhãn cụm và tâm màu thành ảnh phân đoạn hoàn chỉnh.
    """

    def __init__(self):
        """
        Ý LÀM:
        Khởi tạo đối tượng phân đoạn ảnh.
        """
        pass

    def reconstruct_segmented_image(
        self,
        labels: np.ndarray,
        centroids: np.ndarray,
        target_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Ý LÀM (Đáp ứng Task 4 - Phân đoạn ảnh và Reshape về kích thước không gian):
        1. Thay thế mỗi pixel bằng vector màu của tâm cụm được gán:
           segmented_pixels = centroids[labels].
        2. Ép kiểu giá trị màu về uint8 [0, 255] (cắt tỉa np.clip nếu cần).
        3. Reshape mảng 1D (N_pixels, 3) trở lại kích thước ảnh 2D (target_shape[0], target_shape[1], 3).
        4. Trả về mảng ảnh phân đoạn đã định dạng màu chuẩn RGB.
        """
        pass

    def reconstruct_original_size(
        self,
        labels_resized: np.ndarray,
        centroids: np.ndarray,
        original_shape: Tuple[int, int, int],
        resized_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Ý LÀM (Phục hồi ảnh phân đoạn về độ phân giải gốc ban đầu):
        1. Reshape nhãn cụm thành ma trận 2D kích thước (H_resized, W_resized).
        2. Áp dụng phóng đại nhãn về kích thước gốc (H_orig, W_orig) bằng phương pháp
           nội suy láng giềng gần nhất (Nearest Neighbor) để giữ nguyên nhãn cụm rời rạc.
        3. Ánh xạ các nhãn đã phóng đại sang màu tâm cụm tương ứng centroids[labels_orig].
        4. Trả về ảnh phân đoạn ở độ phân giải gốc sắc nét mà không bị mờ nhòe.
        """
        pass

    def extract_cluster_masks(
        self,
        labels: np.ndarray,
        shape: Tuple[int, int],
        k: int,
    ) -> Dict[int, np.ndarray]:
        """
        Ý LÀM:
        1. Reshape nhãn 1D thành bản đồ nhãn 2D (H, W).
        2. Duyệt qua từng cụm c từ 0 đến k - 1:
           - Tạo mặt nạ nhị phân uint8: mask = (label_map == c) * 255.
        3. Trả về từ điển chứa mặt nạ nhị phân của từng phân khúc cụm.
        """
        pass

    def extract_cluster_boundaries(
        self,
        labels: np.ndarray,
        shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Ý LÀM:
        1. Reshape nhãn thành bản đồ 2D (H, W).
        2. Tính toán độ biến thiên nhãn giữa các pixel liền kề (theo chiều ngang và chiều dọc).
        3. Điểm nào có sự thay đổi nhãn với láng giềng được đánh dấu là đường ranh giới (boundary = 255).
        4. Trả về mặt nạ nhị phân chứa các đường ranh giới phân khúc.
        """
        pass

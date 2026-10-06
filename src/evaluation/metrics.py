"""
=============================================================================
MODULE: ĐÁNH GIÁ CHẤT LƯỢNG PHÂN ĐOẠN (SEGMENTATION METRICS)
=============================================================================
Mục đích:
- Tính toán tổng bình phương sai số nội cụm WCSS (Inertia).
- Đánh giá hệ số Silhouette Score để đo độ nén và độ phân tách màu sắc.
- So sánh các chỉ số tương đồng phân cụm (Adjusted Rand Index, NMI).
=============================================================================
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score


def compute_wcss(
    X: np.ndarray,
    centroids: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Ý LÀM:
    1. Tính khoảng cách bình phương từ mỗi điểm X[i] tới centroids[labels[i]].
    2. Tính tổng toàn bộ khoảng cách để ra WCSS (Within-Cluster Sum of Squares).
    3. Trả về giá trị WCSS.
    """
    pass


def compute_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    max_samples: int = 5000,
    random_state: int = 42,
) -> float:
    """
    Ý LÀM:
    1. Kiểm tra số cụm hợp lệ (phải >= 2 và < số mẫu).
    2. Nếu số lượng điểm ảnh lớn (ví dụ 65,536 điểm), thực hiện lấy mẫu ngẫu nhiên
       phân tầng (Stratified subsample) tối đa max_samples điểm để tối ưu hóa thời gian tính toán.
    3. Gọi silhouette_score từ sklearn.metrics trên tập mẫu đã trích xuất.
    4. Trả về hệ số Silhouette trung bình [-1.0, 1.0].
    """
    pass


def compare_clustering_labels(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> Dict[str, float]:
    """
    Ý LÀM:
    1. Tính Adjusted Rand Index (ARI): đo mức độ tương đồng nhãn độc lập với cách đánh số.
    2. Tính Normalized Mutual Information (NMI): đo lượng thông tin tương hỗ giữa 2 cách phân cụm.
    3. Trả về dictionary: {"ari": float, "nmi": float}.
    """
    pass

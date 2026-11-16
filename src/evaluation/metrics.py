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
    Tính tổng bình phương khoảng cách từ mỗi điểm X[i] tới centroids[labels[i]].
    Returns: WCSS (Within-Cluster Sum of Squares) — số thực không âm hữu hạn.
    """
    X = np.asarray(X, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    assigned = centroids[labels]          # (N, D)
    diff = X - assigned                   # (N, D)
    wcss = float(np.sum(diff ** 2))
    return wcss


def compute_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    max_samples: int = 5000,
    random_state: int = 42,
) -> float:
    """
    Tính Silhouette Score trên tập mẫu đại diện (tối đa max_samples điểm).
    Trả về -1.0 nếu số cụm không hợp lệ.
    """
    n_samples = len(X)
    n_labels = len(np.unique(labels))

    # Điều kiện hợp lệ: 2 <= n_labels <= n_samples - 1
    if n_labels < 2 or n_labels >= n_samples:
        return -1.0

    # Stratified subsample nếu dữ liệu lớn
    if n_samples > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_samples, size=max_samples, replace=False)
        X_sample = X[idx]
        labels_sample = labels[idx]
        # Đảm bảo sample vẫn có đủ n_labels
        if len(np.unique(labels_sample)) < 2:
            return -1.0
    else:
        X_sample = X
        labels_sample = labels

    score = silhouette_score(X_sample, labels_sample, metric="euclidean")
    return float(score)


def compare_clustering_labels(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> Dict[str, float]:
    """
    So sánh hai tập nhãn phân cụm bằng ARI và NMI.
    Returns: {"ari": float, "nmi": float}
    """
    ari = float(adjusted_rand_score(labels_a, labels_b))
    nmi = float(normalized_mutual_info_score(labels_a, labels_b))
    return {"ari": ari, "nmi": nmi}

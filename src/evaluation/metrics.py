"""
=============================================================================
MODULE: ĐÁNH GIÁ CHẤT LƯỢNG PHÂN ĐOẠN (SEGMENTATION METRICS)
=============================================================================
Mục đích:
- Tính toán tổng bình phương khoảng cách nội cụm WCSS (Inertia).
- Đánh giá hệ số Silhouette Score với lấy mẫu phân tầng (stratified subsampling)
  an toàn về bộ nhớ và thời gian trên ma trận pixel lớn.
- Đối chiếu độ nhất quán phân cụm giữa hai mô hình qua ARI và NMI.
=============================================================================
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

from src.config import RANDOM_SEED
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_wcss(
    X: np.ndarray,
    centroids: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Tính tổng bình phương khoảng cách nội cụm WCSS (Within-Cluster Sum of Squares).

    Tham số:
        X: Ma trận đặc trưng 2D kích thước (N, D).
        centroids: Ma trận tâm cụm (K, D).
        labels: Mảng 1D nhãn cụm (N,) thuộc [0, K-1].

    Trả về:
        Giá trị WCSS kiểu float không âm.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    c_arr = np.asarray(centroids, dtype=np.float64)
    lbl_arr = np.asarray(labels, dtype=np.int64).reshape(-1)

    if X_arr.ndim != 2 or c_arr.ndim != 2 or X_arr.shape[1] != c_arr.shape[1]:
        raise ValueError("Incompatible shapes between X and centroids.")
    if X_arr.shape[0] == 0 or lbl_arr.size != X_arr.shape[0]:
        raise ValueError("labels length must match number of rows in X and be non-empty.")
    if not (np.isfinite(X_arr).all() and np.isfinite(c_arr).all()):
        raise ValueError("X and centroids must contain finite values.")
    if int(lbl_arr.min()) < 0 or int(lbl_arr.max()) >= c_arr.shape[0]:
        raise ValueError("labels contain cluster indices outside [0, K-1].")

    diff = X_arr - c_arr[lbl_arr]
    wcss = float(np.sum(diff * diff))
    return wcss


def compute_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    max_samples: int = 5000,
    random_state: int = RANDOM_SEED,
) -> float:
    """
    Tính hệ số Silhouette trung bình [-1.0, 1.0], sử dụng lấy mẫu phân tầng khi dữ liệu lớn.

    Tham số:
        X: Ma trận đặc trưng 2D kích thước (N, D).
        labels: Mảng 1D nhãn cụm (N,).
        max_samples: Số lượng điểm mẫu tối đa để tránh quá tải ma trận khoảng cách O(N^2).
        random_state: Seed cố định khi lấy mẫu ngẫu nhiên phân tầng.

    Trả về:
        Hệ số Silhouette Score nằm trong [-1.0, 1.0] (trả về 0.0 nếu chỉ có 1 cụm).
    """
    X_arr = np.asarray(X, dtype=np.float32)
    lbl_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    if X_arr.ndim != 2 or X_arr.shape[0] != lbl_arr.size or X_arr.shape[0] < 2:
        raise ValueError("Invalid inputs for silhouette score computation.")

    unique_labels, counts = np.unique(lbl_arr, return_counts=True)
    if len(unique_labels) < 2 or len(unique_labels) >= X_arr.shape[0]:
        logger.warning(
            f"Không thể tính Silhouette Score hợp lệ với {len(unique_labels)} cụm trên {X_arr.shape[0]} mẫu, trả về 0.0."
        )
        return 0.0

    # Lấy mẫu phân tầng nếu N > max_samples và mỗi cụm có tối thiểu 2 phần tử để split an toàn
    if X_arr.shape[0] > max_samples:
        if np.all(counts >= 2) and max_samples >= len(unique_labels):
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                train_size=int(max_samples),
                random_state=int(random_state),
            )
            sample_idx, _ = next(splitter.split(X_arr, lbl_arr))
        else:
            rng = np.random.default_rng(int(random_state))
            sample_idx = rng.choice(X_arr.shape[0], size=int(max_samples), replace=False)

        X_eval = X_arr[sample_idx]
        lbl_eval = lbl_arr[sample_idx]
    else:
        X_eval = X_arr
        lbl_eval = lbl_arr

    score = float(silhouette_score(X_eval, lbl_eval, metric="euclidean"))
    logger.info(
        f"Đã tính Silhouette Score={score:.4f} (trên {len(X_eval):,} mẫu đại diện, K={len(unique_labels)})."
    )
    return score


def compare_clustering_labels(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> Dict[str, float]:
    """
    So sánh độ tương đồng giữa hai bộ nhãn phân cụm bằng ARI và NMI (bất biến hoán vị nhãn).

    Tham số:
        labels_a: Mảng 1D nhãn phân cụm thứ nhất.
        labels_b: Mảng 1D nhãn phân cụm thứ hai.

    Trả về:
        Từ điển gồm {"ari": float, "nmi": float}.
    """
    a_arr = np.asarray(labels_a, dtype=np.int64).reshape(-1)
    b_arr = np.asarray(labels_b, dtype=np.int64).reshape(-1)
    if a_arr.size == 0 or a_arr.shape != b_arr.shape:
        raise ValueError("labels_a and labels_b must be non-empty 1D arrays of equal length.")

    ari = round(float(adjusted_rand_score(a_arr, b_arr)), 4)
    nmi = round(float(normalized_mutual_info_score(a_arr, b_arr)), 4)
    return {"ari": ari, "nmi": nmi}

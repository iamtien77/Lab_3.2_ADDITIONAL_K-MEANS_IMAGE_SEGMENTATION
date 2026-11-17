"""
=============================================================================
MODULE: K-MEANS TỪ ĐẦU (K-MEANS FROM SCRATCH & CONVERGENCE TRACKING)
=============================================================================
Mục đích (Đáp ứng đầy đủ Task 3 trong đề bài Practice Exercise 3.2):
- Khởi tạo K tâm cụm ngẫu nhiên đại diện cho màu trung bình của từng cụm.
- Gán mỗi pixel về tâm cụm gần nhất dựa trên khoảng cách Euclidean bình phương.
- Cập nhật tâm cụm bằng trung bình cộng các pixel thuộc cụm; xử lý an toàn cụm rỗng.
- Lặp lại hai bước gán và cập nhật cho đến khi hội tụ hoặc đạt ngưỡng max_iter.
- Ghi nhận đầy đủ lịch sử hội tụ (Inertia, centroid shift, label change ratio).
=============================================================================
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import MAX_ITER, RANDOM_SEED, TOLERANCE
from src.utils.logger import get_logger

logger = get_logger(__name__)


class KMeansFromScratch:
    """
    Thuật toán K-Means cài đặt thuần từ đầu (Pure NumPy) kèm theo dõi hội tụ từng bước.
    """

    def __init__(
        self,
        n_clusters: int = 4,
        max_iter: int = MAX_ITER,
        tol: float = TOLERANCE,
        random_state: int = RANDOM_SEED,
    ):
        """
        Khởi tạo siêu tham số và các thuộc tính lưu trạng thái huấn luyện K-Means.

        Tham số:
            n_clusters: Số lượng cụm màu K (> 0).
            max_iter: Giới hạn số vòng lặp tối đa (>= 1).
            tol: Ngưỡng hội tụ độ dịch chuyển lớn nhất của tâm cụm (>= 0).
            random_state: Seed cố định để tái lập kết quả khởi tạo ngẫu nhiên.
        """
        if not isinstance(n_clusters, (int, np.integer)) or n_clusters < 1:
            raise ValueError(f"n_clusters must be a positive integer, got {n_clusters}.")
        if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
            raise ValueError(f"max_iter must be a positive integer, got {max_iter}.")
        if tol < 0.0:
            raise ValueError(f"tol must be non-negative, got {tol}.")

        self.n_clusters: int = int(n_clusters)
        self.max_iter: int = int(max_iter)
        self.tol: float = float(tol)
        self.random_state: int = int(random_state)

        self.centroids: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.n_iter_: int = 0
        self.history_: List[Dict[str, Any]] = []
        self.stop_reason_: Optional[str] = None
        self.empty_cluster_events_: int = 0

    def _validate_matrix(self, X: np.ndarray) -> np.ndarray:
        """Kiểm định ma trận đặc trưng 2D (N, D) phải không rỗng và hoàn toàn hữu hạn."""
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError("Input X must be a non-empty 2D array of shape (N, D).")
        if not np.isfinite(arr).all():
            raise ValueError("Input X contains NaN or Inf values.")
        return arr

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Khởi tạo K tâm cụm màu phân biệt ngẫu nhiên từ tập pixel bằng seed cố định.

        Tham số:
            X: Ma trận điểm ảnh 2D kích thước (N, D).

        Trả về:
            Ma trận K tâm cụm khởi tạo kích thước (K, D).
        """
        rng = np.random.default_rng(self.random_state)
        unique_rows = np.unique(X, axis=0)
        if len(unique_rows) < self.n_clusters:
            raise ValueError(
                f"Cannot initialize n_clusters={self.n_clusters} centroids from only "
                f"{len(unique_rows)} distinct color vectors."
            )

        selected_indices = rng.choice(len(unique_rows), size=self.n_clusters, replace=False)
        return unique_rows[selected_indices].astype(np.float64, copy=True)

    def _assign_clusters(
        self,
        X: np.ndarray,
        centroids: np.ndarray,
        chunk_size: int = 65536,
    ) -> Tuple[np.ndarray, float]:
        """
        Gán mỗi pixel vào tâm cụm gần nhất theo khoảng cách Euclidean bình phương và tính WCSS.

        Tham số:
            X: Ma trận điểm ảnh 2D kích thước (N, D).
            centroids: Ma trận tâm cụm hiện tại kích thước (K, D).
            chunk_size: Kích thước lô xử lý để tối ưu bộ nhớ khi ảnh có hàng triệu pixel.

        Trả về:
            Tuple gồm:
                - labels: Mảng 1D nhãn cụm thuộc [0, K-1] kích thước (N,).
                - inertia: Tổng bình phương khoảng cách nội cụm (WCSS) không âm.
        """
        n_samples = X.shape[0]
        labels = np.empty(n_samples, dtype=np.int64)
        total_inertia = 0.0
        centroids_f64 = np.asarray(centroids, dtype=np.float64)

        # Xử lý theo từng lô để tránh tràn RAM khi X lớn hơn vài trăm nghìn pixel
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            block = np.asarray(X[start:end], dtype=np.float64)
            diff = block[:, np.newaxis, :] - centroids_f64[np.newaxis, :, :]
            sq_dist = np.sum(diff * diff, axis=2)

            # np.argmin đảm bảo quy tắc chọn chỉ số cụm nhỏ nhất khi khoảng cách bằng nhau
            block_labels = np.argmin(sq_dist, axis=1)
            labels[start:end] = block_labels
            total_inertia += float(np.min(sq_dist, axis=1).sum())

        return labels, float(total_inertia)

    def _update_centroids(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """
        Cập nhật tọa độ từng tâm cụm bằng trung bình cộng các pixel được gán;
        nếu phát hiện cụm rỗng, chọn pixel xa nhất so với các tâm hợp lệ hiện có.

        Tham số:
            X: Ma trận điểm ảnh 2D kích thước (N, D).
            labels: Mảng nhãn cụm (N,).
            k: Số lượng cụm.

        Trả về:
            Ma trận K tâm cụm mới kích thước (K, D).
        """
        n_features = X.shape[1]
        new_centroids = np.zeros((k, n_features), dtype=np.float64)
        empty_clusters: List[int] = []
        valid_centroids: List[np.ndarray] = []

        for cluster_id in range(k):
            mask = labels == cluster_id
            if np.any(mask):
                centroid_vec = X[mask].mean(axis=0)
                new_centroids[cluster_id] = centroid_vec
                valid_centroids.append(centroid_vec)
            else:
                empty_clusters.append(cluster_id)

        # Khởi tạo lại cụm rỗng bằng điểm dữ liệu nằm xa nhất so với các tâm đang hoạt động
        if empty_clusters:
            self.empty_cluster_events_ += len(empty_clusters)
            for cluster_id in empty_clusters:
                logger.warning(
                    f"Phát hiện cụm rỗng #{cluster_id}; chọn điểm ảnh xa nhất so với tâm hiện có để tái khởi tạo."
                )
                if valid_centroids:
                    ref_matrix = np.vstack(valid_centroids)
                    # Tính khoảng cách tới tâm gần nhất theo từng lô cho ref_matrix
                    n_samples = X.shape[0]
                    min_sq_dists = np.empty(n_samples, dtype=np.float64)
                    for start in range(0, n_samples, 65536):
                        end = min(start + 65536, n_samples)
                        diff = X[start:end, np.newaxis, :] - ref_matrix[np.newaxis, :, :]
                        min_sq_dists[start:end] = np.min(np.sum(diff * diff, axis=2), axis=1)
                    farthest_idx = int(np.argmax(min_sq_dists))
                    recovered_vec = X[farthest_idx].copy()
                else:
                    recovered_vec = X[0].copy()

                new_centroids[cluster_id] = recovered_vec
                valid_centroids.append(recovered_vec)

        return new_centroids

    def fit(self, X: np.ndarray) -> "KMeansFromScratch":
        """
        Huấn luyện K-Means theo vòng lặp: khởi tạo -> gán -> cập nhật -> kiểm tra điều kiện dừng.

        Tham số:
            X: Ma trận điểm ảnh 2D kích thước (N, D).

        Trả về:
            Chính đối tượng KMeansFromScratch đã được huấn luyện (self).
        """
        X_arr = self._validate_matrix(X)
        if X_arr.shape[0] < self.n_clusters:
            raise ValueError(
                f"Number of samples ({X_arr.shape[0]}) must be >= n_clusters ({self.n_clusters})."
            )

        centroids = self._init_centroids(X_arr)
        self.history_ = []
        self.empty_cluster_events_ = 0
        self.n_iter_ = 0
        self.stop_reason_ = "max_iter"
        prev_labels: Optional[np.ndarray] = None

        logger.info(
            f"Bắt đầu huấn luyện KMeansFromScratch (K={self.n_clusters}, max_iter={self.max_iter}, "
            f"tol={self.tol}, N={X_arr.shape[0]:,})."
        )

        for iteration in range(1, self.max_iter + 1):
            labels, inertia = self._assign_clusters(X_arr, centroids)

            if prev_labels is None:
                label_change_ratio = 1.0
            else:
                label_change_ratio = float(np.mean(labels != prev_labels))

            new_centroids = self._update_centroids(X_arr, labels, self.n_clusters)
            centroid_shift = float(np.max(np.linalg.norm(new_centroids - centroids, axis=1)))

            self.history_.append(
                {
                    "iteration": int(iteration),
                    "inertia": float(inertia),
                    "centroid_shift": float(centroid_shift),
                    "label_change_ratio": float(label_change_ratio),
                }
            )

            centroids = new_centroids
            prev_labels = labels
            self.n_iter_ = iteration

            # Kiểm tra hội tụ theo độ dịch chuyển tâm cụm hoặc khi không còn pixel đổi nhãn
            if centroid_shift < self.tol or label_change_ratio == 0.0:
                self.stop_reason_ = "converged"
                break

        # Gán nhãn theo đúng tâm sẽ lưu để sai số đổi dtype không làm lệch nhãn tại ranh giới cụm.
        self.centroids = centroids.astype(np.float32)
        self.cluster_centers_ = self.centroids
        final_labels, final_inertia = self._assign_clusters(X_arr, self.centroids)
        self.labels_ = final_labels
        self.inertia_ = float(final_inertia)

        logger.info(
            f"Hoàn tất KMeansFromScratch sau {self.n_iter_} vòng lặp | "
            f"trạng thái dừng='{self.stop_reason_}' | WCSS={self.inertia_:,.2f}."
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Gán nhãn cụm gần nhất cho ma trận pixel mới dựa trên các tâm cụm đã học.

        Tham số:
            X: Ma trận điểm ảnh 2D kích thước (M, D).

        Trả về:
            Mảng 1D nhãn cụm kích thước (M,) kiểu int64.
        """
        if self.centroids is None:
            raise RuntimeError("KMeansFromScratch model must be fitted before calling predict().")

        X_arr = self._validate_matrix(X)
        if X_arr.shape[1] != self.centroids.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.centroids.shape[1]}, got {X_arr.shape[1]}."
            )

        labels, _ = self._assign_clusters(X_arr, self.centroids)
        return labels

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Huấn luyện mô hình trên tập dữ liệu X và trả về ngay mảng nhãn cụm tương ứng.

        Tham số:
            X: Ma trận điểm ảnh 2D kích thước (N, D).

        Trả về:
            Mảng 1D nhãn cụm của tập X.
        """
        self.fit(X)
        assert self.labels_ is not None
        return self.labels_

    def get_convergence_history(self) -> pd.DataFrame:
        """
        Chuyển đổi lịch sử các vòng lặp thành DataFrame để phân tích và vẽ đồ thị hội tụ.

        Trả về:
            DataFrame với các cột: ['iteration', 'inertia', 'centroid_shift', 'label_change_ratio'].
        """
        columns = ["iteration", "inertia", "centroid_shift", "label_change_ratio"]
        if not self.history_:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(self.history_, columns=columns)

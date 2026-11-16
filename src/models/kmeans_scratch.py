"""
=============================================================================
MODULE: K-MEANS TỪ ĐẦU (K-MEANS FROM SCRATCH & CONVERGENCE TRACKING)
=============================================================================
Mục đích (Đáp ứng đầy đủ Task 3 trong đề bài Practice Exercise 3.2):
- Khởi tạo K tâm cụm ngẫu nhiên từ tập pixel (không trùng lặp).
- Gán mỗi pixel vào tâm cụm gần nhất bằng khoảng cách Euclidean bình phương.
- Cập nhật tâm cụm bằng trung bình cộng các pixel thuộc cụm.
- Lặp đến khi hội tụ (centroid_shift <= tol) hoặc hết max_iter.
- Theo dõi lịch sử hội tụ (Inertia, centroid_shift, label_change_ratio).
=============================================================================
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from src.config import RANDOM_SEED, MAX_ITER, TOLERANCE
from src.utils.logger import get_logger

logger = get_logger("KMeansFromScratch")


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
        Lưu siêu tham số và khởi tạo các thuộc tính trạng thái.
        """
        if n_clusters < 1:
            raise ValueError(f"n_clusters phải >= 1, nhận được: {n_clusters}")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # Trạng thái sau khi fit()
        self.centroids: Optional[np.ndarray] = None   # (K, D)
        self.labels_: Optional[np.ndarray] = None      # (N,)
        self.inertia_: Optional[float] = None
        self.n_iter_: int = 0
        self.converged_: bool = False
        self.history_: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Bước 1: Chọn ngẫu nhiên K pixel phân biệt từ X làm tâm cụm khởi tạo.
        """
        n_samples = len(X)
        if self.n_clusters > n_samples:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) không được lớn hơn số pixel ({n_samples})."
            )

        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
        centroids = X[indices].astype(np.float64).copy()
        return centroids

    def _assign_clusters(
        self,
        X: np.ndarray,
        centroids: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Bước 2: Gán từng pixel vào tâm cụm gần nhất (khoảng cách Euclidean bình phương).
        Returns: (labels [N,], wcss float)
        """
        # Tính ||X_i - mu_j||^2 bằng broadcasting: (N,1,D) - (1,K,D) → (N,K,D)
        # Hiệu quả hơn: X^2 + mu^2 - 2*X*mu
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)       # (N, 1)
        C_sq = np.sum(centroids ** 2, axis=1, keepdims=True).T  # (1, K)
        cross = X @ centroids.T                              # (N, K)

        dist_sq = X_sq + C_sq - 2.0 * cross                # (N, K)
        # Clip âm do sai số số học
        dist_sq = np.maximum(dist_sq, 0.0)

        labels = np.argmin(dist_sq, axis=1)                 # (N,)
        # WCSS = tổng dist^2 của pixel tới tâm được gán
        wcss = float(np.sum(dist_sq[np.arange(len(X)), labels]))
        return labels, wcss

    def _update_centroids(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """
        Bước 3: Cập nhật tâm cụm = trung bình pixel trong cụm.
        Xử lý cụm rỗng: gán lại tâm đó bằng pixel xa nhất so với tất cả tâm hiện tại.
        """
        new_centroids = np.empty((k, X.shape[1]), dtype=np.float64)

        for c in range(k):
            mask = labels == c
            if np.any(mask):
                new_centroids[c] = X[mask].mean(axis=0)
            else:
                # Cụm rỗng: chọn pixel xa nhất so với các tâm hiện tại
                logger.warning(f"Cụm {c} rỗng! Khởi tạo lại tâm bằng pixel xa nhất.")
                dist_to_all_centroids = np.min(
                    np.sum((X[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2),
                    axis=1,
                )
                farthest_idx = np.argmax(dist_to_all_centroids)
                new_centroids[c] = X[farthest_idx].copy()

        return new_centroids

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "KMeansFromScratch":
        """
        Bước 4: Vòng lặp chính — gán → cập nhật → kiểm tra hội tụ.
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"X phải là 2D array (N, D), nhận được shape: {X.shape}")

        logger.info(
            f"Bắt đầu K-Means from scratch | K={self.n_clusters} | "
            f"max_iter={self.max_iter} | tol={self.tol} | seed={self.random_state}"
        )

        # --- Bước 1: Khởi tạo tâm ---
        centroids = self._init_centroids(X)
        prev_labels = np.full(len(X), -1, dtype=np.int32)
        self.history_ = []

        for iteration in range(1, self.max_iter + 1):
            # --- Bước 2: Gán cụm ---
            labels, wcss = self._assign_clusters(X, centroids)

            # Tỷ lệ điểm đổi nhãn
            label_change_ratio = float(np.mean(labels != prev_labels))

            # --- Bước 3: Cập nhật tâm ---
            new_centroids = self._update_centroids(X, labels, self.n_clusters)

            # Độ dịch chuyển tâm (max Euclidean shift)
            centroid_shift = float(np.max(
                np.sqrt(np.sum((new_centroids - centroids) ** 2, axis=1))
            ))

            # --- Ghi lịch sử ---
            self.history_.append({
                "iteration": iteration,
                "inertia": wcss,
                "centroid_shift": centroid_shift,
                "label_change_ratio": label_change_ratio,
            })

            if iteration % 10 == 0 or iteration == 1:
                logger.info(
                    f"  Iter {iteration:3d} | WCSS={wcss:.2f} | "
                    f"shift={centroid_shift:.6f} | changed={label_change_ratio:.4f}"
                )

            prev_labels = labels.copy()
            centroids = new_centroids

            # --- Kiểm tra dừng ---
            if centroid_shift <= self.tol or label_change_ratio == 0.0:
                self.converged_ = True
                logger.info(
                    f"Hội tụ tại vòng {iteration} | "
                    f"shift={centroid_shift:.6f} | changed={label_change_ratio:.4f}"
                )
                break
        else:
            self.converged_ = False
            logger.warning(
                f"Đạt giới hạn max_iter={self.max_iter} mà chưa hội tụ hoàn toàn."
            )

        # --- Gán nhãn cuối cùng bằng bộ tâm cuối ---
        final_labels, final_wcss = self._assign_clusters(X, centroids)

        self.centroids = centroids
        self.labels_ = final_labels.astype(np.int32)
        self.inertia_ = final_wcss
        self.n_iter_ = len(self.history_)

        logger.info(
            f"Kết thúc K-Means | n_iter={self.n_iter_} | "
            f"WCSS={self.inertia_:.2f} | converged={self.converged_}"
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Gán nhãn cụm cho tập dữ liệu mới dùng bộ tâm đã học.
        """
        if self.centroids is None:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước.")

        X = np.asarray(X, dtype=np.float64)
        labels, _ = self._assign_clusters(X, self.centroids)
        return labels.astype(np.int32)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Gọi fit() rồi trả về labels_."""
        self.fit(X)
        return self.labels_

    def get_convergence_history(self) -> pd.DataFrame:
        """
        Chuyển history_ thành DataFrame theo dõi hội tụ.
        Columns: iteration, inertia, centroid_shift, label_change_ratio
        """
        if not self.history_:
            return pd.DataFrame(
                columns=["iteration", "inertia", "centroid_shift", "label_change_ratio"]
            )
        return pd.DataFrame(self.history_)

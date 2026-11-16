"""
=============================================================================
MODULE: HUẤN LUYỆN K-MEANS VÀ TÌM K TỐI ƯU BẰNG ELBOW (SKLEARN K-MEANS)
=============================================================================
Mục đích:
- Sử dụng Scikit-Learn KMeans với thuật toán khởi tạo tối ưu k-means++.
- Quét số cụm K từ 2 đến 12 để tính quán tính WCSS (Inertia).
- Cài đặt thuật toán Kneedle (KneeLocator) tự động tìm điểm khuỷu tay (Elbow point).
  Fallback sang thuật toán hình học thuần NumPy nếu kneed chưa cài.
- Huấn luyện mô hình K-Means cuối cùng với số cụm K tối ưu.
=============================================================================
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from src.config import RANDOM_SEED, K_SEARCH_RANGE, MAX_ITER, N_INIT
from src.utils.logger import get_logger

logger = get_logger("SklearnKMeansTrainer")


class SklearnKMeansTrainer:
    """
    Lớp điều phối huấn luyện Scikit-Learn KMeans, quét Elbow và xác định K tối ưu.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        """Khởi tạo seed để đảm bảo tính tái lập."""
        self.random_state = random_state
        self.df_elbow_: Optional[pd.DataFrame] = None
        self.optimal_k_: Optional[int] = None
        self.final_model_: Optional[KMeans] = None

    def find_knee_point(
        self,
        k_values: List[int],
        inertias: List[float],
    ) -> int:
        """
        Tìm điểm khuỷu tay Elbow bằng thuật toán Kneedle.
        Ưu tiên dùng thư viện kneed; fallback thuật toán hình học NumPy.
        """
        # --- Thử dùng thư viện kneed ---
        try:
            from kneed import KneeLocator
            kl = KneeLocator(
                k_values, inertias,
                curve="convex",
                direction="decreasing",
                interp_method="interp1d",
            )
            if kl.knee is not None:
                logger.info(f"[kneed] Elbow point: K={kl.knee}")
                return int(kl.knee)
            logger.warning("kneed không tìm được điểm khuỷu, fallback sang NumPy.")
        except ImportError:
            logger.info("Thư viện kneed chưa cài. Dùng thuật toán hình học NumPy.")

        # --- Fallback: Thuật toán hình học (khoảng cách vuông góc) ---
        k_arr = np.array(k_values, dtype=np.float64)
        inertia_arr = np.array(inertias, dtype=np.float64)

        # Chuẩn hóa về [0, 1]
        k_norm = (k_arr - k_arr.min()) / (k_arr.max() - k_arr.min() + 1e-12)
        i_norm = (inertia_arr - inertia_arr.min()) / (inertia_arr.max() - inertia_arr.min() + 1e-12)

        # Đường thẳng nối điểm đầu và cuối
        # Phương trình: ax + by + c = 0
        x1, y1 = k_norm[0], i_norm[0]
        x2, y2 = k_norm[-1], i_norm[-1]
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        norm_vec = np.sqrt(a ** 2 + b ** 2) + 1e-12

        # Khoảng cách từ mỗi điểm tới đường thẳng
        distances = np.abs(a * k_norm + b * i_norm + c) / norm_vec
        knee_idx = int(np.argmax(distances))
        optimal_k = int(k_values[knee_idx])

        logger.info(f"[NumPy Geometric] Elbow point: K={optimal_k}")
        return optimal_k

    def search_elbow(
        self,
        X: np.ndarray,
        k_range: range = K_SEARCH_RANGE,
    ) -> Tuple[pd.DataFrame, int]:
        """
        Quét K từ k_range, tính WCSS mỗi K, tìm optimal_k bằng Elbow.
        Returns: (df_elbow, optimal_k)
        """
        k_values: List[int] = []
        inertias: List[float] = []

        logger.info(
            f"Bắt đầu quét Elbow | K ∈ [{k_range.start}, {k_range.stop - 1}]"
        )

        for k in k_range:
            model = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=N_INIT,
                max_iter=MAX_ITER,
                random_state=self.random_state,
            )
            model.fit(X)
            k_values.append(k)
            inertias.append(float(model.inertia_))
            logger.info(f"  K={k:2d} | WCSS={model.inertia_:.2f}")

        self.df_elbow_ = pd.DataFrame({"K": k_values, "Inertia": inertias})
        self.optimal_k_ = self.find_knee_point(k_values, inertias)

        logger.info(f"Kết quả quét Elbow — Optimal K = {self.optimal_k_}")
        return self.df_elbow_, self.optimal_k_

    def fit_final_model(
        self,
        X: np.ndarray,
        optimal_k: int = 5,
        n_init: int = N_INIT,
        max_iter: int = MAX_ITER,
    ) -> KMeans:
        """
        Huấn luyện mô hình K-Means cuối cùng với K tối ưu.
        Returns: đối tượng KMeans đã fit.
        """
        logger.info(
            f"Huấn luyện mô hình cuối cùng | K={optimal_k} | n_init={n_init}"
        )
        model = KMeans(
            n_clusters=optimal_k,
            init="k-means++",
            n_init=n_init,
            max_iter=max_iter,
            random_state=self.random_state,
        )
        model.fit(X)
        self.final_model_ = model

        logger.info(
            f"Mô hình cuối đã sẵn sàng | WCSS={model.inertia_:.2f} | "
            f"n_iter={model.n_iter_}"
        )
        return model

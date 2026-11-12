"""
=============================================================================
MODULE: HUẤN LUYỆN K-MEANS VÀ TÌM K TỐI ƯU BẰNG ELBOW (SKLEARN K-MEANS)
=============================================================================
Mục đích:
- Sử dụng Scikit-Learn KMeans với thuật toán khởi tạo k-means++.
- Quét dải giá trị K (mặc định 2..12) để ghi nhận đường cong quán tính WCSS.
- Tự động xác định điểm khuỷu tay (Elbow point) bằng Kneedle (thư viện kneed
  hoặc fallback khoảng cách vuông góc thuần NumPy khi kneed chưa được cài).
- Huấn luyện mô hình K-Means cuối cùng với giá trị optimal_k tìm được.
=============================================================================
"""

import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.config import K_SEARCH_RANGE, MAX_ITER, N_INIT, RANDOM_SEED
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SklearnKMeansTrainer:
    """
    Lớp điều phối huấn luyện Scikit-Learn KMeans, quét Elbow và xác định K tối ưu.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        """
        Khởi tạo trainer với seed ngẫu nhiên cố định để bảo đảm tính tái lập.

        Tham số:
            random_state: Seed điều khiển bộ sinh ngẫu nhiên của Scikit-Learn.
        """
        self.random_state: int = int(random_state)

    def find_knee_point(self, k_values: List[int], inertias: List[float]) -> int:
        """
        Xác định điểm khuỷu tay (Elbow point) trên đường cong WCSS theo thuật toán Kneedle.

        Tham số:
            k_values: Danh sách các giá trị K đã khảo sát.
            inertias: Danh sách giá trị quán tính WCSS tương ứng.

        Trả về:
            Giá trị K tối ưu tại điểm gập hình học lớn nhất.

        Ngoại lệ:
            ValueError: Nếu độ dài k_values và inertias không khớp hoặc ít hơn 3 điểm.
        """
        if len(k_values) != len(inertias):
            raise ValueError("k_values and inertias must have the same length.")
        if len(k_values) == 0:
            raise ValueError("k_values must not be empty.")
        if len(k_values) < 3:
            return int(k_values[0])

        pairs = sorted(zip(k_values, inertias), key=lambda pair: pair[0])
        k_sorted = np.array([p[0] for p in pairs], dtype=np.float64)
        inertia_sorted = np.array([p[1] for p in pairs], dtype=np.float64)

        # Thử dùng thư viện kneed nếu khả dụng trong môi trường Python hiện tại
        try:
            from kneed import KneeLocator

            locator = KneeLocator(
                k_sorted.tolist(),
                inertia_sorted.tolist(),
                curve="convex",
                direction="decreasing",
            )
            if locator.knee is not None:
                optimal_k = int(locator.knee)
                logger.info(f"Xác định điểm khuỷu tay bằng thư viện kneed: optimal_k={optimal_k}.")
                return optimal_k
        except ImportError:
            logger.info(
                "Không tìm thấy thư viện 'kneed', sử dụng thuật toán hình học Kneedle thuần NumPy."
            )
        except Exception as exc:
            logger.warning(
                f"KneeLocator gặp ngoại lệ ({exc}), chuyển sang tính khoảng cách vuông góc thuần NumPy."
            )

        # Thuật toán hình học thuần NumPy: chuẩn hóa [0, 1] và đo khoảng cách tới dây cung
        k_span = float(k_sorted.max() - k_sorted.min())
        i_span = float(inertia_sorted.max() - inertia_sorted.min())
        if k_span <= 0.0 or i_span <= 1e-12:
            logger.warning("Đường cong WCSS gần như phẳng, trả về K nhỏ nhất trong dải tìm kiếm.")
            return int(k_sorted[0])

        x_norm = (k_sorted - k_sorted.min()) / k_span
        y_norm = (inertia_sorted - inertia_sorted.min()) / i_span

        # Đường thẳng nối điểm đầu (x=0, y=y_norm[0]) và điểm cuối (x=1, y=y_norm[-1])
        p1 = np.array([x_norm[0], y_norm[0]], dtype=np.float64)
        p2 = np.array([x_norm[-1], y_norm[-1]], dtype=np.float64)
        line_vec = p2 - p1
        line_len = float(np.linalg.norm(line_vec))
        if line_len <= 1e-12:
            return int(k_sorted[0])

        distances = np.abs(
            line_vec[0] * (p1[1] - y_norm) - (p1[0] - x_norm) * line_vec[1]
        ) / line_len
        best_idx = int(np.argmax(distances))
        optimal_k = int(k_sorted[best_idx])

        logger.info(
            f"Xác định điểm khuỷu tay bằng thuật toán Kneedle thuần NumPy: optimal_k={optimal_k} "
            f"(khoảng cách chuẩn hóa={distances[best_idx]:.4f})."
        )
        return optimal_k

    def search_elbow(
        self,
        X: np.ndarray,
        k_range: range = K_SEARCH_RANGE,
    ) -> Tuple[pd.DataFrame, int]:
        """
        Quét dải tham số K để tính WCSS cho từng cấu hình và tự động chọn optimal_k.

        Tham số:
            X: Ma trận điểm ảnh 2D kích thước (N, D).
            k_range: Dải số cụm K cần khảo sát (mặc định 2..12).

        Trả về:
            Tuple gồm:
                - df_elbow: DataFrame với các cột ['K', 'Inertia', 'N_Iter', 'Fit_Time_Sec'].
                - optimal_k: Giá trị số cụm tối ưu xác định bởi find_knee_point().
        """
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2 or X_arr.shape[0] == 0:
            raise ValueError("Input X must be a non-empty 2D array.")
        if not np.isfinite(X_arr).all():
            raise ValueError("Input X contains NaN or Inf values.")

        n_unique = int(len(np.unique(X_arr, axis=0)))
        valid_ks = [int(k) for k in k_range if 1 <= int(k) <= min(X_arr.shape[0], n_unique)]
        if not valid_ks:
            raise ValueError(
                f"No valid K in {list(k_range)} for dataset with {n_unique} distinct color vectors."
            )

        logger.info(
            f"Bắt đầu quét Elbow Method trên dải K={valid_ks[0]}..{valid_ks[-1]} "
            f"(N={X_arr.shape[0]:,} pixels, seed={self.random_state})."
        )

        records: List[Dict[str, Any]] = []
        for k in valid_ks:
            t0 = time.perf_counter()
            model = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=N_INIT,
                max_iter=MAX_ITER,
                random_state=self.random_state,
            )
            model.fit(X_arr)
            elapsed = time.perf_counter() - t0
            records.append(
                {
                    "K": int(k),
                    "Inertia": float(model.inertia_),
                    "N_Iter": int(model.n_iter_),
                    "Fit_Time_Sec": round(float(elapsed), 4),
                }
            )
            logger.info(
                f"  -> K={k:2d} | Inertia={model.inertia_:,.2f} | "
                f"vòng lặp={model.n_iter_} | thời gian={elapsed:.3f}s"
            )

        df_elbow = pd.DataFrame(records, columns=["K", "Inertia", "N_Iter", "Fit_Time_Sec"])
        optimal_k = self.find_knee_point(
            df_elbow["K"].tolist(),
            df_elbow["Inertia"].tolist(),
        )
        return df_elbow, int(optimal_k)

    def fit_final_model(
        self,
        X: np.ndarray,
        optimal_k: int = 5,
        n_init: int = N_INIT,
        max_iter: int = MAX_ITER,
    ) -> KMeans:
        """
        Huấn luyện mô hình Scikit-Learn KMeans chính thức với số cụm optimal_k.

        Tham số:
            X: Ma trận điểm ảnh 2D kích thước (N, D).
            optimal_k: Số cụm mục tiêu (mặc định 5).
            n_init: Số lần khởi tạo k-means++ độc lập để chọn nghiệm tốt nhất.
            max_iter: Số vòng lặp tối đa.

        Trả về:
            Đối tượng sklearn.cluster.KMeans đã huấn luyện xong.
        """
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2 or X_arr.shape[0] < optimal_k:
            raise ValueError(
                f"Dataset shape {X_arr.shape} is invalid for optimal_k={optimal_k}."
            )
        if not np.isfinite(X_arr).all():
            raise ValueError("Input X contains NaN or Inf values.")

        t0 = time.perf_counter()
        model = KMeans(
            n_clusters=int(optimal_k),
            init="k-means++",
            n_init=int(n_init),
            max_iter=int(max_iter),
            random_state=self.random_state,
        )
        model.fit(X_arr)
        elapsed = time.perf_counter() - t0

        logger.info(
            f"Đã huấn luyện mô hình KMeans cuối cùng (K={optimal_k}, n_init={n_init}) "
            f"trong {elapsed:.3f}s | Inertia={model.inertia_:,.2f} | số vòng lặp={model.n_iter_}."
        )
        return model

"""
=============================================================================
MODULE: HUẤN LUYỆN K-MEANS VÀ TÌM K TỐI ƯU BẰNG ELBOW (SKLEARN K-MEANS)
=============================================================================
Mục đích:
- Sử dụng Scikit-Learn KMeans với thuật toán khởi tạo tối ưu k-means++.
- Quét số cụm K từ 2 đến 12 để tính quán tính WCSS (Inertia) và điểm Silhouette.
- Cài đặt thuật toán Kneedle (KneeLocator) tự động tìm điểm khuỷu tay (Elbow point).
- Huấn luyện mô hình K-Means cuối cùng với số cụm K tối ưu (K=5).
=============================================================================
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from src.config import RANDOM_SEED, K_SEARCH_RANGE, MAX_ITER, N_INIT


class SklearnKMeansTrainer:
    """
    Lớp điều phối huấn luyện Scikit-Learn KMeans, quét Elbow và xác định K tối ưu.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        """
        Ý LÀM:
        Khởi tạo seed ngẫu nhiên để đảm bảo tính tái lập (reproducibility).
        """
        pass

    def find_knee_point(self, k_values: List[int], inertias: List[float]) -> int:
        """
        Ý LÀM (Thuật toán Kneedle xác định điểm khuỷu tay Elbow):
        1. Kiểm tra nếu có thư viện kneed: sử dụng KneeLocator(curve="convex", direction="decreasing").
        2. Nếu không có thư viện kneed: áp dụng thuật toán hình học thuần NumPy:
           - Chuẩn hóa K và Inertia về thang đo đoạn [0, 1].
           - Vẽ đường thẳng cát tuyến nối điểm đầu (K_min, Inertia_max) và điểm cuối (K_max, Inertia_min).
           - Tính khoảng cách vuông góc từ mỗi điểm trên đường cong tới đường thẳng cát tuyến.
           - Chọn giá trị K có khoảng cách vuông góc xa nhất làm điểm khuỷu tay (Elbow).
        3. Trả về giá trị K tối ưu tìm được.
        """
        pass

    def search_elbow(
        self,
        X: np.ndarray,
        k_range: range = K_SEARCH_RANGE,
    ) -> Tuple[pd.DataFrame, int]:
        """
        Ý LÀM:
        1. Duyệt qua từng giá trị K trong k_range (ví dụ từ 2 đến 12).
        2. Với mỗi K:
           - Khởi tạo KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=self.random_state).
           - Gọi fit(X) trên ma trận điểm ảnh.
           - Ghi nhận giá trị WCSS (Inertia).
        3. Tạo DataFrame bảng kết quả [K, Inertia].
        4. Gọi hàm find_knee_point để tự động xác định K tối ưu (optimal_k).
        5. Trả về tuple gồm: DataFrame kết quả quét K và giá trị optimal_k (ví dụ K=5).
        """
        pass

    def fit_final_model(
        self,
        X: np.ndarray,
        optimal_k: int = 5,
        n_init: int = N_INIT,
        max_iter: int = MAX_ITER,
    ) -> KMeans:
        """
        Ý LÀM:
        1. Khởi tạo KMeans(n_clusters=optimal_k, init='k-means++', n_init=n_init, max_iter=max_iter).
        2. Gọi fit(X) để huấn luyện mô hình phân cụm cuối cùng.
        3. Trả về đối tượng KMeans đã huấn luyện hoàn chỉnh.
        """
        pass

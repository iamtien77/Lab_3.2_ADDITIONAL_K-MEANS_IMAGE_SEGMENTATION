"""
=============================================================================
MODULE: K-MEANS TỪ ĐẦU (K-MEANS FROM SCRATCH & CONVERGENCE TRACKING)
=============================================================================
Mục đích (Đáp ứng đầy đủ Task 3 trong đề bài Practice Exercise 3.2):
- Khởi tạo K tâm cụm ngẫu nhiên đại diện cho màu trung bình của từng cụm:
  "Initialize K random centroids, each representing the mean color of a cluster."
- Gán mỗi pixel về tâm cụm gần nhất dựa trên khoảng cách Euclidean:
  "Assign each pixel to the nearest centroid based on Euclidean distance."
- Cập nhật tâm cụm bằng trung bình cộng các pixel thuộc cụm:
  "Update the centroids as the mean of the pixels assigned to each cluster."
- Lặp lại các bước gán và cập nhật cho đến khi hội tụ:
  "Repeat the assignment and update steps until convergence."
- Theo dõi lịch sử hội tụ (Inertia, centroid shift, label change ratio) qua từng vòng lặp.
=============================================================================
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from src.config import RANDOM_SEED, MAX_ITER, TOLERANCE


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
        Ý LÀM:
        1. Lưu các siêu tham số: số cụm K (n_clusters), số vòng lặp tối đa (max_iter),
           ngưỡng dừng hội tụ (tol), seed ngẫu nhiên (random_state).
        2. Khởi tạo các thuộc tính lưu trạng thái:
           - centroids: Tọa độ tâm cụm kích thước (K, D).
           - labels_: Nhãn cụm của từng điểm ảnh (N,).
           - inertia_: Quán tính tổng bình phương khoảng cách WCSS.
           - n_iter_: Số vòng lặp thực tế đã chạy trước khi dừng.
           - history_: Danh sách lưu chi tiết lịch sử hội tụ qua từng vòng lặp.
        """
        pass

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Ý LÀM (Task 3 - Bước 1: Khởi tạo K tâm cụm ngẫu nhiên):
        1. Sử dụng bộ sinh số ngẫu nhiên np.random.default_rng(self.random_state).
        2. Chọn ngẫu nhiên K chỉ số mẫu phân biệt (without replacement) từ tập dữ liệu X.
        3. Trích xuất K vector điểm ảnh tương ứng làm tọa độ tâm cụm ban đầu.
        4. Trả về ma trận tâm cụm kích thước (K, 3).
        """
        pass

    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Ý LÀM (Task 3 - Bước 2: Gán pixel vào tâm cụm gần nhất theo Euclidean distance):
        1. Tính khoảng cách bình phương Euclidean từ mỗi pixel tới K tâm cụm:
           distances = ||X_i - Centroid_k||^2.
        2. Xác định nhãn cụm gần nhất cho mỗi pixel bằng argmin theo trục tâm cụm.
        3. Tính WCSS (Inertia) = tổng khoảng cách bình phương từ mỗi pixel tới tâm cụm được gán.
        4. Trả về tuple gồm: mảng nhãn cụm (labels) và giá trị quán tính WCSS (inertia).
        """
        pass

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
        """
        Ý LÀM (Task 3 - Bước 3: Cập nhật tâm cụm bằng trung bình cộng các pixel):
        1. Duyệt qua từng cụm c từ 0 đến k - 1:
        2. Lọc tất cả các pixel có nhãn bằng c: mask = (labels == c).
        3. Nếu cụm c có chứa điểm ảnh: tính mean(axis=0) của các pixel đó để làm tâm cụm mới.
        4. Nếu cụm c bị rỗng (empty cluster): gán tâm cụm mới bằng điểm dữ liệu có khoảng cách xa nhất
           so với các tâm cụm hiện tại để tránh suy biến.
        5. Trả về ma trận K tâm cụm mới cập nhật.
        """
        pass

    def fit(self, X: np.ndarray) -> "KMeansFromScratch":
        """
        Ý LÀM (Task 3 - Bước 4: Lặp lại gán & cập nhật cho tới khi hội tụ):
        1. Khởi tạo K tâm cụm ban đầu bằng _init_centroids(X).
        2. Khởi tạo vòng lặp từ 1 đến max_iter:
           a. Gọi _assign_clusters để gán nhãn và tính Inertia hiện tại.
           b. Tính tỷ lệ điểm bị đổi nhãn so với vòng lặp trước (label_change_ratio).
           c. Gọi _update_centroids để cập nhật tọa độ tâm cụm mới.
           d. Tính độ dịch chuyển tâm cụm (centroid_shift) bằng chuẩn khoảng cách Euclidean.
           e. Ghi nhận thông tin vòng lặp vào history_:
              [iteration, inertia, centroid_shift, label_change_ratio].
           f. Kiểm tra điều kiện dừng hội tụ sớm:
              - Nếu centroid_shift < self.tol hoặc label_change_ratio == 0: dừng lặp.
        3. Lưu kết quả cuối cùng vào self.centroids, self.labels_, self.inertia_, self.n_iter_.
        4. Trả về chính đối tượng (self).
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Ý LÀM:
        1. Tính khoảng cách Euclidean từ mỗi điểm ảnh trong X tới self.centroids.
        2. Trả về nhãn của tâm cụm gần nhất cho từng điểm ảnh.
        """
        pass

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Ý LÀM:
        1. Gọi fit(X).
        2. Trả về self.labels_.
        """
        pass

    def get_convergence_history(self) -> pd.DataFrame:
        """
        Ý LÀM:
        Chuyển đổi danh sách self.history_ thành DataFrame bảng theo dõi tiến trình hội tụ.
        """
        pass

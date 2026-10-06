"""
KIỂM THỬ ĐƠN VỊ: THUẬT TOÁN K-MEANS (TEST K-MEANS FROM SCRATCH & SKLEARN)
"""

import unittest
import numpy as np


class TestKMeansModels(unittest.TestCase):
    """
    Ý LÀM:
    1. test_kmeans_scratch_shape: Kiểm tra K-Means thuần huấn luyện ra đúng K tâm cụm (K, 3).
    2. test_kmeans_scratch_labels: Kiểm tra nhãn cụm nằm trong đoạn [0, K-1].
    3. test_kmeans_scratch_convergence: Kiểm tra giá trị WCSS giảm dần qua các vòng lặp.
    4. test_sklearn_kmeans_fit: Kiểm tra Scikit-Learn KMeans huấn luyện thành công trên dữ liệu pixel mẫu.
    5. test_elbow_kneedle: Kiểm tra thuật toán tìm điểm khuỷu tay xác định đúng điểm gập hình học.
    """
    pass


if __name__ == "__main__":
    unittest.main()

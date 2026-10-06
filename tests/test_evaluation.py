"""
KIỂM THỬ ĐƠN VỊ: ĐÁNH GIÁ CHẤT LƯỢNG (TEST EVALUATION METRICS)
"""

import unittest
import numpy as np


class TestEvaluationMetrics(unittest.TestCase):
    """
    Ý LÀM:
    1. test_compute_wcss_positive: Kiểm tra giá trị quán tính WCSS luôn không âm (>= 0).
    2. test_silhouette_range: Kiểm tra hệ số Silhouette luôn nằm trong đoạn [-1.0, 1.0].
    3. test_ari_identical: Kiểm tra chỉ số ARI giữa hai bộ nhãn giống hệt nhau bằng 1.0.
    """
    pass


if __name__ == "__main__":
    unittest.main()

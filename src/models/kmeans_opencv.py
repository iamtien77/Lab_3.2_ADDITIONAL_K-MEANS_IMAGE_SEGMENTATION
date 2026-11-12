"""
=============================================================================
MODULE: KIỂM ĐỊNH CHÉO VỚI OPENCV K-MEANS (OPENCV CROSS-LIBRARY VALIDATION)
=============================================================================
Mục đích:
- Huấn luyện K-Means bằng thư viện OpenCV (cv2.kmeans) với cờ KMEANS_PP_CENTERS.
- Đối chiếu kết quả giữa Scikit-Learn và OpenCV thông qua thuật toán Hungarian
  (linear_sum_assignment) để căn chỉnh hoán vị nhãn cụm tối ưu trước khi so khớp.
=============================================================================
"""

import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.config import RANDOM_SEED
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OpenCVKMeansTrainer:
    """
    Lớp huấn luyện và đối chiếu chéo thuật toán K-Means bằng thư viện OpenCV.
    """

    def __init__(self):
        """
        Khởi tạo trainer và xác nhận sự sẵn có của hàm cv2.kmeans trong môi trường.
        """
        if not hasattr(cv2, "kmeans"):
            raise RuntimeError("OpenCV cv2.kmeans is not available in the current environment.")
        self.last_fit_time_sec_: Optional[float] = None
        logger.info("Đã khởi tạo OpenCVKMeansTrainer sẵn sàng kiểm định chéo.")

    def train(
        self,
        X: np.ndarray,
        k: int = 5,
        attempts: int = 10,
        max_iter: int = 300,
        epsilon: float = 1.0,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Huấn luyện K-Means bằng cv2.kmeans với khởi tạo KMEANS_PP_CENTERS.

        Tham số:
            X: Ma trận pixel 2D kích thước (N, D).
            k: Số cụm mục tiêu.
            attempts: Số lần chạy lại độc lập với các bộ tâm khác nhau.
            max_iter: Số vòng lặp tối đa trong mỗi lần thử.
            epsilon: Ngưỡng dịch chuyển tâm cụm tối thiểu để dừng sớm.

        Trả về:
            Tuple gồm:
                - compactness: Tổng bình phương khoảng cách (tương đương Inertia/WCSS).
                - labels: Mảng nhãn cụm 1D (N,) kiểu int64.
                - centers: Ma trận tọa độ tâm cụm (K, D) kiểu float32.
        """
        data = np.ascontiguousarray(X, dtype=np.float32)
        if data.ndim != 2 or data.shape[0] < k or k < 1:
            raise ValueError(f"Invalid input shape {data.shape} for k={k}.")
        if not np.isfinite(data).all():
            raise ValueError("Input X contains NaN or Inf values.")

        # Cố định seed của OpenCV trước khi phân cụm để tăng tính tái lập
        if hasattr(cv2, "setRNGSeed"):
            cv2.setRNGSeed(RANDOM_SEED)

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            int(max_iter),
            float(epsilon),
        )
        t0 = time.perf_counter()
        compactness, labels, centers = cv2.kmeans(
            data,
            int(k),
            None,
            criteria,
            int(attempts),
            cv2.KMEANS_PP_CENTERS,
        )
        fit_time = max(time.perf_counter() - t0, 1e-6)
        self.last_fit_time_sec_ = float(fit_time)

        flat_labels = labels.flatten().astype(np.int64)
        centers_f32 = centers.astype(np.float32)
        logger.info(
            f"Đã huấn luyện cv2.kmeans (K={k}, attempts={attempts}) trong {fit_time:.4f}s | Compactness={float(compactness):,.2f}."
        )
        return float(compactness), flat_labels, centers_f32

    def compare_with_sklearn(
        self,
        sklearn_labels: np.ndarray,
        opencv_labels: np.ndarray,
        sklearn_inertia: float,
        opencv_compactness: float,
        sklearn_time_sec: Optional[float] = None,
        opencv_time_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Căn chỉnh hoán vị nhãn tối ưu bằng thuật toán Hungarian và đối chiếu kết quả hai thư viện.

        Tham số:
            sklearn_labels: Mảng nhãn 1D từ Scikit-Learn KMeans.
            opencv_labels: Mảng nhãn 1D từ OpenCV cv2.kmeans.
            sklearn_inertia: Giá trị WCSS của Scikit-Learn.
            opencv_compactness: Giá trị compactness WCSS của OpenCV.
            sklearn_time_sec: Thời gian huấn luyện mô hình Scikit-Learn (giây, tùy chọn).
            opencv_time_sec: Thời gian huấn luyện mô hình OpenCV (giây, mặc định lấy lượt train gần nhất).

        Trả về:
            Từ điển báo cáo đối chiếu gồm chênh lệch quán tính, tỷ lệ trùng khớp nhãn, ARI, NMI và thời gian.
        """
        sk_arr = np.asarray(sklearn_labels, dtype=np.int64).reshape(-1)
        cv_arr = np.asarray(opencv_labels, dtype=np.int64).reshape(-1)
        if sk_arr.size == 0 or sk_arr.shape != cv_arr.shape:
            raise ValueError("sklearn_labels and opencv_labels must be non-empty 1D arrays of equal length.")

        unique_sk = np.unique(sk_arr)
        unique_cv = np.unique(cv_arr)
        max_k = int(max(unique_sk.max(), unique_cv.max()) + 1)

        # Xây dựng ma trận đồng xuất hiện và giải bài toán ghép tối ưu bằng Hungarian algorithm
        contingency = np.zeros((max_k, max_k), dtype=np.int64)
        np.add.at(contingency, (cv_arr, sk_arr), 1)
        row_ind, col_ind = linear_sum_assignment(-contingency)
        label_mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

        remapped_cv = np.vectorize(lambda lbl: label_mapping.get(int(lbl), int(lbl)))(cv_arr)
        matched_pixels = int(np.sum(remapped_cv == sk_arr))
        agreement_pct = round((matched_pixels / float(sk_arr.size)) * 100.0, 2)

        if abs(sklearn_inertia) <= 1e-12:
            rel_diff_pct = 0.0 if abs(opencv_compactness) <= 1e-12 else 100.0
        else:
            rel_diff_pct = round(
                (abs(float(sklearn_inertia) - float(opencv_compactness)) / float(sklearn_inertia)) * 100.0,
                4,
            )

        ari = round(float(adjusted_rand_score(sk_arr, cv_arr)), 4)
        nmi = round(float(normalized_mutual_info_score(sk_arr, cv_arr)), 4)

        resolved_cv_time = (
            float(opencv_time_sec) if opencv_time_sec is not None else self.last_fit_time_sec_
        )
        report: Dict[str, Any] = {
            "sklearn_inertia": round(float(sklearn_inertia), 2),
            "opencv_compactness": round(float(opencv_compactness), 2),
            "inertia_relative_diff_pct": rel_diff_pct,
            "label_agreement_pct": agreement_pct,
            "ari": ari,
            "nmi": nmi,
            "optimal_label_mapping": {str(k): v for k, v in label_mapping.items()},
        }
        if resolved_cv_time is not None:
            report["opencv_fit_time_sec"] = round(resolved_cv_time, 4)
        if sklearn_time_sec is not None and sklearn_time_sec > 0:
            report["sklearn_fit_time_sec"] = round(float(sklearn_time_sec), 4)
            if resolved_cv_time is not None and resolved_cv_time > 0:
                report["opencv_vs_sklearn_speedup"] = round(
                    float(sklearn_time_sec) / resolved_cv_time, 2
                )

        timing_msg = ""
        if resolved_cv_time is not None:
            timing_msg = f", OpenCV_t={resolved_cv_time:.4f}s"
            if sklearn_time_sec is not None:
                timing_msg += f", Sklearn_t={float(sklearn_time_sec):.4f}s"

        logger.info(
            f"Đối chiếu Scikit-Learn vs OpenCV: trùng khớp nhãn={agreement_pct}%, "
            f"chênh lệch WCSS={rel_diff_pct}%, ARI={ari}, NMI={nmi}{timing_msg}."
        )
        return report

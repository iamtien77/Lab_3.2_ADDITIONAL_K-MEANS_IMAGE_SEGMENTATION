"""
=============================================================================
MODULE: KIỂM ĐỊNH CHÉO VỚI OPENCV K-MEANS (OPENCV CROSS-LIBRARY VALIDATION)
=============================================================================
Mục đích:
- Huấn luyện K-Means bằng OpenCV (cv2.kmeans) với cờ KMEANS_PP_CENTERS.
- So sánh kết quả đối đầu giữa scikit-learn và OpenCV:
  * Độ trùng khớp nhãn cụm (Label agreement %).
  * Chênh lệch tương đối về quán tính WCSS (Inertia relative difference %).
=============================================================================
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("OpenCVKMeansTrainer")


class OpenCVKMeansTrainer:
    """
    Lớp huấn luyện K-Means sử dụng thư viện OpenCV.
    """

    def __init__(self):
        """Khởi tạo trainer và kiểm tra sự sẵn có của module cv2."""
        try:
            import cv2 as _cv2
            self._cv2 = _cv2
            logger.info(f"OpenCV sẵn sàng (version={_cv2.__version__})")
        except ImportError:
            self._cv2 = None
            logger.warning(
                "OpenCV không khả dụng. OpenCVKMeansTrainer sẽ không hoạt động. "
                "Cài đặt: pip install opencv-python"
            )

    def _require_cv2(self):
        if self._cv2 is None:
            raise ImportError(
                "Cần OpenCV để sử dụng OpenCVKMeansTrainer. "
                "Cài đặt: pip install opencv-python"
            )

    def train(
        self,
        X: np.ndarray,
        k: int = 5,
        attempts: int = 10,
        max_iter: int = 300,
        epsilon: float = 1.0,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Chạy cv2.kmeans trên ma trận pixel X.
        Returns: (compactness, labels [N,], centers [K, 3])
        """
        self._require_cv2()
        cv2 = self._cv2

        # OpenCV yêu cầu float32 liên tục trong bộ nhớ
        X_f32 = np.ascontiguousarray(X, dtype=np.float32)

        # Tiêu chí dừng: EPS + MAX_ITER
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            max_iter,
            epsilon,
        )

        compactness, labels, centers = cv2.kmeans(
            data=X_f32,
            K=k,
            bestLabels=None,
            criteria=criteria,
            attempts=attempts,
            flags=cv2.KMEANS_PP_CENTERS,
        )

        # Làm phẳng nhãn từ (N, 1) → (N,)
        labels_flat = labels.flatten().astype(np.int32)

        logger.info(
            f"OpenCV K-Means | K={k} | compactness={compactness:.2f}"
        )
        return float(compactness), labels_flat, centers.astype(np.float32)

    def compare_with_sklearn(
        self,
        sklearn_labels: np.ndarray,
        opencv_labels: np.ndarray,
        sklearn_inertia: float,
        opencv_compactness: float,
    ) -> Dict[str, Any]:
        """
        Đối chiếu kết quả OpenCV vs Scikit-Learn.
        Căn chỉnh hoán vị nhãn tối ưu (Hungarian matching) trước khi tính label agreement.
        Returns: dict báo cáo đối chiếu.
        """
        from scipy.optimize import linear_sum_assignment

        sklearn_labels = np.asarray(sklearn_labels, dtype=np.int32)
        opencv_labels = np.asarray(opencv_labels, dtype=np.int32)

        n = len(sklearn_labels)
        k_sklearn = int(sklearn_labels.max()) + 1
        k_opencv = int(opencv_labels.max()) + 1
        k = max(k_sklearn, k_opencv)

        # --- Tính chênh lệch quán tính ---
        inertia_diff_pct = abs(sklearn_inertia - opencv_compactness) / (sklearn_inertia + 1e-12) * 100.0

        # --- Căn chỉnh hoán vị nhãn (Confusion Matrix → Hungarian) ---
        confusion = np.zeros((k, k), dtype=np.int64)
        for sk_label, cv_label in zip(sklearn_labels, opencv_labels):
            if sk_label < k and cv_label < k:
                confusion[sk_label, cv_label] += 1

        # Giải bài toán gán (maximize trùng khớp = minimize âm)
        row_ind, col_ind = linear_sum_assignment(-confusion)
        label_map = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}

        # Áp dụng hoán vị nhãn OpenCV
        opencv_labels_aligned = np.array(
            [label_map.get(lbl, lbl) for lbl in opencv_labels], dtype=np.int32
        )

        # --- Tính tỷ lệ trùng khớp ---
        agreement_pct = float(np.mean(sklearn_labels == opencv_labels_aligned) * 100.0)

        report = {
            "sklearn_inertia": sklearn_inertia,
            "opencv_compactness": opencv_compactness,
            "inertia_diff_pct": round(inertia_diff_pct, 4),
            "label_agreement_pct": round(agreement_pct, 2),
            "label_permutation": label_map,
        }

        logger.info(
            f"OpenCV vs Sklearn | WCSS Δ={inertia_diff_pct:.2f}% | "
            f"Label agreement={agreement_pct:.1f}%"
        )
        return report

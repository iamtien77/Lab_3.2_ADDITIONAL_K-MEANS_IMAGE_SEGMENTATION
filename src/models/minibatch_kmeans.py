"""
=============================================================================
MODULE: PHÂN TÍCH ĐÁNH ĐỔI VỚI MINI-BATCH K-MEANS (SPEED / QUALITY TRADE-OFF)
=============================================================================
Mục đích:
- So sánh hiệu năng và chất lượng phân cụm giữa Full K-Means chuẩn và
  MiniBatchKMeans trên nhiều kích thước batch (256, 512, 1024, 2048, 4096).
- Đo lường speedup thời gian huấn luyện, tỷ lệ tăng WCSS và độ tương đồng
  nhãn phân cụm thông qua ARI và NMI.
=============================================================================
"""

import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.config import MAX_ITER, N_INIT, RANDOM_SEED
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MiniBatchKMeansTrainer:
    """
    Lớp thực nghiệm đánh giá đánh đổi tốc độ/chất lượng của MiniBatchKMeans so với Full K-Means.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        """
        Khởi tạo trainer với seed cố định.

        Tham số:
            random_state: Seed ngẫu nhiên cho MiniBatchKMeans và baseline Full KMeans.
        """
        self.random_state: int = int(random_state)

    def compare_batch_sizes(
        self,
        X: np.ndarray,
        k: int,
        full_kmeans_model: KMeans,
        batch_sizes: List[int] = [256, 512, 1024, 2048, 4096],
    ) -> pd.DataFrame:
        """
        Đo thời gian, tốc độ tăng tốc, tỷ lệ tăng quán tính và ARI/NMI qua nhiều batch_size.

        Tham số:
            X: Ma trận pixel 2D kích thước (N, D).
            k: Số lượng cụm K.
            full_kmeans_model: Mô hình Full KMeans làm mốc so sánh chất lượng.
            batch_sizes: Danh sách các kích thước batch cần khảo sát.

        Trả về:
            DataFrame tổng hợp so sánh Full KMeans và từng cấu hình MiniBatchKMeans.
            Cột batch_size giữ giá trị yêu cầu; effective_batch_size là kích thước thực dùng.
        """
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2 or X_arr.shape[0] < k:
            raise ValueError(f"Invalid dataset shape {X_arr.shape} for k={k}.")
        if not np.isfinite(X_arr).all():
            raise ValueError("Input X contains NaN or Inf values.")

        # Đo lại thời gian và kết quả của Full KMeans trên cùng siêu tham số làm mốc đồng nhất
        t0_full = time.perf_counter()
        timed_full = KMeans(
            n_clusters=int(k),
            init=getattr(full_kmeans_model, "init", "k-means++"),
            n_init=getattr(full_kmeans_model, "n_init", N_INIT),
            max_iter=int(getattr(full_kmeans_model, "max_iter", MAX_ITER)),
            tol=float(getattr(full_kmeans_model, "tol", 1e-4)),
            random_state=getattr(full_kmeans_model, "random_state", self.random_state),
        )
        timed_full.fit(X_arr)
        full_time = max(time.perf_counter() - t0_full, 1e-6)

        baseline_labels = timed_full.labels_
        baseline_inertia = float(timed_full.inertia_)

        rows: List[Dict[str, Any]] = [
            {
                "model": "Full KMeans",
                "batch_size": int(X_arr.shape[0]),
                "effective_batch_size": int(X_arr.shape[0]),
                "fit_time_sec": round(float(full_time), 4),
                "speedup": 1.0,
                "inertia": round(baseline_inertia, 2),
                "inertia_increase_pct": 0.0,
                "ari": 1.0,
                "nmi": 1.0,
            }
        ]

        logger.info(
            f"Bắt đầu benchmark MiniBatchKMeans (K={k}, N={X_arr.shape[0]:,}, "
            f"Full KMeans={full_time:.3f}s, WCSS={baseline_inertia:,.2f})."
        )

        for bs in batch_sizes:
            effective_bs = min(int(bs), X_arr.shape[0])
            batch_label = (
                f"b={effective_bs}, requested={int(bs)}"
                if effective_bs != int(bs)
                else f"b={effective_bs}"
            )
            t0 = time.perf_counter()
            mb_model = MiniBatchKMeans(
                n_clusters=int(k),
                batch_size=effective_bs,
                n_init=N_INIT,
                max_iter=MAX_ITER,
                random_state=self.random_state,
            )
            mb_labels = mb_model.fit_predict(X_arr)
            mb_time = max(time.perf_counter() - t0, 1e-6)
            mb_inertia = float(mb_model.inertia_)

            speedup = round(full_time / mb_time, 2)
            if abs(baseline_inertia) <= 1e-12:
                inertia_inc_pct = 0.0 if abs(mb_inertia) <= 1e-12 else 100.0
            else:
                inertia_inc_pct = round(
                    ((mb_inertia - baseline_inertia) / baseline_inertia) * 100.0, 2
                )

            ari = round(float(adjusted_rand_score(baseline_labels, mb_labels)), 4)
            nmi = round(float(normalized_mutual_info_score(baseline_labels, mb_labels)), 4)

            rows.append(
                {
                    "model": f"MiniBatch ({batch_label})",
                    "batch_size": int(bs),
                    "effective_batch_size": int(effective_bs),
                    "fit_time_sec": round(float(mb_time), 4),
                    "speedup": speedup,
                    "inertia": round(mb_inertia, 2),
                    "inertia_increase_pct": inertia_inc_pct,
                    "ari": ari,
                    "nmi": nmi,
                }
            )
            logger.info(
                f"  -> batch thực dùng={effective_bs}, yêu cầu={int(bs)} | thời gian={mb_time:.4f}s (x{speedup}) | "
                f"tăng WCSS={inertia_inc_pct:+.2f}% | ARI={ari} | NMI={nmi}"
            )

        return pd.DataFrame(rows)

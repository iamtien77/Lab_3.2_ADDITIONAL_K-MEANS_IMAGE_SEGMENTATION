"""
=============================================================================
MODULE: PHÂN TÍCH ĐÁNH ĐỔI VỚI MINI-BATCH K-MEANS (SPEED / QUALITY TRADE-OFF)
=============================================================================
Mục đích:
- Thử nghiệm MiniBatchKMeans trên các kích thước batch khác nhau.
- Đánh giá mức độ tăng tốc thời gian thực thi (Speedup factor).
- Đo lường độ bảo toàn chất lượng thông qua ARI và NMI so với Full K-Means.
=============================================================================
"""

import time
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.utils.logger import get_logger

logger = get_logger("MiniBatchKMeansTrainer")


class MiniBatchKMeansTrainer:
    """
    Lớp thử nghiệm và đánh giá Mini-Batch K-Means so với Full K-Means.
    """

    def __init__(self, random_state: int = 42):
        """Khởi tạo seed ngẫu nhiên."""
        self.random_state = random_state

    def compare_batch_sizes(
        self,
        X: np.ndarray,
        k: int,
        full_kmeans_model: KMeans,
        batch_sizes: List[int] = [256, 512, 1024, 2048, 4096],
    ) -> pd.DataFrame:
        """
        Benchmark MiniBatchKMeans trên nhiều batch_size và so sánh với Full K-Means.

        Returns: DataFrame với columns:
            [batch_size, training_time_ms, speedup, inertia, inertia_increase_pct, ari, nmi]
        """
        X = np.asarray(X, dtype=np.float32)

        # --- Lấy mốc chuẩn từ Full K-Means đã có ---
        full_labels = full_kmeans_model.labels_
        full_inertia = float(full_kmeans_model.inertia_)

        # Đo thời gian Full K-Means (re-fit để lấy thời gian thực)
        t0 = time.perf_counter()
        full_model_timed = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            random_state=self.random_state,
        )
        full_model_timed.fit(X)
        t_full_ms = (time.perf_counter() - t0) * 1000.0

        logger.info(
            f"[Baseline] Full K-Means | K={k} | WCSS={full_inertia:.2f} | "
            f"time={t_full_ms:.1f}ms"
        )

        rows: List[Dict[str, Any]] = []

        for batch_size in batch_sizes:
            mb_model = MiniBatchKMeans(
                n_clusters=k,
                batch_size=batch_size,
                random_state=self.random_state,
                n_init=3,
            )

            t0 = time.perf_counter()
            mb_model.fit(X)
            t_batch_ms = (time.perf_counter() - t0) * 1000.0

            mb_labels = mb_model.labels_
            mb_inertia = float(mb_model.inertia_)

            speedup = t_full_ms / (t_batch_ms + 1e-9)
            inertia_increase_pct = (mb_inertia - full_inertia) / (full_inertia + 1e-9) * 100.0

            ari = float(adjusted_rand_score(full_labels, mb_labels))
            nmi = float(normalized_mutual_info_score(full_labels, mb_labels))

            rows.append({
                "batch_size": batch_size,
                "training_time_ms": round(t_batch_ms, 2),
                "speedup": round(speedup, 3),
                "inertia": round(mb_inertia, 2),
                "inertia_increase_pct": round(inertia_increase_pct, 4),
                "ari": round(ari, 4),
                "nmi": round(nmi, 4),
            })

            logger.info(
                f"  batch={batch_size:5d} | time={t_batch_ms:.1f}ms | "
                f"speedup={speedup:.2f}x | WCSS↑={inertia_increase_pct:.2f}% | "
                f"ARI={ari:.4f} | NMI={nmi:.4f}"
            )

        df = pd.DataFrame(rows)
        logger.info("Hoàn tất benchmark MiniBatchKMeans.")
        return df

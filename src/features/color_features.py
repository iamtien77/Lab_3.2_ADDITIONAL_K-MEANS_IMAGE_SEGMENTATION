"""
=============================================================================
MODULE: TRÍCH XUẤT ĐẶC TRƯNG VÀ THỐNG KÊ MÀU SẮC (COLOR FEATURES)
=============================================================================
Mục đích:
- Chuẩn hóa ma trận điểm ảnh về định dạng float32 an toàn (kiểm tra NaN/Inf và biên độ).
- Thống kê phân phối màu sắc và độ tương phản từng kênh màu trước phân cụm (EDA).
- Định danh tên màu ngữ nghĩa bằng Tiếng Việt kèm Tiếng Anh cho từng tâm cụm K-Means.
- Tổng hợp bảng phân bổ tỷ lệ diện tích màu sắc (share_percentage) phục vụ trực quan hóa.
=============================================================================
"""

from typing import Any, Dict, List

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def normalize_pixels(
    pixel_matrix: np.ndarray,
    to_unit_range: bool = False,
) -> np.ndarray:
    """
    Kiểm định tính hữu hạn, ép kiểu float32 liền mạch và chuẩn hóa dải giá trị điểm ảnh.

    Tham số:
        pixel_matrix: Ma trận pixel 2D kích thước (N, 3).
        to_unit_range: Nếu True, chia cho 255.0 để đưa giá trị từ [0, 255] về [0.0, 1.0].

    Trả về:
        Ma trận pixel kiểu float32 kích thước (N, 3).

    Ngoại lệ:
        ValueError: Nếu shape khác (N, 3), mảng rỗng hoặc chứa NaN/Inf/giá trị âm.
    """
    arr = np.asarray(pixel_matrix)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise ValueError("pixel_matrix must be a non-empty 2D array with shape (N, 3).")
    if not np.isfinite(arr).all():
        raise ValueError("pixel_matrix contains non-finite values (NaN or Inf).")

    float_arr = np.ascontiguousarray(arr, dtype=np.float32)
    min_val = float(float_arr.min())
    max_val = float(float_arr.max())
    if min_val < 0.0:
        raise ValueError(f"pixel_matrix contains negative pixel values (min={min_val}).")

    if to_unit_range:
        if max_val <= 1.0:
            logger.warning(
                "Dữ liệu điểm ảnh đã có max <= 1.0 trước khi chia 255.0, cảnh báo nguy cơ chuẩn hóa hai lần."
            )
        float_arr = np.clip(float_arr / np.float32(255.0), np.float32(0.0), np.float32(1.0))

    return float_arr


def compute_color_statistics(pixel_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Tính toán thống kê mô tả chất lượng dữ liệu màu cho ma trận điểm ảnh (N, 3) trên thang RGB.

    Tham số:
        pixel_matrix: Ma trận 2D (N, 3) hoặc ảnh 3D (H, W, 3).

    Trả về:
        Từ điển chứa thống kê từng kênh R, G, B (mean, std, median, IQR), độ sáng trung bình
        và số lượng màu phân biệt tối đa trên dữ liệu làm việc.
    """
    arr = np.asarray(pixel_matrix)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr.reshape(-1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise ValueError("Input must be a non-empty (N, 3) matrix or (H, W, 3) image.")
    if not np.isfinite(arr).all():
        raise ValueError("Input pixel data contains NaN or Inf values.")

    arr_f = arr.astype(np.float32, copy=False)
    channel_names = ("R", "G", "B")
    channels_summary: Dict[str, Dict[str, float]] = {}

    for idx, ch_name in enumerate(channel_names):
        col = arr_f[:, idx]
        q25, q50, q75 = np.percentile(col, [25.0, 50.0, 75.0])
        channels_summary[ch_name] = {
            "min": round(float(col.min()), 2),
            "max": round(float(col.max()), 2),
            "mean": round(float(col.mean()), 2),
            "std": round(float(col.std()), 2),
            "median": round(float(q50), 2),
            "iqr": round(float(q75 - q25), 2),
        }

    # Tính độ sáng cảm nhận theo chuẩn Rec. 601 (luma)
    luma = 0.299 * arr_f[:, 0] + 0.587 * arr_f[:, 1] + 0.114 * arr_f[:, 2]
    quantized_uint8 = np.clip(np.round(arr_f), 0, 255).astype(np.uint8)
    unique_colors = int(len(np.unique(quantized_uint8, axis=0))) if len(quantized_uint8) <= 500_000 else -1

    stats: Dict[str, Any] = {
        "total_pixels": int(arr_f.shape[0]),
        "unique_colors": unique_colors,
        "mean_brightness": round(float(luma.mean()), 2),
        "std_brightness": round(float(luma.std()), 2),
        "channels": channels_summary,
    }
    return stats


def classify_color(rgb: np.ndarray) -> str:
    """
    Phân loại vector màu [R, G, B] trong đoạn [0, 255] sang tên màu song ngữ Việt - Anh.

    Tham số:
        rgb: Mảng 3 thành phần màu [R, G, B].

    Trả về:
        Chuỗi tên màu bằng Tiếng Việt kèm Tiếng Anh để hiển thị trên biểu đồ và báo cáo.

    Ngoại lệ:
        ValueError: Nếu vector đầu vào không có đúng 3 phần tử hoặc chứa NaN/Inf.
    """
    vec = np.asarray(rgb, dtype=np.float32).reshape(-1)
    if vec.size != 3 or not np.isfinite(vec).all():
        raise ValueError(f"Expected 3 finite RGB components, got shape/size {vec.size}.")

    rgb_uint8 = np.clip(np.round(vec), 0, 255).astype(np.uint8)
    r, g, b = int(rgb_uint8[0]), int(rgb_uint8[1]), int(rgb_uint8[2])
    hsv_pixel = cv2.cvtColor(rgb_uint8.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0]
    hue = int(hsv_pixel[0])
    sat = int(hsv_pixel[1])
    val = int(hsv_pixel[2])

    # Phân nhóm màu vô sắc trước theo độ sáng (V) và độ bão hòa (S)
    if val < 45:
        return "Tối / Đen (Black)"
    if sat < 35 and val > 210:
        return "Sáng / Trắng (White)"
    if sat < 45:
        return "Xám nhạt (Light Gray)" if val >= 135 else "Xám đậm (Dark Gray)"

    # Phân nhóm sắc độ theo Hue (0..179 của OpenCV) kết hợp độ sáng
    if hue < 8 or hue >= 170:
        if val < 120:
            return "Nâu sẫm (Dark Brown)"
        if hue >= 158 and r > 150 and b > 100 and g < 130:
            return "Đỏ mận (Magenta-Red)"
        return "Đỏ (Red)"
    if 8 <= hue < 16:
        if val < 140:
            return "Nâu cam (Orange-Brown)"
        return "Cam rực (Orange)"
    if 16 <= hue < 24:
        return "Nâu đất (Warm Brown)" if val < 140 else "Cam vàng (Yellow-Orange)"
    if 24 <= hue < 36:
        return "Vàng kim (Golden Yellow)" if val > 165 else "Vàng ô liu (Olive)"
    if 36 <= hue < 85:
        return "Xanh lá (Green)"
    if 85 <= hue < 102:
        return "Xanh ngọc (Cyan)"
    if 102 <= hue < 130:
        return "Xanh lam đậm (Navy Blue)" if val < 110 else "Xanh dương (Blue)"
    if 130 <= hue < 155:
        return "Tím than (Dark-Violet)" if val < 120 else "Tím (Purple)"
    # Phân biệt đỏ mận sẫm (ví dụ #88395A có độ sáng trung bình/thấp) với hồng sáng
    if val < 175:
        return "Đỏ mận (Magenta-Red)"
    return "Hồng (Pink)"


def extract_color_distribution(
    labels: np.ndarray,
    centroids: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Thống kê số điểm ảnh, tỷ lệ diện tích (%), mã màu HEX và tên màu ngữ nghĩa cho từng cụm.

    Tham số:
        labels: Mảng 1D nhãn cụm có độ dài N thuộc đoạn [0, K-1].
        centroids: Ma trận 2D tọa độ tâm cụm (K, 3) trong không gian RGB [0, 255].

    Trả về:
        Danh sách dictionary thống kê các cụm được sắp xếp giảm dần theo share_percentage.

    Ngoại lệ:
        ValueError: Nếu labels rỗng, chứa nhãn ngoài biên hoặc centroids sai kích thước.
    """
    labels_arr = np.asarray(labels).reshape(-1)
    centroids_arr = np.asarray(centroids, dtype=np.float64)

    if labels_arr.size == 0:
        raise ValueError("labels array must not be empty.")
    if centroids_arr.ndim != 2 or centroids_arr.shape[1] != 3 or centroids_arr.shape[0] == 0:
        raise ValueError("centroids must be a non-empty 2D array of shape (K, 3).")
    if not np.isfinite(centroids_arr).all():
        raise ValueError("centroids must contain finite numbers.")

    n_clusters = int(centroids_arr.shape[0])
    labels_int = labels_arr.astype(np.int64, copy=False)
    if int(labels_int.min()) < 0 or int(labels_int.max()) >= n_clusters:
        raise ValueError(
            f"labels values must lie in [0, {n_clusters - 1}], "
            f"got [{int(labels_int.min())}, {int(labels_int.max())}]."
        )

    total_pixels = int(labels_int.size)
    counts = np.bincount(labels_int, minlength=n_clusters)

    distribution: List[Dict[str, Any]] = []
    for cluster_id in range(n_clusters):
        centroid = centroids_arr[cluster_id]
        rgb_uint8 = np.clip(np.round(centroid), 0, 255).astype(np.uint8)
        r, g, b = int(rgb_uint8[0]), int(rgb_uint8[1]), int(rgb_uint8[2])
        count = int(counts[cluster_id])
        share_pct = round((count / float(total_pixels)) * 100.0, 2)

        distribution.append(
            {
                "cluster_id": int(cluster_id),
                "centroid_rgb": [round(float(val), 2) for val in centroid],
                "rgb_uint8": [r, g, b],
                "hex_color": f"#{r:02X}{g:02X}{b:02X}",
                "color_name": classify_color(centroid),
                "pixel_count": count,
                "share_percentage": share_pct,
            }
        )

    # Sắp xếp giảm dần theo tỷ lệ diện tích, ưu tiên cluster_id tăng dần nếu tỷ lệ bằng nhau
    distribution.sort(key=lambda item: (-item["share_percentage"], item["cluster_id"]))
    for rank, item in enumerate(distribution, start=1):
        item["dominance_rank"] = rank

    logger.info(
        f"Đã trích xuất phân bố {n_clusters} cụm màu trên {total_pixels:,} pixels; "
        f"cụm chiếm ưu thế nhất: #{distribution[0]['cluster_id']} "
        f"({distribution[0]['color_name']}, {distribution[0]['share_percentage']}%)."
    )
    return distribution

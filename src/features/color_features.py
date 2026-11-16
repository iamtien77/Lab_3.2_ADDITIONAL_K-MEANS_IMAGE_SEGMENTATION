"""
=============================================================================
MODULE: TRÍCH XUẤT ĐẶC TRƯNG MÀU SẮC (COLOR FEATURES)
=============================================================================
Mục đích:
- Chuẩn hóa biên độ giá trị điểm ảnh (Scale / Normalize pixels).
- Định danh tên màu ngữ nghĩa (Semantic color labels) cho từng tâm cụm K-Means.
- Thống kê tỷ lệ phân bổ diện tích màu sắc trên toàn bức ảnh.
=============================================================================
"""

from typing import List, Dict, Any
import numpy as np


def normalize_pixels(
    pixel_matrix: np.ndarray,
    to_unit_range: bool = False,
) -> np.ndarray:
    """
    Chuẩn hóa ma trận điểm ảnh sang float32.
    - to_unit_range=True  → chia 255.0, kết quả [0.0, 1.0]
    - to_unit_range=False → giữ thang [0.0, 255.0] ở float32
    """
    result = np.asarray(pixel_matrix, dtype=np.float32)
    if to_unit_range:
        result = result / 255.0
    return result


def classify_color(rgb: np.ndarray) -> str:
    """
    Phân loại màu heuristic từ vector RGB [R, G, B] ∈ [0, 255].
    Trả về tên màu Tiếng Việt (Tiếng Anh).
    """
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])

    brightness = (r + g + b) / 3.0
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    saturation = (max_c - min_c) / (max_c + 1e-8)

    # Trắng / Sáng
    if brightness > 200 and saturation < 0.15:
        return "Trắng (White)"

    # Đen / Tối
    if brightness < 45:
        return "Đen (Black)"

    # Xám
    if saturation < 0.12:
        if brightness < 100:
            return "Xám tối (Dark Gray)"
        elif brightness < 160:
            return "Xám (Gray)"
        else:
            return "Xám sáng (Light Gray)"

    # Phân loại theo kênh nổi trội
    if max_c == r:
        if g > 150 and b < 100:
            return "Vàng (Yellow)"
        elif g > 100 and b < 80:
            return "Cam (Orange)"
        elif g < 80 and b < 100:
            return "Đỏ (Red)"
        else:
            return "Hồng / Đỏ nhạt (Pink-Red)"
    elif max_c == g:
        if r > 150 and b < 100:
            return "Vàng lục (Yellow-Green)"
        elif b > 150:
            return "Lục lam (Cyan)"
        else:
            return "Xanh lá (Green)"
    else:  # max_c == b
        if r > 150:
            return "Tím (Violet)"
        elif g > 100:
            return "Lam xanh (Cyan-Blue)"
        else:
            return "Xanh dương (Blue)"


def extract_color_distribution(
    labels: np.ndarray,
    centroids: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Thống kê phân bố màu sắc theo từng cụm K-Means.
    Returns: danh sách dict sắp xếp giảm dần theo share_percentage.
    Mỗi phần tử: {cluster_id, centroid_rgb, pixel_count, share_percentage, color_name}
    """
    labels = np.asarray(labels, dtype=np.int32)
    centroids = np.asarray(centroids, dtype=np.float32)

    n_total = len(labels)
    k = len(centroids)
    stats: List[Dict[str, Any]] = []

    for c in range(k):
        mask = labels == c
        pixel_count = int(np.sum(mask))
        share_pct = round((pixel_count / n_total) * 100.0, 2)

        # Tâm cụm làm tròn sang uint8
        centroid_rgb = np.clip(np.round(centroids[c]), 0, 255).astype(np.uint8)
        color_name = classify_color(centroid_rgb)

        stats.append({
            "cluster_id": c,
            "centroid_rgb": centroid_rgb.tolist(),
            "pixel_count": pixel_count,
            "share_percentage": share_pct,
            "color_name": color_name,
        })

    # Sắp xếp giảm dần theo tỷ lệ
    stats.sort(key=lambda x: x["share_percentage"], reverse=True)
    return stats

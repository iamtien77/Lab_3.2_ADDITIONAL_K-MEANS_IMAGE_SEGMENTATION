"""
=============================================================================
MODULE: TRỰC QUAN HÓA KẾT QUẢ PHÂN ĐOẠN (VISUALIZATION)
=============================================================================
Mục đích (Đáp ứng đầy đủ Task 5 trong đề bài Practice Exercise 3.2):
- Hiển thị ảnh gốc và ảnh đã phân đoạn cạnh nhau (side-by-side):
  "Display the original and segmented images side-by-side."
- Trực quan hóa đường cong Elbow Method kèm điểm khuỷu tay tối ưu.
- Trực quan hóa tiến trình hội tụ (Convergence tracking) của thuật toán K-Means thuần.
- Biểu đồ phân bổ tỷ lệ phần trăm diện tích màu sắc của các cụm.
- Lưới tiến hóa ảnh phân khúc qua các giá trị K khác nhau (K-evolution grid).
=============================================================================
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_side_by_side(
    original_img: np.ndarray,
    segmented_img: np.ndarray,
    k: int,
    title: str = "So sánh Ảnh Gốc vs Ảnh Phân Đoạn K-Means",
    save_path: Optional[Path] = None,
) -> None:
    """
    Ý LÀM (Đáp ứng Task 5 - Display original and segmented images side-by-side):
    1. Khởi tạo figure Matplotlib gồm 1 hàng 2 cột với kích thước chuẩn (ví dụ 14x7 inch).
    2. Cột trái: Vẽ ảnh gốc (RGB) kèm tiêu đề kích thước gốc và số điểm ảnh.
    3. Cột phải: Vẽ ảnh đã phân đoạn bằng K-Means với K cụm màu đại diện.
    4. Tắt các trục tọa độ thừa (axis off) để ảnh hiển thị thẩm mỹ nhất.
    5. Đặt tiêu đề chung (Suptitle) cho toàn bộ biểu đồ.
    6. Nếu có save_path: lưu ảnh ra file PNG với độ phân giải DPI=150.
    """
    pass


def plot_color_space_comparison(
    img_rgb: np.ndarray,
    img_hsv: np.ndarray,
    img_lab: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    """
    Ý LÀM:
    1. Tạo figure gồm 1 hàng 3 cột để so sánh 3 không gian màu: RGB, HSV, LAB.
    2. Hiển thị từng ảnh tương ứng kèm tiêu đề đặc trưng của không gian màu đó.
    3. Lưu biểu đồ vào save_path nếu được cung cấp.
    """
    pass


def plot_elbow_curve(
    df_elbow: pd.DataFrame,
    optimal_k: int,
    save_path: Optional[Path] = None,
) -> None:
    """
    Ý LÀM:
    1. Vẽ đường cong WCSS (Inertia) giảm dần theo số cụm K từ 2 đến 12.
    2. Đánh dấu điểm khuỷu tay tối ưu (Elbow point tại K=5) bằng đường nét đứt đỏ và điểm tròn nổi bật.
    3. Ghi nhãn trục X ("Số cụm K"), trục Y ("Quán tính WCSS / Inertia").
    4. Thêm chú thích (Legend) và lưới tọa độ mờ (Grid).
    5. Lưu biểu đồ vào results/figures/elbow_method.png.
    """
    pass


def plot_convergence_tracking(
    history_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """
    Ý LÀM:
    1. Tạo đồ thị gồm 2 đồ thị con (Subplots):
       - Subplot 1: Quán tính WCSS giảm dần qua các vòng lặp (Iterations).
       - Subplot 2: Độ dịch chuyển tâm cụm (Centroid shift) giảm dần về 0.
    2. Đánh dấu điểm thuật toán đạt trạng thái hội tụ hoàn toàn.
    3. Lưu biểu đồ vào results/figures/kmeans_convergence.png.
    """
    pass


def plot_cluster_distribution(
    cluster_stats: List[Dict[str, Any]],
    save_path: Optional[Path] = None,
) -> None:
    """
    Ý LÀM:
    1. Vẽ biểu đồ cột ngang (Horizontal bar chart) thể hiện tỷ lệ phần trăm pixel của từng cụm.
    2. Đặt màu cho từng thanh cột bằng chính mã màu RGB của tâm cụm tương ứng.
    3. Ghi rõ nhãn tên màu và số lượng điểm ảnh trên từng thanh cột.
    4. Lưu biểu đồ vào results/figures/cluster_distribution.png.
    """
    pass


def plot_k_evolution_grid(
    k_images: Dict[int, np.ndarray],
    save_path: Optional[Path] = None,
) -> None:
    """
    Ý LÀM:
    1. Tạo lưới hiển thị ảnh phân đoạn qua các giá trị K tăng dần (ví dụ K=2, 3, 4, 5, 6, 8).
    2. Giúp quan sát trực quan sự cải thiện độ chi tiết màu sắc khi tăng số lượng cụm K.
    3. Lưu ảnh lưới vào results/figures/k_evolution_grid.png.
    """
    pass

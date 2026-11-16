"""
=============================================================================
MODULE: TRỰC QUAN HÓA KẾT QUẢ PHÂN ĐOẠN (VISUALIZATION)
=============================================================================
Mục đích (Đáp ứng đầy đủ Task 5 trong đề bài Practice Exercise 3.2):
- Hiển thị ảnh gốc và ảnh đã phân đoạn cạnh nhau (side-by-side).
- Trực quan hóa đường cong Elbow Method kèm điểm khuỷu tay tối ưu.
- Trực quan hóa tiến trình hội tụ của K-Means thuần.
- Biểu đồ phân bổ màu sắc theo cụm.
- Lưới tiến hóa ảnh phân đoạn qua các K khác nhau.
=============================================================================
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.config import FIGURE_DPI
from src.utils.logger import get_logger

logger = get_logger("Visualization")


def _save_or_show(fig: plt.Figure, save_path: Optional[Path], label: str) -> None:
    """Helper: lưu hình nếu có save_path, luôn đóng figure sau đó."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info(f"Đã lưu {label}: {save_path}")
    plt.close(fig)


def plot_side_by_side(
    original_img: np.ndarray,
    segmented_img: np.ndarray,
    k: int,
    title: str = "So sánh Ảnh Gốc vs Ảnh Phân Đoạn K-Means",
    save_path: Optional[Path] = None,
) -> None:
    """
    Hiển thị ảnh gốc (RGB) và ảnh phân đoạn cạnh nhau.
    Đáp ứng Task 5: Display original and segmented images side-by-side.
    """
    H_orig, W_orig = original_img.shape[:2]
    H_seg, W_seg = segmented_img.shape[:2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(original_img)
    axes[0].set_title(
        f"Ảnh Gốc\n({H_orig}×{W_orig} | {H_orig * W_orig:,} pixels)",
        fontsize=12,
    )
    axes[0].axis("off")

    axes[1].imshow(segmented_img)
    axes[1].set_title(
        f"Ảnh Phân Đoạn K-Means (K={k})\n({H_seg}×{W_seg})",
        fontsize=12,
    )
    axes[1].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_or_show(fig, save_path, "side-by-side comparison")


def plot_color_space_comparison(
    img_rgb: np.ndarray,
    img_hsv: np.ndarray,
    img_lab: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    """
    So sánh 3 không gian màu: RGB, HSV, LAB cạnh nhau.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # RGB — hiển thị trực tiếp
    axes[0].imshow(img_rgb)
    axes[0].set_title("RGB\n(Red, Green, Blue)", fontsize=12)
    axes[0].axis("off")

    # HSV — cần chuẩn hóa về [0,1] để imshow hiểu đúng
    # OpenCV trả về H∈[0,180], S∈[0,255], V∈[0,255]
    hsv_display = img_hsv.astype(np.float32)
    hsv_display[:, :, 0] = hsv_display[:, :, 0] / 180.0
    hsv_display[:, :, 1] = hsv_display[:, :, 1] / 255.0
    hsv_display[:, :, 2] = hsv_display[:, :, 2] / 255.0
    axes[1].imshow(np.clip(hsv_display, 0, 1))
    axes[1].set_title("HSV\n(Hue, Saturation, Value)", fontsize=12)
    axes[1].axis("off")

    # LAB — chuẩn hóa về [0,1]
    lab_display = img_lab.astype(np.float32)
    lab_min = lab_display.min()
    lab_max = lab_display.max()
    lab_display = (lab_display - lab_min) / (lab_max - lab_min + 1e-8)
    axes[2].imshow(np.clip(lab_display, 0, 1))
    axes[2].set_title("LAB\n(Luminance, A-channel, B-channel)", fontsize=12)
    axes[2].axis("off")

    fig.suptitle("So sánh Không Gian Màu", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_path, "color space comparison")


def plot_elbow_curve(
    df_elbow: pd.DataFrame,
    optimal_k: int,
    save_path: Optional[Path] = None,
) -> None:
    """
    Vẽ đường cong Elbow (WCSS vs K) và đánh dấu điểm khuỷu tay tối ưu.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        df_elbow["K"],
        df_elbow["Inertia"],
        marker="o",
        linewidth=2,
        markersize=6,
        color="#2196F3",
        label="WCSS (Inertia)",
    )

    # Đánh dấu Elbow point
    elbow_inertia = df_elbow.loc[df_elbow["K"] == optimal_k, "Inertia"].values
    if len(elbow_inertia) > 0:
        ax.axvline(x=optimal_k, color="#F44336", linestyle="--", linewidth=1.5,
                   label=f"Elbow K = {optimal_k}")
        ax.scatter([optimal_k], [elbow_inertia[0]], color="#F44336", s=120,
                   zorder=5, marker="*")
        ax.annotate(
            f"K = {optimal_k}\nWCSS = {elbow_inertia[0]:.0f}",
            xy=(optimal_k, elbow_inertia[0]),
            xytext=(optimal_k + 0.5, elbow_inertia[0] * 1.05),
            fontsize=10, color="#F44336",
            arrowprops=dict(arrowstyle="->", color="#F44336"),
        )

    ax.set_xlabel("Số cụm K", fontsize=12)
    ax.set_ylabel("Quán tính WCSS / Inertia", fontsize=12)
    ax.set_title("Elbow Method — Xác định số cụm tối ưu K", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xticks(df_elbow["K"].tolist())

    plt.tight_layout()
    _save_or_show(fig, save_path, "elbow curve")


def plot_convergence_tracking(
    history_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """
    Vẽ 2 subplots theo dõi hội tụ K-Means:
    - Subplot 1: WCSS giảm qua các iteration.
    - Subplot 2: Centroid shift giảm về 0.
    """
    if history_df.empty:
        logger.warning("history_df rỗng, bỏ qua plot_convergence_tracking.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    iters = history_df["iteration"]

    # --- Subplot 1: WCSS ---
    axes[0].plot(iters, history_df["inertia"], color="#2196F3",
                 linewidth=2, marker="o", markersize=3)
    axes[0].set_xlabel("Vòng lặp (Iteration)", fontsize=11)
    axes[0].set_ylabel("WCSS / Inertia", fontsize=11)
    axes[0].set_title("Quá trình giảm WCSS", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # --- Subplot 2: Centroid Shift ---
    axes[1].plot(iters, history_df["centroid_shift"], color="#4CAF50",
                 linewidth=2, marker="o", markersize=3)
    axes[1].axhline(y=0, color="#F44336", linestyle="--", linewidth=1,
                    label="Ngưỡng hội tụ (0)")
    axes[1].set_xlabel("Vòng lặp (Iteration)", fontsize=11)
    axes[1].set_ylabel("Centroid Shift (Max Euclidean)", fontsize=11)
    axes[1].set_title("Độ dịch chuyển tâm cụm", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)

    # Đánh dấu điểm hội tụ (shift gần 0)
    tol_mask = history_df["centroid_shift"] <= 1e-4
    if tol_mask.any():
        conv_iter = history_df.loc[tol_mask, "iteration"].iloc[0]
        conv_shift = history_df.loc[tol_mask, "centroid_shift"].iloc[0]
        axes[1].scatter([conv_iter], [conv_shift], color="#F44336", s=100,
                        zorder=5, marker="*", label=f"Hội tụ tại iter={conv_iter}")
        axes[1].legend(fontsize=10)

    fig.suptitle("Theo dõi Hội tụ K-Means From Scratch", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_path, "convergence tracking")


def plot_cluster_distribution(
    cluster_stats: List[Dict[str, Any]],
    save_path: Optional[Path] = None,
) -> None:
    """
    Biểu đồ cột ngang thể hiện tỷ lệ % pixel của từng cụm màu.
    Màu thanh = màu RGB của tâm cụm.
    """
    if not cluster_stats:
        logger.warning("cluster_stats rỗng, bỏ qua plot_cluster_distribution.")
        return

    n = len(cluster_stats)
    labels = [
        f"Cụm {s['cluster_id']}: {s['color_name']}\nRGB={tuple(s['centroid_rgb'])}"
        for s in cluster_stats
    ]
    values = [s["share_percentage"] for s in cluster_stats]
    colors = [
        (s["centroid_rgb"][0] / 255.0,
         s["centroid_rgb"][1] / 255.0,
         s["centroid_rgb"][2] / 255.0)
        for s in cluster_stats
    ]

    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.9)))

    bars = ax.barh(labels, values, color=colors, edgecolor="gray", linewidth=0.5)

    # Ghi % lên từng thanh
    for bar, val, stat in zip(bars, values, cluster_stats):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.1f}% ({stat['pixel_count']:,} px)",
            va="center", ha="left", fontsize=9,
        )

    ax.set_xlabel("Tỷ lệ diện tích (%)", fontsize=11)
    ax.set_title("Phân bổ Màu Sắc theo Cụm K-Means", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(values) * 1.25)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path, "cluster distribution")


def plot_k_evolution_grid(
    k_images: Dict[int, np.ndarray],
    save_path: Optional[Path] = None,
) -> None:
    """
    Lưới hiển thị ảnh phân đoạn tương ứng với các giá trị K tăng dần.
    k_images: {K_value: segmented_img_array}
    """
    if not k_images:
        logger.warning("k_images rỗng, bỏ qua plot_k_evolution_grid.")
        return

    k_values = sorted(k_images.keys())
    n = len(k_values)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for idx, k in enumerate(k_values):
        ax = axes[idx]
        ax.imshow(k_images[k])
        ax.set_title(f"K = {k}", fontsize=12, fontweight="bold")
        ax.axis("off")

    # Ẩn các ô thừa
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Tiến hóa Phân đoạn Ảnh theo Số cụm K",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _save_or_show(fig, save_path, "K-evolution grid")

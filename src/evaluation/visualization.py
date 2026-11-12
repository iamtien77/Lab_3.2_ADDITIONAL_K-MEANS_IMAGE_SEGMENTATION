"""
=============================================================================
MODULE: TRỰC QUAN HÓA KẾT QUẢ PHÂN ĐOẠN (VISUALIZATION)
=============================================================================
Mục đích (Đáp ứng đầy đủ Task 5 trong đề bài Practice Exercise 3.2):
- Vẽ biểu đồ so sánh ảnh gốc và ảnh phân đoạn cạnh nhau (Side-by-Side).
- So sánh trực quan 3 không gian màu RGB, HSV và CIE LAB.
- Vẽ đường cong Elbow Method với điểm khuỷu tay tối ưu (Elbow point).
- Vẽ biểu đồ theo dõi hội tụ (WCSS & Centroid shift) của KMeansFromScratch.
- Vẽ biểu đồ cột ngang phân bổ tỷ lệ % màu sắc theo từng cụm.
- Vẽ lưới tiến hóa mức độ chi tiết theo K tăng dần (K evolution grid).
=============================================================================
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

# Bắt buộc dùng backend không giao diện 'Agg' để xuất ảnh ổn định ở môi trường CLI/Windows
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURE_DPI
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _compute_panel_layout(
    n_cols: int,
    n_rows: int,
    img_height: int,
    img_width: int,
    panel_width: float = 6.8,
    min_panel_height: float = 3.4,
    max_panel_height: float = 7.0,
    header_inches: float = 1.15,
) -> tuple[tuple[float, float], float]:
    """Tính kích thước figure và vùng biên trên theo đúng tỷ lệ ảnh để không thừa khoảng trắng."""
    aspect = float(img_height) / float(max(1, img_width))
    panel_h = float(np.clip(panel_width * aspect, min_panel_height, max_panel_height))
    fig_w = panel_width * n_cols + 0.6
    fig_h = panel_h * n_rows + header_inches
    # Dành cố định ~0.85 inch phía trên cho suptitle và tiêu đề subplot đầu hàng
    rect_top = max(0.75, min(0.94, 1.0 - (0.85 / fig_h)))
    return (fig_w, fig_h), rect_top


def _finalize_figure(fig: plt.Figure, save_path: Optional[Path]) -> None:
    """Lưu figure ra file PNG (nếu truyền save_path) và giải phóng bộ nhớ Matplotlib."""
    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info(f"Đã lưu biểu đồ trực quan tại: {out_path}")
    plt.close(fig)


def plot_side_by_side(
    original_img: np.ndarray,
    segmented_img: np.ndarray,
    k: int,
    title: str = "Original vs K-Means Segmented Image",
    save_path: Optional[Path] = None,
) -> None:
    """
    Hiển thị và lưu ảnh gốc cạnh ảnh đã phân đoạn K-Means với thông tin kích thước và số cụm.

    Tham số:
        original_img: Ảnh gốc RGB kích thước (H, W, 3).
        segmented_img: Ảnh đã phân đoạn RGB kích thước (H, W, 3).
        k: Số cụm màu K đã dùng để phân đoạn.
        title: Tiêu đề tổng thể của biểu đồ (mặc định Tiếng Anh cho đồ thị).
        save_path: Đường dẫn xuất file PNG (tùy chọn).
    """
    h_orig, w_orig = original_img.shape[:2]
    h_seg, w_seg = segmented_img.shape[:2]
    figsize, rect_top = _compute_panel_layout(
        n_cols=2,
        n_rows=1,
        img_height=h_orig,
        img_width=w_orig,
        panel_width=6.8,
        min_panel_height=3.6,
        max_panel_height=7.0,
    )
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(original_img)
    axes[0].set_title(
        f"Original Image ({w_orig}x{h_orig}, {w_orig * h_orig:,} px)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].axis("off")

    axes[1].imshow(segmented_img)
    axes[1].set_title(
        f"K-Means Segmented (K={k}, {w_seg}x{h_seg})",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    # Chừa khoảng trống phía trên tỷ lệ theo chiều cao figure để không đè chữ và không thừa khoảng trắng
    fig.tight_layout(rect=[0.0, 0.0, 1.0, rect_top])
    _finalize_figure(fig, save_path)


def plot_color_space_comparison(
    img_rgb: np.ndarray,
    img_hsv: np.ndarray,
    img_lab: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    """
    Trực quan hóa song song 3 không gian màu RGB, HSV và CIE LAB để đối chiếu đặc trưng.

    Tham số:
        img_rgb: Mảng ảnh RGB (H, W, 3).
        img_hsv: Mảng ảnh HSV (H, W, 3).
        img_lab: Mảng ảnh CIE LAB (H, W, 3).
        save_path: Đường dẫn xuất file PNG.
    """
    h_img, w_img = img_rgb.shape[:2]
    figsize, rect_top = _compute_panel_layout(
        n_cols=3,
        n_rows=1,
        img_height=h_img,
        img_width=w_img,
        panel_width=5.2,
        min_panel_height=3.0,
        max_panel_height=5.5,
    )
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    panels = [
        (img_rgb, "RGB Color Space (Baseline Euclidean)"),
        (img_hsv, "HSV Channels (H, S, V False-Color)"),
        (img_lab, "CIE LAB Channels (L*, a*, b* False-Color)"),
    ]

    for ax, (img, label) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.axis("off")

    fig.suptitle("Color Space Representations Comparison", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, rect_top])
    _finalize_figure(fig, save_path)


def plot_elbow_curve(
    df_elbow: pd.DataFrame,
    optimal_k: int,
    save_path: Optional[Path] = None,
) -> None:
    """
    Vẽ đường cong Elbow Method (WCSS theo K) và đánh dấu điểm khuỷu tay optimal_k.

    Tham số:
        df_elbow: DataFrame chứa tối thiểu 2 cột 'K' và 'Inertia'.
        optimal_k: Giá trị K tối ưu được chọn.
        save_path: Đường dẫn xuất file PNG.
    """
    if df_elbow.empty or "K" not in df_elbow.columns or "Inertia" not in df_elbow.columns:
        raise ValueError("df_elbow must be non-empty and contain 'K' and 'Inertia' columns.")

    fig, ax = plt.subplots(figsize=(10, 6))
    k_vals = df_elbow["K"].to_numpy()
    inertias = df_elbow["Inertia"].to_numpy()

    ax.plot(k_vals, inertias, marker="o", linewidth=2.2, color="#1f77b4", label="WCSS (Inertia)")
    ax.axvline(
        x=optimal_k,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label=f"Optimal Elbow K = {optimal_k}",
    )

    match = df_elbow[df_elbow["K"] == optimal_k]
    if not match.empty:
        opt_inertia = float(match["Inertia"].iloc[0])
        ax.scatter([optimal_k], [opt_inertia], color="#d62728", s=110, zorder=5)
        ax.annotate(
            f"K={optimal_k}\nWCSS={opt_inertia:,.0f}",
            xy=(optimal_k, opt_inertia),
            xytext=(optimal_k + 0.6, opt_inertia * 1.05),
            fontsize=10,
            fontweight="bold",
            color="#d62728",
            arrowprops={"arrowstyle": "->", "color": "#d62728"},
        )

    ax.set_title("Elbow Method for Optimal Number of Clusters (K)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax.set_ylabel("WCSS / Inertia", fontsize=11)
    ax.set_xticks(k_vals)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    _finalize_figure(fig, save_path)


def plot_convergence_tracking(
    history_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """
    Vẽ 2 biểu đồ con theo dõi WCSS và độ dịch chuyển tâm cụm qua từng vòng lặp KMeansFromScratch.

    Tham số:
        history_df: DataFrame lịch sử hội tụ có các cột 'iteration', 'inertia', 'centroid_shift'.
        save_path: Đường dẫn xuất file PNG.
    """
    required = {"iteration", "inertia", "centroid_shift"}
    if history_df.empty or not required.issubset(history_df.columns):
        raise ValueError(f"history_df must contain columns: {sorted(required)}.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    iters = history_df["iteration"].to_numpy()
    inertias = history_df["inertia"].to_numpy()
    shifts = history_df["centroid_shift"].to_numpy()

    axes[0].plot(iters, inertias, marker="o", linewidth=2, color="#1f77b4")
    axes[0].set_title("WCSS (Inertia) per Iteration", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Iteration", fontsize=10)
    axes[0].set_ylabel("WCSS / Inertia", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].plot(iters, shifts, marker="s", linewidth=2, color="#ff7f0e")
    axes[1].set_title("Max Centroid Shift per Iteration", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Iteration", fontsize=10)
    axes[1].set_ylabel("Euclidean Shift", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(
        f"K-Means From Scratch Convergence Tracking ({len(iters)} iterations)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
    _finalize_figure(fig, save_path)


def plot_cluster_distribution(
    cluster_stats: List[Dict[str, Any]],
    save_path: Optional[Path] = None,
) -> None:
    """
    Vẽ biểu đồ cột ngang hiển thị tỷ lệ phần trăm diện tích màu sắc và HEX của các cụm.

    Tham số:
        cluster_stats: Danh sách thống kê trả về từ extract_color_distribution().
        save_path: Đường dẫn xuất file PNG.
    """
    if not cluster_stats:
        raise ValueError("cluster_stats must not be empty.")

    ordered = sorted(
        cluster_stats,
        key=lambda item: float(item.get("share_percentage", item.get("percentage", 0.0))),
    )
    labels = [
        f"Cluster #{item['cluster_id']} ({item['hex_color']})" for item in ordered
    ]
    percentages = [
        float(item.get("share_percentage", item.get("percentage", 0.0))) for item in ordered
    ]
    pixel_counts = [int(item.get("pixel_count", 0)) for item in ordered]
    bar_colors = [
        tuple(np.clip(np.asarray(item["centroid_rgb"], dtype=np.float32) / 255.0, 0.0, 1.0))
        for item in ordered
    ]

    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.9 * len(ordered))))
    bars = ax.barh(labels, percentages, color=bar_colors, edgecolor="#333333", linewidth=0.9)

    for bar, pct, count in zip(bars, percentages, pixel_counts):
        ax.text(
            bar.get_width() + 0.8,
            bar.get_y() + bar.get_height() / 2.0,
            f"{pct:.2f}% ({count:,} px)",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_xlim(0.0, max(percentages) * 1.28 + 3.0)
    ax.set_title("Dominant Color Cluster Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Area Share (%)", fontsize=11)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    fig.tight_layout()
    _finalize_figure(fig, save_path)


def plot_k_evolution_grid(
    k_images: Dict[int, np.ndarray],
    save_path: Optional[Path] = None,
) -> None:
    """
    Vẽ lưới ảnh phân đoạn theo các giá trị K tăng dần để quan sát sự thay đổi chi tiết màu.

    Tham số:
        k_images: Từ điển {k: segmented_rgb_image}.
        save_path: Đường dẫn xuất file PNG.
    """
    if not k_images:
        raise ValueError("k_images dictionary must not be empty.")

    sorted_items = sorted(k_images.items(), key=lambda pair: int(pair[0]))
    n_items = len(sorted_items)
    n_cols = min(3, n_items)
    n_rows = int(np.ceil(n_items / float(n_cols)))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, (k_val, img) in zip(axes_flat, sorted_items):
        ax.imshow(img)
        ax.set_title(f"K = {k_val} Clusters", fontsize=12, fontweight="bold")
        ax.axis("off")

    for unused_ax in axes_flat[len(sorted_items):]:
        unused_ax.axis("off")

    fig.suptitle("K-Means Segmentation Evolution Across K", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    _finalize_figure(fig, save_path)


def plot_minibatch_comparison(
    panel_entries: List[Dict[str, Any]],
    k: int,
    save_path: Optional[Path] = None,
) -> None:
    """
    Vẽ lưới đối chiếu trực quan giữa Full K-Means và các cấu hình Mini-Batch K-Means.

    Tham số:
        panel_entries: Danh sách dict gồm 'label', 'image', 'fit_time_sec', 'speedup',
                       'inertia_increase_pct', 'ari'.
        k: Số cụm màu K sử dụng chung.
        save_path: Đường dẫn xuất file PNG.
    """
    if not panel_entries:
        raise ValueError("panel_entries must not be empty.")

    sample_img = panel_entries[0]["image"]
    n_items = len(panel_entries)
    n_cols = min(4, n_items)
    n_rows = int(np.ceil(n_items / float(n_cols)))
    figsize, rect_top = _compute_panel_layout(
        n_cols=n_cols,
        n_rows=n_rows,
        img_height=sample_img.shape[0],
        img_width=sample_img.shape[1],
        panel_width=4.4,
        min_panel_height=2.8,
        max_panel_height=4.8,
        header_inches=1.25,
    )
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, entry in zip(axes_flat, panel_entries):
        ax.imshow(entry["image"])
        subtitle = (
            f"{entry['label']}\n"
            f"t={entry['fit_time_sec']:.3f}s (x{entry['speedup']}) | "
            f"dWCSS={entry['inertia_increase_pct']:+.2f}% | ARI={entry['ari']:.3f}"
        )
        ax.set_title(subtitle, fontsize=10, fontweight="bold")
        ax.axis("off")

    for unused_ax in axes_flat[len(panel_entries):]:
        unused_ax.axis("off")

    fig.suptitle(
        f"Full K-Means vs Mini-Batch K-Means Visual Quality (K={k})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, rect_top])
    _finalize_figure(fig, save_path)


def plot_color_space_segmentation_comparison(
    original_img: np.ndarray,
    space_entries: List[Dict[str, Any]],
    k: int,
    save_path: Optional[Path] = None,
) -> None:
    """
    Hiển thị song song ảnh gốc và kết quả phân đoạn K-Means trên RGB, HSV, CIE LAB.

    Tham số:
        original_img: Ảnh gốc RGB (H, W, 3).
        space_entries: Danh sách dict gồm 'color_space', 'segmented_rgb', 'silhouette', 'fit_time_sec'.
        k: Số cụm màu K dùng chung cho phép đối chiếu.
        save_path: Đường dẫn xuất file PNG.
    """
    if not space_entries:
        raise ValueError("space_entries must not be empty.")

    n_items = 1 + len(space_entries)
    figsize, rect_top = _compute_panel_layout(
        n_cols=n_items,
        n_rows=1,
        img_height=original_img.shape[0],
        img_width=original_img.shape[1],
        panel_width=4.4,
        min_panel_height=2.8,
        max_panel_height=4.8,
        header_inches=1.2,
    )
    fig, axes = plt.subplots(1, n_items, figsize=figsize)
    axes_flat = np.atleast_1d(axes).ravel()

    axes_flat[0].imshow(original_img)
    axes_flat[0].set_title("Original RGB", fontsize=11, fontweight="bold")
    axes_flat[0].axis("off")

    for ax, entry in zip(axes_flat[1:], space_entries):
        ax.imshow(entry["segmented_rgb"])
        ax.set_title(
            f"{entry['color_space']} (K={k})\n"
            f"Sil={entry['silhouette']:.3f} | t={entry['fit_time_sec']:.3f}s",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.axis("off")

    fig.suptitle(
        f"K-Means Segmentation Comparison Across Color Spaces (K={k})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, rect_top])
    _finalize_figure(fig, save_path)

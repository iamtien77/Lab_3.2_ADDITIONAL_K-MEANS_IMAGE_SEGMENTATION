"""
=============================================================================
DỰ ÁN: PHÂN ĐOẠN ẢNH BẰNG K-MEANS (K-MEANS IMAGE SEGMENTATION PIPELINE)
TẬP TIN ĐIỀU PHỐI CHÍNH (MAIN PIPELINE ENTRYPOINT)
=============================================================================
Theo đúng yêu cầu của đề bài: Practice Exercise 3 (Additional) - Practice Exercise 3.2.pdf
Các bước thực hiện trong pipeline:
  - BƯỚC 1: Load the Image & Convert Color Space (Task 1: OpenCV/PIL, BGR -> RGB/HSV/LAB).
  - BƯỚC 2: Preprocess Image (Task 2: Resize giữ tỷ lệ về tối đa 256 px, Flatten thành 2D pixel matrix).
  - BƯỚC 3: Apply K-Means from Scratch (Task 3: Cài đặt K-Means thuần và theo dõi hội tụ).
  - BƯỚC 4: Elbow Method & Optimal K Selection (Task 3: Quét K=2..12, xác định Elbow K tối ưu).
  - BƯỚC 5: Segment the Image (Task 4: Thay thế pixel bằng tâm màu, phục hồi kích thước gốc).
  - BƯỚC 6: Visualize Results Side-by-Side (Task 5: Hiển thị ảnh gốc vs ảnh phân đoạn side-by-side).
  - BƯỚC 7: Advanced Validation & Deployment (Đối chiếu OpenCV, MiniBatch, Lưu mô hình, Suy luận ảnh mới).
=============================================================================
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans

# Thêm thư mục gốc dự án vào sys.path để chạy trực tiếp main.py ổn định
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BASELINE_K,
    DEFAULT_IMAGE_PATH,
    FIGURES_DIR,
    FINAL_K,
    K_SEARCH_RANGE,
    MAX_ITER,
    METRICS_DIR,
    N_INIT,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
    RESULTS_DIR,
    SAVED_MODELS_DIR,
    TARGET_SIZE,
    TOLERANCE,
    print_section,
)
from src.data.loader import load_image, validate_image
from src.data.preprocessing import (
    convert_centroids_to_rgb,
    convert_color_space,
    flatten_pixels,
    preprocess_for_clustering,
)
from src.deployment.predictor import ImageSegmentationPredictor
from src.evaluation.metrics import compute_silhouette
from src.evaluation.visualization import (
    plot_cluster_distribution,
    plot_color_space_comparison,
    plot_color_space_segmentation_comparison,
    plot_convergence_tracking,
    plot_elbow_curve,
    plot_k_evolution_grid,
    plot_minibatch_comparison,
    plot_side_by_side,
)
from src.features.color_features import (
    compute_color_statistics,
    extract_color_distribution,
)
from src.models.kmeans_opencv import OpenCVKMeansTrainer
from src.models.kmeans_scratch import KMeansFromScratch
from src.models.kmeans_sklearn import SklearnKMeansTrainer
from src.models.minibatch_kmeans import MiniBatchKMeansTrainer
from src.models.persistence import save_model_pipeline
from src.segmentation.segmenter import ImageSegmenter
from src.utils.logger import get_logger

logger = get_logger("MainPipeline", log_file=RESULTS_DIR / "pipeline_execution.log")


def step_1_load_and_color_spaces(
    image_path: Optional[Path] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Bước 1 (Đáp ứng Task 1 - Đọc ảnh và chuyển đổi không gian màu):
    - Nạp ảnh đầu vào mặc định hoặc theo đường dẫn chỉ định và kiểm định chuẩn RGB uint8.
    - Chuyển đổi song song sang không gian màu HSV và CIE LAB để khảo sát đặc trưng.
    - Lưu ảnh so sánh 3 không gian màu vào results/figures/color_spaces_comparison.png.
    """
    print_section(
        "BƯỚC 1: NẠP ẢNH VÀ CHUYỂN ĐỔI KHÔNG GIAN MÀU (TASK 1)",
        f"Ảnh đầu vào: {image_path or DEFAULT_IMAGE_PATH}",
    )
    img_rgb, metadata = load_image(image_path)
    validate_image(img_rgb)

    img_hsv = convert_color_space(img_rgb, source_space="RGB", target_space="HSV")
    img_lab = convert_color_space(img_rgb, source_space="RGB", target_space="LAB")

    out_fig = FIGURES_DIR / "color_spaces_comparison.png"
    plot_color_space_comparison(img_rgb, img_hsv, img_lab, save_path=out_fig)

    print(
        f"Thông tin ảnh: {metadata['file_name']} | "
        f"{metadata['width']}x{metadata['height']} px "
        f"({metadata['total_pixels']:,} pixels) | "
        f"Dung lượng: {metadata['file_size_kb']} KB"
    )
    print(f"Đã xuất biểu đồ so sánh không gian màu: {out_fig}")
    logger.info("Hoàn tất Bước 1: Đã nạp ảnh và tạo biểu đồ so sánh không gian màu.")
    return img_rgb, metadata


def step_2_preprocess_and_flatten(
    img_rgb: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    target_size: int = TARGET_SIZE,
    color_space: str = "RGB",
) -> Tuple[np.ndarray, Tuple[int, int], np.ndarray]:
    """
    Bước 2 (Đáp ứng Task 2 - Tiền xử lý và làm phẳng ảnh):
    - Resize giữ tỷ lệ khung hình về cạnh dài nhất <= target_size (mặc định 256 px).
    - Chuyển sang không gian màu làm việc (RGB/HSV/LAB) và làm phẳng thành ma trận 2D (N_pixels, 3).
    - Thống kê chất lượng dữ liệu màu từng kênh và lưu JSON vào results/metrics/.
    """
    norm_space = color_space.strip().upper()
    print_section(
        "BƯỚC 2: TIỀN XỬ LÝ VÀ LÀM PHẲNG MA TRẬN PIXEL (TASK 2)",
        f"Kích thước chuẩn làm việc: cạnh dài tối đa {target_size} px | Hệ màu: {norm_space}",
    )
    stem = Path(metadata["file_name"]).stem if metadata and "file_name" in metadata else "sample"
    prep_bundle = preprocess_for_clustering(
        img_rgb,
        target_size=target_size,
        color_space=norm_space,
        save_dir=PROCESSED_DATA_DIR,
        file_prefix=stem,
    )
    resized_rgb = prep_bundle["resized_rgb"]
    resized_shape = prep_bundle["resized_shape"]
    pixel_matrix = prep_bundle["pixel_matrix"]

    color_stats = compute_color_statistics(pixel_matrix)
    stats_path = METRICS_DIR / "data_quality_stats.json"
    stats_path.write_text(
        json.dumps(color_stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"Kích thước sau resize: {resized_shape[1]}x{resized_shape[0]} | "
        f"Ma trận pixel ({norm_space}): {pixel_matrix.shape} ({pixel_matrix.dtype}) | "
        f"Số màu phân biệt: {color_stats['unique_colors']:,}"
    )
    logger.info("Hoàn tất Bước 2: Đã tiền xử lý, làm phẳng ma trận pixel và lưu thống kê chất lượng.")
    return resized_rgb, resized_shape, pixel_matrix


def step_3_kmeans_from_scratch(
    pixel_matrix: np.ndarray,
    resized_rgb: Optional[np.ndarray] = None,
    color_space: str = "RGB",
) -> Tuple[KMeansFromScratch, pd.DataFrame]:
    """
    Bước 3 (Đáp ứng Task 3 - Cài đặt và huấn luyện K-Means thuần từ đầu):
    - Huấn luyện KMeansFromScratch với K khởi tạo BASELINE_K (4), MAX_ITER (300), TOLERANCE (1e-4).
    - Trích xuất bảng lịch sử hội tụ và xuất biểu đồ vào results/figures/kmeans_convergence.png.
    - Nếu có ảnh làm việc resized_rgb, xuất thêm ảnh phân đoạn đối chiếu trực tiếp của KMeansFromScratch.
    """
    print_section(
        "BƯỚC 3: THUẬT TOÁN K-MEANS TỪ ĐẦU (PURE NUMPY BASELINE - TASK 3)",
        f"Tham số: K={BASELINE_K}, max_iter={MAX_ITER}, tol={TOLERANCE}, seed={RANDOM_SEED}, hệ màu={color_space}",
    )
    scratch_model = KMeansFromScratch(
        n_clusters=BASELINE_K,
        max_iter=MAX_ITER,
        tol=TOLERANCE,
        random_state=RANDOM_SEED,
    )
    scratch_model.fit(pixel_matrix)
    history_df = scratch_model.get_convergence_history()

    conv_fig = FIGURES_DIR / "kmeans_convergence.png"
    plot_convergence_tracking(history_df, save_path=conv_fig)
    history_df.to_csv(METRICS_DIR / "scratch_convergence_history.csv", index=False)

    if resized_rgb is not None and scratch_model.labels_ is not None and scratch_model.centroids is not None:
        rgb_centroids = convert_centroids_to_rgb(scratch_model.centroids, source_space=color_space)
        scratch_seg = ImageSegmenter().reconstruct_segmented_image(
            scratch_model.labels_,
            rgb_centroids,
            target_shape=(resized_rgb.shape[0], resized_rgb.shape[1]),
        )
        plot_side_by_side(
            resized_rgb,
            scratch_seg,
            k=BASELINE_K,
            title=f"K-Means From Scratch Baseline ({color_space}, K={BASELINE_K})",
            save_path=FIGURES_DIR / "scratch_baseline_side_by_side.png",
        )

    print(
        f"Kết quả KMeansFromScratch: dừng sau {scratch_model.n_iter_} vòng lặp "
        f"(lý do: {scratch_model.stop_reason_}) | WCSS cuối = {scratch_model.inertia_:,.2f}"
    )
    logger.info("Hoàn tất Bước 3: Đã huấn luyện KMeansFromScratch và xuất đồ thị hội tụ.")
    return scratch_model, history_df


def step_4_elbow_method_optimal_k(
    pixel_matrix: np.ndarray,
    resized_shape: Optional[Tuple[int, int]] = None,
    fixed_k: Optional[int] = None,
    color_space: str = "RGB",
) -> Tuple[KMeans, int, float]:
    """
    Bước 4 (Đáp ứng Task 3 mở rộng - Quét tham số K và Elbow Method):
    - Quét dải K từ 2 đến 12 bằng SklearnKMeansTrainer và xác định optimal_k qua Kneedle.
    - Hỗ trợ ghi đè fixed_k nếu người dùng chỉ định trực tiếp qua tham số dòng lệnh.
    - Vẽ đường cong Elbow, lưới ảnh phân đoạn theo K và trả về mô hình kèm thời gian huấn luyện.
    """
    print_section(
        "BƯỚC 4: KHẢO SÁT ELBOW METHOD & CHỌN K TỐI ƯU",
        f"Dải khảo sát K = {K_SEARCH_RANGE.start}..{K_SEARCH_RANGE.stop - 1}"
        + (f" | K chỉ định = {fixed_k}" if fixed_k is not None else ""),
    )
    trainer = SklearnKMeansTrainer(random_state=RANDOM_SEED)
    df_elbow, elbow_k = trainer.search_elbow(pixel_matrix, k_range=K_SEARCH_RANGE)
    selected_k = int(fixed_k) if fixed_k is not None else int(elbow_k)
    if fixed_k is not None and selected_k != elbow_k:
        logger.info(
            f"Ghi đè số cụm theo tham số người dùng: dùng K={selected_k} thay cho Elbow gợi ý K={elbow_k}."
        )

    df_elbow.to_csv(METRICS_DIR / "elbow_metrics.csv", index=False)
    plot_elbow_curve(df_elbow, selected_k, save_path=FIGURES_DIR / "elbow_method.png")

    t0_fit = time.perf_counter()
    final_model = trainer.fit_final_model(
        pixel_matrix,
        optimal_k=selected_k,
        n_init=N_INIT,
        max_iter=MAX_ITER,
    )
    final_fit_time = max(time.perf_counter() - t0_fit, 1e-6)

    # Nếu có kích thước resize, xuất thêm lưới trực quan hóa ảnh phân đoạn theo K tăng dần
    if resized_shape is not None:
        segmenter = ImageSegmenter()
        k_candidates = sorted({2, 3, 4, selected_k, 6, 8})
        k_images: Dict[int, np.ndarray] = {}
        for k_val in k_candidates:
            preview_model = (
                final_model
                if k_val == selected_k
                else trainer.fit_final_model(pixel_matrix, optimal_k=k_val, n_init=5, max_iter=150)
            )
            preview_rgb_centroids = convert_centroids_to_rgb(
                preview_model.cluster_centers_, source_space=color_space
            )
            k_images[k_val] = segmenter.reconstruct_segmented_image(
                preview_model.labels_,
                preview_rgb_centroids,
                target_shape=resized_shape,
            )
        plot_k_evolution_grid(k_images, save_path=FIGURES_DIR / "k_evolution_grid.png")

    print(
        f"K sử dụng cho phân đoạn cuối: K = {selected_k} (Kneedle gợi ý: {elbow_k}) "
        f"| Inertia = {final_model.inertia_:,.2f} | Thời gian fit = {final_fit_time:.4f}s"
    )
    logger.info(
        f"Hoàn tất Bước 4: Đã chọn K={selected_k} và huấn luyện xong mô hình chính thức trong {final_fit_time:.4f}s."
    )
    return final_model, selected_k, float(final_fit_time)


def step_5_segment_and_reconstruct(
    final_model: KMeans,
    pixel_matrix: np.ndarray,
    resized_shape: Tuple[int, int],
    original_shape: Tuple[int, int, int],
    img_rgb: Optional[np.ndarray] = None,
    color_space: str = "RGB",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bước 5 (Đáp ứng Task 4 - Phân đoạn và khôi phục kích thước ảnh):
    - Quy đổi tâm cụm từ không gian màu làm việc sang RGB để tái tạo ảnh đúng màu sắc.
    - Nếu có img_rgb gốc, dự đoán nhãn trực tiếp cho toàn bộ pixel gốc theo từng lô
      để giữ nguyên độ nét chi tiết ở độ phân giải gốc; ngược lại dùng Nearest Neighbor.
    - Trích xuất bản đồ ranh giới cụm và lưu ra results/figures/cluster_boundaries.png.
    """
    norm_space = color_space.strip().upper()
    print_section(
        "BƯỚC 5: TÁI TẠO ẢNH PHÂN ĐOẠN VÀ TRÍCH XUẤT RANH GIỚI CỤM (TASK 4)",
        f"Phục hồi từ {resized_shape[1]}x{resized_shape[0]} lên {original_shape[1]}x{original_shape[0]} ({norm_space})",
    )
    segmenter = ImageSegmenter()
    centroids = final_model.cluster_centers_
    rgb_centroids = convert_centroids_to_rgb(centroids, source_space=norm_space)
    labels_resized = final_model.labels_

    segmented_resized = segmenter.reconstruct_segmented_image(
        labels_resized, rgb_centroids, target_shape=resized_shape
    )

    if img_rgb is not None:
        # Chuyển ảnh gốc sang không gian màu làm việc (nếu khác RGB) rồi dự đoán theo từng lô
        feature_full = (
            convert_color_space(img_rgb, source_space="RGB", target_space=norm_space)
            if norm_space != "RGB"
            else img_rgb
        )
        orig_pixels = flatten_pixels(feature_full, as_float=True)
        n_orig = orig_pixels.shape[0]
        labels_orig = np.empty(n_orig, dtype=np.int64)
        chunk_size = 250000
        for start in range(0, n_orig, chunk_size):
            end = min(start + chunk_size, n_orig)
            labels_orig[start:end] = final_model.predict(orig_pixels[start:end])

        segmented_orig = segmenter.reconstruct_segmented_image(
            labels_orig,
            rgb_centroids,
            target_shape=(original_shape[0], original_shape[1]),
        )
    else:
        segmented_orig = segmenter.reconstruct_original_size(
            labels_resized, rgb_centroids, original_shape=original_shape, resized_shape=resized_shape
        )
        labels_orig = labels_resized

    # Kiểm chứng mặt nạ phân vùng và lưu ảnh đường ranh giới cụm tại kích thước làm việc
    masks = segmenter.extract_cluster_masks(
        labels_resized, shape=resized_shape, k=len(centroids)
    )
    boundaries = segmenter.extract_cluster_boundaries(labels_resized, shape=resized_shape)
    boundary_path = FIGURES_DIR / "cluster_boundaries.png"
    ok, buf = cv2.imencode(".png", boundaries)
    if ok:
        buf.tofile(str(boundary_path))

    print(
        f"Đã tái tạo ảnh phân đoạn kích thước gốc {segmented_orig.shape[1]}x{segmented_orig.shape[0]} "
        f"| Số mặt nạ cụm hợp lệ: {len(masks)}"
    )
    logger.info("Hoàn tất Bước 5: Tái tạo ảnh phân đoạn và mặt nạ ranh giới thành công.")
    return segmented_resized, segmented_orig, labels_orig, rgb_centroids


def step_6_visualize_side_by_side(
    img_rgb: np.ndarray,
    segmented_img_orig: np.ndarray,
    optimal_k: int,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> list:
    """
    Bước 6 (Đáp ứng Task 5 - Hiển thị song song và thống kê màu sắc):
    - Vẽ ảnh gốc cạnh ảnh phân đoạn tại results/figures/final_segmentation_side_by_side.png.
    - Trích xuất phân bố tỷ lệ % từng cụm màu, vẽ biểu đồ cột ngang và lưu cluster_stats.json.
    """
    print_section(
        "BƯỚC 6: HIỂN THỊ SONG SONG VÀ THỐNG KÊ PHÂN BỐ MÀU (TASK 5)",
        f"Số cụm màu phân đoạn: K = {optimal_k}",
    )
    side_by_side_path = FIGURES_DIR / "final_segmentation_side_by_side.png"
    plot_side_by_side(
        img_rgb,
        segmented_img_orig,
        k=optimal_k,
        title=f"Original Image vs K-Means Segmentation (K={optimal_k})",
        save_path=side_by_side_path,
    )

    cluster_stats = extract_color_distribution(labels, centroids)
    dist_fig_path = FIGURES_DIR / "cluster_distribution.png"
    plot_cluster_distribution(cluster_stats, save_path=dist_fig_path)

    stats_json_path = METRICS_DIR / "cluster_stats.json"
    stats_json_path.write_text(
        json.dumps(cluster_stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nBảng phân bố cụm màu chủ đạo:")
    for item in cluster_stats:
        print(
            f"  - Cụm #{item['cluster_id']}: {item['color_name']:28s} | "
            f"HEX={item['hex_color']} | Tỷ lệ={item['share_percentage']:6.2f}% "
            f"({item['pixel_count']:,} pixels)"
        )

    logger.info("Hoàn tất Bước 6: Đã xuất ảnh side-by-side, biểu đồ và JSON phân bố cụm màu.")
    return cluster_stats


def compare_color_space_segmentations(
    resized_rgb: np.ndarray,
    optimal_k: int,
    primary_space: str,
    primary_labels: np.ndarray,
    primary_centroids: np.ndarray,
    primary_silhouette: float,
    primary_fit_time: float,
) -> List[Dict[str, Any]]:
    """
    Thực nghiệm so sánh định lượng và trực quan kết quả K-Means trên 3 không gian màu RGB, HSV và CIE LAB.
    """
    segmenter = ImageSegmenter()
    trainer = SklearnKMeansTrainer(random_state=RANDOM_SEED)
    target_shape = (resized_rgb.shape[0], resized_rgb.shape[1])
    norm_primary = primary_space.strip().upper()

    visual_entries: List[Dict[str, Any]] = []
    json_records: List[Dict[str, Any]] = []

    for space in ("RGB", "HSV", "LAB"):
        if space == norm_primary:
            labels = primary_labels
            centroids = primary_centroids
            sil = primary_silhouette
            fit_time = primary_fit_time
        else:
            space_img = (
                resized_rgb
                if space == "RGB"
                else convert_color_space(resized_rgb, source_space="RGB", target_space=space)
            )
            space_pixels = flatten_pixels(space_img, as_float=True)
            t0 = time.perf_counter()
            model = trainer.fit_final_model(
                space_pixels, optimal_k=optimal_k, n_init=N_INIT, max_iter=MAX_ITER
            )
            fit_time = max(time.perf_counter() - t0, 1e-6)
            labels = model.labels_
            centroids = model.cluster_centers_
            sil = compute_silhouette(
                space_pixels, labels, max_samples=5000, random_state=RANDOM_SEED
            )

        rgb_centers = convert_centroids_to_rgb(centroids, source_space=space)
        seg_rgb = segmenter.reconstruct_segmented_image(labels, rgb_centers, target_shape=target_shape)
        visual_entries.append(
            {
                "color_space": space,
                "segmented_rgb": seg_rgb,
                "silhouette": float(sil),
                "fit_time_sec": float(fit_time),
            }
        )
        json_records.append(
            {
                "color_space": space,
                "k": int(optimal_k),
                "silhouette_score": round(float(sil), 4),
                "fit_time_sec": round(float(fit_time), 4),
            }
        )

    plot_color_space_segmentation_comparison(
        resized_rgb,
        visual_entries,
        k=optimal_k,
        save_path=FIGURES_DIR / "color_space_segmentation_comparison.png",
    )
    (METRICS_DIR / "color_space_comparison.json").write_text(
        json.dumps(json_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Đã hoàn tất đối chiếu phân cụm trên 3 không gian màu RGB, HSV và CIE LAB.")
    return json_records


def step_7_advanced_benchmarks_and_save(
    final_model: KMeans,
    pixel_matrix: np.ndarray,
    metadata: Dict[str, Any],
    resized_rgb: Optional[np.ndarray] = None,
    resized_shape: Optional[Tuple[int, int]] = None,
    final_fit_time: Optional[float] = None,
    color_space: str = "RGB",
    run_color_space_benchmark: bool = True,
) -> None:
    """
    Bước 7: Kiểm định chéo đa thư viện, đo đánh đổi MiniBatch kèm ảnh đối chiếu,
    so sánh không gian màu, lưu mô hình và thử nghiệm suy luận trên ảnh khác.
    """
    norm_space = color_space.strip().upper()
    print_section(
        "BƯỚC 7: KIỂM ĐỊNH NÂNG CAO, LƯU MÔ HÌNH & SUY LUẬN ẢNH MỚI",
        "Đối chiếu OpenCV cv2.kmeans, MiniBatchKMeans, Không gian màu, Silhouette Score & Predictor",
    )
    optimal_k = int(final_model.n_clusters)

    # 1. Kiểm định chéo với OpenCV K-Means (kèm đo thời gian thực thi C++ vs Scikit-Learn)
    cv_trainer = OpenCVKMeansTrainer()
    cv_compactness, cv_labels, _ = cv_trainer.train(
        pixel_matrix, k=optimal_k, attempts=N_INIT, max_iter=MAX_ITER
    )
    cv_report = cv_trainer.compare_with_sklearn(
        sklearn_labels=final_model.labels_,
        opencv_labels=cv_labels,
        sklearn_inertia=float(final_model.inertia_),
        opencv_compactness=cv_compactness,
        sklearn_time_sec=final_fit_time,
        opencv_time_sec=cv_trainer.last_fit_time_sec_,
    )
    (METRICS_DIR / "opencv_comparison.json").write_text(
        json.dumps(cv_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 2. Đánh giá đánh đổi tốc độ / chất lượng với MiniBatchKMeans và xuất ảnh đối chiếu trực quan
    mb_trainer = MiniBatchKMeansTrainer(random_state=RANDOM_SEED)
    df_mb = mb_trainer.compare_batch_sizes(
        pixel_matrix,
        k=optimal_k,
        full_kmeans_model=final_model,
        batch_sizes=[256, 512, 1024, 2048, 4096],
    )
    df_mb.to_csv(METRICS_DIR / "minibatch_comparison.csv", index=False)

    if resized_shape is not None:
        segmenter = ImageSegmenter()
        full_rgb_centers = convert_centroids_to_rgb(
            final_model.cluster_centers_, source_space=norm_space
        )
        full_row = df_mb[df_mb["model"] == "Full KMeans"].iloc[0]
        visual_panels: List[Dict[str, Any]] = [
            {
                "label": "Full K-Means",
                "image": segmenter.reconstruct_segmented_image(
                    final_model.labels_, full_rgb_centers, target_shape=resized_shape
                ),
                "fit_time_sec": float(full_row["fit_time_sec"]),
                "speedup": float(full_row["speedup"]),
                "inertia_increase_pct": float(full_row["inertia_increase_pct"]),
                "ari": float(full_row["ari"]),
            }
        ]
        for preview_bs in (256, 1024, 4096):
            row_match = df_mb[df_mb["batch_size"] == preview_bs]
            if row_match.empty:
                continue
            row = row_match.iloc[0]
            mb_vis = MiniBatchKMeans(
                n_clusters=optimal_k,
                batch_size=min(preview_bs, pixel_matrix.shape[0]),
                n_init=N_INIT,
                max_iter=MAX_ITER,
                random_state=RANDOM_SEED,
            )
            mb_labels = mb_vis.fit_predict(pixel_matrix)
            mb_rgb_centers = convert_centroids_to_rgb(
                mb_vis.cluster_centers_, source_space=norm_space
            )
            visual_panels.append(
                {
                    "label": f"MiniBatch (b={preview_bs})",
                    "image": segmenter.reconstruct_segmented_image(
                        mb_labels, mb_rgb_centers, target_shape=resized_shape
                    ),
                    "fit_time_sec": float(row["fit_time_sec"]),
                    "speedup": float(row["speedup"]),
                    "inertia_increase_pct": float(row["inertia_increase_pct"]),
                    "ari": float(row["ari"]),
                }
            )
        plot_minibatch_comparison(
            visual_panels,
            k=optimal_k,
            save_path=FIGURES_DIR / "minibatch_visual_comparison.png",
        )

    # 3. Tính hệ số Silhouette Score trên mẫu phân tầng
    sil_score = compute_silhouette(
        pixel_matrix,
        final_model.labels_,
        max_samples=5000,
        random_state=RANDOM_SEED,
    )

    # 3b. So sánh thực nghiệm trên RGB, HSV và CIE LAB nếu được bật
    cs_records: List[Dict[str, Any]] = []
    if run_color_space_benchmark and resized_rgb is not None:
        cs_records = compare_color_space_segmentations(
            resized_rgb=resized_rgb,
            optimal_k=optimal_k,
            primary_space=norm_space,
            primary_labels=final_model.labels_,
            primary_centroids=final_model.cluster_centers_,
            primary_silhouette=sil_score,
            primary_fit_time=float(final_fit_time or 0.0),
        )

    # 4. Lưu mô hình K-Means cuối cùng và metadata cấu hình vào đĩa
    model_path = SAVED_MODELS_DIR / "kmeans_segmentation.joblib"
    save_model_pipeline(
        final_model,
        save_path=model_path,
        metadata={
            "optimal_k": optimal_k,
            "silhouette_score": round(float(sil_score), 4),
            "wcss_resized": round(float(final_model.inertia_), 2),
            "opencv_agreement_pct": cv_report["label_agreement_pct"],
            "opencv_fit_time_sec": cv_report.get("opencv_fit_time_sec"),
            "color_space": norm_space,
            "random_seed": RANDOM_SEED,
            "source_image": metadata.get("file_name", ""),
        },
    )

    # 5. Chạy thử nghiệm ImageSegmentationPredictor trên ảnh trong data/raw/
    test_candidates = [
        p
        for p in sorted(RAW_DATA_DIR.glob("*.png")) + sorted(RAW_DATA_DIR.glob("*.jpg"))
        if p.name != metadata.get("file_name")
    ]
    inference_img = test_candidates[0] if test_candidates else Path(DEFAULT_IMAGE_PATH)
    if inference_img.name != metadata.get("file_name"):
        logger.warning(
            f"Lưu ý giới hạn miền màu (Domain Shift): đang áp dụng bộ tâm màu K={optimal_k} học từ "
            f"'{metadata.get('file_name')}' lên ảnh khác '{inference_img.name}' để kiểm chứng năng lực suy luận "
            f"(Cross-Image Palette Transfer); màu đầu ra sẽ bị ánh xạ về bảng màu của ảnh gốc."
        )
    predictor = ImageSegmentationPredictor(model_path=model_path)
    pred_result = predictor.predict(
        inference_img,
        output_dir=FIGURES_DIR / "inference_preview",
        target_size=TARGET_SIZE,
        return_original_size=True,
    )

    cv_time_text = (
        f" | OpenCV={cv_report['opencv_fit_time_sec']:.4f}s"
        if "opencv_fit_time_sec" in cv_report
        else ""
    )
    cs_summary = (
        " | ".join(f"{r['color_space']}:Sil={r['silhouette_score']:.3f}" for r in cs_records)
        if cs_records
        else "Bỏ qua"
    )
    print(
        f"Kết quả kiểm định nâng cao:\n"
        f"  - Silhouette Score ({norm_space}, K={optimal_k}): {sil_score:.4f}\n"
        f"  - Tương đồng Scikit-Learn vs OpenCV: {cv_report['label_agreement_pct']}% "
        f"(chênh lệch WCSS: {cv_report['inertia_relative_diff_pct']}%{cv_time_text})\n"
        f"  - Đối chiếu không gian màu: {cs_summary}\n"
        f"  - Đã lưu mô hình tại: {model_path}\n"
        f"  - Đã suy luận thử nghiệm trên ảnh: {pred_result['original_metadata']['file_name']} "
        f"(lưu ý: gán màu theo bảng {optimal_k} tâm màu đã học từ {metadata.get('file_name', '')})"
    )
    logger.info("Hoàn tất Bước 7: Toàn bộ kiểm định nâng cao, lưu mô hình và suy luận hoàn thành.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Phân tích tham số dòng lệnh để chạy pipeline linh hoạt với ảnh, số cụm K và hệ màu tùy chọn.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline phân đoạn ảnh bằng K-Means (Practice Exercise 3.2)."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Đường dẫn file ảnh đầu vào (mặc định: ảnh cấu hình trong src/config.py).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Số cụm K cố định (mặc định: tự động chọn bằng Kneedle Elbow Method).",
    )
    parser.add_argument(
        "--color-space",
        type=str,
        default="RGB",
        choices=["RGB", "HSV", "LAB", "rgb", "hsv", "lab"],
        help="Không gian màu dùng để phân cụm (RGB, HSV hoặc LAB; mặc định: RGB).",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=TARGET_SIZE,
        help=f"Kích thước cạnh dài tối đa khi thu nhỏ ảnh (mặc định: {TARGET_SIZE}).",
    )
    parser.add_argument(
        "--skip-color-space-benchmark",
        action="store_true",
        help="Bỏ qua bước đối chiếu thực nghiệm đa không gian màu RGB/HSV/LAB ở Bước 7.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """
    Hàm điều phối chính chạy tuần tự toàn bộ 7 bước pipeline phân đoạn ảnh bằng K-Means.
    """
    args = parse_args(argv)
    chosen_space = args.color_space.strip().upper()
    if args.k is not None and args.k < 1:
        raise ValueError(f"--k must be a positive integer, got {args.k}.")
    if args.target_size < 1:
        raise ValueError(f"--target-size must be a positive integer, got {args.target_size}.")

    print_section(
        "BẮT ĐẦU PIPELINE PHÂN ĐOẠN ẢNH BẰNG K-MEANS",
        f"Thực thi Practice Exercise 3.2 | Hệ màu: {chosen_space} | Resize <= {args.target_size} px",
    )
    logger.info(
        f"Bắt đầu chạy toàn bộ pipeline K-Means Image Segmentation (color_space={chosen_space}, "
        f"target_size={args.target_size}, fixed_k={args.k})..."
    )

    # Bước 1: Nạp ảnh và chuyển đổi không gian màu
    img_rgb, metadata = step_1_load_and_color_spaces(image_path=args.image)

    # Bước 2: Tiền xử lý (resize giữ tỷ lệ, hệ màu làm việc, flatten 2D pixel matrix)
    resized_rgb, resized_shape, pixel_matrix = step_2_preprocess_and_flatten(
        img_rgb,
        metadata,
        target_size=args.target_size,
        color_space=chosen_space,
    )

    # Bước 3: Áp dụng K-Means thuần từ đầu và ghi nhận lịch sử hội tụ
    step_3_kmeans_from_scratch(
        pixel_matrix,
        resized_rgb=resized_rgb,
        color_space=chosen_space,
    )

    # Bước 4: Khảo sát Elbow Method tìm K tối ưu và huấn luyện mô hình cuối
    final_model, optimal_k, final_fit_time = step_4_elbow_method_optimal_k(
        pixel_matrix,
        resized_shape=resized_shape,
        fixed_k=args.k,
        color_space=chosen_space,
    )

    # Bước 5: Phân đoạn ảnh và phục hồi kích thước gốc
    _, segmented_orig, labels_orig, rgb_centroids = step_5_segment_and_reconstruct(
        final_model,
        pixel_matrix,
        resized_shape=resized_shape,
        original_shape=img_rgb.shape,
        img_rgb=img_rgb,
        color_space=chosen_space,
    )

    # Bước 6: Trực quan hóa ảnh gốc vs phân đoạn song song và thống kê màu sắc
    step_6_visualize_side_by_side(
        img_rgb,
        segmented_orig,
        optimal_k=optimal_k,
        labels=labels_orig,
        centroids=rgb_centroids,
    )

    # Bước 7: Kiểm định chéo OpenCV, MiniBatchKMeans, Không gian màu, Silhouette, lưu mô hình & suy luận
    step_7_advanced_benchmarks_and_save(
        final_model,
        pixel_matrix,
        metadata,
        resized_rgb=resized_rgb,
        resized_shape=resized_shape,
        final_fit_time=final_fit_time,
        color_space=chosen_space,
        run_color_space_benchmark=not args.skip_color_space_benchmark,
    )

    print_section(
        "HOÀN TẤT PIPELINE THÀNH CÔNG!",
        "Toàn bộ biểu đồ, số liệu và mô hình đã được xuất ra thư mục results/",
    )
    logger.info("Pipeline phân đoạn ảnh K-Means đã kết thúc thành công.")


if __name__ == "__main__":
    main()

"""
=============================================================================
DỰ ÁN: PHÂN ĐOẠN ẢNH BẰNG K-MEANS (K-MEANS IMAGE SEGMENTATION PIPELINE)
TẬP TIN ĐIỀU PHỐI CHÍNH (MAIN PIPELINE ENTRYPOINT)
=============================================================================
Theo đúng yêu cầu của đề bài: Practice Exercise 3 (Additional) - Practice Exercise 3.2.pdf
Các bước thực hiện trong pipeline:
  - BƯỚC 1: Load the Image & Convert Color Space (Task 1: OpenCV/PIL, BGR → RGB/HSV/LAB).
  - BƯỚC 2: Preprocess Image (Task 2: Resize về ≤256px cạnh dài, Flatten thành 2D pixel matrix).
  - BƯỚC 3: Apply K-Means from Scratch (Task 3: Cài đặt K-Means thuần và theo dõi hội tụ).
  - BƯỚC 4: Elbow Method & Optimal K Selection (Task 3: Quét K=2..12, xác định Elbow K).
  - BƯỚC 5: Segment the Image (Task 4: Thay thế pixel bằng tâm màu, reshape về kích thước ảnh).
  - BƯỚC 6: Visualize Results Side-by-Side (Task 5: Hiển thị ảnh gốc vs ảnh phân đoạn).
  - BƯỚC 7: Advanced Validation & Deployment (OpenCV, MiniBatch, Silhouette, Lưu mô hình).
=============================================================================
"""

import sys
import json
from pathlib import Path

# Thêm thư mục gốc vào đường dẫn hệ thống để import trơn tru
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    print_section,
    DEFAULT_IMAGE_PATH,
    FIGURES_DIR,
    METRICS_DIR,
    SAVED_MODELS_DIR,
    TARGET_SIZE,
    BASELINE_K,
    FINAL_K,
    K_SEARCH_RANGE,
    RANDOM_SEED,
)
from src.utils.logger import get_logger

logger = get_logger("MainPipeline")


# ===========================================================================
# BƯỚC 1: Load ảnh & chuyển đổi không gian màu (Task 1)
# ===========================================================================
def step_1_load_and_color_spaces():
    """
    1. Đọc ảnh đầu vào từ DEFAULT_IMAGE_PATH bằng load_image().
    2. Kiểm tra tính toàn vẹn ảnh (validate_image).
    3. Chuyển đổi sang RGB, HSV, LAB bằng convert_color_space().
    4. Trực quan hóa 3 không gian màu → lưu results/figures/color_spaces_comparison.png.
    5. Trả về img_rgb và metadata.
    """
    from src.data.loader import load_image
    from src.data.preprocessing import convert_color_space
    from src.evaluation.visualization import plot_color_space_comparison

    print_section("BƯỚC 1: NẠP ẢNH & CHUYỂN ĐỔI KHÔNG GIAN MÀU", f"Ảnh: {DEFAULT_IMAGE_PATH}")
    logger.info(f"Đọc ảnh từ: {DEFAULT_IMAGE_PATH}")

    if not DEFAULT_IMAGE_PATH.exists():
        logger.error(
            f"Không tìm thấy ảnh tại: {DEFAULT_IMAGE_PATH}\n"
            f"Hãy đặt ít nhất một ảnh (.jpg/.png) vào thư mục: {DEFAULT_IMAGE_PATH.parent}"
        )
        raise FileNotFoundError(f"Không tìm thấy ảnh: {DEFAULT_IMAGE_PATH}")

    img_rgb, metadata = load_image(DEFAULT_IMAGE_PATH)
    logger.info(
        f"Ảnh đã nạp: {metadata['filename']} | "
        f"{metadata['original_height']}×{metadata['original_width']} | "
        f"{metadata['total_pixels']:,} pixels"
    )

    # Chuyển đổi không gian màu (RGB là baseline; HSV, LAB là khảo sát)
    img_hsv = convert_color_space(img_rgb, "RGB", "HSV")
    img_lab = convert_color_space(img_rgb, "RGB", "LAB")
    logger.info("Đã chuyển đổi: RGB → HSV, RGB → LAB")

    # Trực quan hóa
    plot_color_space_comparison(
        img_rgb, img_hsv, img_lab,
        save_path=FIGURES_DIR / "color_spaces_comparison.png",
    )

    logger.info("✔ Bước 1 hoàn tất.")
    return img_rgb, metadata


# ===========================================================================
# BƯỚC 2: Resize & Flatten (Task 2)
# ===========================================================================
def step_2_preprocess_and_flatten(img_rgb):
    """
    1. Resize ảnh về TARGET_SIZE (giữ tỷ lệ khung hình).
    2. Flatten thành ma trận pixel 2D (N, 3) kiểu float32.
    3. Ghi log kích thước.
    4. Trả về (img_resized, resized_shape, pixel_matrix).
    """
    from src.data.preprocessing import resize_image, flatten_pixels

    print_section("BƯỚC 2: TIỀN XỬ LÝ ẢNH", f"Resize cạnh dài ≤ {TARGET_SIZE}px → Flatten")

    img_resized, resized_shape = resize_image(img_rgb, target_size=TARGET_SIZE, keep_aspect_ratio=True)
    pixel_matrix = flatten_pixels(img_resized, as_float=True)

    logger.info(
        f"Resize: {img_rgb.shape[:2]} → {resized_shape} | "
        f"Pixel matrix: {pixel_matrix.shape} | dtype={pixel_matrix.dtype}"
    )
    logger.info("✔ Bước 2 hoàn tất.")
    return img_resized, resized_shape, pixel_matrix


# ===========================================================================
# BƯỚC 3: K-Means from Scratch (Task 3 — Baseline)
# ===========================================================================
def step_3_kmeans_from_scratch(pixel_matrix):
    """
    1. Khởi tạo KMeansFromScratch(n_clusters=BASELINE_K).
    2. Huấn luyện fit(pixel_matrix) theo đúng 4 bước.
    3. Trích xuất history_df và vẽ đồ thị hội tụ.
    4. Trả về (scratch_model, history_df).
    """
    from src.models.kmeans_scratch import KMeansFromScratch
    from src.evaluation.visualization import plot_convergence_tracking

    print_section(
        "BƯỚC 3: K-MEANS TỰ LẬP TRÌNH (FROM SCRATCH)",
        f"K={BASELINE_K} | Thuần NumPy | Theo dõi hội tụ"
    )

    model = KMeansFromScratch(
        n_clusters=BASELINE_K,
        max_iter=300,
        tol=1e-4,
        random_state=RANDOM_SEED,
    )
    model.fit(pixel_matrix)

    logger.info(
        f"K-Means Scratch | n_iter={model.n_iter_} | "
        f"WCSS={model.inertia_:.2f} | converged={model.converged_}"
    )

    history_df = model.get_convergence_history()
    logger.info(f"Lịch sử hội tụ: {len(history_df)} vòng lặp")

    plot_convergence_tracking(
        history_df,
        save_path=FIGURES_DIR / "kmeans_convergence.png",
    )

    logger.info("✔ Bước 3 hoàn tất.")
    return model, history_df


# ===========================================================================
# BƯỚC 4: Elbow Method & Optimal K (Task 3 — Sklearn)
# ===========================================================================
def step_4_elbow_method_optimal_k(pixel_matrix):
    """
    1. Quét K từ K_SEARCH_RANGE bằng SklearnKMeansTrainer.
    2. Vẽ đường cong Elbow và lưu.
    3. Huấn luyện mô hình cuối với optimal_k.
    4. Trả về (final_model, optimal_k).
    """
    from src.models.kmeans_sklearn import SklearnKMeansTrainer
    from src.evaluation.visualization import plot_elbow_curve

    print_section(
        "BƯỚC 4: ELBOW METHOD & XÁC ĐỊNH K TỐI ƯU",
        f"Quét K ∈ [{K_SEARCH_RANGE.start}, {K_SEARCH_RANGE.stop - 1}]"
    )

    trainer = SklearnKMeansTrainer(random_state=RANDOM_SEED)
    df_elbow, optimal_k = trainer.search_elbow(pixel_matrix, k_range=K_SEARCH_RANGE)

    # Lưu bảng Elbow
    elbow_csv = METRICS_DIR / "elbow_results.csv"
    df_elbow.to_csv(elbow_csv, index=False)
    logger.info(f"Bảng Elbow đã lưu: {elbow_csv}")

    plot_elbow_curve(
        df_elbow, optimal_k,
        save_path=FIGURES_DIR / "elbow_method.png",
    )

    # Huấn luyện mô hình cuối
    final_model = trainer.fit_final_model(pixel_matrix, optimal_k=optimal_k)
    logger.info(
        f"Mô hình cuối | K={optimal_k} | WCSS={final_model.inertia_:.2f} | "
        f"n_iter={final_model.n_iter_}"
    )

    logger.info("✔ Bước 4 hoàn tất.")
    return final_model, optimal_k


# ===========================================================================
# BƯỚC 5: Segment the Image (Task 4)
# ===========================================================================
def step_5_segment_and_reconstruct(final_model, pixel_matrix, resized_shape, original_shape):
    """
    1. Lấy labels_ và cluster_centers_ từ final_model.
    2. Tái tạo ảnh tại kích thước làm việc.
    3. Tái tạo ảnh tại kích thước gốc bằng nearest-neighbor upscale.
    4. Trả về (segmented_resized, segmented_orig, labels, centroids).
    """
    from src.segmentation.segmenter import ImageSegmenter

    print_section("BƯỚC 5: PHÂN ĐOẠN ẢNH & TÁI TẠO", "Ánh xạ màu tâm cụm → ảnh phân đoạn")

    labels = final_model.labels_
    centroids = final_model.cluster_centers_
    k = len(centroids)

    segmenter = ImageSegmenter()

    # Tái tạo tại kích thước làm việc
    segmented_resized = segmenter.reconstruct_segmented_image(
        labels, centroids, resized_shape
    )
    logger.info(f"Ảnh phân đoạn (resize): shape={segmented_resized.shape}")

    # Tái tạo tại kích thước gốc
    segmented_orig = segmenter.reconstruct_original_size(
        labels, centroids, original_shape, resized_shape
    )
    logger.info(f"Ảnh phân đoạn (gốc): shape={segmented_orig.shape}")

    logger.info("✔ Bước 5 hoàn tất.")
    return segmented_resized, segmented_orig, labels, centroids


# ===========================================================================
# BƯỚC 6: Visualize Side-by-Side (Task 5)
# ===========================================================================
def step_6_visualize_side_by_side(img_rgb, segmented_img_orig, optimal_k, labels, centroids):
    """
    1. Vẽ so sánh side-by-side ảnh gốc vs phân đoạn → lưu.
    2. Thống kê phân bố màu sắc.
    3. Vẽ biểu đồ phân bố màu sắc → lưu.
    4. Lưu cluster_stats vào JSON.
    """
    from src.features.color_features import extract_color_distribution
    from src.evaluation.visualization import plot_side_by_side, plot_cluster_distribution
    import json

    print_section("BƯỚC 6: TRỰC QUAN HÓA KẾT QUẢ SIDE-BY-SIDE", f"K={optimal_k}")

    # Side-by-side
    plot_side_by_side(
        img_rgb,
        segmented_img_orig,
        k=optimal_k,
        title=f"K-Means Image Segmentation — K={optimal_k}",
        save_path=FIGURES_DIR / "final_segmentation_side_by_side.png",
    )

    # Phân bố màu sắc
    cluster_stats = extract_color_distribution(labels, centroids)
    logger.info("Phân bố màu sắc theo cụm:")
    for s in cluster_stats:
        logger.info(
            f"  Cụm {s['cluster_id']}: {s['color_name']} | "
            f"RGB={tuple(s['centroid_rgb'])} | {s['share_percentage']:.1f}%"
        )

    plot_cluster_distribution(
        cluster_stats,
        save_path=FIGURES_DIR / "cluster_distribution.png",
    )

    # Lưu JSON
    stats_json = METRICS_DIR / "cluster_stats.json"
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(cluster_stats, f, ensure_ascii=False, indent=2)
    logger.info(f"Cluster stats đã lưu: {stats_json}")

    logger.info("✔ Bước 6 hoàn tất.")
    return cluster_stats


# ===========================================================================
# BƯỚC 7: Advanced Benchmarks & Deployment
# ===========================================================================
def step_7_advanced_benchmarks_and_save(final_model, pixel_matrix, metadata, optimal_k):
    """
    1. Đánh giá chéo với OpenCV K-Means → lưu JSON.
    2. Benchmark MiniBatchKMeans vs Full K-Means → lưu CSV.
    3. Tính Silhouette Score (subsample).
    4. Lưu pipeline mô hình vào saved_models/.
    5. Thử nghiệm Predictor inference.
    """
    from src.models.kmeans_opencv import OpenCVKMeansTrainer
    from src.models.minibatch_kmeans import MiniBatchKMeansTrainer
    from src.evaluation.metrics import compute_silhouette, compute_wcss
    from src.models.persistence import save_model_pipeline
    from src.deployment.predictor import ImageSegmentationPredictor

    print_section("BƯỚC 7: ĐÁNH GIÁ NÂNG CAO & LƯU MÔ HÌNH")

    sklearn_labels = final_model.labels_
    sklearn_inertia = float(final_model.inertia_)
    sklearn_centroids = final_model.cluster_centers_

    # --- 7.1 OpenCV Cross-Validation ---
    try:
        opencv_trainer = OpenCVKMeansTrainer()
        compactness, opencv_labels, opencv_centers = opencv_trainer.train(
            pixel_matrix, k=optimal_k
        )
        comparison_report = opencv_trainer.compare_with_sklearn(
            sklearn_labels, opencv_labels,
            sklearn_inertia, compactness,
        )
        opencv_json = METRICS_DIR / "opencv_comparison.json"
        with open(opencv_json, "w", encoding="utf-8") as f:
            json.dump(comparison_report, f, ensure_ascii=False, indent=2)
        logger.info(f"OpenCV comparison report lưu tại: {opencv_json}")
    except Exception as e:
        logger.warning(f"OpenCV cross-validation bỏ qua (lỗi: {e})")

    # --- 7.2 MiniBatch Benchmark ---
    try:
        mb_trainer = MiniBatchKMeansTrainer(random_state=RANDOM_SEED)
        df_mb = mb_trainer.compare_batch_sizes(
            pixel_matrix, k=optimal_k, full_kmeans_model=final_model,
        )
        mb_csv = METRICS_DIR / "minibatch_comparison.csv"
        df_mb.to_csv(mb_csv, index=False)
        logger.info(f"MiniBatch benchmark lưu tại: {mb_csv}\n{df_mb.to_string(index=False)}")
    except Exception as e:
        logger.warning(f"MiniBatch benchmark bỏ qua (lỗi: {e})")

    # --- 7.3 Silhouette Score ---
    try:
        sil_score = compute_silhouette(pixel_matrix, sklearn_labels, max_samples=5000)
        logger.info(f"Silhouette Score (K={optimal_k}): {sil_score:.4f}")
        wcss_check = compute_wcss(pixel_matrix, sklearn_centroids, sklearn_labels)
        logger.info(f"WCSS (kiểm tra nhất quán): {wcss_check:.2f} vs inertia_={sklearn_inertia:.2f}")

        sil_json = METRICS_DIR / "silhouette_score.json"
        with open(sil_json, "w", encoding="utf-8") as f:
            json.dump({
                "optimal_k": optimal_k,
                "silhouette_score": sil_score,
                "wcss": wcss_check,
                "sklearn_inertia": sklearn_inertia,
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Silhouette bỏ qua (lỗi: {e})")

    # --- 7.4 Lưu mô hình ---
    model_path = SAVED_MODELS_DIR / "kmeans_segmentation.joblib"
    save_model_pipeline(
        model=final_model,
        save_path=model_path,
        metadata={
            "optimal_k": optimal_k,
            "centroids": sklearn_centroids.tolist(),
            "inertia": sklearn_inertia,
            "image_filename": metadata.get("filename", "unknown"),
            "target_size": TARGET_SIZE,
        },
    )

    # --- 7.5 Thử nghiệm Predictor ---
    try:
        predictor = ImageSegmentationPredictor(model_path=model_path)
        result = predictor.predict(
            image_path=DEFAULT_IMAGE_PATH,
            output_dir=FIGURES_DIR / "predictor_output",
            target_size=TARGET_SIZE,
            return_original_size=True,
        )
        logger.info(
            f"Predictor inference OK | "
            f"segmented shape={result['segmented_img'].shape}"
        )
    except Exception as e:
        logger.warning(f"Predictor inference bỏ qua (lỗi: {e})")

    logger.info("✔ Bước 7 hoàn tất.")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    """
    Hàm điều phối chính: Gọi tuần tự từ Bước 1 đến Bước 7.
    """
    print_section(
        "BẮT ĐẦU PIPELINE PHÂN ĐOẠN ẢNH BẰNG K-MEANS",
        "Dựa theo yêu cầu đề bài: Practice Exercise 3.2 (K-Means Image Segmentation)"
    )
    logger.info("Khởi chạy pipeline...")

    # Bước 1: Load ảnh & chuyển đổi không gian màu
    img_rgb, metadata = step_1_load_and_color_spaces()
    original_shape = metadata["original_shape"]  # (H, W, 3)

    # Bước 2: Resize & Flatten
    img_resized, resized_shape, pixel_matrix = step_2_preprocess_and_flatten(img_rgb)

    # Bước 3: K-Means thuần NumPy (Baseline)
    scratch_model, history_df = step_3_kmeans_from_scratch(pixel_matrix)

    # Bước 4: Elbow Method → Optimal K
    final_model, optimal_k = step_4_elbow_method_optimal_k(pixel_matrix)

    # Bước 5: Phân đoạn ảnh & Tái tạo
    segmented_resized, segmented_orig, labels, centroids = step_5_segment_and_reconstruct(
        final_model, pixel_matrix, resized_shape, original_shape
    )

    # Bước 6: Trực quan hóa side-by-side
    cluster_stats = step_6_visualize_side_by_side(
        img_rgb, segmented_orig, optimal_k, labels, centroids
    )

    # Bước 7: Đánh giá nâng cao & Lưu mô hình
    step_7_advanced_benchmarks_and_save(final_model, pixel_matrix, metadata, optimal_k)

    print_section(
        "HOÀN TẤT PIPELINE THÀNH CÔNG!",
        f"Toàn bộ kết quả, biểu đồ và mô hình đã được xuất ra thư mục results/"
    )
    logger.info(
        f"Kết quả pipeline:\n"
        f"  - Figures: {FIGURES_DIR}\n"
        f"  - Metrics: {METRICS_DIR}\n"
        f"  - Models:  {SAVED_MODELS_DIR}"
    )


if __name__ == "__main__":
    main()

# Báo Cáo Thực Nghiệm Bài Lab: Phân Đoạn Ảnh Bằng K-Means (K-Means Image Segmentation)

- **Nguồn yêu cầu:** `docs/Practice Exercise 3.2.pdf`
- **Ảnh đầu vào chính:** `data/raw/image-liquid.jpg` (`3264 x 3264 px`, `10,653,696` pixels, `2095.23 KB`)
- **Cấu hình thực nghiệm:** `TARGET_SIZE = 256`, `BASELINE_K = 4`, `FINAL_K = 5`, `MAX_ITER = 300`, `TOLERANCE = 1e-4`, `RANDOM_SEED = 42`

---

## 1. Tiền Xử Lý Và Chất Lượng Dữ Liệu Đầu Vào

- **Thu nhỏ giữ tỷ lệ khung hình:** `3264 x 3264` (`10,653,696 px`) -> `256 x 256` (`65,536 px`), giảm **99.38%** số lượng điểm ảnh tính toán nhưng vẫn bảo toàn cấu trúc mảng màu tổng thể.
- **Thống kê không gian RGB (`results/metrics/data_quality_stats.json`):**
  - Số màu phân biệt trên ma trận `(65536, 3)`: **53,450 màu**.
  - Độ sáng trung bình: `132.82 ± 56.07`.
  - Kênh **R**: `mean = 177.27`, `median = 203.0`, `IQR = 92.0`, `[min..max] = [3.0..253.0]`.
  - Kênh **G**: `mean = 118.06`, `median = 121.0`, `IQR = 106.0`, `[min..max] = [2.0..245.0]`.
  - Kênh **B**: `mean = 92.18`, `median = 83.0`, `IQR = 47.0`, `[min..max] = [2.0..247.0]`.
- Biểu đồ đối chiếu kênh màu lưu tại `results/figures/color_spaces_comparison.png`.

---

## 2. Kết Quả K-Means Tự Cài Đặt (Pure NumPy Baseline, K=4)

- **Mô hình:** `KMeansFromScratch` (`n_clusters=4`, `max_iter=300`, `tol=1e-4`, `random_state=42`).
- **Hội tụ:** Dừng tại vòng lặp thứ **26** với lý do `converged`, tổng quán tính cuối **WCSS = 156,330,549.40** (xấp xỉ sát mốc `156,334,928.0` của Scikit-Learn ở cùng `K=4`).
- **Trực quan hóa:** `results/figures/kmeans_convergence.png`, `results/figures/scratch_baseline_side_by_side.png`, lịch sử hội tụ tại `results/metrics/scratch_convergence_history.csv`.

---

## 3. Khảo Sát Elbow Method Và Chọn Số Cụm Tối Ưu

Quét dải `K = 2..12` bằng `SklearnKMeansTrainer` (`n_init=10`, `results/metrics/elbow_metrics.csv`):

| $K$ | Inertia (WCSS) | Số vòng lặp hội tụ | Thời gian fit (giây) |
| :-: | -------------: | :----------------: | :------------------: |
| 2   | 334,192,736.0  | 10                 | 0.1880               |
| 3   | 232,992,992.0  | 6                  | 0.2102               |
| 4   | 156,334,928.0  | 18                 | 0.2989               |
| **5** | **121,024,704.0** | **7**          | **0.3683**           |
| 6   | 104,840,664.0  | 10                 | 0.5410               |
| 7   | 90,721,552.0   | 23                 | 0.6460               |
| 8   | 78,616,960.0   | 51                 | 0.7688               |
| 9   | 70,459,792.0   | 32                 | 0.6439               |
| 10  | 64,329,860.0   | 37                 | 0.7768               |
| 11  | 58,793,260.0   | 31                 | 1.0022               |
| 12  | 54,090,324.0   | 37                 | 0.8931               |

- Thuật toán Kneedle tự động chọn điểm gập tại **`K = 5`**.
- **Silhouette Score (5,000 mẫu phân tầng, RGB, K=5):** **`0.4249`**.
- Biểu đồ Elbow và lưới tiến hóa theo `K`: `results/figures/elbow_method.png`, `results/figures/k_evolution_grid.png`.

---

## 4. Phân Bố Bảng Màu Phân Đoạn Độ Phân Giải Gốc (3264 x 3264, K=5)

Kết quả gán cụm trực tiếp trên toàn bộ **10,653,696 pixels** ảnh gốc (`results/metrics/cluster_stats.json`):

| Thứ hạng | Cluster ID | Nhãn màu định danh         | Mã HEX    | Tọa độ RGB (`uint8`) | Số pixel   | Tỷ lệ (%) |
| :------: | :--------: | :------------------------- | :-------: | :------------------- | ---------: | --------: |
| 1        | `#0`       | Cam rực (Orange)           | `#D17043` | `[209, 112, 67]`     | 3,139,461  | 29.47%    |
| 2        | `#2`       | Cam vàng (Yellow-Orange)   | `#EDBD5D` | `[237, 189, 93]`     | 2,948,244  | 27.67%    |
| 3        | `#1`       | Đỏ mận (Magenta-Red)       | `#88395A` | `[136, 57, 90]`      | 1,844,562  | 17.31%    |
| 4        | `#4`       | Tím than (Dark-Violet)     | `#28193E` | `[40, 25, 62]`       | 1,425,227  | 13.38%    |
| 5        | `#3`       | Xám nhạt (Light Gray)      | `#ACA2BE` | `[172, 162, 190]`    | 1,296,202  | 12.17%    |

Ảnh trực quan: `results/figures/final_segmentation_side_by_side.png`, `results/figures/cluster_distribution.png`, `results/figures/cluster_boundaries.png`.

---

## 5. Kiểm Định Chéo Và Phân Tích Đánh Đổi Nâng Cao

### 5.1. Scikit-Learn vs OpenCV (`cv2.kmeans`, Hungarian Label Alignment)
Từ `results/metrics/opencv_comparison.json` (`K = 5`):
- **Inertia / Compactness:** Scikit-Learn `121,024,704.0` vs OpenCV `121,043,674.52` (chênh lệch tương đối **0.0157%**).
- **Trùng khớp nhãn pixel (sau căn chỉnh Hungarian):** **99.36%** (`ARI = 0.9834`, `NMI = 0.9743`).
- **Thời gian thực thi:** OpenCV C++ **`0.0942s`** vs Scikit-Learn **`0.2684s`** (OpenCV nhanh hơn **2.85x**).

### 5.2. Đánh đổi tốc độ / chất lượng với Mini-Batch K-Means (`results/metrics/minibatch_comparison.csv`)

| Cấu hình             | Batch size | Thời gian (s) | Speedup | Inertia (WCSS) | Tăng WCSS (%) | ARI    | NMI    |
| :------------------- | ---------: | ------------: | ------: | -------------: | ------------: | -----: | -----: |
| Full KMeans          | 65,536     | 0.3304        | 1.00x   | 121,024,704.0  | +0.00%        | 1.0000 | 1.0000 |
| MiniBatch (`b=256`)  | 256        | 0.7225        | 0.46x   | 121,054,752.0  | +0.02%        | 0.9938 | 0.9869 |
| MiniBatch (`b=512`)  | 512        | 0.6831        | 0.48x   | 121,602,440.0  | +0.48%        | 0.9300 | 0.9148 |
| MiniBatch (`b=1024`) | 1,024      | 0.5978        | 0.55x   | 121,342,200.0  | +0.26%        | 0.9100 | 0.9089 |
| MiniBatch (`b=2048`) | 2,048      | 0.1524        | 2.17x   | 121,410,992.0  | +0.32%        | 0.9085 | 0.9093 |
| MiniBatch (`b=4096`) | 4,096      | 0.1397        | 2.37x   | 121,542,400.0  | +0.43%        | 0.8821 | 0.8866 |

- **Nhận xét:** Ở kích thước `256 x 256` (`65,536 px`), batch nhỏ (`256..1024`) chậm hơn Full KMeans do overhead khởi tạo và lấy mẫu `n_init=10`; từ `batch >= 2048`, Mini-Batch bắt đầu nhanh hơn rõ (`2.17x - 2.37x`) trong khi độ tăng WCSS chỉ dưới `0.5%`. Quan sát trực quan tại `results/figures/minibatch_visual_comparison.png` không thấy hiện tượng bệt/nhiễu hạt màu bất thường.

### 5.3. Đối chiếu không gian màu (`RGB` vs `HSV` vs `CIE LAB`, `K = 5`)
Từ `results/metrics/color_space_comparison.json` và `results/figures/color_space_segmentation_comparison.png`:

| Không gian màu | Silhouette Score (5,000 mẫu) | Thời gian fit (s) | Nhận xét |
| :------------- | ---------------------------: | ----------------: | :------- |
| **RGB**        | `0.4249`                     | `0.2684`          | Baseline đơn giản, bám sát đề bài, tái tạo sắc độ cân bằng. |
| **HSV**        | `0.4172`                     | `0.3240`          | Kênh Hue mang tính tuần hoàn nhưng Euclidean xử lý tuyến tính, dễ gộp nhầm vùng chuyển tiếp cam/vàng. |
| **CIE LAB**    | **`0.4552`**                 | `0.2985`          | Khoảng cách Euclidean gần với cảm nhận thị giác người nhất, đạt Silhouette cao nhất trong 3 hệ màu. |

### 5.4. Thử nghiệm suy luận sang ảnh mới (`ImageSegmentationPredictor`)
- Nạp mô hình lưu tại `results/saved_models/kmeans_segmentation.joblib` (`13.69 KB`) để suy luận trên `data/raw/image-sea.png` (`5824 x 3264`, `19,009,536 pixels`).
- **Lưu ý giới hạn miền màu (Domain Shift):** Do bảng tâm cụm đã học cố định theo tông cam/đỏ mận/tím của `image-liquid.jpg`, toàn bộ pixel của `image-sea.png` bị lượng tử hóa về 5 màu của ảnh gốc (cụm chiếm ưu thế là `#3 Xám nhạt (Light Gray)` với **64.35%** diện tích), được log cảnh báo rõ ràng trong pipeline (`results/figures/inference_preview/image-sea_comparison_k5.png`).

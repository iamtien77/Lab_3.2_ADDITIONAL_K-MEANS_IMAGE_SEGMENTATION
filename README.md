# DỰ ÁN: PHÂN ĐOẠN ẢNH BẰNG K-MEANS (K-MEANS IMAGE SEGMENTATION)

Dự án bài tập thực hành môn Học máy thực hiện phân đoạn ảnh màu thành $K$ phân vùng cụm màu sắc bằng thuật toán **K-Means Clustering** theo đúng yêu cầu của **Practice Exercise 3 (Additional) / Practice Exercise 3.2.pdf**.

---

## 1. Cấu Trúc Thư Mục Dự Án (Project Layout)

Cấu trúc thư mục được tổ chức theo từng phần chức năng rõ ràng, tách biệt giữa dữ liệu, tài liệu thiết kế, mã nguồn thuật toán và kiểm thử:

```text
Lab_3.2_ADDITIONAL_K-MEANS_IMAGE_SEGMENTATION/
├── data/
│   ├── raw/                                      # Dữ liệu ảnh mẫu đầu vào (image-liquid.jpg, image-sea.png, ...)
│   └── processed/                                # Dữ liệu ảnh sau khi xử lý (nếu có)
├── docs/                                         # Đề bài và tài liệu thiết kế chi tiết
│   ├── Practice Exercise 3.2.pdf                 # Đề bài gốc của môn học
│   └── Algorithms/                               # Tài liệu thiết kế thuật toán chi tiết
├── notebooks/                                    # Sổ tay Jupyter minh họa tương tác từng bước
│   ├── 01_image_loading_and_preprocessing.ipynb  # Bước 1 & 2 (Đọc & Tiền xử lý ảnh)
│   ├── 02_kmeans_scratch_and_convergence.ipynb   # Bước 3 (Thuật toán K-Means & Hội tụ)
│   ├── 03_elbow_and_sklearn_segmentation.ipynb   # Bước 4 & 5 (Tái tạo ảnh & Trình bày)
│   └── 04_advanced_comparisons_and_metrics.ipynb # So sánh OpenCV, Mini-Batch K-Means
├── results/                                      # Thư mục lưu trữ kết quả chạy
│   ├── figures/                                  # Ảnh phân đoạn và biểu đồ so sánh (.png)
│   ├── metrics/                                  # Bảng số liệu WCSS, logs (.json, .csv)
│   └── saved_models/                             # Mô hình đã lưu (.joblib)
├── src/                                          # Mã nguồn chính của bài lab
│   ├── __init__.py
│   ├── config.py                                 # Quản lý đường dẫn, tham số K, seed, ngưỡng dừng
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                             # Đọc ảnh (OpenCV / PIL), kiểm tra ảnh hợp lệ
│   │   └── preprocessing.py                      # Chuyển hệ màu, resize giữ tỷ lệ, flatten pixel
│   ├── features/
│   │   ├── __init__.py
│   │   └── color_features.py                     # Thống nhất thang đo màu, chuẩn hóa
│   ├── models/
│   │   ├── __init__.py
│   │   ├── kmeans_scratch.py                     # Thuật toán K-Means tự lập trình thuần NumPy
│   │   ├── kmeans_sklearn.py                     # Scikit-Learn KMeans & quét tìm K bằng Elbow
│   │   ├── kmeans_opencv.py                      # OpenCV cv2.kmeans kiểm tra chéo
│   │   ├── minibatch_kmeans.py                   # MiniBatchKMeans phân tích đánh đổi tốc độ
│   │   └── persistence.py                        # Lưu và nạp thông số mô hình
│   ├── segmentation/
│   │   ├── __init__.py
│   │   └── segmenter.py                          # Tái tạo ảnh phân đoạn ở kích thước làm việc và ảnh gốc
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                            # Đo quán tính WCSS, lịch sử hội tụ
│   │   └── visualization.py                      # Hiển thị Side-by-Side (ảnh gốc cạnh ảnh phân đoạn)
│   ├── deployment/
│   │   ├── __init__.py
│   │   └── predictor.py                          # Áp dụng bộ tâm màu đã học cho ảnh mới
│   └── utils/
│       ├── __init__.py
│       └── logger.py                             # Ghi nhật ký tiến trình ra màn hình
├── tests/                                        # Bộ kiểm thử tự động (Unit Tests)
│   ├── __init__.py
│   ├── test_data.py                              # Kiểm tra đọc ảnh, resize và flatten
│   ├── test_models.py                            # Kiểm tra các hàm gán cụm, cập nhật tâm
│   ├── test_segmentation.py                      # Kiểm tra tái tạo ảnh đúng kích thước
│   └── test_evaluation.py                        # Kiểm tra tính toán WCSS
├── .gitignore
├── main.py                                       # File chính điều phối 5 bước của bài lab
├── README.md
└── requirements.txt
```

---

## 2. Quy Trình Thực Hiện Theo Yêu Cầu Đề Bài (Practice Exercise 3.2)

Quy trình bài lab được thực hiện theo đúng 5 bước đề ra trong tài liệu thực hành:

### Bước 1: Load the Image

- **Đọc ảnh**: Đọc ảnh đầu vào bằng OpenCV (`cv2.imread`) hoặc Pillow (`PIL.Image`).
- **Chuyển đổi không gian màu**: Chuyển đổi từ `BGR` sang `RGB` để hiển thị đúng màu gốc; hỗ trợ chuyển đổi sang `HSV` hoặc `LAB` khi cần khảo sát thêm.

### Bước 2: Preprocess the Image

- **Resize ảnh**: Thu nhỏ ảnh theo tỷ lệ khung hình gốc (giữ nguyên aspect ratio) với cạnh dài tối đa (mặc định 256 pixel) để giảm số lượng pixel từ hàng triệu xuống kích thước làm việc $(h \times w)$, giúp thuật toán chạy nhanh trên máy cá nhân mà không làm méo hình.
- **Flatten ma trận điểm ảnh**: Chuyển đổi ma trận ảnh 3 chiều $(h, w, 3)$ thành ma trận 2 chiều $(N, 3)$ với $N = h \times w$. Mỗi hàng là một vector 3 giá trị màu $[R, G, B]$ ở định dạng số thực `float32` trong đoạn $[0, 255]$.

### Bước 3: Apply K-Means Clustering

Thực hiện phân cụm K-Means theo đúng 4 bước mô tả trong đề bài:

1. **Khởi tạo (Initialize)**: Chọn ngẫu nhiên $K$ tâm cụm màu ban đầu:
   $$
   \mu_1, \mu_2, \dots, \mu_K \in \mathbb{R}^3
   $$
2. **Gán cụm (Assign)**: Gán từng điểm ảnh vào tâm cụm gần nhất dựa trên khoảng cách bình phương Euclidean:
   $$
   c_i = \arg\min_{j \in \{0, \dots, K-1\}} \|x_i - \mu_j\|^2
   $$
3. **Cập nhật (Update)**: Cập nhật tọa độ tâm cụm bằng trung bình cộng các điểm ảnh thuộc cụm:
   $$
   \mu_j = \frac{1}{|S_j|} \sum_{x \in S_j} x
   $$
4. **Lặp lại (Repeat)**: Lặp lại quá trình gán và cập nhật cho đến khi hội tụ (độ dịch chuyển của các tâm nhỏ hơn ngưỡng sai số `tol` hoặc đạt số vòng lặp tối đa `max_iter`).

**Các phương án triển khai cụ thể của nhóm:**

- **Phương án 1 (Tự code thuần NumPy):** Lập trình chi tiết từng bước toán học để hiểu rõ bản chất hội tụ và cách xử lý cụm rỗng $\rightarrow$ Chi tiết tại [`docs/Algorithms/kmeans_from_scratch.md`](docs/Algorithms/kmeans_from_scratch.md).
- **Phương án 2 (Dùng Scikit-Learn & Dò tìm Elbow):** Dùng thư viện chuẩn `sklearn.cluster.KMeans` với thuật toán khởi tạo `k-means++`, quét khoảng $K \in [2, 12]$ để vẽ đường cong Elbow $\rightarrow$ Chi tiết tại [`docs/Algorithms/kmeans_sklearn.md`](docs/Algorithms/kmeans_sklearn.md).
- **Phương án 3 (Dùng OpenCV K-Means):** Dùng hàm `cv2.kmeans` để kiểm tra chéo về độ tương đồng nhãn và độ nén WCSS $\rightarrow$ Chi tiết tại [`docs/Algorithms/kmeans_opencv.md`](docs/Algorithms/kmeans_opencv.md).
- **Khảo sát thêm (Mini-Batch K-Means):** Thử nghiệm phân cụm theo từng lô nhỏ để đánh giá sự đánh đổi giữa thời gian thực thi và chất lượng gom màu $\rightarrow$ Chi tiết tại [`docs/Algorithms/minibatch_kmeans.md`](docs/Algorithms/minibatch_kmeans.md).

### Bước 4: Segment the Image

- **Thay thế màu sắc**: Thay mỗi điểm ảnh bằng màu sắc của tâm cụm mà nó được gán vào.
- **Tái tạo kích thước (Reshape)**: Chuyển ma trận điểm ảnh đã phân đoạn trở lại định dạng ảnh 3 chiều $(h, w, 3)$ ở kích thước làm việc, đồng thời ánh xạ pixel ảnh gốc để tạo ảnh phân đoạn ở kích thước ban đầu $(H, W, 3)$ theo đúng yêu cầu đề.

### Bước 5: Visualize the Results

- **Hiển thị song song (Side-by-Side)**: Đặt ảnh gốc và ảnh đã phân đoạn cạnh nhau, hiển thị rõ giá trị $K$ và kích thước để so sánh trực quan hiệu quả đơn giản hóa màu sắc.
- **Lưu trữ kết quả**: Tự động lưu ảnh so sánh vào thư mục `results/figures/` phục vụ báo cáo bài lab.

---

## 3. Hướng Dẫn Cài Đặt Và Chạy

### 3.1 Cài đặt môi trường

Mở terminal tại thư mục dự án và cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

### 3.2 Chạy toàn bộ pipeline bài lab

Chạy file điều phối chính để thực hiện trọn vẹn từ đọc ảnh, phân cụm đến xuất ảnh so sánh:

```bash
python main.py
```

### 3.3 Chạy bộ kiểm thử tự động

Chạy bộ unit test để kiểm tra tính đúng đắn của các hàm tiền xử lý, thuật toán và tái tạo ảnh:

```bash
python -m unittest discover tests
```

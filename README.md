# DỰ ÁN: PHÂN ĐOẠN ẢNH BẰNG K-MEANS (K-MEANS IMAGE SEGMENTATION)

Dự án Machine Learning thị giác máy tính thực hiện phân đoạn ảnh màu thành $K$ phân vùng cụm màu sắc bằng thuật toán **K-Means Clustering** theo đúng yêu cầu của **Practice Exercise 3 (Additional) / Practice Exercise 3.2.pdf**.

---

## 1. Cấu Trúc Thư Mục Dự Án (Project Layout)

Cấu trúc thư mục được thiết kế chuẩn mực theo mô hình Module hóa (Modular Codebase) tương tự như chuẩn của Lab 3.1:

```text
Lab_3.2_ADDITIONAL_K-MEANS_IMAGE_SEGMENTATION/
├── data/
│   ├── raw/                      # Dữ liệu ảnh thô (image-liquid.jpg, image-sea.png, ...)
│   └── processed/                # Dữ liệu ảnh đã resize và bộ nhớ đệm
├── docs/                         # Đề bài và tài liệu tham khảo
│   ├── LAB3_ADDITIONAL_K-MEANS_IMAGE_SEGMENTATION/ # Codebase mẫu tham khảo
│   ├── Practice Exercise 3.docx
│   └── Practice Exercise 3.2.pdf
├── notebooks/                    # Sổ tay Jupyter thử nghiệm tương tác theo từng bước
│   ├── 01_image_loading_and_preprocessing.ipynb    # Task 1 & Task 2
│   ├── 02_kmeans_scratch_and_convergence.ipynb     # Task 3 (Scratch K-Means)
│   ├── 03_elbow_and_sklearn_segmentation.ipynb     # Task 4 & Task 5 (Elbow & Side-by-Side)
│   └── 04_advanced_comparisons_and_metrics.ipynb   # OpenCV, Mini-Batch, Silhouette
├── results/                      # Lưu trữ kết quả đầu ra
│   ├── figures/                  # Biểu đồ trực quan và ảnh phân đoạn (.png)
│   ├── metrics/                  # Bảng số liệu, json, logs (.json, .csv)
│   └── saved_models/             # Mô hình K-Means đã đóng gói (.joblib)
├── src/                          # Mã nguồn module hóa hoàn chỉnh
│   ├── __init__.py
│   ├── config.py                 # Quản lý đường dẫn, siêu tham số, bảng màu
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py             # Nạp ảnh (OpenCV / PIL), kiểm tra schema
│   │   └── preprocessing.py      # Chuyển đổi không gian màu (RGB/HSV/LAB), resize, flatten
│   ├── features/
│   │   ├── __init__.py
│   │   └── color_features.py     # Chuẩn hóa pixel, định danh tên màu ngữ nghĩa
│   ├── models/
│   │   ├── __init__.py
│   │   ├── kmeans_scratch.py     # K-Means thuần từ đầu (Task 3) kèm theo dõi hội tụ
│   │   ├── kmeans_sklearn.py     # Scikit-Learn KMeans, Elbow Method (Kneedle), chọn K tối ưu
│   │   ├── kmeans_opencv.py      # OpenCV cv2.kmeans kiểm định chéo đa thư viện
│   │   ├── minibatch_kmeans.py   # MiniBatchKMeans phân tích đánh đổi tốc độ/chất lượng
│   │   └── persistence.py        # Lưu và nạp mô hình bằng joblib
│   ├── segmentation/
│   │   ├── __init__.py
│   │   └── segmenter.py          # Tái tạo ảnh phân đoạn (Task 4), phục hồi kích thước gốc, masks
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # WCSS (Inertia), Silhouette Score, ARI, NMI
│   │   └── visualization.py      # Hiển thị Side-by-Side (Task 5), Elbow curve, Convergence plot
│   ├── deployment/
│   │   ├── __init__.py
│   │   └── predictor.py          # Pipeline suy luận trọn gói cho ảnh mới
│   └── utils/
│       ├── __init__.py
│       └── logger.py             # Tiện ích ghi log ra Console và File
├── tests/                        # Bộ kiểm thử tự động (Unit Tests)
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_segmentation.py
│   └── test_evaluation.py
├── .gitignore
├── main.py                       # File điều phối chạy toàn bộ pipeline từ đầu đến cuối
├── README.md
└── requirements.txt
```

---

## 2. Quy Trình Thực Hiện Theo Yêu Cầu Đề Bài (Practice Exercise 3.2)

### Task 1: Load the Image
- **Đọc ảnh**: Đọc ảnh đầu vào bằng OpenCV (`cv2.imread`) hoặc Pillow (`PIL.Image`).
- **Chuyển đổi không gian màu**: Chuyển đổi từ `BGR` sang `RGB` để hiển thị đúng màu; hỗ trợ chuyển đổi sang `HSV` (tách riêng Sắc thái, Độ bão hòa, Độ sáng) và `LAB` (không gian màu cảm nhận đồng đều).

### Task 2: Preprocess the Image
- **Resize ảnh**: Thu nhỏ ảnh về kích thước $256 	imes 256$ (giữ nguyên tỷ lệ khung hình) để giảm số lượng pixel từ hàng triệu xuống khoảng $65,536$ điểm ảnh, giúp tăng tốc độ phân cụm đáng kể.
- **Flatten ma trận điểm ảnh**: Chuyển đổi ma trận ảnh 3 chiều $(H, W, 3)$ thành ma trận 2 chiều $(N_{	ext{pixels}}, 3)$, trong đó mỗi hàng là một vector thuộc tính màu sắc $[R, G, B]$ ở định dạng `float32`.

### Task 3: Apply K-Means Clustering
Triển khai thuật toán theo 4 bước nguyên bản:
1. **Khởi tạo (Initialize)**: Chọn $K$ tâm cụm ngẫu nhiên đại diện cho màu trung bình của từng phân khúc.
2. **Gán cụm (Assign)**: Gán từng pixel vào tâm cụm gần nhất dựa trên khoảng cách Euclidean:
   $$c(i) = rg\min_{j} \|x_i - \mu_j\|^2$$
3. **Cập nhật (Update)**: Cập nhật tọa độ tâm cụm bằng trung bình cộng các pixel thuộc cụm:
   $$\mu_j = rac{1}{|S_j|} \sum_{x_i \in S_j} x_i$$
4. **Lặp lại (Repeat)**: Lặp lại quá trình gán và cập nhật cho đến khi hội tụ (độ dịch chuyển tâm cụm nhỏ hơn ngưỡng tolerance hoặc đạt số vòng lặp tối đa).
- **Theo dõi hội tụ (Convergence Tracking)**: Ghi nhận WCSS, độ dịch chuyển tâm cụm và tỷ lệ thay đổi nhãn qua từng vòng lặp.
- **Dò tìm số cụm $K$ tối ưu (Elbow Method)**: Quét $K \in [2, 12]$ và dùng thuật toán **Kneedle** xác định điểm gập khuỷu tay ($K=5$).

### Task 4: Segment the Image
- **Thay thế màu sắc**: Thay thế vector màu của từng pixel bằng tọa độ màu RGB của tâm cụm tương ứng.
- **Tái tạo kích thước (Reshape)**: Chuyển đổi ma trận điểm ảnh đã phân đoạn trở lại kích thước 2D $(H, W, 3)$ ở cả kích thước thu nhỏ và độ phân giải gốc của bức ảnh.

### Task 5: Visualize the Results
- **Hiển thị song song (Side-by-Side)**: Đặt ảnh gốc và ảnh đã phân đoạn cạnh nhau để so sánh trực quan hiệu quả phân cụm.
- **Biểu đồ phân bổ màu sắc**: Biểu đồ cột ngang thể hiện tỷ lệ % diện tích của từng phân vùng màu.

---

## 3. Hướng Dẫn Cài Đặt Và Chạy

### 3.1 Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### 3.2 Chạy toàn bộ pipeline tự động
```bash
python main.py
```

### 3.3 Chạy bộ kiểm thử tự động
```bash
python -m unittest discover tests
```

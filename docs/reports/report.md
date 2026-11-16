# BÁO CÁO THỰC HÀNH: PHÂN ĐOẠN ẢNH BẰNG K-MEANS
## Practice Exercise 3 (Additional) — K-Means Image Segmentation

- **Môn học:** Machine Learning / Học máy
- **Chủ đề:** Phân đoạn màu sắc hình ảnh (Color-based Image Segmentation) bằng thuật toán phân cụm không giám sát K-Means.
- **Tập dữ liệu thực nghiệm:** `data/raw/image-liquid.jpg` (Độ phân giải gốc: $3264 \times 3264$ pixels, 3 kênh màu RGB).
- **Mã nguồn thực thi:** `main.py` và 4 Jupyter Notebooks tại `notebooks/`.

---

## 1. Dữ liệu và Khám phá Không gian Màu (Task 1: Load Image & Color Space)

### 1.1 Thông tin tệp ảnh gốc
Ảnh đầu vào được lựa chọn là ảnh chất lượng cao chụp hiệu ứng màu chất lỏng pha trộn:
- **Tên tệp:** `image-liquid.jpg`
- **Kích thước ban đầu:** $3264 \times 3264 \times 3$ ($10,653,696$ điểm ảnh).
- **Định dạng dữ liệu:** Mảng số nguyên không dấu `uint8` nằm trong thang giá trị $[0, 255]$.
- **Dung lượng file:** $\approx 2.05 \text{ MB}$.

### 1.2 Khảo sát các không gian màu
Đề bài yêu cầu khảo sát việc đưa ảnh về không gian màu phù hợp. Nhóm đã thực hiện chuyển đổi và đối chiếu 3 không gian màu chính:
1. **RGB (Red, Green, Blue):** Không gian màu gốc cộng tính, biểu thị trực tiếp cường độ 3 màu cơ bản. Đây là không gian màu trực quan và được chọn làm baseline cho toàn bộ quá trình phân cụm.
2. **HSV (Hue, Saturation, Value):** Tách biệt thông tin màu sắc (Hue) khỏi cường độ sáng (Value) và độ bão hòa (Saturation), hữu ích khi ảnh có độ tương phản ánh sáng phức tạp.
3. **CIE-LAB:** Không gian màu cảm nhận đồng đều (Perceptually Uniform), kênh $L$ biểu diễn độ chói, kênh $A$ và $B$ biểu diễn trục màu lục-đỏ và lam-vàng.

![So sánh các không gian màu](../../results/figures/color_spaces_comparison.png)
*Hình 1: Trực quan hóa đối chiếu 3 không gian màu RGB, HSV và CIE-LAB.*

---

## 2. Tiền xử lý dữ liệu ảnh (Task 2: Preprocessing & Reshape)

### 2.1 Thu nhỏ kích thước ảnh (Downsampling)
Ảnh gốc chứa hơn 10.6 triệu điểm ảnh. Nếu chạy thuật toán phân cụm trực tiếp trên toàn bộ ma trận này, chi phí tính khoảng cách Euclidean sẽ là $\mathcal{O}(N \cdot K \cdot d \cdot \text{iter})$, gây quá tải bộ nhớ và tốn nhiều thời gian trên máy tính cá nhân.
- **Giải pháp:** Áp dụng hàm `resize_image` với `target_size = 256` và `keep_aspect_ratio = True`.
- **Kích thước sau khi thu nhỏ:** $(256, 256, 3)$ gồm $65,536$ điểm ảnh.
- **Tỷ lệ giảm tải tính toán:** Giảm $\approx 162.5$ lần số lượng điểm ảnh nhưng vẫn bảo toàn đầy đủ các mảng màu chính và đường bao chi tiết của bố cục ảnh.

### 2.2 Làm phẳng ma trận điểm ảnh (Flatten)
Chuyển ma trận ảnh 3 chiều $(256, 256, 3)$ thành ma trận 2 chiều $(65536, 3)$ kiểu `float32` trong khoảng $[0.0, 255.0]$. Mỗi hàng biểu diễn một vector đặc trưng 3 chiều $[R, G, B]$.

---

## 3. Thuật toán K-Means Tự Lập Trình (Task 3: K-Means from Scratch)

### 3.1 Cài đặt 4 bước toán học thuần NumPy
Theo đúng mô tả trong tài liệu thực hành, thuật toán K-Means được cài đặt độc lập trong lớp `KMeansFromScratch` tại `src/models/kmeans_scratch.py`:
1. **Khởi tạo ($K=4$):** Chọn ngẫu nhiên không hoàn lại 4 điểm ảnh từ ma trận dữ liệu làm 4 tâm màu ban đầu.
2. **Gán cụm:** Tính khoảng cách bình phương Euclidean bằng ma trận hóa:
   $$\text{dist}^2(x_i, \mu_j) = \|x_i - \mu_j\|^2 = \|x_i\|^2 + \|\mu_j\|^2 - 2 x_i \mu_j^T$$
   Gán $c_i = \arg\min_j \text{dist}^2(x_i, \mu_j)$.
3. **Cập nhật tâm & Xử lý cụm rỗng:** Lấy trung bình cộng các điểm ảnh trong từng cụm. Nếu xuất hiện cụm rỗng, tái khởi tạo tâm bằng điểm ảnh đang có khoảng cách xa nhất so với các tâm hiện hữu.
4. **Kiểm tra hội tụ:** Lặp lại cho đến khi độ dịch chuyển tâm $\max_j \|\mu_j^{(t)} - \mu_j^{(t-1)}\| \le 10^{-4}$ hoặc hết 300 vòng lặp.

### 3.2 Kết quả hội tụ thực nghiệm
- **Số vòng lặp hoàn thành:** Thuật toán đạt trạng thái hội tụ hoàn toàn chỉ sau **23 vòng lặp**.
- **Quán tính cuối cùng (WCSS):** $158,324,198.55$.
- **Đặc tính suy giảm:** Quán tính WCSS giảm liên tục và đơn điệu qua từng vòng lặp (vòng 1: $4.67 \times 10^8 \rightarrow$ vòng 23: $1.58 \times 10^8$), chứng minh giải thuật cài đặt đúng theo nguyên lý toán học.

![Tiến trình hội tụ](../../results/figures/kmeans_convergence.png)
*Hình 2: Biểu đồ suy giảm quán tính WCSS và độ dịch chuyển tâm cụm (Centroid Shift) tiệm cận về 0.*

---

## 4. Quét Elbow Method và Tối ưu hóa số cụm K

### 4.1 Khảo sát dải giá trị $K \in [2, 12]$
Sử dụng `sklearn.cluster.KMeans` với phương thức khởi tạo thông minh `k-means++` (chạy 10 lần khởi tạo mỗi $K$) để quét quán tính WCSS:

| Số cụm $K$ | Quán tính WCSS (Inertia) | Tốc độ giảm $\Delta \text{WCSS}$ |
|:---:|:---:|:---:|
| 2 | 337,885,248.0 | — |
| 3 | 235,340,368.0 | -102,544,880.0 |
| 4 | 158,325,536.0 | -77,014,832.0 |
| **5 (Optimal)** | **122,704,000.0** | **-35,621,536.0** |
| 6 | 106,371,408.0 | -16,332,592.0 |
| 7 | 92,153,736.0 | -14,217,672.0 |
| 8 | 80,006,464.0 | -12,147,272.0 |
| 9 | 71,714,416.0 | -8,292,048.0 |
| 10 | 65,363,120.0 | -6,351,296.0 |
| 11 | 59,752,904.0 | -5,610,216.0 |
| 12 | 54,990,224.0 | -4,762,680.0 |

![Đường cong Elbow](../../results/figures/elbow_method.png)
*Hình 3: Đường cong Elbow Method và điểm gập khuỷu tay được xác định tại $K=5$.*

### 4.2 Lựa chọn số cụm $K=5$
Áp dụng thuật toán hình học Kneedle (tìm khoảng cách trực giao cực đại từ đường cong tới cát tuyến nối điểm đầu - cuối), điểm khuỷu tay xuất hiện rõ rệt tại **$K=5$**. Tại mốc này, tốc độ suy giảm WCSS bắt đầu chậm lại rõ rệt; việc tăng thêm $K > 5$ làm tăng độ phức tạp mô hình mà không cải thiện đáng kể độ nén màu sắc.

---

## 5. Tái tạo Ảnh Phân đoạn và Đánh giá Side-by-Side (Task 4 & Task 5)

### 5.1 Tái tạo ảnh phân đoạn và phục hồi kích thước gốc
- Thay thế mỗi điểm ảnh bằng tọa độ tâm màu RGB tương ứng của cụm được gán.
- **Kích thước làm việc:** Reshape về $(256, 256, 3)$ dạng `uint8`.
- **Phục hồi kích thước gốc:** Sử dụng bản đồ nhãn phóng đại bằng nội suy Nearest-Neighbor (`reconstruct_original_size`) để tái tạo ảnh phân đoạn ở kích thước ban đầu $3264 \times 3264 \times 3$ sắc nét, không bị nhòe màu biên giới giữa các phân khúc.

### 5.2 Hiển thị so sánh song song (Side-by-Side)

![So sánh ảnh gốc và ảnh phân đoạn](../../results/figures/final_segmentation_side_by_side.png)
*Hình 4: Hiển thị Side-by-Side ảnh gốc và ảnh phân đoạn tại $K=5$.*

### 5.3 Phân bổ tỷ lệ diện tích các mảng màu
Thống kê từ $65,536$ điểm ảnh cho thấy 5 cụm màu sắc đại diện phân bổ như sau:

| Mã Cụm | Tên màu nhận diện | Vector tâm RGB | Số lượng pixel | Tỷ lệ diện tích (%) |
|:---:|:---:|:---:|:---:|:---:|
| Cụm 0 | Cam (Orange) | `(209, 112, 66)` | 19,713 | 30.08% |
| Cụm 2 | Vàng (Yellow) | `(237, 190, 93)` | 17,839 | 27.22% |
| Cụm 1 | Đỏ (Red) | `(137, 57, 90)` | 11,556 | 17.63% |
| Cụm 4 | Đen (Black) | `(40, 25, 62)` | 8,613 | 13.14% |
| Cụm 3 | Tím (Violet) | `(172, 162, 190)` | 7,815 | 11.92% |

![Phân bổ màu sắc](../../results/figures/cluster_distribution.png)
*Hình 5: Biểu đồ thanh ngang thể hiện tỷ lệ % diện tích từng cụm màu.*

---

## 6. Khảo sát Chuyên sâu và Đánh giá Nâng cao

### 6.1 Chỉ số chất lượng phân cụm Silhouette Score
- Tính toán trên tập mẫu đại diện phân tầng $5,000$ điểm ảnh: **$\text{Silhouette Score} = 0.4245$** (nằm trong khoảng $[-1, 1]$).
- Điểm số dương và trên mức 0.4 cho thấy các cụm màu sắc có sự gắn kết nội cụm tốt và tách biệt rõ rệt giữa các gam màu nóng (Cam, Vàng, Đỏ) và gam màu tối/lạnh (Đen, Tím).

### 6.2 Phân tích đánh đổi với Mini-Batch K-Means
Thử nghiệm phân cụm theo lô nhỏ với $K=5$ đối chiếu với Full K-Means:

| Batch Size | Thời gian (ms) | Tăng tốc (Speedup) | Quán tính WCSS | WCSS tăng thêm (%) | ARI (Đồng thuận nhãn) | NMI (Thông tin tương hỗ) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 256 | 21.00 ms | **12.07x** | 123,366,336.0 | +0.54% | **0.9352** | **0.9170** |
| 512 | 139.22 ms | 1.82x | 140,000,816.0 | +14.10% | 0.5666 | 0.6503 |
| 1024 | 56.80 ms | 4.46x | 122,926,256.0 | +0.18% | 0.9287 | 0.9193 |
| 2048 | 99.40 ms | 2.55x | 125,770,080.0 | +2.50% | 0.8601 | 0.8470 |
| 4096 | 37.25 ms | 6.80x | 123,064,240.0 | +0.29% | 0.9251 | 0.9175 |

**Nhận xét:** Khi chọn `batch_size = 256` hoặc `1024`, Mini-Batch K-Means đạt tốc độ vượt trội (nhanh hơn đến 12 lần) mà mức suy giảm chất lượng chỉ dưới 0.5% ($\text{ARI} > 0.92$), là giải pháp tối ưu cho việc xử lý ảnh trực tuyến thời gian thực.

---

## 7. Kết luận và Hướng phát triển

### 7.1 Kết luận
1. Hoàn thành 100% mục tiêu của đề bài thực hành 3.2 theo quy trình 5 bước chuẩn mực.
2. Thuật toán K-Means tự lập trình thuần NumPy đạt độ chính xác tương đương với thư viện Scikit-Learn chuẩn mực, hội tụ ổn định sau 23 vòng lặp.
3. Cơ chế phục hồi kích thước gốc bằng phóng đại bản đồ nhãn giải quyết triệt để bài toán dung lượng và thời gian thực thi trên máy cá nhân mà vẫn giữ nguyên chất lượng ảnh hiển thị.

### 7.2 Hướng phát triển
- Thử nghiệm mở rộng không gian đặc trưng từ 3 chiều $[R, G, B]$ thành 5 chiều $[R, G, B, x, y]$ có chuẩn hóa tọa độ không gian để các phân vùng cụm có tính liên thông địa lý chặt chẽ hơn thay vì chỉ phân đoạn thuần túy theo màu sắc.


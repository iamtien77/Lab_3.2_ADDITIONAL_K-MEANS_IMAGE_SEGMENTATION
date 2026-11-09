# Thiết kế bài lab K-Means Image Segmentation

## 1. What — Bài lab làm gì?

Thiết kế quy trình phân đoạn ảnh theo màu bằng K-Means cho một bài thực hành nhỏ. Mỗi pixel là một điểm dữ liệu; các pixel có màu gần nhau được gom vào cùng cụm và biểu diễn bằng màu tâm cụm. Cụm màu không mặc nhiên tương ứng với một vật thể hoặc vùng liên thông.

Nguồn yêu cầu: [Practice Exercise 3.2.pdf](<../Practice%20Exercise%203.2.pdf>), trang 1. Phạm vi bắt buộc gồm năm bước:

| Bước            | Yêu cầu                                                                                                                      |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 1. Đọc ảnh     | Chọn ảnh bất kỳ, đọc bằng OpenCV hoặc PIL, đưa về color space phù hợp.                                            |
| 2. Tiền xử lý  | Resize nhỏ hơn để giảm chi phí tính toán; flatten thành ma trận pixel hai chiều.                                    |
| 3. Phân cụm     | Khởi tạo K tâm ngẫu nhiên, gán pixel theo Euclidean distance, cập nhật tâm bằng trung bình và lặp đến hội tụ. |
| 4. Tái tạo ảnh | Thay màu pixel bằng màu tâm cụm tương ứng và đưa về kích thước ảnh.                                            |
| 5. Trình bày    | Hiển thị ảnh ban đầu cạnh ảnh phân đoạn để quan sát.                                                              |

Các lựa chọn RGB, seed, cách resize, xử lý cụm rỗng và tiêu chí kiểm tra dưới đây là quyết định thiết kế của nhóm. Đề không ấn định K, kích thước resize hoặc ngưỡng chất lượng.

### Tài liệu thiết kế chi tiết từng thuật toán:
- **Phương án 1 (Tự lập trình 4 bước theo đề bằng NumPy):** [kmeans_from_scratch.md](kmeans_from_scratch.md)
- **Phương án 2 (Dùng Scikit-Learn & Dò tìm Elbow):** [kmeans_sklearn.md](kmeans_sklearn.md)
- **Phương án 3 (Dùng OpenCV K-Means):** [kmeans_opencv.md](kmeans_opencv.md)
- **Khảo sát thêm (Mini-Batch K-Means):** [minibatch_kmeans.md](minibatch_kmeans.md)

## 2. Why — Vì sao thực hiện?

- Hiểu cách chuyển một ảnh thành tập vector đặc trưng màu.
- Giải thích được ba thao tác cốt lõi của K-Means: khởi tạo, gán cụm và cập nhật tâm.
- Hiểu ảnh hưởng của K, khởi tạo và resize đối với mức độ đơn giản hóa màu.
- Thực hành chia việc, thống nhất dữ liệu bàn giao và kiểm tra một quy trình hoàn chỉnh.

Ưu tiên thuật toán dễ giải thích và phạm vi có thể hoàn thành trên máy cá nhân. Không đặt mục tiêu nhận diện đối tượng hoặc xây dựng dịch vụ vận hành.

## 3. Who — Ai thực hiện và sử dụng?

Nhóm sinh viên thực hành là người thực hiện; các thành viên và giảng viên sử dụng tài liệu để thống nhất cách làm và kiểm tra yêu cầu. Chia trách nhiệm thành ba phần; số người mỗi phần tùy quy mô nhóm.

| Vai trò                   | Trách nhiệm                                                          | Nội dung bàn giao                                           |
| -------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------- |
| Dữ liệu                  | Chọn ảnh, kiểm tra màu, resize và flatten                         | Ảnh RGB, kích thước trước/sau resize và ma trận pixel |
| Thuật toán               | Thiết kế K-Means, điều kiện dừng và xử lý trường hợp biên | Nhãn, tâm cụm và thông tin hội tụ                      |
| Tích hợp và đánh giá | Tái tạo ảnh, trình bày song song, đối chiếu tiêu chí         | Phần minh họa và checklist kiểm tra                       |

Cả nhóm thống nhất quy ước dữ liệu trước khi chia việc; cùng giải thích được toàn bộ năm bước khi trình bày.

## 4. Where — Thực hiện ở đâu?

Thực hành cục bộ trên máy cá nhân với môi trường Python của bài lab. Dữ liệu ảnh đặt tại `data/raw/`; tài liệu thiết kế đặt tại `docs/Algorithms/`; tài liệu báo cáo thuộc `docs/reports/`. Không cần máy chủ hoặc hạ tầng cloud.

### 4.1. Target structure

Cấu trúc được chia theo chức năng và nhiệm vụ trong quy trình thực hành: chuẩn bị dữ liệu, biểu diễn đặc trưng, phân cụm, tái tạo ảnh và đánh giá. Mỗi phần có trách nhiệm riêng để nhóm dễ phân công, kiểm tra, bảo trì và mở rộng. Ký hiệu `[tùy chọn]` dành cho phần ngoài baseline của bài lab.

```text
Lab_3.2_ADDITIONAL_K-MEANS_IMAGE_SEGMENTATION/
├── .agents/                                  # Quy tắc và workflow làm việc
├── .gitignore                                # Quy định tài nguyên loại khỏi Git
├── AGENTS.md                                 # Hướng dẫn làm việc trong workspace
├── README.md                                 # Tổng quan và hướng dẫn thực hành
├── requirements.txt                          # Danh sách thư viện của môi trường
├── data/                                     # Dữ liệu phục vụ bài lab
│   └── raw/                                  # Các ảnh có thể chọn làm đầu vào
├── docs/                                     # Đề bài, thiết kế và báo cáo thực hành
│   ├── Practice Exercise 3.2.pdf              # Đề bài
│   ├── Algorithms/                           # Tài liệu thiết kế thuật toán
│   └── reports/                              # Tài liệu báo cáo thực hành
├── notebooks/                                # Các bước minh họa tương tác
│   ├── 01_image_loading_and_preprocessing.ipynb
│   │                                         # Đọc ảnh và chuẩn bị ma trận pixel
│   ├── 02_kmeans_scratch_and_convergence.ipynb
│   │                                         # Các bước K-Means và điều kiện dừng
│   ├── 03_elbow_and_sklearn_segmentation.ipynb
│   │                                         # [tùy chọn] Khảo sát K và đối chiếu thư viện
│   └── 04_advanced_comparisons_and_metrics.ipynb
│                                             # [tùy chọn] Các phép so sánh bổ sung
├── src/                                      # Phân chia trách nhiệm xử lý
│   ├── __init__.py                           # Đánh dấu package
│   ├── config.py                             # Tham số ảnh, K, seed và điều kiện dừng
│   ├── data/                                 # Đọc và chuẩn bị dữ liệu
│   │   ├── __init__.py                        # Đánh dấu package
│   │   ├── loader.py                          # Đọc ảnh, kiểm tra đầu vào
│   │   └── preprocessing.py                   # Chuyển màu, resize và flatten
│   ├── features/                             # Quy ước đặc trưng màu
│   │   ├── __init__.py                        # Đánh dấu package
│   │   └── color_features.py                  # Biểu diễn và thống nhất thang đo màu
│   ├── models/                               # Thuật toán phân cụm
│   │   ├── __init__.py                        # Đánh dấu package
│   │   ├── kmeans_scratch.py                  # Baseline: khởi tạo, gán, cập nhật, dừng
│   │   ├── kmeans_sklearn.py                  # [tùy chọn] Đối chiếu với scikit-learn
│   │   ├── kmeans_opencv.py                   # [tùy chọn] Đối chiếu với OpenCV
│   │   ├── minibatch_kmeans.py                # [tùy chọn] Khảo sát MiniBatchKMeans
│   │   └── persistence.py                     # [tùy chọn] Lưu và nạp tâm cùng cấu hình
│   ├── segmentation/                         # Tạo ảnh từ nhãn và tâm cụm
│   │   ├── __init__.py                        # Đánh dấu package
│   │   └── segmenter.py                       # Ánh xạ màu tâm và phục hồi shape ảnh
│   ├── evaluation/                           # Kiểm tra và trình bày
│   │   ├── __init__.py                        # Đánh dấu package
│   │   ├── metrics.py                         # WCSS và thông tin phục vụ đánh giá
│   │   └── visualization.py                   # Hiển thị ảnh cạnh nhau và đồ thị
│   ├── deployment/                           # [tùy chọn] Áp dụng cho ảnh khác
│   │   ├── __init__.py                        # Đánh dấu package
│   │   └── predictor.py                       # Gán pixel ảnh khác vào bộ tâm đã chọn
│   └── utils/                                # Tiện ích dùng chung
│       ├── __init__.py                        # Đánh dấu package
│       └── logger.py                          # Thông báo tiến trình và lỗi
├── tests/                                    # Kiểm tra theo trách nhiệm
│   ├── __init__.py                            # Đánh dấu package
│   ├── test_data.py                           # Kiểm tra ảnh, màu, resize và flatten
│   ├── test_models.py                         # Kiểm tra phép gán, cập nhật và dừng
│   ├── test_segmentation.py                   # Kiểm tra tái tạo ảnh và kích thước
│   └── test_evaluation.py                     # Kiểm tra phép đo và dữ liệu hiển thị
└── main.py                                   # Điều phối năm bước của bài lab
```

### 4.2. Tiêu chí phân chia chức năng và nhiệm vụ

| Tiêu chí                 | Cách phân chia                                                                                                                                                 | Mục đích                                                                                                      |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Theo chức năng           | Tách chuẩn bị dữ liệu, đặc trưng màu, phân cụm, tái tạo ảnh và đánh giá thành các phần riêng.                                              | Dễ tìm đúng phần cần làm và theo dõi luồng xử lý.                                                    |
| Theo trách nhiệm         | Mỗi phần đảm nhận một nhiệm vụ chính; chẳng hạn phần phân cụm phụ trách nhãn và tâm, phần tái tạo phụ trách ánh xạ chúng thành ảnh. | Hạn chế chồng chéo khi phân công và bảo trì.                                                            |
| Theo dữ liệu bàn giao   | Thống nhất ảnh RGB, ma trận pixel, nhãn, tâm và kích thước ảnh giữa các bước.                                                                     | Các thành viên có thể thực hiện từng phần rồi ghép lại theo cùng quy ước.                         |
| Theo phạm vi thay đổi   | Tập trung tham số thực hành tại phần cấu hình; đặt phép đo và cách hiển thị tại phần đánh giá.                                              | Dễ điều chỉnh K, cách đánh giá hoặc cách trình bày mà hạn chế ảnh hưởng tới nhiệm vụ khác. |
| Theo khả năng mở rộng  | Tách baseline với các lựa chọn khảo sát thêm; dùng chung quy ước dữ liệu khi bổ sung phương pháp phân cụm hoặc đặc trưng màu.            | Thuận tiện nâng cấp sau khi hoàn thành yêu cầu cơ bản.                                                 |
| Theo phạm vi kiểm tra    | Chia kiểm tra tương ứng với dữ liệu, thuật toán, tái tạo và đánh giá; bổ sung kiểm tra toàn bộ quy trình khi tích hợp.                     | Dễ khoanh vùng sai sót và kiểm tra lại phần chịu ảnh hưởng khi thay đổi.                            |
| Theo mục đích sử dụng | Tách ảnh đầu vào, tài liệu, phần minh họa thực hành và các nhiệm vụ xử lý.                                                                      | Dễ tra cứu, bàn giao và sử dụng lại tài liệu học tập.                                                 |

Với quy mô bài lab nhỏ, nhóm phân công theo ba nhiệm vụ chính ở mục Who. Các phần chức năng giúp xác định trách nhiệm, không yêu cầu mỗi phần phải có một người phụ trách riêng.

## 5. When — Thực hiện theo trình tự nào?

Không ấn định ngày hoặc số buổi; chuyển mốc khi đáp ứng điều kiện bàn giao.

| Mốc                    | Công việc                                                      | Điều kiện chuyển mốc                       |
| ----------------------- | ---------------------------------------------------------------- | ----------------------------------------------- |
| 1. Thống nhất         | Chọn ảnh, K, quy ước RGB, resize và tiêu chí đánh giá  | Cả nhóm hiểu cùng đầu vào và phạm vi   |
| 2. Chuẩn bị dữ liệu | Đọc ảnh, kiểm tra, resize, flatten                           | Ma trận pixel đúng shape và màu            |
| 3. Baseline             | Thực hiện vòng lặp K-Means và kiểm tra trường hợp biên | Nhãn/tâm hợp lệ, có lý do dừng           |
| 4. Tích hợp           | Tái tạo ảnh và bố trí hiển thị song song                 | Đủ năm bước, đúng kích thước và màu |
| 5. Rà soát            | Đối chiếu checklist, chuẩn bị phần giải thích            | Mỗi tiêu chí có cách kiểm tra cụ thể    |

Chỉ khảo sát thêm K hoặc color space sau khi baseline đáp ứng tiêu chí cốt lõi.

## 6. How — Thiết kế giải pháp như thế nào?

### 6.1. Baseline và dữ liệu

Loại bài toán là unsupervised clustering theo màu. Baseline sử dụng K-Means trên RGB với khoảng cách Euclidean vì biểu diễn đơn giản và bám trực tiếp các bước của đề.

Chọn ít nhất một ảnh màu đọc được, có ba kênh. Khi đọc bằng OpenCV, đổi BGR sang RGB trước khi xử lý và hiển thị. Với ảnh grayscale hoặc có alpha, baseline thông báo yêu cầu ảnh RGB; việc hỗ trợ chuyển đổi được thống nhất riêng.

Resize giữ tỷ lệ khung hình. Đề xuất giới hạn cạnh dài ở 256 pixel và không phóng to ảnh nhỏ; đây là cấu hình khởi đầu, có thể điều chỉnh theo ảnh. Lưu kích thước thực tế sau resize, không giả định mọi ảnh đều vuông.

| Dữ liệu       | Quy ước                                                                                  |
| --------------- | ------------------------------------------------------------------------------------------ |
| Ảnh đầu vào | RGB, kích thước `(H, W, 3)`, giá trị nguyên trong `[0,255]`                      |
| Ảnh làm việc | RGB sau resize, kích thước `(h, w, 3)`                                                |
| Ma trận pixel  | `(N,3)`, với `N=h*w`; số thực hữu hạn trong `[0,255]`, giữ thứ tự theo hàng |
| Nhãn cụm      | N số nguyên thuộc `[0,K-1]`, cùng thứ tự với ma trận pixel                       |
| Tâm cụm       | `(K,3)`, số thực hữu hạn, cùng color space và thang đo với dữ liệu             |

K là tham số được nhóm chọn trước mỗi lần thực hành; có thể bắt đầu bằng K=4 nếu ảnh có đủ màu phân biệt. Không coi giá trị khởi đầu là tối ưu. Chọn seed cố định, ví dụ 42, để thuận tiện kiểm tra lặp lại trong cùng môi trường.

### 6.2. Đánh giá được thống nhất trước khi thực hành

| Nội dung            | Cách kiểm tra                                                                                  | Giới hạn diễn giải                                                                                        |
| -------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| Đủ yêu cầu       | Đối chiếu năm bước trong mục 1                                                            | Là tiêu chí nghiệm thu chính                                                                             |
| WCSS / Inertia       | Tổng bình phương khoảng cách từ pixel tới tâm được gán; phải hữu hạn, không âm | Chỉ so sánh khi cùng dữ liệu và thang đo; WCSS nhỏ hơn không tự chứng minh phân đoạn tốt hơn |
| Hội tụ             | Theo dõi độ dịch chuyển tâm, số vòng lặp và lý do dừng                               | Đạt giới hạn vòng lặp không đồng nghĩa hội tụ                                                     |
| Quan sát trực quan | Đặt hai ảnh cạnh nhau, kiểm tra màu, vùng chi tiết và mức đơn giản hóa             | Không suy ra độ chính xác nhận diện vật thể                                                          |
| Tính tái lập      | Ghi ảnh, kích thước, K, seed, color space, ngưỡng dừng và giới hạn vòng lặp          | Chỉ kỳ vọng nhất quán trong môi trường được kiểm soát                                            |

Không sử dụng Accuracy hoặc IoU khi chưa có ground truth. Không đặt trước điểm số hoặc một K tối ưu cần đạt.

### 6.3. Logic K-Means

Với mỗi điểm ảnh $x_i$, tâm cụm $\mu_j$ và nhãn cụm được gán $c_i$, mục tiêu của thuật toán là tối thiểu hóa tổng bình phương khoảng cách nội cụm (WCSS):

$$
\text{WCSS} = \sum_{i=1}^{N} \|x_i - \mu_{c_i}\|^2
$$

1. **Kiểm tra đầu vào:** ảnh và ma trận không rỗng; K là số nguyên dương, không vượt số màu phân biệt của dữ liệu làm việc; các giá trị đều hữu hạn.
2. **Khởi tạo:** lấy ngẫu nhiên K màu phân biệt từ tập pixel bằng seed đã chọn.
3. **Gán cụm:** mỗi pixel nhận nhãn của tâm gần nhất theo Euclidean distance; khi hòa, dùng cùng một quy tắc chọn chỉ số nhỏ nhất.
4. **Cập nhật:** lấy trung bình các pixel của từng cụm. Nếu một cụm rỗng, giữ tâm của cụm đó ở vòng trước để tránh phép lấy trung bình trên tập rỗng; ghi nhận sự kiện để nhóm giải thích.
5. **Kiểm tra dừng:** dùng độ dịch chuyển lớn nhất của các tâm không vượt ngưỡng đã chọn. Đề xuất ngưỡng `1e-4` trên thang RGB `[0,255]`, giới hạn 300 vòng; ghi rõ dừng vì hội tụ hay hết giới hạn.
6. **Hoàn tất:** gán lại nhãn theo bộ tâm cuối và tính WCSS tương ứng để nhãn, tâm và phép đo nhất quán.

Ghi WCSS ở cùng một thời điểm của mỗi vòng lặp để kiểm tra xu hướng không tăng trong sai số số học. Nếu có bất thường, kiểm tra phép gán, cập nhật và độ chính xác số học trước khi diễn giải.

### 6.4. Tái tạo và trình bày ảnh

Thay mỗi pixel bằng màu tâm của nhãn tương ứng, rồi reshape theo đúng thứ tự flatten để tạo ảnh `(h,w,3)`. Khi hiển thị, chặn giá trị trong `[0,255]`, làm tròn và đổi về kiểu nguyên.

Để đáp ứng yêu cầu về kích thước ban đầu, thiết kế phân biệt ảnh làm việc `(h,w)` và ảnh đầu vào `(H,W)`: học tâm trên ảnh làm việc, sau đó gán các pixel RGB của ảnh đầu vào vào bộ tâm đã học để dựng ảnh `(H,W,3)`. Không học lại tâm ở bước này và không reshape trực tiếp `h*w` nhãn thành `H*W` pixel. Với ảnh lớn, có thể gán theo từng phần để hạn chế bộ nhớ.

Hiển thị ảnh đầu vào và ảnh phân đoạn ở kích thước ban đầu cạnh nhau, giữ tỷ lệ, kèm K và kích thước. Số màu phân biệt của ảnh phân đoạn không vượt K; không bắt buộc đúng K vì tâm có thể trùng màu sau làm tròn hoặc không được dùng.

### 6.5. Checklist nghiệm thu dự kiến

- Đọc được một ảnh màu tùy chọn, không đảo kênh hoặc kéo méo ảnh.
- Resize và flatten đúng với ảnh chữ nhật; số pixel bằng `h*w`.
- Giải thích được khởi tạo, gán cụm, cập nhật trung bình và điều kiện dừng.
- Nhãn hợp lệ, tâm hữu hạn, WCSS nhất quán với nhãn và tâm cuối.
- Có cách xử lý cụm rỗng, đường dẫn không hợp lệ và K không hợp lệ.
- Kiểm tra ảnh một màu với K=1: tâm bằng màu đó và WCSS bằng 0 trong sai số số học.
- Phân biệt dừng do hội tụ với dừng do hết giới hạn vòng lặp.
- Ảnh tái tạo đúng kích thước; mỗi màu thuộc bảng màu tâm sau phép làm tròn thống nhất.
- Có bố cục so sánh song song và thông tin cấu hình đủ để thực hành lại.

### 6.6. Các vấn đề cần kiểm chứng khi thực hành

Ở giai đoạn thiết kế, nhóm chưa kết luận mức độ phù hợp của cấu hình đối với ảnh được chọn. Bảng dưới đặt câu hỏi và cách kiểm tra dự kiến; chỉ đưa ra nhận xét cụ thể sau khi thực hành có bằng chứng.

| Quyết định thiết kế                       | Câu hỏi cần kiểm chứng                                                                                  | Cách kiểm tra dự kiến                                                              |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| Resize trước phân cụm                      | Kích thước làm việc có giữ được các vùng màu nhóm muốn quan sát không?                      | Đối chiếu ảnh trước và sau resize; điều chỉnh kích thước nếu cần.       |
| Dùng RGB làm baseline                        | Các vùng màu trên ảnh được chọn có được phân biệt phù hợp với mục tiêu minh họa không? | Quan sát ảnh phân đoạn cạnh ảnh đầu vào, ghi lại vùng cần xem xét.       |
| Chọn một K khởi đầu                       | K đã chọn có phù hợp với mức đơn giản hóa màu mong muốn không?                                | Nếu còn thời gian, thử thêm vài K trên cùng ảnh và cùng quy trình.         |
| Cố định seed                                | Lần thực hành lặp lại có nhất quán với cấu hình đã ghi không?                                  | Lặp lại cùng ảnh, cấu hình và môi trường; đối chiếu nhãn và tâm.       |
| Giữ tâm khi cụm rỗng                       | Trường hợp cụm rỗng có được xử lý hợp lệ không?                                                | Chuẩn bị tình huống kiểm tra riêng, kiểm tra tâm hữu hạn và nhãn hợp lệ. |
| Đặt ngưỡng dừng và giới hạn vòng lặp | Thuật toán dừng vì điều kiện nào?                                                                    | Kiểm tra độ dịch chuyển tâm, số vòng lặp và lý do dừng.                    |

Phạm vi đánh giá là phân cụm màu trên ảnh phục vụ bài lab. Việc đánh giá nhận diện vật thể, tính tối ưu toàn cục hoặc hiệu quả trên tập ảnh lớn nằm ngoài kế hoạch này.

### 6.7. Mở rộng tùy chọn

Nếu còn thời gian, khảo sát một vài K trên cùng ảnh bằng WCSS và quan sát trực quan; không bắt buộc phải tìm thấy điểm Elbow rõ ràng. Có thể so sánh thêm một color space khác khi đã thống nhất cách chuyển đổi và thang đo. Các mở rộng này không thay thế hoặc làm điều kiện tiên quyết cho năm bước cốt lõi.

# Thiết kế thuật toán: Phân đoạn màu bằng OpenCV K-Means (Kiểm tra chéo)

---

## 1. Mục đích

Dùng K-Means của OpenCV làm phương án đối chiếu bổ sung cho bài lab. Nhóm sẽ so sánh WCSS, mức tương đồng nhãn và thời gian với Scikit-Learn trên cùng dữ liệu, qua đó tìm hiểu ảnh hưởng của cách khởi tạo và điều kiện dừng.

Đây là phần khảo sát sau baseline, không phải điều kiện bắt buộc để hoàn thành năm bước của đề. Không đặt trước thư viện nào nhanh hơn hoặc cho phân hoạch tốt hơn.

---

## 2. Trạng thái thiết kế (Status)

Tài liệu nêu phương án đối chiếu dự kiến, dữ liệu cần bàn giao và cách diễn giải các phép đo. Các lựa chọn ban đầu sẽ được xem xét trong phạm vi một số ảnh của bài lab.

---

## 3. Dữ liệu sử dụng (Data used)

Dùng ma trận điểm ảnh RGB $(N,3)$ từ cùng ảnh đã resize như các phương án còn lại. Dữ liệu cần có giá trị hữu hạn trên thang $[0,255]$, biểu diễn bằng số thực độ chính xác đơn phù hợp với OpenCV.

Giữ nguyên thứ tự pixel, hệ màu và thang đo giữa hai phương án. K không vượt số màu phân biệt của ảnh làm việc.

---

## 4. Đầu vào (Input)

| Thành phần | Ý nghĩa | Lựa chọn ban đầu |
| --- | --- | --- |
| Ma trận điểm ảnh | Dữ liệu RGB $(N,3)$ | Ảnh sau resize |
| Số cụm K | Dùng cùng K với phương án đối chiếu | K hợp lệ đã chọn, kể cả K=1 |
| Số lần khởi tạo | Chọn lần có WCSS nhỏ nhất | 10 lần |
| Giới hạn vòng lặp | Giới hạn cho mỗi lần chạy | 300 vòng |
| Ngưỡng hội tụ | Ngưỡng dịch chuyển tâm theo quy ước OpenCV | $10^{-4}$ trên thang RGB $[0,255]$ |
| Seed | Hỗ trợ kiểm tra lặp lại trong cùng môi trường | 42 |

Cùng giá trị ngưỡng hoặc seed không có nghĩa hai thư viện dùng cùng điều kiện dừng hay chọn cùng tâm ban đầu. Cần ghi rõ quy ước của mỗi phương án khi so sánh.

---

## 5. Đầu ra (Output)

- Bộ tâm màu kích thước $(K,3)$ và N nhãn cụm theo thứ tự pixel chung.
- Compactness: tổng bình phương khoảng cách từ pixel đến tâm được gán, cùng ý nghĩa toán học với WCSS.
- Thời gian phân cụm, cấu hình đã dùng và các chỉ số đối chiếu với Scikit-Learn.

Nhãn bàn giao cần thống nhất thành một dãy N phần tử. Các chỉ số đối chiếu là phép đo giữa hai phân hoạch, không phải độ chính xác so với nhãn vật thể.

---

## 6. Cách thuật toán hoạt động (How it works)

Quy trình dự kiến: nhận dữ liệu chung → kiểm tra K và thiết lập điều kiện dừng → phân cụm với OpenCV → chuẩn bị nhãn và tâm → đối chiếu với Scikit-Learn → bàn giao cho phần trình bày.

### Bước 1: Thiết lập tiêu chuẩn dừng cho OpenCV

Kết hợp giới hạn vòng lặp với ngưỡng hội tụ: dừng khi đạt giới hạn hoặc thỏa điều kiện dịch chuyển tâm của OpenCV. Ghi rõ cấu hình; không suy ra hội tụ chỉ từ việc lần chạy đã kết thúc.

### Bước 2: Chạy phân cụm bằng OpenCV

Dự kiến dùng khởi tạo k-means++, thực hiện số lần khởi tạo đã thống nhất và chọn lần có WCSS nhỏ nhất. Đo thời gian phần phân cụm, tách khỏi đọc ảnh và vẽ biểu đồ để các phép đo có cùng phạm vi.

### Bước 3: Thuật toán đối chiếu với Scikit-Learn

1. **So sánh WCSS:** với WCSS của Scikit-Learn lớn hơn 0, tính chênh lệch tương đối:

$$
\Delta_{\mathrm{WCSS}}=\frac{|\mathrm{WCSS}_{\mathrm{sklearn}}-\mathrm{WCSS}_{\mathrm{opencv}}|}{\mathrm{WCSS}_{\mathrm{sklearn}}}\times100\%
$$

Nếu mốc bằng 0, chỉ báo chênh lệch tuyệt đối và ghi tỷ lệ không xác định để tránh chia cho 0.

2. **Căn chỉnh nhãn:** số thứ tự cụm chỉ là tên gọi. Lập bảng đếm pixel theo cặp nhãn rồi dùng phép ghép một-một tối đa hóa tổng số pixel trùng nhau, chẳng hạn thuật toán Hungarian.
3. **Tính mức tương đồng:** sau căn chỉnh, tính tỷ lệ pixel cùng nhãn. Có thể bổ sung ARI và NMI, là các chỉ số không phụ thuộc cách đánh số cụm.

---

## 7. Cách đánh giá kết quả (Metrics evaluation)

- Đối chiếu WCSS trên cùng dữ liệu và K; xem xét khác biệt khởi tạo và điều kiện dừng trước khi kết luận về chênh lệch.
- Kết hợp tỷ lệ trùng nhãn, ARI, NMI với ảnh minh họa. Không đặt một ngưỡng phần trăm tùy ý để kết luận hai phân hoạch tương đương.
- Đo thời gian trong cùng môi trường và cùng phạm vi công việc. Nếu thời gian quá ngắn hoặc dao động lớn, lặp lại phép đo và ghi nhận độ dao động.
- Kiểm tra riêng phép căn chỉnh bằng hai bộ nhãn chỉ khác tên cụm; sau căn chỉnh phải trùng hoàn toàn.

---

## 8. Mấy điểm dễ lỗi cần chú ý (Lưu ý khi làm)

- Nhãn cần được đưa về cùng dạng và thứ tự trước khi so sánh hoặc tái tạo ảnh.
- Không nhầm BGR ở bước đọc ảnh với RGB dùng cho phân cụm và hiển thị.
- Không so sánh trực tiếp số thứ tự nhãn giữa hai phương án khi chưa căn chỉnh.
- Cùng K và cùng seed không bảo đảm cùng nghiệm; khác biệt nhãn không tự động là lỗi.
- Với ảnh một màu, WCSS có thể bằng 0; phép chia tương đối cần có cách diễn giải riêng.

---

## 9. Cách nối với các phần khác trong bài (Tính logic hệ thống)

Phần chuẩn bị dữ liệu cung cấp cùng ma trận pixel cho OpenCV và Scikit-Learn. Mỗi phương án bàn giao bộ tâm và nhãn cho phần tái tạo theo cùng quy ước kích thước và màu.

Phần đánh giá nhận hai bộ nhãn cùng thứ tự pixel, WCSS và thời gian có cùng phạm vi đo. Phần trình bày chỉ dùng thông tin đã được đối chiếu này để so sánh, tránh ghép số liệu của một lần thử với ảnh của một cấu hình khác.

---

## 10. Phân công trong nhóm

- **Người phụ trách:** Tường và Trí phối hợp thực hiện.
- **Nhiệm vụ:** chuẩn bị phép đối chiếu, kiểm tra quy ước nhãn và màu, thống nhất cách đo thời gian và giải thích các giới hạn của việc so sánh trong bài lab.

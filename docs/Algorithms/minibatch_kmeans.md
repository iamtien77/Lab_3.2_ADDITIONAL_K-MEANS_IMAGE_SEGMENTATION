# Thiết kế thuật toán: Thử nghiệm tốc độ với Mini-Batch K-Means

---

## 1. Mục đích

Tìm hiểu sự đánh đổi giữa thời gian phân cụm và mức độ gom màu khi dùng Mini-Batch K-Means thay cho Full K-Means. Đây là khảo sát tùy chọn sau khi baseline đáp ứng yêu cầu bài lab.

Câu hỏi chính là Mini-Batch có lợi trên ảnh làm việc đã resize hay không, và việc chỉ cập nhật bằng từng lô nhỏ ảnh hưởng thế nào đến WCSS và ảnh phân đoạn. Không mặc định Mini-Batch sẽ nhanh hơn hoặc có chất lượng tương đương.

---

## 2. Trạng thái thiết kế (Status)

Đây là thiết kế ban đầu cho phép so sánh trên máy cá nhân. Tài liệu xác định dữ liệu, phạm vi đo và tiêu chí nhận xét, không đặt trước mức tăng tốc hoặc mức chênh lệch cần đạt.

---

## 3. Dữ liệu sử dụng (Data used)

Dùng cùng ma trận RGB $(N,3)$ từ ảnh đã resize, giá trị số thực hữu hạn trong $[0,255]$. Full K-Means và Mini-Batch phải dùng cùng ảnh, cùng K, cùng hệ màu và thứ tự pixel.

K là số nguyên dương, không vượt số màu phân biệt. Nếu khảo sát thêm hệ màu khác, mỗi phép đối chiếu phải thực hiện riêng trong hệ màu đó.

---

## 4. Đầu vào (Input)

| Thành phần | Ý nghĩa | Lựa chọn ban đầu |
| --- | --- | --- |
| Ma trận điểm ảnh | Dữ liệu làm việc $(N,3)$ | Ảnh đã resize |
| Số cụm K | Giữ cố định giữa hai phương án | K hợp lệ đã chọn cho ảnh |
| Kích thước lô đề nghị | Số pixel dự kiến trong mỗi lần cập nhật | 256, 512, 1024, 2048, 4096 |
| Cấu hình Full K-Means | Mốc so sánh WCSS, nhãn và thời gian | Cùng dữ liệu, K và cấu hình đã thống nhất |
| Số lần khởi tạo và giới hạn vòng lặp | Điều kiện thực nghiệm cần ghi rõ | 10 lần và 300 vòng |
| Seed | Giúp lặp lại phép lấy mẫu trong cùng môi trường | 42 |

Kích thước lô phải là số nguyên dương. Nếu lớn hơn N, dùng N và ghi rõ cả giá trị đề nghị lẫn giá trị thực dùng. Nếu chọn số lần khởi tạo tự động theo thư viện, cần ghi rõ lựa chọn đó thay vì coi nó là một số lần cố định.

---

## 5. Đầu ra (Output)

Dự kiến tổng hợp một bảng đối chiếu gồm:

- Kích thước lô đề nghị và kích thước lô thực dùng.
- Thời gian phân cụm và hệ số tăng tốc so với Full K-Means.
- WCSS trên toàn bộ dữ liệu làm việc và chênh lệch so với mốc.
- ARI và NMI giữa hai bộ nhãn trên cùng tập pixel.
- Cấu hình thực nghiệm đủ để lặp lại phép thử trong cùng môi trường.

Ảnh minh họa cần dùng bộ tâm và nhãn của đúng cấu hình được ghi trong bảng. Các giá trị lô đề nghị khác nhau nhưng cùng giá trị thực dùng không được diễn giải là các mức lô khác nhau.

---

## 6. Cách thuật toán hoạt động (How it works)

Mini-Batch K-Means khởi tạo K tâm, lấy một lô pixel ngẫu nhiên, gán các pixel đó vào tâm gần nhất rồi cập nhật tâm theo trung bình tích lũy. Lặp quá trình này đến điều kiện dừng. Sau cùng, gán toàn bộ dữ liệu làm việc vào bộ tâm để tính WCSS và các chỉ số đối chiếu.

### Các bước chi tiết trong thử nghiệm

1. **Lấy mốc Full K-Means:** đo phần phân cụm trên dữ liệu chung, lưu nhãn, tâm, WCSS và thời gian. Phạm vi đo không gồm đọc ảnh hoặc vẽ biểu đồ.
2. **Khảo sát từng kích thước lô:** xác định số pixel thực dùng, giữ các cấu hình còn lại nhất quán, đo thời gian với cùng phạm vi như mốc.
3. **Tính các chỉ số:** hệ số tăng tốc là $t_{\mathrm{full}}/t_{\mathrm{batch}}$. Với WCSS mốc lớn hơn 0, tính:

$$
\Delta_{\mathrm{WCSS}}=\frac{\mathrm{WCSS}_{\mathrm{batch}}-\mathrm{WCSS}_{\mathrm{full}}}{\mathrm{WCSS}_{\mathrm{full}}}\times100\%
$$

Nếu WCSS mốc bằng 0, dùng chênh lệch tuyệt đối và ghi tỷ lệ không xác định. Chênh lệch có thể âm do hai phương án tìm được nghiệm khác nhau, nên không mặc định đây luôn là mức suy giảm.

4. **Đối chiếu nhãn và hình ảnh:** tính ARI, NMI trên cùng N pixel rồi xem ảnh tái tạo. Tổng hợp số liệu cùng các giới hạn của phép đo.

---

## 7. Cách đánh giá kết quả (Metrics evaluation)

- **Thời gian:** hệ số tăng tốc lớn hơn 1 nghĩa là Mini-Batch nhanh hơn trong phép đo đó. Nếu thời gian dao động, lặp lại trong cùng điều kiện trước khi nhận xét.
- **WCSS:** xem mức chênh lệch trên cùng dữ liệu, không đặt trước ngưỡng phần trăm chấp nhận khi chưa có mục tiêu cụ thể.
- **ARI và NMI:** đo mức đồng thuận giữa hai phân hoạch. ARI có thể âm; NMI nằm trong $[0,1]$. Giá trị 1 biểu thị phân hoạch trùng nhau bất kể tên nhãn, không biểu thị độ chính xác nhận diện vật thể.
- **Quan sát ảnh:** kiểm tra các vùng màu lớn và chi tiết bị mất để bổ sung cho các chỉ số. Không quyết định chỉ bằng tốc độ hoặc một điểm số duy nhất.

---

## 8. Mấy điểm dễ lỗi cần chú ý (Lưu ý khi làm)

- Lô quá nhỏ có thể đại diện kém cho các màu ít xuất hiện, làm tâm biến động hoặc bỏ sót vùng màu nhỏ.
- Trên ảnh đã thu nhỏ, chi phí quản lý các lô có thể làm Mini-Batch không nhanh hơn Full K-Means; cần đo thay vì giả định.
- Nhiều lô đề nghị có thể cùng bị giới hạn về N. Phải ghi giá trị thực dùng để tránh kết luận sai về ảnh hưởng của kích thước lô.
- Không lấy thời gian của lần thử này ghép với nhãn hoặc WCSS của lần thử khác nếu cấu hình hay dữ liệu đã thay đổi.
- Full K-Means là mốc đối chiếu của bài lab, không phải ground truth. WCSS, ARI và NMI không thay thế đánh giá phân vùng vật thể có nhãn chuẩn.

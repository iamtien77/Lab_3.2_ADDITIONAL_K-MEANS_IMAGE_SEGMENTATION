# Thiết kế thuật toán: K-Means bằng Scikit-Learn & Dò tìm số cụm K bằng Elbow

---

## 1. Mục đích

Phương án này dùng K-Means của Scikit-Learn để đối chiếu với baseline và khảo sát ảnh hưởng của số cụm K. Khởi tạo k-means++ giúp chọn các tâm ban đầu có độ phân tán, nhưng không bảo đảm tìm được nghiệm tối ưu toàn cục.

Elbow là khảo sát tùy chọn: quan sát mức giảm WCSS khi tăng K để gợi ý mức đơn giản hóa màu phù hợp. Không bắt buộc mọi ảnh phải có điểm khuỷu tay rõ ràng.

---

## 2. Trạng thái thiết kế (Status)

Đây là thiết kế ban đầu cho phần đối chiếu và khảo sát K của bài lab. Phạm vi, dữ liệu và cách đánh giá được thống nhất trước khi thực hành; chưa đặt trước K hoặc điểm số cần đạt.

---

## 3. Dữ liệu sử dụng (Data used)

Dùng chung ma trận RGB $(N,3)$ từ ảnh đã resize, giá trị số thực hữu hạn trong $[0,255]$. Các phép đối chiếu phải giữ nguyên ảnh, kích thước làm việc, thứ tự pixel, hệ màu và thang đo.

Đếm số màu phân biệt U trên ảnh làm việc để giới hạn $1\le K\le U$. Số pixel N và số màu U có thể khác nhau rất nhiều.

---

## 4. Đầu vào (Input)

| Thành phần | Ý nghĩa | Lựa chọn ban đầu |
| --- | --- | --- |
| Ma trận điểm ảnh | Dữ liệu RGB $(N,3)$, không rỗng | Ảnh sau resize |
| K chỉ định | Số cụm người thực hành muốn dùng | Tùy chọn; phải nằm trong giới hạn dữ liệu |
| Dải khảo sát K | Các giá trị để vẽ đường cong WCSS | Từ 2 đến 12, giới hạn lại theo U |
| Số lần khởi tạo | Chọn lần chạy có WCSS nhỏ nhất trong các lần thử | 10 lần, ghi rõ khi thay đổi |
| Giới hạn vòng lặp | Giới hạn cho mỗi lần chạy | 300 vòng |
| Seed | Phục vụ kiểm tra lặp lại trong cùng môi trường | 42 |

Với ảnh một màu, chọn K=1 và bỏ qua tìm Elbow. Nếu K được chỉ định hợp lệ, khảo sát thêm không được làm thay đổi lựa chọn đó hoặc trở thành điều kiện để hoàn thành bài lab.

---

## 5. Đầu ra (Output)

- Bảng K và WCSS cho các giá trị hợp lệ đã khảo sát, nếu có.
- K gợi ý từ đường cong, hoặc ghi nhận không xác định được Elbow rõ ràng.
- K được chọn cho ảnh phân đoạn, phân biệt với K gợi ý nếu người thực hành quyết định khác.
- Bộ tâm $(K,3)$, N nhãn cụm và WCSS tương ứng trên dữ liệu làm việc.

---

## 6. Cách thuật toán hoạt động (How it works)

Trước tiên kiểm tra dữ liệu và giới hạn K. Sau đó thực hiện khảo sát nếu cần, chọn K và học bộ tâm dùng để phân đoạn. Với ảnh một màu, chuyển thẳng sang K=1; với K chỉ định, dùng giá trị hợp lệ đó cho bước cuối.

### Giai đoạn 1: Quét Elbow và tìm K gợi ý

1. Lấy các K hợp lệ trong dải dự kiến; không thử số cụm lớn hơn số màu phân biệt.
2. Với mỗi K, giữ nguyên dữ liệu, cách khởi tạo, số lần thử và điều kiện dừng; ghi WCSS.
3. Quan sát điểm đường cong chuyển từ giảm nhanh sang giảm chậm. Có thể dùng Kneedle để hỗ trợ phát hiện điểm này.
4. Nếu không có Elbow rõ, ghi nhận giới hạn đó và chọn K dựa trên mức chi tiết mong muốn. Với quá ít giá trị K để tạo thành đường cong có ý nghĩa, không diễn giải một điểm bất kỳ là Elbow.

Một cách tham khảo hình học là chuẩn hóa hai trục và đo độ lệch của đường cong so với đoạn nối hai đầu. Điểm lệch lớn nhất chỉ là gợi ý; cách này không chứng minh K tối ưu và không đồng nhất hoàn toàn với mọi bước của Kneedle.

### Giai đoạn 2: Huấn luyện mô hình cuối cùng

Dùng K đã chọn để học các tâm màu trên toàn bộ dữ liệu làm việc. Nhãn, tâm và WCSS được bàn giao cho phần tái tạo và đánh giá; lưu lại cấu hình để có thể lặp lại phép thử trong cùng môi trường.

Các ảnh minh họa theo nhiều K cũng phải tuân theo giới hạn số màu. Không cố tạo đủ một danh sách K định trước khi ảnh không có đủ màu.

---

## 7. Cách đánh giá kết quả (Metrics evaluation)

- **WCSS:** chỉ so sánh trên cùng dữ liệu và thang đo. Giá trị tối ưu lý thuyết không tăng khi tăng K, nhưng các lần chạy độc lập có thể dừng ở nghiệm cục bộ khác nhau; không đòi hỏi đường cong thực nghiệm luôn giảm nghiêm ngặt.
- **Quan sát ảnh:** so sánh một vài K hợp lệ để xem mức đơn giản hóa màu và chi tiết bị mất. Không đặt trước K=4 hay K=5 là tốt nhất.
- **Đối chiếu baseline:** giữ cùng K, dữ liệu và ghi rõ khác biệt khởi tạo, điều kiện dừng. Chênh lệch WCSS cần được xem xét, không tự động là bằng chứng baseline sai.
- **Silhouette, nếu khảo sát thêm:** chỉ tính khi tập đánh giá có ít nhất hai cụm và số cụm nhỏ hơn số mẫu. Nếu lấy mẫu, phân bổ gần theo tỷ lệ cụm với ít nhất một mẫu mỗi cụm khi ngân sách cho phép, dùng seed cố định và kiểm tra lại điều kiện sau lấy mẫu. Khi không đủ điều kiện, ghi rõ chỉ số không xác định, không coi đó là chất lượng kém.

---

## 8. Mấy điểm dễ lỗi cần chú ý (Lưu ý khi làm)

- Elbow có thể không rõ trên ảnh có màu biến thiên liên tục; không coi lựa chọn tự động là kết luận cuối cùng.
- Quét nhiều K và nhiều lần khởi tạo làm tăng thời gian, nên dùng ảnh đã resize phù hợp với máy cá nhân.
- Không để khảo sát K hoặc biểu đồ minh họa chặn trường hợp hợp lệ như ảnh một màu, K=1.
- Mẫu Silhouette có thể bỏ sót cụm rất nhỏ; ngay cả lấy mẫu phân tầng cũng cần kiểm tra phân bổ mẫu và số cụm còn lại.
- Nếu khảo sát HSV hoặc LAB, ghi đúng hệ màu và cách đo khoảng cách. Silhouette giữa các hệ tọa độ khác nhau không phải thước đo chung về chất lượng phân vùng vật thể.

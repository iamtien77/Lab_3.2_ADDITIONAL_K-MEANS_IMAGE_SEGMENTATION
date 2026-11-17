# Thiết kế thuật toán: K-Means từ đầu (K-Means from Scratch)

---

## 1. Mục đích

Phương án này là baseline của bài lab, giúp nhóm giải thích được bốn bước K-Means trong đề: khởi tạo tâm, gán điểm ảnh vào tâm gần nhất, cập nhật tâm bằng trung bình và lặp đến điều kiện dừng. Mỗi pixel được biểu diễn bằng ba giá trị RGB; thuật toán chỉ gom màu, không nhận diện vật thể.

Ưu tiên cách tính dễ theo dõi trên ảnh đã thu nhỏ để nhóm có thể kiểm tra từng bước bằng một tập màu nhỏ trước khi thực hành với ảnh.

---

## 2. Trạng thái thiết kế (Status)

Đây là thiết kế ban đầu để thống nhất dữ liệu, cách tính và tiêu chí kiểm tra. Các tham số dưới đây là lựa chọn khởi đầu cho bài lab, có thể điều chỉnh theo ảnh và thời gian thực hành.

---

## 3. Dữ liệu sử dụng (Data used)

- Nhận ảnh RGB đã resize giữ tỷ lệ từ phần chuẩn bị dữ liệu.
- Ma trận điểm ảnh có kích thước $(N,3)$, với $N=h\times w$; mỗi hàng chứa ba giá trị màu theo đúng thứ tự pixel trong ảnh.
- Dùng số thực hữu hạn trên thang $[0,255]$ để tính khoảng cách và trung bình; giữ lại kích thước $(h,w)$ để tái tạo ảnh.
- Đếm số màu phân biệt trên dữ liệu làm việc để xác định giới hạn hợp lệ của K.

---

## 4. Đầu vào (Input)

| Thành phần | Ý nghĩa | Lựa chọn ban đầu |
| --- | --- | --- |
| Ma trận điểm ảnh $X$ | Dữ liệu RGB kích thước $(N,3)$, không rỗng | Ảnh sau resize |
| Số cụm $K$ | Số nguyên dương, không vượt số màu phân biệt | Bắt đầu với 4 nếu đủ màu; giảm theo dữ liệu nếu ít màu hơn |
| Giới hạn vòng lặp | Chặn thời gian tính toán | 300 vòng |
| Ngưỡng dịch chuyển tâm | Điều kiện dừng theo độ dịch chuyển lớn nhất | $10^{-4}$ trên thang RGB $[0,255]$ |
| Seed | Giúp lặp lại phép khởi tạo trong cùng môi trường | 42 |

Ảnh một màu dùng K=1. Nếu người thực hành chọn K ngoài giới hạn, cần thông báo rõ để chọn lại, không âm thầm thay đổi lựa chọn đó.

---

## 5. Đầu ra (Output)

- Bộ tâm màu kích thước $(K,3)$ và N nhãn nguyên thuộc $[0,K-1]$.
- WCSS được tính từ chính bộ tâm và nhãn bàn giao.
- Số vòng lặp, lý do dừng và lịch sử WCSS, độ dịch chuyển tâm qua từng vòng.
- Thông tin các lần gặp cụm rỗng để giải thích khi kiểm tra thuật toán.

Nhãn phải ứng với tâm gần nhất trong bộ tâm cuối. Việc làm tròn màu để hiển thị là bước riêng, không thay thế bộ tâm dùng cho phân cụm.

---

## 6. Cách thuật toán hoạt động (How it works)

Quy trình dự kiến: kiểm tra dữ liệu và K → chọn tâm → gán nhãn → cập nhật tâm → kiểm tra dừng. Nếu chưa dừng, quay lại bước gán nhãn. Khi dừng, gán nhãn và tính WCSS một lần cuối trên bộ tâm sẽ bàn giao.

### Bước 1: Khởi tạo tâm cụm ban đầu

Chọn ngẫu nhiên K màu phân biệt từ dữ liệu bằng seed đã thống nhất. Hai pixel ở vị trí khác nhau có thể cùng màu, nên chọn K vị trí khác nhau chưa đủ bảo đảm K tâm ban đầu khác nhau.

### Bước 2: Gán điểm ảnh vào cụm gần nhất

Với pixel $x_i=[R_i,G_i,B_i]$ và tâm $\mu_j$, tính bình phương khoảng cách Euclidean:

$$
d(x_i,\mu_j)^2=(R_i-\mu_{j,R})^2+(G_i-\mu_{j,G})^2+(B_i-\mu_{j,B})^2
$$

Gán nhãn $c_i=\arg\min_j d(x_i,\mu_j)^2$. Khi khoảng cách bằng nhau, chọn chỉ số tâm nhỏ nhất để giữ quy tắc nhất quán.

$$
\mathrm{WCSS}=\sum_{i=1}^{N} d(x_i,\mu_{c_i})^2
$$

### Bước 3: Cập nhật lại tâm cụm mới

Với tập pixel $S_j$ thuộc cụm j, tâm mới là:

$$
\mu_j^{\mathrm{new}}=\frac{1}{|S_j|}\sum_{x\in S_j}x
$$

Nếu cụm rỗng, đặt lại tâm bằng pixel xa các tâm đang được giữ lại nhất. Khi có nhiều cụm rỗng, cập nhật tập tâm tham chiếu sau mỗi lần chọn để hạn chế chọn lại cùng một màu.

### Bước 4: Kiểm tra điều kiện dừng và lặp lại

Tính độ dịch chuyển lớn nhất:

$$
\mathrm{shift}=\max_j\|\mu_j^{\mathrm{new}}-\mu_j^{\mathrm{old}}\|
$$

Dừng khi độ dịch chuyển không vượt ngưỡng đã chọn hoặc đã hết số vòng lặp. Hai lý do này phải được phân biệt; hết vòng lặp không chứng minh đã hội tụ.

Trước khi bàn giao, cố định độ chính xác số của bộ tâm cuối, rồi gán lại nhãn và tính WCSS theo bộ tâm đó. Không làm tròn hoặc thay đổi độ chính xác của tâm sau phép gán cuối vì pixel gần ranh giới có thể đổi cụm.

---

## 7. Cách đánh giá kết quả (Metrics evaluation)

1. **Tính đúng của phép tính:** dùng tập màu nhỏ để kiểm tra tâm trung bình, quy tắc hòa khoảng cách, cụm rỗng và nhãn gần nhất. Kiểm tra lại WCSS từ nhãn và tâm cuối.
2. **Hội tụ:** theo dõi WCSS ở cùng thời điểm của mỗi vòng và độ dịch chuyển tâm. WCSS phải hữu hạn, không âm; nếu tăng vượt sai số số học, cần xem lại cách tính và xử lý cụm rỗng.
3. **Quan sát ảnh:** đặt ảnh ban đầu cạnh ảnh phân đoạn, kiểm tra mức đơn giản hóa màu và chi tiết bị mất. Không coi một cụm màu là một vật thể.
4. **Trường hợp biên:** ảnh một màu với K=1 phải giữ được màu đó và có WCSS bằng 0 trong sai số số học; ảnh chữ nhật phải giữ đúng kích thước khi tái tạo.

---

## 8. Mấy điểm dễ lỗi cần chú ý (Lưu ý khi làm)

- Cụm rỗng không có giá trị trung bình; phải xử lý trước khi cập nhật tâm.
- Phép trừ và bình phương khoảng cách cần dùng số thực đủ chính xác, tránh tràn số của biểu diễn màu nguyên.
- K phải được kiểm tra theo số màu phân biệt, không chỉ theo tổng số pixel.
- Seed giúp kiểm tra lặp lại trong cùng môi trường, không bảo đảm mọi thư viện hoặc mọi máy cho cùng một phân hoạch.
- K-Means có thể dừng ở nghiệm cục bộ. WCSS nhỏ hơn trên cùng dữ liệu không đồng nghĩa ảnh phân đoạn phù hợp hơn với mọi mục đích quan sát.

---

## 9. Cách nối với các phần khác trong bài (Tính logic hệ thống)

Phần chuẩn bị dữ liệu bàn giao ma trận RGB, thứ tự pixel và kích thước ảnh. Phần phân cụm bàn giao nhãn, tâm cùng thông tin hội tụ cho phần tái tạo và đánh giá.

Để dựng ảnh ở kích thước ban đầu, phần tái tạo gán từng pixel ảnh gốc vào bộ tâm đã học, không học lại tâm. Sau đó làm tròn màu tâm thống nhất để hiển thị. Thống kê diện tích các cụm phải dùng nhãn của đúng kích thước ảnh đang được trình bày.

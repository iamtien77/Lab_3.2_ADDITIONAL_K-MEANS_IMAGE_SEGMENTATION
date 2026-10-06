"""
=============================================================================
MODULE: TIỀN XỬ LÝ ẢNH (IMAGE PREPROCESSING)
=============================================================================
Mục đích (Đáp ứng Task 1 & Task 2 trong đề bài Practice Exercise 3.2):
- Task 1: Chuyển đổi không gian màu phù hợp (RGB, HSV, LAB) từ ảnh BGR ban đầu.
- Task 2: Resize ảnh về kích thước nhỏ hơn (ví dụ: 256x256) để tăng tốc độ phân cụm.
- Task 2: Làm phẳng ảnh (Flatten) thành mảng 2D điểm ảnh (N_pixels, 3 kênh màu).
=============================================================================
"""

from typing import Tuple, Optional
import numpy as np


def convert_color_space(
    img: np.ndarray,
    source_space: str = "BGR",
    target_space: str = "RGB",
) -> np.ndarray:
    """
    Ý LÀM (Đáp ứng Task 1 - Convert the image to a suitable color space):
    1. Kiểm tra không gian màu nguồn (source_space) và đích (target_space):
       - BGR sang RGB: đảo ngược thứ tự kênh màu để hiển thị đúng chuẩn Matplotlib.
       - RGB sang HSV: tách biệt kênh màu sắc (Hue), độ bão hòa (Saturation), giá trị sáng (Value).
       - RGB sang LAB: không gian màu cảm nhận đồng đều (Luminance, A channel, B channel).
    2. Sử dụng cv2.cvtColor nếu có OpenCV; hoặc thuật toán chuyển đổi thuần NumPy / Pillow nếu không có.
    3. Kiểm tra tính toàn vẹn của mảng kết quả (cùng kích thước HxW, 3 kênh).
    4. Trả về mảng ảnh trong không gian màu mới.
    """
    pass


def resize_image(
    img: np.ndarray,
    target_size: int = 256,
    keep_aspect_ratio: bool = True,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Ý LÀM (Đáp ứng Task 2 - Resize the image to a smaller size for faster processing):
    1. Lấy kích thước gốc của ảnh (h_orig, w_orig).
    2. Tính toán kích thước mới:
       - Nếu keep_aspect_ratio = True: tính scale = target_size / max(h_orig, w_orig),
         tính new_w = int(round(w_orig * scale)), new_h = int(round(h_orig * scale)) để không méo hình.
       - Nếu keep_aspect_ratio = False: đặt thẳng new_w = target_size, new_h = target_size.
    3. Áp dụng thuật toán nội suy thu nhỏ hình ảnh (Inter-area hoặc Bilinear interpolation).
    4. Trả về tuple gồm: mảng ảnh đã thu nhỏ và tuple kích thước mới (new_h, new_w).
    """
    pass


def flatten_pixels(
    img: np.ndarray,
    as_float: bool = True,
) -> np.ndarray:
    """
    Ý LÀM (Đáp ứng Task 2 - Flatten the image into a 2D array of pixels):
    1. Tính tổng số điểm ảnh: N_pixels = H * W.
    2. Reshape ma trận ảnh 3 chiều (H, W, 3) thành ma trận 2 chiều (N_pixels, 3).
       Mỗi dòng đại diện cho một điểm ảnh với vector 3 thuộc tính (ví dụ: R, G, B).
    3. Nếu as_float = True: ép kiểu sang float32 để sẵn sàng tính toán khoảng cách Euclidean
       với hiệu năng cao trong K-Means.
    4. Kiểm tra shape kết quả phải đúng (N_pixels, 3).
    5. Trả về ma trận 2D điểm ảnh.
    """
    pass

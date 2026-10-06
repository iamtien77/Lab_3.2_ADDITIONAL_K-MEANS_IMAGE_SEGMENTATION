"""
=============================================================================
MODULE: NẠP VÀ KIỂM ĐỊNH ẢNH (DATA LOADER)
=============================================================================
Mục đích (Đáp ứng Task 1 trong đề bài Practice Exercise 3.2):
- Đọc file ảnh tự chọn (landscape, portrait, liquid, ...) bằng OpenCV (cv2.imread)
  hoặc thư viện PIL (Pillow fallback).
- Kiểm tra tính toàn vẹn cấu trúc ảnh (validate shape, channels, dtype).
- Trích xuất thông tin siêu dữ liệu (metadata): kích thước gốc, tổng số pixel, dung lượng file.
=============================================================================
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Union, Optional
import numpy as np


def load_image(image_path: Optional[Union[str, Path]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Ý LÀM (Đáp ứng Task 1 - Load the Image):
    1. Kiểm tra sự tồn tại của file ảnh (tại image_path hoặc DEFAULT_IMAGE_PATH).
    2. Đọc file ảnh bằng OpenCV (cv2.imread) nếu có sẵn; nếu không dùng PIL.Image.open() để fallback.
    3. Kiểm tra tính hợp lệ của mảng ảnh thông qua hàm validate_image().
    4. Trích xuất metadata chi tiết:
       - Tên file, đường dẫn tuyệt đối.
       - Kích thước gốc (Height, Width, Channels).
       - Tổng số điểm ảnh (Total pixels = Height * Width).
       - Kiểu dữ liệu (uint8) và dung lượng tệp tin (bytes).
       - Định dạng màu sắc ban đầu (BGR hoặc RGB).
    5. Trả về mảng ảnh numpy 3 chiều (H, W, 3) và từ điển metadata.
    """
    pass


def validate_image(img: np.ndarray) -> bool:
    """
    Ý LÀM:
    1. Kiểm tra mảng ảnh không được rỗng (img is not None).
    2. Kiểm tra số chiều của mảng: bắt buộc phải là mảng 3 chiều (Height, Width, Channels).
    3. Kiểm tra số kênh màu: bắt buộc Channels == 3 (ảnh màu RGB/BGR).
    4. Kiểm tra kiểu dữ liệu (dtype): bắt buộc phải là uint8 với giá trị điểm ảnh nằm trong đoạn [0, 255].
    5. Trả về True nếu thỏa mãn mọi điều kiện, ngược lại ném ngoại lệ ValueError/TypeError phù hợp.
    """
    pass

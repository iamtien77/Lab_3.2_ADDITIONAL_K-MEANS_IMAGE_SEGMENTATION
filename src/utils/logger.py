"""
=============================================================================
MODULE: LOGGER GHI NHẬT KÝ THỰC THI (LOGGER)
=============================================================================
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "ImageSegmentation", log_file: Optional[Path] = None) -> logging.Logger:
    """
    Ý LÀM:
    1. Khởi tạo đối tượng logging.Logger theo tên module (name).
    2. Cấu hình định dạng log gồm: thời gian [YYYY-MM-DD HH:MM:SS], cấp độ [INFO/ERROR], tên module và thông điệp.
    3. Thêm StreamHandler để in log trực tiếp ra console (sys.stdout).
    4. Nếu có truyền log_file: thêm FileHandler để lưu log vào file trên đĩa.
    5. Tránh duplicate handlers khi gọi nhiều lần.
    6. Trả về đối tượng Logger sẵn sàng sử dụng.
    """
    pass

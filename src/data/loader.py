"""
=============================================================================
MODULE: NẠP VÀ KIỂM ĐỊNH ẢNH (DATA LOADER)
=============================================================================
Mục đích (Đáp ứng Task 1 trong đề bài Practice Exercise 3.2):
- Đọc file ảnh tự chọn bằng OpenCV (cv2.imread) hoặc PIL (Pillow fallback).
- Kiểm tra tính toàn vẹn cấu trúc ảnh (validate shape, channels, dtype).
- Trích xuất thông tin siêu dữ liệu (metadata): kích thước gốc, tổng số pixel, dung lượng file.
=============================================================================
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Union, Optional
import numpy as np

from src.utils.logger import get_logger
from src.config import DEFAULT_IMAGE_PATH

logger = get_logger("DataLoader")


def load_image(
    image_path: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Đọc ảnh từ image_path (hoặc DEFAULT_IMAGE_PATH nếu không truyền).
    Trả về: (img_rgb_array [H, W, 3] uint8, metadata dict)
    Ưu tiên dùng OpenCV; fallback sang Pillow nếu cv2 không khả dụng.
    """
    # --- Xác định đường dẫn ảnh ---
    if image_path is None:
        image_path = DEFAULT_IMAGE_PATH
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_path}")

    img_rgb: np.ndarray
    source_format: str

    # --- Thử đọc bằng OpenCV ---
    try:
        import cv2
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError("cv2.imread trả về None — file có thể hỏng hoặc định dạng không hỗ trợ.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        source_format = "OpenCV (BGR→RGB)"
        logger.info(f"Đọc ảnh bằng OpenCV: {image_path.name}")
    except (ImportError, ValueError) as e:
        # --- Fallback sang Pillow ---
        logger.warning(f"OpenCV không khả dụng hoặc lỗi ({e}), dùng Pillow fallback.")
        try:
            from PIL import Image as PILImage
            pil_img = PILImage.open(image_path).convert("RGB")
            img_rgb = np.array(pil_img, dtype=np.uint8)
            source_format = "Pillow (RGB)"
            logger.info(f"Đọc ảnh bằng Pillow: {image_path.name}")
        except Exception as pil_err:
            raise RuntimeError(f"Không thể đọc ảnh bằng cả OpenCV lẫn Pillow: {pil_err}") from pil_err

    # --- Kiểm tra hợp lệ ---
    validate_image(img_rgb)

    # --- Trích xuất metadata ---
    h, w, c = img_rgb.shape
    file_size_bytes = image_path.stat().st_size

    metadata: Dict[str, Any] = {
        "filename": image_path.name,
        "filepath": str(image_path.resolve()),
        "original_height": h,
        "original_width": w,
        "channels": c,
        "original_shape": (h, w, c),
        "total_pixels": h * w,
        "dtype": str(img_rgb.dtype),
        "file_size_bytes": file_size_bytes,
        "color_format": source_format,
    }

    logger.info(
        f"Ảnh đã nạp — {image_path.name} | {h}×{w}×{c} | "
        f"{h * w:,} pixels | {file_size_bytes / 1024:.1f} KB"
    )
    return img_rgb, metadata


def validate_image(img: np.ndarray) -> bool:
    """
    Kiểm tra tính toàn vẹn của mảng ảnh.
    Raises: ValueError hoặc TypeError nếu không hợp lệ.
    Returns: True nếu hợp lệ.
    """
    if img is None:
        raise ValueError("Mảng ảnh không được là None.")

    if not isinstance(img, np.ndarray):
        raise TypeError(f"Mảng ảnh phải là np.ndarray, nhận được: {type(img)}")

    if img.ndim != 3:
        raise ValueError(
            f"Mảng ảnh phải có 3 chiều (H, W, C), nhận được {img.ndim} chiều với shape {img.shape}."
        )

    if img.shape[2] != 3:
        raise ValueError(
            f"Ảnh phải có đúng 3 kênh màu (RGB/BGR), nhận được {img.shape[2]} kênh."
        )

    if img.dtype != np.uint8:
        raise TypeError(
            f"Kiểu dữ liệu ảnh phải là uint8, nhận được: {img.dtype}. "
            "Hãy đảm bảo giá trị điểm ảnh nằm trong [0, 255]."
        )

    return True

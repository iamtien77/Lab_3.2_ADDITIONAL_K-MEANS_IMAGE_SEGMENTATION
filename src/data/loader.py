"""Đọc ảnh màu 8 bit và ghi nhận thứ tự kênh của thư viện giải mã."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

from src.config import DEFAULT_IMAGE_PATH

try:
    import cv2
except ImportError:
    cv2 = None


def load_image(image_path: Optional[Union[str, Path]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Trả ảnh màu cùng metadata; dùng ảnh mẫu nếu không truyền đường dẫn."""
    path = Path(image_path if image_path is not None else DEFAULT_IMAGE_PATH).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp ảnh: {path}")
    if not path.is_file():
        raise ValueError(f"Đường dẫn ảnh không trỏ đến tệp: {path}")

    try:
        if cv2 is not None:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            color_space = "BGR"
        else:
            with Image.open(path) as image:
                # Kiểm tra mode trước khi đọc mảng để giữ nguyên thông tin alpha.
                if "A" in image.getbands() or "transparency" in image.info:
                    raise ValueError("Ảnh có kênh alpha chưa được hỗ trợ trong baseline.")
                if image.mode in {"1", "L", "I", "F"} or image.mode.startswith("I;16"):
                    raise ValueError("Ảnh xám chưa được hỗ trợ trong baseline.")
                if image.mode != "RGB":
                    raise ValueError(f"Baseline yêu cầu ảnh RGB 8 bit; mode {image.mode} chưa được hỗ trợ.")
                img = np.array(image)
            color_space = "RGB"
    except OSError as exc:
        raise ValueError(f"Không đọc được tệp ảnh {path}: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"Ảnh tại {path} không hợp lệ: {exc}") from exc

    if img is None:
        raise ValueError(f"Không đọc được tệp ảnh: {path}")
    try:
        validate_image(img)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"Ảnh tại {path} không hợp lệ: {exc}") from exc

    height, width, _ = img.shape
    return img, {
        "filename": path.name,
        "image_path": str(path),
        "original_shape": img.shape,
        "total_pixels": height * width,
        "file_size_bytes": path.stat().st_size,
        "color_space": color_space,
        "dtype": str(img.dtype),
    }


def validate_image(img: np.ndarray) -> bool:
    """Kiểm tra ảnh màu ba kênh, uint8 và có kích thước dương."""
    if img is None:
        raise ValueError("Ảnh rỗng.")
    if not isinstance(img, np.ndarray):
        raise TypeError("Ảnh phải là mảng NumPy.")
    if img.size == 0:
        raise ValueError("Ảnh rỗng hoặc có chiều bằng 0.")
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        raise ValueError("Ảnh xám chưa được hỗ trợ trong baseline.")
    if img.ndim == 3 and img.shape[2] in (2, 4):
        raise ValueError("Ảnh có kênh alpha chưa được hỗ trợ trong baseline.")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Ảnh phải có dạng (h, w, 3).")
    if img.dtype != np.uint8:
        raise TypeError(f"Ảnh phải có dtype uint8, nhận được {img.dtype}.")
    return True

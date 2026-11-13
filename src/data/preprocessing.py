"""Đưa ảnh về RGB, thu nhỏ và chuẩn bị ma trận pixel cho K-Means."""

from numbers import Integral
from typing import Tuple

import numpy as np
from PIL import Image

from src.data.loader import cv2, validate_image


def convert_color_space(
    img: np.ndarray,
    source_space: str = "BGR",
    target_space: str = "RGB",
) -> np.ndarray:
    """Đổi thứ tự kênh RGB/BGR và trả về bản sao độc lập."""
    validate_image(img)
    source = source_space.upper()
    target = target_space.upper()
    if source not in {"BGR", "RGB"} or target not in {"BGR", "RGB"}:
        raise ValueError("Baseline hỗ trợ hai thứ tự kênh RGB và BGR.")
    return img.copy() if source == target else img[..., ::-1].copy()


def resize_image(
    img: np.ndarray,
    target_size: int = 256,
    keep_aspect_ratio: bool = True,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Giới hạn cạnh dài của ảnh theo target_size khi giữ tỷ lệ."""
    validate_image(img)
    if isinstance(target_size, bool) or not isinstance(target_size, Integral) or target_size <= 0:
        raise ValueError("target_size phải là số nguyên dương.")
    height, width = img.shape[:2]
    if keep_aspect_ratio:
        scale = min(1.0, target_size / max(height, width))
        new_height = max(1, round(height * scale))
        new_width = max(1, round(width * scale))
    else:
        new_height = new_width = int(target_size)

    if (height, width) == (new_height, new_width):
        return img.copy(), (height, width)

    size = (new_width, new_height)
    shrinking = new_height <= height and new_width <= width
    if cv2 is not None:
        interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        resized = cv2.resize(img, size, interpolation=interpolation)
    else:
        resample = Image.Resampling.BOX if shrinking else Image.Resampling.BILINEAR
        resized = np.array(Image.fromarray(img).resize(size, resample=resample))
    return resized, resized.shape[:2]


def flatten_pixels(
    img: np.ndarray,
    as_float: bool = True,
) -> np.ndarray:
    """Duyệt theo hàng để trả ma trận pixel trong miền [0, 255]."""
    validate_image(img)
    pixels = img.reshape(-1, 3, order="C")
    return pixels.astype(np.float32) if as_float else pixels.copy()

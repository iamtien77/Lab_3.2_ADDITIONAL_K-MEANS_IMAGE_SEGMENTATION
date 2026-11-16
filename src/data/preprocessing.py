"""
=============================================================================
MODULE: TIỀN XỬ LÝ ẢNH (IMAGE PREPROCESSING)
=============================================================================
Mục đích (Đáp ứng Task 1 & Task 2 trong đề bài Practice Exercise 3.2):
- Task 1: Chuyển đổi không gian màu phù hợp (RGB, HSV, LAB) từ ảnh BGR ban đầu.
- Task 2: Resize ảnh về kích thước nhỏ hơn (cạnh dài <= target_size) để tăng tốc phân cụm.
- Task 2: Làm phẳng ảnh (Flatten) thành mảng 2D điểm ảnh (N_pixels, 3 kênh màu).
=============================================================================
"""

from typing import Tuple, Optional
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("Preprocessing")


def _rgb_to_hsv_numpy(rgb: np.ndarray) -> np.ndarray:
    """
    Chuyển đổi RGB sang HSV thuần NumPy (khi không có cv2).
    Đầu vào: rgb (H, W, 3) uint8 [0, 255].
    Đầu ra: hsv (H, W, 3) float32 với H in [0, 180], S in [0, 255], V in [0, 255] (tương thích OpenCV).
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[:, :, 0], rgb_f[:, :, 1], rgb_f[:, :, 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    diff = cmax - cmin

    # Hue
    h = np.zeros_like(cmax)
    # diff > 0
    mask_r = (cmax == r) & (diff > 0)
    mask_g = (cmax == g) & (diff > 0)
    mask_b = (cmax == b) & (diff > 0)

    h[mask_r] = (60.0 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360.0) % 360.0
    h[mask_g] = (60.0 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120.0) % 360.0
    h[mask_b] = (60.0 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240.0) % 360.0

    # Chuyển H về thang [0, 180] như OpenCV
    h = h / 2.0

    # Saturation
    s = np.zeros_like(cmax)
    mask_cmax = cmax > 0
    s[mask_cmax] = (diff[mask_cmax] / cmax[mask_cmax]) * 255.0

    # Value
    v = cmax * 255.0

    hsv = np.stack([h, s, v], axis=2).astype(np.float32)
    return hsv


def _rgb_to_lab_numpy(rgb: np.ndarray) -> np.ndarray:
    """
    Chuyển đổi RGB sang CIE-LAB chuẩn thuần NumPy.
    """
    # 1. Chuyển sRGB sang Linear RGB
    rgb_f = rgb.astype(np.float32) / 255.0
    mask = rgb_f > 0.04045
    rgb_lin = np.where(mask, ((rgb_f + 0.055) / 1.055) ** 2.4, rgb_f / 12.92)

    # 2. Linear RGB sang XYZ (Ma trận D65 chuẩn)
    x = rgb_lin[:, :, 0] * 0.4124564 + rgb_lin[:, :, 1] * 0.3575761 + rgb_lin[:, :, 2] * 0.1804375
    y = rgb_lin[:, :, 0] * 0.2126729 + rgb_lin[:, :, 1] * 0.7151522 + rgb_lin[:, :, 2] * 0.0721750
    z = rgb_lin[:, :, 0] * 0.0193339 + rgb_lin[:, :, 1] * 0.1191920 + rgb_lin[:, :, 2] * 0.9503041

    # Chuẩn hóa theo điểm trắng D65 (Xn, Yn, Zn)
    x = x / 0.95047
    y = y / 1.00000
    z = z / 1.08883

    # Hàm f(t) cho LAB
    delta = 6.0 / 29.0
    f_x = np.where(x > delta ** 3, x ** (1.0 / 3.0), (x / (3 * delta ** 2)) + (4.0 / 29.0))
    f_y = np.where(y > delta ** 3, y ** (1.0 / 3.0), (y / (3 * delta ** 2)) + (4.0 / 29.0))
    f_z = np.where(z > delta ** 3, z ** (1.0 / 3.0), (z / (3 * delta ** 2)) + (4.0 / 29.0))

    L = 116.0 * f_y - 16.0
    a = 500.0 * (f_x - f_y)
    b = 200.0 * (f_y - f_z)

    lab = np.stack([L, a, b], axis=2).astype(np.float32)
    return lab


def convert_color_space(
    img: np.ndarray,
    source_space: str = "BGR",
    target_space: str = "RGB",
) -> np.ndarray:
    """
    Chuyển đổi không gian màu của ảnh.
    Hỗ trợ: BGR->RGB, RGB->HSV, RGB->LAB, RGB->BGR.
    Dùng cv2.cvtColor nếu có; fallback thuật toán thuần NumPy nếu không có OpenCV.
    """
    source_space = source_space.upper()
    target_space = target_space.upper()

    if source_space == target_space:
        return img.copy()

    # Thử dùng OpenCV trước
    try:
        import cv2

        conversion_map = {
            ("BGR", "RGB"): cv2.COLOR_BGR2RGB,
            ("RGB", "BGR"): cv2.COLOR_RGB2BGR,
            ("BGR", "HSV"): cv2.COLOR_BGR2HSV,
            ("RGB", "HSV"): cv2.COLOR_RGB2HSV,
            ("BGR", "LAB"): cv2.COLOR_BGR2LAB,
            ("RGB", "LAB"): cv2.COLOR_RGB2LAB,
        }

        key = (source_space, target_space)
        if key not in conversion_map:
            raise ValueError(f"Chuyển đổi {source_space}->{target_space} chưa được hỗ trợ.")

        result = cv2.cvtColor(img, conversion_map[key])
        logger.info(f"Chuyển không gian màu (OpenCV): {source_space} -> {target_space}")
        return result

    except ImportError:
        # Fallback thuật toán thuần NumPy
        logger.warning(f"OpenCV không khả dụng. Sử dụng thuật toán thuần NumPy cho {source_space}->{target_space}.")
        
        if set([source_space, target_space]) == {"BGR", "RGB"}:
            result = img[:, :, ::-1].copy()
        elif source_space == "RGB" and target_space == "HSV":
            result = _rgb_to_hsv_numpy(img)
        elif source_space == "RGB" and target_space == "LAB":
            result = _rgb_to_lab_numpy(img)
        elif source_space == "BGR" and target_space == "HSV":
            rgb_temp = img[:, :, ::-1]
            result = _rgb_to_hsv_numpy(rgb_temp)
        elif source_space == "BGR" and target_space == "LAB":
            rgb_temp = img[:, :, ::-1]
            result = _rgb_to_lab_numpy(rgb_temp)
        else:
            raise NotImplementedError(
                f"Chuyển đổi {source_space}->{target_space} chưa được hỗ trợ trên NumPy fallback."
            )

    return result


def resize_image(
    img: np.ndarray,
    target_size: int = 256,
    keep_aspect_ratio: bool = True,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Resize ảnh về kích thước nhỏ hơn (cạnh dài <= target_size).
    Không phóng to ảnh đã nhỏ hơn target_size.
    Returns: (ảnh đã resize, (new_h, new_w))
    """
    h_orig, w_orig = img.shape[:2]

    if keep_aspect_ratio:
        max_side = max(h_orig, w_orig)
        if max_side <= target_size:
            logger.info(f"Ảnh đã nhỏ hơn target ({h_orig}x{w_orig}), giữ nguyên.")
            return img.copy(), (h_orig, w_orig)

        scale = target_size / max_side
        new_h = max(1, int(round(h_orig * scale)))
        new_w = max(1, int(round(w_orig * scale)))
    else:
        new_h = target_size
        new_w = target_size

    try:
        import cv2
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except ImportError:
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(img)
        pil_resized = pil_img.resize((new_w, new_h), PILImage.LANCZOS)
        resized = np.array(pil_resized, dtype=img.dtype)

    logger.info(
        f"Resize ảnh: ({h_orig}x{w_orig}) -> ({new_h}x{new_w}) | scale={new_h / h_orig:.3f}"
    )
    return resized, (new_h, new_w)


def flatten_pixels(
    img: np.ndarray,
    as_float: bool = True,
) -> np.ndarray:
    """
    Flatten ảnh 3D (H, W, 3) thành ma trận 2D (N_pixels, 3).
    Nếu as_float=True: ép sang float32 để tính khoảng cách Euclidean hiệu quả.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"Ảnh đầu vào phải có shape (H, W, 3), nhận được: {img.shape}"
        )

    h, w, c = img.shape
    n_pixels = h * w
    pixel_matrix = img.reshape(n_pixels, c)

    if as_float:
        pixel_matrix = pixel_matrix.astype(np.float32)

    assert pixel_matrix.shape == (n_pixels, 3), (
        f"flatten_pixels: shape bất thường {pixel_matrix.shape}, kỳ vọng ({n_pixels}, 3)"
    )

    logger.info(
        f"Flatten pixels: ({h}x{w}x{c}) -> ({n_pixels}, {c}) | dtype={pixel_matrix.dtype}"
    )
    return pixel_matrix

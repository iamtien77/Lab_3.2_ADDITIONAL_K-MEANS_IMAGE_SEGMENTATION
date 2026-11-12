"""
=============================================================================
MODULE: NẠP VÀ KIỂM ĐỊNH ẢNH ĐẦU VÀO (DATA LOADER)
=============================================================================
Mục đích (Đáp ứng Task 1 trong đề bài Practice Exercise 3.2):
- Đọc ảnh màu bằng OpenCV (kèm hỗ trợ đường dẫn Unicode) hoặc Pillow fallback.
- Tự động hiệu chỉnh hướng xoay EXIF (nếu có) và xử lý kênh Alpha (RGBA/BGRA)
  bằng cách trộn nền trắng hoặc từ chối nghiêm ngặt theo cấu hình.
- Kiểm tra toàn vẹn mảng ảnh 3 chiều (H, W, 3) kiểu uint8 qua validate_image().
- Trích xuất đầy đủ metadata tệp tin, định dạng màu và thống kê phân bố kênh R, G, B.
=============================================================================
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.config import DEFAULT_IMAGE_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def validate_image(img: np.ndarray) -> bool:
    """
    Kiểm tra tính toàn vẹn nghiêm ngặt của mảng ảnh theo chuẩn hợp đồng (H, W, 3) uint8.

    Tham số:
        img: Mảng NumPy đầu vào cần kiểm định.

    Trả về:
        True nếu mảng ảnh thỏa mãn toàn bộ điều kiện hợp lệ.

    Ngoại lệ:
        TypeError: Nếu img là None hoặc không phải kiểu numpy.ndarray.
        ValueError: Nếu sai số chiều, kích thước rỗng, sai số kênh màu hoặc sai dtype uint8.
    """
    if img is None or not isinstance(img, np.ndarray):
        raise TypeError("Image input must be a valid non-null numpy.ndarray.")
    if img.ndim != 3:
        raise ValueError(f"Expected 3D image array (H, W, 3), got {img.ndim}D shape {img.shape}.")
    if img.shape[0] <= 0 or img.shape[1] <= 0:
        raise ValueError(f"Image dimensions must be positive, got shape {img.shape}.")
    if img.shape[2] != 3:
        raise ValueError(f"Expected 3 color channels (RGB), got {img.shape[2]} channels.")
    if img.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image dtype in [0, 255], got {img.dtype}.")
    return True


def is_valid_image(img: np.ndarray) -> bool:
    """
    Kiểm tra nhanh tính hợp lệ của mảng ảnh và trả về True/False mà không ném ngoại lệ.

    Tham số:
        img: Mảng ảnh ứng viên cần kiểm tra.

    Trả về:
        True nếu hợp lệ, False nếu vi phạm bất kỳ điều kiện nào.
    """
    try:
        return validate_image(img)
    except (TypeError, ValueError):
        return False


def _composite_alpha_on_white(rgba_img: np.ndarray) -> np.ndarray:
    """
    Tách kênh trong suốt (Alpha) và trộn ảnh RGBA lên nền trắng thuần (255, 255, 255)
    để tránh hiện tượng viền đen khi đưa vào K-Means trên không gian RGB.
    """
    rgb = rgba_img[..., :3].astype(np.float32)
    alpha = rgba_img[..., 3:4].astype(np.float32) / np.float32(255.0)
    blended = rgb * alpha + np.float32(255.0) * (np.float32(1.0) - alpha)
    return np.clip(np.round(blended), 0, 255).astype(np.uint8)


def _compute_channel_summary(img_rgb: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Tính thống kê cơ bản (min, max, mean, std) cho từng kênh màu R, G, B."""
    channel_names = ("R", "G", "B")
    summary: Dict[str, Dict[str, float]] = {}
    for idx, ch_name in enumerate(channel_names):
        ch = img_rgb[:, :, idx]
        summary[ch_name] = {
            "min": int(ch.min()),
            "max": int(ch.max()),
            "mean": round(float(ch.mean()), 2),
            "std": round(float(ch.std()), 2),
        }
    return summary


def _load_via_pillow(path: Path, allow_alpha: bool = True) -> Tuple[np.ndarray, str, bool]:
    """Đọc ảnh bằng thư viện Pillow (xử lý EXIF orientation, palette và kiểm tra alpha/grayscale)."""
    with Image.open(path) as raw_pil:
        pil_img = ImageOps.exif_transpose(raw_pil)
        orig_mode = pil_img.mode
        alpha_composited = False

        # Từ chối ảnh đa sắc xám hoặc ảnh 1-bit không mang đủ 3 kênh màu
        if orig_mode in {"L", "1", "I", "F"}:
            raise ValueError(f"Grayscale images are not supported (mode={orig_mode}): {path}")

        # Nếu là ảnh Palette ('P'), kiểm tra xem có thông tin trong suốt không
        if orig_mode == "P":
            has_transparency = "transparency" in pil_img.info
            pil_img = pil_img.convert("RGBA" if has_transparency else "RGB")

        if pil_img.mode in {"RGBA", "LA", "PA"}:
            if not allow_alpha:
                raise ValueError(f"Images with alpha channel are not supported (mode={orig_mode}): {path}")
            rgba_arr = np.asarray(pil_img.convert("RGBA"), dtype=np.uint8)
            rgb_arr = _composite_alpha_on_white(rgba_arr)
            alpha_composited = True
            return rgb_arr, orig_mode, alpha_composited

        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        return np.asarray(pil_img, dtype=np.uint8), orig_mode, alpha_composited


def load_image(
    image_path: Optional[Union[str, Path]] = None,
    allow_alpha: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Đọc ảnh màu từ ổ đĩa, kiểm định chất lượng đầu vào và trích xuất siêu dữ liệu chi tiết.

    Tham số:
        image_path: Đường dẫn đến file ảnh đầu vào. Mặc định lấy DEFAULT_IMAGE_PATH.
        allow_alpha: Nếu True, tự động trộn ảnh 4 kênh (RGBA/BGRA) lên nền trắng thay vì báo lỗi.

    Trả về:
        Tuple gồm:
            - img_rgb: Mảng ảnh chuẩn RGB kích thước (H, W, 3), dtype uint8.
            - metadata: Từ điển chứa kích thước, dung lượng, định dạng màu và thống kê kênh.

    Ngoại lệ:
        FileNotFoundError: Nếu file ảnh không tồn tại.
        ValueError: Nếu file rỗng, không đọc được, là ảnh xám hoặc có kênh Alpha khi allow_alpha=False.
    """
    target_path = Path(DEFAULT_IMAGE_PATH if image_path is None else image_path)
    if not target_path.exists() or not target_path.is_file():
        logger.error(f"Không tìm thấy tệp ảnh tại đường dẫn: {target_path}")
        raise FileNotFoundError(f"Image file not found: {target_path}")

    file_size_bytes = int(target_path.stat().st_size)
    if file_size_bytes == 0:
        logger.error(f"Tệp ảnh bị rỗng (0 byte): {target_path}")
        raise ValueError(f"Image file is empty (0 bytes): {target_path}")

    if target_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        logger.warning(
            f"Phần mở rộng tệp '{target_path.suffix}' ít phổ biến, hệ thống vẫn thử giải mã."
        )

    # Kiểm tra trước EXIF orientation bằng Pillow nếu tệp là JPEG/TIFF để tránh xoay ngược chiều
    exif_orientation = 1
    try:
        with Image.open(target_path) as probe:
            exif_data = probe.getexif()
            if exif_data:
                exif_orientation = int(exif_data.get(274, 1))
    except Exception:
        exif_orientation = 1

    alpha_composited = False
    source_color_mode = "BGR"

    if exif_orientation != 1:
        # Ưu tiên Pillow khi ảnh có thẻ xoay EXIF để đồng bộ chính xác hướng hiển thị
        logger.info(
            f"Phát hiện thẻ xoay EXIF (orientation={exif_orientation}) trên '{target_path.name}', tiến hành hiệu chỉnh hướng ảnh."
        )
        img_rgb, source_color_mode, alpha_composited = _load_via_pillow(
            target_path, allow_alpha=allow_alpha
        )
    else:
        # Đọc byte thô bằng NumPy kết hợp cv2.imdecode để an toàn với đường dẫn Unicode trên Windows
        raw_bytes = np.fromfile(str(target_path), dtype=np.uint8)
        raw_img = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)

        if raw_img is None:
            logger.warning(
                f"OpenCV không giải mã được '{target_path.name}', chuyển sang sử dụng Pillow fallback."
            )
            img_rgb, source_color_mode, alpha_composited = _load_via_pillow(
                target_path, allow_alpha=allow_alpha
            )
        elif raw_img.ndim == 2 or (raw_img.ndim == 3 and raw_img.shape[2] == 1):
            logger.error(f"Từ chối ảnh xám (grayscale) vì pipeline yêu cầu ảnh màu 3 kênh: {target_path.name}")
            raise ValueError(f"Grayscale images are not supported: {target_path}")
        elif raw_img.ndim == 3 and raw_img.shape[2] == 4:
            if not allow_alpha:
                logger.error(f"Từ chối ảnh 4 kênh có Alpha do allow_alpha=False: {target_path.name}")
                raise ValueError(f"Images with alpha channel (4 channels) are not supported: {target_path}")
            logger.warning(
                f"Ảnh '{target_path.name}' có 4 kênh (BGRA/RGBA), tự động trộn kênh Alpha trên nền trắng để đưa về RGB."
            )
            rgba_img = cv2.cvtColor(raw_img, cv2.COLOR_BGRA2RGBA)
            img_rgb = _composite_alpha_on_white(rgba_img)
            source_color_mode = "RGBA"
            alpha_composited = True
        elif raw_img.ndim == 3 and raw_img.shape[2] == 3:
            # Nếu dữ liệu ảnh là 16-bit (uint16), co dải về uint8 một cách rõ ràng
            if raw_img.dtype == np.uint16:
                logger.warning(
                    f"Ảnh '{target_path.name}' có độ sâu 16-bit (uint16), chuyển đổi tuyến tính về 8-bit uint8."
                )
                raw_img = (raw_img / 257.0).round().astype(np.uint8)
            img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            source_color_mode = "BGR"
        else:
            raise ValueError(f"Unsupported image shape {raw_img.shape}: {target_path}")

    # Xác nhận đầu ra thỏa mãn 100% hợp đồng dữ liệu (H, W, 3) uint8
    validate_image(img_rgb)

    height, width, channels = img_rgb.shape
    metadata: Dict[str, Any] = {
        "file_name": target_path.name,
        "file_path": str(target_path.resolve()),
        "height": int(height),
        "width": int(width),
        "channels": int(channels),
        "total_pixels": int(height * width),
        "dtype": str(img_rgb.dtype),
        "file_size_bytes": file_size_bytes,
        "file_size_kb": round(file_size_bytes / 1024.0, 2),
        "aspect_ratio": round(float(width) / float(height), 4),
        "color_format": "RGB",
        "source_color_mode": source_color_mode,
        "alpha_composited": alpha_composited,
        "channel_stats": _compute_channel_summary(img_rgb),
    }

    logger.info(
        f"Đã nạp ảnh '{target_path.name}' [{width}x{height}, {metadata['total_pixels']:,} pixels, "
        f"nguồn={source_color_mode}, dung lượng={metadata['file_size_kb']} KB]."
    )
    return img_rgb, metadata

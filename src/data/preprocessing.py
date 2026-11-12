"""
=============================================================================
MODULE: TIỀN XỬ LÝ ẢNH VÀ CHUẨN BỊ MA TRẬN PIXEL (IMAGE PREPROCESSING)
=============================================================================
Mục đích (Đáp ứng Task 1 & Task 2 trong đề bài Practice Exercise 3.2):
- Chuyển đổi không gian màu linh hoạt giữa BGR, RGB, HSV, LAB bằng OpenCV
  kèm nhánh tính toán dự phòng thuần NumPy cho trường hợp kiểm thử độc lập.
- Thu nhỏ ảnh về kích thước làm việc chuẩn (target_size=256) giữ đúng tỷ lệ khung hình,
  không tự ý phóng to ảnh nhỏ, đo lường và ghi log tỷ lệ nén chi phí tính toán.
- Làm phẳng ảnh thành ma trận 2D (N_pixels, 3) theo thứ tự hàng (C-order), kiểm tra
  tính hữu hạn, thống kê số màu phân biệt và hỗ trợ lưu ảnh đã xử lý vào data/processed/.
=============================================================================
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

_DIRECT_COLOR_MAP: Dict[Tuple[str, str], int] = {
    ("BGR", "RGB"): cv2.COLOR_BGR2RGB,
    ("RGB", "BGR"): cv2.COLOR_RGB2BGR,
    ("BGR", "HSV"): cv2.COLOR_BGR2HSV,
    ("RGB", "HSV"): cv2.COLOR_RGB2HSV,
    ("BGR", "LAB"): cv2.COLOR_BGR2LAB,
    ("RGB", "LAB"): cv2.COLOR_RGB2LAB,
    ("HSV", "BGR"): cv2.COLOR_HSV2BGR,
    ("HSV", "RGB"): cv2.COLOR_HSV2RGB,
    ("LAB", "BGR"): cv2.COLOR_LAB2BGR,
    ("LAB", "RGB"): cv2.COLOR_LAB2RGB,
}

_SUPPORTED_SPACES = {"BGR", "RGB", "HSV", "LAB"}


def _rgb_to_hsv_numpy(rgb_uint8: np.ndarray) -> np.ndarray:
    """Chuyển đổi thuần NumPy từ RGB uint8 sang HSV theo thang đo OpenCV (H:0..179, S/V:0..255)."""
    rgb_f = rgb_uint8.astype(np.float32) / np.float32(255.0)
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    c_max = np.max(rgb_f, axis=-1)
    c_min = np.min(rgb_f, axis=-1)
    delta = c_max - c_min

    hue = np.zeros_like(c_max)
    nonzero = delta > 1e-7
    mask_r = nonzero & (c_max == r)
    mask_g = nonzero & (c_max == g)
    mask_b = nonzero & (c_max == b)

    hue[mask_r] = (60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0))
    hue[mask_g] = (60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0))
    hue[mask_b] = (60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0))

    sat = np.where(c_max > 1e-7, (delta / np.where(c_max > 0, c_max, 1.0)) * 255.0, 0.0)
    val = c_max * 255.0
    # Chuẩn hóa kênh Hue về đoạn tuần hoàn [0, 179] giống quy ước uint8 của OpenCV
    hue_cv = np.round(hue / 2.0) % 180.0
    hsv = np.stack([hue_cv, np.round(sat), np.round(val)], axis=-1)
    return np.clip(hsv, 0, 255).astype(np.uint8)


def convert_color_space(
    img: np.ndarray,
    source_space: str = "BGR",
    target_space: str = "RGB",
    use_numpy_fallback: bool = False,
) -> np.ndarray:
    """
    Chuyển đổi ảnh 3 kênh giữa các không gian màu BGR, RGB, HSV và CIE LAB.

    Tham số:
        img: Mảng ảnh đầu vào kích thước (H, W, 3).
        source_space: Không gian màu nguồn ('BGR', 'RGB', 'HSV', 'LAB').
        target_space: Không gian màu đích ('BGR', 'RGB', 'HSV', 'LAB').
        use_numpy_fallback: Nếu True, ưu tiên dùng cài đặt thuần NumPy khi khả dụng (BGR/RGB/HSV).

    Trả về:
        Mảng ảnh mới trong không gian màu đích có cùng chiều cao và chiều rộng.

    Ngoại lệ:
        ValueError: Nếu đầu vào không phải mảng 3 kênh hợp lệ hoặc tên không gian màu sai.
    """
    if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel NumPy array of shape (H, W, 3).")
    if img.shape[0] <= 0 or img.shape[1] <= 0:
        raise ValueError(f"Input image dimensions must be positive, got shape {img.shape}.")
    if not np.isfinite(img).all():
        raise ValueError("Input image contains NaN or Inf values.")

    src = source_space.strip().upper()
    dst = target_space.strip().upper()
    if src not in _SUPPORTED_SPACES or dst not in _SUPPORTED_SPACES:
        raise ValueError(
            f"Unsupported color space conversion: {source_space} -> {target_space}. "
            f"Supported spaces: {sorted(_SUPPORTED_SPACES)}."
        )

    if src == dst:
        return img.copy()

    # Nhánh dự phòng thuần NumPy cho các cặp chuyển đổi cơ bản
    if use_numpy_fallback and {src, dst}.issubset({"BGR", "RGB", "HSV"}):
        rgb = img[:, :, ::-1].copy() if src == "BGR" else img.copy()
        if src != "HSV":
            if dst == "RGB":
                converted = rgb
            elif dst == "BGR":
                converted = rgb[:, :, ::-1].copy()
            else:
                converted = _rgb_to_hsv_numpy(rgb)
            return converted

    direct_code = _DIRECT_COLOR_MAP.get((src, dst))
    if direct_code is not None:
        converted = cv2.cvtColor(img, direct_code)
    else:
        # Chuyển trung gian qua không gian RGB đối với cặp chuyển đổi HSV <-> LAB
        intermediate_rgb = cv2.cvtColor(img, _DIRECT_COLOR_MAP[(src, "RGB")])
        converted = cv2.cvtColor(intermediate_rgb, _DIRECT_COLOR_MAP[("RGB", dst)])

    if converted.shape != img.shape:
        raise ValueError(
            f"Converted image shape mismatch: expected {img.shape}, got {converted.shape}."
        )

    logger.info(
        f"Đã chuyển đổi không gian màu {src} -> {dst} | dải giá trị từng kênh: "
        f"ch0[{int(converted[..., 0].min())}..{int(converted[..., 0].max())}], "
        f"ch1[{int(converted[..., 1].min())}..{int(converted[..., 1].max())}], "
        f"ch2[{int(converted[..., 2].min())}..{int(converted[..., 2].max())}]."
    )
    return converted


def resize_image(
    img: np.ndarray,
    target_size: int = 256,
    keep_aspect_ratio: bool = True,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Thu nhỏ ảnh đầu vào theo target_size và giữ nguyên tỷ lệ khung hình (không phóng to ảnh nhỏ).

    Tham số:
        img: Mảng ảnh đầu vào (H, W, C) hoặc (H, W).
        target_size: Kích thước cạnh dài nhất mong muốn khi giữ tỷ lệ, hoặc cạnh hình vuông.
        keep_aspect_ratio: Nếu True, giữ nguyên tỷ lệ rộng/cao để ảnh không bị méo.

    Trả về:
        Tuple gồm:
            - resized_img: Mảng ảnh sau khi resize.
            - new_shape: Tuple kích thước mới (new_height, new_width).

    Ngoại lệ:
        ValueError: Nếu kích thước mảng hoặc target_size không hợp lệ.
    """
    if not isinstance(img, np.ndarray) or img.ndim not in (2, 3):
        raise ValueError("Input image must be a 2D or 3D NumPy array.")
    if img.shape[0] <= 0 or img.shape[1] <= 0:
        raise ValueError(f"Input image dimensions must be positive, got shape {img.shape}.")
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}.")

    height, width = img.shape[:2]

    if keep_aspect_ratio:
        # Giữ nguyên ảnh nếu cạnh dài nhất đã nhỏ hơn hoặc bằng ngưỡng mục tiêu
        if max(height, width) <= target_size:
            logger.info(
                f"Ảnh {width}x{height} có cạnh dài <= {target_size} px, giữ nguyên kích thước không phóng to."
            )
            return img.copy(), (int(height), int(width))

        scale = float(target_size) / float(max(height, width))
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
    else:
        new_width = int(target_size)
        new_height = int(target_size)

    # Dùng INTER_AREA khi thu nhỏ để khử răng cưa và giữ màu sắc vùng mượt tốt hơn
    interpolation = (
        cv2.INTER_AREA if (new_height < height or new_width < width) else cv2.INTER_LINEAR
    )
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=interpolation)

    orig_pixels = height * width
    new_pixels = new_height * new_width
    reduction_pct = (1.0 - (new_pixels / float(orig_pixels))) * 100.0
    logger.info(
        f"Đã resize ảnh {width}x{height} ({orig_pixels:,} px) -> {new_width}x{new_height} "
        f"({new_pixels:,} px, giảm {reduction_pct:.2f}% chi phí pixel)."
    )
    return resized_img, (int(new_height), int(new_width))


def flatten_pixels(
    img: np.ndarray,
    as_float: bool = True,
) -> np.ndarray:
    """
    Làm phẳng ảnh 3 chiều (H, W, 3) thành ma trận đặc trưng 2D (N_pixels, 3) theo thứ tự hàng.

    Tham số:
        img: Mảng ảnh đầu vào (H, W, 3).
        as_float: Nếu True, chuyển kiểu dữ liệu sang float32 để tính khoảng cách Euclidean nhanh và chính xác.

    Trả về:
        Ma trận 2D liền mạch bộ nhớ (contiguous) kích thước (H*W, 3).

    Ngoại lệ:
        ValueError: Nếu shape khác (H, W, 3) hoặc tồn tại giá trị không hữu hạn (NaN/Inf).
    """
    if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Input image must have shape (H, W, 3).")
    if img.shape[0] <= 0 or img.shape[1] <= 0:
        raise ValueError(f"Input image dimensions must be positive, got shape {img.shape}.")
    if not np.isfinite(img).all():
        raise ValueError("Input image contains non-finite values (NaN or Inf).")

    height, width, channels = img.shape
    pixels = np.ascontiguousarray(img).reshape(height * width, channels)
    if as_float:
        pixels = pixels.astype(np.float32, copy=False)

    # Kiểm tra nhanh số lượng màu phân biệt trên dữ liệu vừa làm phẳng để phát hiện sớm ảnh quá ít màu
    sample_step = max(1, len(pixels) // 100000)
    unique_count = int(len(np.unique(pixels[::sample_step], axis=0)))
    mem_mb = pixels.nbytes / (1024.0 * 1024.0)
    logger.info(
        f"Đã làm phẳng ảnh ({height}x{width}x{channels}) -> ma trận {pixels.shape} "
        f"({pixels.dtype}, {mem_mb:.2f} MB, ~{unique_count:,} màu phân biệt trên tập mẫu)."
    )
    return pixels


def preprocess_for_clustering(
    img_rgb: np.ndarray,
    target_size: int = 256,
    color_space: str = "RGB",
    save_dir: Optional[Path] = None,
    file_prefix: str = "preprocessed",
) -> Dict[str, Any]:
    """
    Điều phối hoàn chỉnh bước tiền xử lý dữ liệu: chuyển không gian màu (nếu cần),
    resize giữ tỷ lệ, làm phẳng thành ma trận pixel float32 và lưu snapshot vào data/processed/.

    Tham số:
        img_rgb: Mảng ảnh gốc RGB kích thước (H, W, 3) uint8.
        target_size: Kích thước cạnh dài nhất sau khi thu nhỏ.
        color_space: Không gian màu làm việc cho phân cụm ('RGB', 'HSV', 'LAB').
        save_dir: Thư mục tùy chọn (ví dụ PROCESSED_DATA_DIR) để lưu ảnh đã resize và thông số JSON.
        file_prefix: Tiền tố đặt tên tệp lưu trong save_dir.

    Trả về:
        Từ điển chứa ảnh làm việc, kích thước mới, ma trận pixel 2D và đường dẫn snapshot (nếu lưu).
    """
    working_color_space = color_space.strip().upper()
    resized_rgb, resized_shape = resize_image(img_rgb, target_size=target_size, keep_aspect_ratio=True)

    if working_color_space != "RGB":
        working_img = convert_color_space(resized_rgb, source_space="RGB", target_space=working_color_space)
    else:
        working_img = resized_rgb

    pixel_matrix = flatten_pixels(working_img, as_float=True)

    saved_image_path: Optional[str] = None
    saved_meta_path: Optional[str] = None
    if save_dir is not None:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        img_out_path = out_dir / f"{file_prefix}_{working_color_space.lower()}_{resized_shape[1]}x{resized_shape[0]}.png"
        meta_out_path = out_dir / f"{file_prefix}_summary.json"

        # Lưu ảnh RGB đã thu nhỏ bằng cv2.imencode để an toàn đường dẫn Unicode trên Windows
        ok, buf = cv2.imencode(".png", cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR))
        if ok:
            buf.tofile(str(img_out_path))
            saved_image_path = str(img_out_path)

        summary_payload = {
            "original_shape": [int(img_rgb.shape[0]), int(img_rgb.shape[1]), int(img_rgb.shape[2])],
            "resized_shape": [int(resized_shape[0]), int(resized_shape[1])],
            "working_color_space": working_color_space,
            "pixel_matrix_shape": [int(pixel_matrix.shape[0]), int(pixel_matrix.shape[1])],
            "pixel_matrix_dtype": str(pixel_matrix.dtype),
        }
        meta_out_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        saved_meta_path = str(meta_out_path)
        logger.info(f"Đã lưu dữ liệu sau tiền xử lý vào thư mục: {out_dir}")

    return {
        "resized_rgb": resized_rgb,
        "working_image": working_img,
        "resized_shape": resized_shape,
        "pixel_matrix": pixel_matrix,
        "color_space": working_color_space,
        "saved_image_path": saved_image_path,
        "saved_meta_path": saved_meta_path,
    }


def convert_centroids_to_rgb(
    centroids: np.ndarray,
    source_space: str = "RGB",
) -> np.ndarray:
    """
    Chuyển đổi tọa độ tâm cụm (K, 3) từ không gian màu làm việc về hệ RGB [0, 255] để hiển thị.

    Tham số:
        centroids: Mảng tâm cụm kích thước (K, 3).
        source_space: Không gian màu nguồn của tâm cụm ('RGB', 'BGR', 'HSV', 'LAB').

    Trả về:
        Mảng tâm cụm float32 trong không gian RGB [0, 255] kích thước (K, 3).

    Ngoại lệ:
        ValueError: Nếu shape khác (K, 3), chứa NaN/Inf hoặc không gian màu không hợp lệ.
    """
    arr = np.asarray(centroids, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise ValueError("centroids must be a non-empty 2D array of shape (K, 3).")
    if not np.isfinite(arr).all():
        raise ValueError("centroids contains NaN or Inf values.")

    src = source_space.strip().upper()
    if src not in _SUPPORTED_SPACES:
        raise ValueError(
            f"Unsupported source color space: {source_space}. Supported: {sorted(_SUPPORTED_SPACES)}."
        )
    if src == "RGB":
        return np.clip(arr, 0.0, 255.0).astype(np.float32)

    # Kẹp đúng biên giá trị OpenCV uint8 trước khi chuyển đổi ngược về RGB
    palette_uint8 = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    if src == "HSV":
        palette_uint8[:, 0] = np.clip(palette_uint8[:, 0], 0, 179)

    rgb_strip = convert_color_space(
        palette_uint8.reshape(1, -1, 3),
        source_space=src,
        target_space="RGB",
    )
    return rgb_strip.reshape(-1, 3).astype(np.float32)

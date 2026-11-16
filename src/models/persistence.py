"""
=============================================================================
MODULE: LƯU TRỮ VÀ NẠP MÔ HÌNH (MODEL PERSISTENCE)
=============================================================================
Mục đích:
- Đóng gói toàn bộ mô hình K-Means và metadata liên quan vào file .joblib.
- Nạp lại mô hình để phục vụ suy luận phân khúc (inference) cho các ảnh mới.
=============================================================================
"""

from pathlib import Path
from typing import Any, Dict, Tuple, Optional
from datetime import datetime
import joblib

from src.utils.logger import get_logger

logger = get_logger("ModelPersistence")


def save_model_pipeline(
    model: Any,
    save_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Đóng gói model + metadata và lưu xuống đĩa bằng joblib.
    metadata được bổ sung trường saved_at tự động.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if metadata is None:
        metadata = {}

    metadata["saved_at"] = datetime.now().isoformat()

    payload = {
        "model": model,
        "metadata": metadata,
    }

    joblib.dump(payload, save_path, compress=3)
    logger.info(f"Đã lưu mô hình tại: {save_path}")


def load_model_pipeline(
    load_path: Path,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Nạp mô hình từ file .joblib.
    Returns: (model, metadata)
    Raises: FileNotFoundError nếu file không tồn tại.
    """
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file mô hình: {load_path}")

    payload = joblib.load(load_path)
    model = payload["model"]
    metadata = payload.get("metadata", {})
    logger.info(f"Đã nạp mô hình từ: {load_path} (saved_at={metadata.get('saved_at', 'N/A')})")
    return model, metadata

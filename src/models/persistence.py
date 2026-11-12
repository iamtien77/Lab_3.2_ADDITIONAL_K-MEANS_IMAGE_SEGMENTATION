"""
=============================================================================
MODULE: LƯU TRỮ VÀ NẠP MÔ HÌNH (MODEL PERSISTENCE)
=============================================================================
Mục đích:
- Đóng gói mô hình phân cụm K-Means đã huấn luyện cùng metadata cấu hình vào tệp .joblib.
- Nạp lại mô hình và kiểm định tính toàn vẹn trước khi sử dụng để suy luận ảnh mới.
=============================================================================
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib

from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_model_pipeline(
    model: Any,
    save_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Lưu mô hình K-Means và từ điển metadata liên quan xuống tệp tin nén .joblib.

    Tham số:
        model: Đối tượng mô hình đã huấn luyện (KMeans hoặc KMeansFromScratch).
        save_path: Đường dẫn file .joblib đầu ra.
        metadata: Từ điển chứa thông tin bổ sung (optimal_k, color_space, timestamps, v.v.).
    """
    if model is None:
        raise ValueError("Cannot save a None model instance.")

    target_path = Path(save_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    enriched_metadata = dict(metadata) if metadata is not None else {}
    enriched_metadata.setdefault(
        "saved_at_utc", datetime.now(timezone.utc).isoformat()
    )

    payload = {
        "model": model,
        "metadata": enriched_metadata,
    }
    joblib.dump(payload, target_path, compress=3)
    size_kb = round(target_path.stat().st_size / 1024.0, 2)
    logger.info(
        f"Đã lưu gói mô hình và metadata tại '{target_path}' (dung lượng={size_kb} KB)."
    )


def load_model_pipeline(
    load_path: Path,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Nạp lại mô hình K-Means và metadata đã lưu từ tệp .joblib trên ổ đĩa.

    Tham số:
        load_path: Đường dẫn đến file .joblib cần nạp.

    Trả về:
        Tuple gồm: (model, metadata).

    Ngoại lệ:
        FileNotFoundError: Nếu không tìm thấy file tại load_path.
        ValueError: Nếu cấu trúc gói dữ liệu nạp lên không đúng định dạng mong đợi.
    """
    source_path = Path(load_path)
    if not source_path.exists() or not source_path.is_file():
        logger.error(f"Không tìm thấy tệp mô hình tại: {source_path}")
        raise FileNotFoundError(f"Saved model file not found: {source_path}")

    payload = joblib.load(source_path)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(
            f"Invalid saved model payload at '{source_path}': expected dictionary with 'model' key."
        )

    model = payload["model"]
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    logger.info(f"Đã nạp thành công mô hình K-Means từ '{source_path}'.")
    return model, metadata

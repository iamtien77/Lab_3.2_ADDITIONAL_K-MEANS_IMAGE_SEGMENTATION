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
import joblib


def save_model_pipeline(
    model: Any,
    save_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Ý LÀM:
    1. Tạo thư mục cha của save_path nếu chưa tồn tại.
    2. Đóng gói đối tượng dictionary gồm: model và metadata (optimal_k, centroids, timestamps).
    3. Sử dụng joblib.dump để lưu đối tượng xuống ổ đĩa với độ nén tối ưu.
    4. Ghi log hoàn thành lưu trữ.
    """
    pass


def load_model_pipeline(
    load_path: Path,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Ý LÀM:
    1. Kiểm tra sự tồn tại của file tại load_path.
    2. Sử dụng joblib.load để nạp đối tượng từ đĩa.
    3. Tách mô hình (model) và siêu dữ liệu (metadata).
    4. Trả về tuple (model, metadata).
    """
    pass

"""
=============================================================================
MODULE: TIỆN ÍCH GHI NHẬT KÝ THỰC THI (LOGGER)
=============================================================================
Mục đích:
- Cung cấp hàm get_logger() tạo đối tượng logging.Logger thống nhất toàn dự án.
- Hỗ trợ xuất log ra console (stdout) với UTF-8 và lưu ra tệp tin (.log) tùy chọn.
- Ngăn chặn hiện tượng ghi trùng lặp handler khi được import ở nhiều module khác nhau.
=============================================================================
"""

import io
import logging
import sys
from pathlib import Path
from typing import Optional


def _ensure_utf8_stream(stream: object) -> object:
    """Bảo đảm luồng xuất console hỗ trợ UTF-8 trên Windows để hiển thị tiếng Việt có dấu."""
    if hasattr(stream, "reconfigure"):
        try:
            encoding = getattr(stream, "encoding", "") or ""
            if encoding.lower() != "utf-8":
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            # Bỏ qua nếu stream đã bị đóng hoặc bị capture bởi trình test (unittest/pytest)
            pass
    elif isinstance(stream, io.TextIOWrapper):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    return stream


def get_logger(
    name: str = "ImageSegmentation",
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Khởi tạo hoặc lấy lại đối tượng logger đã cấu hình handler console và file (nếu có).

    Tham số:
        name: Tên định danh của logger (thường truyền __name__).
        log_file: Đường dẫn tùy chọn tới file .log trên đĩa; tự động tạo thư mục cha.

    Trả về:
        Đối tượng logging.Logger đã sẵn sàng ghi log và không bị lặp handler.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Kiểm tra StreamHandler console (loại trừ FileHandler vì FileHandler kế thừa StreamHandler)
    has_console_handler = any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    )
    if not has_console_handler:
        safe_stream = _ensure_utf8_stream(sys.stdout)
        console_handler = logging.StreamHandler(safe_stream)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Đăng ký FileHandler nếu người dùng truyền log_file và đường dẫn đó chưa gắn handler
    if log_file is not None:
        resolved_path = Path(log_file).resolve()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        has_file_handler = any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename).resolve() == resolved_path
            for handler in logger.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(resolved_path, encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger

"""
=============================================================================
MODULE: LOGGER GHI NHẬT KÝ THỰC THI (LOGGER)
=============================================================================
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "ImageSegmentation", log_file: Optional[Path] = None) -> logging.Logger:
    """
    Khởi tạo đối tượng Logger với StreamHandler (console) và FileHandler (tùy chọn).
    Tránh duplicate handlers khi hàm được gọi nhiều lần.
    """
    logger = logging.getLogger(name)

    # Nếu đã có handler thì không thêm nữa
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # StreamHandler -> stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # FileHandler -> log file (tùy chọn)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

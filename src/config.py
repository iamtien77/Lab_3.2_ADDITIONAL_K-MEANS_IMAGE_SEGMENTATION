"""
=============================================================================
MODULE: CẤU HÌNH HỆ THỐNG DỰ ÁN (CONFIG)
=============================================================================
Mục đích:
- Quản lý tập trung toàn bộ cấu hình, hằng số, đường dẫn và siêu tham số của dự án.
- Thiết lập đường dẫn thư mục chuẩn: data/raw, data/processed, results/figures,
  results/metrics, results/saved_models.
- Cố định Random Seed để đảm bảo tính tái lập (reproducibility) trong toàn bộ pipeline.
- Định nghĩa các tham số phân đoạn ảnh K-Means (TARGET_SIZE, K_RANGE, MAX_ITER).
- Cung cấp cấu hình thẩm mỹ hiển thị (Matplotlib, Bảng màu doanh nghiệp, DPI).
=============================================================================
"""

import os
import sys
import warnings
from pathlib import Path
import numpy as np

# Vô hiệu hóa các cảnh báo không ảnh hưởng kết quả thực thi
warnings.filterwarnings("ignore")

# 1. ĐỊNH NGHĨA CÁC ĐƯỜNG DẪN THƯ MỤC CHUẨN TRONG DỰ ÁN
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"           # Lưu trữ các biểu đồ trực quan (.png)
METRICS_DIR = RESULTS_DIR / "metrics"           # Lưu trữ các file số liệu đánh giá (.json, .csv)
SAVED_MODELS_DIR = RESULTS_DIR / "saved_models" # Lưu trữ mô hình K-Means đã huấn luyện (.joblib)

DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"

# Tự động khởi tạo cấu trúc thư mục nếu chưa tồn tại trên ổ đĩa
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    SAVED_MODELS_DIR,
    NOTEBOOKS_DIR,
    TESTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# 2. FILE ẢNH MẪU MẶC ĐỊNH
DEFAULT_IMAGE_NAME = "image-liquid.jpg"
DEFAULT_IMAGE_PATH = RAW_DATA_DIR / DEFAULT_IMAGE_NAME

if not DEFAULT_IMAGE_PATH.exists():
    raw_images = list(RAW_DATA_DIR.glob("*.jpg")) + list(RAW_DATA_DIR.glob("*.png"))
    if raw_images:
        DEFAULT_IMAGE_PATH = raw_images[0]

# 3. SIÊU THAM SỐ VÀ CẤU HÌNH THỰC NGHIỆM
RANDOM_SEED: int = 42
TARGET_SIZE: int = 256           # Giới hạn cạnh dài nhất khi thu nhỏ ảnh
BASELINE_K: int = 4              # Số cụm khởi tạo ban đầu
FINAL_K: int = 5                 # Số cụm tối ưu từ Elbow Method
K_SEARCH_MIN: int = 2            # Dải quét K nhỏ nhất
K_SEARCH_MAX: int = 12           # Dải quét K lớn nhất
K_SEARCH_RANGE = range(K_SEARCH_MIN, K_SEARCH_MAX + 1)

MAX_ITER: int = 300              # Số vòng lặp tối đa của K-Means
N_INIT: int = 10                 # Số lần khởi tạo ngẫu nhiên
TOLERANCE: float = 1e-4          # Ngưỡng hội tụ (tolerance)

# 4. BẢNG MÀU & TIỆN ÍCH HIỂN THỊ
FIGURE_DPI: int = 150
COLOR_PALETTE = "tab10"


def print_section(title: str, subtitle: str = "") -> None:
    """In tiêu đề từng bước pipeline ra console một cách chuyên nghiệp."""
    border = "=" * 80
    print(f"\n{border}")
    print(f"  {title.upper()}")
    if subtitle:
        print(f"  -> {subtitle}")
    print(f"{border}\n")

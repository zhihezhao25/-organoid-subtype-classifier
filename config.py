"""
配置中心 —— 所有可调参数集中管理
"""
from pathlib import Path
import torch

# ===== 路径 =====
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"

# ===== 数据 =====
CSV_PATH = DATA_DIR / "dataset_overview.csv"
IMAGE_DIR = DATA_DIR / "imgs"
MASK_DIR = DATA_DIR / "labels"

# ===== 图像处理 =====
IMAGE_SIZE = 512          # 输入分辨率
USE_MASK = False          # False=全视野直接分类 True=裁剪后分类
CROP_MARGIN = 20
MIN_ORGANOID_AREA = 500

# ===== 训练 =====
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
SEED = 42

# ===== 模型 =====
BACKBONE = "convnext_tiny"
DROPOUT = 0.4
FREEZE_STAGES = 0

# ===== 硬件 =====
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
AMP = False

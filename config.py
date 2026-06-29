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
# 实际数据结构:
# data/
#   dataset_overview.csv   ← 图像清单 [org_id, img_id, Day, Clone, Imaging, ...]
#   imgs/                  ← .jpg (LabA) 和 .tif (LabB) 的原始图像
#   labels/                ← .npy 二值分割标注
CSV_PATH = DATA_DIR / "dataset_overview.csv"
IMAGE_DIR = DATA_DIR / "imgs"
MASK_DIR = DATA_DIR / "labels"

# ===== 图像处理 =====
IMAGE_SIZE = 384          # 输入分辨率 (224→384，捕捉更多形态细节)
CROP_MARGIN = 20          # 从mask bbox向外扩展的像素数
MIN_ORGANOID_AREA = 500   # 过滤太小的碎片 (像素)

# ===== 训练 =====
BATCH_SIZE = 16  # 384分辨率吃内存，降到16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0  # MPS 上多进程有问题，设 0
SEED = 42

# ===== 模型 =====
BACKBONE = "convnext_tiny"  # 可选: resnet50, efficientnet_b0/b3, convnext_tiny
DROPOUT = 0.4
FREEZE_STAGES = 0             # 解冻所有层，充分训练

# ===== 硬件 =====
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
AMP = False                   # MPS 不支持 autocast

# ===== 日志 =====
WANDB_PROJECT = "organoid-subtype"
EXPERIMENT_NAME = "efficientnet_b3_baseline"

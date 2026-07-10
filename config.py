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

# ===== 提升泛化 =====
USE_WEIGHTED_SAMPLER = True   # 类别不均衡时按类别均衡采样
MIXUP_ALPHA = 0.2             # 0=关闭；小数据集推荐 0.1~0.4
CUTMIX_ALPHA = 0.0            # 0=关闭；如过拟合可试 0.2~1.0
MIX_PROB = 0.5                # 每个 batch 使用 MixUp/CutMix 的概率
GRAD_CLIP_NORM = 1.0          # 0=关闭；防止微调时梯度爆炸
EARLY_STOP_PATIENCE = 12
SAVE_METRIC = "f1"           # "f1" 比 acc 更适合类别不均衡

# ===== 模型 =====
BACKBONE = "convnext_tiny"
DROPOUT = 0.4
FREEZE_STAGES = 0

# ===== 硬件 =====
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
AMP = False

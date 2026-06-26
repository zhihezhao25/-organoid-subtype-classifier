"""
数据加载管道 —— 从全视野图像 + mask 自动裁剪单个类器官

实际数据结构:
  data/
    dataset_overview.csv   [org_id, img_id, Day, Clone, Imaging, ...]
    imgs/                  .jpg (LabA) 和 .tif (LabB)
    labels/                .npy 二值 mask

核心逻辑:
  1. 读取全视野图像 + 对应 .npy mask
  2. mask → bbox → crop (margin=20px)
  3. resize → 224×224 → DataLoader
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import config


# ============================================================
# 1. 从 mask 提取单个类器官 ROI
# ============================================================

def extract_organoid_roi(image: np.ndarray, mask: np.ndarray, margin: int = 20):
    """
    Args:
        image: 全视野 RGB (H, W, 3)
        mask:  二值 mask (H, W), 1=类器官
        margin: bbox 外扩像素
    Returns:
        cropped 或 None
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None

    y1 = max(0, ys.min() - margin)
    y2 = min(image.shape[0], ys.max() + margin)
    x1 = max(0, xs.min() - margin)
    x2 = min(image.shape[1], xs.max() + margin)

    if (x2 - x1) * (y2 - y1) < config.MIN_ORGANOID_AREA:
        return None

    return image[y1:y2, x1:x2, :]


# ============================================================
# 2. 构建数据集索引
# ============================================================

def build_dataset():
    """
    读取 dataset_overview.csv，构建分类标签。

    CSV 列: org_id, img_id, Day, Clone, Imaging, org_size_px2, org_size_mikrometer2
    克隆列 "Clone" 即亚型标签: wt2D, TH2-7, A1A-1, B2A-2

    返回: df (带 label 列), clone_to_label
    """
    csv_path = config.CSV_PATH

    if not csv_path.exists():
        raise FileNotFoundError(
            f"找不到 {csv_path}。\n"
            f"请确保 data/dataset_overview.csv 存在。"
        )

    df = pd.read_csv(csv_path)

    # 直接用 CSV 中的 Clone 列作为 label
    clone_list = sorted(df["Clone"].unique())
    clone_to_label = {c: i for i, c in enumerate(clone_list)}
    df["label"] = df["Clone"].map(clone_to_label)

    print(f"\n📊 数据集概况:")
    print(f"   总图像数:  {len(df)}")
    print(f"   类器官数:  {df['org_id'].nunique()}")
    print(f"   克隆类型:  {clone_to_label}")
    print(f"   实验室:    {df['Imaging'].unique().tolist()}")
    print(f"   时间点:    {sorted(df['Day'].unique())}")
    print(f"   类别分布:")
    for clone, count in df["Clone"].value_counts().items():
        print(f"     {clone:8s}: {count} 张")

    return df, clone_to_label


# ============================================================
# 3. PyTorch Dataset
# ============================================================

class OrganoidDataset(Dataset):
    """从全视野图 + mask 自动裁剪类器官"""

    def __init__(self, df, image_dir, mask_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["img_id"]
        lab = row["Imaging"]

        # 找图像文件 (LabA → .jpg, LabB → .tif)
        for ext in (["jpg", "tif"] if lab == "LabA" else ["tif", "jpg"]):
            img_path = self.image_dir / f"{img_id}.{ext}"
            if img_path.exists():
                break

        # 读图像
        if img_path.suffix in [".tif", ".tiff"]:
            image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        # 用 mask 裁剪
        mask_path = self.mask_dir / f"{img_id}.npy"
        if mask_path.exists():
            mask = (np.load(str(mask_path)) > 0.5).astype(np.uint8)
            cropped = extract_organoid_roi(image, mask, margin=config.CROP_MARGIN)
            if cropped is not None:
                image = cropped

        # 统一尺寸
        image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row["label"], dtype=torch.long)
        return image, label


# ============================================================
# 4. 数据增强
# ============================================================

def get_train_transforms():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ============================================================
# 5. 数据划分 (按 org_id 分组防泄漏!)
# ============================================================

def create_folds(df, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    groups = df["org_id"].values
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df["label"], groups)):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        yield fold, train_df, val_df


def create_loaders(train_df, val_df, batch_size=None):
    batch_size = batch_size or config.BATCH_SIZE
    train_ds = OrganoidDataset(train_df, config.IMAGE_DIR, config.MASK_DIR, get_train_transforms())
    val_ds = OrganoidDataset(val_df, config.IMAGE_DIR, config.MASK_DIR, get_val_transforms())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=config.NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=config.NUM_WORKERS)
    return train_loader, val_loader


# ============================================================
# 6. 预处理模式 —— 直接读裁剪好的 JPEG，极快
# ============================================================

PROCESSED_CSV = config.DATA_DIR / "processed_dataset.csv"
PROCESSED_DIR = config.DATA_DIR / "processed"


def build_processed_dataset():
    """读取预处理好的数据集"""
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError(
            f"找不到 {PROCESSED_CSV}。请先运行: python preprocess.py"
        )
    df = pd.read_csv(PROCESSED_CSV)
    clone_list = sorted(df["Clone"].unique())
    clone_to_label = {c: i for i, c in enumerate(clone_list)}
    df["label"] = df["Clone"].map(clone_to_label)

    print(f"\n📊 预处理数据集:")
    print(f"   总图像: {len(df)}, 类器官: {df['org_id'].nunique()}")
    print(f"   克隆: {clone_to_label}")
    for clone, count in df["Clone"].value_counts().items():
        print(f"     {clone:8s}: {count} 张")
    return df, clone_to_label


class ProcessedDataset(Dataset):
    """直接读裁剪好的 JPEG，不读 TIFF / mask"""
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_dir = PROCESSED_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / f"{row['img_id']}.jpg"
        image = Image.open(str(img_path)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(row["label"], dtype=torch.long)


def create_processed_loaders(train_df, val_df, batch_size=None):
    batch_size = batch_size or config.BATCH_SIZE
    train_ds = ProcessedDataset(train_df, get_train_transforms())
    val_ds = ProcessedDataset(val_df, get_val_transforms())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=config.NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=config.NUM_WORKERS)
    return train_loader, val_loader


# ============================================================
# 7. 测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 测试数据加载管道")
    print("=" * 60)

    if config.CSV_PATH.exists():
        df, clone_map = build_dataset()

        for fold, train_df, val_df in create_folds(df, n_splits=5):
            print(f"\n🔀 Fold {fold + 1}:")
            print(f"   Train: {len(train_df)} 张 ({train_df['org_id'].nunique()} 个类器官)")
            print(f"   Val:   {len(val_df)} 张 ({val_df['org_id'].nunique()} 个类器官)")

            train_orgs = set(train_df["org_id"])
            val_orgs = set(val_df["org_id"])
            leakage = train_orgs & val_orgs
            print(f"   Leakage: {'✅ 无泄漏' if not leakage else f'❌ {len(leakage)} 个!'}")

            train_loader, val_loader = create_loaders(train_df, val_df)
            images, labels = next(iter(train_loader))
            print(f"   Batch: {images.shape}, Labels: {labels.tolist()}")
            break
    else:
        print(f"\n⚠️  找不到 {config.CSV_PATH}")
        print("请确保 data/dataset_overview.csv 存在")

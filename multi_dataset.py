"""
统一数据加载器 — 支持三种类器官数据集联合训练

数据来源:
  1. OrganoIDNet (脑) — 全视野+mask, 1407张, 4克隆
  2. CLORG (人小肠)  — 预裁剪224×224, 23063张, 4发育阶段
  3. OrgaQuant (小鼠小肠) — YOLO检测格式, 840张, 多类器官/图
"""
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ============================================================
# 通用变换
# ============================================================

def get_train_tf(img_size=224):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def get_val_tf(img_size=224):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ============================================================
# 数据集 1: CLORG — 人小肠，预裁剪224²，按文件夹分类
# ============================================================

class CLORGDataset(Dataset):
    """CLORG 人小肠类器官，已裁剪为 224×224，按类别文件夹存放"""

    def __init__(self, root_dir, split="train", transform=None):
        self.root = Path(root_dir) / f"{split}_folder"
        self.transform = transform
        self.samples = []

        # 遍历 class_0/1/2/3 文件夹
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            label = int(class_dir.name)
            for img_path in class_dir.glob("*.png"):
                self.samples.append((str(img_path), label))

        self.classes = ["cyst", "early_budding", "late_budding", "spheroid"]

        print(f"  CLORG {split}: {len(self)} images, classes: {self.classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


# ============================================================
# 数据集 2: OrganoIDNet — 脑类器官，全视野+mask
# ============================================================

class BrainOrganoidDataset(Dataset):
    """OrganoIDNet 脑类器官，全视野 + mask 自动裁剪"""

    def __init__(self, csv_path, image_dir, mask_dir,
                 use_mask=False, img_size=224, transform=None):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['Clone'] != 'Clone'].reset_index(drop=True)
        clones = sorted(self.df["Clone"].unique())
        self.c2i = {c: i for i, c in enumerate(clones)}
        self.df["label"] = self.df["Clone"].map(self.c2i)
        self.classes = clones

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.use_mask = use_mask
        self.img_size = img_size
        self.transform = transform

        print(f"  Brain: {len(self.df)} images, classes: {self.classes}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["img_id"]

        for ext in (["jpg","tif"] if row["Imaging"]=="LabA" else ["tif","jpg"]):
            p = self.image_dir / f"{img_id}.{ext}"
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED if p.suffix=='.tif' else cv2.IMREAD_COLOR)
                break
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 可选mask裁剪
        if self.use_mask:
            mask_path = self.mask_dir / f"{img_id}.npy"
            if mask_path.exists():
                mask = (np.load(str(mask_path)) > 0.5).astype(np.uint8)
                ys, xs = np.where(mask > 0)
                if len(ys) > 500:
                    h, w = img.shape[:2]
                    y1, y2 = max(0,ys.min()-20), min(h,ys.max()+20)
                    x1, x2 = max(0,xs.min()-20), min(w,xs.max()+20)
                    img = img[y1:y2, x1:x2, :]

        img = cv2.resize(img, (self.img_size, self.img_size))
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(row["label"], dtype=torch.long)


# ============================================================
# 数据集 3: OrgaQuant — 小鼠小肠，YOLO检测格式，多类器官/图
# ============================================================

class OrgaQuantDataset(Dataset):
    """
    OrgaQuant 小鼠小肠类器官 — YOLO 检测格式
    每张图含 5-14 个类器官，用 bbox 裁剪成单个分类样本
    """

    def __init__(self, root_dir, split="train", img_size=224, transform=None):
        self.root = Path(root_dir)
        self.img_dir = self.root / split / "images"
        self.lbl_dir = self.root / split / "labels"
        self.img_size = img_size
        self.transform = transform
        self.classes = ["organoid0_cyst", "organoid1_early", "organoid3_late", "spheroid"]
        self._build_index()

        print(f"  OrgaQuant {split}: {len(self.samples)} organoid crops")

    def _build_index(self):
        """用 YOLO bbox 裁剪出每个类器官"""
        self.samples = []
        for img_path in sorted(self.img_dir.glob("*.jp*")):
            # 读图
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]

            # 找对应的 label 文件
            lbl_path = self.lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            # YOLO 格式: class_id x_center y_center width height (归一化)
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])
                    # 转像素坐标
                    x1 = int((xc - bw/2) * w)
                    y1 = int((yc - bh/2) * w)
                    x2 = int((xc + bw/2) * w)
                    y2 = int((yc + bh/2) * w)
                    # 裁剪
                    crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2), :]
                    if crop.size == 0:
                        continue
                    self.samples.append((crop, cls_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        crop, label = self.samples[idx]
        crop = cv2.resize(crop, (self.img_size, self.img_size))
        crop = Image.fromarray(crop)

        if self.transform:
            crop = self.transform(crop)

        return crop, torch.tensor(label, dtype=torch.long)


# ============================================================
# 联合 DataLoader — 混合三种数据集的 batch
# ============================================================

class MixedBatchLoader:
    """
    每轮迭代从各数据集中等量采样，拼成一个混合 batch。
    每张图带 tissue_idx 标识来源。
    """

    def __init__(self, datasets, batch_size=16, shuffle=True):
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle

        per_ds = max(1, batch_size // len(datasets))
        self.loaders = [
            DataLoader(ds, batch_size=per_ds, shuffle=shuffle,
                      drop_last=True, num_workers=0)
            for ds in datasets
        ]
        self.iters = [iter(ld) for ld in self.loaders]
        self.batches_per_epoch = min(len(ld) for ld in self.loaders)

    def __iter__(self):
        self.iters = [iter(ld) for ld in self.loaders]
        return self

    def __next__(self):
        imgs, labels, tissues = [], [], []
        for i, it in enumerate(self.iters):
            try:
                img_b, lbl_b = next(it)
            except StopIteration:
                self.iters[i] = iter(self.loaders[i])
                img_b, lbl_b = next(self.iters[i])
            imgs.append(img_b)
            labels.append(lbl_b)
            tissues.append(torch.full((len(img_b),), i, dtype=torch.long))

        return (torch.cat(imgs), torch.cat(labels), torch.cat(tissues))

    def __len__(self):
        return self.batches_per_epoch


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    import pandas as pd
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

    print("=" * 60)
    print("三数据集联合加载测试")
    print("=" * 60)

    # 1. CLORG 人小肠
    print("\n[1] CLORG 人小肠...")
    clorg_train = CLORGDataset(
        "/Users/zhaozhihe/Desktop/2026surf/Final_Organoids_Dataset人小肠",
        split="train", transform=get_train_tf(224)
    )
    img, lbl = clorg_train[0]
    print(f"   Image: {img.shape}, Label: {lbl} ({clorg_train.classes[lbl]})")

    # 2. OrganoIDNet 脑
    print("\n[2] OrganoIDNet 脑...")
    brain_train = BrainOrganoidDataset(
        "/Users/zhaozhihe/Desktop/2026surf/organoid_project/data/dataset_overview.csv",
        "/Users/zhaozhihe/Desktop/2026surf/organoid_project/data/imgs",
        "/Users/zhaozhihe/Desktop/2026surf/organoid_project/data/labels",
        use_mask=False, img_size=224, transform=get_train_tf(224),
    )
    img, lbl = brain_train[0]
    print(f"   Image: {img.shape}, Label: {lbl} ({brain_train.classes[lbl]})")

    # 3. OrgaQuant 小鼠小肠
    print("\n[3] OrgaQuant 小鼠小肠...")
    orga_train = OrgaQuantDataset(
        "/Users/zhaozhihe/Desktop/2026surf/OrganoidDataset小鼠小肠",
        split="train", img_size=224, transform=get_train_tf(224),
    )
    img, lbl = orga_train[0]
    print(f"   Image: {img.shape}, Label: {lbl} ({orga_train.classes[lbl]})")

    # 4. 联合加载
    print(f"\n[4] 联合 MixedBatchLoader...")
    loader = MixedBatchLoader([clorg_train, brain_train, orga_train], batch_size=12)
    imgs, labels, tissues = next(iter(loader))
    print(f"   Batch: {imgs.shape}")
    print(f"   Labels: {labels.tolist()}")
    print(f"   Tissues: {tissues.tolist()} (0=CLORG, 1=Brain, 2=OrgaQuant)")

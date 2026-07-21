"""
三组图片联合训练

直接使用三组数据：
1. 人小肠 CLORG: train_folder / val_folder
2. 脑类器官 OrganoIDNet: data/dataset_overview.csv + imgs + labels，自动划分 train/val
3. 小鼠小肠 OrgaQuant: train / val

用法: python multi_train.py
"""
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

from multi_model import MultiOrganoidClassifier
from multi_dataset import CLORGDataset, BrainOrganoidDataset, OrgaQuantDataset, get_train_tf, get_val_tf

ROOT = Path(__file__).resolve().parent
CLORG = ROOT.parent / "Final_Organoids_Dataset人小肠"
BRAIN = ROOT / "data"
ORGA = ROOT.parent / "OrganoidDataset小鼠小肠"

TISSUES = {
    "human_intestine": {"num_classes": 4, "labels": ["cyst", "early", "late", "spheroid"]},
    "brain": {"num_classes": 4, "labels": ["wt2D", "A1A-1", "B2A-2", "TH2-7"]},
    "mouse_intestine": {"num_classes": 4, "labels": ["org0_cyst", "org1_early", "org3_late", "spheroid"]},
}

BACKBONE = "convnext_tiny"
IMG_SIZE = 224
BATCH_PER_DATASET = 4
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
SEED = 42
BRAIN_VAL_RATIO = 0.2
MAX_SAMPLES_PER_EPOCH = None  # None=每轮直接使用三组全部训练图片
MAX_VAL_SAMPLES = None
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_brain_dataset(dataset, val_ratio=0.2):
    labels = dataset.df["label"].to_numpy()
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=SEED,
        stratify=labels,
    )
    return Subset(dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist())


def limit_dataset(dataset, max_samples):
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    indices = random.sample(range(len(dataset)), max_samples)
    return Subset(dataset, indices)


def build_loaders():
    print("\n--- Loading three image datasets ---")
    train_h = CLORGDataset(CLORG, "train", get_train_tf(IMG_SIZE))
    val_h = CLORGDataset(CLORG, "val", get_val_tf(IMG_SIZE))

    brain_train_full = BrainOrganoidDataset(
        BRAIN / "dataset_overview.csv",
        BRAIN / "imgs",
        BRAIN / "labels",
        use_mask=False,
        img_size=IMG_SIZE,
        transform=get_train_tf(IMG_SIZE),
    )
    brain_val_full = BrainOrganoidDataset(
        BRAIN / "dataset_overview.csv",
        BRAIN / "imgs",
        BRAIN / "labels",
        use_mask=False,
        img_size=IMG_SIZE,
        transform=get_val_tf(IMG_SIZE),
    )
    train_b, _ = split_brain_dataset(brain_train_full, BRAIN_VAL_RATIO)
    _, val_b = split_brain_dataset(brain_val_full, BRAIN_VAL_RATIO)

    train_m = OrgaQuantDataset(ORGA, "train", IMG_SIZE, get_train_tf(IMG_SIZE))
    val_m = OrgaQuantDataset(ORGA, "val", IMG_SIZE, get_val_tf(IMG_SIZE))

    train_sets = {
        "human_intestine": limit_dataset(train_h, MAX_SAMPLES_PER_EPOCH),
        "brain": limit_dataset(train_b, MAX_SAMPLES_PER_EPOCH),
        "mouse_intestine": limit_dataset(train_m, MAX_SAMPLES_PER_EPOCH),
    }
    val_sets = {
        "human_intestine": limit_dataset(val_h, MAX_VAL_SAMPLES),
        "brain": limit_dataset(val_b, MAX_VAL_SAMPLES),
        "mouse_intestine": limit_dataset(val_m, MAX_VAL_SAMPLES),
    }

    train_loaders = {
        name: DataLoader(ds, BATCH_PER_DATASET, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)
        for name, ds in train_sets.items()
    }
    val_loaders = {
        name: DataLoader(ds, BATCH_PER_DATASET * 2, shuffle=False, num_workers=NUM_WORKERS)
        for name, ds in val_sets.items()
    }

    print("\nSamples used:")
    for name in TISSUES:
        print(f"  {name:16s} train={len(train_sets[name]):5d} val={len(val_sets[name]):5d}")
    return train_loaders, val_loaders


def train_epoch(model, loaders, optimizer, criterion, device):
    model.train()
    loss_sum, subtype_correct, tissue_correct, total = 0.0, 0, 0, 0
    iterators = {name: iter(loader) for name, loader in loaders.items()}
    steps = max(len(loader) for loader in loaders.values())

    for _ in range(steps):
        for tissue_name in TISSUES:
            try:
                imgs, labels = next(iterators[tissue_name])
            except StopIteration:
                iterators[tissue_name] = iter(loaders[tissue_name])
                imgs, labels = next(iterators[tissue_name])
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            tissue_logits, subtype_logits = model(imgs, tissue=tissue_name)
            tissue_target = torch.full(
                (len(imgs),),
                model.tissue_types.index(tissue_name),
                device=device,
                dtype=torch.long,
            )
            loss = criterion(tissue_logits, tissue_target) + criterion(subtype_logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_size = len(imgs)
            loss_sum += loss.item() * batch_size
            subtype_correct += (subtype_logits.argmax(1) == labels).sum().item()
            tissue_correct += (tissue_logits.argmax(1) == tissue_target).sum().item()
            total += batch_size

    return loss_sum / total, subtype_correct / total, tissue_correct / total


@torch.no_grad()
def validate(model, loaders, criterion, device):
    model.eval()
    stats = {}
    total_loss, total_subtype_correct, total_tissue_correct, total = 0.0, 0, 0, 0

    for tissue_name, loader in loaders.items():
        loss_sum, subtype_correct, tissue_correct, count = 0.0, 0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            tissue_logits, subtype_logits = model(imgs, tissue=tissue_name)
            tissue_target = torch.full(
                (len(imgs),),
                model.tissue_types.index(tissue_name),
                device=device,
                dtype=torch.long,
            )
            loss = criterion(tissue_logits, tissue_target) + criterion(subtype_logits, labels)

            batch_size = len(imgs)
            loss_sum += loss.item() * batch_size
            subtype_correct += (subtype_logits.argmax(1) == labels).sum().item()
            tissue_correct += (tissue_logits.argmax(1) == tissue_target).sum().item()
            count += batch_size

        stats[tissue_name] = {
            "loss": loss_sum / count,
            "subtype_acc": subtype_correct / count,
            "tissue_acc": tissue_correct / count,
            "count": count,
        }
        total_loss += loss_sum
        total_subtype_correct += subtype_correct
        total_tissue_correct += tissue_correct
        total += count

    stats["overall"] = {
        "loss": total_loss / total,
        "subtype_acc": total_subtype_correct / total,
        "tissue_acc": total_tissue_correct / total,
        "count": total,
    }
    return stats


def main():
    set_seed(SEED)
    device = torch.device(DEVICE)

    print("=" * 60)
    print(f"Three-Dataset Organoid Training | Device: {device}")
    print("=" * 60)

    train_loaders, val_loaders = build_loaders()
    model = MultiOrganoidClassifier(TISSUES, backbone_name=BACKBONE).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {BACKBONE} | Params: {total_params/1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    save_dir = ROOT / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "multi_organoid_three_datasets_best.pth"

    print("\n--- Training directly on three image groups ---")
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_sub_acc, train_tissue_acc = train_epoch(model, train_loaders, optimizer, criterion, device)
        val_stats = validate(model, val_loaders, criterion, device)
        scheduler.step()

        val_acc = val_stats["overall"]["subtype_acc"]
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_subtype_acc": best_acc,
                    "tissues": TISSUES,
                    "img_size": IMG_SIZE,
                },
                best_path,
            )

        detail = " | ".join(
            f"{name}:{val_stats[name]['subtype_acc']:.3f}"
            for name in TISSUES
        )
        print(
            f"E{epoch:03d} | "
            f"train_loss={train_loss:.4f} sub_acc={train_sub_acc:.3f} tissue_acc={train_tissue_acc:.3f} | "
            f"val_sub_acc={val_acc:.3f} val_tissue_acc={val_stats['overall']['tissue_acc']:.3f} | "
            f"{detail} | lr={scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:.0f}s"
        )

    final_path = save_dir / "multi_organoid_three_datasets_last.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\nDone in {(time.time()-start)/60:.1f} min")
    print(f"Best checkpoint: {best_path} ({best_acc:.4f})")
    print(f"Last weights:     {final_path}")


if __name__ == "__main__":
    main()

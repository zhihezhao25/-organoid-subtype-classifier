"""
Train an intestinal organoid morphology subtype classifier.

This script is the clean baseline for the main scientific task:
human + mouse intestinal organoid four-class morphology classification.

Example smoke test:
    python train_intestinal_subtype.py --epochs 1 --max-train-samples 200 --max-val-samples 100 --no-pretrained --run-name smoke

Example CPU/MPS training:
    python train_intestinal_subtype.py --backbone efficientnet_b0 --epochs 20 --image-size 224 --batch-size 32 --run-name intestine_b0
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from model import OrganoidSubtypeClassifier, count_parameters


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
MANIFEST_PATH = ROOT / "metadata" / "training_manifest.csv"

CLASSES = ["cyst", "early_budding", "late_budding", "spheroid"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_image_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_image(path: Path) -> Image.Image:
    if path.suffix.lower() in {".tif", ".tiff"}:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image)
    return Image.open(path).convert("RGB")


def crop_yolo(image: Image.Image, row: pd.Series, padding: float) -> Image.Image:
    width, height = image.size
    xc = float(row["bbox_x_center"])
    yc = float(row["bbox_y_center"])
    bw = float(row["bbox_width"])
    bh = float(row["bbox_height"])

    pad_w = bw * padding
    pad_h = bh * padding
    x1 = int((xc - bw / 2 - pad_w) * width)
    y1 = int((yc - bh / 2 - pad_h) * height)
    x2 = int((xc + bw / 2 + pad_w) * width)
    y2 = int((yc + bh / 2 + pad_h) * height)

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return image.crop((x1, y1, x2, y2))


class IntestinalSubtypeDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_size: int, train: bool, bbox_padding: float):
        self.df = dataframe.reset_index(drop=True)
        self.bbox_padding = bbox_padding
        self.transform = self._build_transform(image_size, train)

    @staticmethod
    def _build_transform(image_size: int, train: bool):
        ops = [transforms.Resize((image_size, image_size))]
        if train:
            ops.extend([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(25),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.02),
            ])
        ops.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image = read_image(resolve_image_path(row["image_path"]))
        if row["bbox_format"] == "yolo_normalized_xywh":
            image = crop_yolo(image, row, self.bbox_padding)
        label = int(row["label_id"])
        return self.transform(image), torch.tensor(label, dtype=torch.long)


def load_manifest(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(MANIFEST_PATH, dtype=str, keep_default_na=False)
    df = df[df["task"] == "intestinal_morphology_subtype"].copy()
    if args.domain != "both":
        organoid_type = "human_intestine" if args.domain == "human" else "mouse_intestine"
        df = df[df["organoid_type"] == organoid_type].copy()

    if args.use_test_as_val:
        val_split = "test"
    else:
        val_split = "val"
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == val_split].copy()

    if args.max_train_samples > 0 and len(train_df) > args.max_train_samples:
        train_idx, _ = train_test_split(
            train_df.index,
            train_size=args.max_train_samples,
            random_state=args.seed,
            stratify=train_df["label_id"],
        )
        train_df = train_df.loc[train_idx].copy()
    if args.max_val_samples > 0 and len(val_df) > args.max_val_samples:
        val_idx, _ = train_test_split(
            val_df.index,
            train_size=args.max_val_samples,
            random_state=args.seed,
            stratify=val_df["label_id"],
        )
        val_df = val_df.loc[val_idx].copy()
    return train_df, val_df


def make_sampler(labels: list[int]) -> WeightedRandomSampler:
    counts = np.bincount(np.asarray(labels), minlength=len(CLASSES)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    class_weights = 1.0 / counts
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def make_loss(labels: list[int], device: torch.device, no_class_balance: bool):
    if no_class_balance:
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    counts = np.bincount(np.asarray(labels), minlength=len(CLASSES)).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(CLASSES) * counts)
    weights = weights / weights.mean()
    print(f"Class weights: {weights.round(3).tolist()}")
    return nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=0.1,
    )


def build_backbone_model(args, device: torch.device):
    model = OrganoidSubtypeClassifier(
        num_classes=len(CLASSES),
        backbone_name=args.backbone,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
        freeze_stages=args.freeze_stages,
    ).to(device)
    total, trainable = count_parameters(model)
    print(f"Model: {args.backbone} | total={total/1e6:.1f}M trainable={trainable/1e6:.1f}M")
    return model


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses, y_true, y_pred = [], [], []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item() * images.size(0))
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(logits.argmax(1).detach().cpu().tolist())
    return {
        "loss": float(sum(losses) / len(loader.dataset)),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses, y_true, y_pred = [], [], []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        losses.append(loss.item() * images.size(0))
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())

    labels_range = list(range(len(CLASSES)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels_range,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "loss": float(sum(losses) / len(loader.dataset)),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels_range).tolist(),
        "classification_report": report,
    }


def save_confusion_matrix(matrix, output_path: Path) -> None:
    cm = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["both", "human", "mouse"], default="both")
    parser.add_argument("--backbone", choices=["efficientnet_b0", "efficientnet_b3", "resnet50", "convnext_tiny"], default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--freeze-stages", type=int, default=0)
    parser.add_argument("--bbox-padding", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--use-test-as-val", action="store_true")
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--no-class-balance", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--run-name", default="intestinal_subtype")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("=" * 72)
    print("Intestinal Organoid Morphology Subtype Training")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Args: {vars(args)}")

    train_df, val_df = load_manifest(args)
    print("\nDataset:")
    print(f"  train={len(train_df)} val={len(val_df)}")
    print("  train label counts:")
    print(train_df["label"].value_counts().reindex(CLASSES, fill_value=0).to_string())
    print("  val label counts:")
    print(val_df["label"].value_counts().reindex(CLASSES, fill_value=0).to_string())

    train_ds = IntestinalSubtypeDataset(train_df, args.image_size, train=True, bbox_padding=args.bbox_padding)
    val_ds = IntestinalSubtypeDataset(val_df, args.image_size, train=False, bbox_padding=args.bbox_padding)
    train_labels = train_df["label_id"].astype(int).tolist()

    sampler = make_sampler(train_labels) if args.weighted_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = build_backbone_model(args, device)
    criterion = make_loss(train_labels, device, args.no_class_balance)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    output_dir = ROOT / "logs" / args.run_name
    model_dir = ROOT / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / f"{args.run_name}_best.pth"
    last_path = model_dir / f"{args.run_name}_last.pth"
    metrics_path = output_dir / "metrics.json"
    cm_path = output_dir / "confusion_matrix.png"

    best_f1 = -1.0
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "classes": CLASSES,
                "args": vars(args),
                "best_macro_f1": best_f1,
            }, best_path)
            save_confusion_matrix(val_metrics["confusion_matrix"], cm_path)

        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump({"args": vars(args), "classes": CLASSES, "history": history}, handle, ensure_ascii=False, indent=2)

        print(
            f"E{epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.3f} "
            f"train_F1={train_metrics['macro_f1']:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} "
            f"val_balAcc={val_metrics['balanced_accuracy']:.3f} val_F1={val_metrics['macro_f1']:.3f} | "
            f"{time.time() - t0:.0f}s"
        )

    torch.save(model.state_dict(), last_path)
    print(f"\nDone in {(time.time() - start) / 60:.1f} min")
    print(f"Best model: {best_path} | macro-F1={best_f1:.4f}")
    print(f"Last model: {last_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Confusion matrix: {cm_path}")


if __name__ == "__main__":
    main()

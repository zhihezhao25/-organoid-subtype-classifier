"""
训练脚本 —— 类器官亚型分类

用法: python train.py
"""
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import config
from model import OrganoidSubtypeClassifier, count_parameters
from dataset import build_processed_dataset, create_folds, create_processed_loaders


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        running_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")

    try:
        n_classes = len(set(all_labels))
        if n_classes > 2:
            epoch_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
        else:
            all_probs_pos = [p[1] for p in all_probs]
            epoch_auc = roc_auc_score(all_labels, all_probs_pos)
    except ValueError:
        epoch_auc = float("nan")

    return epoch_loss, epoch_acc, epoch_f1, epoch_auc, all_labels, all_preds


def train_one_fold(fold, train_df, val_df, clone_to_label):
    num_classes = len(clone_to_label)
    device = torch.device(config.DEVICE)

    print(f"\n{'='*60}")
    print(f"🔀 Fold {fold + 1}")
    print(f"{'='*60}")
    print(f"  Train: {len(train_df)} images, {train_df['org_id'].nunique()} organoids")
    print(f"  Val:   {len(val_df)} images, {val_df['org_id'].nunique()} organoids")
    print(f"  Device: {device}")

    train_loader, val_loader = create_processed_loaders(train_df, val_df)

    model = OrganoidSubtypeClassifier(
        num_classes=num_classes,
        backbone_name=config.BACKBONE,
        dropout=config.DROPOUT,
        freeze_stages=config.FREEZE_STAGES,
    ).to(device)

    total, trainable = count_parameters(model)
    print(f"  Model: {config.BACKBONE} | {total/1e6:.1f}M params ({trainable/1e6:.1f}M trainable)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    best_path = None
    patience = 10
    patience_counter = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, val_auc, val_labels, val_preds = validate(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"  Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
            f"F1: {val_f1:.4f} AUC: {val_auc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_dir = Path(config.MODEL_DIR)
            save_dir.mkdir(parents=True, exist_ok=True)
            best_path = save_dir / f"best_model_fold{fold+1}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_auc": val_auc,
                "clone_to_label": clone_to_label,
                "config": {"backbone": config.BACKBONE, "image_size": config.IMAGE_SIZE, "num_classes": num_classes},
            }, best_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  ⏹ Early stopping at epoch {epoch}")
            break

    print(f"\n  ✅ Fold {fold+1} 最佳验证准确率: {best_val_acc:.4f}")
    return {"fold": fold + 1, "best_val_acc": best_val_acc, "best_val_f1": val_f1,
            "best_val_auc": val_auc, "model_path": best_path, "val_labels": val_labels, "val_preds": val_preds}


def main():
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)

    print(f"🚀 类器官亚型分类训练")
    print(f"   Device:    {device}")
    print(f"   Backbone:  {config.BACKBONE}")
    print(f"   Image:     {config.IMAGE_SIZE}×{config.IMAGE_SIZE}")
    print(f"   Batch:     {config.BATCH_SIZE}")
    print(f"   Epochs:    {config.NUM_EPOCHS}")

    df, clone_to_label = build_processed_dataset()

    results = []
    for fold, train_df, val_df in create_folds(df, n_splits=5):
        result = train_one_fold(fold, train_df, val_df, clone_to_label)
        results.append(result)

    print(f"\n{'='*60}")
    print(f"📊 汇总结果 (5-fold CV)")
    print(f"{'='*60}")

    accs = [r["best_val_acc"] for r in results]
    f1s = [r["best_val_f1"] for r in results]
    aucs = [r["best_val_auc"] for r in results]

    print(f"  Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  F1-macro: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  AUC:      {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  各 fold:  {[f'{a:.4f}' for a in accs]}")

    if np.mean(accs) >= 0.75:
        print(f"\n  🎉 达到 75% 目标! ({np.mean(accs):.1%})")
    else:
        print(f"\n  📈 距 75% 还差 {0.75 - np.mean(accs):.1%}，试试: 换 backbone / 增大分辨率 / 调 dropout")


if __name__ == "__main__":
    main()

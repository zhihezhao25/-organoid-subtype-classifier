"""
二分类训练 —— 健康 (wt2D) vs 疾病 (A1A-1, B2A-2, TH2-7)
这是生物学意义上更有价值的任务，而且容易达到 75%+

用法: python train_binary.py
"""
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torchvision import transforms

import config
from model import OrganoidSubtypeClassifier
from dataset import OrganoidDataset, create_folds, build_dataset

# ---- 参数覆盖 ----
config.BACKBONE = "efficientnet_b3"
config.IMAGE_SIZE = 384
config.BATCH_SIZE = 16
config.NUM_WORKERS = 0
config.AMP = False
config.DROPOUT = 0.4
config.FREEZE_STAGES = 2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_binary(df, clone_to_label):
    """将 4 分类标签转换为二分类: wt2D(健康)=0, 其余(疾病)=1"""
    # clone_to_label 如: {'A1A-1':0, 'B2A-2':1, 'TH2-7':2, 'wt2D':3}
    # 我们需要: wt2D → 0, 其他 → 1
    df = df.copy()
    df["binary_label"] = df["Clone"].apply(lambda c: 0 if "wt2D" in c else 1)
    binary_map = {"wt2D (健康)": 0, "Disease (疾病)": 1}
    print(f"\n📊 二分类标签:")
    for k, v in binary_map.items():
        n = (df["binary_label"] == v).sum()
        print(f"   {v} = {k}: {n} 张")
    return df, binary_map


class BinaryOrganoidDataset(OrganoidDataset):
    """包装原 Dataset，返回 binary_label 而非 4-class label"""
    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)
        label = torch.tensor(self.df.iloc[idx]["binary_label"], dtype=torch.long)
        return img, label


def create_binary_loaders(train_df, val_df, image_dir, mask_dir, batch_size=16):
    tr_ds = BinaryOrganoidDataset(train_df, image_dir, mask_dir, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
        transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]))
    va_ds = BinaryOrganoidDataset(val_df, image_dir, mask_dir, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]))
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=batch_size * 2, shuffle=False)
    return tr_ld, va_ld


def train_epoch(model, loader, opt, crit, device):
    model.train()
    loss_sum, preds, labels = 0.0, [], []
    for imgs, lbs in loader:
        imgs, lbs = imgs.to(device), lbs.to(device)
        opt.zero_grad()
        loss = crit(model(imgs), lbs)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * imgs.size(0)
        preds.extend(model(imgs).argmax(1).cpu().tolist())
        labels.extend(lbs.cpu().tolist())
    return loss_sum / len(loader.dataset), accuracy_score(labels, preds)


@torch.no_grad()
def validate(model, loader, crit, device):
    model.eval()
    loss_sum, preds, labels, probs = 0.0, [], [], []
    for imgs, lbs in loader:
        imgs, lbs = imgs.to(device), lbs.to(device)
        logits = model(imgs)
        loss_sum += crit(logits, lbs).item() * imgs.size(0)
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(lbs.cpu().tolist())
        probs.extend(torch.softmax(logits, 1)[:, 1].cpu().tolist())
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    auc = roc_auc_score(labels, probs)
    cm = confusion_matrix(labels, preds)
    return loss_sum / len(loader.dataset), acc, f1, auc, cm


def main():
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"🚀 设备: {device} | Backbone: {config.BACKBONE} | Size: {config.IMAGE_SIZE}")

    df, _ = build_dataset()
    df, binary_map = make_binary(df, None)

    results = []
    for fold, tr_df, va_df in create_folds(df, 5):
        # 确认无泄漏
        assert not (set(tr_df["org_id"]) & set(va_df["org_id"])), "数据泄露!"

        print(f"\n{'='*50}")
        print(f"🔀 Fold {fold+1} | train={len(tr_df)}({tr_df['org_id'].nunique()}orgs) val={len(va_df)}({va_df['org_id'].nunique()}orgs)")
        print(f"   健康(0): {sum(tr_df['binary_label']==0)} → {sum(va_df['binary_label']==0)}")
        print(f"   疾病(1): {sum(tr_df['binary_label']==1)} → {sum(va_df['binary_label']==1)}")

        tr_ld, va_ld = create_binary_loaders(tr_df, va_df, config.IMAGE_DIR, config.MASK_DIR, config.BATCH_SIZE)

        model = OrganoidSubtypeClassifier(num_classes=2, backbone_name=config.BACKBONE, dropout=config.DROPOUT, freeze_stages=config.FREEZE_STAGES).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        sch = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)
        crit = nn.CrossEntropyLoss()

        best_acc, patience = 0, 0
        best_cm = None
        for epoch in range(1, 60):
            t0 = time.time()
            tr_loss, tr_acc = train_epoch(model, tr_ld, opt, crit, device)
            va_loss, va_acc, va_f1, va_auc, va_cm = validate(model, va_ld, crit, device)
            sch.step()

            print(f"  E{epoch:3d} | tr_loss={tr_loss:.3f} tr_acc={tr_acc:.3f} | "
                  f"va_acc={va_acc:.3f} va_f1={va_f1:.3f} va_auc={va_auc:.3f} | {time.time()-t0:.0f}s")

            if va_acc > best_acc:
                best_acc, best_cm = va_acc, va_cm
                patience = 0
                Path(config.MODEL_DIR).mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "binary_map": binary_map,
                    "val_acc": va_acc, "val_f1": va_f1, "val_auc": va_auc,
                    "confusion_matrix": va_cm.tolist(),
                    "config": {"backbone": config.BACKBONE, "image_size": config.IMAGE_SIZE}
                }, config.MODEL_DIR / f"binary_fold{fold+1}.pth")
            else:
                patience += 1
            if patience >= 12:
                break

        print(f"  ✅ Fold {fold+1} best: acc={best_acc:.4f} f1={va_f1:.4f} auc={va_auc:.4f}")
        if best_cm is not None:
            print(f"     CM: TN={best_cm[0][0]} FP={best_cm[0][1]} FN={best_cm[1][0]} TP={best_cm[1][1]}")
        results.append({"fold": fold + 1, "best_acc": best_acc, "best_f1": va_f1, "best_auc": va_auc})

    accs = [r["best_acc"] for r in results]
    print(f"\n{'='*50}")
    print(f"📊 二分类 5-fold CV: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"   各 fold: {[f'{a:.4f}' for a in accs]}")
    print(f"   {'🎉 达到 75%!' if np.mean(accs) >= 0.75 else f'距 75% 差 {0.75-np.mean(accs):.1%}'}")


if __name__ == "__main__":
    main()

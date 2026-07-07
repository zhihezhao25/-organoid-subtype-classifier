"""
三数据集联合训练 — Multi-Organoid Classifier

共享 ConvNeXt 主干 + 组织类型分类头 + 各组织亚型分类头
"""
import os, random, time, numpy as np, torch, torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

from multi_model import MultiOrganoidClassifier, PRESET_CONFIGS
from multi_dataset import (
    CLORGDataset, BrainOrganoidDataset, OrgaQuantDataset,
    MixedBatchLoader, get_train_tf, get_val_tf,
)

# ===== 路径配置 =====
CLORG_DIR   = "/Users/zhaozhihe/Desktop/2026surf/Final_Organoids_Dataset人小肠"
BRAIN_CSV   = "/Users/zhaozhihe/Desktop/2026surf/organoid_project/data/dataset_overview.csv"
BRAIN_IMG   = "/Users/zhaozhihe/Desktop/2026surf/organoid_project/data/imgs"
BRAIN_MSK   = "/Users/zhaozhihe/Desktop/2026surf/organoid_project/data/labels"
ORGA_DIR    = "/Users/zhaozhihe/Desktop/2026surf/OrganoidDataset小鼠小肠"

# ===== 训练参数 =====
IMG_SIZE    = 224
BATCH_SIZE  = 18          # 每种组织 6 张
EPOCHS      = 40
LR          = 1e-4
WD          = 1e-4
SEED        = 42
DEVICE      = "mps" if torch.backends.mps.is_available() else "cpu"

# ===== 三组织配置 =====
TISSUE_CONFIGS = {
    "human_intestine": {
        "num_classes": 4,
        "labels": ["cyst", "early_budding", "late_budding", "spheroid"],
    },
    "brain": {
        "num_classes": 4,
        "labels": ["wt2D (健康)", "A1A-1", "B2A-2", "TH2-7"],
    },
    "mouse_intestine": {
        "num_classes": 4,
        "labels": ["organoid0_cyst", "organoid1_early", "organoid3_late", "spheroid"],
    },
}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def main():
    set_seed(SEED)
    device = torch.device(DEVICE)

    print("=" * 60)
    print("Multi-Organoid Joint Training")
    print("=" * 60)
    print(f"Device: {device} | Tissues: {list(TISSUE_CONFIGS.keys())}")

    # ---- 加载训练数据 ----
    print("\n--- 加载训练数据 ---")
    ds_human = CLORGDataset(CLORG_DIR, "train", get_train_tf(IMG_SIZE))
    ds_brain = BrainOrganoidDataset(BRAIN_CSV, BRAIN_IMG, BRAIN_MSK,
                                     use_mask=False, img_size=IMG_SIZE,
                                     transform=get_train_tf(IMG_SIZE))
    ds_mouse = OrgaQuantDataset(ORGA_DIR, "train", IMG_SIZE, get_train_tf(IMG_SIZE))

    train_loader = MixedBatchLoader([ds_human, ds_brain, ds_mouse], BATCH_SIZE)

    # ---- 加载验证数据 ----
    print("\n--- 加载验证数据 ---")
    ds_human_val = CLORGDataset(CLORG_DIR, "val", get_val_tf(IMG_SIZE))
    ds_brain_val = BrainOrganoidDataset(BRAIN_CSV, BRAIN_IMG, BRAIN_MSK,
                                         use_mask=False, img_size=IMG_SIZE,
                                         transform=get_val_tf(IMG_SIZE))
    ds_mouse_val = OrgaQuantDataset(ORGA_DIR, "val", IMG_SIZE, get_val_tf(IMG_SIZE))

    val_loaders = [
        torch.utils.data.DataLoader(ds, BATCH_SIZE*2, shuffle=False, num_workers=0)
        for ds in [ds_human_val, ds_brain_val, ds_mouse_val]
    ]
    val_names = ["human_intestine", "brain", "mouse_intestine"]

    print(f"\nTrain batches/epoch: {len(train_loader)}")
    print(f"Val sizes: human={len(ds_human_val)}, brain={len(ds_brain_val)}, mouse={len(ds_mouse_val)}")

    # ---- 模型 ----
    model = MultiOrganoidClassifier(TISSUE_CONFIGS, backbone_name="convnext_tiny").to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total/1e6:.1f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ---- 训练 ----
    best_avg_acc = 0
    patience = 8
    p_counter = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()

        # --- Train ---
        tr_loss_sum, tr_correct, tr_total = 0, 0, 0
        for imgs, s_labels, t_labels in train_loader:
            imgs, s_labels, t_labels = imgs.to(device), s_labels.to(device), t_labels.to(device)
            opt.zero_grad()

            # 前向：每个 batch 里的样本来自多个组织
            # 简化处理：以当前 batch 中多数组织的 tissue_idx 为准
            tissue_name = list(TISSUE_CONFIGS.keys())[t_labels[0].item()]
            tissue_logits, subtype_logits = model(imgs, tissue=tissue_name)

            # 计算两个 loss
            t_target = torch.full_like(t_labels, model.tissue_types.index(tissue_name))
            loss = crit(tissue_logits, t_target) + crit(subtype_logits, s_labels)
            loss.backward()
            opt.step()

            tr_loss_sum += loss.item()
            tr_correct += (subtype_logits.argmax(1) == s_labels).sum().item()
            tr_total += len(imgs)

        tr_acc = tr_correct / tr_total

        # --- Validate (分别评估三种组织) ---
        model.eval()
        val_accs = {}
        with torch.no_grad():
            for name, v_loader in zip(val_names, val_loaders):
                correct, total = 0, 0
                for imgs, s_labels in v_loader:
                    imgs, s_labels = imgs.to(device), s_labels.to(device)
                    _, subtype_logits = model(imgs, tissue=name)
                    correct += (subtype_logits.argmax(1) == s_labels).sum().item()
                    total += len(imgs)
                val_accs[name] = correct / total

        avg_acc = np.mean(list(val_accs.values()))
        sch.step()

        print(f"  E{epoch:3d} | tr_loss={tr_loss_sum:.2f} tr_acc={tr_acc:.3f} | "
              f"val_human={val_accs['human_intestine']:.3f} "
              f"val_brain={val_accs['brain']:.3f} "
              f"val_mouse={val_accs['mouse_intestine']:.3f} | "
              f"avg={avg_acc:.3f} | {time.time()-t0:.0f}s")

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            p_counter = 0
            Path("models").mkdir(exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_accs": val_accs,
                "tissue_configs": TISSUE_CONFIGS,
            }, "models/multi_organoid_best.pth")
        else:
            p_counter += 1

        if p_counter >= patience:
            print(f"  ⏹ Early stopping at epoch {epoch}")
            break

    print(f"\n{'='*60}")
    print(f"✅ 完成 | Best avg val acc: {best_avg_acc:.4f}")
    for k, v in val_accs.items():
        print(f"   {k}: {v:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

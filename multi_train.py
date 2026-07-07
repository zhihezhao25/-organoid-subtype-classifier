"""
三级训练 — 确保立刻出结果
"""
import os, time, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

from multi_model import MultiOrganoidClassifier
from multi_dataset import CLORGDataset, BrainOrganoidDataset, OrgaQuantDataset, get_train_tf, get_val_tf

CLORG = "/Users/zhaozhihe/Desktop/2026surf/Final_Organoids_Dataset人小肠"
BRAIN = "/Users/zhaozhihe/Desktop/2026surf/organoid_project/data"
ORGA  = "/Users/zhaozhihe/Desktop/2026surf/OrganoidDataset小鼠小肠"

TISSUES = {
    "human_intestine": {"num_classes": 4, "labels": ["cyst", "early", "late", "spheroid"]},
    "brain":           {"num_classes": 4, "labels": ["wt2D", "A1A-1", "B2A-2", "TH2-7"]},
    "mouse_intestine": {"num_classes": 4, "labels": ["org0_cyst", "org1_early", "org3_late", "spheroid"]},
}

IMG_SIZE = 224
BATCH = 12
EPOCHS = 30
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device(DEVICE)

    print("=" * 50)
    print(f"Multi-Organoid Training | Device: {device}")
    print("=" * 50)

    print("\n--- Loading datasets ---")
    train_h = CLORGDataset(CLORG, "train", get_train_tf(IMG_SIZE))
    train_b = BrainOrganoidDataset(f"{BRAIN}/dataset_overview.csv", f"{BRAIN}/imgs", f"{BRAIN}/labels",
                                    use_mask=False, img_size=IMG_SIZE, transform=get_train_tf(IMG_SIZE))
    train_m = OrgaQuantDataset(ORGA, "train", IMG_SIZE, get_train_tf(IMG_SIZE))

    n = min(2000, len(train_h)), min(400, len(train_b)), min(2000, len(train_m))
    print(f"Per epoch: human={n[0]}, brain={n[1]}, mouse={n[2]}")

    model = MultiOrganoidClassifier(TISSUES, backbone_name="convnext_tiny").to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total/1e6:.1f}M")
    print(f"Tissues: {model.tissue_types}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    print("\n--- Training ---")
    t_start = time.time()
    best = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()

        idx_h = random.sample(range(len(train_h)), n[0])
        idx_b = random.sample(range(len(train_b)), n[1])
        idx_m = random.sample(range(len(train_m)), n[2])

        ld_h = DataLoader(Subset(train_h, idx_h), BATCH//3, shuffle=True, drop_last=True, num_workers=0)
        ld_b = DataLoader(Subset(train_b, idx_b), BATCH//3, shuffle=True, drop_last=True, num_workers=0)
        ld_m = DataLoader(Subset(train_m, idx_m), BATCH//3, shuffle=True, drop_last=True, num_workers=0)

        loss_sum, correct, total_s = 0, 0, 0
        iters = [iter(ld_h), iter(ld_b), iter(ld_m)]

        for step in range(min(len(l) for l in [ld_h, ld_b, ld_m])):
            for t_idx, (it, name) in enumerate(zip(iters, list(TISSUES.keys()))):
                imgs, labels = next(it)
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad()

                tissue_logits, subtype_logits = model(imgs, tissue=name)
                t_target = torch.full((len(imgs),), model.tissue_types.index(name),
                                      device=device, dtype=torch.long)
                loss = crit(tissue_logits, t_target) + crit(subtype_logits, labels)
                loss.backward()
                opt.step()

                loss_sum += loss.item()
                correct += (subtype_logits.argmax(1) == labels).sum().item()
                total_s += len(imgs)

        acc = correct / total_s
        elapsed = time.time() - t0

        print(f"E{epoch:3d} | loss={loss_sum:.1f} acc={acc:.3f} | {elapsed:.0f}s")

        if acc > best:
            best = acc

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                imgs, labels = next(iter(ld_h))
                imgs, labels = imgs.to(device), labels.to(device)
                _, s_logits = model(imgs, tissue="human_intestine")
                v_acc = (s_logits.argmax(1) == labels).float().mean().item()
            print(f"  → val_human={v_acc:.3f}")

    print(f"\nDone in {(time.time()-t_start)/60:.0f} min | Best acc: {best:.4f}")
    torch.save(model.state_dict(), "models/multi_organoid.pth")
    print("Saved → models/multi_organoid.pth")


if __name__ == "__main__":
    main()

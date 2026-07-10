"""
推荐主方案：Attention-MIL + 三数据集多任务训练

为什么适合当前项目：
- 样本有限：使用 ImageNet 预训练 backbone，并先冻结 backbone。
- 三组数据标签语义不同：共享视觉特征，每个数据集独立 subtype head。
- 弱化切割步骤：每张图切成 patch，让 attention 自动关注有用区域。

快速测试：python train_mil_multitask.py --epochs 1 --max-samples 60 --max-val-samples 60 --no-pretrained
正式训练：python train_mil_multitask.py --epochs 30 --max-samples 0 --max-val-samples 0 --backbone convnext_tiny
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

from mil_model import MILMultiTaskOrganoidClassifier
from multi_dataset import CLORGDataset, BrainOrganoidDataset, OrgaQuantDataset

ROOT = Path(__file__).resolve().parent
CLORG = ROOT.parent / "Final_Organoids_Dataset人小肠"
BRAIN = ROOT / "data"
ORGA = ROOT.parent / "OrganoidDataset小鼠小肠"

TISSUES = {
    "human_intestine": {"num_classes": 4, "labels": ["cyst", "early_budding", "late_budding", "spheroid"]},
    "brain": {"num_classes": 4, "labels": ["A1A-1", "B2A-2", "TH2-7", "wt2D"]},
    "mouse_intestine": {"num_classes": 4, "labels": ["organoid0_cyst", "organoid1_early", "organoid3_late", "spheroid"]},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PatchBagDataset(Dataset):
    """把普通图像数据集包装成 patch bag。"""

    def __init__(self, base_dataset, image_size=384, patch_size=128, train=True):
        self.base_dataset = base_dataset
        self.image_size = image_size
        self.patch_size = patch_size
        self.train = train
        self.to_image = transforms.ToPILImage()
        self.transform = self._build_transform(train)

    def _build_transform(self, train):
        ops = [
            transforms.Resize((self.image_size, self.image_size)),
        ]
        if train:
            ops.extend([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ])
        ops.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return transforms.Compose(ops)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = (image * std + mean).clamp(0, 1)
            image = self.to_image(image)
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))

        image_tensor = self.transform(image)
        patches = self._make_patches(image_tensor)
        return patches, label

    def _make_patches(self, image_tensor):
        patches = image_tensor.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        patches = patches.permute(1, 2, 0, 3, 4).contiguous()
        return patches.view(-1, image_tensor.size(0), self.patch_size, self.patch_size)


def identity_transform(image):
    return image


def split_brain_indices(dataset, val_ratio, seed):
    labels = dataset.df["label"].to_numpy()
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=labels,
    )
    return train_idx.tolist(), val_idx.tolist()


def maybe_limit(dataset, max_samples):
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    indices = random.sample(range(len(dataset)), max_samples)
    return Subset(dataset, indices)


def build_datasets(args):
    train_h = CLORGDataset(CLORG, "train", transform=identity_transform)
    val_h = CLORGDataset(CLORG, "val", transform=identity_transform)

    brain_train_base = BrainOrganoidDataset(
        BRAIN / "dataset_overview.csv",
        BRAIN / "imgs",
        BRAIN / "labels",
        use_mask=False,
        img_size=args.image_size,
        transform=identity_transform,
    )
    brain_val_base = BrainOrganoidDataset(
        BRAIN / "dataset_overview.csv",
        BRAIN / "imgs",
        BRAIN / "labels",
        use_mask=False,
        img_size=args.image_size,
        transform=identity_transform,
    )
    train_idx, val_idx = split_brain_indices(brain_train_base, args.brain_val_ratio, args.seed)
    train_b = Subset(brain_train_base, train_idx)
    val_b = Subset(brain_val_base, val_idx)

    train_m = OrgaQuantDataset(ORGA, "train", args.image_size, transform=identity_transform)
    val_m = OrgaQuantDataset(ORGA, "val", args.image_size, transform=identity_transform)

    train_sets = {
        "human_intestine": maybe_limit(PatchBagDataset(train_h, args.image_size, args.patch_size, train=True), args.max_samples),
        "brain": maybe_limit(PatchBagDataset(train_b, args.image_size, args.patch_size, train=True), args.max_samples),
        "mouse_intestine": maybe_limit(PatchBagDataset(train_m, args.image_size, args.patch_size, train=True), args.max_samples),
    }
    val_sets = {
        "human_intestine": maybe_limit(PatchBagDataset(val_h, args.image_size, args.patch_size, train=False), args.max_val_samples),
        "brain": maybe_limit(PatchBagDataset(val_b, args.image_size, args.patch_size, train=False), args.max_val_samples),
        "mouse_intestine": maybe_limit(PatchBagDataset(val_m, args.image_size, args.patch_size, train=False), args.max_val_samples),
    }
    return train_sets, val_sets


def build_loaders(datasets, batch_size, shuffle, workers):
    return {
        name: DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=shuffle)
        for name, dataset in datasets.items()
    }


def soft_cross_entropy(logits, targets):
    log_probs = F.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


def train_epoch(model, loaders, optimizer, criterion, device, args):
    model.train()
    totals = {"loss": 0.0, "n": 0, "tissue_correct": 0, "subtype_correct": 0}
    iterators = {name: iter(loader) for name, loader in loaders.items()}
    steps = max(len(loader) for loader in loaders.values())

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        batches = 0

        for tissue_name, loader in loaders.items():
            try:
                patches, labels = next(iterators[tissue_name])
            except StopIteration:
                iterators[tissue_name] = iter(loader)
                patches, labels = next(iterators[tissue_name])

            patches = patches.to(device)
            labels = labels.to(device)
            tissue_index = model.tissue_types.index(tissue_name)
            tissue_targets = torch.full((patches.size(0),), tissue_index, device=device, dtype=torch.long)

            tissue_logits, subtype_logits, _ = model(patches, tissue=tissue_name)
            loss = criterion(subtype_logits, labels) + args.tissue_loss_weight * criterion(tissue_logits, tissue_targets)
            accumulated_loss = accumulated_loss + loss
            batches += 1

            batch_size = patches.size(0)
            totals["loss"] += loss.item() * batch_size
            totals["n"] += batch_size
            totals["subtype_correct"] += (subtype_logits.argmax(1) == labels).sum().item()
            totals["tissue_correct"] += (tissue_logits.argmax(1) == tissue_targets).sum().item()

        (accumulated_loss / batches).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    return {
        "loss": totals["loss"] / totals["n"],
        "subtype_acc": totals["subtype_correct"] / totals["n"],
        "tissue_acc": totals["tissue_correct"] / totals["n"],
    }


@torch.no_grad()
def validate(model, loaders, criterion, device, args):
    model.eval()
    results = {}
    all_true, all_pred = [], []
    total_loss, total_n, tissue_correct = 0.0, 0, 0

    for tissue_name, loader in loaders.items():
        y_true, y_pred = [], []
        loss_sum, count, tissue_ok = 0.0, 0, 0
        tissue_index = model.tissue_types.index(tissue_name)

        for patches, labels in loader:
            patches = patches.to(device)
            labels = labels.to(device)
            tissue_targets = torch.full((patches.size(0),), tissue_index, device=device, dtype=torch.long)

            tissue_logits, subtype_logits, _ = model(patches, tissue=tissue_name)
            loss = criterion(subtype_logits, labels) + args.tissue_loss_weight * criterion(tissue_logits, tissue_targets)
            preds = subtype_logits.argmax(1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            batch_size = patches.size(0)
            loss_sum += loss.item() * batch_size
            count += batch_size
            tissue_ok += (tissue_logits.argmax(1) == tissue_targets).sum().item()

        labels_count = list(range(TISSUES[tissue_name]["num_classes"]))
        cm = confusion_matrix(y_true, y_pred, labels=labels_count)
        results[tissue_name] = {
            "loss": loss_sum / count,
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "tissue_accuracy": tissue_ok / count,
            "confusion_matrix": cm.tolist(),
        }

        all_true.extend((tissue_name, y) for y in y_true)
        all_pred.extend((tissue_name, y) for y in y_pred)
        total_loss += loss_sum
        total_n += count
        tissue_correct += tissue_ok

    macro_f1 = float(np.mean([v["macro_f1"] for v in results.values()]))
    balanced_acc = float(np.mean([v["balanced_accuracy"] for v in results.values()]))
    accuracy = float(np.mean([v["accuracy"] for v in results.values()]))
    results["overall"] = {
        "loss": total_loss / total_n,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "tissue_accuracy": tissue_correct / total_n,
    }
    return results


def save_metrics(metrics, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="mobilenet_v3_small", choices=["mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "convnext_tiny"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument("--lr-backbone", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--tissue-loss-weight", type=float, default=0.3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means full training set")
    parser.add_argument("--max-val-samples", type=int, default=0, help="0 means full validation set")
    parser.add_argument("--brain-val-ratio", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if args.image_size % args.patch_size != 0:
        raise ValueError("image-size must be divisible by patch-size")

    print("=" * 72)
    print("Attention-MIL Multi-Task Organoid Subtype Training")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone} | pretrained={not args.no_pretrained}")
    print(f"Image: {args.image_size} | Patch: {args.patch_size} | patches/image={(args.image_size // args.patch_size) ** 2}")

    train_sets, val_sets = build_datasets(args)
    print("\nDataset sizes:")
    for name in TISSUES:
        print(f"  {name:16s} train={len(train_sets[name]):5d} val={len(val_sets[name]):5d}")

    train_loaders = build_loaders(train_sets, args.batch_size, True, args.workers)
    val_loaders = build_loaders(val_sets, args.batch_size * 2, False, args.workers)

    model = MILMultiTaskOrganoidClassifier(
        TISSUES,
        backbone_name=args.backbone,
        pretrained=not args.no_pretrained,
        dropout=0.3,
    ).to(device)
    model.freeze_backbone()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_head,
        weight_decay=args.weight_decay,
    )

    save_dir = ROOT / "models"
    log_dir = ROOT / "logs"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "mil_multitask_best.pth"
    metrics_path = log_dir / "mil_multitask_metrics.json"

    best_score = -1.0
    history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW([
                {"params": model.backbone.parameters(), "lr": args.lr_backbone},
                {"params": model.attention_pool.parameters(), "lr": args.lr_head},
                {"params": model.tissue_head.parameters(), "lr": args.lr_head},
                {"params": model.subtype_heads.parameters(), "lr": args.lr_head},
            ], weight_decay=args.weight_decay)
            print("\nBackbone unfrozen for fine-tuning.")

        t0 = time.time()
        train_metrics = train_epoch(model, train_loaders, optimizer, criterion, device, args)
        val_metrics = validate(model, val_loaders, criterion, device, args)
        score = val_metrics["overall"]["macro_f1"]

        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        save_metrics({"args": vars(args), "history": history}, metrics_path)

        if score > best_score:
            best_score = score
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "tissues": TISSUES,
                "best_macro_f1": best_score,
            }, best_path)

        per_dataset = " | ".join(
            f"{name}:F1={val_metrics[name]['macro_f1']:.3f},BalAcc={val_metrics[name]['balanced_accuracy']:.3f}"
            for name in TISSUES
        )
        print(
            f"E{epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['subtype_acc']:.3f} | "
            f"val_F1={score:.3f} val_BalAcc={val_metrics['overall']['balanced_accuracy']:.3f} "
            f"tissue_acc={val_metrics['overall']['tissue_accuracy']:.3f} | "
            f"{per_dataset} | {time.time() - t0:.0f}s"
        )

    last_path = save_dir / "mil_multitask_last.pth"
    torch.save(model.state_dict(), last_path)
    print(f"\nDone in {(time.time() - start) / 60:.1f} min")
    print(f"Best model: {best_path} | macro-F1={best_score:.4f}")
    print(f"Last model: {last_path}")
    print(f"Metrics:    {metrics_path}")


if __name__ == "__main__":
    main()

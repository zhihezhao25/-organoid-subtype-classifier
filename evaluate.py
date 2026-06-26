"""
评估与可视化脚本

用法:
    python evaluate.py --model_path models/best_model_fold1.pth --data_dir data/organoidnet
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # 无头模式
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from model import OrganoidSubtypeClassifier
from dataset import (
    OrganoidDataset,
    get_val_transforms,
    build_dataset,
)


@torch.no_grad()
def extract_features(model, loader, device):
    """提取所有样本的特征向量（用于 t-SNE）"""
    model.eval()
    features_list, labels_list = [], []

    for images, labels in tqdm(loader, desc="Extracting features"):
        images = images.to(device)
        features = model.forward_features(images)
        features_list.append(features.cpu().numpy())
        labels_list.append(labels.numpy())

    return np.vstack(features_list), np.hstack(labels_list)


@torch.no_grad()
def predict_all(model, loader, device):
    """对所有样本预测"""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="Predicting"):
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", ax=ax, colorbar=True, values_format=".2f")
    ax.set_title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 混淆矩阵已保存: {save_path}")


def plot_tsne(features, labels, class_names, save_path):
    """t-SNE 特征可视化"""
    print("  Computing t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=config.SEED)
    embeddings = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(class_names)))

    for i, name in enumerate(class_names):
        mask = labels == i
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=[colors[i]],
            label=name,
            alpha=0.6,
            s=20,
        )

    ax.legend()
    ax.set_title("t-SNE: Organoid Subtype Features")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  🎨 t-SNE 图已保存: {save_path}")


def plot_training_curves(metrics_history, save_path):
    """如果有训练历史记录，绘制曲线"""
    # Placeholder — 可扩展
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="训练好的模型 .pth")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="logs")
    args = parser.parse_args()

    device = torch.device(config.DEVICE)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔍 评估模型: {args.model_path}")

    # ---- Load Model ----
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    clone_to_label = checkpoint["clone_to_label"]
    class_names = list(clone_to_label.keys())
    num_classes = len(class_names)

    model = OrganoidSubtypeClassifier(
        num_classes=num_classes,
        backbone_name=checkpoint["config"]["backbone"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"   Backbone: {checkpoint['config']['backbone']}")
    print(f"   Classes:  {clone_to_label}")
    print(f"   Val Acc:  {checkpoint['val_acc']:.4f}")

    # ---- Load Data ----
    df, _ = build_dataset()

    # 用最后 20% 的 organoid 作为测试集
    all_org_ids = sorted(df["org_id"].unique())
    test_orgs = set(all_org_ids[-len(all_org_ids) // 5:])
    test_df = df[df["org_id"].isin(test_orgs)].reset_index(drop=True)
    print(f"\n   测试集: {len(test_df)} 张图, {len(test_orgs)} 个类器官")

    # DataLoader
    test_ds = OrganoidDataset(
        test_df, config.IMAGE_DIR, config.MASK_DIR, transform=get_val_transforms()
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    # ---- Predict ----
    y_true, y_pred, y_probs = predict_all(model, test_loader, device)

    # ---- Report ----
    print(f"\n{'='*50}")
    print("📊 测试集结果")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # ---- Visualize ----
    plot_confusion_matrix(y_true, y_pred, class_names, output_dir / "confusion_matrix.png")
    plot_tsne(
        *extract_features(model, test_loader, device),
        class_names,
        output_dir / "tsne_plot.png",
    )


if __name__ == "__main__":
    main()

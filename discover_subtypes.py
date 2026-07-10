"""
未知类器官亚型发现工具

用途：
- 当不知道类器官有几种亚型、亚型叫什么名字时，先不要做普通分类。
- 本脚本会提取每张图片的 embedding，然后自动降维和聚类。
- 输出每个 cluster 的代表图片，供人工观察和命名。

推荐运行：
python discover_subtypes.py --dataset all --max-samples 1000

如果安装了 umap-learn / hdbscan，会自动使用 UMAP + HDBSCAN；
否则退化为 PCA + DBSCAN，不影响基本使用。
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tv_models
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
CLORG = ROOT.parent / "Final_Organoids_Dataset人小肠"
BRAIN = ROOT / "data"
ORGA = ROOT.parent / "OrganoidDataset小鼠小肠"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def try_import_umap():
    try:
        import umap
        return umap
    except Exception:
        return None


def try_import_hdbscan():
    try:
        import hdbscan
        return hdbscan
    except Exception:
        return None


class DiscoveryImageDataset(Dataset):
    """读取一批图片，保留路径和可选已有标签，方便后续解释 cluster。"""

    def __init__(self, records, image_size=224):
        self.records = records
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = read_image(record["path"])
        return self.transform(image), idx


def read_image(path):
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(path)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image)
    return Image.open(path).convert("RGB")


def build_backbone(name, pretrained=True):
    if name == "mobilenet_v3_small":
        weights = tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.mobilenet_v3_small(weights=weights)
        return model.features, 576
    if name == "mobilenet_v3_large":
        weights = tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = tv_models.mobilenet_v3_large(weights=weights)
        return model.features, 960
    if name == "efficientnet_b0":
        weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.efficientnet_b0(weights=weights)
        return model.features, 1280
    if name == "convnext_tiny":
        weights = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.convnext_tiny(weights=weights)
        return model.features, 768
    raise ValueError(f"Unknown backbone: {name}")


class FeatureExtractor(nn.Module):
    def __init__(self, backbone_name, pretrained=True):
        super().__init__()
        self.backbone, self.feature_dim = build_backbone(backbone_name, pretrained)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images):
        features = self.backbone(images)
        return self.pool(features).flatten(1)


def collect_clorg_records(root):
    records = []
    for split in ["train", "val", "test"]:
        split_dir = Path(root) / f"{split}_folder"
        if not split_dir.exists():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() in IMAGE_EXTS:
                    records.append({
                        "path": str(image_path),
                        "dataset": "human_intestine",
                        "split": split,
                        "known_label": class_dir.name,
                    })
    return records


def collect_brain_records(root):
    root = Path(root)
    csv_path = root / "dataset_overview.csv"
    image_dir = root / "imgs"
    if not csv_path.exists():
        return []

    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        img_id = row["img_id"]
        exts = ["jpg", "tif"] if row.get("Imaging", "") == "LabA" else ["tif", "jpg"]
        image_path = None
        for ext in exts:
            candidate = image_dir / f"{img_id}.{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            continue
        records.append({
            "path": str(image_path),
            "dataset": "brain",
            "split": "all",
            "known_label": str(row.get("Clone", "unknown")),
        })
    return records


def collect_orga_records(root):
    records = []
    root = Path(root)
    for split in ["train", "val"]:
        image_dir = root / split / "images"
        if not image_dir.exists():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTS:
                records.append({
                    "path": str(image_path),
                    "dataset": "mouse_intestine",
                    "split": split,
                    "known_label": "image_level_unknown",
                })
    return records


def collect_image_folder_records(folder):
    folder = Path(folder)
    records = []
    for image_path in sorted(folder.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
            records.append({
                "path": str(image_path),
                "dataset": folder.name,
                "split": "unknown",
                "known_label": image_path.parent.name,
            })
    return records


def collect_records(args):
    if args.input_dir:
        records = collect_image_folder_records(args.input_dir)
    else:
        records = []
        if args.dataset in {"all", "human_intestine"}:
            records.extend(collect_clorg_records(CLORG))
        if args.dataset in {"all", "brain"}:
            records.extend(collect_brain_records(BRAIN))
        if args.dataset in {"all", "mouse_intestine"}:
            records.extend(collect_orga_records(ORGA))

    if args.max_samples > 0 and len(records) > args.max_samples:
        records = random.sample(records, args.max_samples)
    return records


@torch.no_grad()
def extract_features(records, args, device):
    dataset = DiscoveryImageDataset(records, args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = FeatureExtractor(args.backbone, pretrained=not args.no_pretrained).to(device)
    model.eval()

    features = np.zeros((len(records), model.feature_dim), dtype=np.float32)
    for images, indices in loader:
        images = images.to(device)
        batch_features = model(images).cpu().numpy()
        features[indices.numpy()] = batch_features
    return features


def reduce_dimensions(features, args):
    scaled = StandardScaler().fit_transform(features)
    if args.reducer == "umap" or args.reducer == "auto":
        umap = try_import_umap()
        if umap is not None:
            reducer = umap.UMAP(n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist, random_state=args.seed)
            return reducer.fit_transform(scaled), "umap"

    pca = PCA(n_components=2, random_state=args.seed)
    return pca.fit_transform(scaled), "pca"


def cluster_points(features, reduced, args):
    if args.clusterer == "hdbscan" or args.clusterer == "auto":
        hdbscan = try_import_hdbscan()
        if hdbscan is not None:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size, min_samples=args.min_samples)
            return clusterer.fit_predict(features), "hdbscan"

    if args.clusterer == "kmeans":
        labels = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init="auto").fit_predict(features)
        return labels, "kmeans"

    labels = DBSCAN(eps=args.dbscan_eps, min_samples=args.min_samples).fit_predict(reduced)
    return labels, "dbscan"


def choose_representatives(features, labels, records, output_dir, per_cluster):
    output_dir = Path(output_dir)
    cluster_dirs = {}
    for label in sorted(set(labels)):
        name = "noise" if label == -1 else f"cluster_{label}"
        cluster_dir = output_dir / name
        cluster_dir.mkdir(parents=True, exist_ok=True)
        cluster_dirs[label] = cluster_dir

    for label in sorted(set(labels)):
        indices = np.where(labels == label)[0]
        if len(indices) == 0:
            continue
        cluster_features = features[indices]
        center = cluster_features.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(cluster_features - center, axis=1)
        chosen = indices[np.argsort(distances)[:per_cluster]]

        for rank, idx in enumerate(chosen, start=1):
            src = Path(records[idx]["path"])
            suffix = src.suffix.lower() if src.suffix else ".png"
            dst = cluster_dirs[label] / f"{rank:02d}_{records[idx]['dataset']}_{records[idx]['known_label']}{suffix}"
            try:
                shutil.copy2(src, dst)
            except Exception:
                image = read_image(src)
                image.save(dst.with_suffix(".png"))


def save_plot(reduced, labels, records, output_path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plt.figure(figsize=(10, 8))
    labels_array = np.asarray(labels)
    for label in sorted(set(labels)):
        mask = labels_array == label
        name = "noise" if label == -1 else f"cluster {label}"
        plt.scatter(reduced[mask, 0], reduced[mask, 1], s=12, alpha=0.75, label=name)
    plt.legend(markerscale=2, fontsize=8)
    plt.title("Organoid subtype discovery clusters")
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def summarize_clusters(records, labels):
    df = pd.DataFrame(records)
    df["cluster"] = labels
    summary = []
    for cluster, group in df.groupby("cluster"):
        summary.append({
            "cluster": int(cluster),
            "name": "noise" if cluster == -1 else f"cluster_{cluster}",
            "count": int(len(group)),
            "datasets": group["dataset"].value_counts().to_dict(),
            "known_labels": group["known_label"].value_counts().head(10).to_dict(),
        })
    return df, summary


def compute_quality(reduced, labels):
    valid_mask = labels != -1
    unique = set(labels[valid_mask])
    if len(unique) < 2 or valid_mask.sum() < 3:
        return {"silhouette": None}
    try:
        return {"silhouette": float(silhouette_score(reduced[valid_mask], labels[valid_mask]))}
    except Exception:
        return {"silhouette": None}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["all", "human_intestine", "brain", "mouse_intestine"])
    parser.add_argument("--input-dir", default=None, help="可选：对任意图片文件夹做未知亚型发现")
    parser.add_argument("--output-dir", default=str(ROOT / "logs" / "discovery"))
    parser.add_argument("--backbone", default="mobilenet_v3_small", choices=["mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "convnext_tiny"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=1000, help="0 means all images")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--reducer", default="auto", choices=["auto", "umap", "pca"])
    parser.add_argument("--clusterer", default="auto", choices=["auto", "hdbscan", "dbscan", "kmeans"])
    parser.add_argument("--min-cluster-size", type=int, default=20)
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--dbscan-eps", type=float, default=0.8)
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--umap-neighbors", type=int, default=20)
    parser.add_argument("--umap-min-dist", type=float, default=0.05)
    parser.add_argument("--examples-per-cluster", type=int, default=24)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    print("=" * 72)
    print("Organoid Unknown Subtype Discovery")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset if not args.input_dir else args.input_dir}")
    print(f"Backbone: {args.backbone} | pretrained={not args.no_pretrained}")

    records = collect_records(args)
    if not records:
        raise RuntimeError("No images found for discovery.")
    print(f"Images: {len(records)}")

    features = extract_features(records, args, device)
    reduced, reducer_name = reduce_dimensions(features, args)
    labels, clusterer_name = cluster_points(features, reduced, args)

    assignments, summary = summarize_clusters(records, labels)
    quality = compute_quality(reduced, labels)
    assignments["x"] = reduced[:, 0]
    assignments["y"] = reduced[:, 1]

    np.save(output_dir / "embeddings.npy", features)
    np.save(output_dir / "reduced_2d.npy", reduced)
    assignments.to_csv(output_dir / "cluster_assignments.csv", index=False)
    save_plot(reduced, labels, records, output_dir / "clusters_2d.png")
    choose_representatives(features, labels, records, output_dir / "cluster_examples", args.examples_per_cluster)

    metadata = {
        "args": vars(args),
        "device": str(device),
        "reducer": reducer_name,
        "clusterer": clusterer_name,
        "num_images": len(records),
        "num_clusters_excluding_noise": len([x for x in set(labels) if x != -1]),
        "noise_count": int((labels == -1).sum()),
        "quality": quality,
        "summary": summary,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"Reducer: {reducer_name} | Clusterer: {clusterer_name}")
    print(f"Clusters: {metadata['num_clusters_excluding_noise']} | Noise: {metadata['noise_count']}")
    print(f"Silhouette: {quality['silhouette']}")
    for item in summary:
        print(f"  {item['name']:12s} n={item['count']:4d} datasets={item['datasets']}")
    print(f"\nSaved discovery results to: {output_dir}")
    print("Check cluster_examples/ and clusters_2d.png, then manually name meaningful clusters.")


if __name__ == "__main__":
    main()

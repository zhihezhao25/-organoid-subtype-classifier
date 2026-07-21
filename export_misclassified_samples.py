"""
Export misclassified intestinal organoid samples for visual error analysis.

Example:
    python export_misclassified_samples.py \
      --checkpoint models/intestine_subtype_b0_best.pth \
      --run-name intestine_subtype_b0 \
      --output-dir logs/intestine_subtype_b0/error_examples

The script writes:
- predictions.csv: one validation row per sample with true/predicted labels.
- misclassified.csv: only incorrect samples.
- contact sheets for selected error pairs, such as early_budding -> late_budding.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from model import OrganoidSubtypeClassifier
from train_intestinal_subtype import (
    CLASSES,
    IntestinalSubtypeDataset,
    crop_yolo,
    load_manifest,
    read_image,
    resolve_image_path,
)


ERROR_PAIRS = [
    ("early_budding", "late_budding"),
    ("late_budding", "early_budding"),
    ("spheroid", "cyst"),
    ("cyst", "spheroid"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to *_best.pth checkpoint.")
    parser.add_argument("--run-name", default="intestine_subtype_b0")
    parser.add_argument("--domain", choices=["both", "human", "mouse"], default="both")
    parser.add_argument("--output-dir", default="logs/error_examples")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--max-per-pair", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--bbox-padding", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> tuple[dict, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Re-run the baseline or point --checkpoint to an existing file."
        )
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint.get("args", {})
    return checkpoint, {}


def build_args(base_args: argparse.Namespace, checkpoint_args: dict) -> argparse.Namespace:
    args = argparse.Namespace(**vars(base_args))
    args.backbone = checkpoint_args.get("backbone", "efficientnet_b0")
    args.dropout = float(checkpoint_args.get("dropout", 0.3))
    args.freeze_stages = int(checkpoint_args.get("freeze_stages", 0))
    args.no_pretrained = True
    args.max_train_samples = 0
    args.max_val_samples = 0
    args.use_test_as_val = base_args.split == "test"
    args.weighted_sampler = bool(checkpoint_args.get("weighted_sampler", False))
    args.no_class_balance = bool(checkpoint_args.get("no_class_balance", False))
    args.image_size = int(base_args.image_size or checkpoint_args.get("image_size", 224))
    args.bbox_padding = float(base_args.bbox_padding or checkpoint_args.get("bbox_padding", 0.15))
    return args


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[float]]:
    model.eval()
    preds = []
    confidences = []
    for images, _labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        preds.extend(pred.cpu().tolist())
        confidences.extend(conf.cpu().tolist())
    return preds, confidences


def save_crop(row: pd.Series, output_path: Path, bbox_padding: float) -> None:
    image = read_image(resolve_image_path(row["image_path"]))
    if row["bbox_format"] == "yolo_normalized_xywh":
        image = crop_yolo(image, row, bbox_padding)
    image.thumbnail((224, 224))
    canvas = Image.new("RGB", (224, 224), "white")
    x = (224 - image.width) // 2
    y = (224 - image.height) // 2
    canvas.paste(image.convert("RGB"), (x, y))
    canvas.save(output_path, quality=95)


def make_contact_sheet(image_paths: list[Path], title: str, output_path: Path) -> None:
    if not image_paths:
        return
    thumb = 160
    label_h = 26
    cols = min(5, len(image_paths))
    rows = int(np.ceil(len(image_paths) / cols))
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + label_h) + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 6), title, fill="black", font=font)
    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((thumb, thumb))
        col = i % cols
        row = i // cols
        x0 = col * thumb
        y0 = label_h + row * (thumb + label_h)
        x = x0 + (thumb - image.width) // 2
        y = y0 + (thumb - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text((x0 + 4, y0 + thumb + 4), image_path.stem[:22], fill="black", font=font)
    sheet.save(output_path, quality=95)


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    base_args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint_path = Path(base_args.checkpoint)
    state_dict, checkpoint_args = load_checkpoint(checkpoint_path, device)
    args = build_args(base_args, checkpoint_args)

    _train_df, val_df = load_manifest(args)
    dataset = IntestinalSubtypeDataset(val_df, args.image_size, train=False, bbox_padding=args.bbox_padding)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = OrganoidSubtypeClassifier(
        num_classes=len(CLASSES),
        backbone_name=args.backbone,
        pretrained=False,
        dropout=args.dropout,
        freeze_stages=args.freeze_stages,
    ).to(device)
    model.load_state_dict(state_dict)

    pred_ids, confidences = predict(model, loader, device)
    rows = []
    val_df = val_df.reset_index(drop=True)
    for i, row in val_df.iterrows():
        true_id = int(row["label_id"])
        pred_id = int(pred_ids[i])
        rows.append({
            "training_id": row["training_id"],
            "dataset": row["dataset"],
            "organoid_type": row["organoid_type"],
            "split": row["split"],
            "image_id": row["image_id"],
            "object_id": row["object_id"],
            "image_path": row["image_path"],
            "true_label": CLASSES[true_id],
            "predicted_label": CLASSES[pred_id],
            "confidence": f"{confidences[i]:.6f}",
            "correct": str(true_id == pred_id),
        })

    output_dir = Path(base_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "predictions.csv", rows)
    mistakes = [row for row in rows if row["correct"] == "False"]
    write_rows(output_dir / "misclassified.csv", mistakes)

    by_id = {row["training_id"]: row for _, row in val_df.iterrows()}
    for true_label, pred_label in ERROR_PAIRS:
        pair_dir = output_dir / f"{true_label}_to_{pred_label}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        selected = [
            row for row in mistakes
            if row["true_label"] == true_label and row["predicted_label"] == pred_label
        ][:base_args.max_per_pair]
        saved = []
        for idx, result_row in enumerate(selected, start=1):
            source_row = by_id[result_row["training_id"]]
            image_path = pair_dir / f"{idx:02d}_{source_row['object_id'] or source_row['image_id']}.jpg"
            save_crop(source_row, image_path, args.bbox_padding)
            saved.append(image_path)
        make_contact_sheet(saved, f"{true_label} -> {pred_label}", output_dir / f"{true_label}_to_{pred_label}.jpg")

    print(f"Wrote predictions to {output_dir / 'predictions.csv'}")
    print(f"Wrote {len(mistakes)} misclassified rows to {output_dir / 'misclassified.csv'}")
    print(f"Wrote contact sheets to {output_dir}")


if __name__ == "__main__":
    main()

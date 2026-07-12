"""
Build unified metadata tables for the organoid subtype project.

Outputs:
- metadata/image_manifest.csv: one row per source image or crop.
- metadata/object_manifest.csv: one row per organoid instance when available.

The script does not modify image or label files. It only indexes existing data.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

HUMAN_ROOT = PROJECT_ROOT / "Final_Organoids_Dataset人小肠"
BRAIN_ROOT = ROOT / "data"
MOUSE_ROOT = PROJECT_ROOT / "OrganoidDataset小鼠小肠"
OUTPUT_DIR = ROOT / "metadata"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

HUMAN_LABELS = {
    "0": "cyst",
    "1": "early_budding",
    "2": "late_budding",
    "3": "spheroid",
}

MOUSE_LABELS = {
    "0": "organoid0_cyst",
    "1": "organoid1_early",
    "2": "organoid3_late",
    "3": "spheroid",
}

IMAGE_COLUMNS = [
    "image_id",
    "image_path",
    "dataset",
    "organoid_type",
    "split",
    "subtype_label",
    "label_id",
    "label_source",
    "label_confidence",
    "org_id",
    "clone",
    "day",
    "imaging_source",
    "mask_path",
    "notes",
]

OBJECT_COLUMNS = [
    "object_id",
    "image_id",
    "image_path",
    "dataset",
    "organoid_type",
    "split",
    "subtype_label",
    "label_id",
    "label_source",
    "label_confidence",
    "bbox_x_center",
    "bbox_y_center",
    "bbox_width",
    "bbox_height",
    "bbox_format",
    "mask_path",
    "notes",
]


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def empty_image_row() -> dict[str, str]:
    return {column: "" for column in IMAGE_COLUMNS}


def empty_object_row() -> dict[str, str]:
    return {column: "" for column in OBJECT_COLUMNS}


def collect_human() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    image_rows = []
    object_rows = []

    for split in ("train", "val", "test"):
        split_dir = HUMAN_ROOT / f"{split}_folder"
        if not split_dir.exists():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label_id = class_dir.name
            subtype = HUMAN_LABELS.get(label_id, f"class_{label_id}")
            for image_path in sorted(class_dir.iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                image_id = f"human_intestine:{split}:{label_id}:{image_path.stem}"

                row = empty_image_row()
                row.update({
                    "image_id": image_id,
                    "image_path": rel(image_path),
                    "dataset": "human_intestine_clorg",
                    "organoid_type": "human_intestine",
                    "split": split,
                    "subtype_label": subtype,
                    "label_id": label_id,
                    "label_source": "folder_name",
                    "label_confidence": "high",
                    "notes": "Pre-cropped organoid image; class folder mapped to subtype label.",
                })
                image_rows.append(row)

                obj = empty_object_row()
                obj.update({
                    "object_id": f"{image_id}:object_0",
                    "image_id": image_id,
                    "image_path": rel(image_path),
                    "dataset": "human_intestine_clorg",
                    "organoid_type": "human_intestine",
                    "split": split,
                    "subtype_label": subtype,
                    "label_id": label_id,
                    "label_source": "folder_name",
                    "label_confidence": "high",
                    "bbox_format": "whole_crop",
                    "notes": "The crop is treated as one organoid-level training instance.",
                })
                object_rows.append(obj)

    return image_rows, object_rows


def find_brain_image(img_id: str, imaging_source: str) -> Path | None:
    exts = ("jpg", "tif") if imaging_source == "LabA" else ("tif", "jpg")
    for ext in exts:
        candidate = BRAIN_ROOT / "imgs" / f"{img_id}.{ext}"
        if candidate.exists():
            return candidate
    return None


def collect_brain() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    csv_path = BRAIN_ROOT / "dataset_overview.csv"
    if not csv_path.exists():
        return [], []

    df = pd.read_csv(csv_path)
    image_rows = []
    object_rows = []

    for record in df.to_dict("records"):
        img_id = str(record["img_id"])
        imaging_source = str(record.get("Imaging", ""))
        image_path = find_brain_image(img_id, imaging_source)
        if image_path is None:
            continue

        mask_path = BRAIN_ROOT / "labels" / f"{img_id}.npy"
        mask_value = rel(mask_path) if mask_path.exists() else ""
        subtype = str(record.get("Clone", ""))
        image_id = f"brain:{img_id}"

        row = empty_image_row()
        row.update({
            "image_id": image_id,
            "image_path": rel(image_path),
            "dataset": "brain_organoid_organoidnet",
            "organoid_type": "brain",
            "split": "all",
            "subtype_label": subtype,
            "label_id": subtype,
            "label_source": "clone_metadata",
            "label_confidence": "medium",
            "org_id": str(record.get("org_id", "")),
            "clone": subtype,
            "day": str(record.get("Day", "")),
            "imaging_source": imaging_source,
            "mask_path": mask_value,
            "notes": "Current class label is clone metadata, not necessarily a visual morphology subtype.",
        })
        image_rows.append(row)

        obj = empty_object_row()
        obj.update({
            "object_id": f"{image_id}:object_0",
            "image_id": image_id,
            "image_path": rel(image_path),
            "dataset": "brain_organoid_organoidnet",
            "organoid_type": "brain",
            "split": "all",
            "subtype_label": subtype,
            "label_id": subtype,
            "label_source": "clone_metadata",
            "label_confidence": "medium",
            "bbox_format": "mask_or_whole_image",
            "mask_path": mask_value,
            "notes": "Use mask_path for segmentation-aware crops/features when available.",
        })
        object_rows.append(obj)

    return image_rows, object_rows


def find_mouse_label(image_path: Path, split: str) -> Path:
    return MOUSE_ROOT / split / "labels" / f"{image_path.stem}.txt"


def collect_mouse() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    image_rows = []
    object_rows = []

    for split in ("train", "val"):
        image_dir = MOUSE_ROOT / split / "images"
        if not image_dir.exists():
            continue

        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue

            image_id = f"mouse_intestine:{split}:{image_path.stem}"
            label_path = find_mouse_label(image_path, split)

            row = empty_image_row()
            row.update({
                "image_id": image_id,
                "image_path": rel(image_path),
                "dataset": "mouse_intestine_orgaquant",
                "organoid_type": "mouse_intestine",
                "split": split,
                "subtype_label": "multiple_or_unknown",
                "label_source": "yolo_object_labels",
                "label_confidence": "medium",
                "notes": "Image-level label may contain multiple organoid instances; use object_manifest for labels.",
            })
            image_rows.append(row)

            if not label_path.exists():
                continue

            with label_path.open("r", encoding="utf-8") as handle:
                for object_index, line in enumerate(handle):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    label_id = parts[0]
                    subtype = MOUSE_LABELS.get(label_id, f"class_{label_id}")
                    obj = empty_object_row()
                    obj.update({
                        "object_id": f"{image_id}:object_{object_index}",
                        "image_id": image_id,
                        "image_path": rel(image_path),
                        "dataset": "mouse_intestine_orgaquant",
                        "organoid_type": "mouse_intestine",
                        "split": split,
                        "subtype_label": subtype,
                        "label_id": label_id,
                        "label_source": "yolo_label_file",
                        "label_confidence": "medium",
                        "bbox_x_center": parts[1],
                        "bbox_y_center": parts[2],
                        "bbox_width": parts[3],
                        "bbox_height": parts[4],
                        "bbox_format": "yolo_normalized_xywh",
                        "notes": "Object-level label from YOLO annotation.",
                    })
                    object_rows.append(obj)

    return image_rows, object_rows


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]], keys: tuple[str, ...]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=[*keys, "count"])
    df = pd.DataFrame(rows)
    return df.groupby(list(keys), dropna=False).size().reset_index(name="count")


def main() -> None:
    human_images, human_objects = collect_human()
    brain_images, brain_objects = collect_brain()
    mouse_images, mouse_objects = collect_mouse()

    image_rows = human_images + brain_images + mouse_images
    object_rows = human_objects + brain_objects + mouse_objects

    write_csv(OUTPUT_DIR / "image_manifest.csv", image_rows, IMAGE_COLUMNS)
    write_csv(OUTPUT_DIR / "object_manifest.csv", object_rows, OBJECT_COLUMNS)

    image_summary = summarize(image_rows, ("dataset", "split", "subtype_label"))
    object_summary = summarize(object_rows, ("dataset", "split", "subtype_label"))
    image_summary.to_csv(OUTPUT_DIR / "image_manifest_summary.csv", index=False)
    object_summary.to_csv(OUTPUT_DIR / "object_manifest_summary.csv", index=False)

    print(f"Wrote {len(image_rows)} image rows to {OUTPUT_DIR / 'image_manifest.csv'}")
    print(f"Wrote {len(object_rows)} object rows to {OUTPUT_DIR / 'object_manifest.csv'}")
    print(f"Wrote summaries to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

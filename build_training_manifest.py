"""
Build final training manifests from metadata/image_manifest.csv and object_manifest.csv.

Outputs:
- metadata/training_manifest.csv: all approved training rows.
- metadata/training_manifest_summary.csv: counts by task/dataset/split/label.

The manifest separates:
- intestinal_morphology_subtype: human + mouse intestine mapped to four morphology labels.
- brain_clone_group: brain clone/group labels.
- organoid_domain: domain labels for routing.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
METADATA_DIR = ROOT / "metadata"

INTESTINAL_LABELS = {
    "cyst": "0",
    "early_budding": "1",
    "late_budding": "2",
    "spheroid": "3",
}

MOUSE_TO_FINAL = {
    "organoid0_cyst": "cyst",
    "organoid1_early": "early_budding",
    "organoid3_late": "late_budding",
    "spheroid": "spheroid",
}

BRAIN_LABELS = {
    "A1A-1": "0",
    "B2A-2": "1",
    "TH2-7": "2",
    "wt2D": "3",
}

DOMAIN_LABELS = {
    "human_intestine": "0",
    "mouse_intestine": "1",
    "brain": "2",
}

COLUMNS = [
    "training_id",
    "task",
    "dataset",
    "organoid_type",
    "split",
    "image_id",
    "object_id",
    "image_path",
    "label",
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


def empty_row() -> dict[str, str]:
    return {column: "" for column in COLUMNS}


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def make_human_intestinal_rows(objects: pd.DataFrame) -> list[dict[str, str]]:
    rows = []
    df = objects[objects["dataset"] == "human_intestine_clorg"]
    for record in df.to_dict("records"):
        label = record["subtype_label"]
        if label not in INTESTINAL_LABELS:
            continue
        row = empty_row()
        row.update({
            "training_id": f"intestinal:{record['object_id']}",
            "task": "intestinal_morphology_subtype",
            "dataset": record["dataset"],
            "organoid_type": "human_intestine",
            "split": record["split"],
            "image_id": record["image_id"],
            "object_id": record["object_id"],
            "image_path": record["image_path"],
            "label": label,
            "label_id": INTESTINAL_LABELS[label],
            "label_source": record["label_source"],
            "label_confidence": record["label_confidence"],
            "bbox_format": record["bbox_format"],
            "mask_path": record["mask_path"],
            "notes": "Human intestinal morphology subtype from class folder.",
        })
        rows.append(row)
    return rows


def make_mouse_intestinal_rows(objects: pd.DataFrame) -> list[dict[str, str]]:
    rows = []
    df = objects[objects["dataset"] == "mouse_intestine_orgaquant"]
    for record in df.to_dict("records"):
        original_label = record["subtype_label"]
        label = MOUSE_TO_FINAL.get(original_label)
        if label is None:
            continue
        row = empty_row()
        row.update({
            "training_id": f"intestinal:{record['object_id']}",
            "task": "intestinal_morphology_subtype",
            "dataset": record["dataset"],
            "organoid_type": "mouse_intestine",
            "split": record["split"],
            "image_id": record["image_id"],
            "object_id": record["object_id"],
            "image_path": record["image_path"],
            "label": label,
            "label_id": INTESTINAL_LABELS[label],
            "label_source": record["label_source"],
            "label_confidence": record["label_confidence"],
            "bbox_x_center": record["bbox_x_center"],
            "bbox_y_center": record["bbox_y_center"],
            "bbox_width": record["bbox_width"],
            "bbox_height": record["bbox_height"],
            "bbox_format": record["bbox_format"],
            "mask_path": record["mask_path"],
            "notes": f"Mouse label {original_label} mapped to shared intestinal label {label}.",
        })
        rows.append(row)
    return rows


def make_brain_rows(objects: pd.DataFrame) -> list[dict[str, str]]:
    rows = []
    df = objects[objects["dataset"] == "brain_organoid_organoidnet"]
    for record in df.to_dict("records"):
        label = record["subtype_label"]
        if label not in BRAIN_LABELS:
            continue
        row = empty_row()
        row.update({
            "training_id": f"brain_clone:{record['object_id']}",
            "task": "brain_clone_group",
            "dataset": record["dataset"],
            "organoid_type": "brain",
            "split": "all",
            "image_id": record["image_id"],
            "object_id": record["object_id"],
            "image_path": record["image_path"],
            "label": label,
            "label_id": BRAIN_LABELS[label],
            "label_source": record["label_source"],
            "label_confidence": record["label_confidence"],
            "bbox_format": record["bbox_format"],
            "mask_path": record["mask_path"],
            "notes": "Brain label is clone/group metadata, not confirmed morphology subtype.",
        })
        rows.append(row)
    return rows


def make_domain_rows(images: pd.DataFrame) -> list[dict[str, str]]:
    rows = []
    for record in images.to_dict("records"):
        domain = record["organoid_type"]
        if domain not in DOMAIN_LABELS:
            continue
        row = empty_row()
        row.update({
            "training_id": f"domain:{record['image_id']}",
            "task": "organoid_domain",
            "dataset": record["dataset"],
            "organoid_type": domain,
            "split": record["split"],
            "image_id": record["image_id"],
            "image_path": record["image_path"],
            "label": domain,
            "label_id": DOMAIN_LABELS[domain],
            "label_source": "dataset_domain",
            "label_confidence": "high",
            "mask_path": record["mask_path"],
            "notes": "Domain/type label for routing and dataset shift analysis.",
        })
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    image_manifest = load_csv(METADATA_DIR / "image_manifest.csv")
    object_manifest = load_csv(METADATA_DIR / "object_manifest.csv")

    rows = []
    rows.extend(make_human_intestinal_rows(object_manifest))
    rows.extend(make_mouse_intestinal_rows(object_manifest))
    rows.extend(make_brain_rows(object_manifest))
    rows.extend(make_domain_rows(image_manifest))

    output_path = METADATA_DIR / "training_manifest.csv"
    write_csv(output_path, rows)

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["task", "dataset", "organoid_type", "split", "label"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    summary.to_csv(METADATA_DIR / "training_manifest_summary.csv", index=False)

    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Wrote summary to {METADATA_DIR / 'training_manifest_summary.csv'}")


if __name__ == "__main__":
    main()

"""
一次性预处理：裁剪 + resize → 存为 JPEG，加速训练
运行一次即可: python preprocess.py
"""
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import config

PROCESSED_DIR = config.DATA_DIR / "processed"
PROCESSED_CSV = config.DATA_DIR / "processed_dataset.csv"


def extract_organoid_roi(image, mask, margin=10):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    y1 = max(0, ys.min() - margin)
    y2 = min(image.shape[0], ys.max() + margin)
    x1 = max(0, xs.min() - margin)
    x2 = min(image.shape[1], xs.max() + margin)
    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return None
    return image[y1:y2, x1:x2, :]


def main():
    print("=" * 60)
    print("📦 预处理：裁剪类器官 → resize → 保存 JPEG")
    print("=" * 60)

    df = pd.read_csv(config.CSV_PATH)
    print(f"   总图像: {len(df)}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    processed_rows = []
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        img_id = row["img_id"]
        lab = row["Imaging"]

        # 找图像文件
        img_path = None
        for ext in (["jpg", "tif"] if lab == "LabA" else ["tif", "jpg"]):
            p = config.IMAGE_DIR / f"{img_id}.{ext}"
            if p.exists():
                img_path = p
                break

        if img_path is None:
            failed += 1
            continue

        # 读图像
        if img_path.suffix in [".tif", ".tiff"]:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        if img is None:
            failed += 1
            continue

        # 用 mask 裁剪
        mask_path = config.MASK_DIR / f"{img_id}.npy"
        if mask_path.exists():
            mask = (np.load(str(mask_path)) > 0.5).astype(np.uint8)
            cropped = extract_organoid_roi(img, mask)
            if cropped is not None:
                img = cropped

        # resize + 保存
        img = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        out_path = PROCESSED_DIR / f"{img_id}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        processed_rows.append({
            "org_id": row["org_id"],
            "img_id": img_id,
            "Clone": row["Clone"],
            "label": row["Clone"],  # will be re-mapped
        })

    # 保存新 CSV
    processed_df = pd.DataFrame(processed_rows)
    processed_df.to_csv(PROCESSED_CSV, index=False)

    print(f"\n✅ 完成! {len(processed_df)} 张图像 → {PROCESSED_DIR}")
    print(f"   CSV: {PROCESSED_CSV}")
    if failed > 0:
        print(f"   ⚠️ {failed} 张失败")


if __name__ == "__main__":
    main()

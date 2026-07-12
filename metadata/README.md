# Metadata Manifests

This folder stores unified metadata tables for the organoid subtype project.

Generated files:

- `image_manifest.csv`: one row per source image or crop.
- `object_manifest.csv`: one row per organoid instance when object-level labels are available.
- `image_manifest_summary.csv`: counts by dataset, split, and image-level label.
- `object_manifest_summary.csv`: counts by dataset, split, and object-level label.

Regenerate:

```bash
python build_metadata_manifest.py
```

Notes:

- Human intestine CLORG crops use folder names as labels and are treated as one object per crop.
- Brain organoid rows use clone metadata as the current label. This is useful for supervised experiments, but it is not necessarily a pure visual morphology subtype.
- Mouse intestine OrgaQuant image rows can contain multiple organoids; object-level labels and YOLO boxes are stored in `object_manifest.csv`.

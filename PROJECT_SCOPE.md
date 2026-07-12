# Project Scope

## Current Direction

This project is a prototype framework for image-based organoid type and subtype/group classification.

It does not aim to build a universal zero-shot classifier for all organoid systems. Instead, it provides a reusable workflow for:

- organizing image metadata and labels,
- training classifiers on labelled organoid datasets,
- evaluating domain-specific subtype/group predictions,
- extending the framework when new labelled datasets become available.

## Main Scientific Task

The main scientific task is intestinal organoid morphology subtype classification.

Current shared intestinal labels:

- `cyst`
- `early_budding`
- `late_budding`
- `spheroid`

Current data sources:

- human intestine organoid crops,
- mouse intestine organoid object crops from YOLO annotations.

## Auxiliary Task

Brain organoid images are currently used for clone/group classification:

- `A1A-1`
- `B2A-2`
- `TH2-7`
- `wt2D`

These labels should not be described as confirmed morphology subtypes unless manual morphology annotation is added later.

## Framework Task

The framework also supports organoid domain/type classification:

- `human_intestine`
- `mouse_intestine`
- `brain`

This can be used for routing, quality control, and future dataset expansion.

## Recommended Evaluation

Use:

- accuracy,
- macro-F1,
- balanced accuracy,
- per-class precision/recall/F1,
- confusion matrix.

Accuracy alone is not enough because the datasets are class-imbalanced.

## Extension Strategy

For a new organoid dataset:

1. Register the dataset in metadata.
2. Define label schema and annotation rules.
3. Annotate a small representative subset.
4. Fine-tune or add a new expert head/model.
5. Validate with balanced metrics and error analysis before use.

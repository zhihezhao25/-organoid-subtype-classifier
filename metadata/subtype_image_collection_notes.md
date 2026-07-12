# Subtype Image Source Notes

This file is specifically for organoid image sources with morphology or subtype labels. It is separate from general segmentation, detection, or tool repositories.

## Key Finding

The most valuable data for this project is not another general organoid tool; it is image data with explicit subtype labels. The current project already has two useful local subtype-image sources:

- Human intestine crops labeled as `cyst`, `early_budding`, `late_budding`, and `spheroid`.
- Mouse intestine YOLO object labels mapped to `organoid0_cyst`, `organoid1_early`, `organoid3_late`, and `spheroid`.

These should be treated as the current primary supervised subtype datasets.

## Strong External Candidates

### D-CryptO

- URL: https://arxiv.org/abs/2210.06538
- Labels: crypt formation and opacity.
- Why it matters: it is directly about colorectal organoid morphology classification from brightfield images.
- Current status: paper verified, but public raw image data/code has not been found yet.
- Action: keep searching for data/code or consider contacting authors.

### NOA

- URL: https://arxiv.org/abs/2511.01549
- Local code: `../../external_sources/napari-organoid-analyzer`
- Why it matters: it supports detection, segmentation, feature extraction, custom feature annotation, and ML-based feature prediction.
- Current status: code downloaded; bundled subtype image data not confirmed.
- Action: use as annotation workflow reference.

### MultiOrg

- URL: https://www.kaggle.com/datasets/christinabukas/mutliorg
- Labels: object boxes plus study types such as `Normal` and `Macros`, not clean morphology subtypes.
- Why it matters: very useful for detection pretraining and annotation uncertainty; less direct for subtype classification.
- Current status: Kaggle entry verified; not downloaded.

## Practical Recommendation

Do not wait for a perfect public subtype dataset. The most productive route is:

1. Use the existing human and mouse intestine subtype-labeled images as supervised training data.
2. Build a manual review workflow for a subset of ambiguous images using the ontology and NOA-style annotation.
3. Use D-CryptO as the closest literature model for morphology subtype classification.
4. Use OrganoID/OrgaExtractor for segmentation and feature extraction, not as subtype-label sources.
5. Add external subtype datasets only after confirming image access, label schema, and license.

## Target Manifest Schema

Future downloaded subtype datasets should be converted into:

```text
image_id,image_path,object_id,organoid_type,subtype_label,label_definition,label_source,license,source_url,split
```

For object-level datasets:

```text
object_id,image_id,bbox_or_mask_path,subtype_label,label_confidence,annotator,source_url
```

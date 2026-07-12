# Final Label Schema

Project framing:

This project builds a prototype framework for image-based organoid type and subtype/group classification. It does not claim to be a universal zero-shot classifier for all organoids. The current system is validated on three available data sources and is designed to be extendable when new labelled data are added.

## Task Levels

### Level 1: Organoid Type / Data Domain

The first level identifies which data domain an image belongs to:

| Domain | Meaning | Current status |
| --- | --- | --- |
| `human_intestine` | Human intestinal organoid crops | morphology subtype labels available |
| `mouse_intestine` | Mouse intestinal organoid object crops from YOLO boxes | morphology-like object labels available |
| `brain` | Brain organoid images | clone/group labels available, not confirmed morphology subtypes |

### Level 2: Domain-Specific Labels

#### Human Intestine Morphology Subtype

| Label ID | Final label | Meaning |
| --- | --- | --- |
| `0` | `cyst` | cystic/lumen-forming morphology |
| `1` | `early_budding` | early budding morphology |
| `2` | `late_budding` | mature or more complex budding morphology |
| `3` | `spheroid` | round spheroid-like morphology |

#### Mouse Intestine Object-Level Morphology Label

The original mouse labels are mapped into the same four-class morphology vocabulary for the shared intestinal task.

| Original label | Final label |
| --- | --- |
| `organoid0_cyst` | `cyst` |
| `organoid1_early` | `early_budding` |
| `organoid3_late` | `late_budding` |
| `spheroid` | `spheroid` |

#### Brain Organoid Clone/Group Label

Brain labels are kept as group labels, not treated as verified morphology subtypes.

| Label |
| --- |
| `A1A-1` |
| `B2A-2` |
| `TH2-7` |
| `wt2D` |

## Recommended Experiments

1. Intestinal morphology subtype classifier:
   - Data: human intestine + mouse intestine.
   - Labels: `cyst`, `early_budding`, `late_budding`, `spheroid`.
   - Main scientific task.

2. Brain clone/group classifier:
   - Data: brain organoid images.
   - Labels: `A1A-1`, `B2A-2`, `TH2-7`, `wt2D`.
   - Auxiliary experiment only; do not describe as morphology subtype classification unless labels are manually redefined.

3. Domain/type classifier:
   - Data: all three domains.
   - Labels: `human_intestine`, `mouse_intestine`, `brain`.
   - Used as a routing or quality-control component.

4. Multi-task model:
   - Data: all domains.
   - Purpose: research comparison against expert/domain-specific models.

## Reporting Rules

- Use `accuracy` only as a secondary metric.
- Main metrics should include `macro-F1`, `balanced accuracy`, per-class F1, and confusion matrix.
- Do not compare brain clone labels directly against intestinal morphology labels.
- New organoid datasets should be added through metadata registration, small manual annotation, fine-tuning, and validation.

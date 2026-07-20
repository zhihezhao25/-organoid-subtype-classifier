# Experiment Results

This file records the current intestinal organoid morphology subtype classification experiments.

Current main task:

```text
Input: human intestinal organoid crops and/or mouse intestinal organoid object crops
Output: cyst / early_budding / late_budding / spheroid
```

## Summary Table

| Run | Data domain | Key setting | Best epoch | Accuracy | Balanced accuracy | Macro-F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `intestine_subtype_b0` | human + mouse intestine | EfficientNet-B0, weighted sampler | 25 | 0.9227 | 0.9218 | 0.9099 |
| `human_intestine_b0` | human intestine only | EfficientNet-B0, weighted sampler | 22 | 0.8907 | 0.8958 | 0.8766 |
| `mouse_intestine_b0` | mouse intestine only | EfficientNet-B0, weighted sampler | 25 | 0.8866 | 0.8916 | 0.8667 |
| `intestine_subtype_b0_no_sampler` | human + mouse intestine | EfficientNet-B0, no weighted sampler | 24 | 0.9251 | 0.9177 | 0.9102 |

The structured CSV version is available at:

```text
metadata/experiment_results.csv
```

## Main Findings

1. Combined human + mouse intestinal training gave the strongest overall result.

   The combined-domain models reached about 92% validation accuracy and 91% macro-F1, outperforming both the human-only and mouse-only baselines. This suggests that the two intestinal datasets share transferable morphology information.

2. The four-class intestinal morphology classification task is feasible.

   The model consistently distinguished `cyst`, `early_budding`, `late_budding`, and `spheroid` above the single-domain baselines, supporting the feasibility of image-based intestinal organoid subtype classification.

3. The weighted sampler was not the dominant factor.

   Removing the weighted sampler slightly increased accuracy and macro-F1 but slightly reduced balanced accuracy. The two combined-domain runs were very close, suggesting that the classifier is relatively stable and that class-balanced loss and pretrained image features are more important in the current setting.

4. The main errors occurred between visually similar categories.

   Most confusion occurred between:

   - `early_budding` and `late_budding`
   - `cyst` and `spheroid`

   This is biologically reasonable because intestinal organoid morphology can change continuously, and some examples lie near category boundaries.

## Recommended Result for Reporting

For reports, posters, and project summaries, use `intestine_subtype_b0` as the main baseline:

```text
The combined human and mouse intestinal organoid classifier achieved 92.27% validation accuracy, 92.18% balanced accuracy, and 90.99% macro-F1 across four morphology subtypes.
```

Use `intestine_subtype_b0_no_sampler` as an ablation/control experiment for the weighted sampler.

## Output Files

Main combined baseline:

```text
logs/intestine_subtype_b0/metrics.json
logs/intestine_subtype_b0/confusion_matrix.png
logs/intestine_subtype_b0/result_summary.md
```

Human-only baseline:

```text
logs/human_intestine_b0/metrics.json
logs/human_intestine_b0/confusion_matrix.png
logs/human_intestine_b0/result_summary.md
```

Mouse-only baseline:

```text
logs/mouse_intestine_b0/metrics.json
logs/mouse_intestine_b0/confusion_matrix.png
logs/mouse_intestine_b0/result_summary.md
```

No weighted sampler ablation:

```text
logs/intestine_subtype_b0_no_sampler/metrics.json
logs/intestine_subtype_b0_no_sampler/confusion_matrix.png
logs/intestine_subtype_b0_no_sampler/result_summary.md
```

# Legacy Scripts

This folder keeps early experiments and one-off utilities that are not part of
the current main workflow.

Current recommended workflow:

- `train_intestinal_subtype.py`: main intestinal four-class morphology baseline.
- `train_mil_multitask.py`: Attention-MIL multitask comparison.
- `build_metadata_manifest.py` and `build_training_manifest.py`: reproducible
  metadata and training manifests.

Archived files here are retained for traceability, but they may require running
from the repository root or adding the repository root to `PYTHONPATH`.

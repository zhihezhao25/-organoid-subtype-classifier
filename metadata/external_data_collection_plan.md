# External Data Collection Plan

Goal: collect online resources that can improve organoid subtype classification without mixing unusable or unlicensed data into the project.

## What To Collect First

1. Public organoid image datasets with labels or masks.
2. Object-level annotations, especially bbox/mask files.
3. Morphology subtype label definitions, such as cystic, budding, solid, spheroid, branched, and opacity/crypt-formation labels.
4. Data/code availability and license information for every source.
5. Small metadata records before large file downloads.

## What Not To Bulk Download Yet

- Large imaging archives without clear license.
- Papers/PDFs only, unless they define labels or methods.
- Non-organoid cell image datasets, unless used only as representation-learning or methodology references.
- Clinical datasets without clear reuse permission.

## Proposed Workflow

1. Build a source registry in `external_data_sources.csv`.
2. For each candidate, verify official repository, license, data size, label schema, and citation.
3. Add a local `external_manifest.csv` only after files are downloaded or linked.
4. Keep downloaded external data outside model code and record provenance.
5. Use external data in this order:
   - segmentation/object detection pretraining,
   - morphology feature extraction validation,
   - subtype taxonomy and annotation guidelines,
   - supervised subtype classification only if labels match the project.

## Current Priority

The most useful immediate sources are MultiOrg, OrganoID, OrgaExtractor, and D-CryptO because they are closest to organoid image detection, segmentation, and morphology classification.

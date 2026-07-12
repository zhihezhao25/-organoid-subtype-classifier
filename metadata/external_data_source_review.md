# External Data Source Review

This review records online organoid image/data sources that may help the organoid subtype classification project. Files and notes are kept inside `/Users/zhaozhihe/Desktop/2026surf`.

## Highest Priority

### OrganoID

- Paper: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010584
- Code: https://github.com/jono-m/OrganoID
- Dataset: https://osf.io/xmes4/
- Best use: segmentation baseline, mask-based morphology extraction, longitudinal organoid dynamics.
- Why useful: the paper reports brightfield/phase-contrast organoid detection, segmentation, and tracking. It explicitly says the GitHub repository contains source code plus the training/testing dataset, and also links the dataset on OSF.
- Relevance to current project: high. It can support the segmentation-first branch of the project and help extract area, circularity, solidity, eccentricity, and growth features.

### OrgaExtractor

- Paper: https://www.nature.com/articles/s41598-023-46485-2
- Code/data: https://github.com/tpark16/orgaextractor
- Best use: segmentation baseline and shape-feature extraction.
- Why useful: the paper and repository describe a small colon organoid segmentation dataset with binary masks. The repository lists MIT license and a dataset link.
- Useful feature terms: projected area, diameter, perimeter, major/minor axis length, eccentricity, circularity, roundness, solidity.
- Relevance to current project: high for feature engineering and baseline segmentation, even though the dataset is small.

### MultiOrg

- Paper: https://arxiv.org/abs/2410.14612
- Best use: object detection and annotation uncertainty.
- Why useful: the abstract reports over 400 high-resolution 2D microscopy images, more than 60,000 organoid annotations, and multiple expert label sets for the test data.
- Current limitation: official download URL and license were not confirmed yet.
- Relevance to current project: very high if the data is downloadable, because object-level labels can improve detection/cropping and quantify label uncertainty.

### D-CryptO

- Paper: https://arxiv.org/abs/2210.06538
- Best use: morphology classification label design.
- Why useful: it directly classifies colorectal organoid morphology from brightfield images using crypt formation and opacity, which is close to the subtype-classification goal.
- Current limitation: public raw data/code were not confirmed yet.
- Relevance to current project: high as a label-taxonomy and methodology source; direct training use depends on data availability.

## Secondary Priority

### NOA: Napari Organoid Analyzer

- Paper: https://arxiv.org/abs/2511.01549
- Code: https://github.com/Meleray/napari-organoid-analyzer
- Best use: annotation workflow and feature-based ML design.
- Why useful: it combines detection, segmentation, tracking, feature extraction, manual annotation, and ML prediction. It defines differentiation stages such as spheroid I, spheroid II, budding, and enteroid.
- Relevance to current project: useful for designing annotation standards and a feature-table classifier.

### BOrg

- Paper: https://arxiv.org/abs/2406.19556
- Code/data: https://github.com/awaisrauf/borg
- Best use: brain organoid auxiliary detection/cell-state task.
- Why useful: the repository says the dataset is released in mmdetection format and contains brain organoid confocal images with mitosis phase annotations.
- Limitation: labels are mitotic phases, not whole-organoid morphology subtypes.
- Relevance to current project: secondary. Useful only if brain organoid representation learning or auxiliary tasks are needed.

### SegmentAnything for organoid microscopy analysis

- Paper: https://arxiv.org/abs/2309.04190
- Best use: feature engineering and segmentation workflow.
- Why useful: the paper proposes SAM-based organoid segmentation and morphology measurements including perimeter, area, radius, non-smoothness, and non-circularity.
- Relevance to current project: useful as a method reference.

### Organoid Tracker

- Paper: https://arxiv.org/abs/2509.11063
- Code: https://github.com/hrlblab/OrganoidTracker
- Best use: video/time-series organoid analysis.
- Why useful: it targets kidney organoid videos and quantifies cyst formation rate, growth velocity, and morphology changes.
- Relevance to current project: useful if adding longitudinal growth features.

## Download Recommendation

Download first:

1. OrganoID OSF dataset, if size is manageable.
2. OrgaExtractor small Google Drive dataset, after confirming access and license.

Verify before download:

1. MultiOrg official dataset link and license.
2. D-CryptO raw data/code availability.

Use as references without bulk download:

1. NOA.
2. SegmentAnything organoid paper.
3. Organoid Tracker.
4. CRC/oral-cancer morphology papers.

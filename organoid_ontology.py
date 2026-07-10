"""
类器官类型与常见亚型/形态特征知识库

用途：
- 为初步类器官亚型分类研究提供候选标签体系。
- 辅助解释 discover_subtypes.py 产生的 cluster。
- 注意：这里是文献/综述层面的通用知识，不等于模型可直接无监督准确识别。

字段说明：
- organoid_type: 器官/系统来源
- subtype: 常见亚型、阶段、区域或形态类别
- category: morphology / developmental_stage / regional_identity / lineage / disease_model / culture_state
- visual_features: 显微图像中可能观察到的特征
- biological_notes: 生物学含义或常见背景
- synonyms: 常见同义词或近似叫法
- caution: 使用时的注意事项
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable
import csv
import json
from pathlib import Path


@dataclass(frozen=True)
class OrganoidSubtype:
    organoid_type: str
    subtype: str
    category: str
    visual_features: tuple[str, ...]
    biological_notes: str
    synonyms: tuple[str, ...] = ()
    caution: str = ""


ORGANOID_ONTOLOGY: tuple[OrganoidSubtype, ...] = (
    # ------------------------------------------------------------------
    # General morphology shared across many epithelial/tumor organoids
    # ------------------------------------------------------------------
    OrganoidSubtype(
        "general_morphology",
        "spheroid",
        "morphology",
        ("round or oval outline", "smooth boundary", "compact 3D sphere", "limited protrusions"),
        "A broad morphology seen in many organoid systems and tumor spheroids.",
        ("sphere-like", "round organoid", "spheroid-like"),
        "Spheroid is a morphology, not a tissue-specific lineage.",
    ),
    OrganoidSubtype(
        "general_morphology",
        "cystic",
        "morphology",
        ("clear central lumen", "thin epithelial wall", "smooth circular boundary", "hollow appearance"),
        "Common in intestinal, gastric, pancreatic, liver ductal, and some tumor organoids.",
        ("lumen-forming", "cyst", "hollow"),
        "Requires image quality sufficient to see lumen; brightfield contrast can vary.",
    ),
    OrganoidSubtype(
        "general_morphology",
        "budding",
        "morphology",
        ("multiple bud-like protrusions", "irregular lobulated edge", "crypt-like extensions"),
        "Often indicates intestinal crypt-like growth or branching epithelial expansion.",
        ("budded", "crypt-budding", "lobulated"),
        "Budding can be confused with merged nearby organoids or segmentation artifacts.",
    ),
    OrganoidSubtype(
        "general_morphology",
        "solid_compact",
        "morphology",
        ("dense interior", "weak or absent lumen", "high opacity", "compact mass"),
        "Frequently used for tumor organoids or less differentiated epithelial cultures.",
        ("solid", "compact", "dense"),
        "Solid appearance may also result from focus, illumination, or necrotic center.",
    ),
    OrganoidSubtype(
        "general_morphology",
        "branched",
        "morphology",
        ("branch-like extensions", "tree-like structure", "elongated projections", "complex perimeter"),
        "Seen in airway, mammary, prostate, kidney/ureteric bud, and some glandular organoids.",
        ("branching", "tree-like", "ductal branching"),
        "Branching morphology is tissue-context dependent.",
    ),
    OrganoidSubtype(
        "general_morphology",
        "irregular_fragmented",
        "morphology",
        ("fragmented boundary", "debris-like regions", "low circularity", "heterogeneous texture"),
        "Can represent damaged organoids, poor culture quality, or true invasive/irregular phenotype.",
        ("fragmented", "debris-like", "irregular"),
        "Needs manual review; not always a biological subtype.",
    ),

    # ------------------------------------------------------------------
    # Intestinal / colorectal
    # ------------------------------------------------------------------
    OrganoidSubtype(
        "intestinal",
        "enterosphere",
        "developmental_stage",
        ("small round spheroid", "smooth edge", "early compact/cystic morphology"),
        "Early intestinal organoid state before prominent crypt budding.",
        ("early intestinal spheroid", "early organoid"),
    ),
    OrganoidSubtype(
        "intestinal",
        "cystic_intestinal",
        "morphology",
        ("large central lumen", "thin epithelial ring", "smooth cyst-like morphology"),
        "Common intestinal organoid morphology, often associated with differentiated/cystic growth conditions.",
        ("cystic", "lumen-forming intestinal organoid"),
    ),
    OrganoidSubtype(
        "intestinal",
        "early_budding",
        "developmental_stage",
        ("one or few small buds", "mostly round body", "emerging protrusions"),
        "Intermediate intestinal organoid morphology as crypt domains start forming.",
        ("early budding", "few buds"),
    ),
    OrganoidSubtype(
        "intestinal",
        "late_budding",
        "developmental_stage",
        ("many crypt-like buds", "lobulated boundary", "complex edge", "larger structure"),
        "Mature crypt-like intestinal organoids with pronounced budding domains.",
        ("mature budding", "crypt-like", "multi-budded"),
    ),
    OrganoidSubtype(
        "colorectal_cancer",
        "cystic_crc",
        "morphology",
        ("cystic lumen", "smooth or moderately irregular wall", "hollow tumor organoid"),
        "One common colorectal cancer organoid morphology.",
        ("cystic tumor organoid", "hollow CRC organoid"),
    ),
    OrganoidSubtype(
        "colorectal_cancer",
        "solid_crc",
        "morphology",
        ("solid dense mass", "no clear lumen", "compact opaque interior"),
        "Common tumor organoid morphology associated with compact growth.",
        ("solid tumor organoid", "compact CRC organoid"),
    ),
    OrganoidSubtype(
        "colorectal_cancer",
        "budding_invasive_crc",
        "morphology",
        ("irregular protrusions", "budding edge", "small detached clusters", "invasive-like margins"),
        "Tumor organoid phenotype sometimes linked to invasive/budding behavior.",
        ("budding CRC", "invasive-like"),
        "Needs biological validation; morphology alone is not enough.",
    ),

    # ------------------------------------------------------------------
    # Brain / neural
    # ------------------------------------------------------------------
    OrganoidSubtype(
        "brain",
        "cerebral_organoid",
        "regional_identity",
        ("opaque neuroepithelial tissue", "rosette-like regions", "heterogeneous internal texture"),
        "General cerebral organoid containing mixed forebrain/midbrain/hindbrain-like tissues depending on protocol.",
        ("whole brain organoid", "cerebral"),
    ),
    OrganoidSubtype(
        "brain",
        "forebrain_organoid",
        "regional_identity",
        ("neuroepithelial rosettes", "ventricle-like cavities", "layered radial structures"),
        "Region-specific organoid patterned toward forebrain/cortical identity.",
        ("cortical organoid", "dorsal forebrain", "ventral forebrain"),
    ),
    OrganoidSubtype(
        "brain",
        "midbrain_organoid",
        "regional_identity",
        ("dense neural tissue", "pigmented/darker regions sometimes in dopaminergic protocols", "neural rosettes"),
        "Organoid patterned toward midbrain identity, often used for dopaminergic neuron studies.",
        ("dopaminergic midbrain organoid",),
        "Brightfield morphology alone may not reliably separate regional identity.",
    ),
    OrganoidSubtype(
        "brain",
        "retinal_brain_related_organoid",
        "regional_identity",
        ("optic-cup-like vesicles", "layered retinal tissue", "pigmented RPE-like areas"),
        "Retinal organoids are neural ectoderm-derived but often treated as a separate system.",
        ("optic cup", "retinal vesicle"),
    ),
    OrganoidSubtype(
        "brain",
        "assembloid",
        "culture_state",
        ("fused organoid masses", "boundary between fused regions", "larger composite structure"),
        "Fusion of region-specific organoids, used to model cell migration and circuit interactions.",
        ("brain assembloid", "fused organoid"),
        "Assembloid identity is usually defined by protocol, not only image morphology.",
    ),

    # ------------------------------------------------------------------
    # Liver / biliary / pancreas / gastric
    # ------------------------------------------------------------------
    OrganoidSubtype(
        "liver",
        "hepatic_organoid",
        "lineage",
        ("compact epithelial clusters", "sometimes cystic or dense", "variable lumen visibility"),
        "Hepatocyte-like or fetal liver-like organoids depending on protocol.",
        ("hepatocyte organoid", "hepatic-like"),
    ),
    OrganoidSubtype(
        "liver",
        "cholangiocyte_organoid",
        "lineage",
        ("cystic duct-like spheres", "clear lumen", "epithelial wall"),
        "Bile duct/cholangiocyte organoids often show cystic ductal morphology.",
        ("bile duct organoid", "ductal liver organoid"),
    ),
    OrganoidSubtype(
        "pancreas",
        "pancreatic_ductal_organoid",
        "lineage",
        ("cystic duct-like structures", "smooth epithelial boundary", "lumen"),
        "Pancreatic ductal organoids and PDAC organoids commonly show cystic or compact tumor morphologies.",
        ("ductal pancreatic organoid",),
    ),
    OrganoidSubtype(
        "pancreas",
        "pancreatic_tumor_solid",
        "disease_model",
        ("solid compact mass", "irregular boundary", "dense interior"),
        "Pancreatic cancer organoids often vary from cystic to solid/invasive-like morphologies.",
        ("PDAC organoid", "solid pancreatic cancer organoid"),
    ),
    OrganoidSubtype(
        "gastric",
        "gastric_cystic",
        "morphology",
        ("cystic epithelial sphere", "central lumen", "smooth wall"),
        "Common gastric organoid morphology from antral/corpus/fundic protocols.",
        ("gastric cyst", "gastric spheroid"),
    ),
    OrganoidSubtype(
        "gastric",
        "gastric_glandular_budding",
        "morphology",
        ("budding or gland-like protrusions", "lobulated boundary", "epithelial branches"),
        "Gastric gland-like organoid morphology in some culture conditions.",
        ("glandular gastric organoid",),
    ),

    # ------------------------------------------------------------------
    # Lung / airway
    # ------------------------------------------------------------------
    OrganoidSubtype(
        "lung",
        "airway_organoid",
        "lineage",
        ("cystic or branched airway-like structures", "lumen", "epithelial wall"),
        "Models conducting airway epithelium; morphology can be cystic or branched.",
        ("bronchosphere", "airway sphere"),
    ),
    OrganoidSubtype(
        "lung",
        "alveolar_organoid",
        "lineage",
        ("small rounded spheres", "thin epithelial structures", "alveolosphere-like morphology"),
        "Models alveolar epithelial lineages, often AT2-derived alveolospheres.",
        ("alveolosphere", "AT2 organoid"),
    ),
    OrganoidSubtype(
        "lung",
        "lung_tumor_organoid",
        "disease_model",
        ("solid or cystic tumor spheres", "irregular margins", "heterogeneous opacity"),
        "Lung cancer organoids can show diverse tumor-like morphologies.",
        ("lung cancer organoid", "tumoroid"),
    ),

    # ------------------------------------------------------------------
    # Kidney / urinary / reproductive / other systems
    # ------------------------------------------------------------------
    OrganoidSubtype(
        "kidney",
        "nephron_organoid",
        "lineage",
        ("segmented epithelial tubules", "glomerulus-like round structures", "complex internal pattern"),
        "Kidney organoids often contain nephron-like segmented structures.",
        ("renal organoid", "nephron-like"),
        "Brightfield whole-organoid images may not separate nephron segments without staining.",
    ),
    OrganoidSubtype(
        "kidney",
        "ureteric_bud_organoid",
        "lineage",
        ("branching epithelial tree", "duct-like branches", "tree-like pattern"),
        "Ureteric bud/collecting duct organoids often show branching morphogenesis.",
        ("collecting duct organoid", "ureteric bud"),
    ),
    OrganoidSubtype(
        "retina",
        "optic_cup_organoid",
        "developmental_stage",
        ("optic cup-like folded vesicle", "layered retinal tissue", "curved neuroepithelial sheet"),
        "Retinal organoid stage with optic cup-like architecture.",
        ("retinal cup", "optic vesicle"),
    ),
    OrganoidSubtype(
        "retina",
        "rpe_organoid_region",
        "lineage",
        ("dark pigmented patches", "RPE-like monolayer or islands", "high contrast regions"),
        "Retinal pigment epithelium-like regions can appear dark/pigmented.",
        ("RPE-like", "pigmented retinal organoid"),
    ),
    OrganoidSubtype(
        "mammary",
        "mammosphere",
        "morphology",
        ("round epithelial sphere", "solid or hollow sphere", "smooth boundary"),
        "Mammary epithelial organoids/spheres may be solid or lumen-forming.",
        ("breast organoid", "mammary sphere"),
    ),
    OrganoidSubtype(
        "mammary",
        "branching_mammary_organoid",
        "morphology",
        ("duct-like branches", "tree-like epithelial outgrowth", "elongated projections"),
        "Used to model mammary branching morphogenesis.",
        ("branching breast organoid",),
    ),
    OrganoidSubtype(
        "prostate",
        "prostasphere",
        "morphology",
        ("spherical epithelial structure", "cystic or solid", "smooth boundary"),
        "Prostate organoids can form hollow or solid epithelial spheres.",
        ("prostate organoid",),
    ),
    OrganoidSubtype(
        "prostate",
        "branching_prostate_organoid",
        "morphology",
        ("branched epithelial outgrowth", "ductal projections", "irregular lobulated structure"),
        "Models prostate ductal branching and glandular morphogenesis.",
        ("ductal prostate organoid",),
    ),
    OrganoidSubtype(
        "endometrium",
        "endometrial_gland_organoid",
        "lineage",
        ("cystic gland-like sphere", "central lumen", "epithelial wall"),
        "Endometrial organoids commonly model glandular epithelium.",
        ("uterine organoid", "endometrial gland"),
    ),
    OrganoidSubtype(
        "bladder",
        "urothelial_organoid",
        "lineage",
        ("spherical epithelial organoid", "cystic or compact morphology", "smooth boundary"),
        "Urothelial/bladder organoids represent urinary tract epithelium.",
        ("bladder organoid", "urothelial sphere"),
    ),
)


SOURCE_NOTES = {
    "scope": "Curated high-level ontology from common organoid review concepts and morphology terms.",
    "limitations": [
        "Organoid subtype names are not standardized across papers.",
        "Many labels are protocol-defined or marker-defined, not visually separable in brightfield images.",
        "Morphology terms such as cystic/budding/solid can appear across multiple tissues.",
        "Use this ontology to guide annotation and cluster interpretation, not as ground truth.",
    ],
    "recommended_workflow": [
        "Run discover_subtypes.py to cluster images by visual embedding.",
        "Inspect cluster_examples manually.",
        "Use this ontology to choose candidate names for clusters.",
        "Train a supervised classifier only after cluster labels are reviewed.",
    ],
}


def as_dicts(records: Iterable[OrganoidSubtype] = ORGANOID_ONTOLOGY) -> list[dict]:
    return [asdict(record) for record in records]


def filter_by_type(organoid_type: str) -> list[OrganoidSubtype]:
    return [record for record in ORGANOID_ONTOLOGY if record.organoid_type == organoid_type]


def filter_by_category(category: str) -> list[OrganoidSubtype]:
    return [record for record in ORGANOID_ONTOLOGY if record.category == category]


def search_ontology(query: str) -> list[OrganoidSubtype]:
    query = query.lower()
    matches = []
    for record in ORGANOID_ONTOLOGY:
        haystack = " ".join([
            record.organoid_type,
            record.subtype,
            record.category,
            record.biological_notes,
            " ".join(record.visual_features),
            " ".join(record.synonyms),
        ]).lower()
        if query in haystack:
            matches.append(record)
    return matches


def export_json(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump({"source_notes": SOURCE_NOTES, "records": as_dicts()}, file, ensure_ascii=False, indent=2)


def export_csv(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = as_dicts()
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            row = row.copy()
            row["visual_features"] = "; ".join(row["visual_features"])
            row["synonyms"] = "; ".join(row["synonyms"])
            writer.writerow(row)


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "ontology"
    export_json(output_dir / "organoid_ontology.json")
    export_csv(output_dir / "organoid_ontology.csv")
    print(f"Exported {len(ORGANOID_ONTOLOGY)} ontology records to {output_dir}")

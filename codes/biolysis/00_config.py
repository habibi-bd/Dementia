from pathlib import Path

PROJECT = Path(r"D:\Dementia2")

DATA = PROJECT / "data"
MATRIX_DIR = DATA / "matrix"
GPL_DIR = DATA / "gpl"

METADATA_DIR = PROJECT / "metadata"
PROCESSED_DIR = PROJECT / "processed"
EXPR_DIR = PROCESSED_DIR / "gene_expression"
QC_DIR = PROCESSED_DIR / "qc"
MAPPING_DIR = PROCESSED_DIR / "mappings"

DE_DIR = PROJECT / "de_analysis"
META_DIR = PROJECT / "meta_analysis"
ENRICH_DIR = PROJECT / "enrichment"
NETWORK_DIR = PROJECT / "network"
ML_DIR = PROJECT / "machine_learning"
FIG_DIR = PROJECT / "figures"
RESULTS_DIR = PROJECT / "results"

for d in [
    METADATA_DIR,
    EXPR_DIR,
    QC_DIR,
    MAPPING_DIR,
    DE_DIR,
    META_DIR,
    ENRICH_DIR,
    NETWORK_DIR,
    ML_DIR,
    FIG_DIR,
    RESULTS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)


GSE_GPL_MAP = {
    "GSE12685": ["GPL96"],
    "GSE1297": ["GPL96"],
    "GSE4757": ["GPL570"],
    "GSE5281": ["GPL570"],
    "GSE6834": ["GPL4757"],
    "GSE15222": ["GPL2700"],
    "GSE16759": ["GPL570", "GPL8757"],
    "GSE26927": ["GPL6255"],
    "GSE28146": ["GPL570"],
    "GSE29378": ["GPL6947"],
    "GSE33000": ["GPL4372"],
    "GSE36980": ["GPL6244"],
    "GSE37263": ["GPL5175"],
    "GSE44770": ["GPL4372"],
}


# mRNA platforms to use in the main dementia analysis
MAIN_GSE_GPL = {
    "GSE12685": "GPL96",
    "GSE1297": "GPL96",
    "GSE4757": "GPL570",
    "GSE5281": "GPL570",
    "GSE6834": "GPL4757",
    "GSE15222": "GPL2700",
    "GSE16759": "GPL570",
    "GSE26927": "GPL6255",
    "GSE28146": "GPL570",
    "GSE29378": "GPL6947",
    "GSE33000": "GPL4372",
    "GSE36980": "GPL6244",
    "GSE37263": "GPL5175",
    "GSE44770": "GPL4372",
}


RANDOM_STATE = 42

# Primary DE thresholds
FDR_THRESHOLD = 0.05
LOG2FC_THRESHOLD = 1.0

# Minimum proportion of samples where a gene should be reasonably measurable
MIN_GENE_DETECTION = 0.20

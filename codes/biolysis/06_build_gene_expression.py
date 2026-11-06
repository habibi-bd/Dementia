# ============================================================
# STEP 6 — BUILD GENE-LEVEL EXPRESSION MATRICES
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import re
import warnings

warnings.filterwarnings("ignore")

BASE = Path(r"D:\Dementia2")
PARSED_DIR = Path(r"D:\Dementia2\codes\biolysis\step02_parsed")
ANNOT_DIR = Path(r"D:\Dementia2\codes\biolysis\step05_probe_annotation")
OUT_DIR = Path(r"D:\Dementia2\codes\biolysis\step06_gene_expression")

OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "GSE1297": "GPL96",
    "GSE4757": "GPL570",
    "GSE5281": "GPL570",
    "GSE6834": "GPL4757",
    "GSE15222": "GPL2700",
    "GSE16759": "GPL570",
    "GSE28146": "GPL570",
    "GSE29378": "GPL6947",
    "GSE33000": "GPL4372",
    "GSE36980": "GPL6244",
    "GSE37263": "GPL5175",
    "GSE44770": "GPL4372",
}


# ------------------------------------------------------------
# NORMALIZE
# ------------------------------------------------------------


def norm_id(x):
    if pd.isna(x):
        return ""

    x = str(x).strip().strip('"').strip("'")

    if re.fullmatch(r"\d+\.0", x):
        x = x[:-2]

    return x.lower()


# ------------------------------------------------------------
# FIND EXPRESSION FILE
# ------------------------------------------------------------


def find_expression_file(gse):

    candidates = [
        PARSED_DIR / f"{gse}.csv",
        PARSED_DIR / f"{gse}_parsed.csv",
        PARSED_DIR / f"{gse}.tsv",
        PARSED_DIR / f"{gse}_parsed.tsv",
    ]

    for p in candidates:
        if p.exists():
            return p

    matches = list(PARSED_DIR.glob(f"{gse}*"))

    for p in matches:
        if p.suffix.lower() in [".csv", ".tsv", ".txt"]:
            return p

    return None


# ------------------------------------------------------------
# LOAD EXPRESSION
# ------------------------------------------------------------


def load_expression(gse):

    path = find_expression_file(gse)

    if path is None:
        raise FileNotFoundError(f"Expression file not found: {gse}")

    try:
        df = pd.read_csv(path, sep=None, engine="python", index_col=0)
    except Exception:
        df = pd.read_csv(path, sep="\t", index_col=0)

    df.index = df.index.astype(str).map(norm_id)

    df = df[df.index != ""]

    # Remove duplicated probe IDs
    df = df[~df.index.duplicated(keep="first")]

    # Convert expression values
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ------------------------------------------------------------
# LOAD PROBE → GENE MAPPING
# ------------------------------------------------------------


def load_mapping(gse, gpl):

    path = ANNOT_DIR / f"{gse}_{gpl}_probe_gene_mapping.csv"

    if not path.exists():
        raise FileNotFoundError(f"Mapping not found: {path}")

    mapping = pd.read_csv(path, dtype=str)

    required = {"ProbeID", "GeneSymbol"}

    if not required.issubset(mapping.columns):
        raise ValueError(f"Invalid mapping columns: " f"{mapping.columns.tolist()}")

    mapping["ProbeID"] = mapping["ProbeID"].map(norm_id)

    mapping["GeneSymbol"] = mapping["GeneSymbol"].astype(str).str.strip()

    # Remove invalid symbols
    invalid = {"", "nan", "none", "null", "---", "na", "n/a"}

    mapping = mapping[~mapping["GeneSymbol"].str.lower().isin(invalid)]

    # Handle annotations containing multiple symbols
    mapping["GeneSymbol"] = (
        mapping["GeneSymbol"]
        .str.replace(r"\s*///\s*", ";", regex=True)
        .str.replace(r"\s*//\s*", ";", regex=True)
        .str.replace(r"\s*;\s*", ";", regex=True)
    )

    mapping = mapping.drop_duplicates(subset=["ProbeID"])

    return mapping


# ------------------------------------------------------------
# SPLIT MULTIPLE GENE SYMBOLS
# ------------------------------------------------------------


def expand_mapping(mapping):

    rows = []

    for _, row in mapping.iterrows():

        probe = row["ProbeID"]
        genes = row["GeneSymbol"]

        for gene in str(genes).split(";"):

            gene = gene.strip()

            if not gene:
                continue

            if gene.lower() in {"nan", "none", "null", "---", "na", "n/a"}:
                continue

            rows.append({"ProbeID": probe, "GeneSymbol": gene})

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# BUILD GENE MATRIX
# ------------------------------------------------------------


def build_gene_matrix(expression, mapping):

    mapping = mapping[mapping["ProbeID"].isin(expression.index)].copy()

    if mapping.empty:
        raise ValueError("No probe IDs overlap expression matrix")

    mapping = expand_mapping(mapping)

    # Keep only valid expression probes
    mapping = mapping[mapping["ProbeID"].isin(expression.index)]

    # --------------------------------------------------------
    # Attach gene symbol
    # --------------------------------------------------------

    expr = expression.loc[mapping["ProbeID"]].copy()

    expr["GeneSymbol"] = mapping["GeneSymbol"].values

    # --------------------------------------------------------
    # Median probe summarization
    # --------------------------------------------------------

    gene_matrix = expr.groupby("GeneSymbol").median(numeric_only=True)

    gene_matrix = gene_matrix.replace([np.inf, -np.inf], np.nan)

    # Remove genes completely missing
    gene_matrix = gene_matrix.dropna(axis=0, how="all")

    # Remove genes containing no usable samples
    gene_matrix = gene_matrix.dropna(
        axis=0, thresh=max(1, int(gene_matrix.shape[1] * 0.5))
    )

    return gene_matrix, mapping


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

summary = []

print("=" * 80)
print("STEP 6 — BUILD GENE-LEVEL EXPRESSION MATRICES")
print("=" * 80)

for gse, gpl in DATASETS.items():

    print(f"\nProcessing {gse} → {gpl}")

    try:

        # --------------------------------------------
        # Load expression
        # --------------------------------------------

        expression = load_expression(gse)

        print(
            f"  Probe matrix: "
            f"{expression.shape[0]:,} probes × "
            f"{expression.shape[1]:,} samples"
        )

        # --------------------------------------------
        # Load annotation
        # --------------------------------------------

        mapping = load_mapping(gse, gpl)

        print(f"  Mapping rows: " f"{len(mapping):,}")

        # --------------------------------------------
        # Build gene matrix
        # --------------------------------------------

        gene_matrix, used_mapping = build_gene_matrix(expression, mapping)

        print(
            f"  Gene matrix: "
            f"{gene_matrix.shape[0]:,} genes × "
            f"{gene_matrix.shape[1]:,} samples"
        )

        # --------------------------------------------
        # Save
        # --------------------------------------------

        out_file = OUT_DIR / f"{gse}_gene_expression.csv"

        gene_matrix.to_csv(out_file, index_label="GeneSymbol")

        mapping_file = OUT_DIR / f"{gse}_used_probe_gene_mapping.csv"

        used_mapping.to_csv(mapping_file, index=False)

        # --------------------------------------------
        # Statistics
        # --------------------------------------------

        summary.append(
            {
                "GSE": gse,
                "GPL": gpl,
                "Probe_Features": expression.shape[0],
                "Samples": expression.shape[1],
                "Mapped_Probes": len(used_mapping),
                "Genes": gene_matrix.shape[0],
                "Missing_Values": int(gene_matrix.isna().sum().sum()),
                "Status": "PASS",
            }
        )

    except Exception as e:

        print(f"  ERROR: {type(e).__name__}: {e}")

        summary.append(
            {
                "GSE": gse,
                "GPL": gpl,
                "Probe_Features": 0,
                "Samples": 0,
                "Mapped_Probes": 0,
                "Genes": 0,
                "Missing_Values": 0,
                "Status": f"ERROR_{type(e).__name__}",
            }
        )


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

summary_df = pd.DataFrame(summary)

summary_df.to_csv(OUT_DIR / "step06_gene_expression_summary.csv", index=False)

print("\n")
print("=" * 80)
print("STEP 6 SUMMARY")
print("=" * 80)

print(summary_df.to_string(index=False))

print("\nOutput:")
print(OUT_DIR)

print("=" * 80)

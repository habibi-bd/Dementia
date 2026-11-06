# 09_cross_dataset_replication.py

from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(r"D:\Dementia2")

DE_DIR = BASE / "codes" / "biolysis" / "step08_DE"
OUT_DIR = BASE / "codes" / "biolysis" / "step09_replication"

OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "GSE1297",
    "GSE4757",
    "GSE5281",
    "GSE6834",
    "GSE15222",
    "GSE16759",
    "GSE28146",
    "GSE29378",
    "GSE33000",
    "GSE36980",
    "GSE37263",
    "GSE44770",
]

# ============================================================
# LOAD DE RESULTS
# ============================================================

results = []

for gse in DATASETS:

    file = DE_DIR / f"{gse}_DE.csv"

    if not file.exists():
        print(f"{gse}: missing")
        continue

    df = pd.read_csv(file)

    required = {"GeneSymbol", "logFC", "p_value", "adj_p_value"}

    if not required.issubset(df.columns):
        print(f"{gse}: invalid DE file")
        continue

    df = df[["GeneSymbol", "logFC", "p_value", "adj_p_value"]].copy()

    df["GSE"] = gse

    df["GeneSymbol"] = df["GeneSymbol"].astype(str).str.strip().str.upper()

    df = df[(df["GeneSymbol"] != "") & (df["GeneSymbol"] != "NAN")]

    results.append(df)

    print(f"{gse}: {len(df):,} genes")

if not results:
    raise RuntimeError("No DE results found.")

all_de = pd.concat(results, ignore_index=True)

# ============================================================
# DEFINE PER-DATASET SIGNIFICANCE
# ============================================================

all_de["FDR05"] = all_de["adj_p_value"] < 0.05

all_de["FDR10"] = all_de["adj_p_value"] < 0.10

all_de["FC1"] = all_de["logFC"].abs() >= 1

all_de["FC05"] = all_de["logFC"].abs() >= 0.5

all_de["STRICT"] = all_de["FDR05"] & all_de["FC1"]

all_de["EXPLORATORY"] = all_de["FDR10"] & all_de["FC05"]

all_de["DIRECTION"] = np.where(all_de["logFC"] > 0, "UP", "DOWN")

# ============================================================
# GENE-LEVEL REPLICATION
# ============================================================

gene_rows = []

for gene, g in all_de.groupby("GeneSymbol"):

    n_studies = len(g)

    strict = g[g["STRICT"]]
    exploratory = g[g["EXPLORATORY"]]

    # --------------------------------------------------------
    # Direction among all datasets
    # --------------------------------------------------------

    signs = np.sign(g["logFC"])

    n_up = int((signs > 0).sum())
    n_down = int((signs < 0).sum())

    if n_up >= n_down:
        dominant_direction = "UP"
        direction_count = n_up
    else:
        dominant_direction = "DOWN"
        direction_count = n_down

    direction_consistency = direction_count / n_studies

    # --------------------------------------------------------
    # Significant direction consistency
    # --------------------------------------------------------

    if len(strict) > 0:

        strict_signs = np.sign(strict["logFC"])

        strict_up = int((strict_signs > 0).sum())

        strict_down = int((strict_signs < 0).sum())

        if strict_up >= strict_down:
            strict_direction = "UP"
            strict_direction_count = strict_up
        else:
            strict_direction = "DOWN"
            strict_direction_count = strict_down

        strict_direction_consistency = strict_direction_count / len(strict)

    else:

        strict_direction = "NONE"
        strict_direction_count = 0
        strict_direction_consistency = 0

    # --------------------------------------------------------
    # Effect sizes
    # --------------------------------------------------------

    median_logFC = g["logFC"].median()

    median_abs_logFC = g["logFC"].abs().median()

    # --------------------------------------------------------
    # Minimum FDR
    # --------------------------------------------------------

    min_fdr = g["adj_p_value"].min()

    # --------------------------------------------------------
    # Replication criteria
    # --------------------------------------------------------

    # Strong replication:
    # significant in >=3 datasets
    # and >=75% same direction
    replicated_3 = len(strict) >= 3 and strict_direction_consistency >= 0.75

    # Very strong replication:
    # significant in >=4 datasets
    # and >=75% same direction
    replicated_4 = len(strict) >= 4 and strict_direction_consistency >= 0.75

    # Broad replication:
    # exploratory significance in >=4 datasets
    replicated_exploratory = len(exploratory) >= 4

    gene_rows.append(
        {
            "GeneSymbol": gene,
            "Studies": n_studies,
            "Strict_DE_Studies": len(strict),
            "Exploratory_DE_Studies": len(exploratory),
            "Up_Studies": n_up,
            "Down_Studies": n_down,
            "Direction": dominant_direction,
            "Direction_Consistency": direction_consistency,
            "Strict_Direction": strict_direction,
            "Strict_Direction_Consistency": strict_direction_consistency,
            "Median_logFC": median_logFC,
            "Median_abs_logFC": median_abs_logFC,
            "Minimum_FDR": min_fdr,
            "Replicated_3plus": replicated_3,
            "Replicated_4plus": replicated_4,
            "Exploratory_Replicated_4plus": replicated_exploratory,
        }
    )

replication = pd.DataFrame(gene_rows)

# ============================================================
# SORT
# ============================================================

replication = replication.sort_values(
    [
        "Replicated_4plus",
        "Replicated_3plus",
        "Strict_DE_Studies",
        "Direction_Consistency",
        "Minimum_FDR",
    ],
    ascending=[False, False, False, False, True],
)

# ============================================================
# SAVE COMPLETE REPLICATION TABLE
# ============================================================

replication.to_csv(OUT_DIR / "step09_gene_replication.csv", index=False)

# ============================================================
# STRONG REPLICATION
# ============================================================

strong = replication[replication["Replicated_3plus"]].copy()

strong.to_csv(OUT_DIR / "step09_strong_replicated_genes.csv", index=False)

# ============================================================
# VERY STRONG REPLICATION
# ============================================================

very_strong = replication[replication["Replicated_4plus"]].copy()

very_strong.to_csv(OUT_DIR / "step09_very_strong_replicated_genes.csv", index=False)

# ============================================================
# EXPLORATORY REPLICATION
# ============================================================

exploratory = replication[replication["Exploratory_Replicated_4plus"]].copy()

exploratory.to_csv(OUT_DIR / "step09_exploratory_replicated_genes.csv", index=False)

# ============================================================
# STUDY-BY-GENE MATRIX
# ============================================================

logfc_matrix = all_de.pivot_table(
    index="GeneSymbol", columns="GSE", values="logFC", aggfunc="median"
)

logfc_matrix.to_csv(OUT_DIR / "step09_logFC_matrix.csv")

# ============================================================
# SIGNIFICANCE MATRIX
# ============================================================

sig_matrix = all_de.pivot_table(
    index="GeneSymbol", columns="GSE", values="STRICT", aggfunc="max", fill_value=False
)

sig_matrix.to_csv(OUT_DIR / "step09_strict_DE_matrix.csv")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("STEP 9 SUMMARY")
print("=" * 80)

print(f"Genes evaluated              : {len(replication):,}")

print(f"Replicated >=3 studies       : " f"{replication['Replicated_3plus'].sum():,}")

print(f"Replicated >=4 studies       : " f"{replication['Replicated_4plus'].sum():,}")

print(
    f"Exploratory >=4 studies      : "
    f"{replication['Exploratory_Replicated_4plus'].sum():,}"
)

print("\nTop replicated genes:")

cols = [
    "GeneSymbol",
    "Strict_DE_Studies",
    "Direction",
    "Strict_Direction_Consistency",
    "Median_logFC",
    "Minimum_FDR",
]

print(replication[cols].head(30).to_string(index=False))

print("\nOutput:")
print(OUT_DIR)

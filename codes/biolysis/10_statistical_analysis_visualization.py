# =============================================================================
# STEP 10 — STATISTICAL ANALYSIS + VOLCANO PLOTS + REPLICATED GENE VISUALIZATION
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

BASE = Path(r"D:\Dementia2\codes\biolysis")

DE_DIR = BASE / "step08_DE"
REP_DIR = BASE / "step09_replication"

OUT_DIR = BASE / "step10_statistics_visualization"
VOLCANO_DIR = OUT_DIR / "volcano"
TABLE_DIR = OUT_DIR / "tables"
PLOT_DIR = OUT_DIR / "plots"

for d in [OUT_DIR, VOLCANO_DIR, TABLE_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# DATASETS
# GSE26927 EXCLUDED
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# PARAMETERS
# -----------------------------------------------------------------------------

FDR_CUTOFF = 0.05
FDR_EXPLORATORY = 0.10

LOGFC_STRICT = 1.0
LOGFC_EXPLORATORY = 0.5

TOP_N = 30

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------


def find_column(df, candidates):
    lower = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    for c in df.columns:
        cl = c.lower()
        for candidate in candidates:
            if candidate.lower() in cl:
                return c

    return None


def clean_de_table(path):
    df = pd.read_csv(path)

    gene_col = find_column(df, ["GeneSymbol", "Gene Symbol", "gene_symbol", "Gene"])

    logfc_col = find_column(df, ["logFC", "log2FoldChange", "log_fold_change"])

    p_col = find_column(df, ["p_value", "P.Value", "PValue", "pvalue", "p"])

    fdr_col = find_column(
        df, ["adj_p_value", "adj.P.Val", "FDR", "adjusted_p_value", "padj"]
    )

    if gene_col is None:
        raise ValueError(f"No gene column found in {path}")

    if logfc_col is None:
        raise ValueError(f"No logFC column found in {path}")

    if p_col is None:
        raise ValueError(f"No raw p-value column found in {path}")

    if fdr_col is None:
        raise ValueError(f"No FDR column found in {path}")

    out = pd.DataFrame(
        {
            "GeneSymbol": df[gene_col].astype(str).str.upper().str.strip(),
            "logFC": pd.to_numeric(df[logfc_col], errors="coerce"),
            "PValue": pd.to_numeric(df[p_col], errors="coerce"),
            "FDR": pd.to_numeric(df[fdr_col], errors="coerce"),
        }
    )

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna()

    out = out[out["GeneSymbol"].notna()]
    out = out[out["GeneSymbol"] != ""]
    out = out[out["GeneSymbol"] != "NAN"]

    # Valid probabilities
    out["PValue"] = out["PValue"].clip(lower=np.finfo(float).tiny, upper=1.0)
    out["FDR"] = out["FDR"].clip(lower=np.finfo(float).tiny, upper=1.0)

    # Significance variables
    out["neg_log10_PValue"] = -np.log10(out["PValue"])
    out["neg_log10_FDR"] = -np.log10(out["FDR"])

    out["Significant_FDR05"] = (out["FDR"] < FDR_CUTOFF) & (
        np.abs(out["logFC"]) >= LOGFC_STRICT
    )

    out["Significant_FDR10"] = (out["FDR"] < FDR_EXPLORATORY) & (
        np.abs(out["logFC"]) >= LOGFC_EXPLORATORY
    )

    out["Direction"] = np.where(
        out["logFC"] > 0, "UP", np.where(out["logFC"] < 0, "DOWN", "ZERO")
    )

    return out


# -----------------------------------------------------------------------------
# 1. PROCESS EVERY DATASET
# -----------------------------------------------------------------------------

all_results = []
dataset_summary = []

print("=" * 90)
print("STEP 10 — STATISTICAL ANALYSIS")
print("=" * 90)

for gse in DATASETS:

    path = DE_DIR / f"{gse}_DE.csv"

    if not path.exists():
        print(f"[WARNING] Missing: {path}")
        continue

    df = clean_de_table(path)

    df["GSE"] = gse

    # Save complete statistical table
    df = df[
        [
            "GSE",
            "GeneSymbol",
            "logFC",
            "PValue",
            "FDR",
            "neg_log10_PValue",
            "neg_log10_FDR",
            "Direction",
            "Significant_FDR05",
            "Significant_FDR10",
        ]
    ]

    df.to_csv(TABLE_DIR / f"{gse}_complete_statistics.csv", index=False)

    # -------------------------------------------------------------------------
    # TOP BY FDR
    # -------------------------------------------------------------------------

    top_fdr = df.sort_values(["FDR", "PValue"], ascending=[True, True]).head(TOP_N)

    top_fdr.to_csv(TABLE_DIR / f"{gse}_top{TOP_N}_by_FDR.csv", index=False)

    # -------------------------------------------------------------------------
    # TOP BY ABSOLUTE LOGFC
    # -------------------------------------------------------------------------

    top_fc = (
        df.assign(abs_logFC=np.abs(df["logFC"]))
        .sort_values(["abs_logFC", "FDR"], ascending=[False, True])
        .head(TOP_N)
    )

    top_fc.to_csv(TABLE_DIR / f"{gse}_top{TOP_N}_by_abs_logFC.csv", index=False)

    # -------------------------------------------------------------------------
    # SIGNIFICANT COUNTS
    # -------------------------------------------------------------------------

    fdr05 = df["FDR"] < 0.05
    fdr10 = df["FDR"] < 0.10

    strict = fdr05 & (np.abs(df["logFC"]) >= LOGFC_STRICT)

    exploratory = fdr10 & (np.abs(df["logFC"]) >= LOGFC_EXPLORATORY)

    up_strict = strict & (df["logFC"] > 0)
    down_strict = strict & (df["logFC"] < 0)

    dataset_summary.append(
        {
            "GSE": gse,
            "Genes": len(df),
            "FDR05": int(fdr05.sum()),
            "FDR10": int(fdr10.sum()),
            "Strict_DE": int(strict.sum()),
            "Strict_UP": int(up_strict.sum()),
            "Strict_DOWN": int(down_strict.sum()),
            "Exploratory_DE": int(exploratory.sum()),
            "Median_abs_logFC": float(np.median(np.abs(df["logFC"]))),
            "Max_abs_logFC": float(np.max(np.abs(df["logFC"]))),
            "Min_PValue": float(df["PValue"].min()),
            "Min_FDR": float(df["FDR"].min()),
        }
    )

    all_results.append(df)

    print(
        f"{gse:10s} | "
        f"Genes={len(df):7d} | "
        f"FDR<0.05={fdr05.sum():7d} | "
        f"Strict={strict.sum():6d} | "
        f"UP={up_strict.sum():6d} | "
        f"DOWN={down_strict.sum():6d}"
    )

# -----------------------------------------------------------------------------
# SAVE SUMMARY
# -----------------------------------------------------------------------------

summary = pd.DataFrame(dataset_summary)

summary.to_csv(TABLE_DIR / "step10_dataset_statistics_summary.csv", index=False)

# -----------------------------------------------------------------------------
# 2. VOLCANO PLOTS
# -----------------------------------------------------------------------------

print()
print("=" * 90)
print("GENERATING VOLCANO PLOTS")
print("=" * 90)

for df in all_results:

    gse = df["GSE"].iloc[0]

    plot_df = df.copy()

    plot_df["Category"] = "Not significant"

    plot_df.loc[
        (plot_df["FDR"] < FDR_CUTOFF) & (plot_df["logFC"] >= LOGFC_STRICT), "Category"
    ] = "Upregulated"

    plot_df.loc[
        (plot_df["FDR"] < FDR_CUTOFF) & (plot_df["logFC"] <= -LOGFC_STRICT), "Category"
    ] = "Downregulated"

    plt.figure(figsize=(12, 9))

    sns.scatterplot(
        data=plot_df,
        x="logFC",
        y="neg_log10_FDR",
        hue="Category",
        palette={
            "Not significant": "lightgray",
            "Upregulated": "red",
            "Downregulated": "blue",
        },
        s=20,
        alpha=0.65,
        linewidth=0,
    )

    plt.axvline(LOGFC_STRICT, linestyle="--", linewidth=1)

    plt.axvline(-LOGFC_STRICT, linestyle="--", linewidth=1)

    plt.axhline(-np.log10(FDR_CUTOFF), linestyle="--", linewidth=1)

    # Label strongest genes
    label_df = (
        plot_df[
            (plot_df["FDR"] < FDR_CUTOFF) & (np.abs(plot_df["logFC"]) >= LOGFC_STRICT)
        ]
        .sort_values("FDR")
        .head(10)
    )

    for _, row in label_df.iterrows():

        plt.text(row["logFC"], row["neg_log10_FDR"], row["GeneSymbol"], fontsize=8)

    plt.title(f"{gse} — Alzheimer vs Control\nVolcano Plot")

    plt.xlabel("log2 Fold Change")
    plt.ylabel("-log10(FDR)")
    plt.legend(title="Category")
    plt.tight_layout()

    plt.savefig(VOLCANO_DIR / f"{gse}_volcano.png", dpi=300, bbox_inches="tight")

    plt.savefig(VOLCANO_DIR / f"{gse}_volcano.pdf", bbox_inches="tight")

    plt.close()

# -----------------------------------------------------------------------------
# 3. CROSS-STUDY REPLICATED GENE ANALYSIS
# -----------------------------------------------------------------------------

rep_path = REP_DIR / "step09_gene_replication.csv"

if rep_path.exists():

    rep = pd.read_csv(rep_path)

    rep["GeneSymbol"] = rep["GeneSymbol"].astype(str).str.upper().str.strip()

    # -------------------------------------------------------------------------
    # TOP REPLICATED GENES
    # -------------------------------------------------------------------------

    rep = rep.sort_values(
        [
            "Strict_DE_Studies",
            "Strict_Direction_Consistency",
            "Minimum_FDR",
            "Median_abs_logFC",
        ],
        ascending=[False, False, True, False],
    )

    rep.to_csv(TABLE_DIR / "step10_replicated_genes_ranked.csv", index=False)

    # -------------------------------------------------------------------------
    # STRONG REPLICATION
    # -------------------------------------------------------------------------

    strong = rep[
        (rep["Strict_DE_Studies"] >= 3) & (rep["Strict_Direction_Consistency"] >= 0.75)
    ].copy()

    strong.to_csv(TABLE_DIR / "step10_strong_replicated_genes.csv", index=False)

    # -------------------------------------------------------------------------
    # VERY STRONG
    # -------------------------------------------------------------------------

    very_strong = rep[
        (rep["Strict_DE_Studies"] >= 4) & (rep["Strict_Direction_Consistency"] >= 0.75)
    ].copy()

    very_strong.to_csv(
        TABLE_DIR / "step10_very_strong_replicated_genes.csv", index=False
    )

    print()
    print("Strong replicated genes:", len(strong))
    print("Very strong replicated genes:", len(very_strong))

# -----------------------------------------------------------------------------
# 4. LOGFC HEATMAP
# -----------------------------------------------------------------------------

logfc_path = REP_DIR / "step09_logFC_matrix.csv"

if logfc_path.exists():

    logfc = pd.read_csv(logfc_path)

    gene_col = "GeneSymbol"

    logfc[gene_col] = logfc[gene_col].astype(str).str.upper().str.strip()

    # Keep genes replicated in >=2 studies
    study_columns = [c for c in logfc.columns if c != gene_col]

    logfc["Studies_with_Data"] = logfc[study_columns].notna().sum(axis=1)

    heatmap_df = (
        logfc[logfc["Studies_with_Data"] >= 2]
        .drop(columns=["Studies_with_Data"])
        .set_index(gene_col)
    )

    # Top 50 by maximum absolute effect
    ranking = heatmap_df.abs().max(axis=1).sort_values(ascending=False).head(50).index

    heatmap_df = heatmap_df.loc[ranking]

    plt.figure(figsize=(14, 14))

    sns.heatmap(
        heatmap_df,
        cmap="vlag",
        center=0,
        linewidths=0.2,
        xticklabels=True,
        yticklabels=True,
    )

    plt.title("Cross-Study log2 Fold Change\nTop 50 Replicated Genes")

    plt.xlabel("GEO Dataset")
    plt.ylabel("Gene")

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "top50_replicated_genes_logFC_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        PLOT_DIR / "top50_replicated_genes_logFC_heatmap.pdf", bbox_inches="tight"
    )

    plt.close()

# -----------------------------------------------------------------------------
# 5. REPLICATION COUNT DISTRIBUTION
# -----------------------------------------------------------------------------

if rep_path.exists():

    rep = pd.read_csv(rep_path)

    plt.figure(figsize=(10, 7))

    counts = rep["Strict_DE_Studies"].value_counts().sort_index()

    plt.bar(counts.index.astype(str), counts.values)

    plt.xlabel("Number of datasets with strict DE")
    plt.ylabel("Number of genes")

    plt.title("Distribution of Cross-Study Gene Replication")

    plt.tight_layout()

    plt.savefig(PLOT_DIR / "replication_distribution.png", dpi=300, bbox_inches="tight")

    plt.savefig(PLOT_DIR / "replication_distribution.pdf", bbox_inches="tight")

    plt.close()

# -----------------------------------------------------------------------------
# 6. TOP REPLICATED GENES — EFFECT SIZE
# -----------------------------------------------------------------------------

if rep_path.exists():

    rep = pd.read_csv(rep_path)

    top = (
        rep[rep["Strict_DE_Studies"] >= 2]
        .sort_values(
            ["Strict_DE_Studies", "Strict_Direction_Consistency", "Median_abs_logFC"],
            ascending=False,
        )
        .head(30)
        .copy()
    )

    plt.figure(figsize=(12, 10))

    top = top.sort_values("Median_logFC")

    plt.barh(top["GeneSymbol"], top["Median_logFC"])

    plt.axvline(0, linestyle="-", linewidth=1)

    plt.xlabel("Median log2 Fold Change")
    plt.ylabel("Gene")

    plt.title("Top Cross-Study Replicated Genes")

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "top_replicated_genes_effect_size.png", dpi=300, bbox_inches="tight"
    )

    plt.savefig(PLOT_DIR / "top_replicated_genes_effect_size.pdf", bbox_inches="tight")

    plt.close()

# -----------------------------------------------------------------------------
# 7. FINAL REPORT
# -----------------------------------------------------------------------------

print()
print("=" * 90)
print("STEP 10 COMPLETE")
print("=" * 90)

print()
print("Output directory:")
print(OUT_DIR)

print()
print("Important files:")
print(TABLE_DIR / "step10_dataset_statistics_summary.csv")
print(TABLE_DIR / "step10_replicated_genes_ranked.csv")
print(TABLE_DIR / "step10_strong_replicated_genes.csv")
print(TABLE_DIR / "step10_very_strong_replicated_genes.csv")
print(PLOT_DIR / "top50_replicated_genes_logFC_heatmap.png")
print(PLOT_DIR / "replication_distribution.png")
print(PLOT_DIR / "top_replicated_genes_effect_size.png")

print()
print("Volcano plots:")
for gse in DATASETS:
    print(VOLCANO_DIR / f"{gse}_volcano.png")

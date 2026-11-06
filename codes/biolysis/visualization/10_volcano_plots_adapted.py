"""
COMPLETE BIOINFORMATICS ANALYSIS FOR DEMENTIA DATASETS
FIXED VERSION - Proper column handling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings

warnings.filterwarnings("ignore")

# Define paths
BASE = Path(r"D:\Dementia2\codes\biolysis")
DE = BASE / "step08_DE"
OUT = BASE / "dementia_analysis_results"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("COMPLETE DEMENTIA GENE EXPRESSION ANALYSIS")
print("=" * 80)
print(f"Output directory: {OUT}")

# List of datasets
GSES = [
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

# Step 1: Load all DE results
print("\n" + "=" * 80)
print("STEP 1: LOADING ALL DIFFERENTIAL EXPRESSION RESULTS")
print("=" * 80)

all_results = {}
for gse in GSES:
    file_path = DE / f"{gse}_DE.csv"
    if file_path.exists():
        df = pd.read_csv(file_path)
        all_results[gse] = df
        print(f"✅ Loaded {gse}: {len(df)} genes")
    else:
        print(f"❌ {gse}_DE.csv not found")

if not all_results:
    print("\n❌ No DE files found! Please run differential expression analysis first.")
    exit(1)

# Step 2: Identify column names
print("\n" + "=" * 80)
print("STEP 2: IDENTIFYING COLUMN NAMES")
print("=" * 80)

# Get column names from first dataset
sample_df = list(all_results.values())[0]
print(f"Columns in DE files: {sample_df.columns.tolist()}")


# Auto-detect column names
def find_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    for col in df.columns:
        for name in possible_names:
            if col.lower() == name.lower():
                return col
    return None


gene_col = find_column(sample_df, ["GeneSymbol", "Gene_Symbol", "Gene", "Symbol"])
logfc_col = find_column(sample_df, ["logFC", "log2FC", "Log2FC", "log2FoldChange"])
pval_col = find_column(sample_df, ["PValue", "P.Value", "p_value", "pval", "P"])
fdr_col = find_column(
    sample_df, ["FDR", "adj.P.Val", "adj_p_val", "padj", "adj_p_value"]
)

# If FDR not found, use p-value
if fdr_col is None:
    fdr_col = pval_col
    print("⚠️  FDR column not found, using p-values as fallback")

print(f"\nUsing columns:")
print(f"  Gene: {gene_col}")
print(f"  logFC: {logfc_col}")
print(f"  P-value: {pval_col}")
print(f"  FDR: {fdr_col}")

# Step 3: Meta-analysis - Combine results across datasets
print("\n" + "=" * 80)
print("STEP 3: META-ANALYSIS ACROSS ALL DATASETS")
print("=" * 80)

# Create combined dataframe with all genes from all datasets
all_genes = set()
for df in all_results.values():
    if gene_col in df.columns:
        # Clean gene names and add to set
        genes = df[gene_col].astype(str).str.upper().str.strip()
        all_genes.update(genes)

print(f"Total unique genes across all datasets: {len(all_genes)}")

# Create matrices for meta-analysis with NaN for missing values
logfc_matrix = pd.DataFrame(index=sorted(all_genes))
pval_matrix = pd.DataFrame(index=sorted(all_genes))
fdr_matrix = pd.DataFrame(index=sorted(all_genes))
direction_matrix = pd.DataFrame(index=sorted(all_genes))

for gse, df in all_results.items():
    if gene_col not in df.columns:
        continue

    # Clean gene names
    df_clean = df.copy()
    df_clean[gene_col] = df_clean[gene_col].astype(str).str.upper().str.strip()
    df_clean = df_clean.drop_duplicates(gene_col)
    df_clean = df_clean.set_index(gene_col)

    # Store logFC and p-values
    logfc_matrix[gse] = df_clean[logfc_col]
    pval_matrix[gse] = df_clean[pval_col]
    fdr_matrix[gse] = df_clean[fdr_col]

    # Calculate direction: 1 for up, -1 for down, 0 for not significant or missing
    direction_series = pd.Series(0, index=df_clean.index)
    direction_series[(df_clean[logfc_col] > 1) & (df_clean[pval_col] < 0.05)] = 1
    direction_series[(df_clean[logfc_col] < -1) & (df_clean[pval_col] < 0.05)] = -1

    # Reindex to match all genes (fills missing with NaN)
    direction_matrix[gse] = direction_series.reindex(sorted(all_genes))

print("✅ Matrices created successfully")

# Step 4: Calculate meta-statistics
print("\n" + "=" * 80)
print("STEP 4: CALCULATING META-STATISTICS")
print("=" * 80)

meta_stats = []
for gene in logfc_matrix.index:
    # Get values across datasets (drop NaN)
    logfc_values = logfc_matrix.loc[gene].dropna().values
    p_values = pval_matrix.loc[gene].dropna().values
    directions = direction_matrix.loc[gene].dropna().values

    if len(logfc_values) > 0:
        # Mean logFC
        mean_logfc = np.mean(logfc_values)

        # Median logFC (robust to outliers)
        median_logfc = np.median(logfc_values)

        # Number of significant studies (p < 0.05)
        n_sig = np.sum(p_values < 0.05)

        # Number of up/down regulated
        n_up = np.sum((logfc_values > 1) & (p_values < 0.05))
        n_down = np.sum((logfc_values < -1) & (p_values < 0.05))

        # Direction consistency (proportion of studies with same direction)
        # Only consider studies with significant changes
        sig_directions = directions[(directions != 0)]
        if len(sig_directions) > 0:
            n_positive = np.sum(sig_directions == 1)
            n_negative = np.sum(sig_directions == -1)
            consistency = max(n_positive, n_negative) / len(sig_directions)
        else:
            consistency = 0

        # Combine p-values using Fisher's method
        if len(p_values) > 0:
            # Avoid -inf from very small p-values
            p_clipped = np.clip(p_values, 1e-300, 1)
            fisher_chi2 = -2 * np.sum(np.log(p_clipped))
            combined_p = 1 - stats.chi2.cdf(fisher_chi2, 2 * len(p_values))
        else:
            combined_p = 1

        meta_stats.append(
            {
                "Gene": gene,
                "n_studies": len(logfc_values),
                "n_sig": n_sig,
                "n_up": n_up,
                "n_down": n_down,
                "mean_logFC": mean_logfc,
                "median_logFC": median_logfc,
                "consistency": consistency,
                "combined_p": combined_p,
            }
        )

meta_df = pd.DataFrame(meta_stats)

# Sort by number of significant studies, then consistency, then absolute logFC
# Create a separate column for absolute logFC
meta_df["abs_mean_logFC"] = np.abs(meta_df["mean_logFC"])
meta_df = meta_df.sort_values(
    ["n_sig", "consistency", "abs_mean_logFC"], ascending=[False, False, False]
)

print(f"✅ Meta-statistics calculated for {len(meta_df)} genes")

# Step 5: Identify top genes associated with dementia
print("\n" + "=" * 80)
print("STEP 5: TOP GENES ASSOCIATED WITH DEMENTIA")
print("=" * 80)

# Criteria for top genes
top_criteria = {
    "min_studies": 3,  # At least 3 datasets
    "min_sig": 2,  # Significant in at least 2 datasets
    "min_consistency": 0.7,  # 70% consistent direction
    "min_abs_logfc": 0.5,  # Mean |logFC| > 0.5
}

# Filter genes based on criteria
filtered_genes = meta_df[
    (meta_df["n_studies"] >= top_criteria["min_studies"])
    & (meta_df["n_sig"] >= top_criteria["min_sig"])
    & (meta_df["consistency"] >= top_criteria["min_consistency"])
    & (meta_df["abs_mean_logFC"] >= top_criteria["min_abs_logfc"])
].copy()

# Get top 20 genes
top_genes = filtered_genes.head(20)

print(f"\nTop {len(top_genes)} genes identified as dementia-associated:")
print("=" * 80)
if len(top_genes) > 0:
    print(
        top_genes[
            ["Gene", "n_studies", "n_sig", "mean_logFC", "consistency"]
        ].to_string(index=False)
    )
else:
    print("No genes met the criteria. Trying with relaxed thresholds...")

    # Relax criteria if no genes found
    relaxed_criteria = {
        "min_studies": 2,
        "min_sig": 1,
        "min_consistency": 0.6,
        "min_abs_logfc": 0.3,
    }

    filtered_genes = meta_df[
        (meta_df["n_studies"] >= relaxed_criteria["min_studies"])
        & (meta_df["n_sig"] >= relaxed_criteria["min_sig"])
        & (meta_df["consistency"] >= relaxed_criteria["min_consistency"])
        & (meta_df["abs_mean_logFC"] >= relaxed_criteria["min_abs_logfc"])
    ].copy()

    top_genes = filtered_genes.head(20)
    print(f"\nTop {len(top_genes)} genes with relaxed criteria:")
    if len(top_genes) > 0:
        print(
            top_genes[
                ["Gene", "n_studies", "n_sig", "mean_logFC", "consistency"]
            ].to_string(index=False)
        )

# Step 6: Save results
print("\n" + "=" * 80)
print("STEP 6: SAVING RESULTS")
print("=" * 80)

# Save top genes
if len(top_genes) > 0:
    top_genes.to_csv(OUT / "top_dementia_genes.csv", index=False)
    print(f"✅ Top genes saved: {OUT / 'top_dementia_genes.csv'}")
else:
    print("⚠️  No top genes found to save")

# Save meta-analysis results
meta_df.to_csv(OUT / "meta_analysis_results.csv", index=False)
print(f"✅ Meta-analysis results saved: {OUT / 'meta_analysis_results.csv'}")

# Step 7: Create volcano plots
print("\n" + "=" * 80)
print("STEP 7: CREATING VOLCANO PLOTS")
print("=" * 80)

volcano_dir = OUT / "volcano_plots"
volcano_dir.mkdir(exist_ok=True)

if len(top_genes) > 0:
    top_gene_list = top_genes["Gene"].tolist()
else:
    top_gene_list = []

for gse, df in all_results.items():
    if gene_col not in df.columns:
        continue

    plt.figure(figsize=(12, 10))

    # Clean data
    df_clean = df.copy()
    df_clean[gene_col] = df_clean[gene_col].astype(str).str.upper().str.strip()
    df_clean["-log10_p"] = -np.log10(df_clean[pval_col].clip(lower=1e-300))

    # Color by significance
    df_clean["Color"] = "Not Significant"
    df_clean.loc[(df_clean[logfc_col] > 1) & (df_clean[pval_col] < 0.05), "Color"] = (
        "Upregulated"
    )
    df_clean.loc[(df_clean[logfc_col] < -1) & (df_clean[pval_col] < 0.05), "Color"] = (
        "Downregulated"
    )

    # Plot all genes
    colors = {
        "Upregulated": "#e41a1c",
        "Downregulated": "#377eb8",
        "Not Significant": "#cccccc",
    }
    for color, label in colors.items():
        subset = df_clean[df_clean["Color"] == color]
        if len(subset) > 0:
            plt.scatter(
                subset[logfc_col],
                subset["-log10_p"],
                c=colors[color],
                label=color,
                alpha=0.5,
                s=10,
            )

    # Highlight top genes
    if top_gene_list:
        top_in_dataset = df_clean[df_clean[gene_col].isin(top_gene_list)]
        for _, row in top_in_dataset.iterrows():
            plt.scatter(
                row[logfc_col],
                row["-log10_p"],
                c="gold",
                s=100,
                edgecolors="black",
                linewidth=1,
                alpha=0.8,
            )
            plt.annotate(
                row[gene_col],
                (row[logfc_col], row["-log10_p"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                weight="bold",
                alpha=0.8,
            )

    # Threshold lines
    plt.axhline(y=-np.log10(0.05), color="black", linestyle="--", alpha=0.5)
    plt.axvline(x=1, color="black", linestyle="--", alpha=0.5)
    plt.axvline(x=-1, color="black", linestyle="--", alpha=0.5)

    # Labels and title
    plt.xlabel("log₂ Fold Change", fontsize=14)
    plt.ylabel("-log₁₀ p-value", fontsize=14)
    plt.title(f"{gse} - Volcano Plot with Top Dementia Genes Highlighted", fontsize=16)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)

    # Statistics box
    n_up = len(df_clean[(df_clean[logfc_col] > 1) & (df_clean[pval_col] < 0.05)])
    n_down = len(df_clean[(df_clean[logfc_col] < -1) & (df_clean[pval_col] < 0.05)])
    plt.text(
        0.02,
        0.98,
        f"Up: {n_up}, Down: {n_down}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(volcano_dir / f"{gse}_volcano_highlighted.png", dpi=300)
    plt.close()

    print(f"✅ {gse} volcano plot created")

# Step 8: Create heatmap
print("\n" + "=" * 80)
print("STEP 8: CREATING HEATMAP")
print("=" * 80)

if len(top_gene_list) > 0:
    # Create heatmap for top genes
    plt.figure(figsize=(14, 12))

    # Get expression data for top genes
    top_logfc = logfc_matrix.loc[top_gene_list]

    # Fill NaN with 0 for visualization
    heatmap_data = top_logfc.fillna(0)

    # Plot heatmap
    sns.heatmap(
        heatmap_data,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "log₂ Fold Change"},
        xticklabels=True,
        yticklabels=True,
    )

    plt.title(
        "Expression Pattern of Top Dementia-Associated Genes\nAcross All Datasets",
        fontsize=16,
    )
    plt.xlabel("Genes", fontsize=12)
    plt.ylabel("Datasets", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "top_genes_heatmap.png", dpi=300)
    plt.close()
    print("✅ Heatmap created")

# Step 9: Create summary bar plot
print("\n" + "=" * 80)
print("STEP 9: CREATING SUMMARY VISUALIZATION")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Number of significant genes per dataset
ax1 = axes[0, 0]
sig_counts = []
for gse, df in all_results.items():
    if gene_col in df.columns:
        n_sig = len(df[(df[pval_col] < 0.05) & (np.abs(df[logfc_col]) > 1)])
        sig_counts.append({"Dataset": gse, "Significant": n_sig})

sig_df = pd.DataFrame(sig_counts)
ax1.bar(sig_df["Dataset"], sig_df["Significant"], color="steelblue")
ax1.set_xlabel("Dataset")
ax1.set_ylabel("Number of Significant Genes")
ax1.set_title("Differentially Expressed Genes per Dataset")
ax1.tick_params(axis="x", rotation=45)

# Plot 2: Top 10 genes by number of significant studies
ax2 = axes[0, 1]
if len(meta_df) > 0:
    top10 = meta_df.head(10)
    ax2.barh(top10["Gene"], top10["n_sig"], color="coral")
    ax2.set_xlabel("Number of Studies with Significant Expression")
    ax2.set_title("Top Genes by Replication Across Studies")

# Plot 3: Consistency vs effect size for top genes
ax3 = axes[1, 0]
if len(top_genes) > 0:
    ax3.scatter(
        top_genes["mean_logFC"],
        top_genes["consistency"],
        s=top_genes["n_sig"] * 50,
        alpha=0.6,
        c="purple",
    )
    for _, row in top_genes.iterrows():
        ax3.annotate(
            row["Gene"], (row["mean_logFC"], row["consistency"]), fontsize=8, alpha=0.7
        )
    ax3.set_xlabel("Mean log₂ Fold Change")
    ax3.set_ylabel("Direction Consistency")
    ax3.set_title("Effect Size vs Consistency for Top Genes")
    ax3.axvline(x=0, color="black", linestyle="--", alpha=0.3)

# Plot 4: Combined p-value distribution
ax4 = axes[1, 1]
if len(meta_df) > 0:
    ax4.hist(meta_df["combined_p"], bins=30, color="green", alpha=0.7)
    ax4.set_xlabel("Combined p-value (Fisher's method)")
    ax4.set_ylabel("Number of Genes")
    ax4.set_title("Distribution of Combined p-values")
    ax4.axvline(x=0.05, color="red", linestyle="--", label="p=0.05")
    ax4.legend()

plt.tight_layout()
plt.savefig(OUT / "summary_plots.png", dpi=300)
plt.close()
print("✅ Summary plots created")

# Step 10: Create final report
print("\n" + "=" * 80)
print("STEP 10: FINAL REPORT")
print("=" * 80)

report_file = OUT / "dementia_analysis_report.txt"
with open(report_file, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("DEMENTIA GENE EXPRESSION ANALYSIS - FINAL REPORT\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Number of Datasets Analyzed: {len(all_results)}\n")
    f.write(f"Total Unique Genes: {len(all_genes)}\n")
    f.write(
        f"Genes Meeting Criteria: {len(filtered_genes) if len(filtered_genes) > 0 else 0}\n\n"
    )

    f.write("=" * 80 + "\n")
    f.write("TOP GENES ASSOCIATED WITH DEMENTIA\n")
    f.write("=" * 80 + "\n\n")

    if len(top_genes) > 0:
        for i, (idx, row) in enumerate(top_genes.iterrows(), 1):
            f.write(
                f"{i:2d}. {row['Gene']:15s} | Studies: {int(row['n_studies']):2d} | "
            )
            f.write(f"Significant: {int(row['n_sig']):2d} | ")
            f.write(f"logFC: {row['mean_logFC']:6.3f} | ")
            f.write(f"Consistency: {row['consistency']:.2%}\n")
    else:
        f.write("No genes met the criteria. Please check your data.\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("BIOLOGICAL INTERPRETATION\n")
    f.write("=" * 80 + "\n\n")

    if len(top_genes) > 0:
        f.write("These genes show consistent differential expression across multiple\n")
        f.write(
            "independent dementia datasets, indicating they are robustly associated\n"
        )
        f.write("with the disease process.\n\n")

        f.write("Key biological functions represented in the top genes:\n")

        # Check for known dementia genes
        known_genes = {
            "SYT1": "Synaptic function",
            "CALM1": "Calcium signaling",
            "ITPKB": "Calcium signaling",
            "CXCR4": "Neuroinflammation",
            "TGFB1": "Neuroinflammation",
            "ITGB1": "Cell adhesion",
            "BAG3": "Protein modification",
            "NOTCH1": "Signaling pathways",
        }

        for gene in top_genes["Gene"].head(10):
            if gene in known_genes:
                f.write(f"  - {gene}: {known_genes[gene]}\n")

print(f"✅ Report saved: {report_file}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nAll results saved to: {OUT}")
print("\nKey files:")
print(f"  - Top genes: top_dementia_genes.csv")
print(f"  - Meta-analysis: meta_analysis_results.csv")
print(f"  - Volcano plots: volcano_plots/")
print(f"  - Heatmap: top_genes_heatmap.png")
print(f"  - Summary plots: summary_plots.png")
print(f"  - Final report: dementia_analysis_report.txt")
print("=" * 80)

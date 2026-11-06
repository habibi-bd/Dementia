"""
STEP 11: QQ Plots for Differential Expression
Purpose: Create QQ plots to assess the distribution of p-values
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot

BASE = Path(r"D:\Dementia2\codes\biolysis")
DE = BASE / "step08_DE"
OUT = BASE / "step10_visualization" / "11_QQ_plots"
OUT.mkdir(parents=True, exist_ok=True)

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

print("=" * 60)
print("CREATING QQ PLOTS")
print("=" * 60)

# First, identify the correct column name from the first DE file
sample_file = DE / f"{GSES[0]}_DE.csv"
if not sample_file.exists():
    print(f"❌ No DE file found: {sample_file}")
    print("Please check if step08_DE directory exists and contains DE files")
    exit(1)

sample_df = pd.read_csv(sample_file)
print(f"\nColumns in DE files: {sample_df.columns.tolist()}")

# Find p-value column
pval_col = None
for col in sample_df.columns:
    col_lower = col.lower()
    if "p" in col_lower and ("val" in col_lower or "value" in col_lower):
        pval_col = col
        break
    elif col_lower == "p":
        pval_col = col
        break

if pval_col is None:
    # Check if there's a column with just 'p' or similar
    for col in sample_df.columns:
        if col.lower() == "p" or col.lower() == "pvalue":
            pval_col = col
            break

if pval_col is None:
    print(f"\n❌ Could not find p-value column!")
    print(f"Available columns: {sample_df.columns.tolist()}")
    exit(1)

print(f"\nUsing p-value column: '{pval_col}'")

# Find logFC column for additional info
logfc_col = None
for col in sample_df.columns:
    if "log" in col.lower() and (
        "fc" in col.lower() or "fold" in col.lower() or "change" in col.lower()
    ):
        logfc_col = col
        break

if logfc_col:
    print(f"Using logFC column: '{logfc_col}'")

print("\n" + "-" * 40)

# Create QQ plots for each dataset
successful = 0
failed = 0

for gse in GSES:

    f = DE / f"{gse}_DE.csv"

    if not f.exists():
        print(f"⚠️  File not found: {gse}_DE.csv")
        failed += 1
        continue

    try:
        d = pd.read_csv(f)

        # Check if p-value column exists in this file
        if pval_col not in d.columns:
            print(f"⚠️  Column '{pval_col}' not found in {gse}")
            # Try to find alternative
            alt_col = None
            for col in d.columns:
                if "p" in col.lower() and (
                    "val" in col.lower() or "value" in col.lower()
                ):
                    alt_col = col
                    break
            if alt_col:
                print(f"   Using alternative: '{alt_col}'")
                pval_col = alt_col
            else:
                print(f"   ❌ Skipping {gse}")
                failed += 1
                continue

        # Extract p-values, clip to avoid extreme values
        p_values = d[pval_col].clip(lower=1e-300, upper=1).dropna().values

        if len(p_values) == 0:
            print(f"⚠️  No valid p-values for {gse}")
            failed += 1
            continue

        # Sort p-values and calculate expected values
        p_sorted = np.sort(p_values)
        expected = -np.log10((np.arange(1, len(p_sorted) + 1) - 0.5) / len(p_sorted))
        observed = -np.log10(p_sorted)

        # Create QQ plot
        plt.figure(figsize=(8, 8))

        plt.scatter(expected, observed, s=8, alpha=0.5, color="blue")

        # Add diagonal line
        lim = max(expected.max(), observed.max()) * 1.05
        plt.plot(
            [0, lim],
            [0, lim],
            linestyle="--",
            color="red",
            alpha=0.7,
            label="Expected distribution",
        )

        # Add labels
        plt.xlabel("Expected -log10(p)", fontsize=12)
        plt.ylabel("Observed -log10(p)", fontsize=12)
        plt.title(f"{gse} — QQ Plot", fontsize=14)

        # Add grid
        plt.grid(True, alpha=0.3)

        # Add legend
        plt.legend()

        # Add annotation with number of genes
        n_genes = len(p_sorted)
        plt.text(
            0.02,
            0.98,
            f"Genes: {n_genes}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()

        # Save plot
        plt.savefig(OUT / f"{gse}_QQ.png", dpi=300)
        plt.savefig(OUT / f"{gse}_QQ.pdf")
        plt.close()

        print(f"✅ Created QQ plot for {gse} (genes: {n_genes})")
        successful += 1

    except Exception as e:
        print(f"❌ Error creating QQ plot for {gse}: {e}")
        failed += 1
        continue

print("\n" + "=" * 60)
print("QQ PLOTS COMPLETE")
print("=" * 60)
print(f"✅ Successful: {successful}")
print(f"❌ Failed: {failed}")
print(f"Output directory: {OUT}")
print("=" * 60)

# from pathlib import Path
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# BASE = Path(r"D:\Dementia2\codes\biolysis")

# REP = BASE / "step09_replication"
# DE = BASE / "step08_DE"
# OUT = BASE / "step10_visualization" / "17_top_gene_crossstudy"

# OUT.mkdir(parents=True, exist_ok=True)

# GSES = [
#     "GSE1297",
#     "GSE4757",
#     "GSE5281",
#     "GSE6834",
#     "GSE15222",
#     "GSE16759",
#     "GSE28146",
#     "GSE29378",
#     "GSE33000",
#     "GSE36980",
#     "GSE37263",
#     "GSE44770",
# ]

# rep_file = REP / "step09_gene_replication.csv"

# rep = pd.read_csv(rep_file)

# # Strong replicated genes
# genes = (
#     rep[rep["Strict_DE_Studies"] >= 2]
#     .sort_values(
#         ["Strict_DE_Studies", "Strict_Direction_Consistency", "Minimum_FDR"],
#         ascending=[False, False, True],
#     )
#     .head(50)["GeneSymbol"]
#     .astype(str)
#     .str.upper()
#     .tolist()
# )

# # -------------------------------------------------------------------------
# # BUILD LOGFC MATRIX
# # -------------------------------------------------------------------------

# logfc = pd.DataFrame(index=genes)
# pvalues = pd.DataFrame(index=genes)
# fdr = pd.DataFrame(index=genes)
# direction = pd.DataFrame(index=genes)

# for gse in GSES:

#     f = DE / f"{gse}_DE.csv"

#     if not f.exists():
#         continue

#     d = pd.read_csv(f)

#     d["GeneSymbol"] = d["GeneSymbol"].astype(str).str.upper().str.strip()

#     d = d.drop_duplicates("GeneSymbol")
#     d = d.set_index("GeneSymbol")

#     logfc[gse] = d["logFC"].reindex(genes)
#     pvalues[gse] = d["PValue"].reindex(genes)
#     fdr[gse] = d["FDR"].reindex(genes)

#     direction[gse] = np.where(
#         d["logFC"].reindex(genes) > 0, 1, np.where(d["logFC"].reindex(genes) < 0, -1, 0)
#     )

# # -------------------------------------------------------------------------
# # SAVE TABLES
# # -------------------------------------------------------------------------

# logfc.to_csv(OUT / "top50_replicated_logFC.csv")

# pvalues.to_csv(OUT / "top50_replicated_pvalues.csv")

# fdr.to_csv(OUT / "top50_replicated_FDR.csv")

# direction.to_csv(OUT / "top50_replicated_direction.csv")

# # -------------------------------------------------------------------------
# # HEATMAP — EFFECT SIZE
# # -------------------------------------------------------------------------

# plt.figure(figsize=(15, 14))

# sns.heatmap(logfc, cmap="vlag", center=0, xticklabels=True, yticklabels=True)

# plt.title("Top Replicated Dementia-Associated Genes\n" "Cross-Study log2 Fold Change")

# plt.xlabel("GEO Dataset")
# plt.ylabel("Gene")

# plt.tight_layout()

# plt.savefig(OUT / "top50_crossstudy_logFC_heatmap.png", dpi=300)

# plt.savefig(OUT / "top50_crossstudy_logFC_heatmap.pdf")

# plt.close()

# # -------------------------------------------------------------------------
# # FDR HEATMAP
# # -------------------------------------------------------------------------

# neglogfdr = -np.log10(fdr.clip(lower=1e-300))

# plt.figure(figsize=(15, 14))

# sns.heatmap(neglogfdr, cmap="viridis", xticklabels=True, yticklabels=True)

# plt.title("Top Replicated Genes\n" "-log10(FDR)")

# plt.xlabel("GEO Dataset")
# plt.ylabel("Gene")

# plt.tight_layout()

# plt.savefig(OUT / "top50_crossstudy_FDR_heatmap.png", dpi=300)

# plt.close()

# # -------------------------------------------------------------------------
# # DIRECTION HEATMAP
# # -------------------------------------------------------------------------

# plt.figure(figsize=(15, 14))

# sns.heatmap(
#     direction,
#     cmap="vlag",
#     center=0,
#     vmin=-1,
#     vmax=1,
#     xticklabels=True,
#     yticklabels=True,
# )

# plt.title("Top Replicated Genes\n" "Direction of Expression Change")

# plt.xlabel("GEO Dataset")
# plt.ylabel("Gene")

# plt.tight_layout()

# plt.savefig(OUT / "top50_crossstudy_direction_heatmap.png", dpi=300)

# plt.close()

# print("=" * 80)
# print("STEP 17 COMPLETE")
# print("=" * 80)
# print("Genes analyzed:", len(genes))
# print("Output:", OUT)


"""
STEP: Check DE file structure
Purpose: Inspect the DE file to see what column names are available
"""

import pandas as pd
from pathlib import Path

# Define paths
BASE = Path(r"D:\Dementia2\codes\biolysis")
DE = BASE / "step08_DE"

print("=" * 60)
print("CHECKING DE FILE STRUCTURE")
print("=" * 60)

# Find a DE file
de_files = list(DE.glob("*_DE.csv"))

if de_files:
    print(f"\nFound {len(de_files)} DE files")

    # Check the first file
    sample_file = de_files[0]
    print(f"\nChecking file: {sample_file.name}")

    df = pd.read_csv(sample_file)
    print(f"\nColumns in this file:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1}. {col}")

    print(f"\nFirst 5 rows:")
    print(df.head())

    print(f"\nData types:")
    print(df.dtypes)

    print(f"\nShape: {df.shape}")
else:
    print("\n❌ No DE files found in step08_DE directory")
    print(f"Looking in: {DE}")

print("\n" + "=" * 60)

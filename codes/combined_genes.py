"""
===============================================================================
CONSOLIDATED GENE SIGNATURE
Rank aggregation across two independent methods:
  - Method 1: Ensemble ML (Step 18 consensus)
  - Method 2: SGAPC deep learning

Methods compared:
  1. Borda Count (sum of ranks)
  2. Reciprocal Rank Fusion (RRF)
  3. Weighted Rank Fusion
===============================================================================
"""

import pandas as pd
import numpy as np

# =============================================================================
# INPUT DATA
# =============================================================================

# Method 1: Ensemble ML top-10 (from Step 18 consensus)
method_ml = [
    "CA10",
    "LSM4",
    "PHYHD1",
    "GFAP",
    "DGKG",
    "DUSP4",
    "TSPAN7",
    "ITPKB",
    "MAST3",
    "TMEM158",
]

# Method 2: SGAPC top-10 (from deep learning consensus)
method_sgapc = [
    "ITPKB",
    "GFAP",
    "DUSP4",
    "DGKG",
    "DUSP6",
    "NPTX2",
    "BDNF",
    "EGR1",
    "GJA1",
    "PHYHD1",
]

# Assign ranks (1 = best)
rank_ml = {gene: i + 1 for i, gene in enumerate(method_ml)}
rank_sgapc = {gene: i + 1 for i, gene in enumerate(method_sgapc)}

# Collect all unique genes
all_genes = sorted(set(method_ml) | set(method_sgapc))
print(f"Total unique genes across both methods: {len(all_genes)}\n")


# =============================================================================
# METHOD 1: BORDA COUNT
# =============================================================================
# Score = sum of ranks (lower = better)
# Missing genes get a penalty rank = max_rank + 1

MAX_RANK = 10  # top-10 lists
PENALTY = MAX_RANK + 1  # = 11

borda_scores = {}
for gene in all_genes:
    r1 = rank_ml.get(gene, PENALTY)
    r2 = rank_sgapc.get(gene, PENALTY)
    borda_scores[gene] = r1 + r2


# =============================================================================
# METHOD 2: RECIPROCAL RANK FUSION (RRF)
# =============================================================================
# Score = sum of 1/(k + rank) across methods
# k = 60 (standard value from Cormack et al. 2009)

K_RRF = 60

rrf_scores = {}
for gene in all_genes:
    score = 0.0
    if gene in rank_ml:
        score += 1.0 / (K_RRF + rank_ml[gene])
    if gene in rank_sgapc:
        score += 1.0 / (K_RRF + rank_sgapc[gene])
    rrf_scores[gene] = score


# =============================================================================
# METHOD 3: WEIGHTED RANK FUSION
# =============================================================================
# Score = w1 * rank1 + w2 * rank2
# Lower = better
# Example weights: 50/50 or 60/40 (adjust as desired)

W_ML = 0.5
W_SGAPC = 0.5

weighted_scores = {}
for gene in all_genes:
    r1 = rank_ml.get(gene, PENALTY)
    r2 = rank_sgapc.get(gene, PENALTY)
    weighted_scores[gene] = W_ML * r1 + W_SGAPC * r2


# =============================================================================
# BUILD RESULTS TABLE
# =============================================================================

df = pd.DataFrame(
    {
        "Gene": all_genes,
        "ML_Rank": [rank_ml.get(g, None) for g in all_genes],
        "SGAPC_Rank": [rank_sgapc.get(g, None) for g in all_genes],
        "Borda_Score": [borda_scores[g] for g in all_genes],
        "RRF_Score": [rrf_scores[g] for g in all_genes],
        "Weighted_Score": [weighted_scores[g] for g in all_genes],
    }
)

# Support flag
df["Support"] = df.apply(
    lambda row: (
        "Both"
        if (row["ML_Rank"] is not None and row["SGAPC_Rank"] is not None)
        else ("ML only" if row["ML_Rank"] is not None else "SGAPC only")
    ),
    axis=1,
)

# =============================================================================
# RANK BY EACH METHOD
# =============================================================================

df["Rank_Borda"] = df["Borda_Score"].rank(method="min").astype(int)
df["Rank_RRF"] = df["RRF_Score"].rank(ascending=False, method="min").astype(int)
df["Rank_Weighted"] = df["Weighted_Score"].rank(method="min").astype(int)

# =============================================================================
# DISPLAY RESULTS
# =============================================================================

print("=" * 100)
print("FULL RANK AGGREGATION RESULTS")
print("=" * 100)
print(df.sort_values("Rank_RRF").to_string(index=False))
print()

# =============================================================================
# TOP-10 BY EACH METHOD
# =============================================================================

print("=" * 100)
print("TOP-10 GENES BY EACH AGGREGATION METHOD")
print("=" * 100)

for method, col in [
    ("Borda Count (lowest score)", "Borda_Score"),
    ("RRF (highest score)", "RRF_Score"),
    ("Weighted Rank (lowest score)", "Weighted_Score"),
]:
    ascending = "RRF" not in method  # RRF is highest=best, others lowest=best
    top10 = df.sort_values(col, ascending=ascending).head(10)["Gene"].tolist()
    print(f"\n{method}:")
    for i, g in enumerate(top10, 1):
        print(f"  {i:2d}. {g}")


# =============================================================================
# FINAL CONSENSUS (RRF-BASED, RECOMMENDED)
# =============================================================================

print("\n" + "=" * 100)
print("RECOMMENDED FINAL CONSENSUS SIGNATURE (RRF)")
print("=" * 100)

final = df.sort_values("RRF_Score", ascending=False).head(10).reset_index(drop=True)
final["Final_Rank"] = final.index + 1

print(
    final[
        ["Final_Rank", "Gene", "ML_Rank", "SGAPC_Rank", "Support", "RRF_Score"]
    ].to_string(index=False)
)


# =============================================================================
# SAVE OUTPUT
# =============================================================================

output_file = "consolidated_gene_signature.csv"
final[["Final_Rank", "Gene", "ML_Rank", "SGAPC_Rank", "Support", "RRF_Score"]].to_csv(
    output_file, index=False
)
print(f"\n[OK] Results saved to: {output_file}")

# Save full comparison
df_sorted = df.sort_values("Rank_RRF")
df_sorted.to_csv("full_rank_aggregation.csv", index=False)
print(f"[OK] Full results saved to: full_rank_aggregation.csv")

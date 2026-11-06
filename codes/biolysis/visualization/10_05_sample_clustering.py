from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(r"D:\Dementia2\codes\biolysis")
EXPR = BASE / "step07_phenotype"
OUT = BASE / "step10_visualization" / "05_sample_clustering"
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

for gse in GSES:

    f = EXPR / f"{gse}_case_control_expression.csv"

    if not f.exists():
        continue

    x = pd.read_csv(f, index_col=0)

    genes = x.var(axis=1).nlargest(min(1000, len(x))).index
    x = x.loc[genes]

    x = x.T

    g = sns.clustermap(
        x,
        metric="correlation",
        method="average",
        cmap="vlag",
        figsize=(12, 10),
        xticklabels=False,
        yticklabels=False,
    )

    g.fig.suptitle(f"{gse} — Hierarchical Sample Clustering", y=1.02)

    g.savefig(OUT / f"{gse}_hierarchical_clustering.png", dpi=300, bbox_inches="tight")

    plt.close("all")

print("Hierarchical clustering complete.")

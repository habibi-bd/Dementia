from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(r"D:\Dementia2\codes\biolysis")
EXPR = BASE / "step07_phenotype"
OUT = BASE / "step10_visualization" / "04_sample_correlation"
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

    # Select variable genes
    genes = x.var(axis=1).nlargest(min(1000, len(x))).index
    x = x.loc[genes]

    corr = x.corr(method="pearson")

    plt.figure(figsize=(11, 9))

    sns.heatmap(
        corr, cmap="vlag", center=0, square=True, xticklabels=False, yticklabels=False
    )

    plt.title(f"{gse} — Sample Correlation")
    plt.tight_layout()

    plt.savefig(OUT / f"{gse}_sample_correlation.png", dpi=300)
    plt.close()

print("Sample correlation complete.")

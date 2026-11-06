from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE = Path(r"D:\Dementia2\codes\biolysis")
EXPR = BASE / "step07_phenotype"
OUT = BASE / "step10_visualization" / "06_expression_distribution"
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

    # Sample distributions
    values = np.log2(
        x.replace([np.inf, -np.inf], np.nan).dropna(how="all").clip(lower=1e-8) + 1
    )

    plt.figure(figsize=(13, 7))

    plt.boxplot([values[c].dropna() for c in values.columns], showfliers=False)

    plt.xlabel("Samples")
    plt.ylabel("log2(Expression + 1)")
    plt.title(f"{gse} — Expression Distribution")
    plt.tight_layout()

    plt.savefig(OUT / f"{gse}_expression_distribution.png", dpi=300)
    plt.close()

print("Expression distribution complete.")

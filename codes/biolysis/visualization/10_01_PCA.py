from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE = Path(r"D:\Dementia2\codes\biolysis")
EXPR = BASE / "step07_phenotype"
OUT = BASE / "step10_visualization" / "01_PCA"
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
    xfile = EXPR / f"{gse}_case_control_expression.csv"
    mfile = EXPR / f"{gse}_curated_metadata.csv"

    if not xfile.exists() or not mfile.exists():
        continue

    x = pd.read_csv(xfile, index_col=0)
    meta = pd.read_csv(mfile)

    sample_col = next(
        c for c in meta.columns if c.lower() in ["sample_id", "sample", "gsm"]
    )

    label_col = next(c for c in meta.columns if c.lower() in ["label", "group"])

    common = [s for s in x.columns if s in set(meta[sample_col])]

    x = x[common]
    meta = meta.set_index(sample_col).loc[common]

    X = x.T.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    # Keep manageable feature count
    variances = X.var(axis=0)
    genes = variances.nlargest(min(5000, len(variances))).index
    X = X[genes]

    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(9, 7))

    for group in sorted(meta[label_col].unique()):
        idx = meta[label_col].values == group
        ax.scatter(pcs[idx, 0], pcs[idx, 1], label=str(group), alpha=0.75, s=45)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"{gse} — PCA")
    ax.legend()
    plt.tight_layout()

    plt.savefig(OUT / f"{gse}_PCA.png", dpi=300)
    plt.savefig(OUT / f"{gse}_PCA.pdf")
    plt.close()

print("PCA complete.")




from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

BASE = Path(r"D:\Dementia2\codes\biolysis")
EXPR = BASE / "step07_phenotype"
OUT = BASE / "step10_visualization" / "03_tSNE"
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
    xf = EXPR / f"{gse}_case_control_expression.csv"
    mf = EXPR / f"{gse}_curated_metadata.csv"

    if not xf.exists() or not mf.exists():
        continue

    x = pd.read_csv(xf, index_col=0)
    meta = pd.read_csv(mf)

    sample_col = next(
        c for c in meta.columns if c.lower() in ["sample_id", "sample", "gsm"]
    )
    label_col = next(c for c in meta.columns if c.lower() in ["label", "group"])

    meta = meta.set_index(sample_col)
    common = [s for s in x.columns if s in meta.index]

    x = x[common]
    meta = meta.loc[common]

    X = x.T.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    genes = X.var().nlargest(min(2000, X.shape[1])).index
    X = StandardScaler().fit_transform(X[genes])

    if len(X) < 6:
        continue

    perplexity = min(30, max(2, (len(X) - 1) // 3))

    emb = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
    ).fit_transform(X)

    plt.figure(figsize=(9, 7))

    for group in sorted(meta[label_col].unique()):
        idx = meta[label_col].values == group
        plt.scatter(emb[idx, 0], emb[idx, 1], label=str(group), alpha=0.75, s=45)

    plt.xlabel("t-SNE1")
    plt.ylabel("t-SNE2")
    plt.title(f"{gse} — t-SNE")
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUT / f"{gse}_tSNE.png", dpi=300)
    plt.savefig(OUT / f"{gse}_tSNE.pdf")
    plt.close()

print("t-SNE complete.")

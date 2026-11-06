# import os
# import warnings
# import numpy as np
# import pandas as pd
# from combat.pycombat import pycombat

# warnings.filterwarnings("ignore")

# # ================================================================
# # PATHS
# # ================================================================

# BASE = r"D:\Dementia2"
# DATA = os.path.join(BASE, "data")
# MATRIX = os.path.join(DATA, "matrix")
# MAPS = os.path.join(BASE, "codes", "platform_gene_maps")
# OUT = os.path.join(BASE, "codes", "ml_data")
# os.makedirs(OUT, exist_ok=True)

# # ================================================================
# # DATASETS
# # ================================================================

# DATASETS = [
#     ("GSE33000", "GPL4372", "GSE33000_series_matrix.txt"),
#     ("GSE29378", "GPL6947", "GSE29378_series_matrix.txt"),
#     ("GSE44770", "GPL4372", "GSE44770_series_matrix.txt"),
#     ("GSE36980", "GPL6244", "GSE36980_series_matrix.txt"),
#     ("GSE37263", "GPL5175", "GSE37263_series_matrix.txt"),
#     ("GSE28146", "GPL570", "GSE28146_series_matrix.txt"),
#     ("GSE16759", "GPL570", "GSE16759-GPL570_series_matrix.txt"),
#     ("GSE5281", "GPL570", "GSE5281_series_matrix.txt"),
#     ("GSE1297", "GPL96", "GSE1297_series_matrix.txt"),
#     ("GSE4757", "GPL570", "GSE4757_series_matrix.txt"),
# ]

# META = ["condition", "label", "dataset", "platform"]


# # ================================================================
# # FIND GENE MAP
# # ================================================================


# def find_map(gpl):
#     names = [
#         f"{gpl}_probe_to_gene.csv",
#         f"{gpl}_probe_gene_map.csv",
#         f"{gpl}_gene_map.csv",
#         f"{gpl}_map.csv",
#     ]

#     for name in names:
#         path = os.path.join(MAPS, name)
#         if os.path.exists(path):
#             return path

#     return None


# # ================================================================
# # READ GEO SERIES MATRIX
# # ================================================================


# def read_matrix(path):

#     header = None

#     with open(path, "r", encoding="utf-8", errors="replace") as f:
#         for i, line in enumerate(f):
#             if line.startswith("ID_REF\t") or line.startswith('"ID_REF"\t'):
#                 header = i
#                 break

#     if header is None:
#         raise ValueError(f"ID_REF header not found: {path}")

#     x = pd.read_csv(
#         path,
#         sep="\t",
#         skiprows=header,
#         compression="infer",
#         quotechar='"',
#         low_memory=False,
#     )

#     x = x.rename(columns={x.columns[0]: "probe"})

#     x["probe"] = x["probe"].astype(str).str.strip().str.strip('"')

#     x = x.set_index("probe")

#     x = x.apply(pd.to_numeric, errors="coerce")

#     x = x[~x.index.duplicated(keep="first")]

#     return x


# # ================================================================
# # FIND COLUMN
# # ================================================================


# def find_column(df, names):

#     names = [x.lower() for x in names]

#     for c in df.columns:
#         if str(c).strip().lower() in names:
#             return c

#     for c in df.columns:
#         c2 = str(c).strip().lower()
#         if any(n in c2 for n in names):
#             return c

#     return None


# # ================================================================
# # PROBE -> GENE
# # ================================================================


# def probe_to_gene(expr, mapping):

#     probe_col = find_column(
#         mapping, ["probe_id", "probe", "id", "spot_id", "platform_spotid"]
#     )

#     gene_col = find_column(
#         mapping, ["gene_symbol", "gene symbol", "symbol", "unigene symbol"]
#     )

#     if probe_col is None or gene_col is None:
#         raise ValueError(f"Cannot identify probe/gene columns: {list(mapping.columns)}")

#     m = mapping[[probe_col, gene_col]].copy()
#     m.columns = ["probe", "gene"]

#     m["probe"] = m["probe"].astype(str).str.strip()
#     m["gene"] = m["gene"].astype(str).str.strip()

#     bad = ["", "nan", "none", "na", "---", "null"]

#     m = m[(m["gene"] != "") & (~m["gene"].str.lower().isin(bad))]

#     m = m[m["probe"].isin(expr.index)]

#     m = m.drop_duplicates("probe")

#     if len(m) == 0:
#         raise ValueError("No probes could be mapped to genes.")

#     x = expr.loc[m["probe"]].copy()

#     x["gene"] = m.set_index("probe").loc[x.index, "gene"].values

#     # Multiple probes -> median gene expression
#     x = x.groupby("gene").median()

#     return x


# # ================================================================
# # MAIN
# # ================================================================

# print("=" * 70)
# print("STEP 16 - PYCOMBAT INTEGRATED GENE EXPRESSION MATRIX")
# print("=" * 70)

# # ------------------------------------------------
# # Labels
# # ------------------------------------------------

# labels = pd.read_csv(os.path.join(DATA, "labels.csv"), dtype=str)

# labels["gse_id"] = labels["gse_id"].astype(str).str.strip()
# labels["sample_id"] = labels["sample_id"].astype(str).str.strip()
# labels["condition"] = labels["condition"].astype(str).str.strip()
# labels["label"] = pd.to_numeric(labels["label"], errors="coerce")

# print()
# print("LABELS")
# print(f"Samples : {len(labels):,}")
# print(f"AD      : {(labels['label'] == 1).sum():,}")
# print(f"Control : {(labels['label'] == 0).sum():,}")


# frames = []
# summary = []


# # ================================================================
# # PROCESS DATASETS
# # ================================================================

# for gse, gpl, filename in DATASETS:

#     print()
#     print("-" * 70)
#     print(f"{gse} -> {gpl}")
#     print("-" * 70)

#     matrix_path = os.path.join(MATRIX, filename)
#     map_path = find_map(gpl)

#     if not os.path.exists(matrix_path):
#         print("STATUS: MATRIX NOT FOUND")
#         continue

#     if map_path is None:
#         print("STATUS: GENE MAP NOT FOUND")
#         continue

#     print("Reading expression matrix...")

#     expr = read_matrix(matrix_path)

#     lab = labels[labels["gse_id"] == gse].copy()

#     labelled = [s for s in expr.columns if s in set(lab["sample_id"])]

#     if len(labelled) < 4:
#         print("STATUS: TOO FEW LABELLED SAMPLES")
#         continue

#     expr = expr[labelled]

#     print(f"Samples retained : {len(labelled):,}")
#     print(f"Probe features   : {len(expr):,}")

#     # ------------------------------------------------------------
#     # Log transformation
#     # ------------------------------------------------------------

#     finite = expr.to_numpy(dtype=float)
#     finite = finite[np.isfinite(finite)]

#     if len(finite) > 0 and np.nanpercentile(finite, 99) > 50:
#         print("Applying log2(x + 1)...")
#         expr = np.log2(expr + 1)

#     # ------------------------------------------------------------
#     # Probe -> gene
#     # ------------------------------------------------------------

#     mapping = pd.read_csv(map_path, dtype=str, low_memory=False)

#     expr = probe_to_gene(expr, mapping)

#     print(f"Genes after mapping : {len(expr):,}")

#     # ------------------------------------------------------------
#     # Remove unusable genes
#     # ------------------------------------------------------------

#     expr = expr.replace([np.inf, -np.inf], np.nan)

#     expr = expr.dropna(axis=0, how="all")

#     # Median-impute occasional missing values
#     expr = expr.T.fillna(expr.T.median()).T

#     # ------------------------------------------------------------
#     # Samples x genes
#     # ------------------------------------------------------------

#     sample_gene = expr.T

#     sample_gene["sample_id"] = sample_gene.index

#     sample_gene = sample_gene.merge(
#         lab[["sample_id", "condition", "label"]], on="sample_id", how="inner"
#     )

#     sample_gene["dataset"] = gse
#     sample_gene["platform"] = gpl

#     sample_gene = sample_gene.set_index("sample_id")

#     frames.append(sample_gene)

#     summary.append(
#         {
#             "gse_id": gse,
#             "gpl_id": gpl,
#             "samples": len(sample_gene),
#             "alzheimers": int((sample_gene["label"] == 1).sum()),
#             "controls": int((sample_gene["label"] == 0).sum()),
#             "genes": len(expr),
#         }
#     )

#     print("STATUS: OK")


# # ================================================================
# # CHECK DATASETS
# # ================================================================

# if not frames:
#     raise RuntimeError("No datasets were successfully processed.")

# print()
# print("=" * 70)
# print("COMBINING DATASETS")
# print("=" * 70)

# # ------------------------------------------------
# # Find genes present on every usable dataset
# # ------------------------------------------------

# common = set(frames[0].columns) - set(META)

# for frame in frames[1:]:
#     common &= set(frame.columns) - set(META)

# genes = sorted(common)

# print(f"Usable datasets : {len(frames)}")
# print(f"Common genes    : {len(genes):,}")


# # ================================================================
# # BUILD EXPRESSION + METADATA
# # ================================================================

# metadata = pd.concat([f[META] for f in frames], axis=0)

# X = pd.concat([f[genes] for f in frames], axis=0)

# X = X[genes]

# # Genes x samples for pyComBat
# combat_X = X.T

# batch = metadata["dataset"].to_numpy()

# # Biological condition:
# # 0 = Control
# # 1 = Alzheimer
# condition = metadata["label"].astype(int).to_numpy()

# print()
# print("COMBAT INPUT")
# print(f"Samples : {combat_X.shape[1]:,}")
# print(f"Genes   : {combat_X.shape[0]:,}")
# print(f"Batches : {len(np.unique(batch))}")


# # ================================================================
# # PYCOMBAT
# # ================================================================

# print()
# print("=" * 70)
# print("RUNNING PYCOMBAT")
# print("=" * 70)

# # IMPORTANT:
# # pycombat 0.20 expects mod as a list.
# # Do NOT use DataFrame or numpy array here.

# combat_X = pycombat(combat_X, batch=batch, mod=[condition.tolist()])

# print("PyCombat completed.")


# # ================================================================
# # BACK TO SAMPLES x GENES
# # ================================================================

# combat_X = pd.DataFrame(combat_X, index=genes, columns=X.index).T

# combat_X.index.name = "sample_id"


# # ================================================================
# # FINAL MATRIX
# # ================================================================

# final = pd.concat(
#     [metadata.reset_index(drop=True), combat_X.reset_index(drop=True)], axis=1
# )

# # Remove accidental duplicate columns
# final = final.loc[:, ~final.columns.duplicated()]


# # ================================================================
# # SANITY CHECK
# # ================================================================

# print()
# print("=" * 70)
# print("FINAL SANITY CHECK")
# print("=" * 70)

# print(f"Samples : {len(final):,}")
# print(f"Genes   : {len(genes):,}")
# print(f"AD      : {(final['label'] == 1).sum():,}")
# print(f"Control : {(final['label'] == 0).sum():,}")

# print()
# print("Samples per dataset:")

# print(
#     final.groupby(["dataset", "platform"])["label"].agg(
#         samples="count", AD="sum", Control=lambda x: (x == 0).sum()
#     )
# )


# # ================================================================
# # SAVE
# # ================================================================

# matrix_out = os.path.join(OUT, "integrated_combat_gene_expression.csv")

# labels_out = os.path.join(OUT, "integrated_combat_labels.csv")

# summary_out = os.path.join(OUT, "combat_dataset_summary.csv")

# final.to_csv(matrix_out, index=False)

# final[["sample_id", "condition", "label", "dataset", "platform"]].to_csv(
#     labels_out, index=False
# )

# pd.DataFrame(summary).to_csv(summary_out, index=False)


# # ================================================================
# # DONE
# # ================================================================

# print()
# print("=" * 70)
# print("STEP 16 COMPLETE")
# print("=" * 70)

# print("Output:")
# print(matrix_out)
# print(labels_out)
# print(summary_out)

# print()
# print("Ready for leakage-safe ML.")
# print("=" * 70)


import os
import warnings
import numpy as np
import pandas as pd
from combat.pycombat import pycombat

warnings.filterwarnings("ignore")

BASE = r"D:\Dementia2"
DATA = os.path.join(BASE, "data")
MATRIX = os.path.join(DATA, "matrix")
MAPS = os.path.join(BASE, "codes", "platform_gene_maps")
OUT = os.path.join(BASE, "codes", "ml_data")
os.makedirs(OUT, exist_ok=True)

DATASETS = [
    ("GSE33000", "GPL4372", "GSE33000_series_matrix.txt"),
    ("GSE29378", "GPL6947", "GSE29378_series_matrix.txt"),
    ("GSE44770", "GPL4372", "GSE44770_series_matrix.txt"),
    ("GSE36980", "GPL6244", "GSE36980_series_matrix.txt"),
    ("GSE37263", "GPL5175", "GSE37263_series_matrix.txt"),
    ("GSE28146", "GPL570", "GSE28146_series_matrix.txt"),
    ("GSE16759", "GPL570", "GSE16759-GPL570_series_matrix.txt"),
    ("GSE5281", "GPL570", "GSE5281_series_matrix.txt"),
    ("GSE1297", "GPL96", "GSE1297_series_matrix.txt"),
    ("GSE4757", "GPL570", "GSE4757_series_matrix.txt"),
]

META = ["sample_id", "condition", "label", "dataset", "platform"]


def find_map(gpl):
    for n in [
        f"{gpl}_probe_to_gene.csv",
        f"{gpl}_probe_gene_map.csv",
        f"{gpl}_gene_map.csv",
        f"{gpl}_map.csv",
    ]:
        p = os.path.join(MAPS, n)
        if os.path.exists(p):
            return p
    return None


def read_matrix(path):
    header = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if line.startswith("ID_REF\t") or line.startswith('"ID_REF"\t'):
                header = i
                break

    if header is None:
        raise ValueError(f"ID_REF header not found: {path}")

    x = pd.read_csv(
        path,
        sep="\t",
        skiprows=header,
        compression="infer",
        quotechar='"',
        low_memory=False,
    )

    x = x.rename(columns={x.columns[0]: "probe"})
    x["probe"] = x["probe"].astype(str).str.strip().str.strip('"')
    x = x.set_index("probe")
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x[~x.index.duplicated()]

    return x


def find_column(df, names):
    names = [n.lower() for n in names]

    for c in df.columns:
        if str(c).strip().lower() in names:
            return c

    for c in df.columns:
        if any(n in str(c).lower() for n in names):
            return c

    return None


def probe_to_gene(expr, mapping):

    p = find_column(mapping, ["probe_id", "probe", "id", "spot_id", "platform_spotid"])

    g = find_column(mapping, ["gene_symbol", "gene symbol", "symbol", "unigene symbol"])

    if p is None or g is None:
        raise ValueError(f"Cannot identify probe/gene columns: {list(mapping.columns)}")

    m = mapping[[p, g]].copy()
    m.columns = ["probe", "gene"]

    m["probe"] = m["probe"].astype(str).str.strip()
    m["gene"] = m["gene"].astype(str).str.strip()

    bad = ["", "nan", "none", "na", "---", "null"]

    m = m[(m["gene"] != "") & (~m["gene"].str.lower().isin(bad))]

    m = m[m["probe"].isin(expr.index)]
    m = m.drop_duplicates("probe")

    x = expr.loc[m["probe"]].copy()

    x["gene"] = m.set_index("probe").loc[x.index, "gene"].values

    return x.groupby("gene").median()


print("=" * 70)
print("STEP 16 - PYCOMBAT INTEGRATED GENE EXPRESSION MATRIX")
print("=" * 70)

# ================================================================
# LABELS
# ================================================================

labels = pd.read_csv(os.path.join(DATA, "labels.csv"), dtype=str)

labels["gse_id"] = labels["gse_id"].str.strip()
labels["sample_id"] = labels["sample_id"].str.strip()
labels["condition"] = labels["condition"].str.strip()
labels["label"] = pd.to_numeric(labels["label"], errors="coerce")

print()
print("LABELS")
print(f"Samples : {len(labels):,}")
print(f"AD      : {(labels.label == 1).sum():,}")
print(f"Control : {(labels.label == 0).sum():,}")

frames = []
summary = []

# ================================================================
# DATASET PROCESSING
# ================================================================

for gse, gpl, filename in DATASETS:

    print()
    print("-" * 70)
    print(f"{gse} -> {gpl}")
    print("-" * 70)

    matrix_path = os.path.join(MATRIX, filename)
    map_path = find_map(gpl)

    if not os.path.exists(matrix_path):
        print("STATUS: MATRIX NOT FOUND")
        continue

    if map_path is None:
        print("STATUS: GENE MAP NOT FOUND")
        continue

    print("Reading expression matrix...")

    expr = read_matrix(matrix_path)

    lab = labels[labels.gse_id == gse].copy()

    samples = [s for s in expr.columns if s in set(lab.sample_id)]

    if len(samples) < 4:
        print("STATUS: TOO FEW LABELLED SAMPLES")
        continue

    expr = expr[samples]

    print(f"Samples retained : {len(samples):,}")
    print(f"Probe features   : {len(expr):,}")

    # ------------------------------------------------------------
    # Log2 transform if raw-scale expression
    # ------------------------------------------------------------

    values = expr.to_numpy(dtype=float)
    values = values[np.isfinite(values)]

    if len(values) and np.nanpercentile(values, 99) > 50:
        print("Applying log2(x + 1)...")
        expr = np.log2(expr + 1)

    # ------------------------------------------------------------
    # Probe -> gene
    # ------------------------------------------------------------

    mapping = pd.read_csv(map_path, dtype=str, low_memory=False)

    expr = probe_to_gene(expr, mapping)

    print(f"Genes after mapping : {len(expr):,}")

    # ------------------------------------------------------------
    # Clean missing values
    # ------------------------------------------------------------

    expr = expr.replace([np.inf, -np.inf], np.nan)
    expr = expr.dropna(axis=0, how="all")
    expr = expr.T.fillna(expr.T.median()).T

    # ------------------------------------------------------------
    # Samples x genes
    # ------------------------------------------------------------

    sample_gene = expr.T

    sample_gene["sample_id"] = sample_gene.index

    sample_gene = sample_gene.merge(
        lab[["sample_id", "condition", "label"]], on="sample_id", how="inner"
    )

    sample_gene["dataset"] = gse
    sample_gene["platform"] = gpl

    sample_gene = sample_gene.set_index("sample_id")

    frames.append(sample_gene)

    summary.append(
        {
            "gse_id": gse,
            "gpl_id": gpl,
            "samples": len(sample_gene),
            "alzheimers": int((sample_gene.label == 1).sum()),
            "controls": int((sample_gene.label == 0).sum()),
            "genes": len(expr),
        }
    )

    print("STATUS: OK")


# ================================================================
# COMMON GENES
# ================================================================

if not frames:
    raise RuntimeError("No datasets were successfully processed.")

print()
print("=" * 70)
print("COMBINING DATASETS")
print("=" * 70)

common = set(frames[0].columns) - {"condition", "label", "dataset", "platform"}

for f in frames[1:]:
    common &= set(f.columns) - {"condition", "label", "dataset", "platform"}

genes = sorted(common)

print(f"Usable datasets : {len(frames)}")
print(f"Common genes    : {len(genes):,}")


# ================================================================
# METADATA + EXPRESSION
# ================================================================

metadata = pd.concat(
    [f[["condition", "label", "dataset", "platform"]] for f in frames], axis=0
)

# Preserve sample IDs explicitly
metadata.insert(0, "sample_id", metadata.index.astype(str))

X = pd.concat([f[genes] for f in frames], axis=0)

# Ensure exact sample order
X = X.loc[metadata.index]

# Genes x samples for PyCombat
combat_X = X.T

batch = metadata["dataset"].to_numpy()

condition = metadata["label"].astype(int).to_numpy()

print()
print("=" * 70)
print("COMBAT INPUT")
print("=" * 70)

print(f"Samples : {combat_X.shape[1]:,}")
print(f"Genes   : {combat_X.shape[0]:,}")
print(f"Batches : {len(np.unique(batch))}")
print(f"AD      : {(condition == 1).sum():,}")
print(f"Control : {(condition == 0).sum():,}")


# ================================================================
# PYCOMBAT
# ================================================================

print()
print("=" * 70)
print("RUNNING PYCOMBAT")
print("=" * 70)

# pycombat 0.20 expects mod as a LIST.
combat_X = pycombat(combat_X, batch=batch, mod=[condition.tolist()])

print("PyCombat completed.")


# ================================================================
# BACK TO SAMPLES x GENES
# ================================================================

combat_X = pd.DataFrame(combat_X, index=genes, columns=metadata.index).T

# Make sure the sample ID is preserved
combat_X.index.name = "sample_id"

combat_X.insert(0, "sample_id", combat_X.index.astype(str))

# ================================================================
# FINAL MATRIX
# ================================================================

final = pd.concat(
    [
        metadata.reset_index(drop=True),
        combat_X.reset_index(drop=True).drop(columns=["sample_id"]),
    ],
    axis=1,
)

# Remove duplicate columns if any
final = final.loc[:, ~final.columns.duplicated()]

# Put metadata first
gene_columns = [c for c in final.columns if c not in META]

final = final[META + gene_columns]

# ================================================================
# SANITY CHECK
# ================================================================

print()
print("=" * 70)
print("FINAL SANITY CHECK")
print("=" * 70)

print(f"Samples : {len(final):,}")
print(f"Genes   : {len(gene_columns):,}")
print(f"AD      : {(final.label == 1).sum():,}")
print(f"Control : {(final.label == 0).sum():,}")

print()
print("Samples per dataset:")

print(
    final.groupby(["dataset", "platform"])["label"].agg(
        samples="count", AD="sum", Control=lambda x: (x == 0).sum()
    )
)


# ================================================================
# SAVE
# ================================================================

matrix_out = os.path.join(OUT, "integrated_combat_gene_expression.csv")

labels_out = os.path.join(OUT, "integrated_combat_labels.csv")

summary_out = os.path.join(OUT, "combat_dataset_summary.csv")

final.to_csv(matrix_out, index=False)

final[META].to_csv(labels_out, index=False)

pd.DataFrame(summary).to_csv(summary_out, index=False)


# ================================================================
# FINAL
# ================================================================

print()
print("=" * 70)
print("STEP 16 COMPLETE")
print("=" * 70)

print(f"Samples : {len(final):,}")
print(f"Genes   : {len(gene_columns):,}")
print(f"AD      : {(final.label == 1).sum():,}")
print(f"Control : {(final.label == 0).sum():,}")

print()
print("OUTPUT FILES")
print("-" * 70)
print(matrix_out)
print(labels_out)
print(summary_out)

print()
print("Ready for Step 17: leakage-safe ML.")
print("=" * 70)

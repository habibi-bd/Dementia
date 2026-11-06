from pathlib import Path
import pandas as pd

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"D:\Dementia2")

MATRIX_FILE = PROJECT_DIR / "data" / "matrix" / "GSE33000_series_matrix.txt"

ANNOTATION_FILE = PROJECT_DIR / "data" / "gpl" / "GPL4372_annotation.csv"


# ============================================================
# GET ALL MATRIX PROBE IDS
# ============================================================

print("\n" + "=" * 80)
print("FULL GPL4372 → GENE COVERAGE CHECK")
print("=" * 80)

print("\nReading GSE33000 feature IDs...")

header_line = None

with open(MATRIX_FILE, "r", encoding="utf-8", errors="replace") as f:

    for i, line in enumerate(f):

        if line.startswith("!"):
            continue

        if "\t" in line:

            header_line = i
            break


if header_line is None:

    raise RuntimeError("Could not find expression matrix header.")


matrix = pd.read_csv(
    MATRIX_FILE,
    sep="\t",
    skiprows=header_line,
    usecols=[0],
    dtype=str,
    low_memory=False,
)

matrix.columns = ["probe"]

matrix["probe"] = matrix["probe"].astype(str).str.strip()

matrix_probes = set(matrix["probe"])

print(f"Expression probes : {len(matrix_probes):,}")


# ============================================================
# READ ANNOTATION
# ============================================================

print("Reading GPL4372 annotation...")

annotation = pd.read_csv(ANNOTATION_FILE, dtype=str, low_memory=False)

annotation.columns = [str(c).strip() for c in annotation.columns]

annotation["ID"] = annotation["ID"].astype(str).str.strip()

print(f"Annotation probes  : " f"{annotation['ID'].nunique():,}")


# ============================================================
# MATCH PROBES
# ============================================================

annotation_ids = set(annotation["ID"])

matched = matrix_probes.intersection(annotation_ids)

print(f"Matched probes     : {len(matched):,}")

print(f"Probe match rate   : " f"{len(matched) / len(matrix_probes) * 100:.2f}%")


# ============================================================
# SUBSET MATCHED ANNOTATION
# ============================================================

mapped = annotation[annotation["ID"].isin(matched)].copy()


# ============================================================
# GENE SYMBOL COVERAGE
# ============================================================

symbol = mapped["Gene symbol"].fillna("").astype(str).str.strip()

symbol_valid = symbol.ne("") & symbol.ne("nan") & symbol.ne("NA") & symbol.ne("---")


# ============================================================
# GENE ID COVERAGE
# ============================================================

gene_id = mapped["Gene ID"].fillna("").astype(str).str.strip()

gene_id_valid = (
    gene_id.ne("") & gene_id.ne("nan") & gene_id.ne("NA") & gene_id.ne("---")
)


# ============================================================
# BOTH
# ============================================================

both_valid = symbol_valid & gene_id_valid


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "-" * 80)
print("GENE ANNOTATION COVERAGE")
print("-" * 80)

print(f"Matched probes            : " f"{len(mapped):,}")

print(
    f"With gene symbol          : "
    f"{symbol_valid.sum():,} "
    f"({symbol_valid.mean() * 100:.2f}%)"
)

print(
    f"With gene ID              : "
    f"{gene_id_valid.sum():,} "
    f"({gene_id_valid.mean() * 100:.2f}%)"
)

print(
    f"With symbol + gene ID     : "
    f"{both_valid.sum():,} "
    f"({both_valid.mean() * 100:.2f}%)"
)


# ============================================================
# UNIQUE GENES
# ============================================================

unique_symbols = symbol[symbol_valid].nunique()

unique_gene_ids = gene_id[gene_id_valid].nunique()

print("\n" + "-" * 80)
print("UNIQUE GENE COUNTS")
print("-" * 80)

print(f"Unique gene symbols       : " f"{unique_symbols:,}")

print(f"Unique gene IDs           : " f"{unique_gene_ids:,}")


# ============================================================
# UNMAPPED EXAMPLES
# ============================================================

unmapped_gene = mapped[~both_valid][["ID", "Gene symbol", "Gene ID"]]

print("\n" + "-" * 80)
print("EXAMPLES WITHOUT COMPLETE GENE ANNOTATION")
print("-" * 80)

print(unmapped_gene.head(10).to_string(index=False))


print("\n" + "=" * 80)
print("STEP 9 COMPLETE")
print("=" * 80)

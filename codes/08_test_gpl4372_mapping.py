from pathlib import Path
import pandas as pd

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"D:\Dementia2")

MATRIX_FILE = PROJECT_DIR / "data" / "matrix" / "GSE33000_series_matrix.txt"

ANNOTATION_FILE = PROJECT_DIR / "data" / "gpl" / "GPL4372_annotation.csv"


# ============================================================
# READ FIRST PROBES FROM MATRIX
# ============================================================

print("\n" + "=" * 80)
print("GPL4372 PROBE → GENE TEST")
print("=" * 80)

print("\nReading GSE33000 probe IDs...")

header_line = None

with open(MATRIX_FILE, "r", encoding="utf-8", errors="replace") as f:

    for i, line in enumerate(f):

        line = line.rstrip("\n")

        if line.startswith("!"):
            continue

        if "\t" in line:

            header_line = i
            break


if header_line is None:
    raise RuntimeError("Could not find expression matrix header.")


matrix = pd.read_csv(
    MATRIX_FILE, sep="\t", skiprows=header_line, usecols=[0], nrows=20, dtype=str
)

matrix_probe_col = matrix.columns[0]

matrix_probes = matrix[matrix_probe_col].astype(str).str.strip()

print(f"Matrix feature column : {matrix_probe_col}")

print("First 10 matrix probes:")

for probe in matrix_probes.head(10):
    print(f"  {probe}")


# ============================================================
# READ GPL4372 ANNOTATION
# ============================================================

print("\nReading GPL4372 annotation...")

annotation = pd.read_csv(ANNOTATION_FILE, dtype=str, low_memory=False)

annotation.columns = [str(c).strip() for c in annotation.columns]

print(f"Annotation rows    : {len(annotation):,}")

print(f"Annotation columns : {len(annotation.columns)}")

print("\nRelevant columns:")

for col in annotation.columns:

    if "ID" in col.upper() or "SYMBOL" in col.upper() or "GENE" in col.upper():

        print(f"  {col}")


# ============================================================
# CHECK ID COLUMN
# ============================================================

if "ID" not in annotation.columns:

    raise RuntimeError("GPL4372 annotation does not contain expected ID column.")


# ============================================================
# MATCH
# ============================================================

annotation["ID"] = annotation["ID"].astype(str).str.strip()

matrix_probe_set = set(matrix_probes)

annotation_probe_set = set(annotation["ID"])

matched = matrix_probe_set.intersection(annotation_probe_set)


print("\n" + "-" * 80)
print("MATCH RESULT")
print("-" * 80)

print(f"Matrix probes tested : {len(matrix_probe_set)}")

print(f"Annotation probes    : {len(annotation_probe_set):,}")

print(f"Matched probes       : {len(matched)}")

print(f"Match percentage     : " f"{len(matched) / len(matrix_probe_set) * 100:.2f}%")


# ============================================================
# SHOW MAPPED GENES
# ============================================================

print("\n" + "-" * 80)
print("MAPPED PROBES")
print("-" * 80)

mapping_columns = ["ID", "Gene symbol", "Gene ID"]

available_columns = [c for c in mapping_columns if c in annotation.columns]

mapped = annotation[annotation["ID"].isin(matrix_probe_set)][available_columns]

print(mapped.to_string(index=False))


print("\n" + "=" * 80)
print("STEP 8 COMPLETE")
print("=" * 80)

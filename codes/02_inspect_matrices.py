from pathlib import Path
import pandas as pd

# ============================================================
# 1. Project directories
# ============================================================

PROJECT_DIR = Path(r"D:\Dementia2")
MATRIX_DIR = PROJECT_DIR / "data" / "matrix"
LABEL_FILE = PROJECT_DIR / "data" / "labels.csv"


# ============================================================
# 2. Datasets
# ============================================================

GSE_LIST = [
    "GSE33000",
    "GSE29378",
    "GSE44770",
    "GSE36980",
    "GSE37263",
    "GSE28146",
    "GSE16759",
    "GSE15222",
    "GSE6834",
    "GSE5281",
    "GSE1297",
    "GSE4757",
]


# ============================================================
# 3. Read labels
# ============================================================

labels = pd.read_csv(LABEL_FILE)

labels["gse_id"] = labels["gse_id"].astype(str)
labels["sample_id"] = labels["sample_id"].astype(str)


# ============================================================
# 4. Function to inspect one matrix
# ============================================================


def inspect_matrix(file_path):

    print("\n" + "=" * 70)
    print(file_path.name)
    print("=" * 70)

    # --------------------------------------------------------
    # Read first 100 lines
    # --------------------------------------------------------

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [next(f).rstrip("\n") for _ in range(100)]

    # --------------------------------------------------------
    # Print important GEO metadata lines
    # --------------------------------------------------------

    print("\nGEO metadata:")

    for line in lines:
        if line.startswith("!"):
            if (
                "platform_id" in line
                or "sample_id" in line
                or "title" in line
                or "source_name" in line
            ):
                print(line)

    # --------------------------------------------------------
    # Find expression table header
    # --------------------------------------------------------

    header_line = None

    for i, line in enumerate(lines):
        if line.startswith('"ID_REF"'):
            header_line = i
            break

    if header_line is None:

        for i, line in enumerate(lines):
            if line.startswith("ID_REF"):
                header_line = i
                break

    if header_line is None:
        print("\nCould not find expression table header.")
        return

    print("\nExpression table header found at line:", header_line + 1)

    # --------------------------------------------------------
    # Read expression table
    # --------------------------------------------------------

    df = pd.read_csv(
        file_path, sep="\t", comment="!", header=0, index_col=0, low_memory=False
    )

    # Remove GEO comment/metadata rows that may remain
    df = df.loc[~df.index.astype(str).str.startswith("!")]

    print("\nExpression matrix shape:")
    print("Rows (features/probes):", df.shape[0])
    print("Columns (samples):", df.shape[1])

    # --------------------------------------------------------
    # Show first few feature IDs
    # --------------------------------------------------------

    print("\nFirst 5 feature IDs:")
    print(df.index[:5].tolist())

    # --------------------------------------------------------
    # Show first 5 sample IDs
    # --------------------------------------------------------

    print("\nFirst 5 sample IDs:")
    print(df.columns[:5].tolist())

    # --------------------------------------------------------
    # Check sample IDs against labels
    # --------------------------------------------------------

    gse_name = file_path.name.split("_")[0]

    if gse_name == "GSE16759-GPL570":
        gse_name = "GSE16759"
    elif gse_name == "GSE16759-GPL8757":
        gse_name = "GSE16759"

    label_samples = set(labels.loc[labels["gse_id"] == gse_name, "sample_id"])

    matrix_samples = set(df.columns.astype(str))

    matched = label_samples.intersection(matrix_samples)
    missing = label_samples - matrix_samples
    extra = matrix_samples - label_samples

    print("\nSample matching:")
    print("Samples in labels.csv:", len(label_samples))
    print("Samples in matrix:", len(matrix_samples))
    print("Matched samples:", len(matched))
    print("Labels missing from matrix:", len(missing))
    print("Matrix samples without labels:", len(extra))

    if missing:
        print("\nFirst missing labeled samples:")
        print(sorted(missing)[:10])

    if extra:
        print("\nFirst unlabeled matrix samples:")
        print(sorted(extra)[:10])


# ============================================================
# 5. Find and inspect matrices
# ============================================================

for gse in GSE_LIST:

    files = sorted(MATRIX_DIR.glob(f"{gse}*series_matrix.txt"))

    for file_path in files:
        inspect_matrix(file_path)


print("\n" + "=" * 70)
print("STEP 2 COMPLETE")
print("=" * 70)

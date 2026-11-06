from pathlib import Path
import pandas as pd

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(r"D:\Dementia2")
MATRIX_DIR = PROJECT_DIR / "data" / "matrix"


# ============================================================
# MATRICES WE ACTUALLY NEED
# ============================================================

MATRICES = [
    ("GSE33000", "GSE33000_series_matrix.txt"),
    ("GSE29378", "GSE29378_series_matrix.txt"),
    ("GSE44770", "GSE44770_series_matrix.txt"),
    ("GSE36980", "GSE36980_series_matrix.txt"),
    ("GSE37263", "GSE37263_series_matrix.txt"),
    ("GSE28146", "GSE28146_series_matrix.txt"),
    ("GSE16759-GPL570", "GSE16759-GPL570_series_matrix.txt"),
    ("GSE16759-GPL8757", "GSE16759-GPL8757_series_matrix.txt"),
    ("GSE15222", "GSE15222_series_matrix.txt"),
    ("GSE6834", "GSE6834_series_matrix.txt"),
    ("GSE5281", "GSE5281_series_matrix.txt"),
    ("GSE1297", "GSE1297_series_matrix.txt"),
    ("GSE4757", "GSE4757_series_matrix.txt"),
]


# ============================================================
# FIND EXPRESSION TABLE HEADER
# ============================================================


def find_header(file_path):

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:

        for line_number, line in enumerate(f):

            line = line.rstrip("\n")

            if not line:
                continue

            # GEO metadata
            if line.startswith("!"):
                continue

            # Expression table normally contains tabs.
            if "\t" in line:

                fields = line.split("\t")

                return line_number, fields

    return None, None


# ============================================================
# INSPECT ONE MATRIX
# ============================================================


def inspect_matrix(gse, filename):

    file_path = MATRIX_DIR / filename

    print(f"\n{gse}")
    print("-" * 70)

    if not file_path.exists():

        print("STATUS : FILE NOT FOUND")
        return

    print(f"File size : " f"{file_path.stat().st_size / (1024 * 1024):.1f} MB")

    header_line, header = find_header(file_path)

    if header is None:

        print("STATUS : HEADER NOT FOUND")
        return

    print(f"Header line : {header_line + 1}")

    print(f"Columns     : {len(header)}")

    print(f"Sample cols : {max(0, len(header) - 1)}")

    print(f"Feature col : {header[0]}")

    # --------------------------------------------------------
    # Read only first 10 expression rows
    # --------------------------------------------------------

    try:

        df = pd.read_csv(
            file_path,
            sep="\t",
            skiprows=header_line,
            nrows=10,
            dtype=str,
            low_memory=False,
        )

        feature_column = df.columns[0]

        features = df[feature_column].astype(str).str.strip()

        print("First features:")

        for feature in features.head(5):

            print(f"  {feature}")

        print(f"Expression columns detected : " f"{len(df.columns) - 1}")

        print("STATUS : OK")

    except Exception as e:

        print(f"STATUS : ERROR")

        print(f"ERROR  : {e}")


# ============================================================
# RUN
# ============================================================

print("\n" + "=" * 80)
print("FAST EXPRESSION MATRIX FEATURE CHECK")
print("=" * 80)

for gse, filename in MATRICES:

    inspect_matrix(gse, filename)


print("\n" + "=" * 80)
print("STEP 7A COMPLETE")
print("=" * 80)

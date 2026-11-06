from pathlib import Path
import pandas as pd

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"D:\Dementia2")

MATRIX_DIR = PROJECT_DIR / "data" / "matrix"
GPL_DIR = PROJECT_DIR / "data" / "gpl"


# ============================================================
# DATASETS TO INVESTIGATE
# ============================================================

DATASETS = [
    ("GSE37263", "GSE37263_series_matrix.txt", "GPL5175"),
    ("GSE15222", "GSE15222_series_matrix.txt", "GPL2700"),
    ("GSE6834", "GSE6834_series_matrix.txt", "GPL4757"),
    ("GSE16759-GPL570", "GSE16759-GPL570_series_matrix.txt", "GPL570"),
    ("GSE16759-GPL8757", "GSE16759-GPL8757_series_matrix.txt", "GPL8757"),
]


# ============================================================
# AVAILABLE ANNOTATION FILES
# ============================================================

ANNOTATION_FILES = {
    "GPL5175": [
        "GPL5175 (2).txt",
        "GPL5175.txt",
    ],
    "GPL2700": [
        "GPL2700.txt",
    ],
    "GPL4757": [
        "GPL4757.txt",
    ],
    "GPL570": [
        "GPL570_annotation.csv",
        "GPL570.txt",
    ],
    "GPL8757": [
        "GPL8757.soft",
    ],
}


# ============================================================
# FIND MATRIX HEADER
# ============================================================


def find_header(file_path):

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:

        for i, line in enumerate(f):

            if line.startswith("!"):
                continue

            if "\t" in line:
                return i

    return None


# ============================================================
# READ MATRIX PROBES
# ============================================================


def read_matrix_probes(file_path):

    header = find_header(file_path)

    if header is None:
        return set(), None

    df = pd.read_csv(
        file_path, sep="\t", skiprows=header, usecols=[0], dtype=str, low_memory=False
    )

    column = str(df.columns[0]).strip()

    probes = df.iloc[:, 0].astype(str).str.strip()

    probes = probes[
        probes.ne("") & probes.ne("nan") & ~probes.str.startswith("!series")
    ]

    return set(probes), column


# ============================================================
# INSPECT TEXT FILE
# ============================================================


def inspect_text_file(file_path):

    print(f"\n  Annotation file: {file_path.name}")

    try:

        df = pd.read_csv(file_path, sep=None, engine="python", dtype=str, nrows=5)

        print(f"  Detected columns: {len(df.columns)}")

        print("  Columns:")

        for col in df.columns:

            print(f"    {str(col)[:100]}")

    except Exception as e:

        print(f"  Could not read as table: {e}")

        # Show first useful lines
        try:

            with open(file_path, "r", encoding="utf-8", errors="replace") as f:

                lines = []

                for line in f:

                    line = line.strip()

                    if line:

                        lines.append(line)

                    if len(lines) >= 8:
                        break

            print("  First non-empty lines:")

            for line in lines:

                print("    " + line[:160])

        except Exception:
            pass


# ============================================================
# GPL8757 SPECIAL PARSER
# ============================================================


def inspect_gpl8757(file_path):

    print("\n  GPL8757 special inspection:")

    try:

        found_header = False
        header = None
        rows = 0

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:

            for line in f:

                line = line.rstrip("\n\r")

                if not found_header:

                    if line.startswith("ID\t"):

                        header = line.split("\t")
                        found_header = True

                else:

                    if line.strip():

                        rows += 1

        if header:

            print(f"  Header columns: {len(header)}")

            print("  Header:")

            print("    " + " | ".join(header))

            print(f"  Data rows: {rows:,}")

        else:

            print("  Could not locate table header.")

    except Exception as e:

        print(f"  ERROR: {e}")


# ============================================================
# MAIN
# ============================================================

print("\n" + "=" * 80)
print("REMAINING PLATFORM RESOLUTION")
print("=" * 80)


results = []


for gse, matrix_name, gpl in DATASETS:

    print("\n" + "-" * 80)
    print(f"{gse}  →  {gpl}")
    print("-" * 80)

    matrix_file = MATRIX_DIR / matrix_name

    if not matrix_file.exists():

        print(f"ERROR: matrix file not found: " f"{matrix_file.name}")

        continue

    # --------------------------------------------------------
    # Matrix
    # --------------------------------------------------------

    probes, feature_column = read_matrix_probes(matrix_file)

    print(f"Matrix feature column : {feature_column}")

    print(f"Matrix probes         : {len(probes):,}")

    if probes:

        print("First 5 probes:")

        for probe in list(probes)[:5]:

            print(f"  {probe}")

    # --------------------------------------------------------
    # Annotation files
    # --------------------------------------------------------

    annotation_files = ANNOTATION_FILES[gpl]

    for annotation_name in annotation_files:

        annotation_file = GPL_DIR / annotation_name

        if not annotation_file.exists():

            print(f"\n  MISSING: " f"{annotation_name}")

            continue

        # GPL8757
        if gpl == "GPL8757" and annotation_name == "GPL8757.soft":

            inspect_gpl8757(annotation_file)

            continue

        # Normal inspection
        inspect_text_file(annotation_file)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("STEP 11 COMPLETE")
print("=" * 80)

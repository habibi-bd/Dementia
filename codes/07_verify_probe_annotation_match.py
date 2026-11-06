from pathlib import Path
import gzip
import pandas as pd

# ============================================================
# 1. Directories
# ============================================================

PROJECT_DIR = Path(r"D:\Dementia2")
MATRIX_DIR = PROJECT_DIR / "data" / "matrix"
GPL_DIR = PROJECT_DIR / "data" / "gpl"

OUTPUT_FILE = PROJECT_DIR / "codes" / "probe_annotation_match_summary.csv"


# ============================================================
# 2. Dataset → matrix → GPL → annotation
# ============================================================

DATASETS = [
    (
        "GSE33000",
        "GSE33000_series_matrix.txt",
        "GPL4372",
        "GPL4372_annotation.csv",
        "csv",
    ),
    (
        "GSE29378",
        "GSE29378_series_matrix.txt",
        "GPL6947",
        "GPL6947_annotation.csv",
        "csv",
    ),
    (
        "GSE44770",
        "GSE44770_series_matrix.txt",
        "GPL4372",
        "GPL4372_annotation.csv",
        "csv",
    ),
    (
        "GSE36980",
        "GSE36980_series_matrix.txt",
        "GPL6244",
        "GPL6244_annotation.csv",
        "csv",
    ),
    (
        "GSE37263",
        "GSE37263_series_matrix.txt",
        "GPL5175",
        "GPL5175.annot.gz",
        "annot_gz",
    ),
    (
        "GSE28146",
        "GSE28146_series_matrix.txt",
        "GPL570",
        "GPL570_annotation.csv",
        "csv",
    ),
    (
        "GSE16759",
        "GSE16759-GPL570_series_matrix.txt",
        "GPL570",
        "GPL570_annotation.csv",
        "csv",
    ),
    (
        "GSE16759",
        "GSE16759-GPL8757_series_matrix.txt",
        "GPL8757",
        "GPL8757.soft",
        "soft",
    ),
    (
        "GSE15222",
        "GSE15222_series_matrix.txt",
        "GPL2700",
        "GPL2700.annot.gz",
        "annot_gz",
    ),
    ("GSE6834", "GSE6834_series_matrix.txt", "GPL4757", "GPL4757.annot.gz", "annot_gz"),
    ("GSE5281", "GSE5281_series_matrix.txt", "GPL570", "GPL570_annotation.csv", "csv"),
    ("GSE1297", "GSE1297_series_matrix.txt", "GPL96", "GPL96_annotation.csv", "csv"),
    ("GSE4757", "GSE4757_series_matrix.txt", "GPL570", "GPL570_annotation.csv", "csv"),
]


# ============================================================
# 3. Get feature IDs from GEO series matrix
# ============================================================


def get_matrix_features(file_path):

    # GEO series matrices contain metadata lines beginning
    # with ! and a tab-separated expression table.

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:

        for line in f:

            line = line.rstrip("\n")

            # Ignore metadata
            if line.startswith("!"):
                continue

            if not line.strip():
                continue

            # First non-metadata tabular line = expression header
            if "\t" in line:

                header = line.split("\t")

                feature_column = header[0].strip().strip('"')

                break

        else:

            raise ValueError("Could not find expression matrix header.")

    # Read only the first column, skipping metadata.
    df = pd.read_csv(
        file_path, sep="\t", comment="!", dtype=str, usecols=[0], low_memory=False
    )

    column = df.columns[0]

    features = df[column].astype(str).str.strip()

    # Correct filtering — no tuple indexer
    features = features[
        features.ne("") & features.ne("nan") & ~features.str.startswith("!")
    ]

    return set(features)


# ============================================================
# 4. Read CSV annotation
# ============================================================


def read_csv_annotation(file_path):

    df = pd.read_csv(file_path, dtype=str, low_memory=False)

    df.columns = [str(c).strip().strip('"') for c in df.columns]

    return df


# ============================================================
# 5. Read GEO .annot.gz
# ============================================================


def read_annot_gz(file_path):

    header_line = None

    with gzip.open(file_path, "rt", encoding="utf-8", errors="replace") as f:

        for i, line in enumerate(f):

            line = line.rstrip("\n")

            if not line.strip():
                continue

            if line.startswith("#"):
                continue

            fields = line.split("\t")

            if len(fields) >= 2:

                header_line = i
                break

    if header_line is None:

        raise ValueError("Could not locate annotation table header.")

    df = pd.read_csv(
        file_path,
        sep="\t",
        skiprows=header_line,
        compression="gzip",
        dtype=str,
        low_memory=False,
        on_bad_lines="skip",
    )

    df.columns = [str(c).strip().strip('"') for c in df.columns]

    return df


# ============================================================
# 6. Read GPL8757 SOFT
# ============================================================


def read_gpl8757_soft(file_path):

    header_line = None

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:

        for i, line in enumerate(f):

            line = line.rstrip("\n")

            if not line.strip():
                continue

            if line.startswith("!"):
                continue

            if line.startswith("^"):
                continue

            fields = line.split("\t")

            if len(fields) >= 2 and fields[0].strip() == "ID":

                header_line = i
                break

    if header_line is None:

        raise ValueError("Could not locate GPL8757 annotation table.")

    df = pd.read_csv(
        file_path,
        sep="\t",
        skiprows=header_line,
        dtype=str,
        low_memory=False,
        on_bad_lines="skip",
    )

    df.columns = [str(c).strip() for c in df.columns]

    return df


# ============================================================
# 7. Read annotation according to type
# ============================================================


def read_annotation(file_path, annotation_type):

    if annotation_type == "csv":
        return read_csv_annotation(file_path)

    elif annotation_type == "annot_gz":
        return read_annot_gz(file_path)

    elif annotation_type == "soft":
        return read_gpl8757_soft(file_path)

    else:
        raise ValueError(f"Unknown annotation type: {annotation_type}")


# ============================================================
# 8. Find annotation ID column
# ============================================================


def find_id_column(annotation_df):

    candidates = [
        "ID",
        "ID_REF",
        "Probe ID",
        "Probe_ID",
        "PROBE_ID",
        "probe_id",
        "SPOT_ID",
        "SPOTID",
    ]

    for candidate in candidates:

        if candidate in annotation_df.columns:
            return candidate

    return None


# ============================================================
# 9. Find gene symbol column
# ============================================================


def find_symbol_column(annotation_df):

    candidates = [
        "Gene Symbol",
        "Gene symbol",
        "GENE_SYMBOL",
        "gene_symbol",
        "Symbol",
    ]

    for candidate in candidates:

        if candidate in annotation_df.columns:
            return candidate

    return None


# ============================================================
# 10. Find gene ID column
# ============================================================


def find_gene_id_column(annotation_df):

    candidates = [
        "ENTREZ_GENE_ID",
        "Gene ID",
        "Gene_ID",
        "GENE_ID",
        "gene_id",
        "Entrez Gene",
        "ENTREZID",
        "Ensembl Gene ID",
        "miRNA_ID",
    ]

    for candidate in candidates:

        if candidate in annotation_df.columns:
            return candidate

    return None


# ============================================================
# 11. Analyze one dataset
# ============================================================


def analyze_dataset(gse, matrix_filename, gpl, annotation_filename, annotation_type):

    matrix_file = MATRIX_DIR / matrix_filename
    annotation_file = GPL_DIR / annotation_filename

    result = {
        "gse_id": gse,
        "gpl_id": gpl,
        "matrix_file": matrix_filename,
        "annotation_file": annotation_filename,
        "matrix_features": 0,
        "annotation_rows": 0,
        "annotation_id_column": "",
        "matched_features": 0,
        "match_percent": 0.0,
        "gene_symbol_column": "",
        "gene_id_column": "",
        "status": "ERROR",
        "issue": "",
    }

    try:

        # ----------------------------------------------------
        # Files
        # ----------------------------------------------------

        if not matrix_file.exists():

            result["issue"] = "Matrix file missing"
            return result

        if not annotation_file.exists():

            result["issue"] = "Annotation file missing"
            return result

        # ----------------------------------------------------
        # Matrix feature IDs
        # ----------------------------------------------------

        matrix_features = get_matrix_features(matrix_file)

        result["matrix_features"] = len(matrix_features)

        # ----------------------------------------------------
        # Annotation
        # ----------------------------------------------------

        annotation = read_annotation(annotation_file, annotation_type)

        result["annotation_rows"] = len(annotation)

        # ----------------------------------------------------
        # Columns
        # ----------------------------------------------------

        id_column = find_id_column(annotation)

        symbol_column = find_symbol_column(annotation)

        gene_id_column = find_gene_id_column(annotation)

        result["annotation_id_column"] = id_column or ""

        result["gene_symbol_column"] = symbol_column or ""

        result["gene_id_column"] = gene_id_column or ""

        if id_column is None:

            result["issue"] = "No annotation ID column found"

            return result

        # ----------------------------------------------------
        # Annotation IDs
        # ----------------------------------------------------

        annotation_ids = annotation[id_column].astype(str).str.strip()

        annotation_ids = annotation_ids[
            annotation_ids.ne("") & annotation_ids.ne("nan")
        ]

        annotation_ids = set(annotation_ids)

        # ----------------------------------------------------
        # Match
        # ----------------------------------------------------

        matched = matrix_features.intersection(annotation_ids)

        result["matched_features"] = len(matched)

        if result["matrix_features"] > 0:

            result["match_percent"] = round(
                len(matched) / len(matrix_features) * 100, 2
            )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if result["match_percent"] >= 95:

            result["status"] = "EXCELLENT_MATCH"

        elif result["match_percent"] >= 80:

            result["status"] = "GOOD_MATCH"

        elif result["match_percent"] >= 50:

            result["status"] = "PARTIAL_MATCH"

        else:

            result["status"] = "POOR_MATCH"

    except Exception as e:

        result["issue"] = str(e)

    return result


# ============================================================
# 12. Run validation
# ============================================================

results = []

print("\n" + "=" * 80)
print("PROBE → ANNOTATION MATCH VALIDATION")
print("=" * 80)

for item in DATASETS:

    gse, matrix, gpl, annotation, annotation_type = item

    print(f"Checking {gse} ({gpl})...")

    result = analyze_dataset(gse, matrix, gpl, annotation, annotation_type)

    results.append(result)


# ============================================================
# 13. Save
# ============================================================

summary = pd.DataFrame(results)

summary.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# 14. Final match summary
# ============================================================

print("\n" + "=" * 80)
print("FINAL MATCH SUMMARY")
print("=" * 80)

for _, row in summary.iterrows():

    print(
        f"{row['gse_id']:<10} "
        f"{row['gpl_id']:<8} "
        f"features={row['matrix_features']:<8,} "
        f"annotation={row['annotation_rows']:<8,} "
        f"matched={row['matched_features']:<8,} "
        f"match={row['match_percent']:>6.2f}% "
        f"{row['status']}"
    )


# ============================================================
# 15. Identifier summary
# ============================================================

print("\n" + "-" * 80)
print("ANNOTATION IDENTIFIERS")
print("-" * 80)

for _, row in summary.iterrows():

    print(
        f"{row['gse_id']:<10} "
        f"{row['gpl_id']:<8} "
        f"ID={row['annotation_id_column'] or 'NONE':<15} "
        f"SYMBOL={row['gene_symbol_column'] or 'NONE':<20} "
        f"GENE_ID={row['gene_id_column'] or 'NONE'}"
    )


# ============================================================
# 16. Problems
# ============================================================

print("\n" + "-" * 80)
print("ITEMS REQUIRING REVIEW")
print("-" * 80)

problems = summary[
    summary["status"].isin(["POOR_MATCH", "PARTIAL_MATCH", "ERROR"])
    | summary["issue"].ne("")
]

if problems.empty:

    print("No major matching problems detected.")

else:

    for _, row in problems.iterrows():

        issue = row["issue"]

        print(
            f"{row['gse_id']} / "
            f"{row['gpl_id']} : "
            f"{row['status']}" + (f" - {issue}" if issue else "")
        )


# ============================================================
# 17. Output
# ============================================================

print("\n" + "=" * 80)
print("OUTPUT")
print("=" * 80)

print(OUTPUT_FILE)

print("\nSTEP 7 COMPLETE")

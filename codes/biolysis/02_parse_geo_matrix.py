from pathlib import Path
import pandas as pd
import gzip

# ============================================================
# STEP 2
# GEO SERIES MATRIX PARSER
# ============================================================

# This script ONLY parses the GEO matrix files.
#
# Outputs are stored inside:
#
# D:\Dementia2\codes\biolysis\step02_parsed\
#
# ============================================================


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

BIOLOGY_DIR = Path(__file__).resolve().parent

PROJECT_DIR = BIOLOGY_DIR.parent.parent

DATA_DIR = PROJECT_DIR / "data"
MATRIX_DIR = DATA_DIR / "matrix"

OUTPUT_DIR = BIOLOGY_DIR / "step02_parsed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. DATASET CONFIGURATION
# ------------------------------------------------------------

# IMPORTANT:
# GSE16759 has two local matrices.
# For the main dementia/mRNA analysis we use GPL570.
#
# GSE12685 is currently missing locally and is skipped.

DATASETS = {
    "GSE1297": "GSE1297_series_matrix.txt",
    "GSE4757": "GSE4757_series_matrix.txt",
    "GSE5281": "GSE5281_series_matrix.txt",
    "GSE6834": "GSE6834_series_matrix.txt",
    "GSE15222": "GSE15222_series_matrix.txt",
    "GSE16759": "GSE16759-GPL570_series_matrix.txt",
    "GSE26927": "GSE26927_series_matrix.txt",
    "GSE28146": "GSE28146_series_matrix.txt",
    "GSE29378": "GSE29378_series_matrix.txt",
    "GSE33000": "GSE33000_series_matrix.txt",
    "GSE36980": "GSE36980_series_matrix.txt",
    "GSE37263": "GSE37263_series_matrix.txt",
    "GSE44770": "GSE44770_series_matrix.txt",
}


# ------------------------------------------------------------
# 3. OPEN GEO FILE
# ------------------------------------------------------------


def open_geo_file(path):

    if str(path).lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")

    return open(path, "r", encoding="utf-8", errors="replace")


# ------------------------------------------------------------
# 4. PARSE ONE GEO MATRIX
# ------------------------------------------------------------


def parse_geo_matrix(path):

    print()
    print("=" * 80)
    print(f"PARSING: {path.name}")
    print("=" * 80)

    sample_accessions = []
    sample_titles = []
    sample_characteristics = []

    table_lines = []

    inside_table = False

    # --------------------------------------------------------
    # Read file
    # --------------------------------------------------------

    with open_geo_file(path) as f:

        for raw_line in f:

            line = raw_line.rstrip("\r\n")

            # ------------------------------------------------
            # Sample GEO accession
            # ------------------------------------------------

            if line.startswith("!Sample_geo_accession"):

                parts = line.split("\t")

                sample_accessions = [x.strip().strip('"') for x in parts[1:]]

                continue

            # ------------------------------------------------
            # Sample title
            # ------------------------------------------------

            if line.startswith("!Sample_title"):

                parts = line.split("\t")

                sample_titles = [x.strip().strip('"') for x in parts[1:]]

                continue

            # ------------------------------------------------
            # Sample characteristics
            # ------------------------------------------------

            if line.startswith("!Sample_characteristics_ch1"):

                parts = line.split("\t")

                sample_characteristics = [x.strip().strip('"') for x in parts[1:]]

                continue

            # ------------------------------------------------
            # Expression table START
            # ------------------------------------------------

            if line.startswith("!series_matrix_table_begin"):

                inside_table = True
                continue

            # ------------------------------------------------
            # Expression table END
            # ------------------------------------------------

            if line.startswith("!series_matrix_table_end"):

                inside_table = False
                break

            # ------------------------------------------------
            # Collect expression rows
            # ------------------------------------------------

            if inside_table:

                if line.strip():
                    table_lines.append(line)

    # --------------------------------------------------------
    # Check expression table
    # --------------------------------------------------------

    if not table_lines:

        raise RuntimeError("No expression table found.")

    # --------------------------------------------------------
    # First line = expression header
    # --------------------------------------------------------

    header = table_lines[0].split("\t")

    print(f"Header columns: {len(header)}")

    if len(header) < 2:

        raise RuntimeError("Expression table header contains fewer than " "2 columns.")

    # First column is normally ID_REF
    feature_column = header[0].strip().strip('"')

    sample_columns = [x.strip().strip('"') for x in header[1:]]

    print(f"Feature column: {feature_column}")
    print(f"Expression sample columns: {len(sample_columns)}")

    # --------------------------------------------------------
    # Build expression dataframe
    # --------------------------------------------------------

    data_rows = []

    for line in table_lines[1:]:

        parts = line.split("\t")

        # Handle malformed rows
        if len(parts) < len(header):

            parts += [""] * (len(header) - len(parts))

        elif len(parts) > len(header):

            parts = parts[: len(header)]

        data_rows.append(parts)

    expression = pd.DataFrame(data_rows, columns=header)

    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    expression.columns = [str(c).strip().strip('"') for c in expression.columns]

    feature_column = expression.columns[0]

    # --------------------------------------------------------
    # Set feature IDs as index
    # --------------------------------------------------------

    expression[feature_column] = (
        expression[feature_column].astype(str).str.strip().str.strip('"')
    )

    expression = expression.set_index(feature_column)

    # --------------------------------------------------------
    # Convert expression values to numeric
    # --------------------------------------------------------

    for column in expression.columns:

        expression[column] = pd.to_numeric(expression[column], errors="coerce")

    # --------------------------------------------------------
    # Remove completely empty columns
    # --------------------------------------------------------

    expression = expression.dropna(axis=1, how="all")

    # --------------------------------------------------------
    # Remove completely empty rows
    # --------------------------------------------------------

    expression = expression.dropna(axis=0, how="all")

    # --------------------------------------------------------
    # Match metadata sample count
    # --------------------------------------------------------

    expression_samples = list(expression.columns)

    print()
    print("Metadata samples:")
    print(f"  GEO accessions : {len(sample_accessions)}")
    print(f"  Titles         : {len(sample_titles)}")
    print(f"  Characteristics: {len(sample_characteristics)}")

    print()
    print("Expression:")
    print(f"  Features       : {expression.shape[0]}")
    print(f"  Samples        : {expression.shape[1]}")

    # --------------------------------------------------------
    # Verify sample IDs
    # --------------------------------------------------------

    if sample_accessions:

        print()
        print("First sample accession:")
        print(f"  {sample_accessions[0]}")

        print("First expression column:")
        print(f"  {expression_samples[0]}")

        # GEO expression headers should normally match GSM IDs
        if sample_accessions == expression_samples:

            print("Sample ID check  : MATCH")

        else:

            print("Sample ID check  : DIFFERENT")

            # Check overlap
            overlap = set(sample_accessions) & set(expression_samples)

            print(f"Common sample IDs: {len(overlap)}")

    # --------------------------------------------------------
    # Build sample metadata
    # --------------------------------------------------------

    metadata = pd.DataFrame()

    if sample_accessions:

        metadata["GSM"] = sample_accessions

    else:

        metadata["GSM"] = expression_samples

    # --------------------------------------------------------
    # Add titles
    # --------------------------------------------------------

    if len(sample_titles) == len(metadata):

        metadata["Title"] = sample_titles

    else:

        metadata["Title"] = ""

    # --------------------------------------------------------
    # Add characteristics
    # --------------------------------------------------------

    if len(sample_characteristics) == len(metadata):

        metadata["Characteristics"] = sample_characteristics

    else:

        metadata["Characteristics"] = ""

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    gse = path.name.split("_")[0]

    if gse.startswith("GSE16759-GPL570"):

        gse = "GSE16759"

    expression_file = OUTPUT_DIR / f"{gse}_expression.csv"

    metadata_file = OUTPUT_DIR / f"{gse}_sample_metadata.csv"

    expression.to_csv(expression_file)

    metadata.to_csv(metadata_file, index=False)

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("Saved:")
    print(f"  Expression: {expression_file}")
    print(f"  Metadata  : {metadata_file}")

    return expression, metadata


# ------------------------------------------------------------
# 5. MAIN
# ------------------------------------------------------------


def main():

    print()
    print("=" * 80)
    print("STEP 2 — GEO SERIES MATRIX PARSING")
    print("=" * 80)

    print()
    print(f"Project : {PROJECT_DIR}")
    print(f"Matrix  : {MATRIX_DIR}")
    print(f"Output  : {OUTPUT_DIR}")

    success = []
    failed = []

    # --------------------------------------------------------
    # Process datasets one by one
    # --------------------------------------------------------

    for gse, filename in DATASETS.items():

        path = MATRIX_DIR / filename

        print()
        print("-" * 80)
        print(f"{gse}")
        print("-" * 80)

        if not path.exists():

            print(f"WARNING: matrix not found: {filename}")

            failed.append((gse, "FILE_NOT_FOUND"))

            continue

        try:

            expression, metadata = parse_geo_matrix(path)

            success.append(gse)

        except Exception as e:

            print()
            print(f"ERROR: {type(e).__name__}: {e}")

            failed.append((gse, f"{type(e).__name__}: {e}"))

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 2 SUMMARY")
    print("=" * 80)

    print()
    print(f"Successful datasets: {len(success)}")

    for gse in success:
        print(f"  ✅ {gse}")

    print()
    print(f"Failed datasets: {len(failed)}")

    for gse, reason in failed:
        print(f"  ❌ {gse} — {reason}")

    print()
    print(f"Output directory:")
    print(f"  {OUTPUT_DIR}")

    print()
    print("=" * 80)
    print("STEP 2 FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    main()

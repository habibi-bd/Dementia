from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"D:\Dementia2")
MATRIX_DIR = PROJECT_DIR / "data" / "matrix"
GPL_DIR = PROJECT_DIR / "data" / "gpl"


# ============================================================
# DATASETS
# ============================================================

DATASETS = [
    ("GSE33000", "GSE33000_series_matrix.txt", "GPL4372"),
    ("GSE29378", "GSE29378_series_matrix.txt", "GPL6947"),
    ("GSE44770", "GSE44770_series_matrix.txt", "GPL4372"),
    ("GSE36980", "GSE36980_series_matrix.txt", "GPL6244"),
    ("GSE28146", "GSE28146_series_matrix.txt", "GPL570"),
    ("GSE5281", "GSE5281_series_matrix.txt", "GPL570"),
    ("GSE1297", "GSE1297_series_matrix.txt", "GPL96"),
    ("GSE4757", "GSE4757_series_matrix.txt", "GPL570"),
]


# ============================================================
# ANNOTATION FILES
# ============================================================

ANNOTATIONS = {
    "GPL96": "GPL96_annotation.csv",
    "GPL570": "GPL570_annotation.csv",
    "GPL4372": "GPL4372_annotation.csv",
    "GPL6244": "GPL6244_annotation.csv",
    "GPL6947": "GPL6947_annotation.csv",
}


# ============================================================
# FIND MATRIX HEADER
# ============================================================

def find_header_line(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as f:

        for i, line in enumerate(f):

            if line.startswith("!"):
                continue

            if "\t" in line:
                return i

    raise RuntimeError(
        "Expression matrix header not found."
    )


# ============================================================
# READ ALL MATRIX PROBE IDS
# ============================================================

def get_matrix_probes(file_path):

    header_line = find_header_line(file_path)

    df = pd.read_csv(
        file_path,
        sep="\t",
        skiprows=header_line,
        usecols=[0],
        dtype=str,
        low_memory=False
    )

    column = df.columns[0]

    probes = (
        df[column]
        .astype(str)
        .str.strip()
    )

    probes = probes[
        probes.ne("")
        & probes.ne("nan")
    ]

    return set(probes)


# ============================================================
# ANALYZE ONE DATASET
# ============================================================

def analyze_dataset(
    gse,
    matrix_filename,
    gpl,
    annotation_filename
):

    matrix_file = MATRIX_DIR / matrix_filename
    annotation_file = GPL_DIR / annotation_filename

    print(
        f"\nChecking {gse} / {gpl}..."
    )

    # --------------------------------------------------------
    # Matrix
    # --------------------------------------------------------

    probes = get_matrix_probes(
        matrix_file
    )

    # --------------------------------------------------------
    # Annotation
    # --------------------------------------------------------

    annotation = pd.read_csv(
        annotation_file,
        dtype=str,
        low_memory=False
    )

    annotation.columns = [
        str(c).strip()
        for c in annotation.columns
    ]

    annotation["ID"] = (
        annotation["ID"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Probe matching
    # --------------------------------------------------------

    annotation_ids = set(
        annotation["ID"]
    )

    matched = (
        probes
        .intersection(annotation_ids)
    )

    match_percent = (
        len(matched)
        / len(probes)
        * 100
    )

    # --------------------------------------------------------
    # Matched annotation
    # --------------------------------------------------------

    mapped = annotation[
        annotation["ID"].isin(matched)
    ].copy()

    # --------------------------------------------------------
    # Gene symbol
    # --------------------------------------------------------

    if "Gene symbol" in mapped.columns:

        symbols = (
            mapped["Gene symbol"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    elif "Gene Symbol" in mapped.columns:

        symbols = (
            mapped["Gene Symbol"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    else:

        symbols = pd.Series(
            "",
            index=mapped.index
        )

    valid_symbol = (
        symbols.ne("")
        & symbols.ne("nan")
        & symbols.ne("NA")
        & symbols.ne("---")
    )

    # --------------------------------------------------------
    # Gene ID
    # --------------------------------------------------------

    if "Gene ID" in mapped.columns:

        gene_ids = (
            mapped["Gene ID"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    elif "ENTREZ_GENE_ID" in mapped.columns:

        gene_ids = (
            mapped["ENTREZ_GENE_ID"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    else:

        gene_ids = pd.Series(
            "",
            index=mapped.index
        )

    valid_gene_id = (
        gene_ids.ne("")
        & gene_ids.ne("nan")
        & gene_ids.ne("NA")
        & gene_ids.ne("---")
    )

    valid_both = (
        valid_symbol
        & valid_gene_id
    )

    # --------------------------------------------------------
    # Unique genes
    # --------------------------------------------------------

    unique_symbols = (
        symbols[valid_symbol]
        .nunique()
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {
        "gse_id": gse,
        "gpl_id": gpl,
        "matrix_probes": len(probes),
        "annotation_probes": len(annotation_ids),
        "matched_probes": len(matched),
        "match_percent": match_percent,
        "valid_gene_symbol_probes": int(
            valid_symbol.sum()
        ),
        "valid_gene_id_probes": int(
            valid_gene_id.sum()
        ),
        "valid_both_probes": int(
            valid_both.sum()
        ),
        "unique_gene_symbols": int(
            unique_symbols
        ),
    }


# ============================================================
# RUN
# ============================================================

print("\n" + "=" * 80)
print("MAJOR PLATFORM GENE MAPPING COVERAGE")
print("=" * 80)

results = []

for gse, matrix, gpl in DATASETS:

    annotation = ANNOTATIONS[gpl]

    result = analyze_dataset(
        gse,
        matrix,
        gpl,
        annotation
    )

    results.append(result)


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame(results)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

for _, row in summary.iterrows():

    print(
        f"{row['gse_id']:<10} "
        f"{row['gpl_id']:<8} "
        f"probes={row['matrix_probes']:<8,} "
        f"matched={row['matched_probes']:<8,} "
        f"match={row['match_percent']:>6.2f}% "
        f"gene_probes={row['valid_both_probes']:<8,} "
        f"genes={row['unique_gene_symbols']:<8,}"
    )


# ============================================================
# SAVE
# ============================================================

output_file = (
    PROJECT_DIR
    / "codes"
    / "major_platform_mapping_summary.csv"
)

summary.to_csv(
    output_file,
    index=False
)

print("\n" + "-" * 80)
print("OUTPUT")
print("-" * 80)

print(output_file)

print("\nSTEP 10 COMPLETE")
from pathlib import Path
import pandas as pd

# ============================================================
# 1. Directories
# ============================================================

PROJECT_DIR = Path(r"D:\Dementia2")
GPL_DIR = PROJECT_DIR / "data" / "gpl"
OUTPUT_FILE = PROJECT_DIR / "codes" / "gpl_annotation_summary.csv"


# ============================================================
# 2. Platforms used by our datasets
# ============================================================

GPL_LIST = [
    "GPL96",
    "GPL570",
    "GPL2700",
    "GPL4372",
    "GPL4757",
    "GPL5175",
    "GPL6244",
    "GPL6947",
    "GPL8757",
]


# ============================================================
# 3. Candidate annotation files
# ============================================================


def get_annotation_files(gpl_id):

    files = []

    for file in sorted(GPL_DIR.glob(f"{gpl_id}*")):

        # Skip compressed annotation files for now
        if file.suffix.lower() == ".gz":
            continue

        # Skip SOFT files in this step
        if file.suffix.lower() == ".soft":
            continue

        files.append(file)

    return files


# ============================================================
# 4. Read only the header and a few rows
# ============================================================


def inspect_file(file_path):

    result = {
        "file": file_path.name,
        "status": "ERROR",
        "separator": "",
        "n_columns": 0,
        "columns": "",
        "probe_candidates": "",
        "gene_symbol_candidates": "",
        "gene_id_candidates": "",
        "error": "",
    }

    try:

        # ----------------------------------------------------
        # Try comma-separated
        # ----------------------------------------------------

        df = pd.read_csv(file_path, sep=",", nrows=5, low_memory=False)

        separator = ","

        # ----------------------------------------------------
        # If only one column, try tab-separated
        # ----------------------------------------------------

        if len(df.columns) == 1:

            df = pd.read_csv(file_path, sep="\t", nrows=5, low_memory=False)

            separator = "\\t"

        result["status"] = "OK"
        result["separator"] = separator
        result["n_columns"] = len(df.columns)

        columns = [str(col).strip() for col in df.columns]

        result["columns"] = " | ".join(columns)

        # ----------------------------------------------------
        # Identify likely probe columns
        # ----------------------------------------------------

        probe_keywords = [
            "probe",
            "probe_id",
            "probeid",
            "id_ref",
            "id ref",
            "platform",
        ]

        probe_candidates = [
            col
            for col in columns
            if any(keyword in col.lower() for keyword in probe_keywords)
        ]

        result["probe_candidates"] = " | ".join(probe_candidates)

        # ----------------------------------------------------
        # Identify likely gene-symbol columns
        # ----------------------------------------------------

        symbol_keywords = [
            "gene symbol",
            "gene_symbol",
            "genesymbol",
            "symbol",
            "gene name",
        ]

        symbol_candidates = [
            col
            for col in columns
            if any(keyword in col.lower() for keyword in symbol_keywords)
        ]

        result["gene_symbol_candidates"] = " | ".join(symbol_candidates)

        # ----------------------------------------------------
        # Identify likely gene-ID columns
        # ----------------------------------------------------

        gene_id_keywords = [
            "entrez",
            "gene id",
            "gene_id",
            "geneid",
            "ensembl",
            "gene accession",
        ]

        gene_id_candidates = [
            col
            for col in columns
            if any(keyword in col.lower() for keyword in gene_id_keywords)
        ]

        result["gene_id_candidates"] = " | ".join(gene_id_candidates)

    except Exception as e:

        result["error"] = str(e)

    return result


# ============================================================
# 5. Inspect all annotation files
# ============================================================

results = []

for gpl_id in GPL_LIST:

    files = get_annotation_files(gpl_id)

    if not files:

        results.append(
            {
                "gpl_id": gpl_id,
                "file": "",
                "status": "NO_FILE",
                "separator": "",
                "n_columns": 0,
                "columns": "",
                "probe_candidates": "",
                "gene_symbol_candidates": "",
                "gene_id_candidates": "",
                "error": "",
            }
        )

        continue

    for file_path in files:

        result = inspect_file(file_path)

        result["gpl_id"] = gpl_id

        results.append(result)


# ============================================================
# 6. Create summary table
# ============================================================

summary = pd.DataFrame(results)

summary = summary[
    [
        "gpl_id",
        "file",
        "status",
        "separator",
        "n_columns",
        "probe_candidates",
        "gene_symbol_candidates",
        "gene_id_candidates",
        "columns",
        "error",
    ]
]


# ============================================================
# 7. Save summary
# ============================================================

summary.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# 8. Print compact diagnostic summary
# ============================================================

print("\n" + "=" * 80)
print("GPL ANNOTATION DIAGNOSTIC SUMMARY")
print("=" * 80)

print(f"Platforms checked : {len(GPL_LIST)}")

print(f"Files inspected   : {len(summary)}")

print(f"Readable files    : " f"{(summary['status'] == 'OK').sum()}")

print(f"Failed files      : " f"{(summary['status'] == 'ERROR').sum()}")

print(f"Missing platforms : " f"{(summary['status'] == 'NO_FILE').sum()}")


# ============================================================
# 9. Compact platform-level report
# ============================================================

print("\n" + "-" * 80)
print("PLATFORM SUMMARY")
print("-" * 80)

for gpl_id in GPL_LIST:

    rows = summary[summary["gpl_id"] == gpl_id]

    print(f"\n{gpl_id}")

    if rows.empty:
        print("  STATUS: NO FILE")
        continue

    for _, row in rows.iterrows():

        print(f"  File       : {row['file']}")

        print(f"  Status     : {row['status']}")

        if row["status"] == "OK":

            print(f"  Columns    : {row['n_columns']}")

            print(f"  Probe      : " f"{row['probe_candidates'] or 'NONE DETECTED'}")

            print(
                f"  Gene symbol: " f"{row['gene_symbol_candidates'] or 'NONE DETECTED'}"
            )

            print(f"  Gene ID    : " f"{row['gene_id_candidates'] or 'NONE DETECTED'}")

        elif row["status"] == "ERROR":

            print(f"  Error      : {row['error']}")


# ============================================================
# 10. Important warnings
# ============================================================

print("\n" + "-" * 80)
print("DIAGNOSTIC WARNINGS")
print("-" * 80)

for _, row in summary.iterrows():

    if row["status"] == "ERROR":

        print(f"[ERROR] {row['gpl_id']} / {row['file']}")

    elif row["status"] == "NO_FILE":

        print(f"[MISSING] {row['gpl_id']}")

    elif not row["probe_candidates"] or not row["gene_symbol_candidates"]:

        print(
            f"[REVIEW] {row['gpl_id']} / {row['file']} "
            f"needs manual annotation-column inspection"
        )


# ============================================================
# 11. Output location
# ============================================================

print("\n" + "=" * 80)
print("OUTPUT FILE")
print("=" * 80)

print(OUTPUT_FILE)

print("\nSTEP 4 COMPLETE")

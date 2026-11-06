from pathlib import Path
import csv
import re
import pandas as pd

# ============================================================
# 1. Directories
# ============================================================

PROJECT_DIR = Path(r"D:\Dementia2")
GPL_DIR = PROJECT_DIR / "data" / "gpl"
OUTPUT_FILE = PROJECT_DIR / "codes" / "gpl_annotation_validation.csv"


# ============================================================
# 2. Platforms
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
# 3. Keywords used for detecting columns
# ============================================================

PROBE_KEYWORDS = [
    "probe",
    "probe id",
    "probe_id",
    "probeid",
    "id_ref",
    "id ref",
    "ilmn",
    "affy",
    "cloneid",
    "clone id",
    "spotid",
    "spot id",
    "sequence",
]

SYMBOL_KEYWORDS = [
    "gene symbol",
    "gene_symbol",
    "genesymbol",
    "symbol",
]

GENE_ID_KEYWORDS = [
    "entrez",
    "entrez gene",
    "entrez_gene",
    "gene id",
    "gene_id",
    "geneid",
    "ensembl",
]


# ============================================================
# 4. Determine delimiter
# ============================================================


def detect_delimiter(file_path):

    try:

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:

            sample = f.read(20000)

        # Try common delimiters
        delimiters = ["\t", ",", ";"]

        counts = {d: sample.count(d) for d in delimiters}

        best = max(counts, key=counts.get)

        if counts[best] == 0:
            return None

        return best

    except Exception:
        return None


# ============================================================
# 5. Find a likely table header
# ============================================================


def find_header(file_path):

    try:

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:

            for line_number, line in enumerate(f):

                stripped = line.strip()

                if not stripped:
                    continue

                # Ignore GEO metadata
                if stripped.startswith("!"):
                    continue

                # Ignore SOFT metadata
                if stripped.startswith("^"):
                    continue

                # Ignore comments
                if stripped.startswith("#"):
                    continue

                delimiter = detect_delimiter(file_path)

                if delimiter is None:
                    continue

                fields = [x.strip().strip('"') for x in stripped.split(delimiter)]

                # A useful annotation header normally has
                # more than one column.
                if len(fields) >= 2:

                    return line_number, fields, delimiter

    except Exception:
        pass

    return None, None, None


# ============================================================
# 6. Detect candidate columns
# ============================================================


def find_candidates(columns, keywords):

    candidates = []

    for column in columns:

        normalized = re.sub(r"[^a-z0-9]+", " ", str(column).lower()).strip()

        for keyword in keywords:

            keyword_normalized = re.sub(r"[^a-z0-9]+", " ", keyword.lower()).strip()

            if keyword_normalized in normalized:
                candidates.append(column)
                break

    return candidates


# ============================================================
# 7. Validate a table
# ============================================================


def validate_table(file_path, gpl_id):

    result = {
        "gpl_id": gpl_id,
        "file": file_path.name,
        "status": "ERROR",
        "rows": 0,
        "columns": 0,
        "delimiter": "",
        "probe_candidates": "",
        "symbol_candidates": "",
        "gene_id_candidates": "",
        "probe_nonempty": "",
        "symbol_nonempty": "",
        "gene_id_nonempty": "",
        "error": "",
    }

    try:

        line_number, header, delimiter = find_header(file_path)

        if header is None:

            result["status"] = "NO_TABLE_HEADER"
            return result

        # ----------------------------------------------------
        # Read using detected header
        # ----------------------------------------------------

        df = pd.read_csv(
            file_path,
            sep=delimiter,
            skiprows=line_number,
            nrows=1000,
            low_memory=False,
            dtype=str,
            on_bad_lines="skip",
        )

        df.columns = [str(c).strip().strip('"') for c in df.columns]

        result["status"] = "OK"
        result["columns"] = len(df.columns)
        result["rows"] = len(df)
        result["delimiter"] = "\\t" if delimiter == "\t" else delimiter

        # ----------------------------------------------------
        # Candidate columns
        # ----------------------------------------------------

        probe_candidates = find_candidates(df.columns, PROBE_KEYWORDS)

        symbol_candidates = find_candidates(df.columns, SYMBOL_KEYWORDS)

        gene_id_candidates = find_candidates(df.columns, GENE_ID_KEYWORDS)

        result["probe_candidates"] = " | ".join(probe_candidates)

        result["symbol_candidates"] = " | ".join(symbol_candidates)

        result["gene_id_candidates"] = " | ".join(gene_id_candidates)

        # ----------------------------------------------------
        # Non-empty percentages
        # ----------------------------------------------------

        if probe_candidates:

            values = df[probe_candidates[0]]

            nonempty = (
                values.notna() & values.astype(str).str.strip().ne("")
            ).mean() * 100

            result["probe_nonempty"] = f"{nonempty:.1f}%"

        if symbol_candidates:

            values = df[symbol_candidates[0]]

            nonempty = (
                values.notna() & values.astype(str).str.strip().ne("")
            ).mean() * 100

            result["symbol_nonempty"] = f"{nonempty:.1f}%"

        if gene_id_candidates:

            values = df[gene_id_candidates[0]]

            nonempty = (
                values.notna() & values.astype(str).str.strip().ne("")
            ).mean() * 100

            result["gene_id_nonempty"] = f"{nonempty:.1f}%"

    except Exception as e:

        result["error"] = str(e)

    return result


# ============================================================
# 8. Files to inspect
# ============================================================

files_to_check = []

for gpl_id in GPL_LIST:

    for file_path in sorted(GPL_DIR.glob(f"{gpl_id}*")):

        # Skip compressed files
        if file_path.suffix.lower() == ".gz":
            continue

        files_to_check.append((gpl_id, file_path))


# ============================================================
# 9. Validate files
# ============================================================

results = []

for gpl_id, file_path in files_to_check:

    result = validate_table(file_path, gpl_id)

    results.append(result)


# ============================================================
# 10. Save detailed validation
# ============================================================

validation = pd.DataFrame(results)

validation.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# 11. Compact terminal summary
# ============================================================

print("\n" + "=" * 80)
print("GPL ANNOTATION VALIDATION SUMMARY")
print("=" * 80)

print(f"Platforms checked : {len(GPL_LIST)}")

print(f"Files checked     : {len(validation)}")

print(f"Usable tables     : " f"{(validation['status'] == 'OK').sum()}")

print(f"Problem files     : " f"{(validation['status'] != 'OK').sum()}")


# ============================================================
# 12. Platform-level summary
# ============================================================

print("\n" + "-" * 80)
print("ANNOTATION TABLES")
print("-" * 80)

for gpl_id in GPL_LIST:

    rows = validation[validation["gpl_id"] == gpl_id]

    print(f"\n{gpl_id}")

    if rows.empty:

        print("  NO FILE FOUND")
        continue

    for _, row in rows.iterrows():

        print(f"  {row['file']}")

        print(f"    Status     : {row['status']}")

        if row["status"] == "OK":

            print(
                f"    Size       : "
                f"{row['rows']} rows x "
                f"{row['columns']} columns"
            )

            print(f"    Delimiter  : " f"{row['delimiter']}")

            print(f"    Probe/ID   : " f"{row['probe_candidates'] or 'NONE'}")

            print(f"    Gene symbol: " f"{row['symbol_candidates'] or 'NONE'}")

            print(f"    Gene ID    : " f"{row['gene_id_candidates'] or 'NONE'}")

            if row["probe_nonempty"]:
                print(f"    Probe fill : " f"{row['probe_nonempty']}")

            if row["symbol_nonempty"]:
                print(f"    Symbol fill: " f"{row['symbol_nonempty']}")

            if row["gene_id_nonempty"]:
                print(f"    GeneID fill: " f"{row['gene_id_nonempty']}")

        else:

            if row["error"]:
                print(f"    Error      : " f"{row['error']}")


# ============================================================
# 13. Final warnings
# ============================================================

print("\n" + "-" * 80)
print("ITEMS REQUIRING REVIEW")
print("-" * 80)

for _, row in validation.iterrows():

    if row["status"] != "OK":

        print(f"[{row['status']}] " f"{row['gpl_id']} / {row['file']}")

    elif not row["probe_candidates"]:

        print(f"[NO PROBE COLUMN] " f"{row['gpl_id']} / {row['file']}")

    elif not row["symbol_candidates"]:

        print(f"[NO SYMBOL COLUMN] " f"{row['gpl_id']} / {row['file']}")


# ============================================================
# 14. Output
# ============================================================

print("\n" + "=" * 80)
print("OUTPUT")
print("=" * 80)

print(OUTPUT_FILE)

print("\nSTEP 5 COMPLETE")

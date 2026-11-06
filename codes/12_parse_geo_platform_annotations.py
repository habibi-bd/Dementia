from pathlib import Path
import pandas as pd

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"D:\Dementia2")

GPL_DIR = PROJECT_DIR / "data" / "gpl"


# ============================================================
# FILES
# ============================================================

FILES = {
    "GPL5175": GPL_DIR / "GPL5175 (2).txt",
    "GPL2700": GPL_DIR / "GPL2700.txt",
    "GPL4757": GPL_DIR / "GPL4757.txt",
}


# ============================================================
# FIND GEO TABLE
# ============================================================


def find_table_start(file_path):

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:

        lines = f.readlines()

    for i, line in enumerate(lines):

        line = line.rstrip("\r\n")

        # GEO platform tables usually start with
        # a line beginning with !platform_table_begin
        if "!platform_table_begin" in line.lower():

            return i + 1, lines

    return None, lines


# ============================================================
# PARSE GEO PLATFORM TABLE
# ============================================================


def parse_geo_table(file_path):

    start, lines = find_table_start(file_path)

    if start is None:

        return None, None

    # Find table end
    end = len(lines)

    for i in range(start, len(lines)):

        if "!platform_table_end" in lines[i].lower():

            end = i
            break

    table_lines = lines[start:end]

    if not table_lines:

        return None, None

    # First line = header
    header = table_lines[0].rstrip("\r\n").split("\t")

    rows = []

    for line in table_lines[1:]:

        line = line.rstrip("\r\n")

        if not line:
            continue

        values = line.split("\t")

        # Normalize row length
        if len(values) < len(header):

            values += [""] * (len(header) - len(values))

        elif len(values) > len(header):

            values = values[: len(header)]

        rows.append(values)

    df = pd.DataFrame(rows, columns=header)

    return df, header


# ============================================================
# FIND LIKELY GENE COLUMNS
# ============================================================


def find_columns(columns):

    probe_candidates = []
    symbol_candidates = []
    gene_id_candidates = []

    for col in columns:

        c = str(col).strip().lower()

        # Probe / identifier
        if (
            c in ["id", "id_ref", "probe", "probe_id", "spot_id", "ilmnid"]
            or "probe" in c
        ):

            probe_candidates.append(col)

        # Gene symbol
        if (
            "gene symbol" in c
            or c == "symbol"
            or "gene_symbol" in c
            or "gene name" in c
        ):

            symbol_candidates.append(col)

        # Gene ID
        if "entrez" in c or "gene id" in c or "gene_id" in c or "geneid" in c:

            gene_id_candidates.append(col)

    return (probe_candidates, symbol_candidates, gene_id_candidates)


# ============================================================
# MAIN
# ============================================================

print("\n" + "=" * 80)
print("GEO PLATFORM ANNOTATION PARSER")
print("=" * 80)


results = []


for gpl, file_path in FILES.items():

    print("\n" + "-" * 80)
    print(f"{gpl}")
    print("-" * 80)

    print(f"File: {file_path.name}")

    if not file_path.exists():

        print("STATUS: FILE NOT FOUND")
        continue

    df, header = parse_geo_table(file_path)

    if df is None:

        print("STATUS: GEO PLATFORM TABLE NOT FOUND")

        continue

    print(f"Rows    : {len(df):,}")

    print(f"Columns : {len(df.columns)}")

    print("\nColumns:")

    for col in df.columns:

        print(f"  {col}")

    # --------------------------------------------------------
    # Candidate columns
    # --------------------------------------------------------

    probe_cols, symbol_cols, gene_id_cols = find_columns(df.columns)

    print("\nLikely probe columns:")

    if probe_cols:

        for col in probe_cols:

            print(f"  {col}")

    else:

        print("  NONE")

    print("Likely gene symbol columns:")

    if symbol_cols:

        for col in symbol_cols:

            print(f"  {col}")

    else:

        print("  NONE")

    print("Likely gene ID columns:")

    if gene_id_cols:

        for col in gene_id_cols:

            print(f"  {col}")

    else:

        print("  NONE")

    # --------------------------------------------------------
    # Non-empty percentages for likely columns
    # --------------------------------------------------------

    print("\nNon-empty percentage:")

    candidate_cols = probe_cols + symbol_cols + gene_id_cols

    candidate_cols = list(dict.fromkeys(candidate_cols))

    for col in candidate_cols:

        values = df[col].fillna("").astype(str).str.strip()

        valid = values.ne("") & values.ne("nan") & values.ne("NA") & values.ne("---")

        print(f"  {col:<35} " f"{valid.mean() * 100:6.2f}%")

    # --------------------------------------------------------
    # First 5 rows
    # --------------------------------------------------------

    print("\nFirst 3 rows:")

    print(df.head(3).to_string(index=False, max_cols=12))

    results.append(
        {
            "gpl_id": gpl,
            "file": file_path.name,
            "rows": len(df),
            "columns": len(df.columns),
            "probe_candidates": "; ".join(map(str, probe_cols)),
            "symbol_candidates": "; ".join(map(str, symbol_cols)),
            "gene_id_candidates": "; ".join(map(str, gene_id_cols)),
        }
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(results)

output_file = PROJECT_DIR / "codes" / "geo_platform_parser_summary.csv"

summary.to_csv(output_file, index=False)


print("\n" + "=" * 80)
print("OUTPUT")
print("=" * 80)

print(output_file)

print("\nSTEP 12 COMPLETE")

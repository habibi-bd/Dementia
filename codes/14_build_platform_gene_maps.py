from pathlib import Path
import pandas as pd
import re

# ============================================================
# PATHS
# ============================================================

GPL_DIR = Path(r"D:\Dementia2\data\gpl")
OUTPUT = Path(r"D:\Dementia2\codes\platform_gene_maps")

OUTPUT.mkdir(exist_ok=True)


# ============================================================
# HELPERS
# ============================================================


def clean_text(value):
    if pd.isna(value):
        return ""

    value = str(value).strip()

    if value.lower() in {"", "nan", "na", "---", "null", "none"}:
        return ""

    return value


def first_valid(values):
    for value in values:

        value = clean_text(value)

        if value:
            return value

    return ""


# ============================================================
# GPL4372 / GPL6244 / GPL6947
# ============================================================


def build_clone_platform_map(gpl):

    file = GPL_DIR / f"{gpl}_annotation.csv"

    df = pd.read_csv(file, dtype=str, low_memory=False)

    df = df.fillna("")

    if gpl in {"GPL4372", "GPL6244", "GPL6947"}:

        result = pd.DataFrame()

        result["probe_id"] = df["ID"].map(clean_text)

        result["gene_symbol"] = df["Gene symbol"].map(clean_text)

        result["gene_id"] = df["Gene ID"].map(clean_text)

    return result


# ============================================================
# GPL570 / GPL96
# ============================================================


def build_affymetrix_map(gpl):

    file = GPL_DIR / f"{gpl}_annotation.csv"

    df = pd.read_csv(file, dtype=str, low_memory=False)

    df = df.fillna("")

    result = pd.DataFrame()

    result["probe_id"] = df["ID"].map(clean_text)

    result["gene_symbol"] = df["Gene Symbol"].map(clean_text)

    result["gene_id"] = df["ENTREZ_GENE_ID"].map(clean_text)

    return result


# ============================================================
# GPL4757
# ============================================================
def build_gpl4757_map():

    file = GPL_DIR / "GPL4757.txt"

    # --------------------------------------------------------
    # Locate the actual annotation table.
    # GPL4757 contains GEO metadata before the table.
    # The real header begins with:
    #
    # ID    Gene Symbol    GB_ACC (ref) ...
    # --------------------------------------------------------

    with open(file, "r", encoding="utf-8", errors="replace") as f:

        lines = f.readlines()

    header_index = None

    for i, line in enumerate(lines):

        clean = line.rstrip("\r\n")

        if clean.startswith("ID\tGene Symbol"):

            header_index = i
            break

    if header_index is None:

        raise ValueError("Could not find GPL4757 annotation table header.")

    # --------------------------------------------------------
    # Read only the actual annotation table
    # --------------------------------------------------------

    from io import StringIO

    table_text = "".join(lines[header_index:])

    df = pd.read_csv(StringIO(table_text), sep="\t", dtype=str, low_memory=False)

    df = df.fillna("")

    # --------------------------------------------------------
    # Probe ID
    # --------------------------------------------------------

    result = pd.DataFrame()

    result["probe_id"] = df["ID"].map(clean_text)

    # --------------------------------------------------------
    # Gene Symbol
    #
    # The "Gene Symbol" field is actually a gene
    # description containing the symbol in parentheses.
    #
    # Example:
    #
    # Homo sapiens ... (ACCN1), transcript variant...
    #
    # Extract:
    #
    # ACCN1
    # --------------------------------------------------------

    descriptions = df["Gene Symbol"].map(clean_text)

    def extract_symbol(text):

        if not text:
            return ""

        match = re.search(r"\(([A-Za-z0-9][A-Za-z0-9.-]*)\)", text)

        if match:

            return match.group(1)

        return ""

    result["gene_symbol"] = descriptions.map(extract_symbol)

    # GPL4757 does not provide Entrez IDs
    result["gene_id"] = ""

    return result


# ============================================================
# GPL5175
# ============================================================


def build_gpl5175_map():

    file = GPL_DIR / "GPL5175 (2).txt"

    # Actual table starts at line 13.
    df = pd.read_csv(file, sep="\t", skiprows=12, dtype=str, low_memory=False)

    df = df.fillna("")

    result = pd.DataFrame()

    result["probe_id"] = df["ID"].map(clean_text)

    assignments = df["gene_assignment"].map(clean_text)

    def parse_assignment(text):

        if not text or text == "---":
            return "", ""

        # Example:
        # NR_024005 // DDX11L2 // description // 2q13 // 84771

        first_assignment = text.split("///")[0].strip()

        parts = [p.strip() for p in first_assignment.split("//")]

        if len(parts) < 2:
            return "", ""

        accession = parts[0]

        symbol = parts[1]

        gene_id = ""

        # Search all fields from the assignment
        # for a numeric Entrez Gene ID.
        for part in reversed(parts):

            if re.fullmatch(r"\d+", part):
                gene_id = part
                break

        return symbol, gene_id

    parsed = assignments.map(parse_assignment)

    result["gene_symbol"] = [item[0] for item in parsed]

    result["gene_id"] = [item[1] for item in parsed]

    return result


# ============================================================
# BUILD ALL MAPS
# ============================================================

print("\n" + "=" * 80)
print("BUILDING PLATFORM → GENE MAPS")
print("=" * 80)


platform_maps = {}


# ------------------------------------------------------------
# Clone platforms
# ------------------------------------------------------------

for gpl in ["GPL4372", "GPL6244", "GPL6947"]:

    print(f"\nProcessing {gpl}...")

    platform_maps[gpl] = build_clone_platform_map(gpl)


# ------------------------------------------------------------
# Affymetrix
# ------------------------------------------------------------

for gpl in ["GPL570", "GPL96"]:

    print(f"\nProcessing {gpl}...")

    platform_maps[gpl] = build_affymetrix_map(gpl)


# ------------------------------------------------------------
# GPL4757
# ------------------------------------------------------------

print("\nProcessing GPL4757...")

platform_maps["GPL4757"] = build_gpl4757_map()


# ------------------------------------------------------------
# GPL5175
# ------------------------------------------------------------

print("\nProcessing GPL5175...")

platform_maps["GPL5175"] = build_gpl5175_map()


# ============================================================
# SAVE + SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("PLATFORM MAP SUMMARY")
print("=" * 80)

summary = []


for gpl, df in platform_maps.items():

    # Remove empty probes
    df = df[df["probe_id"] != ""].copy()

    # Remove duplicate probe IDs
    df = df.drop_duplicates(subset=["probe_id"], keep="first")

    output_file = OUTPUT / f"{gpl}_probe_to_gene.csv"

    df.to_csv(output_file, index=False)

    probe_count = len(df)

    symbol_count = df["gene_symbol"].ne("").sum()

    gene_id_count = df["gene_id"].ne("").sum()

    both_count = (df["gene_symbol"].ne("") & df["gene_id"].ne("")).sum()

    summary.append(
        {
            "gpl_id": gpl,
            "probes": probe_count,
            "gene_symbol": symbol_count,
            "gene_id": gene_id_count,
            "symbol_and_id": both_count,
        }
    )

    print(
        f"{gpl:<9}"
        f" probes={probe_count:,}"
        f" symbols={symbol_count:,}"
        f" gene_ids={gene_id_count:,}"
        f" both={both_count:,}"
    )


summary_df = pd.DataFrame(summary)

summary_file = OUTPUT / "platform_gene_map_summary.csv"

summary_df.to_csv(summary_file, index=False)


print("\n" + "-" * 80)

print(f"Output directory : {OUTPUT}")

print(f"Summary file     : {summary_file}")

print("\n" + "=" * 80)
print("STEP 14 COMPLETE")
print("=" * 80)

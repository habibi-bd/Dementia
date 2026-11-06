from pathlib import Path
import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE = Path(r"D:\Dementia2")

MATRIX_DIR = BASE / "data" / "matrix"

MAP_DIR = BASE / "codes" / "platform_gene_maps"


# ============================================================
# DATASET → PLATFORM
# ============================================================

DATASETS = {
    "GSE33000": ("GSE33000_series_matrix.txt", "GPL4372"),
    "GSE29378": ("GSE29378_series_matrix.txt", "GPL6947"),
    "GSE44770": ("GSE44770_series_matrix.txt", "GPL4372"),
    "GSE36980": ("GSE36980_series_matrix.txt", "GPL6244"),
    "GSE37263": ("GSE37263_series_matrix.txt", "GPL5175"),
    "GSE28146": ("GSE28146_series_matrix.txt", "GPL570"),
    "GSE5281": ("GSE5281_series_matrix.txt", "GPL570"),
    "GSE1297": ("GSE1297_series_matrix.txt", "GPL96"),
    "GSE4757": ("GSE4757_series_matrix.txt", "GPL570"),
    "GSE6834": ("GSE6834_series_matrix.txt", "GPL4757"),
}


# ============================================================
# FIND MATRIX HEADER
# ============================================================
 
## ttttttttttttttttttttttttttttttttttttttttttttttttttttttt
def find_header(file):

    with open(file, "r", encoding="utf-8", errors="replace") as f:

        for i, line in enumerate(f):

            if line.startswith('"ID_REF"') or line.startswith("ID_REF"):

                return i

    raise ValueError(f"ID_REF header not found: {file}")


# ============================================================
# LOAD PROBE IDS ONLY
# ============================================================


def load_probe_ids(file):

    header = find_header(file)

    df = pd.read_csv(
        file, sep="\t", skiprows=header, usecols=[0], dtype=str, low_memory=False
    )

    probe_col = df.columns[0]

    probes = df[probe_col].astype(str).str.strip()

    probes = probes[probes != ""]

    probes = probes[probes != "nan"]

    return probes


# ============================================================
# LOAD PLATFORM MAP
# ============================================================


def load_map(gpl):

    file = MAP_DIR / f"{gpl}_probe_to_gene.csv"

    df = pd.read_csv(file, dtype=str).fillna("")

    df["probe_id"] = df["probe_id"].astype(str).str.strip()

    df["gene_symbol"] = df["gene_symbol"].astype(str).str.strip()

    return df


# ============================================================
# MAIN
# ============================================================

print("\n" + "=" * 80)
print("DATASET → GENE MAPPING VALIDATION")
print("=" * 80)


results = []


for gse, (matrix_name, gpl) in DATASETS.items():

    print(f"\nChecking {gse} ({gpl})...")

    matrix_file = MATRIX_DIR / matrix_name

    map_file = MAP_DIR / f"{gpl}_probe_to_gene.csv"

    if not matrix_file.exists():

        print("  STATUS : MATRIX MISSING")

        continue

    if not map_file.exists():

        print("  STATUS : PLATFORM MAP MISSING")

        continue

    # --------------------------------------------------------
    # Matrix probes
    # --------------------------------------------------------

    probes = load_probe_ids(matrix_file)

    matrix_probe_set = set(probes)

    # --------------------------------------------------------
    # Annotation map
    # --------------------------------------------------------

    mapping = load_map(gpl)

    annotation_probe_set = set(mapping["probe_id"])

    matched = matrix_probe_set & annotation_probe_set

    matched_mapping = mapping[mapping["probe_id"].isin(matched)]

    with_gene = matched_mapping[matched_mapping["gene_symbol"] != ""]

    unique_genes = with_gene["gene_symbol"].nunique()

    # --------------------------------------------------------
    # Rates
    # --------------------------------------------------------

    probe_match_rate = len(matched) / len(matrix_probe_set) * 100

    gene_annotation_rate = len(with_gene) / len(matched) * 100 if matched else 0

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print(f"  Matrix probes       : " f"{len(matrix_probe_set):,}")

    print(f"  Matched probes      : " f"{len(matched):,}")

    print(f"  Probe match rate    : " f"{probe_match_rate:.2f}%")

    print(f"  With gene symbol    : " f"{len(with_gene):,}")

    print(f"  Gene annotation     : " f"{gene_annotation_rate:.2f}%")

    print(f"  Unique genes        : " f"{unique_genes:,}")

    results.append(
        {
            "gse_id": gse,
            "gpl_id": gpl,
            "matrix_probes": len(matrix_probe_set),
            "matched_probes": len(matched),
            "probe_match_rate": probe_match_rate,
            "gene_probes": len(with_gene),
            "gene_annotation_rate": gene_annotation_rate,
            "unique_genes": unique_genes,
        }
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = pd.DataFrame(results)

output = BASE / "codes" / "dataset_gene_mapping_validation.csv"

summary.to_csv(output, index=False)


print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

for _, row in summary.iterrows():

    print(
        f"{row['gse_id']:<9} "
        f"{row['gpl_id']:<8} "
        f"probes={int(row['matrix_probes']):,} "
        f"matched={int(row['matched_probes']):,} "
        f"match={row['probe_match_rate']:6.2f}% "
        f"gene_probes={int(row['gene_probes']):,} "
        f"genes={int(row['unique_genes']):,}"
    )


print("\n" + "-" * 80)

print(f"Output : {output}")

print("\n" + "=" * 80)
print("STEP 15 COMPLETE")
print("=" * 80)





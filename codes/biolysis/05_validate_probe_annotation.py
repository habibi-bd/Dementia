# # ============================================================
# # STEP 5 — ROBUST PROBE ↔ GPL ANNOTATION VALIDATION
# # ============================================================

# from pathlib import Path
# import pandas as pd
# import numpy as np
# import gzip
# import re
# import csv
# import warnings

# warnings.filterwarnings("ignore")

# BASE = Path(r"D:\Dementia2")
# DATA = BASE / "data"
# GPL_DIR = DATA / "gpl"
# MATRIX_DIR = DATA / "matrix"
# PARSED_DIR = Path(r"D:\Dementia2\codes\biolysis\step02_parsed")
# OUT_DIR = Path(r"D:\Dementia2\codes\biolysis\step05_probe_annotation")

# OUT_DIR.mkdir(parents=True, exist_ok=True)

# # ------------------------------------------------------------
# # ONLY THESE 12 DATASETS
# # ------------------------------------------------------------

# DATASETS = {
#     "GSE1297": "GPL96",
#     "GSE4757": "GPL570",
#     "GSE5281": "GPL570",
#     "GSE6834": "GPL4757",
#     "GSE15222": "GPL2700",
#     "GSE16759": "GPL570",
#     "GSE28146": "GPL570",
#     "GSE29378": "GPL6947",
#     "GSE33000": "GPL4372",
#     "GSE36980": "GPL6244",
#     "GSE37263": "GPL5175",
#     "GSE44770": "GPL4372",
# }

# # ------------------------------------------------------------
# # NORMALIZE IDENTIFIERS
# # ------------------------------------------------------------


# def norm_id(x):
#     if pd.isna(x):
#         return ""

#     x = str(x).strip().strip('"').strip("'")

#     if not x:
#         return ""

#     # Remove accidental pandas float conversion
#     if re.fullmatch(r"\d+\.0", x):
#         x = x[:-2]

#     return x.lower()


# def clean_string(x):
#     if pd.isna(x):
#         return ""

#     x = str(x).strip().strip('"').strip("'")

#     if x.lower() in {
#         "",
#         "nan",
#         "none",
#         "null",
#         "na",
#         "n/a",
#         "---",
#         "null",
#         "not available",
#         "not applicable",
#     }:
#         return ""

#     return x


# # ------------------------------------------------------------
# # FIND PARSED EXPRESSION FILE
# # ------------------------------------------------------------


# def find_expression_file(gse):

#     candidates = [
#         PARSED_DIR / f"{gse}.csv",
#         PARSED_DIR / f"{gse}_parsed.csv",
#         PARSED_DIR / f"{gse}.tsv",
#         PARSED_DIR / f"{gse}_parsed.tsv",
#     ]

#     for p in candidates:
#         if p.exists():
#             return p

#     matches = list(PARSED_DIR.glob(f"{gse}*"))

#     for p in matches:
#         if p.suffix.lower() in {".csv", ".tsv", ".txt"}:
#             return p

#     return None


# # ------------------------------------------------------------
# # LOAD EXPRESSION PROBE IDS
# # ------------------------------------------------------------


# def load_expression_ids(gse):

#     path = find_expression_file(gse)

#     if path is None:
#         raise FileNotFoundError(f"No parsed expression file found for {gse}")

#     print(f"  Parsed expression: {path.name}")

#     # Try normal CSV
#     try:
#         df = pd.read_csv(path, sep=None, engine="python", nrows=10)
#     except Exception:
#         df = pd.read_csv(path, sep="\t", nrows=10)

#     if df.shape[1] < 2:
#         raise ValueError("Expression file has fewer than 2 columns")

#     # First column is normally probe ID
#     probe_col = df.columns[0]

#     # If first column looks like unnamed index, still use it
#     ids = df[probe_col].astype(str).map(norm_id)

#     # If first column is not suitable, test all columns
#     if ids.replace("", np.nan).notna().sum() < 5:

#         best_col = None
#         best_count = 0

#         for col in df.columns:
#             vals = df[col].astype(str).map(norm_id)
#             count = vals.replace("", np.nan).notna().sum()

#             if count > best_count:
#                 best_count = count
#                 best_col = col

#         if best_col is not None:
#             probe_col = best_col

#     # Read only first column from complete file
#     try:
#         full = pd.read_csv(path, sep=None, engine="python", usecols=[probe_col])
#     except Exception:
#         full = pd.read_csv(path, sep="\t", usecols=[probe_col])

#     ids = full.iloc[:, 0].astype(str).map(norm_id)

#     ids = ids[(ids != "") & (~ids.isin(["sample", "probe", "probe_id", "id"]))]

#     ids = pd.Index(ids.unique())

#     return ids, probe_col, path


# # ------------------------------------------------------------
# # FIND ALL POSSIBLE GPL FILES
# # ------------------------------------------------------------


# def find_gpl_files(gpl):

#     candidates = []

#     patterns = [
#         f"{gpl}*",
#         f"{gpl.lower()}*",
#     ]

#     for pattern in patterns:
#         candidates.extend(GPL_DIR.glob(pattern))

#     # remove duplicates
#     unique = []
#     seen = set()

#     for p in candidates:
#         if p.is_file() and p not in seen:
#             unique.append(p)
#             seen.add(p)

#     # preferred order
#     priority = {
#         ".soft": 0,
#         ".txt": 1,
#         ".csv": 2,
#         ".gz": 3,
#     }

#     unique.sort(key=lambda p: (priority.get(p.suffix.lower(), 9), p.name.lower()))

#     return unique


# # ------------------------------------------------------------
# # OPEN TEXT FILE
# # ------------------------------------------------------------


# def open_text(path):

#     if path.suffix.lower() == ".gz":
#         return gzip.open(path, "rt", encoding="utf-8", errors="replace")

#     return open(path, "r", encoding="utf-8", errors="replace")


# # ------------------------------------------------------------
# # DETECT GEO PLATFORM TABLE
# # ------------------------------------------------------------


# def extract_platform_table(path):

#     with open_text(path) as f:

#         lines = f.readlines()

#     begin = None
#     end = None

#     for i, line in enumerate(lines):

#         s = line.strip().lower()

#         if "!platform_table_begin" in s:
#             begin = i + 1

#         elif "!platform_table_end" in s and begin is not None:
#             end = i
#             break

#     if begin is not None:

#         if end is None:
#             end = len(lines)

#         table_lines = lines[begin:end]

#         # Remove empty/comment lines
#         table_lines = [x for x in table_lines if x.strip() and not x.startswith("!")]

#         if not table_lines:
#             return None

#         # GEO SOFT tables are tab-separated
#         try:
#             df = pd.read_csv(
#                 pd.io.common.StringIO("".join(table_lines)),
#                 sep="\t",
#                 dtype=str,
#                 low_memory=False,
#             )

#             if df.shape[1] > 1:
#                 return df

#         except Exception:
#             pass

#     return None


# # ------------------------------------------------------------
# # GENERIC TABLE PARSER
# # ------------------------------------------------------------


# def parse_generic_table(path):

#     # First try GEO platform table
#     df = extract_platform_table(path)

#     if df is not None and df.shape[1] > 1:
#         return df

#     # Read raw lines
#     with open_text(path) as f:
#         lines = f.readlines()

#     # Remove GEO metadata/comment lines
#     useful = []

#     for line in lines:

#         s = line.strip()

#         if not s:
#             continue

#         if s.startswith("!"):
#             continue

#         if s.startswith("^"):
#             continue

#         useful.append(line)

#     if len(useful) < 2:
#         return None

#     sample = useful[:20]

#     # Detect delimiter
#     delimiters = ["\t", ",", ";"]

#     best_delim = "\t"
#     best_score = 0

#     for delim in delimiters:

#         score = 0

#         for line in sample:
#             score += line.count(delim)

#         if score > best_score:
#             best_score = score
#             best_delim = delim

#     try:

#         df = pd.read_csv(
#             pd.io.common.StringIO("".join(useful)),
#             sep=best_delim,
#             dtype=str,
#             low_memory=False,
#             on_bad_lines="skip",
#         )

#         if df.shape[1] > 1:
#             return df

#     except Exception:
#         pass

#     # Last attempt: Python CSV sniffer
#     try:

#         dialect = csv.Sniffer().sniff("".join(sample))

#         df = pd.read_csv(
#             pd.io.common.StringIO("".join(useful)),
#             sep=dialect.delimiter,
#             dtype=str,
#             low_memory=False,
#             on_bad_lines="skip",
#         )

#         if df.shape[1] > 1:
#             return df

#     except Exception:
#         pass

#     return None


# # ------------------------------------------------------------
# # DETECT PROBE COLUMN
# # ------------------------------------------------------------


# def detect_probe_column(df, expression_ids):

#     normalized_expression = set(expression_ids)

#     candidates = []

#     for col in df.columns:

#         values = df[col].astype(str).map(norm_id)

#         nonempty = values[values != ""]

#         if len(nonempty) == 0:
#             continue

#         overlap = len(set(nonempty).intersection(normalized_expression))

#         name = str(col).lower()

#         score = overlap * 100

#         if any(
#             x in name
#             for x in [
#                 "probe",
#                 "probe_id",
#                 "id",
#                 "identifier",
#                 "affy",
#                 "ilmn",
#                 "reporter",
#                 "feature",
#             ]
#         ):
#             score += 10

#         candidates.append((score, overlap, col))

#     if not candidates:
#         return None, 0

#     candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

#     return candidates[0][2], candidates[0][1]


# # ------------------------------------------------------------
# # DETECT GENE SYMBOL COLUMN
# # ------------------------------------------------------------


# def detect_gene_column(df):

#     preferred = [
#         "gene symbol",
#         "gene_symbol",
#         "genesymbol",
#         "gene_symbol",
#         "symbol",
#         "gene",
#         "gene name",
#         "gene_name",
#         "genename",
#         "gene_assignment",
#         "gene_assignment",
#         "symbol",
#     ]

#     normalized = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}

#     for wanted in preferred:

#         key = re.sub(r"[^a-z0-9]", "", wanted.lower())

#         if key in normalized:
#             return normalized[key]

#     # Fuzzy search
#     for col in df.columns:

#         name = str(col).lower()

#         if "gene" in name and (
#             "symbol" in name or "name" in name or "assignment" in name
#         ):
#             return col

#     return None


# # ------------------------------------------------------------
# # LOAD BEST GPL ANNOTATION
# # ------------------------------------------------------------


# def load_best_gpl(gpl, expression_ids):

#     files = find_gpl_files(gpl)

#     if not files:
#         return None, None, None, None

#     results = []

#     print("  GPL candidates:")

#     for path in files:

#         try:

#             df = parse_generic_table(path)

#             if df is None:
#                 print(f"    {path.name}: PARSE_FAILED")
#                 continue

#             probe_col, overlap = detect_probe_column(df, expression_ids)

#             gene_col = detect_gene_column(df)

#             annotated = 0

#             if gene_col is not None:

#                 genes = df[gene_col].astype(str).map(clean_string)

#                 annotated = int(genes.ne("").sum())

#             print(
#                 f"    {path.name}: "
#                 f"rows={len(df):,}, "
#                 f"cols={len(df.columns):,}, "
#                 f"probe={probe_col}, "
#                 f"overlap={overlap:,}, "
#                 f"gene_col={gene_col}, "
#                 f"annotated={annotated:,}"
#             )

#             if probe_col is not None:

#                 results.append((overlap, annotated, path, df, probe_col, gene_col))

#         except Exception as e:

#             print(f"    {path.name}: ERROR: {e}")

#     if not results:
#         return None, None, None, None

#     # Highest expression↔GPL probe overlap first
#     results.sort(key=lambda x: (x[0], x[1]), reverse=True)

#     return results[0][2], results[0][3], results[0][4], results[0][5]


# # ------------------------------------------------------------
# # MAIN
# # ------------------------------------------------------------

# summary = []

# print("=" * 80)
# print("STEP 5 — ROBUST PROBE ↔ GPL ANNOTATION VALIDATION")
# print("=" * 80)

# for gse, gpl in DATASETS.items():

#     print(f"\nProcessing {gse} → {gpl}")

#     try:

#         # --------------------------------------------
#         # EXPRESSION
#         # --------------------------------------------

#         expression_ids, probe_col_expr, expr_path = load_expression_ids(gse)

#         print(f"  Expression features: " f"{len(expression_ids):,}")

#         # --------------------------------------------
#         # GPL
#         # --------------------------------------------

#         gpl_path, gpl_df, gpl_probe_col, gpl_gene_col = load_best_gpl(
#             gpl, expression_ids
#         )

#         if gpl_df is None:

#             print("  ERROR: GPL could not be parsed")

#             summary.append(
#                 {
#                     "GSE": gse,
#                     "GPL": gpl,
#                     "Expression_Features": len(expression_ids),
#                     "GPL_Features": 0,
#                     "Exact_Probe_Matches": 0,
#                     "Match_Percent": 0,
#                     "Annotated_Probes": 0,
#                     "Annotation_Percent": 0,
#                     "GPL_File": "",
#                     "Probe_Column": "",
#                     "Gene_Column": "",
#                     "Status": "FAIL_GPL_PARSE",
#                 }
#             )

#             continue

#         # --------------------------------------------
#         # NORMALIZE GPL PROBES
#         # --------------------------------------------

#         gpl_probes = gpl_df[gpl_probe_col].astype(str).map(norm_id)

#         valid = gpl_probes != ""

#         gpl_probes = gpl_probes[valid]

#         # --------------------------------------------
#         # OVERLAP
#         # --------------------------------------------

#         expr_set = set(expression_ids)
#         gpl_set = set(gpl_probes)

#         matched = expr_set.intersection(gpl_set)

#         exact_matches = len(matched)

#         match_percent = (
#             exact_matches / len(expression_ids) * 100 if len(expression_ids) else 0
#         )

#         # --------------------------------------------
#         # ANNOTATION
#         # --------------------------------------------

#         annotated = 0

#         if gpl_gene_col is not None:

#             tmp = pd.DataFrame(
#                 {
#                     "probe": gpl_probes,
#                     "gene": (
#                         gpl_df.loc[gpl_probes.index, gpl_gene_col]
#                         .astype(str)
#                         .map(clean_string)
#                     ),
#                 }
#             )

#             # Only expression-matched probes
#             tmp = tmp[tmp["probe"].isin(expression_ids)]

#             annotated = int(tmp["gene"].ne("").sum())

#             # Save mapping
#             mapping = tmp[tmp["gene"].ne("")].drop_duplicates("probe")

#             mapping = mapping.rename(columns={"probe": "ProbeID", "gene": "GeneSymbol"})

#             mapping.to_csv(OUT_DIR / f"{gse}_{gpl}_probe_gene_mapping.csv", index=False)

#         annotation_percent = annotated / exact_matches * 100 if exact_matches else 0

#         # --------------------------------------------
#         # STATUS
#         # --------------------------------------------

#         if match_percent >= 95 and annotation_percent >= 80:
#             status = "PASS"

#         elif match_percent >= 80 and annotation_percent >= 70:
#             status = "PASS_REVIEW"

#         elif match_percent >= 50:
#             status = "REVIEW"

#         else:
#             status = "FAIL_LOW_MATCH"

#         print(f"  Selected GPL file: {gpl_path.name}")

#         print(f"  GPL features: " f"{len(gpl_set):,}")

#         print(f"  Exact probe matches: " f"{exact_matches:,}")

#         print(f"  Match percentage: " f"{match_percent:.2f}%")

#         print(f"  Annotated probes: " f"{annotated:,}")

#         print(f"  Annotation percentage: " f"{annotation_percent:.2f}%")

#         print(f"  STATUS: {status}")

#         summary.append(
#             {
#                 "GSE": gse,
#                 "GPL": gpl,
#                 "Expression_Features": len(expression_ids),
#                 "GPL_Features": len(gpl_set),
#                 "Exact_Probe_Matches": exact_matches,
#                 "Match_Percent": round(match_percent, 2),
#                 "Annotated_Probes": annotated,
#                 "Annotation_Percent": round(annotation_percent, 2),
#                 "GPL_File": gpl_path.name,
#                 "Probe_Column": str(gpl_probe_col),
#                 "Gene_Column": str(gpl_gene_col),
#                 "Status": status,
#             }
#         )

#     except Exception as e:

#         print(f"  ERROR: {type(e).__name__}: {e}")

#         summary.append(
#             {
#                 "GSE": gse,
#                 "GPL": gpl,
#                 "Expression_Features": 0,
#                 "GPL_Features": 0,
#                 "Exact_Probe_Matches": 0,
#                 "Match_Percent": 0,
#                 "Annotated_Probes": 0,
#                 "Annotation_Percent": 0,
#                 "GPL_File": "",
#                 "Probe_Column": "",
#                 "Gene_Column": "",
#                 "Status": f"ERROR_{type(e).__name__}",
#             }
#         )


# # ------------------------------------------------------------
# # SAVE SUMMARY
# # ------------------------------------------------------------

# summary_df = pd.DataFrame(summary)

# summary_df.to_csv(OUT_DIR / "step05_probe_annotation_summary.csv", index=False)

# print("\n")
# print("=" * 80)
# print("STEP 5 SUMMARY")
# print("=" * 80)

# if len(summary_df):

#     print(
#         summary_df[
#             [
#                 "GSE",
#                 "GPL",
#                 "Expression_Features",
#                 "GPL_Features",
#                 "Exact_Probe_Matches",
#                 "Match_Percent",
#                 "Annotated_Probes",
#                 "Annotation_Percent",
#                 "Status",
#             ]
#         ].to_string(index=False)
#     )

# print("\nOutput:")
# print(OUT_DIR)
# print("=" * 80)


# ============================================================
# STEP 10.17 — TOP PROBES → GENE SYMBOL MAPPING
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = Path(r"D:\Dementia2")

DE_DIR = BASE_DIR / "data" / "step08_DE"
MAPPING_DIR = BASE_DIR / "data" / "step05_probe_annotation"

OUT_DIR = (
    BASE_DIR / "codes" / "biolysis" / "step10_visualization" / "17_top_gene_crossstudy"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# DATASETS
# ------------------------------------------------------------

DATASETS = [
    "GSE1297",
    "GSE4757",
    "GSE5281",
    "GSE6834",
    "GSE15222",
    "GSE16759",
    "GSE28146",
    "GSE29378",
    "GSE33000",
    "GSE36980",
    "GSE37263",
    "GSE44770",
]

# ------------------------------------------------------------
# FIND COLUMN
# ------------------------------------------------------------


def find_column(df, candidates):

    normalized = {
        str(c).strip().lower().replace("_", "").replace(".", "").replace("-", ""): c
        for c in df.columns
    }

    for candidate in candidates:

        key = candidate.lower().replace("_", "").replace(".", "").replace("-", "")

        if key in normalized:
            return normalized[key]

    return None


# ============================================================
# LOAD TOP PROBES FROM EACH DE FILE
# ============================================================

all_results = []

for gse in DATASETS:

    de_file = DE_DIR / f"{gse}_DE.csv"

    if not de_file.exists():
        print(f"[WARNING] Missing DE file: {gse}")
        continue

    de = pd.read_csv(de_file)

    print(f"\n{'=' * 70}")
    print(gse)
    print("DE columns:", list(de.columns))

    # --------------------------------------------------------
    # PROBE COLUMN
    # --------------------------------------------------------

    probe_col = find_column(de, ["Probe", "ProbeID", "Probe_ID", "ID", "ID_REF"])

    if probe_col is None:

        print(f"[WARNING] No probe column found for {gse}")

        continue

    # --------------------------------------------------------
    # LOGFC
    # --------------------------------------------------------

    logfc_col = find_column(de, ["logFC", "log2FC", "log_fold_change"])

    if logfc_col is None:

        print(f"[WARNING] No logFC column found for {gse}")

        continue

    # --------------------------------------------------------
    # ADJUSTED P
    # --------------------------------------------------------

    adjp_col = find_column(de, ["adj.P.Val", "FDR", "adjPValue", "adjusted_p_value"])

    if adjp_col is None:

        print(f"[WARNING] No adjusted p-value column found for {gse}")

        continue

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    de[probe_col] = de[probe_col].astype(str).str.strip()

    de[logfc_col] = pd.to_numeric(de[logfc_col], errors="coerce")

    de[adjp_col] = pd.to_numeric(de[adjp_col], errors="coerce")

    de = de.dropna(subset=[probe_col, logfc_col, adjp_col])

    # --------------------------------------------------------
    # EFFECT SIZE RANKING
    # --------------------------------------------------------

    de["abs_logFC"] = de[logfc_col].abs()

    de = de.sort_values("abs_logFC", ascending=False)

    # --------------------------------------------------------
    # TOP 10
    # --------------------------------------------------------

    top = de.head(10).copy()

    top["GSE"] = gse

    top = top.rename(
        columns={probe_col: "Probe", logfc_col: "logFC", adjp_col: "adj_p_value"}
    )

    # --------------------------------------------------------
    # FIND MAPPING FILE
    # --------------------------------------------------------

    mapping_files = list(MAPPING_DIR.glob(f"{gse}_*_probe_gene_mapping.csv"))

    if not mapping_files:

        print(f"[WARNING] No mapping file found for {gse}")

        top["GeneSymbol"] = np.nan

        all_results.append(top[["GSE", "Probe", "GeneSymbol", "logFC", "adj_p_value"]])

        continue

    mapping_file = mapping_files[0]

    print(f"Mapping file: {mapping_file.name}")

    mapping = pd.read_csv(mapping_file, dtype=str)

    print("Mapping columns:", list(mapping.columns))

    # --------------------------------------------------------
    # FIND MAPPING PROBE COLUMN
    # --------------------------------------------------------

    map_probe_col = find_column(
        mapping, ["Probe", "ProbeID", "Probe_ID", "ID", "ID_REF"]
    )

    # --------------------------------------------------------
    # FIND GENE SYMBOL COLUMN
    # --------------------------------------------------------

    gene_col = find_column(
        mapping, ["GeneSymbol", "Gene_Symbol", "Gene Symbol", "Symbol", "Gene"]
    )

    if map_probe_col is None:

        print(f"[WARNING] Probe column not found in mapping")

        top["GeneSymbol"] = np.nan

    elif gene_col is None:

        print(f"[WARNING] GeneSymbol column not found in mapping")

        top["GeneSymbol"] = np.nan

    else:

        mapping[map_probe_col] = mapping[map_probe_col].astype(str).str.strip()

        mapping[gene_col] = mapping[gene_col].astype(str).str.strip()

        # ----------------------------------------------------
        # NORMALIZE PROBE IDs
        # ----------------------------------------------------

        top["Probe"] = top["Probe"].astype(str).str.strip()

        # ----------------------------------------------------
        # ONE PROBE → ONE GENE
        # ----------------------------------------------------

        probe_to_gene = (
            mapping[[map_probe_col, gene_col]]
            .dropna()
            .drop_duplicates(subset=[map_probe_col])
            .set_index(map_probe_col)[gene_col]
        )

        top["GeneSymbol"] = top["Probe"].map(probe_to_gene)

    # --------------------------------------------------------
    # SAVE DATASET RESULT
    # --------------------------------------------------------

    result = top[["GSE", "Probe", "GeneSymbol", "logFC", "adj_p_value"]]

    result.to_csv(OUT_DIR / f"{gse}_top10_probe_gene.csv", index=False)

    all_results.append(result)

# ============================================================
# COMBINE
# ============================================================

if not all_results:

    raise RuntimeError("No datasets were successfully processed.")

final = pd.concat(all_results, ignore_index=True)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

final.to_csv(OUT_DIR / "ALL_DATASETS_top10_probe_gene.csv", index=False)

# ============================================================
# PRINT
# ============================================================

print("\n")
print("=" * 80)
print("TOP 10 PROBES → GENE SYMBOLS")
print("=" * 80)

print(final.to_string(index=False))

print("\n")
print("=" * 80)
print("OUTPUT")
print("=" * 80)

print(OUT_DIR / "ALL_DATASETS_top10_probe_gene.csv")

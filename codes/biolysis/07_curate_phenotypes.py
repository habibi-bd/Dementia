from pathlib import Path
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

BASE = Path(r"D:\Dementia2")

LABEL_FILE = BASE / "data" / "labels.csv"
EXPR_DIR = BASE / "codes" / "biolysis" / "step06_gene_expression"
OUT_DIR = BASE / "codes" / "biolysis" / "step07_phenotype"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# GSE26927 explicitly excluded
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

# ============================================================
# LOAD LABELS
# ============================================================

labels = pd.read_csv(LABEL_FILE)

required = {"gse_id", "sample_id", "condition", "label"}

missing = required - set(labels.columns)

if missing:
    raise ValueError(f"labels.csv missing columns: {missing}")

labels["gse_id"] = labels["gse_id"].astype(str).str.strip()
labels["sample_id"] = labels["sample_id"].astype(str).str.strip()
labels["condition"] = labels["condition"].astype(str).str.strip()
labels["label"] = pd.to_numeric(labels["label"], errors="coerce")

# Remove excluded dataset
labels = labels[labels["gse_id"] != "GSE26927"].copy()

# Keep only binary labels
labels = labels[labels["label"].isin([0, 1])].copy()

labels["condition"] = labels["label"].map({0: "Control", 1: "Alzheimer"})

# ============================================================
# PROCESS DATASETS
# ============================================================

summary = []

for gse in DATASETS:

    print("\n" + "=" * 80)
    print(f"Processing {gse}")
    print("=" * 80)

    expr_file = EXPR_DIR / f"{gse}_gene_expression.csv"

    if not expr_file.exists():
        print(f"  ERROR: expression file not found")
        summary.append(
            {
                "GSE": gse,
                "Expression_Samples": 0,
                "Label_Samples": 0,
                "Matched_Samples": 0,
                "Alzheimer": 0,
                "Control": 0,
                "Status": "MISSING_EXPRESSION",
            }
        )
        continue

    # --------------------------------------------------------
    # Load expression
    # --------------------------------------------------------

    expr = pd.read_csv(expr_file)

    gene_col = expr.columns[0]

    expression_samples = [c for c in expr.columns if c != gene_col]

    # --------------------------------------------------------
    # Labels for this GSE
    # --------------------------------------------------------

    g_labels = labels[labels["gse_id"] == gse].copy()

    # Remove duplicate sample labels
    g_labels = g_labels.drop_duplicates(subset=["sample_id"], keep="last")

    label_samples = set(g_labels["sample_id"])

    # --------------------------------------------------------
    # Match samples
    # --------------------------------------------------------

    matched_samples = [s for s in expression_samples if s in label_samples]

    missing_labels = [s for s in expression_samples if s not in label_samples]

    unused_labels = [s for s in label_samples if s not in set(expression_samples)]

    # --------------------------------------------------------
    # Build phenotype metadata
    # --------------------------------------------------------

    meta = g_labels[g_labels["sample_id"].isin(matched_samples)].copy()

    meta = meta.set_index("sample_id").loc[matched_samples].reset_index()

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    n_ad = int((meta["label"] == 1).sum())
    n_control = int((meta["label"] == 0).sum())

    # --------------------------------------------------------
    # Require both groups
    # --------------------------------------------------------

    if n_ad < 3 or n_control < 3:

        status = "INSUFFICIENT_GROUPS"

        print(f"  Expression samples : {len(expression_samples)}")
        print(f"  Label samples      : {len(g_labels)}")
        print(f"  Matched samples    : {len(matched_samples)}")
        print(f"  Alzheimer          : {n_ad}")
        print(f"  Control            : {n_control}")
        print(f"  STATUS             : {status}")

        summary.append(
            {
                "GSE": gse,
                "Expression_Samples": len(expression_samples),
                "Label_Samples": len(g_labels),
                "Matched_Samples": len(matched_samples),
                "Unmatched_Expression": len(missing_labels),
                "Unused_Labels": len(unused_labels),
                "Alzheimer": n_ad,
                "Control": n_control,
                "Status": status,
            }
        )

        continue

    # --------------------------------------------------------
    # Save curated metadata
    # --------------------------------------------------------

    meta_file = OUT_DIR / f"{gse}_curated_metadata.csv"
    meta.to_csv(meta_file, index=False)

    # --------------------------------------------------------
    # Expression matrix
    # --------------------------------------------------------

    keep_samples = [gene_col] + matched_samples

    expr_matched = expr[keep_samples].copy()

    expression_out = OUT_DIR / f"{gse}_case_control_expression.csv"

    expr_matched.to_csv(expression_out, index=False)

    # --------------------------------------------------------
    # Save sample order
    # --------------------------------------------------------

    sample_order = meta[["sample_id", "condition", "label"]].copy()

    sample_order.to_csv(OUT_DIR / f"{gse}_sample_order.csv", index=False)

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(f"  Expression samples : {len(expression_samples)}")
    print(f"  Label samples      : {len(g_labels)}")
    print(f"  Matched samples    : {len(matched_samples)}")
    print(f"  Alzheimer          : {n_ad}")
    print(f"  Control            : {n_control}")
    print(f"  Unmatched expr     : {len(missing_labels)}")
    print(f"  Unused labels      : {len(unused_labels)}")
    print(f"  STATUS             : PASS")

    summary.append(
        {
            "GSE": gse,
            "Expression_Samples": len(expression_samples),
            "Label_Samples": len(g_labels),
            "Matched_Samples": len(matched_samples),
            "Unmatched_Expression": len(missing_labels),
            "Unused_Labels": len(unused_labels),
            "Alzheimer": n_ad,
            "Control": n_control,
            "Status": "PASS",
        }
    )

# ============================================================
# SUMMARY
# ============================================================

summary_df = pd.DataFrame(summary)

summary_file = OUT_DIR / "step07_phenotype_summary.csv"

summary_df.to_csv(summary_file, index=False)

print("\n" + "=" * 80)
print("STEP 7 SUMMARY")
print("=" * 80)

print(summary_df.to_string(index=False))

print("\nSaved:")
print(summary_file)

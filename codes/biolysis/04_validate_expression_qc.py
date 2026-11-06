from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json

# =============================================================================
# STEP 4
# EXPRESSION MATRIX QUALITY + SCALE VALIDATION
# =============================================================================

BIOLOGY_DIR = Path(__file__).resolve().parent
STEP02_DIR = BIOLOGY_DIR / "step02_parsed"
OUTPUT_DIR = BIOLOGY_DIR / "step04_expression_qc"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# GSE26927 intentionally excluded
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


# =============================================================================
# FILE DISCOVERY
# =============================================================================


def find_expression_file(dataset):

    candidates = []

    for f in STEP02_DIR.glob("*"):

        if not f.is_file():
            continue

        name = f.name.lower()

        if dataset.lower() not in name:
            continue

        if f.suffix.lower() not in [".csv", ".tsv", ".txt"]:
            continue

        if any(x in name for x in ["expression", "expr", "matrix"]):
            candidates.append(f)

    if not candidates:
        return None

    # Prefer expression files explicitly named expression
    candidates.sort(
        key=lambda x: (0 if "expression" in x.name.lower() else 1, len(x.name))
    )

    return candidates[0]


# =============================================================================
# LOAD EXPRESSION
# =============================================================================


def load_expression(file):

    try:

        if file.suffix.lower() == ".csv":

            df = pd.read_csv(file, index_col=0)

        else:

            df = pd.read_csv(file, sep="\t", index_col=0)

        return df

    except Exception as e:

        print(f"ERROR loading {file.name}: {e}")
        return None


# =============================================================================
# NUMERIC CONVERSION
# =============================================================================


def prepare_numeric(df):

    numeric = df.apply(pd.to_numeric, errors="coerce")

    return numeric


# =============================================================================
# SCALE HEURISTIC
# =============================================================================


def determine_scale(values):

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return "NO_VALID_VALUES"

    minimum = float(np.min(values))
    maximum = float(np.max(values))
    median = float(np.median(values))

    q01 = float(np.percentile(values, 1))
    q99 = float(np.percentile(values, 99))

    # Strong evidence for log2-like microarray data
    if minimum < 0 or (maximum < 30 and q99 < 25 and median < 15):
        return "LIKELY_LOG_SCALE"

    # Strong evidence for raw intensity
    if maximum > 100:

        return "LIKELY_RAW_SCALE"

    return "AMBIGUOUS_SCALE"


# =============================================================================
# SAMPLE STATISTICS
# =============================================================================


def sample_statistics(X, dataset):

    rows = []

    for sample in X.columns:

        values = X[sample].to_numpy(dtype=float)

        finite = values[np.isfinite(values)]

        if len(finite) == 0:

            rows.append(
                {
                    "GSE": dataset,
                    "Sample": sample,
                    "N_Features": len(values),
                    "Valid_Values": 0,
                    "Missing": len(values),
                    "Missing_Percent": 100.0,
                    "Min": np.nan,
                    "Q01": np.nan,
                    "Median": np.nan,
                    "Q99": np.nan,
                    "Max": np.nan,
                    "Mean": np.nan,
                    "SD": np.nan,
                }
            )

            continue

        rows.append(
            {
                "GSE": dataset,
                "Sample": sample,
                "N_Features": len(values),
                "Valid_Values": len(finite),
                "Missing": int(len(values) - len(finite)),
                "Missing_Percent": 100.0 * (len(values) - len(finite)) / len(values),
                "Min": np.min(finite),
                "Q01": np.percentile(finite, 1),
                "Median": np.median(finite),
                "Q99": np.percentile(finite, 99),
                "Max": np.max(finite),
                "Mean": np.mean(finite),
                "SD": np.std(finite),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# FEATURE STATISTICS
# =============================================================================


def feature_statistics(X, dataset):

    rows = []

    for feature in X.index:

        values = X.loc[feature].to_numpy(dtype=float)

        finite = values[np.isfinite(values)]

        if len(finite) == 0:

            rows.append(
                {
                    "GSE": dataset,
                    "Feature": feature,
                    "Valid_Samples": 0,
                    "Missing_Samples": len(values),
                    "Variance": np.nan,
                    "Mean": np.nan,
                    "Median": np.nan,
                    "Min": np.nan,
                    "Max": np.nan,
                }
            )

            continue

        rows.append(
            {
                "GSE": dataset,
                "Feature": feature,
                "Valid_Samples": len(finite),
                "Missing_Samples": int(len(values) - len(finite)),
                "Variance": np.var(finite),
                "Mean": np.mean(finite),
                "Median": np.median(finite),
                "Min": np.min(finite),
                "Max": np.max(finite),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# EXPRESSION DISTRIBUTION PLOT
# =============================================================================


def create_sample_boxplot(X, dataset):

    # Limit features only for plotting to keep large matrices manageable
    if len(X) > 10000:

        rng = np.random.default_rng(42)

        indices = rng.choice(len(X), size=10000, replace=False)

        plot_X = X.iloc[indices]

    else:

        plot_X = X

    plt.figure(figsize=(max(10, len(X.columns) * 0.18), 7))

    plt.boxplot(
        [
            pd.to_numeric(plot_X[col], errors="coerce").dropna()
            for col in plot_X.columns
        ],
        showfliers=False,
    )

    plt.xlabel("Samples")
    plt.ylabel("Expression")
    plt.title(f"{dataset} — Sample Expression Distribution")

    plt.xticks(
        range(1, len(plot_X.columns) + 1), plot_X.columns, rotation=90, fontsize=5
    )

    plt.tight_layout()

    output = OUTPUT_DIR / f"{dataset}_sample_boxplot.png"

    plt.savefig(output, dpi=180, bbox_inches="tight")

    plt.close()


# =============================================================================
# SAMPLE MEDIAN PLOT
# =============================================================================


def create_sample_median_plot(stats, dataset):

    plt.figure(figsize=(12, 6))

    plt.plot(
        range(len(stats)),
        stats["Median"].to_numpy(),
        marker="o",
        markersize=3,
        linewidth=1,
    )

    plt.xlabel("Sample")
    plt.ylabel("Median Expression")
    plt.title(f"{dataset} — Sample Median Expression")

    plt.xticks(range(len(stats)), stats["Sample"], rotation=90, fontsize=5)

    plt.tight_layout()

    output = OUTPUT_DIR / f"{dataset}_sample_medians.png"

    plt.savefig(output, dpi=180, bbox_inches="tight")

    plt.close()


# =============================================================================
# PROCESS DATASET
# =============================================================================


def process_dataset(dataset):

    print()
    print("=" * 80)
    print(dataset)
    print("=" * 80)

    expression_file = find_expression_file(dataset)

    if expression_file is None:

        print("ERROR: Expression matrix not found.")
        return None

    print(f"Expression file: {expression_file.name}")

    df = load_expression(expression_file)

    if df is None:
        return None

    print(f"Original shape: {df.shape}")

    # -------------------------------------------------------------------------
    # Clean feature IDs
    # -------------------------------------------------------------------------

    df.index = df.index.astype(str).str.strip().str.replace('"', "", regex=False)

    # -------------------------------------------------------------------------
    # Numeric matrix
    # -------------------------------------------------------------------------

    X = prepare_numeric(df)

    # -------------------------------------------------------------------------
    # Remove columns that contain no numeric values
    # -------------------------------------------------------------------------

    valid_columns = [col for col in X.columns if X[col].notna().sum() > 0]

    X = X[valid_columns]

    # -------------------------------------------------------------------------
    # Global values
    # -------------------------------------------------------------------------

    values = X.to_numpy(dtype=float).ravel()

    finite_values = values[np.isfinite(values)]

    if len(finite_values) == 0:

        print("ERROR: No valid numeric expression values.")
        return None

    minimum = float(np.min(finite_values))
    maximum = float(np.max(finite_values))
    mean = float(np.mean(finite_values))
    median = float(np.median(finite_values))

    q01 = float(np.percentile(finite_values, 1))
    q25 = float(np.percentile(finite_values, 25))
    q75 = float(np.percentile(finite_values, 75))
    q99 = float(np.percentile(finite_values, 99))

    missing = int(X.isna().sum().sum())

    infinite = int(np.isinf(X.to_numpy(dtype=float)).sum())

    # -------------------------------------------------------------------------
    # Duplicate features
    # -------------------------------------------------------------------------

    duplicated_features = int(X.index.duplicated().sum())

    unique_features = int(X.index.nunique())

    # -------------------------------------------------------------------------
    # Constant features
    # -------------------------------------------------------------------------

    feature_variance = X.var(axis=1, skipna=True)

    constant_features = int((feature_variance.fillna(0) == 0).sum())

    # -------------------------------------------------------------------------
    # Scale
    # -------------------------------------------------------------------------

    scale = determine_scale(finite_values)

    # -------------------------------------------------------------------------
    # Sample statistics
    # -------------------------------------------------------------------------

    sample_stats = sample_statistics(X, dataset)

    sample_stats_file = OUTPUT_DIR / f"{dataset}_sample_statistics.csv"

    sample_stats.to_csv(sample_stats_file, index=False)

    # -------------------------------------------------------------------------
    # Feature statistics
    # -------------------------------------------------------------------------

    feature_stats = feature_statistics(X, dataset)

    feature_stats_file = OUTPUT_DIR / f"{dataset}_feature_statistics.csv"

    feature_stats.to_csv(feature_stats_file, index=False)

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    create_sample_boxplot(X, dataset)

    create_sample_median_plot(sample_stats, dataset)

    # -------------------------------------------------------------------------
    # Dataset summary
    # -------------------------------------------------------------------------

    summary = {
        "GSE": dataset,
        "Features": X.shape[0],
        "Samples": X.shape[1],
        "Unique_Features": unique_features,
        "Duplicate_Feature_Rows": duplicated_features,
        "Constant_Features": constant_features,
        "Missing_Values": missing,
        "Infinite_Values": infinite,
        "Min": minimum,
        "Q01": q01,
        "Q25": q25,
        "Median": median,
        "Q75": q75,
        "Q99": q99,
        "Max": maximum,
        "Mean": mean,
        "Likely_Scale": scale,
        "Status": "PASS",
    }

    # -------------------------------------------------------------------------
    # Automatic QC flags
    # -------------------------------------------------------------------------

    problems = []

    if X.shape[1] < 2:
        problems.append("Too few samples")

    if X.shape[0] < 100:
        problems.append("Very few features")

    if missing > 0:
        problems.append("Missing expression values")

    if infinite > 0:
        problems.append("Infinite expression values")

    if duplicated_features > 0:
        problems.append(f"{duplicated_features} duplicate feature IDs")

    if constant_features > 0:
        problems.append(f"{constant_features} constant features")

    if scale == "NO_VALID_VALUES":
        problems.append("No valid expression values")

    if problems:
        summary["Status"] = "REVIEW"
        summary["QC_Flags"] = "; ".join(problems)
    else:
        summary["QC_Flags"] = ""

    print(f"Features:             {X.shape[0]}")
    print(f"Samples:              {X.shape[1]}")
    print(f"Unique features:      {unique_features}")
    print(f"Duplicate features:   {duplicated_features}")
    print(f"Constant features:    {constant_features}")
    print(f"Missing values:       {missing}")
    print(f"Infinite values:      {infinite}")
    print(f"Minimum:              {minimum:.4f}")
    print(f"Median:               {median:.4f}")
    print(f"Maximum:              {maximum:.4f}")
    print(f"Likely scale:         {scale}")
    print(f"QC status:            {summary['Status']}")

    return summary


# =============================================================================
# MAIN
# =============================================================================


def main():

    print("=" * 80)
    print("STEP 4 — EXPRESSION MATRIX QUALITY + SCALE VALIDATION")
    print("=" * 80)

    print()
    print("GSE26927 is EXCLUDED.")
    print(f"Datasets analyzed: {len(DATASETS)}")
    print(f"Output: {OUTPUT_DIR}")

    results = []

    for dataset in DATASETS:

        result = process_dataset(dataset)

        if result is not None:
            results.append(result)

    # =========================================================================
    # COMBINED SUMMARY
    # =========================================================================

    summary = pd.DataFrame(results)

    summary_file = OUTPUT_DIR / "expression_qc_summary.csv"

    summary.to_csv(summary_file, index=False)

    # =========================================================================
    # SAMPLE QC SUMMARY
    # =========================================================================

    sample_files = sorted(OUTPUT_DIR.glob("*_sample_statistics.csv"))

    sample_summary_rows = []

    for file in sample_files:

        try:

            df = pd.read_csv(file)

            dataset = file.name.split("_sample_statistics.csv")[0]

            for _, row in df.iterrows():

                sample_summary_rows.append({"GSE": dataset, **row.to_dict()})

        except Exception as e:

            print(f"WARNING reading {file.name}: {e}")

    if sample_summary_rows:

        sample_summary = pd.DataFrame(sample_summary_rows)

        sample_summary.to_csv(
            OUTPUT_DIR / "all_sample_expression_statistics.csv", index=False
        )

    # =========================================================================
    # RUN REPORT
    # =========================================================================

    report = {
        "step": 4,
        "description": "Expression matrix quality and scale validation",
        "datasets_requested": len(DATASETS),
        "datasets_processed": len(results),
        "excluded_datasets": ["GSE26927"],
        "output_directory": str(OUTPUT_DIR),
    }

    with open(OUTPUT_DIR / "step04_run_report.json", "w", encoding="utf-8") as f:

        json.dump(report, f, indent=2)

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 4 SUMMARY")
    print("=" * 80)

    print(f"Datasets processed: {len(results)} / {len(DATASETS)}")

    if not summary.empty:

        print()

        columns = [
            "GSE",
            "Features",
            "Samples",
            "Missing_Values",
            "Duplicate_Feature_Rows",
            "Constant_Features",
            "Min",
            "Median",
            "Max",
            "Likely_Scale",
            "Status",
        ]

        available = [c for c in columns if c in summary.columns]

        print(summary[available].to_string(index=False))

    print()
    print("Output directory:")
    print(f"  {OUTPUT_DIR}")

    print()
    print("=" * 80)
    print("STEP 4 FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    main()

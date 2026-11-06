# # import pandas as pd

# # # Use a relative path from where dataset_details.py is located
# # df = pd.read_csv("ml_data/integrated_labels.csv")

# # print(df.head())


# import pandas as pd

# # Load your dataset
# df = pd.read_csv("ml_data/integrated_labels.csv")

# # Group by dataset and condition/label, then count the samples
# # Adjust 'condition' or 'label' values based on your exact text (e.g., "Alzheimer", "Normal", etc.)
# summary_table = (
#     df.groupby(["dataset", "platform", "condition"]).size().unstack(fill_value=0)
# )

# # If your conditions are named differently (e.g., 'Normal' and 'Alzheimer' instead of 'Cancer'),
# # map or rename the columns to match your desired format.
# # For example: summary_table = summary_table.rename(columns={'Alzheimer': 'Alzheimer Samples', 'Normal': 'Normal Samples'})

# # Calculate Total Samples
# summary_table["Total Samples"] = summary_table.sum(axis=1)

# # Reset index to make dataset and platform normal columns
# summary_table = summary_table.reset_index()

# print(summary_table.to_markdown(index=False))


import os
import pandas as pd
import numpy as np
from pathlib import Path

# ================================================================
# PATHS
# ================================================================

BASE = r"D:\Dementia2"
ML_DATA = os.path.join(BASE, "codes", "ml_data")
ML_RESULTS = os.path.join(BASE, "codes", "ml_results")

# ================================================================
# 1. LOAD PREPROCESSED DATA (Ready for ML)
# ================================================================

print("=" * 80)
print("PREPROCESSED DATASET SIZE (Ready for ML/DL Training)")
print("=" * 80)

# Main integrated file
integrated_file = os.path.join(ML_DATA, "integrated_combat_gene_expression.csv")
labels_file = os.path.join(ML_DATA, "integrated_combat_labels.csv")
summary_file = os.path.join(ML_DATA, "combat_dataset_summary.csv")

# ================================================================
# 2. LOAD THE ACTUAL TRAINING DATA
# ================================================================

print("\n" + "-" * 80)
print("1. FINAL PREPROCESSED DATASET")
print("-" * 80)

if os.path.exists(integrated_file):
    df = pd.read_csv(integrated_file, low_memory=False)

    # Identify columns
    meta_cols = ["sample_id", "condition", "label", "dataset", "platform"]
    meta_present = [c for c in meta_cols if c in df.columns]
    gene_cols = [c for c in df.columns if c not in meta_cols]

    # Clean data
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
    df[gene_cols] = df[gene_cols].apply(pd.to_numeric, errors="coerce")

    print(f"\n📊 DATASET SIZE:")
    print(f"   ├── Total samples:        {len(df):,}")
    print(f"   ├── Total genes:          {len(gene_cols):,}")
    print(f"   ├── Alzheimer's (label=1): {(df['label'] == 1).sum():,}")
    print(f"   ├── Control (label=0):     {(df['label'] == 0).sum():,}")
    print(f"   └── Total features:       {len(gene_cols):,}")

    print(f"\n📊 DATA TYPE:")
    print(
        f"   ├── Matrix shape:         {df.shape[0]:,} samples × {df.shape[1]:,} columns"
    )
    print(f"   ├── Gene columns:         {len(gene_cols):,}")
    print(
        f"   ├── Metadata columns:     {len(meta_present)} ({', '.join(meta_present)})"
    )
    print(
        f"   └── Memory usage:         {df.memory_usage(deep=True).sum() / (1024**2):.1f} MB"
    )

    # Check for missing values
    missing_genes = df[gene_cols].isnull().sum().sum()
    if missing_genes > 0:
        print(
            f"\n⚠️ Missing values:          {missing_genes:,} ({missing_genes/(len(df)*len(gene_cols))*100:.4f}%)"
        )
    else:
        print(f"\n✅ No missing values in gene expression data")

else:
    print(f"❌ Integrated file not found at: {integrated_file}")
    df = None
    gene_cols = []

# ================================================================
# 3. DATASET BREAKDOWN (FIXED - No Apostrophe Issues)
# ================================================================

print("\n" + "-" * 80)
print("2. SAMPLES PER DATASET")
print("-" * 80)

if df is not None:
    # OPTION 1: Using .agg() with dictionary
    ds_summary = (
        df.groupby(["dataset", "platform"])["label"]
        .agg(
            [
                ("Total", "count"),
                ("AD", lambda x: (x == 1).sum()),
                ("Control", lambda x: (x == 0).sum()),
            ]
        )
        .reset_index()
    )

    print(ds_summary.to_string(index=False))
    print(f"\n{'='*50}")
    print(f"TOTAL: {ds_summary['Total'].sum():,} samples")
    print(f"       AD: {ds_summary['AD'].sum():,}")
    print(f"       Control: {ds_summary['Control'].sum():,}")

    # OPTION 2: Alternative using value_counts (if you prefer)
    # ds_summary2 = df.groupby(['dataset', 'platform']).apply(
    #     lambda x: pd.Series({
    #         'Total': len(x),
    #         'AD': (x['label'] == 1).sum(),
    #         'Control': (x['label'] == 0).sum()
    #     })
    # ).reset_index()

# ================================================================
# 4. TRAIN/TEST SPLIT (Ready for ML)
# ================================================================

print("\n" + "-" * 80)
print("3. TRAIN/TEST SPLIT (Actual ML/DL Ready Data)")
print("-" * 80)

test_file = os.path.join(ML_RESULTS, "test_set_samples.csv")

if os.path.exists(test_file) and df is not None:
    test_samples = pd.read_csv(test_file)

    # Determine which samples are in test set
    test_ids = set(test_samples["sample_id"].astype(str))
    all_ids = set(df["sample_id"].astype(str))

    train_ids = all_ids - test_ids

    X_train = df[df["sample_id"].astype(str).isin(train_ids)]
    X_test = df[df["sample_id"].astype(str).isin(test_ids)]

    print(f"\n📊 TRAIN SET:")
    print(f"   ├── Samples:   {len(X_train):,}")
    print(f"   ├── AD:        {(X_train['label'] == 1).sum():,}")
    print(f"   ├── Control:   {(X_train['label'] == 0).sum():,}")
    print(f"   └── Genes:     {len(gene_cols):,}")

    print(f"\n📊 TEST SET:")
    print(f"   ├── Samples:   {len(X_test):,}")
    print(f"   ├── AD:        {(X_test['label'] == 1).sum():,}")
    print(f"   ├── Control:   {(X_test['label'] == 0).sum():,}")
    print(f"   └── Genes:     {len(gene_cols):,}")

    print(f"\n📊 SPLIT RATIO:")
    print(f"   ├── Train:     {len(X_train)/len(df)*100:.1f}%")
    print(f"   ├── Test:      {len(X_test)/len(df)*100:.1f}%")
    print(f"   └── Total:     {len(df):,}")

# ================================================================
# 5. ACTUAL INPUT FOR ML MODELS
# ================================================================

print("\n" + "-" * 80)
print("4. ML/DL MODEL INPUT MATRICES")
print("-" * 80)

if df is not None:
    # X (features) and y (labels) as used in Step 17
    X = df[gene_cols].values
    y = df["label"].values

    print(f"\n📊 FEATURE MATRIX (X):")
    print(f"   ├── Shape:     {X.shape[0]:,} samples × {X.shape[1]:,} genes")
    print(f"   ├── Type:      {X.dtype}")
    print(f"   ├── Range:     [{X.min():.4f}, {X.max():.4f}]")
    print(f"   ├── Mean:      {X.mean():.4f}")
    print(f"   ├── Std Dev:   {X.std():.4f}")
    print(f"   └── Memory:    {X.nbytes / (1024**2):.1f} MB")

    print(f"\n📊 LABEL VECTOR (y):")
    print(f"   ├── Shape:     {len(y):,}")
    print(f"   ├── AD:        {(y == 1).sum():,} ({ (y == 1).sum()/len(y)*100:.1f}%)")
    print(f"   ├── Control:   {(y == 0).sum():,} ({ (y == 0).sum()/len(y)*100:.1f}%)")
    print(f"   └── Imbalance: {(y == 1).sum()/(y == 0).sum():.2f}:1 (AD:Control)")

# ================================================================
# 6. DATASET SUMMARY TABLE FOR PAPER (FIXED - No Apostrophe Issues)
# ================================================================

print("\n" + "-" * 80)
print("5. PAPER-READY SUMMARY TABLE")
print("-" * 80)

if df is not None:
    # Calculate values first
    n_total = len(df)
    n_ad = (df["label"] == 1).sum()
    n_control = (df["label"] == 0).sum()
    n_genes = len(gene_cols)
    n_datasets = df["dataset"].nunique()
    n_platforms = df["platform"].nunique()

    if os.path.exists(test_file):
        n_train = len(X_train)
        n_test = len(X_test)
        split_ratio = f"{n_train/n_total*100:.1f}% / {n_test/n_total*100:.1f}%"
    else:
        n_train = "N/A"
        n_test = "N/A"
        split_ratio = "N/A"

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│ FINAL PREPROCESSED DATASET CHARACTERISTICS                  │")
    print("├─────────────────────────────────────────────────────────────┤")
    print(f"│ Total Samples              │ {n_total:>8,}                    │")
    print(f"│ Total Genes                │ {n_genes:>8,}                    │")
    print(f"│ Alzheimer's Disease        │ {n_ad:>8,}                    │")
    print(f"│ Control                    │ {n_control:>8,}                    │")
    print(f"│ Datasets                   │ {n_datasets:>8,}                    │")
    print(f"│ Platforms                  │ {n_platforms:>8,}                    │")
    print(f"│ Train Samples              │ {str(n_train):>8}                    │")
    print(f"│ Test Samples               │ {str(n_test):>8}                    │")
    print(f"│ Train/Test Split           │ {split_ratio:>8}                    │")
    print("└─────────────────────────────────────────────────────────────┘")

# ================================================================
# 7. SAVE SUMMARY
# ================================================================

output_file = os.path.join(BASE, "codes", "preprocessed_dataset_summary.csv")

if df is not None:
    # Calculate values without apostrophes in column names
    n_total = len(df)
    n_ad = (df["label"] == 1).sum()
    n_control = (df["label"] == 0).sum()
    n_genes = len(gene_cols)
    n_datasets = df["dataset"].nunique()
    n_platforms = df["platform"].nunique()

    if os.path.exists(test_file):
        n_train = len(X_train)
        n_test = len(X_test)
        split_ratio = f"{n_train/n_total*100:.1f}% / {n_test/n_total*100:.1f}%"
    else:
        n_train = "N/A"
        n_test = "N/A"
        split_ratio = "N/A"

    summary_data = {
        "Metric": [
            "Total Samples",
            "Alzheimer_Disease",  # Using underscore instead of apostrophe
            "Control",
            "Total_Genes",
            "Number_Datasets",
            "Number_Platforms",
            "Train_Samples",
            "Test_Samples",
            "Train_Test_Split",
            "Feature_Matrix_Shape",
            "AD_Control_Ratio",
        ],
        "Value": [
            n_total,
            n_ad,
            n_control,
            n_genes,
            n_datasets,
            n_platforms,
            n_train,
            n_test,
            split_ratio,
            f"{X.shape[0]:,} x {X.shape[1]:,}",
            f"1:{n_control/n_ad:.2f}" if n_ad > 0 else "N/A",
        ],
    }

    pd.DataFrame(summary_data).to_csv(output_file, index=False)
    print(f"\n📊 Summary saved to: {output_file}")

# ================================================================
# 8. FINAL SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("READY FOR ML/DL TRAINING!")
print("=" * 80)

if df is not None:
    n_total = len(df)
    n_genes = len(gene_cols)
    n_ad = (df["label"] == 1).sum()
    n_control = (df["label"] == 0).sum()
    n_datasets = df["dataset"].nunique()
    n_platforms = df["platform"].nunique()
    memory_mb = df.memory_usage(deep=True).sum() / (1024**2)

    print(f"""
✅ Preprocessed data is ready:
   • {n_total:,} samples
   • {n_genes:,} genes
   • {n_ad:,} AD / {n_control:,} Control
   • {n_datasets} datasets
   • {n_platforms} platforms
   • Memory: {memory_mb:.1f} MB
""")

print("=" * 80)

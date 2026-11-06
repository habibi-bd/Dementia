from pathlib import Path
import pandas as pd

# ============================================================
# 1. Project directories
# ============================================================

PROJECT_DIR = Path(r"D:\Dementia2")
DATA_DIR = PROJECT_DIR / "data"
MATRIX_DIR = DATA_DIR / "matrix"
GPL_DIR = DATA_DIR / "gpl"
LABEL_FILE = DATA_DIR / "labels.csv"
OUTPUT_DIR = PROJECT_DIR / "codes"


# ============================================================
# 2. Dataset list
# ============================================================

GSE_LIST = [
    "GSE33000",
    "GSE29378",
    "GSE44770",
    "GSE36980",
    "GSE37263",
    "GSE28146",
    "GSE16759",
    "GSE15222",
    "GSE6834",
    "GSE5281",
    "GSE1297",
    "GSE4757",
]


# ============================================================
# 3. Check directories and label file
# ============================================================

print("\n=== PROJECT STRUCTURE ===")

for path in [DATA_DIR, MATRIX_DIR, GPL_DIR, LABEL_FILE]:
    print(f"{path}: {'OK' if path.exists() else 'MISSING'}")


# ============================================================
# 4. Read labels
# ============================================================

print("\n=== LABEL FILE ===")

labels = pd.read_csv(LABEL_FILE)

print("Shape:", labels.shape)
print("Columns:", list(labels.columns))

required_columns = [
    "gse_id",
    "sample_id",
    "condition",
    "label",
]

missing_columns = [col for col in required_columns if col not in labels.columns]

if missing_columns:
    print("Missing columns:", missing_columns)
else:
    print("All required columns are present.")


# ============================================================
# 5. Basic label information
# ============================================================

print("\n=== LABEL SUMMARY ===")

print("\nGSE datasets:")
print(labels["gse_id"].value_counts().sort_index())

print("\nConditions:")
print(labels["condition"].value_counts())

print("\nLabel values:")
print(labels["label"].value_counts(dropna=False))


# ============================================================
# 6. Dataset-by-dataset sample summary
# ============================================================

print("\n=== DATASET SAMPLE SUMMARY ===")

summary = labels.groupby(["gse_id", "condition"]).size().unstack(fill_value=0)

print(summary)


# ============================================================
# 7. Check for duplicate sample IDs
# ============================================================

print("\n=== DUPLICATE SAMPLE IDs ===")

duplicate_samples = labels[labels["sample_id"].duplicated(keep=False)].sort_values(
    "sample_id"
)

if duplicate_samples.empty:
    print("No duplicate sample IDs found.")
else:
    print(duplicate_samples.to_string(index=False))


# ============================================================
# 8. Check whether requested GSE datasets are present
# ============================================================

print("\n=== REQUESTED DATASETS ===")

label_gses = set(labels["gse_id"].astype(str))

for gse in GSE_LIST:
    if gse in label_gses:
        count = (labels["gse_id"] == gse).sum()
        print(f"{gse}: {count} labeled samples")
    else:
        print(f"{gse}: NOT FOUND in labels.csv")


# ============================================================
# 9. Find matrix files
# ============================================================

print("\n=== MATRIX FILES ===")

matrix_files = sorted(MATRIX_DIR.glob("*"))

for file in matrix_files:
    print(file.name)


# ============================================================
# 10. Match expected GSEs to matrix files
# ============================================================

print("\n=== GSE MATRIX CHECK ===")

for gse in GSE_LIST:
    matches = [file.name for file in matrix_files if gse in file.name]

    print(f"\n{gse}")
    if matches:
        for match in matches:
            print("   ", match)
    else:
        print("    NO MATRIX FILE FOUND")


print("\n=== STEP 1 COMPLETE ===")

from pathlib import Path
import re
import pandas as pd

# ============================================================
# 1. Directories
# ============================================================

PROJECT_DIR = Path(r"D:\Dementia2")
MATRIX_DIR = PROJECT_DIR / "data" / "matrix"
GPL_DIR = PROJECT_DIR / "data" / "gpl"
LABEL_FILE = PROJECT_DIR / "data" / "labels.csv"
OUTPUT_FILE = PROJECT_DIR / "codes" / "gse_gpl_mapping.csv"


# ============================================================
# 2. Datasets
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
# 3. Find GPL from a series matrix file
# ============================================================


def get_gpl_from_matrix(file_path):

    gpl_ids = set()

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:

        for line in f:

            if not line.startswith("!"):
                break

            if "!Series_platform_id" in line:

                matches = re.findall(r"GPL\d+", line)

                gpl_ids.update(matches)

    return sorted(gpl_ids)


# ============================================================
# 4. Find annotation files for a GPL
# ============================================================


def find_gpl_files(gpl_id):

    files = sorted(GPL_DIR.glob(f"{gpl_id}*"))

    return [file.name for file in files]


# ============================================================
# 5. Build mapping
# ============================================================

results = []


for gse in GSE_LIST:

    matrix_files = sorted(MATRIX_DIR.glob(f"{gse}*series_matrix.txt"))

    for matrix_file in matrix_files:

        gpl_ids = get_gpl_from_matrix(matrix_file)

        # ----------------------------------------------------
        # If GPL could not be read from metadata
        # ----------------------------------------------------

        if not gpl_ids:

            print(f"WARNING: No GPL found for {matrix_file.name}")

            results.append(
                {
                    "gse_id": gse,
                    "matrix_file": matrix_file.name,
                    "gpl_id": "",
                    "gpl_annotation_files": "",
                }
            )

            continue

        # ----------------------------------------------------
        # Add one row for each GPL
        # ----------------------------------------------------

        for gpl_id in gpl_ids:

            annotation_files = find_gpl_files(gpl_id)

            results.append(
                {
                    "gse_id": gse,
                    "matrix_file": matrix_file.name,
                    "gpl_id": gpl_id,
                    "gpl_annotation_files": "; ".join(annotation_files),
                }
            )


# ============================================================
# 6. Create dataframe
# ============================================================

mapping = pd.DataFrame(results)


# ============================================================
# 7. Save result
# ============================================================

mapping.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# 8. Display result
# ============================================================

print("\n" + "=" * 70)
print("GSE → GPL MAPPING")
print("=" * 70)

print(mapping.to_string(index=False))


print("\n" + "=" * 70)
print("UNIQUE PLATFORMS")
print("=" * 70)

print(mapping["gpl_id"].replace("", pd.NA).dropna().value_counts().sort_index())


print("\n" + "=" * 70)
print("OUTPUT")
print("=" * 70)

print(OUTPUT_FILE)

print("\nSTEP 3 COMPLETE")

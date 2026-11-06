from pathlib import Path
import pandas as pd

# ============================================================
# 1. Directories
# ============================================================

PROJECT_DIR = Path(r"D:\Dementia2")
GPL_DIR = PROJECT_DIR / "data" / "gpl"
OUTPUT_FILE = PROJECT_DIR / "codes" / "annotation_column_summary.csv"


# ============================================================
# 2. Exact files to inspect
# ============================================================

FILES = [
    ("GPL96", "GPL96_annotation.csv", ","),
    ("GPL570", "GPL570_annotation.csv", ","),
    ("GPL4372", "GPL4372_annotation.csv", ","),
    ("GPL6244", "GPL6244_annotation.csv", ","),
    ("GPL6947", "GPL6947_annotation.csv", ","),
    ("GPL4757", "GPL4757.txt", "\t"),
    ("GPL5175", "GPL5175 (2).txt", "\t"),
]


# ============================================================
# 3. Read and summarize one annotation table
# ============================================================


def inspect_table(gpl_id, filename, separator):

    file_path = GPL_DIR / filename

    result = {
        "gpl_id": gpl_id,
        "file": filename,
        "status": "ERROR",
        "rows": 0,
        "columns": 0,
        "column_names": "",
        "error": "",
    }

    if not file_path.exists():

        result["status"] = "MISSING"
        return result

    try:

        df = pd.read_csv(
            file_path, sep=separator, dtype=str, low_memory=False, on_bad_lines="skip"
        )

        df.columns = [str(c).strip().strip('"') for c in df.columns]

        result["status"] = "OK"
        result["rows"] = len(df)
        result["columns"] = len(df.columns)
        result["column_names"] = " | ".join(df.columns)

        return result, df

    except Exception as e:

        result["error"] = str(e)
        return result, None


# ============================================================
# 4. Analyze columns
# ============================================================

all_results = []


print("\n" + "=" * 80)
print("ANNOTATION COLUMN INSPECTION")
print("=" * 80)


for gpl_id, filename, separator in FILES:

    output, df = inspect_table(gpl_id, filename, separator)

    if isinstance(output, tuple):
        output = output[0]

    all_results.append(output)

    print("\n" + "-" * 80)
    print(f"{gpl_id} : {filename}")
    print("-" * 80)

    if df is None:

        print(f"STATUS: {output['status']}")

        if output["error"]:
            print(f"ERROR : {output['error']}")

        continue

    print(f"Rows    : {len(df):,}")

    print(f"Columns : {len(df.columns)}")

    print("\nColumns and non-empty percentage:")

    for column in df.columns:

        values = df[column].astype(str).str.strip()

        nonempty = (values.ne("") & values.ne("nan") & values.ne("None")).mean() * 100

        print(f"  {column:<35} {nonempty:6.1f}%")


# ============================================================
# 5. GPL2700 special check
# ============================================================

print("\n" + "=" * 80)
print("GPL2700 SPECIAL CHECK")
print("=" * 80)

gpl2700 = GPL_DIR / "GPL2700.txt"

if not gpl2700.exists():

    print("GPL2700.txt: MISSING")

else:

    try:

        with open(gpl2700, "r", encoding="utf-8", errors="replace") as f:

            lines = []

            for i, line in enumerate(f):

                if i >= 40:
                    break

                line = line.rstrip()

                if line:
                    lines.append(line)

        print(f"File size: {gpl2700.stat().st_size:,} bytes")

        print(f"First {len(lines)} non-empty lines inspected.")

        # Only print structural information, not whole data.
        print("\nLine structure:")

        for i, line in enumerate(lines[:15], start=1):

            if "\t" in line:
                fields = line.split("\t")
                delimiter = "TAB"

            elif "," in line:
                fields = line.split(",")
                delimiter = "COMMA"

            else:
                fields = [line]
                delimiter = "NONE"

            print(f"  Line {i:02d}: " f"{delimiter}, " f"{len(fields)} field(s)")

    except Exception as e:

        print("ERROR:", e)


# ============================================================
# 6. GPL8757 special check
# ============================================================

print("\n" + "=" * 80)
print("GPL8757 SPECIAL CHECK")
print("=" * 80)

gpl8757 = GPL_DIR / "GPL8757.soft"

if not gpl8757.exists():

    print("GPL8757.soft: MISSING")

else:

    try:

        with open(gpl8757, "r", encoding="utf-8", errors="replace") as f:

            data_table_found = False
            table_header = None
            table_columns = 0
            table_rows = 0

            for line in f:

                line = line.rstrip("\n")

                # GEO platform metadata
                if line.startswith("!"):
                    continue

                # SOFT section metadata
                if line.startswith("^"):
                    continue

                if not line.strip():
                    continue

                # Look for tabular content
                if "\t" in line:

                    fields = line.split("\t")

                    if not data_table_found:

                        table_header = fields
                        table_columns = len(fields)
                        data_table_found = True

                    else:

                        table_rows += 1

            print(f"Table found      : " f"{'YES' if data_table_found else 'NO'}")

            if data_table_found:

                print(f"Header columns   : {table_columns}")

                print(f"Data rows        : {table_rows:,}")

                print("\nHeader:")

                print(" | ".join(table_header))

    except Exception as e:

        print("ERROR:", e)


# ============================================================
# 7. Save general summary
# ============================================================

summary = pd.DataFrame(all_results)

summary.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# 8. Final output
# ============================================================

print("\n" + "=" * 80)
print("OUTPUT")
print("=" * 80)

print(OUTPUT_FILE)

print("\nSTEP 6 COMPLETE")

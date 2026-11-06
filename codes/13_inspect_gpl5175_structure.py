from pathlib import Path

# ============================================================
# PATH
# ============================================================

FILE = Path(r"D:\Dementia2\data\gpl\GPL5175 (2).txt")


# ============================================================
# READ FILE
# ============================================================

print("\n" + "=" * 80)
print("GPL5175 STRUCTURE INSPECTION")
print("=" * 80)

if not FILE.exists():

    raise FileNotFoundError(FILE)


with open(FILE, "r", encoding="utf-8", errors="replace") as f:

    lines = f.readlines()


print(f"\nFile size : {FILE.stat().st_size:,} bytes")

print(f"Total lines : {len(lines):,}")


# ============================================================
# FIND IMPORTANT GEO MARKERS
# ============================================================

print("\n" + "-" * 80)
print("GEO TABLE MARKERS")
print("-" * 80)

markers = [
    "!platform_table_begin",
    "!platform_table_end",
    "#ID",
    "#GB_LIST",
    "#SPOT_ID",
]


for marker in markers:

    found = []

    for i, line in enumerate(lines):

        if marker.lower() in line.lower():

            found.append(i + 1)

    print(f"{marker:<30} : " f"{found[:10]}")


# ============================================================
# PRINT FIRST 30 NON-EMPTY LINES
# ============================================================

print("\n" + "-" * 80)
print("FIRST 30 NON-EMPTY LINES")
print("-" * 80)

count = 0

for i, line in enumerate(lines):

    line = line.rstrip("\r\n")

    if not line.strip():
        continue

    count += 1

    print(f"{i + 1:6d}: {line[:300]}")

    if count >= 30:
        break


# ============================================================
# FIND POSSIBLE TABLE HEADER
# ============================================================

print("\n" + "-" * 80)
print("POSSIBLE TABLE HEADER")
print("-" * 80)

for i, line in enumerate(lines):

    clean = line.strip()

    if clean.startswith("#ID") or clean.startswith("ID\t") or clean.startswith("ID,"):

        print(f"Line {i + 1}:")

        print(clean[:1000])


# ============================================================
# FIND GENE-RELATED TERMS
# ============================================================

print("\n" + "-" * 80)
print("GENE-RELATED TERMS FOUND")
print("-" * 80)

terms = [
    "gene",
    "symbol",
    "entrez",
    "refseq",
    "accession",
    "gb_list",
    "transcript",
    "unigene",
]


for term in terms:

    matches = []

    for i, line in enumerate(lines):

        if term.lower() in line.lower():

            matches.append(i + 1)

            if len(matches) >= 5:
                break

    if matches:

        print(f"{term:<15}: lines {matches}")

    else:

        print(f"{term:<15}: NONE")


# ============================================================
# SHOW LINES AROUND TABLE START
# ============================================================

print("\n" + "-" * 80)
print("LINES AROUND POSSIBLE TABLE START")
print("-" * 80)

table_start = None

for i, line in enumerate(lines):

    if "#ID" in line and ("=" in line or "\t" in line):

        table_start = i
        break


if table_start is not None:

    start = max(0, table_start - 3)

    end = min(len(lines), table_start + 12)

    for i in range(start, end):

        print(f"{i + 1:6d}: " f"{lines[i].rstrip()[:500]}")

else:

    print("No obvious #ID table header found.")


print("\n" + "=" * 80)
print("STEP 13 COMPLETE")
print("=" * 80)

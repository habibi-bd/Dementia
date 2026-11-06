import re
from pathlib import Path
import pandas as pd

from importlib.util import spec_from_file_location, module_from_spec

spec = spec_from_file_location("config", Path(__file__).parent / "00_config.py")
config = module_from_spec(spec)
spec.loader.exec_module(config)


def file_size_mb(path):
    return path.stat().st_size / (1024**2)


print("=" * 80)
print("DEMENTIA GEO DATASET AUDIT")
print("=" * 80)

for gse, gpl in config.MAIN_GSE_GPL.items():

    print(f"\n{gse}")
    print("-" * 60)
    print("Expected GPL:", gpl)

    candidates = list(config.MATRIX_DIR.glob(f"{gse}*"))

    if not candidates:
        print("WARNING: matrix not found")
        continue

    for f in candidates:
        print(f"  {f.name:55s}" f"{file_size_mb(f):10.2f} MB")

print("\nGPL FILES")
print("=" * 80)

for gpl in sorted(set(config.MAIN_GSE_GPL.values())):

    print(f"\n{gpl}")

    for f in config.GPL_DIR.glob(f"{gpl}*"):
        print(f"  {f.name:55s}" f"{file_size_mb(f):10.2f} MB")

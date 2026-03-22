"""Quick diagnostic: check CSVs are intact and columns are as expected."""

import sys
import pandas as pd

DFP_CSV = "/data/dataset/DFP/DFP0395_comprehensiveSft.csv"
CNS_CSV = "/data/dataset/CNS/CNS0091_comprehensiveSft.csv"
CNS_META = "/data/dataset/CNS/CNS0091_sourceMetadata.csv"

# Allow override from CLI
if len(sys.argv) > 1:
    DFP_CSV = sys.argv[1]
if len(sys.argv) > 2:
    CNS_CSV = sys.argv[2]
if len(sys.argv) > 3:
    CNS_META = sys.argv[3]

print("=" * 60)
print("DFP comprehensiveSft")
print("=" * 60)
dfp = pd.read_csv(DFP_CSV)
print(f"  Path:    {DFP_CSV}")
print(f"  Shape:   {dfp.shape}")
print(f"  Columns: {len(dfp.columns)}")
print()
print(f"  'drug1' present: {'drug1' in dfp.columns}")
if "drug1" in dfp.columns:
    print(f"  drug1 NaN:       {dfp['drug1'].isna().sum()} / {len(dfp)}")
    print(f"  drug1 unique:    {dfp['drug1'].nunique()}")
    print(f"  drug1 samples:   {dfp['drug1'].value_counts().head(5).to_dict()}")
    print(f"  drug1 dtype:     {dfp['drug1'].dtype}")
else:
    print("  WARNING: drug1 column MISSING")
    drug_cols = [c for c in dfp.columns if "drug" in c.lower()]
    print(f"  Columns with 'drug': {drug_cols}")

if "drug1Concentration" in dfp.columns:
    print(f"  drug1Concentration NaN: {dfp['drug1Concentration'].isna().sum()} / {len(dfp)}")
    print(f"  drug1Concentration unique: {dfp['drug1Concentration'].nunique()}")

print()
print("=" * 60)
print("CNS comprehensiveSft")
print("=" * 60)
cns = pd.read_csv(CNS_CSV)
print(f"  Path:    {CNS_CSV}")
print(f"  Shape:   {cns.shape}")
print(f"  Columns: {len(cns.columns)}")
print()
print(f"  'treatment' present: {'treatment' in cns.columns}")
print(f"  'screeningWellContents' present: {'screeningWellContents' in cns.columns}")
if "screeningWellContents" in cns.columns:
    print(f"  screeningWellContents unique: {cns['screeningWellContents'].nunique()}")

print()
print("=" * 60)
print("CNS sourceMetadata")
print("=" * 60)
try:
    meta = pd.read_csv(CNS_META)
    print(f"  Path:    {CNS_META}")
    print(f"  Shape:   {meta.shape}")
    print(f"  'treatment' present: {'treatment' in meta.columns}")
    if "treatment" in meta.columns:
        print(f"  treatment NaN:    {meta['treatment'].isna().sum()} / {len(meta)}")
        print(f"  treatment unique: {meta['treatment'].nunique()}")
    print(f"  'WellDescription_x' present: {'WellDescription_x' in meta.columns}")
    print(f"  'SourceNumber' present: {'SourceNumber' in meta.columns}")
except FileNotFoundError:
    print(f"  NOT FOUND: {CNS_META}")

print()
print("=" * 60)
print("Feature column overlap")
print("=" * 60)
shared = sorted(set(dfp.columns) & set(cns.columns))
print(f"  Shared columns: {len(shared)}")
numeric_shared = [c for c in shared if dfp[c].dtype in ("float64", "float32", "int64", "int32")]
print(f"  Shared numeric: {len(numeric_shared)}")

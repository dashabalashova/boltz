#!/usr/bin/env python3
import pandas as pd
import shutil
from pathlib import Path

# Paths
processed_dir = Path("data/cache2/processed")
smiles_tsv   = processed_dir / "smiles.tsv"
fasta_in     = Path("data/cache2/raw/YP_009725308.1.fasta")
fasta_out    = processed_dir / "protein.fasta"

# 1) make sure the directory is there
processed_dir.mkdir(parents=True, exist_ok=True)

# 2) read Excel & write TSV
df = pd.read_excel(
    "data/cache2/raw/CACHE2_Round1_09092024.xlsx",
    sheet_name=0,
    header=1,
    usecols=["CACHE ID", "Smiles"]
)
df.columns = ["id_raw", "smiles"]
df.to_csv(smiles_tsv, sep="\t", index=False)
print(f"Wrote {len(df)} rows to {smiles_tsv}")

# 3) copy FASTA
shutil.copy(fasta_in, fasta_out)
print(f"Copied {fasta_in} → {fasta_out}")

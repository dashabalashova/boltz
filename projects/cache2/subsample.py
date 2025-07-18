#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample SMILES from CACHE2 Excel and write to a TSV."
    )
    parser.add_argument(
        "--output-tsv", "-o",
        type=Path,
        required=True,
        help="Path to the output TSV file"
    )
    parser.add_argument(
        "--n-samples", "-n",
        type=int,
        default=8,
        help="Number of samples to draw (default: 8)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=0,
        help="Random seed for sampling (default: 0)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Read the first sheet, assuming headers on the second row
    df = pd.read_excel("data/cache2/raw/CACHE2_Round1_09092024.xlsx", 
                       sheet_name=0, header=1)

    # Sample CACHE ID and Smiles
    df_sample = (
        df[["CACHE ID", "Smiles"]]
        .sample(n=args.n_samples, random_state=args.seed)
        .reset_index(drop=True)
    )

    # Create ligand IDs
    df_sample["id"] = "lig_" + df_sample.index.astype(str)

    # Rename columns and select output order
    df_sample = df_sample.rename(columns={"Smiles": "smiles"})
    df_out = df_sample[["id", "smiles"]]

    # Ensure output directory exists
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)

    # Write TSV
    df_out.to_csv(args.output_tsv, sep="\t", index=False)

    print(
        f"Wrote {len(df_out)} samples to {args.output_tsv} "
        f"(seed={args.seed}, n={args.n_samples})"
    )


if __name__ == "__main__":
    main()

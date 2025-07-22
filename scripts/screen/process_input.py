#!/usr/bin/env python3  # noqa: EXE001

import argparse
from pathlib import Path

import pandas as pd
import string


def read_all_fasta_sequences(path: Path) -> list[tuple[str,str]]:
    """
    Returns a list of (header_id, sequence) tuples in the order they appear.
    header_id is whatever follows the '>' up to whitespace.
    """
    seqs = []
    header = None
    seq_lines = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                seqs.append((header, "".join(seq_lines)))
            header    = line[1:].split()[0]
            seq_lines = []
        else:
            seq_lines.append(line.strip())
    if header is not None:
        seqs.append((header, "".join(seq_lines)))
    return seqs


def parse_args() -> argparse.Namespace:  # noqa: D103
    parser = argparse.ArgumentParser(
        description="Generate YAML files from a single FASTA and a smiles.tsv"
    )
    parser.add_argument(
        "--fasta_path", "-f",
        type=Path,
        required=True,
        help="Path to the input FASTA file"
    )
    parser.add_argument(
        "--smiles_path", "-s",
        type=Path,
        required=True,
        help="Path to the TSV file with columns 'id' and 'smiles'"
    )
    parser.add_argument(
        "--out_dir", "-o",
        type=Path,
        required=True,
        help="Directory to write the YAML files"
    )
    return parser.parse_args()


def main() -> None:  # noqa: D103
    args = parse_args()
    fasta_path: Path = args.fasta_path
    smiles_path: Path = args.smiles_path
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read the protein sequence
    protein_seq = read_all_fasta_sequences(fasta_path)

    # Read the TSV with 'id' and 'smiles'
    df = pd.read_csv(smiles_path, sep="\t", dtype=str)
    if not {"id", "smiles"}.issubset(df.columns):
        msg = "TSV must contain columns 'id' and 'smiles'"
        raise KeyError(
            msg
        )

    proteins = read_all_fasta_sequences(fasta_path)  

    labels = list(string.ascii_uppercase)

    for _, row in df.iterrows():
        lig_id = row["id"]
        lig_sm = row["smiles"]

        # Assign A, B, … to each protein in order:
        seq_yaml_lines = ["sequences:"]
        for i, (header, seq) in enumerate(proteins):
            letter = labels[i]
            seq_yaml_lines += [
                f"  - protein:",
                f"      id: {letter}",
                f"      sequence: \"{seq}\"",
            ]

        # Now pick the next free letter for your ligand:
        ligand_letter = labels[len(proteins)]
        seq_yaml_lines += [
            "  - ligand:",
            f"      id: {ligand_letter}",
            f"      smiles: '{lig_sm}'",
        ]

        # Build the rest:
        yaml_text = "\n".join(
            ["version: 1", *seq_yaml_lines,
            "properties:",
            "  - affinity:",
            f"      binder: {ligand_letter}", ""]
        )

        out_path = out_dir / f"{lig_id}.yaml"
        out_path.write_text(yaml_text, encoding="utf-8")


if __name__ == "__main__":
    main()

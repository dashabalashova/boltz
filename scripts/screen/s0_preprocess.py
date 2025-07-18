#!/usr/bin/env python3  # noqa: EXE001

import argparse
from pathlib import Path

import pandas as pd


def read_first_fasta_sequence(fasta_path: Path) -> str:
    """
    Reads the first FASTA record from the given file.
    Returns the sequence as a string.
    """  # noqa: D205, D401
    header = None
    seq_lines: list[str] = []
    with fasta_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    break
                header = line[1:]
            else:
                seq_lines.append(line)
    return "".join(seq_lines)


def parse_args() -> argparse.Namespace:  # noqa: D103
    parser = argparse.ArgumentParser(
        description="Generate YAML files from a single FASTA and a smiles.tsv"
    )
    parser.add_argument(
        "--fasta-path", "-f",
        type=Path,
        required=True,
        help="Path to the input FASTA file"
    )
    parser.add_argument(
        "--smiles-path", "-s",
        type=Path,
        required=True,
        help="Path to the TSV file with columns 'id' and 'smiles'"
    )
    parser.add_argument(
        "--out-dir", "-o",
        type=Path,
        default=Path("yamls_CACHE2_v3"),
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
    protein_seq = read_first_fasta_sequence(fasta_path)

    # Read the TSV with 'id' and 'smiles'
    df = pd.read_csv(smiles_path, sep="\t", dtype=str)
    if not {"id", "smiles"}.issubset(df.columns):
        msg = "TSV must contain columns 'id' and 'smiles'"
        raise KeyError(
            msg
        )

    # Generate YAML for each ligand
    for _, row in df.iterrows():
        lig_id = row["id"]
        lig_sm = row["smiles"]
        yaml_text = (
            "version: 1\n"
            "sequences:\n"
            "  - protein:\n"
            "      id: A\n"
            f"      sequence: \"{protein_seq}\"\n"
            "  - ligand:\n"
            "      id: B\n"
            f"      smiles: '{lig_sm}'\n"
            "properties:\n"
            "  - affinity:\n"
            "      binder: B\n"
        )
        (out_dir / f"{lig_id}.yaml").write_text(yaml_text, encoding="utf-8")

    print(f"Generated {len(df)} YAML files in {out_dir}")  # noqa: T201


if __name__ == "__main__":
    main()

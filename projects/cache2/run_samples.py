#!/usr/bin/env python3  # noqa: EXE001

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:  # noqa: D103
    print("+", " ".join(cmd))  # noqa: T201
    subprocess.run(cmd, check=True)  # noqa: S603


def main() -> None:  # noqa: D103
    parser = argparse.ArgumentParser(
        description="Pipeline: subsample, preprocess, screen, structure & affinity runs"
    )
    parser.add_argument(
        "--n_samples", "-n", type=int, default=64,
        help="Number of SMILES to sample"
    )
    parser.add_argument(
        "--num_seeds", "-S", type=int, default=5,
        help="Number of different random seeds"
    )
    parser.add_argument(
        "--num_repeats", "-R", type=int, default=2,
        help="Number of repeats per seed"
    )
    parser.add_argument(
        "--batch_sizes", "-B", type=int, nargs="+",
        default=[1, 2, 4, 8],
        help="List of batch sizes to test"
    )
    parser.add_argument(
        "--raw_fasta", "-f", type=Path,
        default=Path("data/cache2/raw/YP_009725308.1.fasta"),
        help="Input FASTA file"
    )
    parser.add_argument(
        "--processed_dir", "-p", type=Path,
        default=Path("data/cache2/processed"),
        help="Base processed directory (for smiles.tsv & yamls)"
    )
    parser.add_argument(
        "--results_dir", "-r", type=Path,
        default=Path("results"),
        help="Base results directory"
    )
    parser.add_argument(
        "--subsample_script", type=Path,
        default=Path("projects/cache2/subsample.py"),
        help="Path to subsample.py"
    )
    parser.add_argument(
        "--preprocess_script", type=Path,
        default=Path("scripts/screen/process_input.py"),
        help="Path to s0_preprocess.py"
    )
    parser.add_argument(
        "--screen_script", type=Path,
        default=Path("scripts/screen/process_msa.py"),
        help="Path to s1_input_screen.py"
    )
    parser.add_argument(
        "--structure_script", type=Path,
        default=Path("scripts/predict/s2_structure.py"),
        help="Path to s2_structure.py"
    )
    parser.add_argument(
        "--affinity_script", type=Path,
        default=Path("scripts/predict/s3_affinity.py"),
        help="Path to s3_affinity.py"
    )
    args = parser.parse_args()

    # ensure dirs exist
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    for seed in range(args.num_seeds):
        # 1) subsample
        smiles_tsv = args.processed_dir / f"smiles_n{args.n_samples}_s{seed}.tsv"
        run([
            sys.executable,
            str(args.subsample_script),
            "-n", str(args.n_samples),
            "-s", str(seed),
            "-o", str(smiles_tsv),
        ])

        # 2) preprocess
        yamls_out = args.processed_dir / "yamls" / f"n{args.n_samples}_s{seed}"
        run([
            sys.executable,
            str(args.preprocess_script),
            "-f", str(args.raw_fasta),
            "-s", str(smiles_tsv),
            "-o", str(yamls_out),
        ])

        # 3) screening / MSA
        screen_out = args.results_dir / f"cache2_n{args.n_samples}_s{seed}"
        run([
            sys.executable,
            str(args.screen_script),
            "-y", str(yamls_out),
            "-o", str(screen_out),
        ])

        # 4) for each repeat and batch size: copy and run structure+affinity
        for repeat in range(args.num_repeats):
            for b in args.batch_sizes:
                name = f"cache2_n{args.n_samples}_s{seed}_b{b}_r{repeat}"
                target = args.results_dir / name

                # copy base screen results into new dir
                shutil.copytree(screen_out, target, dirs_exist_ok=True)

                # structure
                run([
                    sys.executable,
                    str(args.structure_script),
                    "-b", str(b),
                    "--out_dir", str(target),
                ])
                # affinity
                run([
                    sys.executable,
                    str(args.affinity_script),
                    "-b", str(b),
                    "--out_dir", str(target),
                ])


if __name__ == "__main__":
    main()

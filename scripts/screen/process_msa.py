#!/usr/bin/env python3

import argparse
import subprocess
import sys
import shutil
from pathlib import Path


def run_single(yaml: Path, script: Path, out_dir: Path) -> None:
    """
    Run the prediction script on a single YAML (lig_0.yaml).
    """
    cmd = [
        sys.executable,
        str(script),
        "--yamls", str(yaml),
        "--out_dir", str(out_dir)
    ]
    print(f"Running single: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def copy_msa_files(out_dir: Path, n_files: int) -> None:
    """
    Copy lig_0_0.csv → lig_{i}_0.csv for i=1..n_files-1 in the msa subdirectory.
    """
    msa_dir = out_dir / "msa"
    src = msa_dir / "lig_0_0.csv"
    if not src.exists():
        raise FileNotFoundError(f"Source MSA file not found: {src}")
    for i in range(1, n_files):
        dst = msa_dir / f"lig_{i}_0.csv"
        shutil.copy(src, dst)
        print(f"Copied {src.name} → {dst.name}")

    src = msa_dir / "lig_0_1.csv"
    if src.exists():
        for i in range(1, n_files):
            dst = msa_dir / f"lig_{i}_1.csv"
            shutil.copy(src, dst)
            print(f"Copied {src.name} → {dst.name}")


def run_batch(yamls: list[Path], script: Path, out_dir: Path) -> None:
    """
    Run the prediction script on all ligand YAMLs in one invocation.
    """
    cmd = [
        sys.executable,
        str(script),
        "--yamls", *(str(p) for p in yamls),
        "--out_dir", str(out_dir)
    ]
    print(f"Running batch: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run s1_inputs.py on a set of ligand YAMLs, replicate MSA outputs, then rerun in batch."
    )
    parser.add_argument(
        "--yamls_dir", "-y",
        type=Path,
        required=True,
        help="Directory containing ligand YAMLs (lig_*.yaml), excluding screen.yaml"
    )
    parser.add_argument(
        "--out_dir", "-o",
        type=Path,
        required=True,
        help="Output directory for MSA results (must contain an 'msa' subfolder)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script = "scripts/predict/s1_inputs.py"
    yamls_dir = args.yamls_dir
    out_dir = args.out_dir

    # Gather ligand YAML files
    yamls = sorted(
        p for p in yamls_dir.iterdir()
        if p.suffix == ".yaml" and p.name.startswith("lig_") and p.name != "screen.yaml"
    )
    n_files = len(yamls)
    if n_files == 0:
        print(f"No ligand YAMLs found in {yamls_dir}")
        sys.exit(1)

    print(f"Found {n_files} ligand YAML files in {yamls_dir}")

    # Step 1: run on the first YAML
    run_single(yamls[0], script, out_dir)

    # Step 2: replicate the first MSA output to all others
    copy_msa_files(out_dir, n_files)

    # Step 3: run on all YAMLs in batch
    run_batch(yamls, script, out_dir)


if __name__ == "__main__":
    main()

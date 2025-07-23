#!/usr/bin/env python3
import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

import glob
import json

import pandas as pd
import time

def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)

def parse_args():
    p = argparse.ArgumentParser(
        description="Split smiles.tsv into sequential blocks and run preprocess → screen → structure → affinity"
    )
    p.add_argument(
        "--smiles_tsv", "-s",
        type=Path, required=True,
        help="Path to the full smiles.tsv"
    )
    p.add_argument(
        "--protein_fasta", "-f",
        type=Path, required=True,
        help="Path to protein.fasta"
    )
    p.add_argument(
        "--g_samples", "-g",
        type=int, default=64,
        help="Number of SMILES per group"
    )
    p.add_argument(
        "--batch_size", "-B",
        type=int, default=1,
        help="Batch size"
    )
    p.add_argument(
        "--results_dir", "-r",
        type=Path, default=Path("results"),
        help="Where to write final outputs"
    )
    p.add_argument("--preprocess_script", type=Path,
        default=Path("scripts/screen/process_input.py"))
    p.add_argument("--screen_script", type=Path,
        default=Path("scripts/screen/process_msa.py"))
    p.add_argument("--structure_script", type=Path,
        default=Path("scripts/predict/s2_structure.py"))
    p.add_argument("--affinity_script", type=Path,
        default=Path("scripts/predict/s3_affinity.py"))
    return p.parse_args()

def main():
    group_start = time.monotonic()
    args = parse_args()
    processed_dir = args.results_dir / "processed_data"
    processed_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # load full smiles list
    df_all = pd.read_csv(args.smiles_tsv, sep="\t")
    total = len(df_all)
    groups = math.ceil(total / args.g_samples)

    group_end = time.monotonic()
    elapsed = round(group_end - group_start, 3)
    group_times = []
    group_times.append((-1, elapsed))
    for g in range(groups):

        group_start = time.monotonic()
        # 1) extract block
        start, end = g*args.g_samples, (g+1)*args.g_samples
        df_block = df_all.iloc[start:end].reset_index(drop=True)
        df_block["id"] = "lig_" + df_block.index.astype(str)
        df_block = df_block[["id_raw", "id", "smiles"]]
        block_tsv = processed_dir / f"{args.smiles_tsv.stem}_g{g}.tsv"
        df_block.to_csv(block_tsv, sep="\t", index=False)
        print(f"[block {g}] Wrote {len(df_block)} rows to {block_tsv}")

        # 2) preprocess block → yamls
        yamls_dir = processed_dir / "yamls" / f"g{g}"
        yamls_dir.mkdir(parents=True, exist_ok=True)
        run([
            sys.executable, str(args.preprocess_script),
            "--smiles_path", str(block_tsv),
            "--fasta_path", str(args.protein_fasta),
            "--out_dir", str(yamls_dir),
        ])

        # 3) screening / MSA → screen_out
        screen_out = args.results_dir / f"screen_g{g}"
        run([
            sys.executable, str(args.screen_script),
            "--yamls_dir", str(yamls_dir),
            "--out_dir",   str(screen_out),
        ])

        # 4) structure & affinity per batch & repeat
        b = args.batch_size
        name   = f"screen_g{g}_b{b}"
        target = args.results_dir / name
        if target.exists():
            shutil.rmtree(target)
        shutil.move(screen_out, target)

        # structure
        run([
            sys.executable, str(args.structure_script),
            "--b_size", str(b),
            "--out_dir",     str(target),
        ])
        # affinity
        run([
            sys.executable, str(args.affinity_script),
            "--b_size", str(b),
            "--out_dir",     str(target),
        ])

        group_end = time.monotonic()
        elapsed = round(group_end - group_start, 3)
        group_times.append((g, elapsed))
    
    # out_path = args.results_dir / f"screen_n{args.g_samples}_group_times.txt"
    # with open(out_path, 'w') as f:
    #     for g, t in group_times:
    #         f.write(f"{g} {t}\n")

    merged = []
    for g in range(groups):
        df = pd.read_csv(processed_dir / f"smiles_g{g}.tsv", sep='\t')
        df['group'] = g

        preds = []

        pattern = str(args.results_dir / f"screen_g{g}_b{args.batch_size}" / "predictions" / "*" / "affinity_*.json")
        for fp in glob.glob(pattern):
            data = json.load(open(fp))
            ligand = fp.split('/')[-2]
            preds.append({'id': ligand, 'affinity_pred_value': data['affinity_pred_value']})

        df_pred = pd.DataFrame(preds)
        merged.append(df.merge(df_pred, on='id', how='left'))

    result = pd.concat(merged, ignore_index=True)[['id_raw', 'smiles', 'affinity_pred_value']]
    result.to_csv(args.results_dir / Path(f"affinity_pred_values.tsv"), sep='\t', index=False)
    


if __name__ == "__main__":
    main()

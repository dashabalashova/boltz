#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Run Boltz2 preprocessing with process_inputs")
    parser.add_argument(
        '--yamls', '-y',
        nargs='+',
        required=True,
        help='Paths to input YAML files'
    )
    parser.add_argument(
        '--out_dir', '-o',
        type=str,
        default='output',
        help='Base output directory (default: output)'
    )
    parser.add_argument(
        '--cache_dir', '-c',
        type=str,
        default='cache',
        help='Cache directory (default: cache)'
    )
    parser.add_argument(
        '--ccd_path',
        type=str,
        default=None,
        help='Path to CCD pickle file (default: cache_dir/ccd.pkl)'
    )
    parser.add_argument(
        '--mol_tar',
        type=str,
        default=None,
        help='Path to mols.tar (default: cache_dir/mols.tar)'
    )
    parser.add_argument(
        '--msa_server_url',
        type=str,
        default='https://api.colabfold.com',
        help='MSA server URL (default: https://api.colabfold.com)'
    )
    parser.add_argument(
        '--msa_pairing_strategy',
        choices=['greedy', 'sequential'],
        default='greedy',
        help='MSA pairing strategy (default: greedy)'
    )
    parser.add_argument(
        '--preprocessing_threads',
        type=int,
        default=1,
        help='Number of preprocessing threads (default: 1)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    start_time = time.time()

    # Setup paths
    cache_dir = Path(args.cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = cache_dir / 'boltz2_aff.ckpt'
    mol_dir = cache_dir / 'mols'
    mol_tar_path = Path(args.mol_tar) if args.mol_tar else cache_dir / 'mols.tar'
    ccd_path = Path(args.ccd_path) if args.ccd_path else cache_dir / 'ccd.pkl'

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import and run preprocessing
    from boltz.main import process_inputs
    process_inputs(
        data=[Path(y) for y in args.yamls],
        out_dir=out_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        use_msa_server=True,
        msa_server_url=args.msa_server_url,
        msa_pairing_strategy=args.msa_pairing_strategy,
        preprocessing_threads=args.preprocessing_threads,
        boltz2=True
    )

    # Print execution time
    end_time = time.time()
    hrs, rem = divmod(end_time - start_time, 3600)
    mins, secs = divmod(rem, 60)
    print(f"Total execution time: {int(hrs)}h {int(mins)}m {secs:.2f}s")

if __name__ == '__main__':
    main()

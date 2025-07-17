#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import torch, gc
from pytorch_lightning import Trainer

from boltz.main import (
    Boltz2DiffusionParams,
    PairformerArgsV2,
    MSAModuleArgs,
    BoltzSteeringParams
)
from boltz.model.models.boltz2 import Boltz2
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import Manifest
from boltz.data.write.writer import BoltzAffinityWriter, BoltzWriter


def parse_args():
    parser = argparse.ArgumentParser(description="Run Boltz2 affinity predictions")
    parser.add_argument(
        "-b", "--b_size",
        type=int,
        default=1,
        help="Batch size for Boltz2InferenceDataModule (default: 1)"
    )
    parser.add_argument(
        "--diffusion_samples",
        type=int,
        default=3,
        help="Number of diffusion samples (default: 3)"
    )
    parser.add_argument(
        "--max_parallel_samples",
        type=int,
        default=5,
        help="Maximum parallel diffusion samples (default: 5)"
    )
    parser.add_argument(
        "--step_scale",
        type=float,
        default=1.5,
        help="Scale for diffusion step size (default: 1.5)"
    )
    parser.add_argument(
        "--out_dir", "-o",
        type=str,
        default="output",
        help="Base output directory (default: output)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    start_time = time.time()

    # Paths
    out_dir = Path(args.out_dir)
    processed_dir = out_dir / "processed"
    cache_dir = Path("~/.boltz").expanduser()
    mol_dir = cache_dir / "mols"

    # Load manifest
    manifest = Manifest.load(processed_dir / "manifest.json")

    # Configure diffusion parameters
    diffusion_params = Boltz2DiffusionParams()
    diffusion_params.step_scale = args.step_scale

    # Load model checkpoint
    model = Boltz2.load_from_checkpoint(
        cache_dir / "boltz2_aff.ckpt",
        strict=True,
        predict_args={
            "recycling_steps": 3,
            "sampling_steps": 100,
            "diffusion_samples": args.diffusion_samples,
            "max_parallel_samples": args.max_parallel_samples,
            "write_confidence_summary": False,
            "write_full_pae": False,
            "write_full_pde": False,
        },
        map_location="cuda",
        diffusion_process_args=diffusion_params.__dict__,
        ema=False,
        pairformer_args=PairformerArgsV2().__dict__,
        msa_args=MSAModuleArgs().__dict__,
        steering_args=BoltzSteeringParams().__dict__,
        affinity_mw_correction=False
    )
    model.eval()

    # Writers
    affinity_writer = BoltzAffinityWriter(
        data_dir=processed_dir / "structures",
        output_dir=out_dir / "predictions"
    )
    structure_writer = BoltzWriter(
        data_dir=processed_dir / "structures",
        output_dir=out_dir / "predictions/structure",
        output_format="mmcif",
        boltz2=True
    )

    # Trainer
    trainer = Trainer(
        default_root_dir=out_dir,
        callbacks=[structure_writer, affinity_writer],
        accelerator="gpu",
        devices=1,
        precision=32
    )

    # Data module
    data_module = Boltz2InferenceDataModule(
        manifest=manifest,
        target_dir=out_dir / "predictions",
        msa_dir=processed_dir / "msa",
        mol_dir=mol_dir,
        num_workers=8,
        constraints_dir=processed_dir / "constraints",
        template_dir=processed_dir / "templates",
        extra_mols_dir=processed_dir / "mols",
        override_method="other",
        affinity=True,
        batch_size=args.b_size
    )

    # Replace writer callback
    trainer.callbacks[0] = affinity_writer

    print(f"Running affinity predict with batch_size={args.b_size}, diffusion_samples={args.diffusion_samples}, max_parallel_samples={args.max_parallel_samples}, step_scale={args.step_scale}...")
    
    predict_start = time.time()
    trainer.predict(model, datamodule=data_module, return_predictions=False)
    predict_end = time.time()

    # Cleanup
    del model, trainer, data_module
    torch.cuda.empty_cache()
    gc.collect()

    end_time = time.time()

    # Print execution time
    hrs, rem = divmod(predict_end - predict_start, 3600)
    mins, secs = divmod(rem, 60)
    print(f"Prediction time: {int(hrs)}h {int(mins)}m {secs:.2f}s")
    
    hrs, rem = divmod(end_time - start_time, 3600)
    mins, secs = divmod(rem, 60)
    print(f"Total execution time: {int(hrs)}h {int(mins)}m {secs:.2f}s")

    log_path = Path(args.out_dir) / "run_times.csv"
    header = not log_path.exists()
    with open(log_path, "a") as log:
        if header:
            log.write("script,b_size,diffusion_samples,max_parallel_samples,predict_time_s\n")
        log.write(f"s3_affinity,{args.b_size},{args.diffusion_samples},{args.max_parallel_samples},"
              f"{predict_end-predict_start:.2f}\n")


if __name__ == "__main__":
    main()

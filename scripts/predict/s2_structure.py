#!/usr/bin/env python3  # noqa: EXE001
import argparse
import gc
import time
from dataclasses import asdict
from pathlib import Path

import torch
from pytorch_lightning import Trainer

from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import Manifest
from boltz.data.write.writer_screen import BoltzWriter_screen
from boltz.main import (
    Boltz2DiffusionParams,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgsV2,
)
from boltz.model.models.boltz2 import Boltz2


def parse_args():  # noqa: ANN201, D103
    parser = argparse.ArgumentParser(description="Run Boltz2 inference")
    parser.add_argument(
        "-b", "--b_size", type=int, default=1,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--diffusion_samples", type=int, default=1,
        help="Number of diffusion samples"
    )
    parser.add_argument(
        "--max_parallel_samples", type=int, default=5,
        help="Max parallel diffusion samples"
    )
    parser.add_argument(
        "--out_dir", type=str, default="output",
        help="Output directory"
    )
    return parser.parse_args()


def main() -> None:  # noqa: D103
    args = parse_args()
    start_time = time.time()

    out_dir = Path(args.out_dir)
    processed_dir = out_dir / "processed"
    cache_dir = Path("~/.boltz").expanduser()
    mol_dir = cache_dir / "mols"
    manifest = Manifest.load(processed_dir / "manifest.json")

    diffusion_params = Boltz2DiffusionParams()
    diffusion_params.step_scale = 1.5

    steering_args = BoltzSteeringParams()
    steering_args.fk_steering = False
    steering_args.physical_guidance_update = False

    model = Boltz2.load_from_checkpoint(
        cache_dir / "boltz2_conf.ckpt",
        strict=True,
        predict_args={
            "recycling_steps": 3,
            "sampling_steps": 200,
            "diffusion_samples": args.diffusion_samples,
            "max_parallel_samples": args.max_parallel_samples,
            "write_confidence_summary": True,
            "write_full_pae": False,
            "write_full_pde": False,
        },
        use_kernels=True,
        map_location="cuda",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(MSAModuleArgs()),
        steering_args=asdict(steering_args)
    )
    model.eval()

    writer = BoltzWriter_screen(
        data_dir=processed_dir / "structures",
        output_dir=out_dir / "predictions",
        output_format="mmcif",
        boltz2=True,
    )

    trainer = Trainer(
        default_root_dir=out_dir,
        callbacks=[writer],
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
    )

    data_module = Boltz2InferenceDataModule(
        manifest=manifest,
        target_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        mol_dir=mol_dir,
        num_workers=8,
        constraints_dir=processed_dir / "constraints",
        template_dir=processed_dir / "templates",
        extra_mols_dir=processed_dir / "mols",
        batch_size=args.b_size,
    )

    print(
        f"Running strucuture predict with batch_size={args.b_size}, "
        f"diffusion_samples={args.diffusion_samples}, "
        f"max_parallel_samples={args.max_parallel_samples}"
    )

    predict_start = time.time()
    trainer.predict(model, datamodule=data_module, return_predictions=False)
    predict_end = time.time()

    torch.cuda.empty_cache()
    gc.collect()

    hrs, rem = divmod(predict_end - predict_start, 3600)
    mins, secs = divmod(rem, 60)
    print(f"Prediction time: {int(hrs)}h {int(mins)}m {secs:.2f}s")
    total_time = time.time() - start_time
    hrs, rem = divmod(total_time, 3600)
    mins, secs = divmod(rem, 60)
    print(f"Total execution time: {int(hrs)}h {int(mins)}m {secs:.2f}s")

    log_path = out_dir / "run_times.csv"
    header = not log_path.exists()
    with open(log_path, "a") as log:
        if header:
            log.write(
                "script,b_size,diffusion_samples,max_parallel_samples,predict_time_s,total_time_s\n"
            )
        log.write(
            f"s2_structure,{args.b_size},{args.diffusion_samples},"
            f"{args.max_parallel_samples},{predict_end - predict_start:.2f},{total_time:.2f}\n"
        )


if __name__ == "__main__":
    main()

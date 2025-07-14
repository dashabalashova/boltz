#!/usr/bin/env python3
import subprocess, time, csv
from pathlib import Path

EXPERIMENTS = [
    (1, [f"yamls/ex{i}.yaml" for i in range(1, 31)]),
    (2, [f"yamls/ex{i}.yaml" for i in range(1, 31)]),
    (4, [f"yamls/ex{i}.yaml" for i in range(1, 31)]),
    (8, [f"yamls/ex{i}.yaml" for i in range(1, 31)])
]

LOG = Path("all_runs.csv")
with LOG.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["batch","out_dir","script","load_time_s","predict_time_s"])
    subprocess.run(["python3","projects/test_run_30/s1_inputs.py","--yamls", *EXPERIMENTS[0][1],"--out_dir","msa_30"], check=True)

    for batch, yamls in EXPERIMENTS:
        out_dir = f"output_test_run_30_b{batch}"
        
        subprocess.run(["cp","-r","msa_30",out_dir], check=True)
        
        # Шаг 1: preprocess
        subprocess.run(["python3","projects/test_run_30/s1_inputs.py","--yamls", *yamls,"--out_dir",out_dir], check=True)

        # Шаг 2: structure
        t0 = time.time()
        subprocess.run(["python3","projects/test_run_30/s2_structure.py","-b",str(batch),"--out_dir",out_dir], check=True)
        t1 = time.time()

        # Шаг 3: affinity
        t2 = time.time()
        subprocess.run(["python3","projects/test_run_30/s3_affinity.py","-b",str(batch),"--out_dir",out_dir], check=True)
        t3 = time.time()

        writer.writerow([batch,out_dir,"s2_structure","",f"{t1-t0:.2f}"])
        writer.writerow([batch,out_dir,"s3_affinity","",f"{t3-t2:.2f}"])

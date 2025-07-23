#!/usr/bin/env python3
import glob
import json
import re
import pandas as pd

def main():
    # find group indices
    paths = glob.glob("data/cache2/processed/smiles_g*.tsv")
    groups = [int(m.group(1)) for p in paths
              if (m := re.search(r"smiles_g(\d+)\.tsv$", p))]

    merged = []
    for g in range(max(groups) + 1):
        df = pd.read_csv(f"data/cache2/processed/smiles_g{g}.tsv", sep='\t')
        df['group'] = g

        preds = []
        for fp in glob.glob(f"results/cache2_n64_g{g}_b8/predictions/*/affinity_*.json"):
            data = json.load(open(fp))
            ligand = fp.split('/')[-2]
            preds.append({'id': ligand, 'affinity_pred_value': data['affinity_pred_value']})

        df_pred = pd.DataFrame(preds)
        merged.append(df.merge(df_pred, on='id', how='left'))

    result = pd.concat(merged, ignore_index=True)[['id_raw', 'smiles', 'affinity_pred_value']]
    result.to_csv('results/cache2_affinity.tsv', sep='\t', index=False)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Difficulty profile of the suite in the representation learned by each deep method.

For every dataset of the suite, the profile is computed (clustaudit.metrics.profile.
difficulty_profile, the chain used for external data) on the standardized input, which is
what the deep methods receive, and on the embedding each of IDEC, DCN and DipDECK learned
with the first seed (results/deep_embeddings_seed42.npz, written by run_deep_clustering.py).
The reference labels are the suite's; noise points keep their label -1.

Usage:   python benchmarks/deep/embedding_profile.py [--data_dir datasets/suite]
Output:  results/deep_embedding_profiles.csv   one row per (dataset, representation):
         representation in {input, IDEC, DCN, DipDECK}, then every descriptor of the profile.
Needs only the core install (the embeddings are read from the result file); about a minute.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from clustaudit.metrics.profile import difficulty_profile

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
METHODS = ("IDEC", "DCN", "DipDECK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=os.path.join(HERE, os.pardir, os.pardir, "datasets", "suite"))
    a = ap.parse_args()
    emb = np.load(os.path.join(RESULTS, "deep_embeddings_seed42.npz"))
    rows = []
    for path in sorted(glob.glob(os.path.join(a.data_dir, "*.npz"))):
        name = os.path.splitext(os.path.basename(path))[0]
        d = np.load(path)
        X, y = d["X"], d["y"]
        spaces = [("input", StandardScaler().fit_transform(X.astype(np.float64)))]
        spaces += [(m, emb[f"{name}|{m}"].astype(np.float64)) for m in METHODS if f"{name}|{m}" in emb.files]
        for space, Z in spaces:
            row = {"dataset": name, "representation": space, "embedding_dim": Z.shape[1]}
            row.update(difficulty_profile(Z, y, seed=0).to_dict())
            rows.append(row)
        print(f"{name}: {len(spaces)} representations", flush=True)
    out = os.path.join(RESULTS, "deep_embedding_profiles.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

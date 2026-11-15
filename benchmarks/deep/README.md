# benchmarks/deep/

Deep tier: IDEC, DCN (K given) and DipDECK (K estimated), from ClustPy, run under
their reference defaults on the 75 datasets of `datasets/suite`. Each method is
treated as a black box `(X, seed) -> labels`; internal indices are computed in the
original space, not in the learned embedding.

```bash
python -m pip install clustpy torch
python benchmarks/deep/run_deep_clustering.py --data_dir datasets/suite [--resume]
```

Five runs per dataset; results are saved incrementally to `results/deep_results.csv`
(`--resume` continues an interrupted run). About ten hours on CPU.

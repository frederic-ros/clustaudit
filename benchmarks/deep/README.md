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
(`--resume` continues an interrupted run). Ten to fifteen hours on CPU.

Every partition is evaluated by `clustaudit.audit.evaluation.evaluate_partition`, as in
the classical tier: `ARI` is the inlier ARI, `ARI_full` counts noise as a class (the
convention of this tier up to release 2.2.1), and the smallest-cluster F1 and the failure
mode are recorded. The script also writes every label vector (`results/deep_labels.npz`,
key `dataset|method|seed`), the embedding learned with the first seed
(`results/deep_embeddings_seed42.npz`, key `dataset|method`) and the versions of the run
(`results/deep_run_info.csv`).

```bash
python benchmarks/deep/embedding_profile.py --data_dir datasets/suite
```

computes the difficulty profile in the standardized input and in each learned
representation (`results/deep_embedding_profiles.csv`, about a minute).

The shipped files come from a run on Windows (see `results/deep_run_info.csv`); under the
former convention they reproduce the 2.2.1 file exactly, for all 1125 runs.

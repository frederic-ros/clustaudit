# benchmarks/real-data/

Nineteen UCI / OpenML datasets (from `iris` to `waveform`), labels taken as given.
The difficulty profile is computed with the same instrument as on generated data;
the eight classical methods are run and two label-free quantities are recorded: the
median Silhouette of the non-degenerate partitions and the fraction of degenerate
outputs.

```bash
python benchmarks/real-data/run_transfer.py            # labels as given
python benchmarks/real-data/run_transfer.py --relabel  # labels made consistent with the k-NN rule
```

The datasets are fetched from scikit-learn and OpenML on first use (network access
required). Results: `results/transfer_results.csv`, `transfer_results_relabel.csv`,
`transfer_per_algo.csv`.

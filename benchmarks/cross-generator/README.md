# benchmarks/cross-generator/

Two independent generators, MDCGen (compactness factor, seven levels) and HAWKS
(target Silhouette 0.9 to 0.3), five seeds each, `n = 1000`, `d = 20`, `K = 5`, the
nine-method pool under protocols (F) and (S). For every dataset the difficulty
profile is reconstructed from `(X, y)` exactly as on the suite.

```bash
python -m pip install git+https://github.com/CN-TU/mdcgenpy hawks
python benchmarks/cross-generator/run_crossgen.py
python benchmarks/cross-generator/analyze_crossgen.py
```

`hawks` 0.2.0 expects `pandas < 2`; `run_crossgen.py` contains a small compatibility
shim for newer pandas. Results: `results/crossgen_results.csv`;
`results/predictive_summary.csv` is written by `analysis/audit_value_v2.py`.

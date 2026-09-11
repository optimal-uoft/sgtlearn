# Baseline: regularized outer growth

Pinned revision `90fb148e8b9b4181534dd2a19a5837aa85fba506`; 99 fresh-process runs completed: 45 measured SGT repetitions, 27 SGT warmup/quality runs, 24 CART references and 3 explicit CART exclusions. No benchmark failures. See `raw.jsonl`, `summary.json`, `manifest.json`, `dependencies.txt` and `build-provenance.txt` for reproduction.

Apple M5 Pro (15 cores, 24 GiB), macOS 26.6.2, Python 3.14.5, scikit-learn 1.9.0; isolated noneditable wheel, Apple Clang 21, Release `-O3 -DNDEBUG`, Accelerate. Native/import paths and SHA256 hashes verified in every successful worker. BLAS/OpenMP threads fixed at one; parallel forest uses two joblib threads. Root paused other builds/tests during timing.

| Workload | Fit median ± MAD (ms) | Predict (M rows/s) | Fit RSS (MiB) | SGT training quality, seeds min–max | CART | Worse than CART | Leaves, seed 17 |
|---|---:|---:|---:|---:|---:|---|---:|
| entropy_multiway_pair_budget | 70.30 ± 0.26 | 12.30 | 189.5 | accuracy 0.6875–0.6892 | 0.7132–0.7132 | 3/3 | 12–12 |
| gini_binary_queue | 35.38 ± 0.33 | 8.46 | 190.1 | accuracy 0.7602–0.7602 | 0.7812–0.7812 | 3/3 | 31–31 |
| gini_categorical_missing_multioutput | 15.50 ± 0.11 | 9.33 | 189.3 | accuracy 0.6111–0.6111 | Excluded | 0/3 (excluded) | 11–11 |
| gini_default_tao | 16.67 ± 0.08 | 9.89 | 188.7 | accuracy 0.6667–0.6667 | 0.6913–0.6913 | 3/3 | 11–11 |
| gini_forest_parallel | 71.80 ± 0.70 | 2.29 | 213.0 | accuracy 0.7073–0.7300 | 0.7300–0.7300 | 2/3 | 15–16 / tree |
| gini_forest_serial | 60.72 ± 0.17 | 2.28 | 191.8 | accuracy 0.7073–0.7300 | 0.7300–0.7300 | 2/3 | 15–16 / tree |
| mae_cd_off | 84.75 ± 0.25 | 16.39 | 191.1 | mae 0.9869–0.9869 | 0.8668–0.8668 | 3/3 | 11–11 |
| mae_cd_on | 330.10 ± 0.98 | 17.62 | 191.3 | mae 0.8888–0.8888 | 0.8668–0.8668 | 3/3 | 11–11 |
| mse_uncapped_multioutput | 69.95 ± 0.16 | 9.10 | 189.7 | mse 0.4369–0.4388 | 0.9057–0.9057 | 0/3 | 100–100 |

Training quality is the primary soft comparison signal. Every worse-than-CART seed is explicitly flagged in `summary.json`; these flags are reference findings, not failures. The categorical/missing case excludes CART because reproducing grouped category and missing-category semantics would change preprocessing. Classification multioutput quality averages per-output weighted accuracies; regression averages per-output weighted loss. No held-out metric replaces these training results.

The old multiway implementation already exceeds several requested structural leaf caps; preserve these observed structures when interpreting before/after performance. Detailed per-seed/per-tree structural and occupied leaf counts, depths and node counts are in the JSON. The parallel forest has appreciable per-run variation, so a threshold breach must be confirmed with matched reruns. Fit batches are 33–670 ms (individual fits ≥15 ms); imports/startup are excluded. Prediction timing spans ≥250 ms per process. Memory is 5 ms sampled process-tree RSS, with the OS lifetime high-water RSS also saved; it includes Python/import/data footprint.

Frozen review gates: investigate >25% median fit or prediction slowdown and >50% peak-memory increase; confirm beyond measured noise. Similar/improved training accuracy expected, slight degradation acceptable, large reproducible drops investigated without a strict numeric cutoff. All common configurations have zero penalties. Baseline cannot accept public branching_penalty; equal positive old/new parameter values have different units.

Validation: benchmark metric/summary self-check passed; ruff passed; raw-data audit confirmed all 9 workloads × 5 measured repetitions, 3 quality seeds and identical dataset hashes across models/seeds. Separate pre-implementation test record from root: 77 initialization/classifier/regressor fidelity checks passed; mypy over sgtlearn passed (12 files). No known baseline failures were supplied. The full implementation test suite remains separate from benchmark status.

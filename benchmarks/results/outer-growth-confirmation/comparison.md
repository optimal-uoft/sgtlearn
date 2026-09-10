# Matched outer-growth comparison

Training quality is the primary soft signal. Changes below are candidate minus baseline: accuracy changes use percentage points (pp); regression losses show absolute and relative changes. Higher accuracy and lower loss are better. No numeric quality cutoff is imposed. Every worse-than-CART seed remains flagged.

| Workload / seed | Metric | Baseline | Candidate | Change | CART | Worse than CART (old/new) | Structural leaves (old/new) |
|---|---|---:|---:|---:|---:|---|---|
| gini_binary_queue / 17 | accuracy | 0.760167 | 0.765167 | +0.500 pp | 0.781167 | True/True | [31]/[31] |
| gini_binary_queue / 29 | accuracy | 0.760167 | 0.765167 | +0.500 pp | 0.781167 | True/True | [31]/[31] |
| gini_binary_queue / 43 | accuracy | 0.760167 | 0.765167 | +0.500 pp | 0.781167 | True/True | [31]/[31] |
| gini_default_tao / 17 | accuracy | 0.666667 | 0.694667 | +2.800 pp | 0.691333 | True/False | [11]/[10] |
| gini_default_tao / 29 | accuracy | 0.666667 | 0.694667 | +2.800 pp | 0.691333 | True/False | [11]/[10] |
| gini_default_tao / 43 | accuracy | 0.666667 | 0.694667 | +2.800 pp | 0.691333 | True/False | [11]/[10] |
| gini_forest_parallel / 17 | accuracy | 0.707250 | 0.700750 | -0.650 pp | 0.730000 | True/True | [15, 15, 15, 16, 16, 15, 16, 15]/[15, 15, 15, 15, 15, 15, 15, 15] |
| gini_forest_parallel / 29 | accuracy | 0.711750 | 0.719500 | +0.775 pp | 0.730000 | True/True | [16, 15, 16, 15, 15, 15, 16, 15]/[15, 15, 15, 15, 15, 15, 15, 15] |
| gini_forest_parallel / 43 | accuracy | 0.730000 | 0.748750 | +1.875 pp | 0.730000 | False/False | [16, 15, 16, 15, 16, 15, 15, 16]/[15, 15, 15, 15, 15, 15, 15, 15] |
| gini_forest_serial / 17 | accuracy | 0.707250 | 0.700750 | -0.650 pp | 0.730000 | True/True | [15, 15, 15, 16, 16, 15, 16, 15]/[15, 15, 15, 15, 15, 15, 15, 15] |
| gini_forest_serial / 29 | accuracy | 0.711750 | 0.719500 | +0.775 pp | 0.730000 | True/True | [16, 15, 16, 15, 15, 15, 16, 15]/[15, 15, 15, 15, 15, 15, 15, 15] |
| gini_forest_serial / 43 | accuracy | 0.730000 | 0.748750 | +1.875 pp | 0.730000 | False/False | [16, 15, 16, 15, 16, 15, 15, 16]/[15, 15, 15, 15, 15, 15, 15, 15] |
| mae_cd_off / 17 | mae | 0.986915 | 0.888282 | -0.098633 (-9.99%) | 0.866791 | True/True | [11]/[11] |
| mae_cd_off / 29 | mae | 0.986915 | 0.888282 | -0.098633 (-9.99%) | 0.866791 | True/True | [11]/[11] |
| mae_cd_off / 43 | mae | 0.986915 | 0.888282 | -0.098633 (-9.99%) | 0.866791 | True/True | [11]/[11] |

Ratios are candidate / baseline medians. Investigate fit or prediction >1.25× and peak RSS >1.50×; confirm breaches with matched reruns beyond measured noise. Review changed tree sizes alongside costs.

| Workload | Fit ratio | Predict latency ratio | Peak RSS ratio | Review gates |
|---|---:|---:|---:|---|
| gini_binary_queue | 1.330× | 1.206× | 1.003× | fit_seconds |
| gini_default_tao | 1.669× | 1.122× | 1.002× | fit_seconds |
| gini_forest_parallel | 1.005× | 1.404× | 1.006× | predict_seconds_per_row |
| gini_forest_serial | 1.321× | 1.374× | 1.006× | fit_seconds, predict_seconds_per_row |
| mae_cd_off | 1.184× | 1.426× | 1.003× | predict_seconds_per_row |

Raw per-output quality, occupied leaves, depth, node counts, timing dispersion and provenance are preserved in raw.jsonl and comparison.json. Each matched pair uses the same manifest, dataset hash and seed; order alternates by round. Warmup and extra quality runs are excluded from timing medians. The runner documents memory scope in outer_growth.md.

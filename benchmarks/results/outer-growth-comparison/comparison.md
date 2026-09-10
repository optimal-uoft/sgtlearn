# Matched outer-growth comparison

Training quality is the primary soft signal. Changes below are candidate minus baseline: accuracy changes use percentage points (pp); regression losses show absolute and relative changes. Higher accuracy and lower loss are better. No numeric quality cutoff is imposed. Every worse-than-CART seed remains flagged.

| Workload / seed | Metric | Baseline | Candidate | Change | CART | Worse than CART (old/new) | Structural leaves (old/new) |
|---|---|---:|---:|---:|---:|---|---|
| entropy_multiway_pair_budget / 17 | accuracy | 0.687489 | 0.702714 | +1.523 pp | 0.713233 | True/True | [12]/[12] |
| entropy_multiway_pair_budget / 29 | accuracy | 0.687497 | 0.702714 | +1.522 pp | 0.713233 | True/True | [12]/[12] |
| entropy_multiway_pair_budget / 43 | accuracy | 0.689244 | 0.702714 | +1.347 pp | 0.713233 | True/True | [12]/[12] |
| gini_binary_queue / 17 | accuracy | 0.760167 | 0.765167 | +0.500 pp | 0.781167 | True/True | [31]/[31] |
| gini_binary_queue / 29 | accuracy | 0.760167 | 0.765167 | +0.500 pp | 0.781167 | True/True | [31]/[31] |
| gini_binary_queue / 43 | accuracy | 0.760167 | 0.765167 | +0.500 pp | 0.781167 | True/True | [31]/[31] |
| gini_categorical_missing_multioutput / 17 | accuracy | 0.611118 | 0.617539 | +0.642 pp | Excluded: grouped categorical/missing | excluded/excluded | [11]/[10] |
| gini_categorical_missing_multioutput / 29 | accuracy | 0.611118 | 0.617539 | +0.642 pp | Excluded: grouped categorical/missing | excluded/excluded | [11]/[10] |
| gini_categorical_missing_multioutput / 43 | accuracy | 0.611118 | 0.617539 | +0.642 pp | Excluded: grouped categorical/missing | excluded/excluded | [11]/[10] |
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
| mae_cd_on / 17 | mae | 0.888761 | 0.838960 | -0.049801 (-5.60%) | 0.866791 | True/False | [11]/[11] |
| mae_cd_on / 29 | mae | 0.888761 | 0.838960 | -0.049801 (-5.60%) | 0.866791 | True/False | [11]/[11] |
| mae_cd_on / 43 | mae | 0.888761 | 0.838960 | -0.049801 (-5.60%) | 0.866791 | True/False | [11]/[11] |
| mse_uncapped_multioutput / 17 | mse | 0.437620 | 0.438829 | +0.001208 (+0.28%) | 0.905711 | False/False | [100]/[102] |
| mse_uncapped_multioutput / 29 | mse | 0.436905 | 0.438765 | +0.001860 (+0.43%) | 0.905711 | False/False | [101]/[100] |
| mse_uncapped_multioutput / 43 | mse | 0.438753 | 0.438693 | -0.000060 (-0.01%) | 0.905711 | False/False | [100]/[99] |

Ratios are candidate / baseline medians. Investigate fit or prediction >1.25× and peak RSS >1.50×; confirm breaches with matched reruns beyond measured noise. Review changed tree sizes alongside costs.

| Workload | Fit ratio | Predict latency ratio | Peak RSS ratio | Review gates |
|---|---:|---:|---:|---|
| entropy_multiway_pair_budget | 1.019× | 1.134× | 1.001× | No breach |
| gini_binary_queue | 1.363× | 1.185× | 1.003× | fit_seconds |
| gini_categorical_missing_multioutput | 0.925× | 1.040× | 1.001× | No breach |
| gini_default_tao | 1.717× | 1.142× | 1.002× | fit_seconds |
| gini_forest_parallel | 1.007× | 1.381× | 1.007× | predict_seconds_per_row |
| gini_forest_serial | 1.357× | 1.363× | 1.005× | fit_seconds, predict_seconds_per_row |
| mae_cd_off | 1.172× | 1.426× | 1.001× | predict_seconds_per_row |
| mae_cd_on | 1.113× | 1.167× | 1.001× | No breach |
| mse_uncapped_multioutput | 1.014× | 0.988× | 1.000× | No breach |

Raw per-output quality, occupied leaves, depth, node counts, timing dispersion and provenance are preserved in raw.jsonl and comparison.json. Each matched pair uses the same manifest, dataset hash and seed; order alternates by round. Warmup and extra quality runs are excluded from timing medians. The runner documents memory scope in outer_growth.md.

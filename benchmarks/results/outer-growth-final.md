# Regularized outer growth: before/after/CART evidence

**Training quality: soft review passes. Performance: pending explicit user acceptance of confirmed timing exceptions; it has not passed sign-off.** Candidate `0d7d8f6d35859e4958564e65a3ac3592592e047f` is compared with pinned pre-change `90fb148e8b9b4181534dd2a19a5837aa85fba506`. No model was retuned to remove a regression.

Training quality is mostly improved. The material seed-specific decline is forest seed 17: **70.725% → 70.075%, −0.650 percentage points** (identical serial/parallel predictions), reproduced in the confirmation batch. Seeds 29 and 43 improve +0.775 and +1.875 pp. Uncapped multioutput MSE changes by −0.014% to +0.426%; these small losses remain much better than CART. MAE falls 9.99% with CD off and 5.60% with CD on. No large unexplained training-quality drop was found.

## Training quality first

Triples below are seeds **17 / 29 / 43**; repeated values are shown once with “all”. Accuracy is weighted where applicable, and multioutput accuracy is the mean of per-output accuracy rather than exact-match accuracy. Regression is the weighted output-averaged loss. Higher accuracy and lower loss are better. CART uses the frozen matching criterion/depth/leaf/sample limits; its binary axis-aligned model is a reference, not an equivalent model family. Every worse-than-CART flag is retained.

| Workload | Metric | Old | New | New − old | CART | Worse than CART old / new |
|---|---|---|---|---|---|---|
| entropy_multiway_pair_budget | accuracy | 68.749% / 68.750% / 68.924% | 70.271% (all) | +1.523 pp / +1.522 pp / +1.347 pp | 71.323% (all) | yes (all) / yes (all) |
| gini_binary_queue | accuracy | 76.017% (all) | 76.517% (all) | +0.500 pp (all) | 78.117% (all) | yes (all) / yes (all) |
| gini_categorical_missing_multioutput | accuracy | 61.112% (all) | 61.754% (all) | +0.642 pp (all) | Excluded: grouped/missing | N/A |
| gini_default_tao | accuracy | 66.667% (all) | 69.467% (all) | +2.800 pp (all) | 69.133% (all) | yes (all) / no (all) |
| gini_forest_parallel | accuracy | 70.725% / 71.175% / 73.000% | 70.075% / 71.950% / 74.875% | -0.650 pp / +0.775 pp / +1.875 pp | 73.000% (all) | old: yes / yes / no; new: yes / yes / no |
| gini_forest_serial | accuracy | 70.725% / 71.175% / 73.000% | 70.075% / 71.950% / 74.875% | -0.650 pp / +0.775 pp / +1.875 pp | 73.000% (all) | old: yes / yes / no; new: yes / yes / no |
| mae_cd_off | mae | 0.986915 (all) | 0.888282 (all) | -0.098633 (-9.994%) (all) | 0.866791 (all) | yes (all) / yes (all) |
| mae_cd_on | mae | 0.888761 (all) | 0.838960 (all) | -0.049801 (-5.603%) (all) | 0.866791 (all) | yes (all) / no (all) |
| mse_uncapped_multioutput | mse | 0.437620 / 0.436905 / 0.438753 | 0.438829 / 0.438765 / 0.438693 | +0.001208 (+0.276%) / +0.001860 (+0.426%) / -0.000060 (-0.014%) | 0.905711 (all) | no (all) / no (all) |

The candidate remains below CART on binary Gini, entropy, forest seeds 17/29, and MAE CD off; each gap also existed before. Default TAO now exceeds CART, and MAE CD on improves from worse than CART to better. Grouped-categorical/missing CART is explicitly excluded because matching its logical routing would require different preprocessing.

Per-output review finds small opposing changes hidden by the mean: categorical classification output 1 declines −0.239 pp while output 2 improves +1.523 pp. MSE output 2 at seed 29 increases +0.005974 (+2.000%), and output 1 at seed 43 increases +0.009683 (+1.695%); the other output improves in each case. Both MSE outputs remain better than CART. MAE improves on both outputs, but its second output remains worse than CART with CD both off and on even when the mean beats CART. These per-output changes do not reveal a large hidden degradation.

Per-output scores (accuracy fractions or criterion loss) are included below; means and individual seed flags also appear in the [complete matched report](outer-growth-comparison/comparison.md) and [structured results](outer-growth-comparison/comparison.json).

| Multioutput workload / seed | Old per-output scores | New per-output scores | CART per-output scores |
|---|---|---|---|
| gini_categorical_missing_multioutput / 17 | 0.479133, 0.743104 | 0.476741, 0.758337 | Excluded |
| gini_categorical_missing_multioutput / 29 | 0.479133, 0.743104 | 0.476741, 0.758337 | Excluded |
| gini_categorical_missing_multioutput / 43 | 0.479133, 0.743104 | 0.476741, 0.758337 | Excluded |
| mae_cd_off / 17 | 1.220211, 0.753619 | 1.192311, 0.584253 | 1.224657, 0.508925 |
| mae_cd_off / 29 | 1.220211, 0.753619 | 1.192311, 0.584253 | 1.224657, 0.508925 |
| mae_cd_off / 43 | 1.220211, 0.753619 | 1.192311, 0.584253 | 1.224657, 0.508925 |
| mae_cd_on / 17 | 1.160644, 0.616878 | 1.111396, 0.566525 | 1.224657, 0.508925 |
| mae_cd_on / 29 | 1.160644, 0.616878 | 1.111396, 0.566525 | 1.224657, 0.508925 |
| mae_cd_on / 43 | 1.160644, 0.616878 | 1.111396, 0.566525 | 1.224657, 0.508925 |
| mse_uncapped_multioutput / 17 | 0.571506, 0.303735 | 0.573294, 0.304363 | 1.124026, 0.687396 |
| mse_uncapped_multioutput / 29 | 0.575099, 0.298711 | 0.572846, 0.304685 | 1.124026, 0.687396 |
| mse_uncapped_multioutput / 43 | 0.571187, 0.306319 | 0.580870, 0.296516 | 1.124026, 0.687396 |

## Common zero-penalty performance

All shared penalties are zero; candidate branching_penalty is also zero. The main run contains **198 raw rows** (99 per revision): nine workloads, one warmup, five measured repetitions at seed 17, extra quality seeds 29/43, and CART references. Each old/new pair uses fresh subprocesses, alternates order, and verifies matching data hashes, dependency versions and thread settings. Startup/imports and quality scoring are outside fit timing. Predictions repeat for at least 0.25 seconds. No other builds or tests were active.

Values are **old → new median ± MAD**. Fit is milliseconds per fit, prediction is microseconds per row, and RSS is MiB for the sampled fit process tree. Ratios are new/old. Every workload is shown; no aggregate hides a slow path.

| Workload | Fit ms (± MAD) | Ratio | Predict µs/row (± MAD) | Ratio | RSS MiB (± MAD) | Ratio |
|---|---|---:|---|---:|---|---:|
| entropy_multiway_pair_budget | 75.084 ± 0.139 → 76.489 ± 0.083 | 1.019× | 0.0882 ± 0.0003 → 0.1001 ± 0.0006 | 1.134× | 189.02 ± 0.11 → 189.28 ± 0.30 | 1.001× |
| gini_binary_queue | 37.975 ± 0.180 → 51.767 ± 0.321 | 1.363× | 0.1289 ± 0.0008 → 0.1528 ± 0.0009 | 1.185× | 189.97 ± 0.30 → 190.52 ± 0.02 | 1.003× |
| gini_categorical_missing_multioutput | 16.749 ± 0.169 → 15.499 ± 0.063 | 0.925× | 0.1154 ± 0.0009 → 0.1201 ± 0.0018 | 1.040× | 189.16 ± 0.14 → 189.34 ± 0.06 | 1.001× |
| gini_default_tao | 17.856 ± 0.316 → 30.660 ± 0.745 | 1.717× | 0.1091 ± 0.0014 → 0.1246 ± 0.0033 | 1.142× | 188.27 ± 0.20 → 188.66 ± 0.31 | 1.002× |
| gini_forest_parallel | 91.788 ± 0.902 → 92.439 ± 1.014 | 1.007× | 0.4659 ± 0.0080 → 0.6433 ± 0.0033 | 1.381× | 212.95 ± 0.17 → 214.34 ± 0.14 | 1.007× |
| gini_forest_serial | 65.174 ± 1.162 → 88.463 ± 3.509 | 1.357× | 0.4747 ± 0.0056 → 0.6472 ± 0.0112 | 1.363× | 191.17 ± 0.08 → 192.09 ± 0.27 | 1.005× |
| mae_cd_off | 87.359 ± 0.244 → 102.364 ± 0.798 | 1.172× | 0.0637 ± 0.0004 → 0.0908 ± 0.0005 | 1.426× | 190.56 ± 0.11 → 190.72 ± 0.12 | 1.001× |
| mae_cd_on | 349.119 ± 0.862 → 388.574 ± 0.220 | 1.113× | 0.0589 ± 0.0006 → 0.0687 ± 0.0005 | 1.167× | 191.47 ± 0.23 → 191.61 ± 0.17 | 1.001× |
| mse_uncapped_multioutput | 73.995 ± 0.861 → 75.039 ± 0.676 | 1.014× | 0.1172 ± 0.0005 → 0.1158 ± 0.0010 | 0.988× | 189.66 ± 0.25 → 189.58 ± 0.03 | 1.000× |

Peak RSS increases remain below 1% in both full and confirmation batches, far below the 50% review threshold. RSS includes interpreter/import/data footprint; 5 ms sampling can miss brief allocations and does not isolate native candidate storage. Raw data also retains the OS lifetime high-water RSS through fit. This bounded suite is not evidence of asymptotic memory safety for arbitrarily large frontiers.

## Confirmed timing exceptions: acceptance required

All five workloads with a >25% fit/prediction breach were rerun in a fresh **110-row matched batch**, with the same warmup/five measured repetitions/quality seeds/CART protocol. The breaches persisted beyond their observed MAD. The following confirmed values are the quantified tradeoff awaiting user acceptance:

| Workload | Fit ms (± MAD) | Ratio | Predict µs/row (± MAD) | Ratio | RSS MiB (± MAD) | Ratio |
|---|---|---:|---|---:|---|---:|
| gini_binary_queue | 39.864 ± 0.720 → 53.009 ± 0.844 | 1.330× | 0.1303 ± 0.0012 → 0.1570 ± 0.0008 | 1.206× | 189.81 ± 0.08 → 190.31 ± 0.28 | 1.003× |
| gini_default_tao | 17.981 ± 0.112 → 30.012 ± 0.161 | 1.669× | 0.1089 ± 0.0015 → 0.1221 ± 0.0011 | 1.122× | 188.36 ± 0.19 → 188.77 ± 0.17 | 1.002× |
| gini_forest_parallel | 91.065 ± 6.297 → 91.551 ± 1.240 | 1.005× | 0.4606 ± 0.0068 → 0.6467 ± 0.0138 | 1.404× | 212.86 ± 0.09 → 214.14 ± 0.16 | 1.006× |
| gini_forest_serial | 66.428 ± 1.916 → 87.747 ± 0.803 | 1.321× | 0.4646 ± 0.0079 → 0.6385 ± 0.0098 | 1.374× | 191.42 ± 0.06 → 192.55 ± 0.19 | 1.006× |
| mae_cd_off | 89.057 ± 0.125 → 105.445 ± 0.430 | 1.184× | 0.0643 ± 0.0006 → 0.0916 ± 0.0004 | 1.426× | 190.58 ± 0.08 → 191.20 ± 0.31 | 1.003× |

Confirmed breaches: **binary Gini fit +33.0%; serial forest fit +32.1% and prediction +37.4%; parallel forest prediction +40.4%; MAE CD-off prediction +42.6%; default-TAO fit +66.9%.** The parallel forest fit itself has no confirmed breach. Absolute default-TAO fit cost rises by 12.03 ms on 1,500 samples; binary fit rises by 13.15 ms on 6,000 samples; serial eight-tree forest fit rises by 21.32 ms on 4,000 samples. [Confirmation raw data and dispersion](outer-growth-confirmation/comparison.json) retain all observations.

Non-timed public-fit/export diagnostics explain substantial additional topology work despite equal or smaller model sizes:

| Workload | Leaves old/new | Internal reached sample sum old/new | Ratio | Split-eligible node sample sum old/new | Ratio |
|---|---|---|---:|---|---:|
| Binary Gini | 31 / 31 | 21,590 / 30,202 | 1.399× | 26,929 / 36,166 | 1.343× |
| MAE CD off | 11 / 11 | 2,629 / 3,838 | 1.460× | 3,755 / 4,453 | 1.186× |
| Serial forest, eight trees | 123 / 120 | 61,183 / 94,405 | 1.543× | 92,220 / 125,204 | 1.358× |
| Default TAO, induction-only control | 11 / 10 | 2,424 / 3,581 | 1.477× | 3,885 / 5,081 | 1.308× |

Binary Gini has 30 internal nodes in both revisions. Maximum depth falls from 8 to 6, but mean training path rises from 3.598 to 5.034 nodes. After excluding the candidate’s final 131 eligible child rows that are not discovered at the exhausted cap, its discovery proxy is 36,035/26,929 = **1.338×**, closely matching 1.330× fit cost. Forest/MAE prediction work also rises because samples visit more internal nodes. Strict leaf caps remove baseline overshoot: e.g. forest seed 17 shrinks from 123 to 120 total leaves, and default TAO from 11 to 10.

These are measured topology counts, not an exclusive causal profile. “Split-eligible” means exported depth below max_depth, at least twice min_samples_leaf rows, and positive impurity. It cannot reconstruct failed searches, capped rediscovery, inner-router effort, allocation/copy costs, or TAO iterations. Forest paths count bootstrap fitting samples rather than the prediction matrix. The default-TAO objective/algorithm remains unchanged, but its input topology changes; the 66.9% full-fit increase is not fully attributed by export counts. No evidence justified speculative copying/routing rewrites or changing required search behavior to improve the benchmark. [Reproducible diagnostic and complete exports](outer-growth-work-diagnostic/README.md) document these limits.

## Separate branching-penalty experiments

Two additional matched runs contain **22 rows each**. Their baseline stays unregularized; the new model changes only branching_penalty. These are intentional regularization/model-size comparisons, excluded from the common zero-cost conclusion.

| Workload | Candidate lambda | Candidate zero → positive quality (seeds 17/29/43) | Candidate zero → positive leaves | Fit/predict/RSS ratio vs old zero |
|---|---:|---|---|---|
| entropy_multiway_pair_budget | 25 | 0.702714 → 0.702714; 0.702714 → 0.702714; 0.702714 → 0.702714 | 12 → 12; 12 → 12; 12 → 12 | 1.080×/1.159×/1.003× |
| mse_uncapped_multioutput | 10 | 0.438829 → 0.445438; 0.438765 → 0.445832; 0.438693 → 0.443913 | 102 → 87; 100 → 85; 99 → 86 | 1.011×/1.004×/1.000× |

Entropy lambda25 does not change the selected model on these data. MSE lambda10 removes 13–15 leaves and increases MSE by 1.19–1.61% relative to candidate lambda0; it remains substantially better than CART. All separate-experiment performance ratios pass the review thresholds. Their full old/new/CART scores and individual seed flags are in [entropy results](outer-growth-branching-entropy/comparison.md) and [MSE results](outer-growth-branching-mse/comparison.md).

Positive penalty units deliberately changed. Outer gain is total sample-weight mass times the mean per-output impurity reduction; min_impurity_decrease is a fixed total-loss split cost, branching_penalty charges each child beyond two, and pairwise_penalty is charged once for a pair. Identical positive numbers in old and new versions therefore do **not** imply equal regularization. No such equivalence is asserted by these measurements. Correctness evidence is supplied by `test_sample_mass_scales_gain_but_not_growth_cost` (four criteria: multiplying weight mass changes admission at a fixed cost), `test_outer_scores_average_outputs_and_preserve_uniform_output_replication` (output averaging), and `test_pair_cost_is_constant_total_loss_and_charged_once` in [the final revision’s tests](../../tests/test_outer_regularized_growth.py). Those tests were included in the passing final suite; no extra timing-sensitive CI assertions were added.

## Reproduction, provenance and scope

Hardware: Apple M5 Pro, 15 physical/logical cores, 24 GiB RAM; macOS26.6.2. Both isolated noneditable builds use Python3.14.5, the same 28 pinned dependencies, Apple Clang21, CMake4.4.2, Release `-O3 -DNDEBUG -std=gnu++20 -arch arm64`, and one native BLAS/OpenMP thread. Forest n_jobs is fixed at1/2 independently. Candidate tests are enabled in the same build; all three native modules and Python package paths/hashes were verified inside their own environments. The candidate full validation passed once before timing: 471 Python tests passed, 2 skipped, 31 expected MAE warnings; CTest44/44 passed.

- [Frozen workload and measurement protocol](../outer_growth.md), [manifest](../outer_growth_manifest.json), [shared worker](../outer_growth.py), [matched driver](../compare_outer_growth.py).
- [Original baseline capture](outer-growth-baseline/summary.md) remains unchanged as an audit artifact.
- [Candidate build flags/hardware/dependency source provenance](outer-growth-candidate-build/build-provenance.txt), [dependency pins](outer-growth-candidate-build/dependencies.txt), [verified import paths/hashes](outer-growth-candidate-build/imports.json).
- [Common run settings and revisions](outer-growth-comparison/settings.json), [198 common raw rows](outer-growth-comparison/raw.jsonl), [110 confirmation rows](outer-growth-confirmation/raw.jsonl), [22 entropy rows](outer-growth-branching-entropy/raw.jsonl), [22 MSE rows](outer-growth-branching-mse/raw.jsonl).

From the repository root, reproduce the common comparison:

```sh
.venv/bin/python benchmarks/compare_outer_growth.py \
  --baseline-python /tmp/sgtlearn-baseline-90fb148/.venv/bin/python \
  --baseline-checkout /tmp/sgtlearn-baseline-90fb148 \
  --candidate-python /tmp/sgtlearn-candidate-env/bin/python \
  --candidate-checkout /tmp/sgtlearn-candidate \
  --output benchmarks/results/outer-growth-comparison-reproduced
```

Use a fresh output directory for new measurements. Confirmations add `--workload ID` for each of the five rows above; positive experiments add `--workload entropy_multiway_pair_budget --candidate-branching-penalty 25` or `--workload mse_uncapped_multioutput --candidate-branching-penalty 10`. Preserve the frozen manifest, runner, environments and final source revision; run without competing builds/tests. Each result directory records its exact CLI settings. The [candidate build protocol](outer-growth-candidate-build/README.md) records reconstruction of the tested wheel.

This is a practical nine-workload synthetic suite covering all four criteria, weighted/multioutput/missing/categorical cases, binary/high-arity capped/finite-depth uncapped growth, pairs, MAE CD off/on, forests, and default TAO. It does not measure held-out generalization, long-running production data, exhaustive frontier/arity scaling, or exact native allocation peaks. Model nodes/depth/structural and occupied leaves are preserved for every quality seed in the structured results. Training-quality/CART flags remain soft review signals. Confirmed timing gates remain open until the user explicitly accepts the quantified tradeoff or remediation resolves them.

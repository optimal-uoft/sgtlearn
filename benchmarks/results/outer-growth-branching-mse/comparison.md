# Matched outer-growth comparison

Training quality is the primary soft signal. Changes below are candidate minus baseline: accuracy changes use percentage points (pp); regression losses show absolute and relative changes. Higher accuracy and lower loss are better. No numeric quality cutoff is imposed. Every worse-than-CART seed remains flagged.

Separate regularization experiment: candidate branching_penalty=10.0; baseline has no public branching penalty. This is not the common zero-penalty comparison.

| Workload / seed | Metric | Baseline | Candidate | Change | CART | Worse than CART (old/new) | Structural leaves (old/new) |
|---|---|---:|---:|---:|---:|---|---|
| mse_uncapped_multioutput / 17 | mse | 0.437620 | 0.445438 | +0.007817 (+1.79%) | 0.905711 | False/False | [100]/[87] |
| mse_uncapped_multioutput / 29 | mse | 0.436905 | 0.445832 | +0.008927 (+2.04%) | 0.905711 | False/False | [101]/[85] |
| mse_uncapped_multioutput / 43 | mse | 0.438753 | 0.443913 | +0.005160 (+1.18%) | 0.905711 | False/False | [100]/[86] |

Ratios are candidate / baseline medians. Investigate fit or prediction >1.25× and peak RSS >1.50×; confirm breaches with matched reruns beyond measured noise. Review changed tree sizes alongside costs.

| Workload | Fit ratio | Predict latency ratio | Peak RSS ratio | Review gates |
|---|---:|---:|---:|---|
| mse_uncapped_multioutput | 1.011× | 1.004× | 1.000× | No breach |

Raw per-output quality, occupied leaves, depth, node counts, timing dispersion and provenance are preserved in raw.jsonl and comparison.json. Each matched pair uses the same manifest, dataset hash and seed; order alternates by round. Warmup and extra quality runs are excluded from timing medians. The runner documents memory scope in outer_growth.md.

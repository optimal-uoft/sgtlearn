# Matched outer-growth comparison

Training quality is the primary soft signal. Changes below are candidate minus baseline: accuracy changes use percentage points (pp); regression losses show absolute and relative changes. Higher accuracy and lower loss are better. No numeric quality cutoff is imposed. Every worse-than-CART seed remains flagged.

Separate regularization experiment: candidate branching_penalty=25.0; baseline has no public branching penalty. This is not the common zero-penalty comparison.

| Workload / seed | Metric | Baseline | Candidate | Change | CART | Worse than CART (old/new) | Structural leaves (old/new) |
|---|---|---:|---:|---:|---:|---|---|
| entropy_multiway_pair_budget / 17 | accuracy | 0.687489 | 0.702714 | +1.523 pp | 0.713233 | True/True | [12]/[12] |
| entropy_multiway_pair_budget / 29 | accuracy | 0.687497 | 0.702714 | +1.522 pp | 0.713233 | True/True | [12]/[12] |
| entropy_multiway_pair_budget / 43 | accuracy | 0.689244 | 0.702714 | +1.347 pp | 0.713233 | True/True | [12]/[12] |

Ratios are candidate / baseline medians. Investigate fit or prediction >1.25× and peak RSS >1.50×; confirm breaches with matched reruns beyond measured noise. Review changed tree sizes alongside costs.

| Workload | Fit ratio | Predict latency ratio | Peak RSS ratio | Review gates |
|---|---:|---:|---:|---|
| entropy_multiway_pair_budget | 1.080× | 1.159× | 1.003× | No breach |

Raw per-output quality, occupied leaves, depth, node counts, timing dispersion and provenance are preserved in raw.jsonl and comparison.json. Each matched pair uses the same manifest, dataset hash and seed; order alternates by round. Warmup and extra quality runs are excluded from timing medians. The runner documents memory scope in outer_growth.md.

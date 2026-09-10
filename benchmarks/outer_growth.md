# Outer growth baseline and comparison

Issue [#55](https://github.com/optimal-uoft/sgtlearn/issues/55), baseline commit
`90fb148e8b9b4181534dd2a19a5837aa85fba506`. Run the same manifest and runner
against separately built baseline and candidate environments. The runner rejects
SGT Python/native imports outside its virtual environment and records their SHA256
hashes, source revision, resolved estimator parameters and data hashes.

```sh
git worktree add --detach /tmp/sgtlearn-baseline-90fb148 90fb148e8b9b4181534dd2a19a5837aa85fba506
uv venv /tmp/sgtlearn-baseline-90fb148/.venv --python 3.14.5
uv pip install --python /tmp/sgtlearn-baseline-90fb148/.venv/bin/python -r benchmarks/results/outer-growth-baseline/dependencies.txt
CMAKE_BUILD_PARALLEL_LEVEL=4 uv pip install --python /tmp/sgtlearn-baseline-90fb148/.venv/bin/python --no-build-isolation --no-deps /tmp/sgtlearn-baseline-90fb148 --config-settings build-dir=/tmp/sgtlearn-baseline-90fb148/build
/tmp/sgtlearn-baseline-90fb148/.venv/bin/python benchmarks/check_outer_growth.py
/tmp/sgtlearn-baseline-90fb148/.venv/bin/python benchmarks/outer_growth.py --checkout /tmp/sgtlearn-baseline-90fb148 --output benchmarks/results/outer-growth-baseline
```

Use an absolute runner path when invoking from another directory. An interrupted
run resumes completed rows; use a new output directory for matched reruns or a
different build/configuration. `--workload ID` selects a workload. To compare new
branching regularization separately, use `--branching-penalty VALUE` with a new
output directory. The baseline does not support that public parameter. Equal
positive old/new `min_impurity_decrease` or `pairwise_penalty` values do **not**
represent equal regularization units. The frozen common suite uses zero penalties.

The nine bounded synthetic workloads cover Gini, entropy, MSE, MAE; binary and
multiway; capped and finite-depth uncapped growth; pairs; weights; multiple
outputs; missing/grouped categorical data; serial and two-thread forests; MAE CD
off/on; and induction only versus the unchanged automatic default TAO (10 runs).
The queue workload has depth 8 and 31 leaves; the four-way/12-leaf pair workload
forces repeated feasible-arity reductions near its cap. These characterize
behavior; public exports cannot establish private queue sizes. The duplicate-X
MAE case reuses the stress pattern in `tests/test_mae_regression_stress.py`.
The existing native `cpp/tests/bench_mae_branch_assignment.cpp` remains the
standalone branch-optimizer diagnostic; no duplicate microbenchmark is added.

Each workload has one fresh-process warmup at timing seed 17, then five
fresh-process measured repetitions. Extra warmup/quality processes use seeds 29
and 43. Raw rows retain all measurements. Startup, imports, dataset creation,
quality scoring, exports and warning collection are outside fit timing. Multiple
fits per process amortize timers where needed; their durations remain in raw
data. Predictions repeat for at least 0.25 seconds on the full training matrix.
Warnings are suppressed during timing and captured by a separate fit in quality
processes. Run while no other builds/tests are active. The OS, compiler, build
flags and exact package versions are preserved alongside results.

Peak fit memory is the maximum RSS sampled every 5 ms over the process and its
descendants, including worker memory. Existing forests use joblib threads, so
workers share the process RSS. Summing RSS can double count shared pages in a
future subprocess backend. The separate OS lifetime high-water RSS ends just
after timed fits and includes interpreter/import/data overhead. Sampling can
miss brief allocations; these figures are process footprint, not isolated native
allocation counts. Native BLAS/OpenMP threads are fixed at one; forest n_jobs
is fixed independently. Medians and median absolute deviations (MAD) are reported
with min/max; gates are review thresholds, never ordinary CI timing assertions.

Classification training accuracy is sample-weighted where weights are supplied;
multioutput results are per-output accuracy plus their arithmetic mean (not
subset/exact-match accuracy). Regression reports criterion-appropriate weighted
MSE or MAE per output and their mean. Each quality seed has a scikit-learn CART
reference on the same data/weights, with the same criterion, outer maximum depth,
maximum leaves and minimum leaf samples. CART remains binary with axis-aligned
thresholds; SGT can route noncontiguous bins, group categories, use pairs, multiple
children, TAO, and forests. Thus constraints are comparable rather than model
families equivalent. CART forests are deliberately not substituted for CART.
Grouped-categorical/missing comparison is explicitly excluded because preserving
SGT's logical category/missing routing needs different preprocessing; no silent
imputation changes the task. Exported structural/occupied leaves, depth and node
counts accompany all quality measurements (per constituent for forests).

Investigate >25% median fit/prediction slowdown or >50% peak-memory increase,
confirming any breach beyond measured noise with repeated matched runs. Interpret
performance alongside learned tree size. Training quality is the primary soft
gate: similar or improved accuracy is expected, slight degradation is acceptable,
and large reproducible drops need investigation without an invented numeric
cutoff. Lower regression loss is better. Every worse-than-CART quality result is
flagged without automatically failing the change. Default TAO and future positive
regularization comparisons stay separate from induction-only zero-cost results.

For the final common-configuration comparison, build the candidate in another
isolated checkout/venv with the same dependency pins and Release settings, then
interleave matching fresh-process runs:

```sh
.venv/bin/python benchmarks/compare_outer_growth.py \
  --baseline-python /tmp/sgtlearn-baseline-90fb148/.venv/bin/python \
  --baseline-checkout /tmp/sgtlearn-baseline-90fb148 \
  --candidate-python /tmp/sgtlearn-candidate/.venv/bin/python \
  --candidate-checkout /tmp/sgtlearn-candidate \
  --output benchmarks/results/outer-growth-comparison
```

The driver alternates old/new order each round, verifies matching data hashes,
Python/NumPy/sklearn versions, environment and native thread pools, and writes
`raw.jsonl`, `comparison.json`, `comparison.md` and settings with revision/runner
hashes. The Markdown report lists performance review gates and every quality seed
with old/new/CART scores and leaf counts. Use `--workload ID` and a fresh output
directory to confirm a suspected breach. Optional
`--candidate-branching-penalty VALUE` produces a clearly labeled separate
regularization experiment; use another output directory. Do not change/rebuild
either environment during a run. Resume support skips completed rows; if a run
is interrupted mid-pair, use a fresh directory for strict temporal matching.

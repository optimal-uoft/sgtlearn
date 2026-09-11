# Regularized outer-growth verification

Code revision: `0d7d8f6d35859e4958564e65a3ac3592592e047f`.
Reference revision: `90fb148e8b9b4181534dd2a19a5837aa85fba506`.
Specification: [#54](https://github.com/optimal-uoft/sgtlearn/issues/54), associated with [#52](https://github.com/optimal-uoft/sgtlearn/issues/52).

## Isolated checks

The final package and all three native extensions were rebuilt together in a
fresh Release build at `/tmp/sgtlearn-candidate/build`, with native tests enabled.
The detached checkout is `/tmp/sgtlearn-candidate`; its environment is
`/tmp/sgtlearn-candidate-env`. The 28 dependency pins and core compiler flags match
the frozen baseline. Only verified dependency source trees were reused, not
compiled libraries. See [build provenance](outer-growth-candidate-build/build-provenance.txt)
and [module paths/hashes](outer-growth-candidate-build/imports.json).

| Check | Command | Result |
|---|---|---|
| Full Python suite, run once from isolated checkout | `/tmp/sgtlearn-candidate-env/bin/python -m pytest -q` | **471 passed, 2 skipped**, 12.98 s |
| Full configured native suite | `ctest --test-dir /tmp/sgtlearn-candidate/build --output-on-failure` | **44/44 passed**, 0.10 s |
| Typecheck | `.venv/bin/mypy sgtlearn --ignore-missing-imports` | Passed, 12 source files |
| Python lint | `.venv/bin/ruff check sgtlearn` | Passed |
| Python formatting | `.venv/bin/black --check --target-version py314 sgtlearn` | Passed, 12 source files |
| Documentation | `.venv/bin/sphinx-build -W --keep-going -b html docs /tmp/sgt52-docs` | Passed |
| Benchmark scoring/report checks | `.venv/bin/python benchmarks/check_outer_growth.py` | Passed |

The two skips are existing Python 3.14+ exclusions for scikit-learn MAE
best-first reference comparisons in the categorical and univariate discretizer
tests. No new skips were introduced. All 31 reported warnings are the requested
MAE CD warning from existing fit calls; dedicated tests separately verify its
aliases, exact enablement values, and one-warning forest behavior. There were no
test failures requiring baseline comparison or unexplained failures.

## Specification coverage

Paths below are relative to the repository root. The full suites include both
new acceptance checks and existing regression checks.

| Requirement | Evidence | Result |
|---|---|---|
| Weighted best-first priority, including unlimited growth | `tests/test_outer_regularized_growth.py`: competing nodes with opposite local/total gain ordering; classifier/regressor and nonuniform weights | Pass |
| Constant alpha/lambda/gamma costs, raw gain separate from priority | Public mass-scaling/pair-cost cases and native branch-cost cases | Pass |
| Fixed finite score strictly above double epsilon | Native below/equal/above-epsilon and nonfinite cases in `cpp/tests/test_shape_assignment_search.cpp` | Pass |
| Mean impurity across outputs for all four criteria | Public independently computed weighted impurities and uniform output replication | Pass |
| Structural caps, later budget reduction, different-router fallback and heap reprioritization | Eight cases in `tests/test_outer_leaf_budget.py`, including independent exact-SSE regression fixtures | Pass |
| Actual occupied arity and coherent retained metadata | Native independently replayable per-arity snapshots, empty/missing-bin compaction, zero-weight occupancy | Pass |
| Preserve observed/rejected CD states and raw CD moves | Existing native observed-state tests plus `tests/test_coordinate_descent_initialization.py`; shared CD code unchanged | Pass |
| Raw-positive pair admission and raw proxy independent of costs | `tests/test_pair_screening.py` and native raw-ternary/regularized-binary proxy case | Pass |
| Final recomputed raw eligibility | New six-bin native regression; independent original reproduction and 10,000 probe seeds | Pass |
| Exact ties and stable selection | Native fewer-child tie; existing public univariate-before-pair and logical-feature pair-ranking ties | Pass |
| Public parameter cloning, validation and forest forwarding | `tests/test_branching_penalty.py`, `tests/test_outer_penalty_validation.py`, existing forest cases | Pass |
| Exactly one top-level MAE warning | `tests/test_mae_warning.py`, standalone and serial/parallel forest fitting | Pass |
| Weighted multioutput/missing/pair/budget interaction | `tests/test_outer_growth_integration.py`, all four criteria and MAE CD off/on; exported routing replay and leaf statistics | Pass |
| Inner CART and TAO boundaries | Existing discretizer, fidelity, weighted, multioutput, pairwise, missing-routing and TAO suites | Pass |

Export checks respect existing conventions: classification `n_samples` is
rounded sample mass, and univariate exported bin statistics omit the separately
routed NaN bin. Weighted histograms and replayed predictions check complete
sample routing without changing those interfaces.

## Intentional expectation changes

Old positive normalized penalties in existing fixtures were converted to the
new total-loss units: classifier pair penalty `0.5 → 42.5` for 85 samples;
TAO-starting-tree classifier penalty `1 → 85`; and its two-output regression
counterpart `1 → 42.5`. Native branching fixtures were similarly rescaled by
sample mass. Their original topology/integration assertions remain intact.
These fixture conversions are not a claim that arbitrary old/new positive
penalty settings are globally equivalent.

## Standards

No hard documented-standard violations. Independent Ponytail reviews were run
between implementation tickets. The final review identified one optional
simplification: remove the legacy single `best` result copy from the shared
search result, whose production consumers now use `rawBest` and `byArity`.
It remains part of the existing native selection-test contract; it is not a
correctness failure or an added abstraction.

## Spec

No missing core requirements, incorrect implementations, or scope creep found
in the final independent review. One earlier numerical eligibility finding was
fixed and rechecked before the isolated build. Shared inner criterion/CD code
and the separate TAO objective/defaults remain unchanged.

Review totals: Standards 0 hard violations and 1 optional simplification;
Spec 0 unresolved findings.

## Training quality and performance

Matched old/new/CART measurements on this same code revision are complete in
the [quality-first comparison report](outer-growth-final.md). Correctness and
the soft training-quality review pass: most workloads improve, the largest
classification decline is 0.650 percentage points on one forest seed, and no
large unexplained per-output degradation was found. All worse-than-CART cases
remain visible.

The 25% runtime threshold was breached on five workloads and confirmed in a
fresh matched batch. Fit slowdowns are 32.1–66.9% on three workloads; prediction
slowdowns are 37.4–42.6% on three workloads, with overlap. Diagnostics explain
substantial additional sample visits in the learned trees, without establishing
exact operation counts. Peak-memory increases stay below 1%, within the 50%
threshold. Performance sign-off under #61 remains pending explicit user
acceptance of the quantified timing tradeoff or successful remediation.

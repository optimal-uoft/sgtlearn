# Exported topology work diagnostic

Non-timed public fits, seed 17, baseline `90fb148e8b9b4181534dd2a19a5837aa85fba506` and candidate `0d7d8f6d35859e4958564e65a3ac3592592e047f`, using the same isolated packages, manifest and datasets as the matched comparison. Run only after all timing stopped. Raw JSON includes complete reachable node exports and dataset hashes.

| Workload | Internal reached sample sum old/new | Ratio | Eligible node sample sum old/new | Ratio | Structural leaves old/new |
|---|---:|---:|---:|---:|---:|
| Binary Gini | 21,590 / 30,202 | 1.399 | 26,929 / 36,166 | 1.343 | 31 / 31 |
| MAE CD off | 2,629 / 3,838 | 1.460 | 3,755 / 4,453 | 1.186 | 11 / 11 |
| Serial forest (sum of 8 trees) | 61,183 / 94,405 | 1.543 | 92,220 / 125,204 | 1.358 | 123 / 120 |
| Default TAO, exported after TAO | 2,424 / 3,604 | 1.487 | 3,871 / 5,087 | 1.314 | 11 / 10 |
| Default TAO, induction-only control | 2,424 / 3,581 | 1.477 | 3,885 / 5,081 | 1.308 | 11 / 10 |

The binary tree has the same 30 internal nodes and 31 leaves, but the mean training sample path increases from 3.598 to 5.034 internal nodes while maximum depth falls from 8 to 6. Its larger discovered sample workload closely tracks the confirmed 1.330× fit ratio. The other models also route more samples through internal nodes despite equal or smaller leaf counts. This provides evidence for topology-dependent work; these counts do not establish the exclusive cause or measure allocation/copy overhead.

“Eligible” is an exported-topology proxy: depth below max_depth, at least twice min_samples_leaf samples, and positive impurity. It cannot reconstruct failed searches, repeated capped rediscovery, exact inner-router work, or TAO trajectories. The final children are not discovered by the candidate when the leaf cap is exhausted; subtracting their 131 eligible samples for binary Gini gives 36,035, or 1.338× the baseline 26,929. Classification diagnostic cases are unweighted, so exported n_samples is an actual count. Forest counts refer to bootstrap training samples, not paths of the whole prediction matrix. MAE is weighted, and path sums count rows rather than weight mass. Post-TAO exports do not reconstruct induction work; a separate tao_n_runs=0 control is provided.

Reproduce each JSON from the repository root with the matching isolated Python, replacing PYTHON and OUTPUT:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHON benchmarks/results/outer-growth-work-diagnostic/diagnostic.py benchmarks/outer_growth_manifest.json gini_binary_queue gini_forest_serial mae_cd_off gini_default_tao > OUTPUT
```

Python paths: `/tmp/sgtlearn-baseline-90fb148/.venv/bin/python` and `/tmp/sgtlearn-candidate-env/bin/python`. The captured script differs from the executed scratch copy only in locating the shared runner relative to itself instead of by absolute path. Build/import provenance is in `../outer-growth-candidate-build` and each matched raw row.

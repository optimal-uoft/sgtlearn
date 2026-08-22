# MAE / shape-tree fit benchmarks
#
# Run from the repo root after building the C++ extensions:
#
#   cmake --build build -j
#   # optional: copy *.so into the active venv site-packages
#   MPLCONFIGDIR=/tmp/mpl PYTHONPATH=build python benchmarks/bench_sgt_mae_fit.py
#
# The script force-loads ``build/*.so`` when present so a stale venv copy is not used.
#
# Environment knobs used by the native AbsoluteError path:
#   SGTLEARN_MAE_CD=1              enable MAE coordinate descent during fit
#   SGTLEARN_MAE_BACKEND=sort|bst  branch-assignment implementation
#
# The bench sets ``tao_n_runs=0`` so timings isolate tree growth / CD (not TAO).
#
# Results land in benchmarks/results/.

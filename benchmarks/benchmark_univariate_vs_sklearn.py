#!/usr/bin/env python3
"""Benchmark UnivariateClassificationDiscretizer against sklearn DecisionTreeClassifier.

Usage:
  python benchmarks/benchmark_univariate_vs_sklearn.py
  python benchmarks/benchmark_univariate_vs_sklearn.py --sizes 10000 50000 --repeats 7
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from Discretizers import UnivariateClassificationDiscretizer
from sklearn.tree import DecisionTreeClassifier


def timed_seconds(fn) -> float:
    """Wall time in seconds to execute a no-argument callable once."""
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


@dataclass
class BenchResult:
    """Median sklearn vs native fit duration (ms) for one synthetic benchmark setting."""

    n_samples: int
    min_leaf: int
    max_depth: int
    sklearn_fit_ms: float
    sgt_fit_ms: float


def run_one(
    *,
    n_samples: int,
    min_leaf: int,
    max_depth: int,
    repeats: int,
    warmup: int,
    seed: int,
) -> BenchResult:
    """Compare fit time of ``DecisionTreeClassifier`` vs ``UnivariateClassificationDiscretizer`` on random 1-D data."""
    num_classes = 2
    rng = np.random.default_rng(seed)
    x64 = rng.random((n_samples, 1), dtype=np.float64)
    x32 = x64.astype(np.float32, copy=False)
    labels = rng.integers(0, num_classes, size=n_samples, dtype=np.uintp)
    features = np.array([0], dtype=np.uintp)

    def make_sklearn() -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            criterion="gini",
            splitter="best",
            min_samples_leaf=min_leaf,
            min_impurity_decrease=0.0,
            max_depth=None if max_depth == 0 else max_depth,
            max_leaf_nodes=None,
            random_state=0,
        )

    def make_sgt() -> UnivariateClassificationDiscretizer:
        return UnivariateClassificationDiscretizer()

    for _ in range(warmup):
        sk = make_sklearn()
        sk.fit(x32, labels)
        ud = make_sgt()
        ud.Train(x32, features, labels, num_classes, min_leaf, 0.0, max_depth, 0)

    sk_fit_runs: list[float] = []
    ud_fit_runs: list[float] = []

    for _ in range(repeats):
        sk = make_sklearn()
        ud = make_sgt()
        sk_fit_runs.append(timed_seconds(lambda: sk.fit(x32, labels)))
        ud_fit_runs.append(
            timed_seconds(lambda: ud.Train(x32, features, labels, num_classes, min_leaf, 0.0, max_depth, 0))
        )

    return BenchResult(
        n_samples=n_samples,
        min_leaf=min_leaf,
        max_depth=max_depth,
        sklearn_fit_ms=statistics.median(sk_fit_runs) * 1e3,
        sgt_fit_ms=statistics.median(ud_fit_runs) * 1e3,
    )


def run_case(case: tuple[int, int, int, int, int, int]) -> BenchResult:
    """Run ``run_one`` for a packed tuple ``(n_samples, min_leaf, max_depth, repeats, warmup, seed)``."""
    n, leaf, depth, repeats, warmup, seed = case
    return run_one(
        n_samples=n,
        min_leaf=leaf,
        max_depth=depth,
        repeats=repeats,
        warmup=warmup,
        seed=seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1_000, 10_000, 50_000])
    parser.add_argument("--min-leaf", nargs="+", type=int, default=[1, 10])
    parser.add_argument("--max-depth", nargs="+", type=int, default=[0])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Number of worker processes for parallel trial execution.",
    )
    parser.add_argument("--show-detail", action="store_true", help="Also print per-configuration fit timings.")
    return parser.parse_args()


def print_average_fit_by_size(results: list[BenchResult]) -> None:
    """Print mean fit milliseconds per implementation, grouped by ``n_samples``."""
    sk_by_n: dict[int, list[float]] = defaultdict(list)
    sgt_by_n: dict[int, list[float]] = defaultdict(list)

    for r in results:
        sk_by_n[r.n_samples].append(r.sklearn_fit_ms)
        sgt_by_n[r.n_samples].append(r.sgt_fit_ms)

    print("n_samples | avg_sk_fit_ms avg_sgt_fit_ms faster_impl fit_x")
    print("----------------------------------------------------------")
    for n in sorted(sk_by_n):
        avg_sk = statistics.mean(sk_by_n[n])
        avg_sgt = statistics.mean(sgt_by_n[n])
        fit_x = avg_sk / avg_sgt if avg_sgt > 0 else float("inf")
        faster_impl = "sgtlearn" if avg_sgt < avg_sk else "sklearn"
        print(f"{n:8d} | {avg_sk:13.3f} {avg_sgt:14.3f} {faster_impl:11s} {fit_x:5.2f}")


def main() -> None:
    """Execute the full Cartesian grid of benchmarks and print summaries."""
    args = parse_args()
    cases: list[tuple[int, int, int, int, int, int]] = []

    for idx, (n, leaf, depth) in enumerate(
        (n, leaf, depth) for n in args.sizes for leaf in args.min_leaf for depth in args.max_depth
    ):
        cases.append((n, leaf, depth, args.repeats, args.warmup, args.seed + idx))

    if args.jobs == 1:
        results = [run_case(case) for case in cases]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            results = list(executor.map(run_case, cases))

    print_average_fit_by_size(results)

    if args.show_detail:
        print()
        header = "n_samples leaf depth | sk_fit_ms sgt_fit_ms fit_x"
        print(header)
        print("-" * len(header))
        for r in results:
            fit_x = r.sklearn_fit_ms / r.sgt_fit_ms if r.sgt_fit_ms > 0 else float("inf")
            print(
                f"{r.n_samples:8d} {r.min_leaf:4d} {r.max_depth:5d} | "
                f"{r.sklearn_fit_ms:9.3f} {r.sgt_fit_ms:10.3f} {fit_x:5.2f}"
            )


if __name__ == "__main__":
    main()

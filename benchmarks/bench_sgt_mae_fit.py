#!/usr/bin/env python3
"""Benchmark SGTRegressor(absolute_error) fit: sort vs BST branch assignment.

Requires the native extension built with MAE CD / backend env hooks.
Coordinate descent for MAE is enabled via ``SGTLEARN_MAE_CD=1`` so the two
backends are exercised during ``fit`` (default product path still skips MAE CD).
``tao_n_runs=0`` so timings reflect tree growth / branch assignment, not TAO.

Usage (from repo root, with build/ on PYTHONPATH or an editable install)::

    PYTHONPATH=build python benchmarks/bench_sgt_mae_fit.py

Writes::

    benchmarks/results/sgt_mae_fit_<timestamp>.json   # raw runs
    benchmarks/results/sgt_mae_fit_<timestamp>.csv    # per-case rows
    benchmarks/results/sgt_mae_fit_latest.json        # copy of newest
    benchmarks/results/sgt_mae_fit_summary.json       # aggregates
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class Case:
    name: str
    n_samples: int
    n_features: int
    max_depth: int
    inner_max_depth: int
    num_partitions: int
    repeats: int


CASES: list[Case] = [
    # Deeper inner trees + higher fan-out → more bins and CD moves (where BST wins).
    Case("small_2k_x_8", 2_000, 8, 3, 3, 4, 3),
    Case("medium_5k_x_12", 5_000, 12, 4, 4, 4, 3),
    Case("large_8k_x_16", 8_000, 16, 4, 4, 6, 2),
]


def _make_data(n_samples: int, n_features: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float64)
    # Nonlinear + heavy tails so absolute_error is a reasonable criterion.
    y = (
        np.sin(X[:, 0])
        + 0.5 * X[:, 1] ** 2
        + 0.1 * rng.standard_t(df=3, size=n_samples)
    ).astype(np.float64)
    return X, y


def _safe_n_leaves(model: Any) -> int | None:
    est = getattr(model, "_est", None)
    if est is None:
        return None
    if hasattr(est, "num_leaves"):
        try:
            return int(est.num_leaves)
        except Exception:
            return None
    return None


def run_backend(
    backend: str,
    cases: list[Case],
    *,
    seed: int,
    warmup: bool,
) -> list[dict[str, Any]]:
    from sgtlearn import SGTRegressor

    rows: list[dict[str, Any]] = []
    os.environ["SGTLEARN_MAE_CD"] = "1"
    os.environ["SGTLEARN_MAE_BACKEND"] = backend

    if warmup:
        Xw, yw = _make_data(400, 4, seed)
        m = SGTRegressor(
            criterion="absolute_error",
            max_depth=2,
            inner_max_depth=2,
            num_partitions=2,
            coordinate_descent_max_iters=5,
            coordinate_descent_patience=2,
            tao_n_runs=0,
            random_state=seed,
        )
        m.fit(Xw, yw)

    for case in cases:
        times: list[float] = []
        n_nodes_last = None
        n_leaves_last = None
        for r in range(case.repeats):
            os.environ["SGTLEARN_MAE_CD"] = "1"
            os.environ["SGTLEARN_MAE_BACKEND"] = backend
            Xr, yr = _make_data(case.n_samples, case.n_features, seed + r * 17)
            model = SGTRegressor(
                criterion="absolute_error",
                max_depth=case.max_depth,
                inner_max_depth=case.inner_max_depth,
                num_partitions=case.num_partitions,
                coordinate_descent_max_iters=15,
                coordinate_descent_patience=5,
                tao_n_runs=0,
                random_state=seed + r,
            )
            t0 = time.perf_counter()
            model.fit(Xr, yr)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            n_leaves_last = _safe_n_leaves(model)
            est = getattr(model, "_est", None)
            n_nodes_last = int(getattr(est, "num_nodes", -1)) if est is not None else None
            print(
                f"  [{backend}] {case.name} rep={r + 1}/{case.repeats}: "
                f"{elapsed:.3f}s",
                flush=True,
            )

        rows.append(
            {
                "backend": backend,
                "case": case.name,
                "n_samples": case.n_samples,
                "n_features": case.n_features,
                "max_depth": case.max_depth,
                "inner_max_depth": case.inner_max_depth,
                "num_partitions": case.num_partitions,
                "repeats": case.repeats,
                "fit_seconds_mean": statistics.fmean(times),
                "fit_seconds_std": statistics.stdev(times) if len(times) > 1 else 0.0,
                "fit_seconds_min": min(times),
                "fit_seconds_max": max(times),
                "fit_seconds_all": times,
                "n_leaves": n_leaves_last,
                "n_nodes": n_nodes_last,
                "mae_cd": True,
            }
        )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(row["case"], {})[row["backend"]] = row

    out: list[dict[str, Any]] = []
    for case_name, backends in by_case.items():
        sort_row = backends.get("sort")
        bst_row = backends.get("bst")
        if not sort_row or not bst_row:
            continue
        sort_t = float(sort_row["fit_seconds_mean"])
        bst_t = float(bst_row["fit_seconds_mean"])
        out.append(
            {
                "case": case_name,
                "sort_fit_seconds_mean": sort_t,
                "bst_fit_seconds_mean": bst_t,
                "speedup_sort_over_bst": (sort_t / bst_t) if bst_t > 0 else None,
                "n_samples": sort_row["n_samples"],
                "n_features": sort_row["n_features"],
            }
        )
    return out


def _preload_native_extensions(build_dir: Path) -> None:
    """Prefer freshly built ``*.so`` from cmake ``build/`` over a stale venv copy."""
    import importlib.util

    for name in (
        "ShapeGeneralizedTrees",
        "Discretizers",
        "TreeAlternatingOptimization",
    ):
        matches = sorted(build_dir.glob(f"{name}.cpython-*.so"))
        if not matches:
            continue
        so_path = matches[-1]
        sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location(name, so_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Optional subset of case names to run",
    )
    args = parser.parse_args()

    build_dir = ROOT / "build"
    if build_dir.is_dir():
        sys.path.insert(0, str(build_dir))
        _preload_native_extensions(build_dir)

    try:
        import ShapeGeneralizedTrees  # noqa: F401
        from sgtlearn import SGTRegressor  # noqa: F401

        print(f"Using ShapeGeneralizedTrees from {ShapeGeneralizedTrees.__file__}")
    except ImportError as exc:
        print(
            "Failed to import native ShapeGeneralizedTrees / sgtlearn.\n"
            "Build the extension (e.g. cmake --build build) and set "
            "PYTHONPATH=build, or pip install -e .",
            file=sys.stderr,
        )
        print(exc, file=sys.stderr)
        return 1

    cases = CASES
    if args.cases:
        wanted = set(args.cases)
        cases = [c for c in CASES if c.name in wanted]
        if not cases:
            print(f"No matching cases for {args.cases}", file=sys.stderr)
            return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print("SGTRegressor absolute_error fit bench (MAE CD enabled)")
    print(f"backends: sort, bst | cases: {[c.name for c in cases]}")
    print("-" * 60)

    all_rows: list[dict[str, Any]] = []
    for backend in ("sort", "bst"):
        print(f"\n=== backend={backend} ===", flush=True)
        all_rows.extend(
            run_backend(
                backend,
                cases,
                seed=args.seed,
                warmup=not args.no_warmup,
            )
        )

    summary = aggregate(all_rows)
    payload = {
        "timestamp_utc": ts,
        "criterion": "absolute_error",
        "mae_cd_env": "SGTLEARN_MAE_CD=1",
        "backend_env": "SGTLEARN_MAE_BACKEND",
        "tao_n_runs": 0,
        "seed": args.seed,
        "python": sys.version,
        "cases": [asdict(c) for c in cases],
        "runs": all_rows,
        "aggregates": summary,
    }

    json_path = RESULTS_DIR / f"sgt_mae_fit_{ts}.json"
    latest_path = RESULTS_DIR / "sgt_mae_fit_latest.json"
    summary_path = RESULTS_DIR / "sgt_mae_fit_summary.json"
    csv_path = RESULTS_DIR / f"sgt_mae_fit_{ts}.csv"

    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    latest_path.write_text(json.dumps(payload, indent=2) + "\n")
    summary_path.write_text(json.dumps({"timestamp_utc": ts, "aggregates": summary}, indent=2) + "\n")

    fieldnames = [
        "backend",
        "case",
        "n_samples",
        "n_features",
        "max_depth",
        "inner_max_depth",
        "num_partitions",
        "repeats",
        "fit_seconds_mean",
        "fit_seconds_std",
        "fit_seconds_min",
        "fit_seconds_max",
        "n_leaves",
        "n_nodes",
        "mae_cd",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print("\n=== aggregates (sort / bst) ===")
    for agg in summary:
        print(
            f"{agg['case']}: sort={agg['sort_fit_seconds_mean']:.3f}s  "
            f"bst={agg['bst_fit_seconds_mean']:.3f}s  "
            f"speedup={agg['speedup_sort_over_bst']:.2f}x"
        )
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {latest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

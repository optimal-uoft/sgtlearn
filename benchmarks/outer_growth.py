"""Frozen, fresh-process outer-growth benchmark; see outer_growth.md."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import threading
import time
import warnings

import numpy as np


def quality(y, prediction, weights, criterion):
    y = np.asarray(y).reshape(len(y), -1)
    prediction = np.asarray(prediction).reshape(y.shape)
    if criterion in ("gini", "entropy"):
        values, metric = y == prediction, "accuracy"
    elif criterion == "absolute_error":
        values, metric = abs(y - prediction), "mae"
    else:
        values, metric = (y - prediction) ** 2, "mse"
    outputs = np.average(values, axis=0, weights=weights)
    return {"metric": metric, "mean": float(np.mean(outputs)), "per_output": outputs.tolist()}


def summarize(values):
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    return {"median": median, "mad": float(np.median(abs(values - median))),
            "min": float(values.min()), "max": float(values.max())}


def dataset(work):
    rng = np.random.default_rng(work["data_seed"])
    X = rng.standard_normal((work["n"], work["features"])).astype(np.float32)
    signal = 2 * (X[:, 0] > 0) + (X[:, 1] > 0) + 0.5 * X[:, 2] + X[:, 3] * X[:, 4]
    criterion = work["params"]["criterion"]
    if criterion in ("gini", "entropy"):
        y = np.digitize(signal + rng.normal(0, 0.3, len(X)), [0.5, 2.0]).astype(np.int64)
        other = (X[:, 1] + X[:, 4] > 0).astype(np.int64)
    else:
        y = (signal + rng.normal(0, 0.25, len(X))).astype(np.float32)
        other = (X[:, 1] ** 2 + X[:, 4]).astype(np.float32)
    if work.get("duplicate_x"):
        # Same repeated-feature stress pattern as test_mae_regression_stress.py.
        X[: len(X) // 2, 0] = 0.5
    feature_dict = None
    if work.get("categorical_missing"):
        category = rng.integers(0, 3, len(X))
        X[:, :3] = np.eye(3, dtype=np.float32)[category]
        X[rng.random(len(X)) < 0.08, :3] = 0
        X[rng.random(len(X)) < 0.08, 3] = np.nan
        feature_dict = {"category": [0, 1, 2]}
    if work.get("outputs", 1) == 2:
        y = np.column_stack([y, other])
    weights = rng.uniform(0.5, 2, len(X)) if work.get("weighted") else None
    digest = hashlib.sha256(X.tobytes() + y.tobytes() + (weights.tobytes() if weights is not None else b"")).hexdigest()
    return X, y, weights, feature_dict, digest


def structure(model):
    if hasattr(model, "tree_"):
        tree = model.tree_
        leaves = tree.children_left < 0
        return [{"nodes": int(tree.node_count), "depth": int(tree.max_depth),
                 "structural_leaves": int(sum(leaves)), "occupied_leaves": int(sum(leaves & (tree.n_node_samples > 0)))}]
    trees = model.estimators_ if hasattr(model, "estimators_") else [model]
    result = []
    for tree in trees:
        nodes = tree.tree_export()["nodes"]
        leaves = [n for n in nodes if n["is_leaf"]]
        result.append({"nodes": len(nodes), "depth": max(n["depth"] for n in nodes),
                       "structural_leaves": len(leaves), "occupied_leaves": sum(n["n_samples"] > 0 for n in leaves)})
    return result


def provenance(checkout):
    import sklearn
    from threadpoolctl import threadpool_info
    root = Path(sys.prefix).resolve()
    modules = {}
    for name in ("sgtlearn", "ShapeGeneralizedTrees", "Discretizers", "TreeAlternatingOptimization"):
        path = Path(importlib.import_module(name).__file__).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(f"Contaminated import: {name}={path}, expected under {root}")
        modules[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "machine": platform.machine(), "cpu_count": os.cpu_count(), "numpy": np.__version__,
            "sklearn": sklearn.__version__, "modules": modules, "threadpools": threadpool_info(),
            "revision": subprocess.check_output(["git", "-C", checkout, "rev-parse", "HEAD"], text=True).strip(),
            "environment": {k: v for k, v in os.environ.items() if k.startswith(("SGTLEARN", "OMP", "OPENBLAS", "MKL", "VECLIB", "NUMEXPR", "PYTHONHASH", "JOBLIB"))}}


def worker(args, manifest):
    import psutil
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sgtlearn import SGTClassifier, SGTRegressor
    from sgtlearn.ensemble import RandomSGForestClassifier, RandomSGForestRegressor

    work = next(w for w in manifest["workloads"] if w["id"] == args.workload)
    os.environ["SGTLEARN_MAE_CD"] = str(work.get("mae_cd", 0))
    info = provenance(args.checkout)
    X, y, weights, feature_dict, digest = dataset(work)
    params = dict(work["params"], random_state=args.seed)
    classifier = params["criterion"] in ("gini", "entropy")
    fit_kwargs = {"sample_weight": weights}
    if args.model == "cart":
        if feature_dict:
            return {"workload": args.workload, "model": "cart", "seed": args.seed,
                    "excluded": "SGT groups one-hot columns as one categorical feature and handles missing categories; CART cannot preserve this routing task without changed preprocessing."}
        params = {k: v for k, v in params.items() if k in ("criterion", "random_state", "max_depth", "max_leaf_nodes", "min_samples_leaf", "min_impurity_decrease")}
        cls = DecisionTreeClassifier if classifier else DecisionTreeRegressor
    elif work.get("forest"):
        cls = RandomSGForestClassifier if classifier else RandomSGForestRegressor
    else:
        cls = SGTClassifier if classifier else SGTRegressor
    if args.model == "sgt" and feature_dict:
        fit_kwargs["feature_dict"] = feature_dict
    if args.branching_penalty is not None and args.model == "sgt":
        params["branching_penalty"] = args.branching_penalty

    # Sample RSS for the whole fit process and descendants; threaded workers are
    # already included. ru_maxrss separately captures the process lifetime peak.
    process = psutil.Process()
    peak = [0]
    stop = threading.Event()
    def sample():
        while not stop.is_set():
            rss = 0
            for proc in [process, *process.children(recursive=True)]:
                try:
                    rss += proc.memory_info().rss
                except psutil.Error:
                    pass
            peak[0] = max(peak[0], rss)
            stop.wait(0.005)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    fit_times = []
    # Explicit batches amortize timer overhead, without timing imports/startup.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(work.get("fit_batch", 1)):
            model = cls(**params)
            start = time.perf_counter()
            model.fit(X, y, **fit_kwargs)
            fit_times.append(time.perf_counter() - start)
    lifetime_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    stop.set()
    monitor.join()
    train = quality(y, model.predict(X), weights, params["criterion"])
    topology = structure(model)
    prediction_rows = 0
    start = time.perf_counter()
    while time.perf_counter() - start < manifest["predict_min_seconds"]:
        model.predict(X)
        prediction_rows += len(X)
    predict_seconds = time.perf_counter() - start
    captured = []
    if args.rep == -1:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            cls(**params).fit(X, y, **fit_kwargs)
        captured = [str(w.message) for w in records]
    return {"workload": args.workload, "model": args.model, "seed": args.seed, "rep": args.rep,
            "dataset_sha256": digest, "params": model.get_params(), "fit_seconds": float(np.mean(fit_times)),
            "fit_batch_seconds": sum(fit_times), "fit_samples": fit_times,
            "predict_seconds_per_row": predict_seconds / prediction_rows,
            "predict_rows_per_second": prediction_rows / predict_seconds,
            "predict_rows": prediction_rows, "predict_seconds": predict_seconds,
            "peak_fit_process_tree_rss_bytes": peak[0],
            "process_lifetime_peak_rss_bytes": lifetime_peak,
            "train_quality": train, "structure": topology, "warnings": captured, "provenance": info}


def report(rows):
    output = {}
    for workload in sorted({r["workload"] for r in rows}):
        selected = [r for r in rows if r["workload"] == workload]
        measured = [r for r in selected if r["model"] == "sgt" and r.get("rep", -1) >= 0]
        entry = {k: summarize([r[k] for r in measured]) for k in
                 ("fit_seconds", "predict_seconds_per_row", "peak_fit_process_tree_rss_bytes")} if measured else {}
        entry["quality"] = []
        for row in selected:
            if row["model"] != "sgt" or row.get("rep") != -1:
                continue
            cart = next((c for c in selected if c["model"] == "cart" and c["seed"] == row["seed"]), None)
            q = row["train_quality"]
            item = {"seed": row["seed"], "sgt": q, "structure": row["structure"]}
            if cart and "train_quality" in cart:
                reference = cart["train_quality"]
                item["cart"] = reference
                item["worse_than_cart"] = q["mean"] < reference["mean"] if q["metric"] == "accuracy" else q["mean"] > reference["mean"]
            elif cart:
                item["cart_excluded"] = cart["excluded"]
            entry["quality"].append(item)
        output[workload] = entry
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("outer_growth_manifest.json"))
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--workload")
    parser.add_argument("--model", choices=("sgt", "cart"), default="sgt")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rep", type=int, default=-1)
    parser.add_argument("--branching-penalty", type=float)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.worker:
        print(json.dumps(worker(args, manifest)))
        return
    if args.output is None:
        parser.error("--output required")
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.output / "raw.jsonl"
    rows = [json.loads(line) for line in raw.read_text().splitlines()] if raw.exists() else []
    env = dict(os.environ, **manifest["environment"])
    env.pop("PYTHONPATH", None)
    for work in manifest["workloads"]:
        if args.workload and work["id"] != args.workload:
            continue
        cases = [("sgt", manifest["timing_seed"], r) for r in range(-1, manifest["repetitions"])]
        cases += [("sgt", s, -1) for s in manifest["quality_seeds"] if s != manifest["timing_seed"]]
        cases += [("cart", s, -1) for s in manifest["quality_seeds"]]
        for model, seed, rep in cases:
            if any(r["workload"] == work["id"] and r["model"] == model and r["seed"] == seed and r.get("rep", -1) == rep for r in rows):
                continue
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--manifest", str(args.manifest.resolve()),
                       "--checkout", args.checkout, "--workload", work["id"], "--model", model, "--seed", str(seed), "--rep", str(rep)]
            if args.branching_penalty is not None:
                command += ["--branching-penalty", str(args.branching_penalty)]
            run = subprocess.run(command, cwd=args.checkout, env=env, capture_output=True, text=True)
            with (args.output / "stderr.log").open("a") as log:
                log.write(f"{work['id']} {model} {seed} {rep}\n{run.stderr}\n")
            if run.returncode:
                raise RuntimeError(f"Worker failed: {command}\n{run.stderr}\n{run.stdout}")
            row = json.loads(run.stdout)
            rows.append(row)
            with raw.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            (args.output / "summary.json").write_text(json.dumps(report(rows), indent=2) + "\n")
            print(work["id"], model, seed, rep, row.get("fit_seconds", row.get("excluded")), flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Reproducible two-objective search on a small TabArena subset.

Install ``pip install -e '.[tune]' openml pandas``. Run from the repo root:
``python -m benchmarks.tune_defaults --trials 512 --hours 11``.
The official repeat-0/fold-0 test sets are never used by the objective.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from multiprocessing import Pool
from multiprocessing import TimeoutError as FitTimeout
from pathlib import Path

import numpy as np
import openml
import optuna
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from sgtlearn import SGTClassifier, SGTRegressor

SEARCH_DATASETS = [
    ("credit-g", 363626, "classification"),
    ("diabetes", 363629, "classification"),
    ("maternal_health_risk", 363685, "classification"),
    ("airfoil_self_noise", 363612, "regression"),
    ("concrete_compressive_strength", 363625, "regression"),
    ("blood-transfusion-service-center", 363621, "classification"),
    ("website_phishing", 363707, "classification"),
    ("churn", 363623, "classification"),
    ("QSAR_fish_toxicity", 363698, "regression"),
    ("healthcare_insurance_expenses", 363675, "regression"),
    ("Another-Dataset-on-used-Fiat-500", 363615, "regression"),
    ("miami_housing", 363686, "regression"),
]
HOLDOUT_DATASETS = [
    ("qsar-biodeg", 363696, "classification"),
    ("wine_quality", 363708, "regression"),
]
SEED = 42
# Previous tree HPs under the requested target scaling and fixed TAO rate.
BASELINE = {
    "max_depth": None,
    "max_leaf_nodes": None,
    "num_partitions": 2,
    "inner_max_depth": None,
    "inner_max_leaf_nodes": 8,
    "min_samples_leaf": 1,
    "inner_min_samples_leaf": 1,
    "min_impurity_decrease": 0.0,
    "inner_min_impurity_decrease": 0.0,
    "coordinate_descent_max_iters": 20,
    "coordinate_descent_patience": 5,
    "pairwise_candidates": 0,
    "pairwise_penalty": 0.0,
    "branching_penalty": 0.0,
    "max_features": None,
    "tao_n_runs": 10,
    "tao_lambda": 1e-4,
    "tao_pair_scale": 1.1,
}


def suggest_broad_params(trial):
    """Exploratory space, deliberately frozen separately from the public helper."""
    return BASELINE | {
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "inner_max_leaf_nodes": trial.suggest_int(
            "inner_max_leaf_nodes", 2, 256, log=True
        ),
        "num_partitions": trial.suggest_int("num_partitions", 2, 8),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 32, log=True),
        "inner_min_samples_leaf": trial.suggest_int(
            "inner_min_samples_leaf", 1, 32, log=True
        ),
        "min_impurity_decrease": trial.suggest_categorical(
            "min_impurity_decrease", [0.0, 0.01, 0.1, 1.0, 10.0]
        ),
        "inner_min_impurity_decrease": trial.suggest_categorical(
            "inner_min_impurity_decrease", [0.0, 0.001, 0.01, 0.1]
        ),
        "coordinate_descent_max_iters": trial.suggest_categorical(
            "coordinate_descent_max_iters", [5, 10, 20, 40]
        ),
        "coordinate_descent_patience": trial.suggest_int(
            "coordinate_descent_patience", 1, 5
        ),
        "pairwise_candidates": trial.suggest_categorical(
            "pairwise_candidates", [0.0, 0.25, 0.5, 1.0, 2.0]
        ),
        "pairwise_penalty": trial.suggest_categorical(
            "pairwise_penalty", [0.0, 0.01, 0.1, 1.0, 10.0]
        ),
        "branching_penalty": trial.suggest_categorical(
            "branching_penalty", [0.0, 0.01, 0.1, 1.0, 10.0]
        ),
        "tao_n_runs": trial.suggest_categorical("tao_n_runs", [1, 3, 10, 30]),
        "tao_pair_scale": trial.suggest_categorical(
            "tao_pair_scale", [1.0, 1.1, 2.0, 5.0]
        ),
    }


def encode_features(train, valid, categorical):
    """Fit categorical vocabularies on training rows only; retain logical groups."""
    train_parts, valid_parts, groups = [], [], {}
    offset = 0
    for i, column in enumerate(train.columns):
        if categorical[i]:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            a = train[[column]].astype("string").fillna("<missing>")
            b = valid[[column]].astype("string").fillna("<missing>")
            a, b = encoder.fit_transform(a), encoder.transform(b)
        else:
            a = train[[column]].to_numpy(dtype=float)
            b = valid[[column]].to_numpy(dtype=float)
        groups[str(column)] = list(range(offset, offset + a.shape[1]))
        offset += a.shape[1]
        train_parts.append(a)
        valid_parts.append(b)
    return (
        np.concatenate(train_parts, axis=1),
        np.concatenate(valid_parts, axis=1),
        groups,
    )


def prepare_dataset(name, task_id, kind, cache, *, test=False):
    """Cache arrays and exact row indices; test data require an explicit request."""
    suffix = "test" if test else "validation"
    path = cache / f"{name}-{suffix}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            meta = json.loads(str(saved["metadata"]))
            folds = [
                tuple(
                    saved[f"{key}_{i}"]
                    for key in ("X_train", "X_valid", "y_train", "y_valid")
                )
                + (meta["groups"][i],)
                for i in range(len(meta["groups"]))
            ]
        return {"name": name, "kind": kind, "folds": folds}
    task = openml.tasks.get_task(task_id, download_splits=True)
    dataset = task.get_dataset()
    X, y, categorical, _ = dataset.get_data(
        target=task.target_name, dataset_format="dataframe"
    )
    y = y.to_numpy(dtype=str if kind == "classification" else float)
    outer_train, outer_test = task.get_train_test_split_indices(
        repeat=0, fold=0, sample=0
    )
    if test:
        splits = [(outer_train, outer_test)]
    else:
        splitter = (
            StratifiedKFold(3, shuffle=True, random_state=SEED)
            if kind == "classification"
            else KFold(3, shuffle=True, random_state=SEED)
        )
        splits = [
            (outer_train[a], outer_train[b])
            for a, b in splitter.split(X.iloc[outer_train], y[outer_train])
        ]
    payload, folds = {}, []
    meta = {
        "name": name,
        "task_id": task_id,
        "dataset_id": dataset.dataset_id,
        "kind": kind,
        "outer_repeat": 0,
        "outer_fold": 0,
        "groups": [],
        "indices": [],
    }
    for i, (a, b) in enumerate(splits):
        assert not set(a) & set(b)
        if not test:
            assert not set(np.concatenate([a, b])) & set(outer_test)
        xa, xb, groups = encode_features(X.iloc[a], X.iloc[b], categorical)
        folds.append((xa, xb, y[a], y[b], groups))
        meta["groups"].append(groups)
        meta["indices"].append({"train": a.tolist(), "valid": b.tolist()})
        for key, value in zip(
            ("X_train", "X_valid", "y_train", "y_valid"), (xa, xb, y[a], y[b])
        ):
            payload[f"{key}_{i}"] = value
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, metadata=json.dumps(meta), **payload)
    print(f"Prepared {name}: {len(X)} rows, {len(folds)} {suffix} splits", flush=True)
    return {"name": name, "kind": kind, "folds": folds}


def evaluate_dataset(dataset, params, seed=SEED):
    records = []
    for fold, (xa, xb, ya, yb, groups) in enumerate(dataset["folds"]):
        estimator = (
            SGTClassifier if dataset["kind"] == "classification" else SGTRegressor
        )
        started = time.monotonic()
        model = estimator(**params, random_state=seed).fit(xa, ya, feature_dict=groups)
        seconds = time.monotonic() - started
        prediction = model.predict(xb)
        if dataset["kind"] == "classification":
            score = float(balanced_accuracy_score(yb, prediction))
            record = {"balanced_accuracy": score, "loss": 1 - score}
        else:
            mse = float(mean_squared_error(yb, prediction))
            variance = float(np.var(ya))
            if variance <= 0:
                raise ValueError(f"{dataset['name']} has constant training targets")
            record = {"mse": mse, "training_variance": variance, "loss": mse / variance}
        records.append(record | {"fold": fold, "fit_seconds": seconds})
    return {
        "name": dataset["name"],
        "kind": dataset["kind"],
        "folds": records,
        "loss": float(np.mean([r["loss"] for r in records])),
        "fit_seconds": sum(r["fit_seconds"] for r in records),
    }


def evaluate_all(datasets, params, workers=5):
    """Bound a configuration's native fits; terminate its workers on timeout."""
    with Pool(processes=workers) as pool:
        return pool.starmap_async(
            evaluate_dataset, [(data, params) for data in datasets]
        ).get(timeout=600)


def aggregate(results):
    """Equal weight for each dataset within each of the two objectives."""
    return tuple(
        float(np.mean([r["loss"] for r in results if r["kind"] == kind]))
        for kind in ("classification", "regression")
    )


def choose_compromise(front):
    """Nearest to the ideal after front-range normalization; trial ID breaks ties."""
    if not front:
        raise ValueError("No completed Pareto trials")
    front = sorted(front, key=lambda trial: trial.number)
    values = np.asarray([trial.values for trial in front])
    span = np.ptp(values, axis=0)
    normalized = np.divide(
        values - values.min(axis=0), span, out=np.zeros_like(values), where=span > 0
    )
    return front[int(np.argmin(np.linalg.norm(normalized, axis=1)))]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/tuning-29-expanded")
    )
    parser.add_argument("--trials", type=int, default=512)
    parser.add_argument("--hours", type=float, default=11)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument(
        "--warm-start",
        type=Path,
        help="Previous output directory; reevaluate its Pareto configurations on this panel",
    )
    args = parser.parse_args()
    if args.trials < 1 or args.hours <= 0 or args.workers < 1:
        parser.error("trials, hours and workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "data"
    datasets = [prepare_dataset(*spec, cache) for spec in SEARCH_DATASETS]
    if args.pilot:
        cases = {
            "baseline": BASELINE,
            "small": BASELINE
            | {"max_depth": 4, "inner_max_leaf_nodes": 2, "tao_n_runs": 1},
            "rich": BASELINE
            | {
                "max_depth": 10,
                "inner_max_leaf_nodes": 256,
                "num_partitions": 8,
                "pairwise_candidates": 2.0,
                "tao_n_runs": 30,
            },
        }
        results = {}
        for name, params in cases.items():
            start = time.monotonic()
            try:
                rows = evaluate_all(datasets, params, args.workers)
            except FitTimeout:
                results[name] = {
                    "params": params,
                    "timeout_seconds": 600,
                    "wall_seconds": time.monotonic() - start,
                }
                write_json(args.output / "pilot.json", results)
                print(f"{name}: exceeded 600 seconds", flush=True)
                continue
            results[name] = {
                "params": params,
                "datasets": rows,
                "objectives": aggregate(rows),
                "wall_seconds": time.monotonic() - start,
            }
            print(
                name,
                results[name]["objectives"],
                results[name]["wall_seconds"],
                flush=True,
            )
            write_json(args.output / "pilot.json", results)
        return
    storage = f"sqlite:///{args.output.resolve() / 'study.sqlite3'}"
    study = optuna.create_study(
        study_name="shared-defaults-v1",
        storage=storage,
        directions=["minimize", "minimize"],
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=64),
    )
    if args.confirm:
        selected = choose_compromise(study.best_trials)
        params = selected.user_attrs["estimator_params"]
        test_data = [
            prepare_dataset(*spec, cache, test=True)
            for spec in SEARCH_DATASETS + HOLDOUT_DATASETS
        ]
        results = {"selected_trial": selected.number, "params": params}
        for name, config in (("selected", params), ("baseline", BASELINE)):
            results[name] = evaluate_all(test_data, config, args.workers)
            write_json(args.output / "confirmation.json", results)
        return
    source_files = [
        Path(__file__),
        Path("sgtlearn/base.py"),
        Path("sgtlearn/tao.py"),
    ]
    manifest = {
        "datasets": SEARCH_DATASETS,
        "seed": SEED,
        "inner_folds": 3,
        "configuration_timeout_seconds": 600,
        "classification": "mean(1 - balanced_accuracy)",
        "regression": "mean(MSE / train_variance)",
        "sources": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files
        },
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("sgtlearn", "numpy", "scikit-learn", "optuna", "openml")
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "data_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(cache.glob("*-validation.npz"))
        },
    }
    previous = study.user_attrs.get("manifest")
    if previous is not None and previous != json.loads(json.dumps(manifest)):
        raise ValueError("Study inputs changed; use a new output directory")
    study.set_user_attr("manifest", manifest)
    write_json(args.output / "manifest.json", manifest)
    if not study.trials:
        study.enqueue_trial(BASELINE | {"max_depth": 6, "pairwise_candidates": 0.0})
        if args.warm_start is not None:
            previous_study = optuna.load_study(
                study_name="shared-defaults-v1",
                storage=f"sqlite:///{args.warm_start.resolve() / 'study.sqlite3'}",
            )
            for trial in previous_study.best_trials:
                study.enqueue_trial(trial.params, skip_if_exists=True)

    def objective(trial):
        params = suggest_broad_params(trial)
        trial.set_user_attr("estimator_params", params)
        start = time.monotonic()
        try:
            rows = evaluate_all(datasets, params, args.workers)
        except FitTimeout:
            trial.set_user_attr("timeout_seconds", 600)
            raise
        trial.set_user_attr("datasets", rows)
        trial.set_user_attr("wall_seconds", time.monotonic() - start)
        return aggregate(rows)

    remaining = max(0, args.trials - sum(t.state.is_finished() for t in study.trials))
    study.optimize(
        objective, n_trials=remaining, timeout=args.hours * 3600, catch=(FitTimeout,)
    )
    selected = choose_compromise(study.best_trials)
    write_json(
        args.output / "selection.json",
        {
            "trial": selected.number,
            "values": selected.values,
            "params": selected.user_attrs["estimator_params"],
            "pareto_trials": [t.number for t in study.best_trials],
        },
    )
    study.trials_dataframe().to_csv(args.output / "trials.csv", index=False)


if __name__ == "__main__":
    main()

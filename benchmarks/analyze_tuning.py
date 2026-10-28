"""Plot the fixed Pareto choice and measure one-parameter validation sensitivity.

Run after ``benchmarks.tune_defaults`` completes. This never changes the selected
configuration and never uses test scores to choose parameters or ranges.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from optuna.distributions import CategoricalDistribution
from optuna.importance import FanovaImportanceEvaluator, get_param_importances

from benchmarks.tune_defaults import (
    choose_compromise,
    evaluate_all,
    prepare_dataset,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/tuning-29-expanded")
    )
    args = parser.parse_args()
    out = args.output
    study = optuna.load_study(
        study_name="shared-defaults-v1",
        storage=f"sqlite:///{out.resolve() / 'study.sqlite3'}",
    )
    trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
    if any(t.state == optuna.trial.TrialState.RUNNING for t in study.trials):
        raise ValueError("Wait until the search finishes before analyzing")
    selected = choose_compromise(study.best_trials)
    dataset_specs = study.user_attrs["manifest"]["datasets"]
    params = selected.user_attrs["estimator_params"]
    rows = []
    for trial in trials:
        for dataset in trial.user_attrs["datasets"]:
            for fold in dataset["folds"]:
                rows.append(
                    {
                        "trial": trial.number,
                        "dataset": dataset["name"],
                        "kind": dataset["kind"],
                        **fold,
                        **trial.params,
                    }
                )
    pd.DataFrame(rows).to_csv(out / "per_dataset_trials.csv", index=False)
    best = {}
    importance = {}
    for i, (name, _, _) in enumerate(dataset_specs):
        target = lambda t, i=i: t.user_attrs["datasets"][i]["loss"]
        winner = min(trials, key=target)
        best[name] = {
            "trial": winner.number,
            "loss": target(winner),
            "shared_loss": target(selected),
            "params": winner.user_attrs["estimator_params"],
        }
        importance[name] = get_param_importances(
            study, target=target, evaluator=FanovaImportanceEvaluator(seed=42)
        )
    for i, name in enumerate(("classification", "regression")):
        importance[name] = get_param_importances(
            study,
            target=lambda t, i=i: t.values[i],
            evaluator=FanovaImportanceEvaluator(seed=42),
        )
    write_json(out / "dataset_best.json", best)
    write_json(out / "importance.json", importance)

    fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
    ax.scatter(
        [t.values[1] for t in trials],
        [1 - t.values[0] for t in trials],
        s=15,
        alpha=0.25,
        color="#6c849d",
        label="Search trials",
    )
    front = sorted(study.best_trials, key=lambda t: t.values[1])
    ax.plot(
        [t.values[1] for t in front],
        [1 - t.values[0] for t in front],
        "o-",
        color="#176c66",
        label="Pareto front",
    )
    ax.scatter(
        selected.values[1],
        1 - selected.values[0],
        marker="*",
        s=220,
        color="#c45c27",
        edgecolor="white",
        zorder=5,
        label=f"Shared default: trial {selected.number}",
    )
    pilot = json.loads((out / "pilot.json").read_text())["baseline"].get("objectives")
    if pilot is not None:
        ax.scatter(
            pilot[1],
            1 - pilot[0],
            marker="X",
            s=90,
            color="#8b3346",
            label="Previous HPs, scaled targets",
        )
    ax.set(
        xlabel="Mean regression MSE / training-target variance (lower is better)",
        ylabel="Mean classification balanced accuracy (higher is better)",
        title=f"Shared SGT defaults · {len(trials)} configurations · {len(dataset_specs)} datasets",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.15)
    fig.savefig(out / "pareto.png", dpi=180)
    fig.savefig(out / "pareto.svg")
    plt.close(fig)

    datasets = [prepare_dataset(*spec, out / "data") for spec in dataset_specs]
    sensitivity_path = out / "sensitivity.json"
    sensitivity = (
        json.loads(sensitivity_path.read_text()) if sensitivity_path.exists() else {}
    )
    if sensitivity and sensitivity["selected_trial"] != selected.number:
        raise ValueError("Sensitivity belongs to a different selected trial")
    sensitivity = sensitivity or {"selected_trial": selected.number, "parameters": {}}
    for name, distribution in selected.distributions.items():
        if name in sensitivity["parameters"]:
            continue
        if isinstance(distribution, CategoricalDistribution):
            grid = list(distribution.choices)
        elif distribution.log:
            grid = sorted(
                {distribution.low, distribution.high, params[name]}
                | {
                    2**i
                    for i in range(9)
                    if distribution.low <= 2**i <= distribution.high
                }
            )
        else:
            grid = list(range(distribution.low, distribution.high + 1))
        values = []
        for value in grid:
            result = evaluate_all(datasets, params | {name: value})
            values.append({"value": value, "datasets": result})
        sensitivity["parameters"][name] = values
        write_json(sensitivity_path, sensitivity)
        print(f"Sensitivity {name}: {len(grid)} values", flush=True)
    summaries = {}
    for i, (dataset, _, kind) in enumerate(dataset_specs):
        summaries[dataset] = {}
        for name, entries in sensitivity["parameters"].items():
            losses = np.array([e["datasets"][i]["loss"] for e in entries])
            tolerance = 0.01 if kind == "classification" else 0.05 * losses.min()
            good = [
                e["value"]
                for e, loss in zip(entries, losses)
                if loss <= losses.min() + tolerance
            ]
            summaries[dataset][name] = {
                "fanova_importance": importance[dataset][name],
                "best_values": good,
                "best_loss": float(losses.min()),
                "worst_loss": float(losses.max()),
                "improvement_over_shared": float(
                    best[dataset]["shared_loss"] - losses.min()
                ),
            }
    write_json(out / "tuning_ranges.json", summaries)
    print(
        json.dumps(
            {
                "selected_trial": selected.number,
                "objectives": selected.values,
                "params": params,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

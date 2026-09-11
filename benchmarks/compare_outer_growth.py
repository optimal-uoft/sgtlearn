"""Interleave matched old/new workers from outer_growth.py; no training at import."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from outer_growth import report


def performance_ratios(old, new):
    return {key: {"ratio": new[key]["median"] / old[key]["median"],
                  "investigate": new[key]["median"] / old[key]["median"] > limit}
            for key, limit in (("fit_seconds", 1.25), ("predict_seconds_per_row", 1.25),
                               ("peak_fit_process_tree_rss_bytes", 1.5))}


def quality_change(before, after):
    delta = after["mean"] - before["mean"]
    if before["metric"] == "accuracy":
        return f"{100 * delta:+.3f} pp"
    relative = f"{100 * delta / before['mean']:+.2f}%" if before["mean"] != 0 else "relative undefined (baseline 0)"
    return f"{delta:+.6f} ({relative})"


def comparison(rows):
    summaries = {label: report([r for r in rows if r["build"] == label])
                 for label in ("baseline", "candidate")}
    result = {}
    for name in summaries["baseline"].keys() & summaries["candidate"].keys():
        old, new = (summaries[label][name] for label in ("baseline", "candidate"))
        if "fit_seconds" not in old or "fit_seconds" not in new:
            continue
        quality = []
        for before in old["quality"]:
            after = next((q for q in new["quality"] if q["seed"] == before["seed"]), None)
            if after is None:
                continue
            delta = after["sgt"]["mean"] - before["sgt"]["mean"]
            quality.append({"seed": before["seed"], "baseline": before, "candidate": after,
                            "candidate_minus_baseline": delta,
                            "training_quality_decreased": delta < 0 if before["sgt"]["metric"] == "accuracy" else delta > 0})
        result[name] = {"performance": performance_ratios(old, new), "baseline": old,
                        "candidate": new, "quality": quality}
    return dict(sorted(result.items()))


def write_report(output, rows, settings):
    result = comparison(rows)
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Matched outer-growth comparison", "",
             "Training quality is the primary soft signal. Changes below are candidate minus baseline: accuracy changes use percentage points (pp); regression losses show absolute and relative changes. Higher accuracy and lower loss are better. No numeric quality cutoff is imposed. Every worse-than-CART seed remains flagged.", ""]
    if settings["candidate_branching_penalty"] is not None:
        lines += [f"Separate regularization experiment: candidate branching_penalty={settings['candidate_branching_penalty']}; baseline has no public branching penalty. This is not the common zero-penalty comparison.", ""]
    lines += ["| Workload / seed | Metric | Baseline | Candidate | Change | CART | Worse than CART (old/new) | Structural leaves (old/new) |",
              "|---|---|---:|---:|---:|---:|---|---|"]
    for name, entry in result.items():
        for q in entry["quality"]:
            before, after = q["baseline"], q["candidate"]
            cart = f"{before['cart']['mean']:.6f}" if "cart" in before else "Excluded: grouped categorical/missing"
            flags = f"{before.get('worse_than_cart', 'excluded')}/{after.get('worse_than_cart', 'excluded')}"
            leaves = [[t["structural_leaves"] for t in row["structure"]] for row in (before, after)]
            lines.append(f"| {name} / {q['seed']} | {before['sgt']['metric']} | {before['sgt']['mean']:.6f} | {after['sgt']['mean']:.6f} | {quality_change(before['sgt'], after['sgt'])} | {cart} | {flags} | {leaves[0]}/{leaves[1]} |")
    lines += ["", "Ratios are candidate / baseline medians. Investigate fit or prediction >1.25× and peak RSS >1.50×; confirm breaches with matched reruns beyond measured noise. Review changed tree sizes alongside costs.", "",
              "| Workload | Fit ratio | Predict latency ratio | Peak RSS ratio | Review gates |",
              "|---|---:|---:|---:|---|"]
    for name, entry in result.items():
        perf = entry["performance"]
        flags = [key for key, value in perf.items() if value["investigate"]]
        values = [perf[key]["ratio"] for key in ("fit_seconds", "predict_seconds_per_row", "peak_fit_process_tree_rss_bytes")]
        lines.append(f"| {name} | {values[0]:.3f}× | {values[1]:.3f}× | {values[2]:.3f}× | {', '.join(flags) or 'No breach'} |")
    lines += ["", "Raw per-output quality, occupied leaves, depth, node counts, timing dispersion and provenance are preserved in raw.jsonl and comparison.json. Each matched pair uses the same manifest, dataset hash and seed; order alternates by round. Warmup and extra quality runs are excluded from timing medians. The runner documents memory scope in outer_growth.md.", ""]
    (output / "comparison.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-python", required=True)
    parser.add_argument("--baseline-checkout", required=True)
    parser.add_argument("--candidate-python", required=True)
    parser.add_argument("--candidate-checkout", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("outer_growth_manifest.json"))
    parser.add_argument("--workload")
    parser.add_argument("--candidate-branching-penalty", type=float)
    args = parser.parse_args()
    runner = Path(__file__).with_name("outer_growth.py").resolve()
    manifest = json.loads(args.manifest.read_text())
    settings = {"baseline_python": str(Path(args.baseline_python).absolute()),
                "baseline_checkout": str(Path(args.baseline_checkout).resolve()),
                "candidate_python": str(Path(args.candidate_python).absolute()),
                "candidate_checkout": str(Path(args.candidate_checkout).resolve()),
                "candidate_branching_penalty": args.candidate_branching_penalty,
                "manifest": manifest, "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest()}
    for label in ("baseline", "candidate"):
        settings[f"{label}_revision"] = subprocess.check_output(
            ["git", "-C", settings[f"{label}_checkout"], "rev-parse", "HEAD"], text=True).strip()
    args.output.mkdir(parents=True, exist_ok=True)
    settings_path = args.output / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("Output directory belongs to different settings; use a new directory")
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    raw = args.output / "raw.jsonl"
    rows = [json.loads(line) for line in raw.read_text().splitlines()] if raw.exists() else []
    env = dict(os.environ, **manifest["environment"])
    env.pop("PYTHONPATH", None)
    for work in manifest["workloads"]:
        if args.workload and args.workload != work["id"]:
            continue
        cases = [("sgt", manifest["timing_seed"], rep) for rep in range(-1, manifest["repetitions"])]
        cases += [("sgt", seed, -1) for seed in manifest["quality_seeds"] if seed != manifest["timing_seed"]]
        cases += [("cart", seed, -1) for seed in manifest["quality_seeds"]]
        for round_index, (model, seed, rep) in enumerate(cases):
            order = ("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")
            pair = []
            for label in order:
                existing = next((r for r in rows if r["build"] == label and r["workload"] == work["id"] and r["model"] == model and r["seed"] == seed and r.get("rep", -1) == rep), None)
                if existing is not None:
                    pair.append(existing)
                    continue
                command = [settings[f"{label}_python"], str(runner), "--worker", "--manifest", str(args.manifest.resolve()),
                           "--checkout", settings[f"{label}_checkout"], "--workload", work["id"], "--model", model, "--seed", str(seed), "--rep", str(rep)]
                if label == "candidate" and args.candidate_branching_penalty is not None:
                    command += ["--branching-penalty", str(args.candidate_branching_penalty)]
                run = subprocess.run(command, cwd=settings[f"{label}_checkout"], env=env, capture_output=True, text=True, timeout=300)
                with (args.output / "stderr.log").open("a") as log:
                    log.write(f"{label} {work['id']} {model} {seed} {rep}\n{run.stderr}\n")
                if run.returncode:
                    raise RuntimeError(f"Worker failed: {command}\n{run.stderr}\n{run.stdout}")
                row = dict(json.loads(run.stdout), build=label, round=round_index)
                if label == "baseline" and "provenance" in row and row["provenance"]["revision"] != manifest["baseline_revision"]:
                    raise ValueError("Baseline source revision does not match frozen manifest")
                rows.append(row)
                pair.append(row)
                with raw.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(label, work["id"], model, seed, rep, row.get("fit_seconds", row.get("excluded")), flush=True)
            if all("provenance" in row for row in pair):
                if pair[0]["dataset_sha256"] != pair[1]["dataset_sha256"]:
                    raise ValueError("Matched workers produced different datasets")
                for key in ("python", "platform", "machine", "numpy", "sklearn", "environment"):
                    if pair[0]["provenance"][key] != pair[1]["provenance"][key]:
                        raise ValueError(f"Matched environments differ: {key}")
                pools = [[{k: v for k, v in p.items() if k != "filepath"} for p in r["provenance"]["threadpools"]] for r in pair]
                if pools[0] != pools[1]:
                    raise ValueError("Matched native thread pools differ")
            write_report(args.output, rows, settings)
    print(args.output / "comparison.md")


if __name__ == "__main__":
    main()

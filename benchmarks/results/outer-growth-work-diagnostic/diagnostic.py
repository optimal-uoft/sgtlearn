"""Non-timed public-fit/export diagnostic; invoked separately in each env."""
import json
import os
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from outer_growth import dataset, quality
from sgtlearn import SGTClassifier, SGTRegressor
from sgtlearn.ensemble import RandomSGForestClassifier, RandomSGForestRegressor

manifest = json.loads(Path(sys.argv[1]).read_text())
result = []
warnings.simplefilter("ignore")
for work in manifest["workloads"]:
    if work["id"] not in sys.argv[2:]:
        continue
    os.environ["SGTLEARN_MAE_CD"] = str(work.get("mae_cd", 0))
    X, y, weights, feature_dict, digest = dataset(work)
    params = dict(work["params"], random_state=17)
    classifier = params["criterion"] in ("gini", "entropy")
    cls = (RandomSGForestClassifier if classifier else RandomSGForestRegressor) if work.get("forest") else (SGTClassifier if classifier else SGTRegressor)
    for induction_only in ([False, True] if work["id"] == "gini_default_tao" else [False]):
        if induction_only:
            params["tao_n_runs"] = 0
        model = cls(**params).fit(X, y, sample_weight=weights)
        trees = model.estimators_ if work.get("forest") else [model]
        structures = []
        for tree in trees:
            export = tree.tree_export()
            indexed = {n["id"]: n for n in export["nodes"]}
            reachable = []
            pending = [export["root_index"]]
            while pending:
                node = indexed[pending.pop()]
                reachable.append(node)
                pending.extend(node["children"])
            internals = [n for n in reachable if not n["is_leaf"]]
            eligible = [n for n in reachable if (params.get("max_depth") is None or n["depth"] < params["max_depth"]) and n["n_samples"] >= 2 * params["min_samples_leaf"] and n["impurity"] > 0]
            last_children = max(internals, key=lambda n: max(n["children"]))["children"] if internals else []
            structures.append({
                "nodes": len(reachable), "internal_nodes": len(internals),
                "leaves": sum(n["is_leaf"] for n in reachable),
                "max_depth": max(n["depth"] for n in reachable),
                "internal_samples": sum(n["n_samples"] for n in internals),
                "mean_training_path": sum(n["n_samples"] for n in internals) / len(X),
                "eligible_nodes": len(eligible), "eligible_samples": sum(n["n_samples"] for n in eligible),
                "last_created_eligible_children_samples": sum(n["n_samples"] for n in eligible if n["id"] in last_children),
                "inner_bins": sum(len(n["bin_sample_counts"]) for n in internals),
                "nodes_export": reachable,
            })
        result.append({"workload": work["id"], "induction_only_control": induction_only,
                       "params": model.get_params(), "dataset_sha256": digest,
                       "quality": quality(y, model.predict(X), weights, params["criterion"]),
                       "trees": structures})
print(json.dumps(result))

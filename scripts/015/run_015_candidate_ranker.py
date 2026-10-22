"""015 固定 CPU 候选排序对照；只手动运行，不扫描参数、不提交比赛。

运行：.venv-015/bin/python scripts/015/run_015_candidate_ranker.py
重放：.venv-015/bin/python scripts/015/run_015_candidate_ranker.py --replay

输入由 build_015_candidate_features.py 冻结。NP/G1 共用模型、canonical
整组五折；同分子两个参考场景同折。只重排原 v9 top50 内相同分子式的
槽位，其他位置及候选集合不变。OOF、模型、完整排名和人工重放结果都写
入 ignored artifacts，不建立测试体系。最终全量模型仅供后续实验，未部署。
"""
from __future__ import annotations

import os

# Freeze before native numerical libraries initialize; also use threadpoolctl.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
import warnings
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "artifacts/runs/015-analog/features"
OUTPUT = ROOT / "artifacts/runs/015-analog/ranker"
SEED = 20261017
BOOTSTRAPS = 20000
FOLDS = 5
TOP_K = 50
CONFIG = {
    "seed": SEED,
    "folds": FOLDS,
    "fold_rule": "Sort unique truth canonical by SHA256('20261017|015|'+canonical), round-robin 0..4; every domain/scenario copy shares its canonical fold.",
    "top_k": TOP_K,
    "scenarios": ["B_seen", "A_absent"],
    "sample_weight": "Each query total 1.0; each scenario total 0.5; candidates equal weight within query/scenario. No class_weight and no rescaling of total weights.",
    "scaler": "Logistic only: StandardScaler.fit(X_train, sample_weight=train_weight); no held-out rows enter statistics. HGB uses raw features.",
    "logistic": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
    "hgb": {"max_iter": 150, "learning_rate": 0.05, "max_leaf_nodes": 15,
            "min_samples_leaf": 30, "l2_regularization": 1.0,
            "early_stopping": False, "random_state": SEED},
    "threads": 1,
    "bootstrap_draws": BOOTSTRAPS,
    "bootstrap_seed": SEED,
    "models": ["lr_baseline", "lr_analog", "hgb_baseline", "hgb_analog", "hgb_shuffled"],
    "ranking": "Descending predicted positive probability within same-formula original top50 slots; original candidate_position breaks exact ties; remaining v9 order unchanged.",
    "interpretation": "OOF development evidence, not unseen-encoder-structure proof or leaderboard gain. HGB_analog versus HGB_baseline measures added direct+analog feature value together; shuffled is an imperfect negative control retaining direct features and maximum reference Tanimoto.",
}


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs(input_dir):
    manifest_path = input_dir / "manifest.json"
    features_path = input_dir / "features.npz"
    manifest = json.loads(manifest_path.read_text())
    with np.load(features_path, allow_pickle=False) as f:
        arrays = {name: f[name] for name in f.files}
    required = {"X", "X_shuffled", "y", "query_index", "scenario_index", "candidate_position"}
    if not required.issubset(arrays):
        raise ValueError(f"Missing arrays: {required - set(arrays)}")
    names = manifest["feature_names"]
    base_names = manifest["baseline_feature_names"]
    analog_names = manifest["analog_feature_names"]
    queries = manifest["queries"]
    if manifest["scenarios"] != CONFIG["scenarios"]:
        raise ValueError("Scenario order does not match frozen contract")
    if len(set(names)) != len(names) or set(base_names) & set(analog_names):
        raise ValueError("Feature names duplicated or feature families overlap")
    if set(names) != set(base_names) | set(analog_names):
        raise ValueError("Feature families do not partition X")
    X = arrays["X"]
    if X.ndim != 2 or X.shape[1] != len(names) or arrays["X_shuffled"].shape != X.shape:
        raise ValueError("Feature array shape does not match manifest")
    if not np.isfinite(X).all() or not np.isfinite(arrays["X_shuffled"]).all():
        raise ValueError("Nonfinite feature values must be handled before ranker")
    for name in required - {"X", "X_shuffled"}:
        if arrays[name].shape != (len(X),):
            raise ValueError((name, arrays[name].shape, len(X)))
    base_indices = [names.index(name) for name in base_names]
    if not np.array_equal(X[:, base_indices], arrays["X_shuffled"][:, base_indices]):
        raise ValueError("Negative control changed baseline features")
    pairs = defaultdict(list)
    for row, (qi, si, pos) in enumerate(zip(arrays["query_index"], arrays["scenario_index"], arrays["candidate_position"])):
        qi, si, pos = int(qi), int(si), int(pos)
        if not 0 <= qi < len(queries) or not 0 <= si < 2:
            raise ValueError("Unknown query/scenario index")
        pairs[(qi, si)].append((pos, row))
    ordered_rows = {}
    weights = np.empty(len(X), dtype=np.float64)
    for qi, query in enumerate(queries):
        keys = query["candidate_keys"]
        order = query["baseline_order"]
        if keys != order[:TOP_K] or len(set(order)) != len(order):
            raise ValueError((qi, "candidate list is not canonical unique original v9 top50"))
        formulas = query["candidate_formulas"]
        if isinstance(formulas, dict):
            query["candidate_formulas"] = [formulas.get(k) for k in keys]
        if len(query["candidate_formulas"]) != len(keys):
            raise ValueError((qi, "formula length"))
        for si in range(2):
            positions = sorted(pairs.get((qi, si), []))
            if [p for p, _ in positions] != list(range(len(keys))):
                raise ValueError((qi, si, "candidate positions must be unique 0-based top50 positions"))
            rows = np.asarray([row for _, row in positions], dtype=np.int64)
            ordered_rows[(qi, si)] = rows
            weights[rows] = 0.5 / len(rows)
            expected = np.asarray([k == query["truth"] for k in keys], dtype=np.uint8)
            if not np.array_equal(arrays["y"][rows], expected):
                raise ValueError((qi, si, "label does not match canonical truth"))
    if len(ordered_rows) != len(pairs):
        raise ValueError("Unexpected query/scenario pair")
    groups = sorted(set(q["truth"] for q in queries),
                    key=lambda g: hashlib.sha256(f"{SEED}|015|{g}".encode()).hexdigest())
    fold_by_canonical = {g: i % FOLDS for i, g in enumerate(groups)}
    query_fold = np.asarray([fold_by_canonical[q["truth"]] for q in queries], dtype=np.int8)
    row_fold = query_fold[arrays["query_index"]]
    return manifest, arrays, ordered_rows, weights, query_fold, row_fold, fold_by_canonical


def model_spec(name, manifest):
    feature_names = (manifest["baseline_feature_names"] if name.endswith("baseline")
                     else manifest["feature_names"])
    return {"name": name, "family": "lr" if name.startswith("lr_") else "hgb",
            "feature_names": feature_names,
            "feature_indices": [manifest["feature_names"].index(x) for x in feature_names],
            "array": "X_shuffled" if name.endswith("shuffled") else "X"}


def fit_model(spec, arrays, rows, weights):
    X = arrays[spec["array"]][rows][:, spec["feature_indices"]]
    y = arrays["y"][rows]
    scaler = None
    if spec["family"] == "lr":
        scaler = StandardScaler().fit(X, sample_weight=weights[rows])
        X = scaler.transform(X)
        estimator = LogisticRegression(**CONFIG["logistic"])
    else:
        estimator = HistGradientBoostingClassifier(**CONFIG["hgb"])
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(X, y, sample_weight=weights[rows])
    bundle = {"spec": spec, "scaler": scaler, "estimator": estimator}
    info = {"rows": int(len(rows)), "positives": int(y.sum()),
            "weight_sum": float(weights[rows].sum()),
            "positive_weight_sum": float(weights[rows][y == 1].sum()),
            "fit_wall_seconds": time.perf_counter() - start,
            "warnings": [{"category": x.category.__name__, "message": str(x.message)} for x in caught]}
    if scaler is not None:
        info.update({"scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                     "coefficient": estimator.coef_[0].tolist(), "intercept": estimator.intercept_.tolist(),
                     "n_iter": estimator.n_iter_.tolist()})
    else:
        info["n_iter"] = int(estimator.n_iter_)
    return bundle, info


def predict(bundle, arrays, rows):
    spec = bundle["spec"]
    X = arrays[spec["array"]][rows][:, spec["feature_indices"]]
    if bundle["scaler"] is not None:
        X = bundle["scaler"].transform(X)
    return bundle["estimator"].predict_proba(X)[:, 1]


def rank_within_formula(query, scores):
    original = query["baseline_order"]
    result = list(original)
    groups = defaultdict(list)
    for pos, (key, formula) in enumerate(zip(query["candidate_keys"], query["candidate_formulas"])):
        # Missing formula cannot group unrelated structures together.
        groups[formula or "unknown:" + key].append(pos)
    for positions in groups.values():
        order = sorted(positions, key=lambda p: (-float(scores[p]), p))
        for slot, source in zip(positions, order):
            result[slot] = original[source]
    if result[TOP_K:] != original[TOP_K:] or set(result) != set(original):
        raise ValueError("Reranking changed candidate set or tail")
    return result


def truth_rank(order, truth):
    return order.index(truth) + 1 if truth in order else None


def rr(rank):
    return 1.0 / rank if rank is not None and rank <= 25 else 0.0


def bootstrap(delta, canonical):
    # Mean within canonical first so duplicate queries never become independent
    # bootstrap units. In current NP/G1 per-domain inputs all canonical are unique.
    by_group = defaultdict(list)
    for g, value in zip(canonical, delta):
        by_group[g].append(float(value))
    values = np.asarray([np.mean(by_group[g]) for g in sorted(by_group)], dtype=float)
    rng = np.random.default_rng(SEED)
    means = np.empty(BOOTSTRAPS, dtype=float)
    for start in range(0, BOOTSTRAPS, 1000):
        n = min(1000, BOOTSTRAPS - start)
        indices = rng.integers(0, len(values), size=(n, len(values)))
        means[start:start + n] = values[indices].mean(axis=1)
    return {"n_canonical_groups": len(values), "delta_canonical_mean": float(values.mean()),
            "ci95": np.quantile(means, [0.025, 0.975]).tolist()}


def compare(rows, before_name, after_name):
    a = np.asarray([r["reciprocal_ranks"][before_name] for r in rows], dtype=float)
    b = np.asarray([r["reciprocal_ranks"][after_name] for r in rows], dtype=float)
    delta = b - a
    return {"n_queries": len(rows), "before": before_name, "after": after_name,
            "before_mrr25": float(a.mean()), "after_mrr25": float(b.mean()),
            "delta_mrr25": float(delta.mean()),
            "paired_canonical_bootstrap": bootstrap(delta, [r["truth"] for r in rows]),
            "before_top1": int((a == 1).sum()), "after_top1": int((b == 1).sum()),
            "before_top25": int((a > 0).sum()), "after_top25": int((b > 0).sum()),
            "improved": int((delta > 1e-12).sum()), "worsened": int((delta < -1e-12).sum()),
            "top1_gained": int(((a < 1) & (b == 1)).sum()), "top1_lost": int(((a == 1) & (b < 1)).sum()),
            "top25_gained": int(((a == 0) & (b > 0)).sum()), "top25_lost": int(((a > 0) & (b == 0)).sum()),
            "changed_orders": sum(r["orders"][before_name] != r["orders"][after_name] for r in rows)}


def assemble_results(manifest, oof, ordered_rows, query_fold):
    rows = []
    for qi, query in enumerate(manifest["queries"]):
        for si, scenario in enumerate(manifest["scenarios"]):
            orders = {"v9": query["baseline_order"]}
            for name in CONFIG["models"]:
                orders[name] = rank_within_formula(query, oof[name][ordered_rows[(qi, si)]])
            ranks = {name: truth_rank(order, query["truth"]) for name, order in orders.items()}
            rows.append({"query_index": qi, "dataset": query["dataset"], "key": query["key"],
                         "truth": query["truth"], "scenario": scenario, "fold": int(query_fold[qi]),
                         "truth_in_top50": query["truth"] in query["candidate_keys"],
                         "ranks": ranks, "reciprocal_ranks": {name: rr(rank) for name, rank in ranks.items()},
                         "orders": orders})
    metrics = {}
    for domain in sorted(set(r["dataset"] for r in rows)):
        metrics[domain] = {}
        for scenario in manifest["scenarios"]:
            selected = [r for r in rows if r["dataset"] == domain and r["scenario"] == scenario]
            comparisons = {f"{name}_vs_v9": compare(selected, "v9", name) for name in CONFIG["models"]}
            for family in ("lr", "hgb"):
                name = f"{family}_analog_vs_{family}_baseline"
                comparisons[name] = compare(selected, f"{family}_baseline", f"{family}_analog")
            comparisons["hgb_analog_vs_hgb_shuffled"] = compare(selected, "hgb_shuffled", "hgb_analog")
            metrics[domain][scenario] = comparisons
    return rows, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--replay", action="store_true", help="Reload own saved fold models; recompute OOF scores and full rankings without fitting.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wall, cpu = time.perf_counter(), time.process_time()
    manifest, arrays, ordered_rows, weights, query_fold, row_fold, fold_map = load_inputs(args.input_dir)
    input_shas = {name: sha(args.input_dir / name) for name in ("manifest.json", "features.npz")}
    models_dir = args.output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    oof = {name: np.full(len(arrays["X"]), np.nan, dtype=float) for name in CONFIG["models"]}
    if args.replay:
        frozen = json.loads((args.output_dir / "run_manifest.json").read_text())
        if frozen["input_sha256"] != input_shas or frozen["config"] != CONFIG:
            raise ValueError("Input/config changed since model training")
    else:
        environment = {"python": sys.version, "platform": platform.platform(),
                       "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                       "joblib": joblib.__version__, "device": "CPU", "threads": 1}
        frozen = {"created_beijing": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                  "config": CONFIG, "input_sha256": input_shas, "script_sha256": sha(Path(__file__)),
                  "environment": environment, "features": manifest["feature_names"],
                  "baseline_features": manifest["baseline_feature_names"],
                  "analog_features": manifest["analog_feature_names"],
                  "queries": len(manifest["queries"]), "canonical_groups": len(fold_map),
                  "candidate_rows": len(arrays["X"]), "sample_weight_sum": float(weights.sum()),
                  "query_fold_counts": dict(Counter(int(x) for x in query_fold)),
                  "no_top50_truth_by_domain": dict(Counter(q["dataset"] for q in manifest["queries"]
                                                          if q["truth"] not in q["candidate_keys"]))}
        save(args.output_dir / "run_manifest.json", frozen)
        save(args.output_dir / "folds.json", {"rule": CONFIG["fold_rule"], "fold_by_canonical": fold_map,
                                               "query_fold": query_fold.tolist()})
        np.savez_compressed(args.output_dir / "row_alignment.npz", query_index=arrays["query_index"],
                            scenario_index=arrays["scenario_index"], candidate_position=arrays["candidate_position"],
                            y=arrays["y"], fold=row_fold, sample_weight=weights)
    training = {}
    with threadpool_limits(limits=1):
        for name in CONFIG["models"]:
            spec = model_spec(name, manifest)
            training[name] = {"spec": spec, "folds": {}}
            for fold in range(FOLDS):
                train = np.flatnonzero(row_fold != fold)
                valid = np.flatnonzero(row_fold == fold)
                path = models_dir / f"{name}_fold{fold}.joblib"
                if args.replay:
                    expected_sha = frozen["model_sha256"][path.name]
                    if sha(path) != expected_sha:
                        raise ValueError(f"Saved model hash changed: {path.name}")
                    bundle = joblib.load(path)
                else:
                    bundle, info = fit_model(spec, arrays, train, weights)
                    bundle.update({"fold": fold, "train_query_indices": sorted(set(int(x) for x in arrays["query_index"][train])),
                                   "validation_query_indices": sorted(set(int(x) for x in arrays["query_index"][valid])),
                                   "input_sha256": input_shas})
                    joblib.dump(bundle, path, compress=3)
                    training[name]["folds"][str(fold)] = info
                    print(json.dumps({"model": name, "fold": fold, "fit_seconds": round(info["fit_wall_seconds"], 3),
                                      "warnings": len(info["warnings"])}, ensure_ascii=False), flush=True)
                oof[name][valid] = predict(bundle, arrays, valid)
            if not np.isfinite(oof[name]).all():
                raise ValueError((name, "OOF prediction missing/nonfinite"))
            if not args.replay:
                full_rows = np.arange(len(arrays["X"]))
                bundle, info = fit_model(spec, arrays, full_rows, weights)
                bundle.update({"status": "experimental full-data fit, not deployed, not OOF-evaluated",
                               "input_sha256": input_shas})
                joblib.dump(bundle, models_dir / f"{name}_experimental_full.joblib", compress=3)
                training[name]["experimental_full_fit"] = info
        if not args.replay:
            frozen["threadpools_during_fit"] = threadpool_info()
    rows, metrics = assemble_results(manifest, oof, ordered_rows, query_fold)
    if args.replay:
        with np.load(args.output_dir / "oof_candidate_scores.npz", allow_pickle=False) as f:
            max_error = {name: float(np.max(np.abs(oof[name] - f[name]))) for name in CONFIG["models"]}
        saved_rows = json.loads((args.output_dir / "oof_full_rankings.json").read_text())
        saved_metrics = json.loads((args.output_dir / "metrics.json").read_text())
        result = {"input_hashes_unchanged": True, "model_hashes_unchanged": True,
                  "oof_max_absolute_error": max_error,
                  "full_ranking_records_equal": rows == saved_rows,
                  "metrics_equal": metrics == saved_metrics,
                  "n_query_scenario_records": len(rows),
                  "wall_seconds": time.perf_counter() - wall,
                  "cpu_seconds": time.process_time() - cpu}
        save(args.output_dir / "manual_replay.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if any(v > 1e-12 for v in max_error.values()) or rows != saved_rows or metrics != saved_metrics:
            raise ValueError("Manual saved-model replay did not reproduce recorded outputs")
    else:
        np.savez_compressed(args.output_dir / "oof_candidate_scores.npz", **oof)
        save(args.output_dir / "oof_full_rankings.json", rows)
        save(args.output_dir / "metrics.json", metrics)
        save(args.output_dir / "training_details.json", training)
        frozen.update({"wall_seconds": time.perf_counter() - wall, "cpu_seconds": time.process_time() - cpu,
                       "maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                       "maxrss_platform_note": "Darwin ru_maxrss is bytes.",
                       "model_sha256": {p.name: sha(p) for p in sorted(models_dir.glob("*.joblib"))},
                       "output_sha256": {name: sha(args.output_dir / name) for name in
                                         ("oof_candidate_scores.npz", "oof_full_rankings.json", "metrics.json", "folds.json")}})
        save(args.output_dir / "run_manifest.json", frozen)
        save(args.output_dir / "reproduction.json", {
            "train_command": ".venv-015/bin/python scripts/015/run_015_candidate_ranker.py",
            "manual_replay_command": ".venv-015/bin/python scripts/015/run_015_candidate_ranker.py --replay",
            "input_directory": str(args.input_dir), "output_directory": str(args.output_dir),
            "candidate_score_alignment": "oof_candidate_scores.npz arrays use the exact feature row order. row_alignment.npz provides query_index, scenario_index, candidate_position, label, fold and sample_weight; candidate keys/formulas are in the input manifest.",
            "prediction": "Use saved model spec feature_indices and array name. Apply saved scaler only if non-null; predict_proba[:,1]. Use the fold matching each query's canonical group. No fitted alpha, clip or post-hoc gating is added at inference.",
            "ranking": CONFIG["ranking"],
            "model_files": "models/*_fold[0-4].joblib generate OOF predictions. *_experimental_full.joblib were fitted on all current research rows and have no OOF performance of their own; not integrated or deployed.",
            "serialization": "joblib files are own locally generated Python objects and require a compatible saved sklearn environment; loading untrusted foreign pickle/joblib files is outside this workflow.",
            "environment": frozen["environment"], "input_sha256": input_shas,
            "frozen_config": CONFIG,
        })
        print(json.dumps({"status": "finished", "wall_seconds": frozen["wall_seconds"],
                          "queries": len(manifest["queries"]), "models": CONFIG["models"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

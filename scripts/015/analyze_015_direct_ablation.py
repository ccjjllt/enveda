"""015 事后归因诊断：固定 baseline+direct_quality，检查直接证据贡献。

手动运行：.venv-015/bin/python scripts/015/analyze_015_direct_ablation.py
人工重放：.venv-015/bin/python scripts/015/analyze_015_direct_ablation.py --replay

在原五方案结果已知后提出，必须标注 post-hoc。复用原分组、权重、固定
参数及同式 top50 槽位；不覆盖原方案，不扫描参数，不生成部署模型。
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import resource
import sys
import time
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from workspace_paths import workspace_path

import run_015_candidate_ranker as r

np, joblib = r.np, r.joblib
OUT = r.ROOT / "artifacts/runs/015-analog/direct-only-ablation"
ORIGINAL = r.ROOT / "artifacts/runs/015-analog/ranker"
NAMES = ["lr_direct_only", "hgb_direct_only"]


def results(manifest, original_rows, oof, ordered_rows):
    rows = []
    for original in original_rows:
        row = {**original, "orders": dict(original["orders"]), "ranks": dict(original["ranks"]),
               "reciprocal_ranks": dict(original["reciprocal_ranks"])}
        qi = row["query_index"]
        si = manifest["scenarios"].index(row["scenario"])
        for name in NAMES:
            order = r.rank_within_formula(manifest["queries"][qi], oof[name][ordered_rows[(qi, si)]])
            rank = r.truth_rank(order, row["truth"])
            row["orders"][name] = order
            row["ranks"][name] = rank
            row["reciprocal_ranks"][name] = r.rr(rank)
        rows.append(row)
    metrics = {}
    for dataset in sorted({x["dataset"] for x in rows}):
        metrics[dataset] = {}
        for scenario in manifest["scenarios"]:
            selected = [x for x in rows if x["dataset"] == dataset and x["scenario"] == scenario]
            comparisons = {}
            for family in ("lr", "hgb"):
                direct = f"{family}_direct_only"
                comparisons[f"{direct}_vs_v9"] = r.compare(selected, "v9", direct)
                comparisons[f"{direct}_vs_{family}_baseline"] = r.compare(selected, f"{family}_baseline", direct)
                comparisons[f"{family}_analog_vs_{direct}"] = r.compare(selected, direct, f"{family}_analog")
            metrics[dataset][scenario] = comparisons
    return rows, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)
    wall, cpu = time.perf_counter(), time.process_time()
    manifest, arrays, ordered_rows, weights, query_fold, row_fold, fold_map = r.load_inputs(r.INPUT)
    input_sha = {str(p.relative_to(r.ROOT)): r.sha(p) for p in
                 (r.INPUT / "manifest.json", r.INPUT / "features.npz", ORIGINAL / "oof_full_rankings.json",
                  ORIGINAL / "folds.json", ORIGINAL / "run_manifest.json")}
    original_rows = json.loads((ORIGINAL / "oof_full_rankings.json").read_text())
    original_run = json.loads((ORIGINAL / "run_manifest.json").read_text())
    original_folds = json.loads((ORIGINAL / "folds.json").read_text())
    if fold_map != original_folds["fold_by_canonical"] or query_fold.tolist() != original_folds["query_fold"]:
        raise ValueError("Post-hoc diagnosis must retain original folds")
    names = manifest["baseline_feature_names"] + ["direct_quality"]
    indices = [manifest["feature_names"].index(name) for name in names]
    specs = {name: {"name": name, "family": name.split("_")[0], "feature_names": names,
                    "feature_indices": indices, "array": "X"} for name in NAMES}
    if args.replay:
        frozen = json.loads((OUT / "run_manifest.json").read_text())
        frozen_input_sha = {str(workspace_path(k).relative_to(r.ROOT)): value
                            for k, value in frozen["input_sha256"].items()}
        if input_sha != frozen_input_sha:
            raise ValueError("Ablation input changed")
    else:
        frozen = {"created_beijing": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                  "status": "post-hoc attribution diagnosis after the original five-model results, not an independent confirmation or deployment decision",
                  "question": "Can baseline+direct_quality reproduce the observed direct+analog gain?",
                  "input_sha256": input_sha, "script_sha256": r.sha(Path(__file__)),
                  "frozen_original_config": r.CONFIG, "specs": specs,
                  "environment": original_run["environment"], "same_canonical_folds": True,
                  "no_full_data_fit": True}
        r.save(OUT / "run_manifest.json", frozen)
    oof = {name: np.full(len(arrays["X"]), np.nan) for name in NAMES}
    training = {}
    with r.threadpool_limits(limits=1):
        for name, spec in specs.items():
            training[name] = {}
            for fold in range(r.FOLDS):
                train, valid = np.flatnonzero(row_fold != fold), np.flatnonzero(row_fold == fold)
                path = OUT / "models" / f"{name}_fold{fold}.joblib"
                if args.replay:
                    if r.sha(path) != frozen["model_sha256"][path.name]:
                        raise ValueError("Ablation model changed")
                    bundle = joblib.load(path)
                else:
                    bundle, info = r.fit_model(spec, arrays, train, weights)
                    bundle.update({"fold": fold, "input_sha256": input_sha,
                                   "status": "post-hoc OOF attribution diagnosis, not deployment model"})
                    joblib.dump(bundle, path, compress=3)
                    training[name][str(fold)] = info
                    print(json.dumps({"model": name, "fold": fold, "fit_seconds": info["fit_wall_seconds"],
                                      "warnings": len(info["warnings"])}, ensure_ascii=False), flush=True)
                oof[name][valid] = r.predict(bundle, arrays, valid)
            if not np.isfinite(oof[name]).all():
                raise ValueError("Incomplete post-hoc OOF scores")
    rows, metrics = results(manifest, original_rows, oof, ordered_rows)
    if args.replay:
        with np.load(OUT / "oof_candidate_scores.npz", allow_pickle=False) as saved:
            error = {name: float(np.max(np.abs(oof[name] - saved[name]))) for name in NAMES}
        same_rows = rows == json.loads((OUT / "oof_full_rankings.json").read_text())
        same_metrics = metrics == json.loads((OUT / "metrics.json").read_text())
        report = {"input_hashes_unchanged": True, "model_hashes_unchanged": True,
                  "oof_max_absolute_error": error, "full_rankings_equal": same_rows,
                  "metrics_equal": same_metrics, "wall_seconds": time.perf_counter() - wall}
        r.save(OUT / "manual_replay.json", report)
        print(json.dumps(report), flush=True)
        if max(error.values()) > 1e-12 or not same_rows or not same_metrics:
            raise ValueError("Ablation replay mismatch")
    else:
        np.savez_compressed(OUT / "oof_candidate_scores.npz", **oof)
        r.save(OUT / "oof_full_rankings.json", rows)
        r.save(OUT / "metrics.json", metrics)
        r.save(OUT / "training_details.json", training)
        frozen.update({"wall_seconds": time.perf_counter() - wall, "cpu_seconds": time.process_time() - cpu,
                       "maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                       "warning_count": sum(len(f["warnings"]) for model in training.values() for f in model.values()),
                       "fold_fits": sum(len(model) for model in training.values()),
                       "model_sha256": {p.name: r.sha(p) for p in sorted((OUT / "models").glob("*.joblib"))},
                       "replay_command": ".venv-015/bin/python scripts/015/analyze_015_direct_ablation.py --replay"})
        r.save(OUT / "run_manifest.json", frozen)
        print(json.dumps({"status": "finished", "wall_seconds": frozen["wall_seconds"],
                          "warnings": frozen["warning_count"]}), flush=True)


if __name__ == "__main__":
    main()

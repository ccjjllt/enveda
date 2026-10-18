"""016 panel frozen-model prediction (manual, CPU only).

Applies the three frozen 015 full-data LR models (baseline-only,
direct-only, direct+analog combination) to the 016 panel features, ranks
within same-formula top50 slots (tail unchanged), and reports MRR@25/Top1/
Top25 per panel group x reference scenario, plus paired bootstrap CIs for
the pre-registered contrasts. No fitting, no tuning, no submission.

Manual run: .venv-015/bin/python scripts/run_016_panel_predict.py
017 correction: writes artifacts/017-corrected-features/predict_metrics.json;
keeps the original 016 artifacts. Full rankings and source/model hashes are saved.
"""
import json
import hashlib
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "artifacts/017-corrected-features"
MODELS = {
    "lr_baseline": ROOT / "artifacts/015-analog/ranker/models/lr_baseline_experimental_full.joblib",
    "lr_direct": ROOT / "artifacts/015-analog/direct-only-ablation/models/lr_direct_only_experimental_full.joblib",
    "lr_comb": ROOT / "artifacts/015-analog/ranker/models/lr_analog_experimental_full.joblib",
}
SEED = 20261017


def rank_in_formula(order, formulas, scores):
    result = list(order)
    groups = defaultdict(list)
    for pos, (key, f) in enumerate(zip(order, formulas)):
        groups[f or "unknown:" + key].append(pos)
    for positions in groups.values():
        ranked = sorted(positions, key=lambda p: (-float(scores[p]), p))
        for slot, src in zip(positions, ranked):
            result[slot] = order[src]
    return result


def rr_of(order, truth):
    if truth in order:
        r = order.index(truth) + 1
        return 1.0 / r if r <= 25 else 0.0, r
    return 0.0, None


def boot_ci(a, b, seed=SEED, n=20000):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    diffs = []
    for _ in range(n):
        ix = rng.integers(0, len(a), len(a))
        diffs.append(a[ix].mean() - b[ix].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # The point estimate is the observed paired mean, not the resampling mean.
    return float((a - b).mean()), [float(lo), float(hi)]


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    import argparse
    from threadpoolctl import threadpool_limits
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=str, default=None)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--diagnostic-zero-direct", action="store_true",
                    help="Post-hoc attribution only: also predict frozen LR direct model with its direct_quality column set to zero.")
    a = ap.parse_args()
    start = time.perf_counter()
    in_dir = Path(a.input_dir) if a.input_dir else IN
    out_path = Path(a.output) if a.output else in_dir / "predict_metrics.json"
    with np.load(in_dir / "features.npz", allow_pickle=False) as z:
        X, y = z["X"], z["y"]
        qi, si, pos = z["query_index"], z["scenario_index"], z["candidate_position"]
    manifest = json.loads((in_dir / "manifest.json").read_text())
    feature_names = manifest["feature_names"]
    queries = manifest["queries"]
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite candidate features")
    bundles = {name: joblib.load(p) for name, p in MODELS.items()}
    for name, b in bundles.items():
        if [feature_names[i] for i in b["spec"]["feature_indices"]] != b["spec"]["feature_names"]:
            raise ValueError((name, "feature schema differs from frozen model"))
    rows_of = defaultdict(list)
    for r in range(len(X)):
        rows_of[(int(qi[r]), int(si[r]))].append(r)
    scen_names = manifest["scenarios"]
    if len(rows_of) != len(queries) * len(scen_names):
        raise ValueError("Incomplete query/scenario rows")
    names = ["v9", *MODELS]
    if a.diagnostic_zero_direct:
        names.append("diagnostic_lr_direct_zero")
    grouped, details = defaultdict(list), []
    with threadpool_limits(limits=1):
        for (qii, sii), rows in sorted(rows_of.items()):
            q = queries[qii]
            order = q["baseline_order"]
            forms = q["candidate_formulas"]
            keys = q["candidate_keys"]
            if keys != order[:50] or len(set(order)) != len(order) or len(forms) != len(keys):
                raise ValueError((q["key"], "candidate identity or formula mismatch"))
            by_pos = {int(pos[r]): r for r in rows}
            if len(by_pos) != len(rows) or sorted(by_pos) != list(range(len(keys))):
                raise ValueError((q["key"], sii, "candidate row mismatch"))
            aligned = [by_pos[p] for p in range(len(keys))]
            truth = q["truth"]
            if not np.array_equal(y[aligned], np.asarray([k == truth for k in keys], dtype=y.dtype)):
                raise ValueError((q["key"], "label mismatch"))
            orders, probabilities = {"v9": order}, {}
            for name in names[1:]:
                source = "lr_direct" if name == "diagnostic_lr_direct_zero" else name
                b = bundles[source]
                Xa = X[aligned][:, b["spec"]["feature_indices"]].copy()
                if name == "diagnostic_lr_direct_zero":
                    j = b["spec"]["feature_names"].index("direct_quality")
                    Xa[:, j] = 0.0
                if b["scaler"] is not None:
                    Xa = b["scaler"].transform(Xa)
                scores = b["estimator"].predict_proba(Xa)[:, 1]
                if not np.isfinite(scores).all():
                    raise ValueError((name, q["key"], "nonfinite predictions"))
                new_order = rank_in_formula(order, forms, scores)
                if new_order[50:] != order[50:] or set(new_order) != set(order):
                    raise ValueError("Ranking changed candidate set or tail")
                orders[name] = new_order
                probabilities[name] = scores.tolist()
            paired = {name: rr_of(value, truth) for name, value in orders.items()}
            rrs = {name: value[0] for name, value in paired.items()}
            ranks = {name: value[1] for name, value in paired.items()}
            rec = {"case": q["key"], "truth": truth, "query_index": qii,
                   "group": q["group"], "scenario": scen_names[sii],
                   **rrs, "ranks": ranks}
            grouped[(q["group"], scen_names[sii])].append(rec)
            details.append({**rec, "orders": orders, "candidate_probabilities": probabilities})
    report = {}
    for (group, scen), recs in sorted(grouped.items()):
        if len({r["truth"] for r in recs}) != len(recs):
            raise ValueError("Bootstrap expects one record per canonical within a group/scenario")
        series = {k: np.array([r[k] for r in recs]) for k in names}
        cell = {"n": len(recs), "mrr": {k: float(v.mean()) for k, v in series.items()},
                "top1": {k: int((series[k] == 1).sum()) for k in names},
                "top25": {k: int((series[k] > 0).sum()) for k in names}, "contrasts": {}}
        comparisons = [("lr_comb", "v9"), ("lr_direct", "v9"), ("lr_comb", "lr_direct"),
                       ("lr_baseline", "v9"), ("lr_direct", "lr_baseline"), ("lr_comb", "lr_baseline")]
        if a.diagnostic_zero_direct:
            comparisons += [("diagnostic_lr_direct_zero", "v9"), ("lr_direct", "diagnostic_lr_direct_zero")]
        for after, before in comparisons:
            da, db = series[after], series[before]
            delta, ci = boot_ci(da, db)
            cell["contrasts"][f"{after}_vs_{before}"] = {
                "delta": delta, "ci95": ci,
                "improved": int((da > db + 1e-12).sum()), "worsened": int((da < db - 1e-12).sum()),
                "top1_gained": int(((da == 1) & (db < 1)).sum()),
                "top1_lost": int(((db == 1) & (da < 1)).sum()),
                "top25_gained": int(((da > 0) & (db == 0)).sum()),
                "top25_lost": int(((db > 0) & (da == 0)).sum())}
        report[f"{group}__{scen}"] = cell
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save(out_path, report)
    detail_path = out_path.with_name(out_path.stem + "_details.json")
    save(detail_path, details)
    provenance = {
        "date_beijing": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "scope": "017 corrective reanalysis of the already used 016 panel; frozen models, no fitting or tuning",
        "input_sha256": {name: file_sha(in_dir / name) for name in ("features.npz", "manifest.json")},
        "model_sha256": {name: file_sha(p) for name, p in MODELS.items()},
        "script_sha256": file_sha(Path(__file__)), "seed": SEED, "bootstrap_draws": 20000,
        "bootstrap_unit": "one canonical molecule within each group/scenario; scenarios never pooled as independent",
        "point_estimate": "observed paired mean difference, not bootstrap mean",
        "diagnostic_zero_direct": a.diagnostic_zero_direct,
        "diagnostic_note": "zero-direct is post-hoc attribution on the frozen model, never a chosen deployment gate",
        "wall_seconds": time.perf_counter() - start,
        "output_sha256": {p.name: file_sha(p) for p in (out_path, detail_path)}}
    save(out_path.with_name(out_path.stem + "_run.json"), provenance)
    for k, v in report.items():
        print(k, {m: round(x, 6) for m, x in v["mrr"].items()}, "Top1", v["top1"], flush=True)
        for c, d in v["contrasts"].items():
            print("   ", c, round(d["delta"], 6), [round(x, 6) for x in d["ci95"]], flush=True)


if __name__ == "__main__":
    main()

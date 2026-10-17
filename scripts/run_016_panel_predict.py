"""016 panel frozen-model prediction (manual, CPU only).

Applies the three frozen 015 full-data LR models (baseline-only,
direct-only, direct+analog combination) to the 016 panel features, ranks
within same-formula top50 slots (tail unchanged), and reports MRR@25/Top1/
Top25 per panel group x reference scenario, plus paired bootstrap CIs for
the pre-registered contrasts. No fitting, no tuning, no submission.

Manual run: .venv-015/bin/python scripts/run_016_panel_predict.py
Writes artifacts/016-panel-features/predict_metrics.json.
"""
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "artifacts/016-panel-features"
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
    return float(np.mean(diffs)), [float(lo), float(hi)]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=str, default=None)
    ap.add_argument("--output", type=str, default=None)
    a = ap.parse_args()
    in_dir = Path(a.input_dir) if a.input_dir else IN
    out_path = Path(a.output) if a.output else in_dir / "predict_metrics.json"
    z = np.load(in_dir / "features.npz")
    X, y = z["X"], z["y"]
    qi, si, pos = z["query_index"], z["scenario_index"], z["candidate_position"]
    manifest = json.load(open(in_dir / "manifest.json"))
    queries = manifest["queries"]
    bundles = {name: joblib.load(p) for name, p in MODELS.items()}
    # row lookup per (qi, si)
    rows_of = defaultdict(list)
    for r in range(len(X)):
        rows_of[(int(qi[r]), int(si[r]))].append(r)
    scen_names = manifest["scenarios"]
    out = {"scenarios": scen_names, "groups": {}}
    for (qii, sii), rows in sorted(rows_of.items()):
        q = queries[qii]
        order = q["baseline_order"][:50]
        forms = q["candidate_formulas"][:50]
        # align rows to candidate positions
        by_pos = {int(pos[r]): r for r in rows}
        assert sorted(by_pos) == list(range(len(order))), (q["key"], sii)
        truth = q["truth"]
        base_rr, base_rank = rr_of(order, truth)
        rrs = {"v9": base_rr}
        ranks = {"v9": base_rank}
        for name, b in bundles.items():
            spec = b["spec"]
            Xa = X[[by_pos[p] for p in range(len(order))]][:, spec["feature_indices"]]
            if b["scaler"] is not None:
                Xa = b["scaler"].transform(Xa)
            s = b["estimator"].predict_proba(Xa)[:, 1]
            new_order = rank_in_formula(order, forms, s)
            r, rank = rr_of(new_order, truth)
            rrs[name] = r
            ranks[name] = rank
        out["groups"].setdefault((q["group"], scen_names[sii]), []).append(
            {"case": q["key"], **rrs, "ranks": ranks})
    report = {}
    for (group, scen), recs in sorted(out["groups"].items()):
        series = {k: np.array([r[k] for r in recs]) for k in ["v9", *MODELS]}
        top1 = {k: int((np.array([r["ranks"][k] for r in recs]) == 1).sum()) for k in ["v9", *MODELS]}
        top25 = {k: int(sum(1 for r in recs if r["ranks"][k] is not None and r["ranks"][k] <= 25))
                 for k in ["v9", *MODELS]}
        cell = {"n": len(recs),
                "mrr": {k: float(v.mean()) for k, v in series.items()},
                "top1": top1, "top25": top25, "contrasts": {}}
        for a, b in [("lr_comb", "v9"), ("lr_direct", "v9"), ("lr_comb", "lr_direct"),
                      ("lr_baseline", "v9")]:
            d, ci = boot_ci(series[a], series[b])
            cell["contrasts"][f"{a}_vs_{b}"] = {"delta": d, "ci95": ci}
        report[f"{group}__{scen}"] = cell
    json.dump(report, open(out_path, "w"), indent=2)
    for k, v in report.items():
        print(k, {m: round(x, 4) for m, x in v["mrr"].items()},
              "Top1", v["top1"], "Top25", v["top25"])
        for c, d in v["contrasts"].items():
            print("   ", c, round(d["delta"], 4), [round(x, 4) for x in d["ci95"]])


if __name__ == "__main__":
    main()

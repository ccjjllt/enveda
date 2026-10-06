"""手动复算 005 已缓存 forward 分数，比较 shortlist 与融合；不运行模型。"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def zscore(values):
    a = np.array([np.nan if v is None else float(v) for v in values])
    ok = np.isfinite(a)
    out = np.zeros(len(a))
    if ok.sum() >= 2:
        sd = a[ok].std(ddof=1)
        if sd > 0:
            out[ok] = (a[ok] - a[ok].mean()) / sd
    return out


def ordered(molecule, channels, weights, k=50, append_missing=True):
    keys = molecule["canon_order_top50"][:k]
    base = np.array([molecule["base_scores"][key] for key in keys])
    fused = zscore(base)
    scored = np.zeros(len(keys), dtype=bool)
    for channel, weight in zip(channels, weights):
        scores = [channel.get(molecule["canon_to_smiles"][key]) for key in keys]
        available = np.array([v is not None and np.isfinite(v) for v in scores])
        if weight != 0:
            scored |= available
            fused += weight * zscore(scores)
    if not scored.any() or not append_missing:
        return [keys[i] for i in sorted(range(len(keys)), key=lambda i: (-fused[i], i))]
    head = sorted(np.flatnonzero(scored), key=lambda i: (-fused[i], i))
    tail = sorted(np.flatnonzero(~scored), key=lambda i: (-base[i], i))
    return [keys[i] for i in head + tail]


def rr(keys, truth):
    for i, key in enumerate(keys[:25], 1):
        if key == truth:
            return 1.0 / i
    return 0.0


def metrics(values):
    a = np.array(values)
    return {"n": len(a), "mrr_at_25": float(a.mean()),
            "top1": float((a == 1).mean()), "top25": float((a > 0).mean())}


def paired(a, b, seed=20261005):
    d = np.array(b) - np.array(a)
    rng = np.random.default_rng(seed)
    draws = d[rng.integers(0, len(d), (10000, len(d)))].mean(axis=1)
    return {"delta_mrr_at_25": float(d.mean()),
            "molecule_bootstrap_95pct": np.quantile(draws, [.025, .975]).tolist(),
            "improved": int((d > 1e-12).sum()), "worsened": int((d < -1e-12).sum()),
            "tied": int((abs(d) <= 1e-12).sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, default=Path("artifacts/005"))
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/006"))
    ap.add_argument("--summary", type=Path, default=Path("data/references/forward-reanalysis-2026-10-05.json"))
    args = ap.parse_args()
    source_files = {}

    def read(name):
        p = args.input_dir / name
        blob = p.read_bytes()
        source_files[name] = {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
        return json.loads(blob)

    molecules = read("np250_shortlists_top50.json")["molecules"]
    gl = read("glacier_scores_np250.json")
    ice = read("iceberg_scores_np250.json")
    filed = read("iceberg_np250_fusion_eval.json")
    filed_by_key = {r["mid"]: r for r in filed["per_molecule"]}
    per_molecule = []
    all_rr = {name: [] for name in ["base", "gl25", "gl50", "ice25", "dual25", "gl50_neutral_missing"]}
    recall = Counter()
    errors = Counter()
    subsets = {"forward_scored": [], "forward_unscored": [], "has_positive_and_negative": [], "negative_only": []}
    reconstruction_errors = Counter()
    formula_oracle_rr = []
    adduct_counts, supported_ce_counts, unsupported_adduct_sets = Counter(), Counter(), Counter()
    planned_jobs, unscored_fiora_eligibility = Counter(), Counter()
    for m in molecules:
        key, truth = m["key"], m["truth_canon"]
        configs = {
            "base": ([], [], 50, True),
            "gl25": ([gl[key]], [1.0], 25, True),
            "gl50": ([gl[key]], [1.0], 50, True),
            "ice25": ([ice[key]], [1.0], 25, True),
            "dual25": ([gl[key], ice[key]], [.5, .5], 25, True),
            "gl50_neutral_missing": ([gl[key]], [1.0], 50, False),
        }
        orders = {name: ordered(m, *cfg) for name, cfg in configs.items()}
        rs = {name: rr(order, truth) for name, order in orders.items()}
        for name, v in rs.items():
            all_rr[name].append(v)
        for name, field in [("base", "base_rr"), ("gl25", "glacier_fused_top25_rr"),
                            ("gl50", "glacier_fused_top50_rr"), ("ice25", "ice_fused_rr")]:
            if abs(rs[name] - filed_by_key[key][field]) > 1e-12:
                reconstruction_errors[name] += 1
        for k in [10, 25, 50]:
            recall[f"base_top{k}"] += truth in m["canon_order_top50"][:k]
        recall["mass_window"] += truth in m["base_scores"]
        scored = int(sum(v is not None and np.isfinite(v) for v in gl[key].values()))
        subset = "forward_scored" if scored else "forward_unscored"
        subsets[subset].append(len(per_molecule))
        if not scored:
            unsupported_adduct_sets[" / ".join(sorted({s["adduct"] for s in m["spectra"]}))] += 1
            mh_negative = any(s["mode"] == "negative" and s["adduct"] == "[M-H]-" for s in m["spectra"])
            unscored_fiora_eligibility["has_MH_negative" if mh_negative else "no_MH_negative"] += 1
        mean_conditions, each_conditions = set(), set()
        for spectrum in m["spectra"]:
            adduct_counts[spectrum["adduct"]] += 1
            if spectrum["mode"] == "positive" and spectrum["adduct"] in {"[M+H]+", "[M+Na]+"}:
                supported_ce_counts[str(spectrum["ce_ev"])] += 1
                ce = spectrum["ce_ev"]
                if ce:
                    bucket = lambda value: max(5.0, round(float(value) / 5) * 5.0)
                    mean_conditions.add((spectrum["adduct"], bucket(sum(ce) / len(ce))))
                    each_conditions.update((spectrum["adduct"], bucket(v)) for v in ce)
        for k in [25, 50, 100]:
            n_candidates = min(k, len(m["base_scores"]))
            planned_jobs[f"mean_ce_top{k}"] += n_candidates * len(mean_conditions)
            planned_jobs[f"each_ce_top{k}"] += n_candidates * len(each_conditions)
        modes = {s["mode"] for s in m["spectra"]}
        if modes == {"negative"}:
            subsets["negative_only"].append(len(per_molecule))
        if {"positive", "negative"}.issubset(modes):
            subsets["has_positive_and_negative"].append(len(per_molecule))
        truth_formula = m["formulas"].get(truth)
        full_order = sorted(m["base_scores"], key=lambda k: (-m["base_scores"][k], k))
        for k in [100, 200]:
            recall[f"base_top{k}"] += truth in full_order[:k]
        oracle = [k for k in full_order if m["formulas"].get(k) == truth_formula] if truth_formula else []
        formula_oracle_rr.append(rr(oracle, truth))
        top_key = orders["gl50"][0] if orders["gl50"] else None
        if rs["gl50"] != 1:
            if truth not in m["base_scores"]:
                errors["truth_outside_mass_window"] += 1
            elif truth not in m["canon_order_top50"]:
                errors["truth_in_window_outside_top50"] += 1
            else:
                errors["truth_in_top50_wrong_top1"] += 1
            if truth_formula and top_key:
                errors["wrong_top1_same_formula" if m["formulas"].get(top_key) == truth_formula else "wrong_top1_different_formula"] += 1
        per_molecule.append({"key": key, "n_spectra": len(m["spectra"]), "modes": sorted(modes),
                             "n_window_canon": m["n_win_canon"], "gl_candidates_scored": scored,
                             "rr": rs, "oracle_formula_fp_rr": formula_oracle_rr[-1],
                             "truth_base_rank_full": m["base_rank_full"]})

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "local_research_date": "2026-10-05 Asia/Shanghai",
        "caliber": "NP250 saved canonical keys, RDKit2026.03.6 artifacts; development analysis, not new GPU inference or leaderboard",
        "seed": 20261005, "source_files": source_files,
        "reconstruction_mismatches_vs_filed": dict(reconstruction_errors),
        "methods": {k: metrics(v) for k, v in all_rr.items()},
        "paired": {name: paired(all_rr[a], all_rr[b]) for name, a, b in [
            ("ice25_minus_gl25", "gl25", "ice25"), ("ice25_minus_gl50", "gl50", "ice25"),
            ("dual25_minus_ice25", "ice25", "dual25"), ("gl50_minus_gl25", "gl25", "gl50"),
            ("neutral_missing_minus_append", "gl50", "gl50_neutral_missing")]},
        "strata": {s: {name: metrics([v[i] for i in indices]) for name, v in all_rr.items()}
                   for s, indices in subsets.items() if indices},
        "candidate_recall_counts": dict(recall), "gl50_error_breakdown": dict(errors),
        "oracle_formula_fp": metrics(formula_oracle_rr),
        "oracle_choose_gl50_or_ice25": metrics(np.maximum(all_rr["gl50"], all_rr["ice25"])),
        "condition_metadata_diagnostic": {
            "np_spectrum_adduct_counts": dict(adduct_counts),
            "positive_supported_spectrum_count": sum(supported_ce_counts.values()),
            "positive_supported_CE_list_counts": dict(supported_ce_counts),
            "unscored_molecule_adduct_set_counts": dict(unsupported_adduct_sets),
            "forward_unscored_fiora_MH_negative_adduct_eligibility": dict(unscored_fiora_eligibility),
            "planned_unique_condition_candidate_jobs": dict(planned_jobs),
            "job_count_note": "按分子内(adduct,5eV bucket)去重，timsTOF映射统一；不含候选解析失败/跨分子缓存/推理成本，非实际运行时长。",
        },
        "limitations": [
            "同一个反复使用的NP开发集；bootstrap不消除checkpoint暴露或选择偏差。",
            "GLACIER已算top50而ICEBERG只算top25，不能重构ICEBERG-top50。",
            "dual25固定权重 .5/.5 只是一次缓存分数的探索，不是独立验证的新模型。",
            "oracle公式及oracle择模型用真值，仅诊断，不能部署。",
            "保存分数已经聚合谱，无法从中恢复不同CE/adduct的逐谱效果。",
            "合法候选身份按已有canonical key，不等同官方2026.03.3 SMILES复评。",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "forward_per_molecule.json").write_text(json.dumps(per_molecule, ensure_ascii=False, indent=2) + "\n")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "source_files"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""手动复盘 004 的已保存名次；不运行模型，不用隐藏标签，不是自动化测试。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(ranks):
    rr = np.array([1.0 / r if r is not None and 1 <= r <= 25 else 0.0 for r in ranks])
    return {
        "n": len(ranks),
        "mrr_at_25": float(rr.mean()),
        "top1": sum(r == 1 for r in ranks) / len(ranks),
        "top25": float((rr > 0).mean()),
        "rank_gt25": sum(r is not None and r > 25 for r in ranks),
        "rank_missing": sum(r is None for r in ranks),
        "untruncated_mrr": sum(1.0 / r for r in ranks if r) / len(ranks),
    }


def paired(a, b, seed, repetitions):
    """按分子成对重采样；区间仅描述这个已反复使用的观察集。"""
    def rr(r):
        return 1.0 / r if r is not None and 1 <= r <= 25 else 0.0
    delta = np.array([rr(y) - rr(x) for x, y in zip(a, b)])
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), (repetitions, len(delta)))].mean(axis=1)
    return {
        "delta_mrr_at_25": float(delta.mean()),
        "paired_molecule_bootstrap_95pct": np.quantile(means, [.025, .975]).tolist(),
        "improved_molecules": int((delta > 0).sum()),
        "worsened_molecules": int((delta < 0).sum()),
        "unchanged_molecules": int((delta == 0).sum()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, default=Path("artifacts/runs/004-research"))
    ap.add_argument("--output", type=Path, default=Path("data/references/004-metric-reanalysis-2026-10-04.json"))
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--bootstrap-repetitions", type=int, default=10000)
    args = ap.parse_args()
    files = {}

    def read(name):
        blob = (args.input_dir / name).read_bytes()
        files[name] = {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
        return json.loads(blob)

    queries = read("np_queries_1184.json")
    r1 = read("r1_np1184.json")
    r2 = read("r2_np250.json")
    merged = read("merged_np250.json")["ranks"]
    cosine = read("r1_cosine_np1184.json")
    grouped = defaultdict(list)
    for i, q in enumerate(queries):
        grouped[q["inchikey14"]].append(i)
    r1_by_id = {r["id"]: r for r in r1}
    if len(merged) != len(grouped) or [r["key"] for r in r2] != list(grouped):
        raise ValueError("保存文件顺序与 notebook 的 Q 分组顺序不一致，不能做 paired 比较")
    # merged 输出未保存 key；这里只能按已审阅 notebook 的相同 G 构造顺序对齐。
    keys = list(grouped)
    variants = {name: [r[name] for r in r2] for name in
                ["single_rank", "mean_all_rank", "mean_ad_rank", "rrfuse_rank"]}
    variants["merged_rank"] = merged
    oracle = []
    selection_mismatches = []
    per_molecule = []
    for row, key in zip(r2, keys):
        ids = grouped[key]
        selected_id = max(ids, key=lambda i: len(queries[i]["ms2_mzs"]))
        ranks = [r1_by_id[i]["rank"] for i in ids]
        best = min((r for r in ranks if r is not None), default=None)
        oracle.append(best)
        if r1_by_id[selected_id]["rank"] != row["single_rank"]:
            selection_mismatches.append(key)
        q = queries[selected_id]
        per_molecule.append({
            "key": key, "n_spectra": len(ids), "selected_query_id": selected_id,
            "selected_adduct": q["adduct"], "selected_mode": q["ionization_mode"],
            "selected_raw_peaks": len(q["ms2_mzs"]),
            "selected_mass_window_size": r1_by_id[selected_id]["win"],
            "single_rank": row["single_rank"], "oracle_best_spectrum_rank": best,
        })
    single = variants["single_rank"]
    stratified = {}
    for field in ["selected_adduct", "selected_mode"]:
        stratified[field] = {
            label: summarize([r["single_rank"] for r in per_molecule if r[field] == label])
            for label in sorted({r[field] for r in per_molecule})
        }
    stratified["n_spectra"] = {
        label: summarize([r["single_rank"] for r in per_molecule if lo <= r["n_spectra"] <= hi])
        for label, lo, hi in [("1", 1, 1), ("2-4", 2, 4), ("5+", 5, 100000)]
        if any(lo <= r["n_spectra"] <= hi for r in per_molecule)
    }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python scripts/005/analyze_005_existing_results.py",
        "seed": args.seed, "bootstrap_repetitions": args.bootstrap_repetitions,
        "caliber": "saved raw inchikey14 ranks; cutoff corrected to 25, NOT re-scored official normalized SMILES",
        "candidate_pool": "004 train-only 275810 structures; NOT v7 union 712199",
        "limitations": [
            "公开 checkpoint 训练暴露未知；NP250 是反复使用的开发观察集，非独立隐藏集。",
            "single/mean/merged 使用不同的质量窗口或窗口并集，不能将差异全归因于融合方法。",
            "merged 文件没有 key；按已审阅 notebook 的相同 Q 分组顺序对齐，无法独立证明 runtime 未重排。",
            "oracle 用真值选最佳谱，只是现有输出中的可达上界，不是可提交的方法。",
            "R3 只保存汇总且未截断25，不能还原逐分子 MRR@25 或 bootstrap。",
            "旧 R4 保存的是窗内原下标，完全不能用于 ranker 效果结论；本脚本不报告为指标。",
            "bootstrap 区间不消除训练暴露、候选池变化或反复挑选方法造成的偏差。",
        ],
        "source_files": files,
        "spectrum_level_r1": summarize([r["rank"] for r in r1]),
        "molecule_level": {name: summarize(ranks) for name, ranks in variants.items()},
        "paired_vs_single": {
            name: paired(single, ranks, args.seed, args.bootstrap_repetitions)
            for name, ranks in variants.items() if name != "single_rank"
        },
        "oracle_best_spectrum": summarize(oracle),
        "oracle_vs_selected": paired(single, oracle, args.seed, args.bootstrap_repetitions),
        "single_reconstruction_mismatches": selection_mismatches,
        "selected_spectrum_strata": stratified,
        "saved_cosine_artifact": summarize([r["rank"] for r in cosine]),
        "spectra_per_molecule": dict(sorted(Counter(len(v) for v in grouped.values()).items())),
        "per_molecule": per_molecule,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in
                      {"per_molecule", "source_files", "spectra_per_molecule"}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

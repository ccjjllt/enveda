"""008-A D2 固定对照（CPU 单配置，不扫描，不重跑 GPU，不重建库）。

冻结输入（只读，不覆盖 D 原始结果）：
  artifacts/007-r1d/feature_table_np_d.json / feature_table_g1_d.json（seen）
  artifacts/007-r1d/feature_table_np_d_source.json / _g1_d_source.json（source 重建）
  artifacts/007-r1/A_full_np.json / A_full_g1.json（forward 缓存：v9/fused_merge/ranked/has_fwd）
  artifacts/007-r1d/folds_np_d.json / folds_g1_d.json（canonical 原分子分折）
  artifacts/007-r1d/config_d.json（特征顺序/种子口径）

固定单配置：α=0.25、τ=1.0，D2 13 维特征与 D 一致，l2=1.0/lr=0.5/iters=500 与 D 一致。
训练与推理共用同一门控评分函数（src/evidence_residual.py）：
  gated = has_lib * model(standardized)，final = v9 + clip(α*gated)，
  分组排序（scored 先、unscored 后，组内按 final，tie 按 order 序号）。
D2 允许无 forward 候选在同一组内按 final 重排（D 按 v9 锁死尾部）；
α=0 必须复现 A 全序（250/250 全序一致才算通过）。

NP/G1 分域独立 5 折，增强 4 副本同折（seen/truth_off/source_off/struct_rand，
与 run_D 同 seed/同 mask 逻辑），标准化只用训练折（含增强分布，不读验证折）。

场景（与 D 同名，不合成总分）：
  seen（有库原场景）、nolib_all（全无库）、pressure_truth_off（真值缺库）、
  source_transfer（来源移除：NP riken / G1 enveda-180 重建表）。

输出（新目录 artifacts/008-d2/，不动 007-r1d）：
  models_D2_{np,g1}.json、oof_scores_D2_{np,g1}.json、
  metrics_D2_{np,g1}.json、run_meta_D2.json、consistency_D2.json（训练/推理一致性）。

零 GPU：不导入 torch，不调用 cuda，不 push/submit，不读保留集。
手动运行：先 `--dry-run` 小样本，再全量；记录墙钟、seed、输入 SHA。

用法：
  /opt/homebrew/bin/python3.11 scripts/run_008_d2.py --dry-run
  /opt/homebrew/bin/python3.11 scripts/run_008_d2.py
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import evidence_residual as er
import numpy as np

ALPHA = 0.25
TAU = 1.0
BOOT = 20000
BSEED = 20261004
AUG_SEED = 20261004
N_COPIES = 4
L2 = 1.0
LR = 0.5
ITERS_FULL = 500

OUTDIR = ROOT / "artifacts" / "008-d2"


def sha256_file(p: Path) -> dict:
    b = p.read_bytes()
    return {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def zero_lib_row(r: dict) -> dict:
    """与 run_D.py:53-71 逐字一致的缺失行（D2 评估场景用，不改口径）。"""
    q = dict(r)
    q["has_lib"] = 0.0
    q["lib_max"] = 0.0
    q["lib_second"] = 0.0
    q["lib_gap"] = 0.0
    q["lib_n_spectra"] = 0.0
    q["lib_n_sources"] = 0.0
    q["best_n_matched"] = 0.0
    q["best_explained"] = 0.0
    q["best_ent_sim"] = 0.0
    q["best_same_adduct"] = 0.0
    q["ce_missing"] = 1.0
    q["neg_ce_diff"] = 0.0
    q["ana_max"] = 0.0
    q["ana_explained"] = 0.0
    q["has_lib_x_base"] = 0.0
    q["sources_sorted"] = []
    return q


def derive_truth_off(seen_rows, truth):
    return [zero_lib_row(r) if r["c"] == truth else dict(r) for r in seen_rows]


def derive_nolib(seen_rows):
    return [zero_lib_row(r) for r in seen_rows]


def struct_rand_mask(order, truth, mol_key, seed=AUG_SEED):
    """与 run_D.py:79-85 逐字一致的每分子确定性 mask（含真值）。"""
    h = int(hashlib.sha256(("%d:%s" % (seed, mol_key)).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(h)
    drop = {}
    for c in order:
        drop[c] = bool(rng.random() < 0.5)
    return drop


def derive_struct_rand(seen_rows, order, mol_key):
    drop = struct_rand_mask(order, None, mol_key)
    out = []
    for r in seen_rows:
        if drop.get(r["c"], False):
            out.append(zero_lib_row(r))
        else:
            out.append(dict(r))
    return out, drop


def rr_of(ranked, truth):
    if truth not in ranked:
        return 0.0, None
    rk = ranked.index(truth) + 1
    return (1.0 / rk if rk <= 25 else 0.0), rk


def boot_ci(delta, seed=BSEED, draws=BOOT):
    d = np.array(delta, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(draws, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(d.mean()), [float(lo), float(hi)]


def run_one_ds(ds: str, iters: int, limit_mols, input_shas: dict) -> dict:
    seen_path = ROOT / ("artifacts/007-r1d/feature_table_%s_d.json" % ds)
    src_path = ROOT / ("artifacts/007-r1d/feature_table_%s_d_source.json" % ds)
    afull_path = ROOT / ("artifacts/007-r1/A_full_%s.json" % ds)
    folds_path = ROOT / ("artifacts/007-r1d/folds_%s_d.json" % ds)
    for p in (seen_path, src_path, afull_path, folds_path):
        input_shas[str(p.relative_to(ROOT))] = sha256_file(p)

    FT = json.loads(seen_path.read_bytes())
    FTSRC = json.loads(src_path.read_bytes())
    AF = json.loads(afull_path.read_bytes())
    folds = json.loads(folds_path.read_bytes())["folds"]

    if limit_mols is not None:
        keep = sorted(FT.keys())[: int(limit_mols)]
        FT = {k: FT[k] for k in keep}
        FTSRC = {k: FTSRC[k] for k in keep if k in FTSRC}
        AF = {k: AF[k] for k in keep if k in AF}

    mol_fold = {}
    for mk, v in FT.items():
        mol_fold[mk] = folds[v["truth"]]
    mols = {}
    for mk, v in FT.items():
        order = [str(c) for c in v["order"]]
        truth = str(v["truth"])
        v9map = {str(c): float(x) for c, x in AF[mk]["v9"].items()}
        fused = AF[mk]["fused_merge"]
        scored_set = set(c for c in order if fused.get(c) is not None)
        mols[mk] = {"truth": truth, "order": order, "v9map": v9map,
                    "seen_rows": v["rows"], "src_rows": FTSRC[mk]["rows"],
                    "fold": int(mol_fold[mk]), "A_rank": AF[mk]["rank"],
                    "A_rr": float(AF[mk]["rr"]),
                    "has_fwd": bool(v.get("has_fwd", False)),
                    "scored_set": scored_set,
                    "A_ranked": [str(c) for c in AF[mk]["ranked"]]}

    # ---- OOF 训练（含增强，门控可学习对过滤） ----
    models = {}
    train_stats = {}
    consistency = {}
    for fold in range(5):
        t0 = time.time()
        train_mks = [mk for mk, m in mols.items()
                     if m["fold"] != fold and m["truth"] in m["order"]]
        X_all: list = []
        pair_index: list = []
        pair_weight: list = []
        base_diff: list = []
        offset = 0
        n_excl_cross = 0
        n_excl_both_nolib = 0
        n_learn_total = 0
        n_skipped_copies = 0
        n_skipped_mols = 0
        for mk in train_mks:
            m = mols[mk]
            order = m["order"]
            truth = m["truth"]
            v9map = m["v9map"]
            scored_set = m["scored_set"]
            truth_group = 1 if truth in scored_set else 0
            seen_rows = m["seen_rows"]
            truth_off_rows = derive_truth_off(seen_rows, truth)
            src_rows = m["src_rows"]
            struct_rows, _ = derive_struct_rand(seen_rows, order, mk)
            copies = [seen_rows, truth_off_rows, src_rows, struct_rows]
            vpos = float(v9map[truth])
            for rows in copies:
                row_by_c = {str(r["c"]): r for r in rows}
                base_idx = offset
                for c in order:
                    X_all.append(er.feat_vec_d2(row_by_c[c]))
                # 本副本可学习对：同组且至少一端有库
                learn_here = []
                for j, c in enumerate(order):
                    if c == truth:
                        continue
                    neg_group = 1 if c in scored_set else 0
                    if neg_group != truth_group:
                        n_excl_cross += 1
                        continue
                    hl_pos = 1.0 if float(row_by_c[truth].get("has_lib", 0.0)) > 0.0 else 0.0
                    hl_neg = 1.0 if float(row_by_c[c].get("has_lib", 0.0)) > 0.0 else 0.0
                    if hl_pos <= 0.0 and hl_neg <= 0.0:
                        n_excl_both_nolib += 1
                        continue
                    pos_global = base_idx + order.index(truth)
                    neg_global = base_idx + j
                    learn_here.append((pos_global, neg_global,
                                       float(vpos - float(v9map[c]))))
                if not learn_here:
                    n_skipped_copies += 1
                else:
                    w_each = (1.0 / N_COPIES) / float(len(learn_here))
                    for a, b, bd in learn_here:
                        pair_index.append((a, b))
                        pair_weight.append(w_each)
                        base_diff.append(bd)
                    n_learn_total += len(learn_here)
                offset += len(order)
        # 若整折零可学习对则如实失败（不静默）。
        if not pair_index:
            raise RuntimeError("fold %d has zero learnable pairs" % fold)
        model = er.train_gated_pairwise_residual(
            X_all, pair_index, pair_weight, base_diff,
            l2=L2, lr=LR, iters=int(iters))
        # 数值一致性断言：同一候选训练打分 == 推理打分。
        chk = er.verify_gated_consistency(X_all, model)
        if not chk["ok"]:
            raise RuntimeError("fold %d gated consistency failed: %s" % (fold, chk))
        model["fold"] = int(fold)
        model["n_mols"] = int(len(train_mks))
        model["n_pairs"] = int(len(pair_index))
        model["n_copies_per_mol"] = int(N_COPIES)
        model["features"] = list(er.D2_FEATURES_13)
        model["alpha_frozen"] = float(ALPHA)
        model["tau"] = float(TAU)
        model["caliber"] = "008d2_gated_canonical_source_struct_v1"
        model["aug"] = ("seen+truth_off_canonical+source_off_exact"
                        "+struct_rand_p0.5_samefold_seed20261004")
        model["fallback"] = "gated residual = has_lib * model(x_std) (train/infer shared)"
        model["ranking"] = ("grouped scored-by-final + unscored-by-final "
                            "(D2 allows unscored reorder within group)")
        model["pair_filter"] = ("same-group only; both-nolib excluded; "
                                "per-copy 1/4 split among learnable")
        model["train_elapsed_s"] = float(time.time() - t0)
        models[str(fold)] = model
        train_stats[str(fold)] = {
            "n_train_mols": int(len(train_mks)),
            "n_X_rows": int(len(X_all)),
            "n_learnable_pairs": int(len(pair_index)),
            "n_excluded_cross_group": int(n_excl_cross),
            "n_excluded_both_nolib": int(n_excl_both_nolib),
            "n_skipped_copies_zero_learnable": int(n_skipped_copies),
            "consistency_max_abs_diff": float(chk["max_abs_diff"]),
            "elapsed_s": float(model["train_elapsed_s"]),
        }
        print("fold %d train mols %d X %d learnable %d excl_cross %d excl_both0 %d "
              "skip_copies %d consist %.3g elapsed %.1fs"
              % (fold, len(train_mks), len(X_all), len(pair_index),
                 n_excl_cross, n_excl_both_nolib, n_skipped_copies,
                 chk["max_abs_diff"], model["train_elapsed_s"]), flush=True)
        # 记录整折零可学习分子数（4 副本全空）。
        # （pair 层已按副本落空统计；分子层需重算以便如实记录。）
        n0 = 0
        for mk in train_mks:
            m = mols[mk]
            order = m["order"]
            truth = m["truth"]
            scored_set = m["scored_set"]
            truth_group = 1 if truth in scored_set else 0
            seen_rows = m["seen_rows"]
            copies = [seen_rows,
                      derive_truth_off(seen_rows, truth),
                      m["src_rows"],
                      derive_struct_rand(seen_rows, order, mk)[0]]
            tot = 0
            for rows in copies:
                rb = {str(r["c"]): r for r in rows}
                for c in order:
                    if c == truth:
                        continue
                    if (1 if c in scored_set else 0) != truth_group:
                        continue
                    hp = 1.0 if float(rb[truth].get("has_lib", 0.0)) > 0.0 else 0.0
                    hn = 1.0 if float(rb[c].get("has_lib", 0.0)) > 0.0 else 0.0
                    if hp > 0.0 or hn > 0.0:
                        tot += 1
                        break
                if tot:
                    break
            if tot == 0:
                n0 += 1
        train_stats[str(fold)]["n_mols_zero_learnable"] = int(n0)
        consistency[str(fold)] = chk

    # ---- OOF 评估（同一门控评分函数） ----
    scenarios = ["seen", "nolib_all", "pressure_truth_off", "source_transfer"]
    results = {}
    oof = {}
    clip_stats = {}
    for sc in scenarios:
        clip_stats[sc] = {"n_cands": 0, "n_clipped": 0}
    n0match = 0
    for mk, m in mols.items():
        fold = m["fold"]
        model = models[str(fold)]
        order = m["order"]
        truth = m["truth"]
        v9map = m["v9map"]
        scored_set = m["scored_set"]
        is_scored = {c: bool(c in scored_set) for c in order}
        has_lib_seen = {}
        for r in m["seen_rows"]:
            has_lib_seen[str(r["c"])] = float(r.get("has_lib", 0.0))
        entry = {"truth": truth, "order": list(order), "fold": int(fold),
                 "A_rank": m["A_rank"], "A_rr": float(m["A_rr"]),
                 "A_ranked": list(m["A_ranked"]),
                 "scored_n": int(len(scored_set)),
                 "unscored_n": int(len(order) - len(scored_set)),
                 "is_scored": {c: bool(is_scored[c]) for c in order},
                 "has_lib_seen": {c: float(has_lib_seen.get(c, 0.0)) for c in order},
                 "scenarios": {}}
        for sc in scenarios:
            if sc == "seen":
                rows = m["seen_rows"]
            elif sc == "nolib_all":
                rows = derive_nolib(m["seen_rows"])
            elif sc == "pressure_truth_off":
                rows = derive_truth_off(m["seen_rows"], truth)
            elif sc == "source_transfer":
                rows = m["src_rows"]
            else:
                raise ValueError(sc)
            row_by_c = {str(r["c"]): r for r in rows}
            gated_map = {}
            masks = {}
            for c in order:
                fv = er.feat_vec_d2(row_by_c[c])
                g = er.gated_residual_row(fv, model)
                gated_map[c] = float(g)
                masks[c] = float(row_by_c[c].get("has_lib", 0.0))
            ranked, finals, nclip = er.rank_grouped_d2(
                order, v9map, gated_map, scored_set, ALPHA, TAU)
            clip_stats[sc]["n_cands"] += len(order)
            clip_stats[sc]["n_clipped"] += int(nclip)
            crr, crank = rr_of(ranked, truth)
            entry["scenarios"][sc] = {
                "ranked": list(ranked), "rr": float(crr),
                "rank": (int(crank) if crank is not None else None),
                "residuals_gated": {c: float(gated_map[c]) for c in order},
                "finals": {c: float(finals[c]) for c in order},
                "masks_has_lib": {c: float(masks[c]) for c in order},
                "n_clipped": int(nclip)}
        # α=0 全序一致性（同一分组排序函数，α=0 应精确复现 A）。
        gated0 = {c: 0.0 for c in order}
        ranked0, _, _ = er.rank_grouped_d2(order, v9map, gated0, scored_set, 0.0, TAU)
        entry["alpha0_grouped_ranked"] = list(ranked0)
        entry["alpha0_matches_A"] = bool(list(ranked0) == list(m["A_ranked"]))
        if entry["alpha0_matches_A"]:
            n0match += 1
        oof[mk] = entry
    print("alpha0 full-order match %d/%d" % (n0match, len(oof)), flush=True)

    for sc in scenarios:
        C_rr = []
        A_rr = []
        for mk, m in mols.items():
            C_rr.append(float(oof[mk]["scenarios"][sc]["rr"]))
            A_rr.append(float(m["A_rr"]))
        C_arr = np.array(C_rr)
        A_arr = np.array(A_rr)

        def mrr_top(a):
            return float(a.mean()), float((a == 1.0).mean()), float((a > 0).mean())

        cm, ct1, ct25 = mrr_top(C_arr)
        am, at1, at25 = mrr_top(A_arr)
        delta = C_arr - A_arr
        dmean, ci = boot_ci(delta)
        imp = int((delta > 1e-12).sum())
        wor = int((delta < -1e-12).sum())
        tie = int((abs(delta) <= 1e-12).sum())
        w2r = int(((A_arr < 1.0) & (C_arr == 1.0)).sum())
        r2w = int(((A_arr == 1.0) & (C_arr < 1.0)).sum())
        cs = clip_stats[sc]
        results[sc] = {"n": int(len(mols)), "A_mrr": float(am),
                       "A_top1": float(at1), "A_top25": float(at25),
                       "C_mrr": float(cm), "C_top1": float(ct1),
                       "C_top25": float(ct25),
                       "delta": float(C_arr.mean() - A_arr.mean()),
                       "ci95": [float(ci[0]), float(ci[1])],
                       "improved": int(imp), "worsened": int(wor),
                       "tied": int(tie), "w2r": int(w2r), "r2w": int(r2w),
                       "clip_n_cands": int(cs["n_cands"]),
                       "clip_n_clipped": int(cs["n_clipped"])}
        print("sc %s: A %.6f C %.6f d %.6f CI [%.6f,%.6f] imp %d wor %d "
              "w2r %d r2w %d clip %d/%d"
              % (sc, am, cm, dmean, ci[0], ci[1], imp, wor, w2r, r2w,
                 cs["n_clipped"], cs["n_cands"]), flush=True)

    metrics = {"scenarios": results, "clip_by_scenario": clip_stats,
               "alpha0_full_order_match": {
                   "n_match": int(n0match), "n_total": int(len(oof)),
                   "mismatched_keys": [k for k, v in oof.items()
                                       if not v["alpha0_matches_A"]][:10]},
               "train_stats": train_stats,
               "alpha": float(ALPHA), "tau": float(TAU),
               "caliber": ("008d2 single-config gated; train/infer shared gated "
                           "residual + clip + grouped sort; "
                           "unscored reorder allowed within group")}
    return {"models": models, "oof": oof, "metrics": metrics,
            "consistency": consistency, "n_mols": len(mols)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="小样本 dry-run（默认前 5 分子/域，iters=50），输出 dryrun 文件")
    ap.add_argument("--limit-mols", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20261004)
    args = ap.parse_args()

    if args.dry_run:
        limit = args.limit_mols if args.limit_mols is not None else 5
        iters = args.iters if args.iters is not None else 50
        suffix = "_dryrun"
    else:
        limit = args.limit_mols
        iters = args.iters if args.iters is not None else ITERS_FULL
        suffix = ""
    assert iters > 0
    assert ALPHA == 0.25 and TAU == 1.0, "D2 只允许固定 α=0.25、τ=1.0"

    # 零 GPU：本脚本全程只用标准库 + numpy，不导入 torch，不读保留集。
    # 此处不做任何 torch/cuda 调用，仅以静态约定保证（见文件头）。

    OUTDIR.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    input_shas: dict = {}
    input_shas["src/evidence_residual.py"] = sha256_file(ROOT / "src/evidence_residual.py")
    input_shas["scripts/run_008_d2.py"] = sha256_file(ROOT / "scripts/run_008_d2.py")
    input_shas["artifacts/007-r1d/config_d.json"] = sha256_file(
        ROOT / "artifacts/007-r1d/config_d.json")

    all_out = {}
    for ds in ("np", "g1"):
        print("=== %s (iters=%d limit=%s) ===" % (ds, iters, limit), flush=True)
        t0 = time.time()
        out = run_one_ds(ds, iters=iters, limit_mols=limit, input_shas=input_shas)
        elapsed = time.time() - t0
        print("%s done in %.1fs" % (ds, elapsed), flush=True)
        (OUTDIR / ("models_D2_%s%s.json" % (ds, suffix))).write_text(
            json.dumps(out["models"], indent=1) + "\n")
        (OUTDIR / ("oof_scores_D2_%s%s.json" % (ds, suffix))).write_text(
            json.dumps(out["oof"]) + "\n")
        (OUTDIR / ("metrics_D2_%s%s.json" % (ds, suffix))).write_text(
            json.dumps(out["metrics"], indent=1, ensure_ascii=False) + "\n")
        all_out[ds] = {"n_mols": out["n_mols"], "elapsed_s": elapsed,
                       "metrics": out["metrics"]}

    total_elapsed = time.time() - t_all
    meta = {"alpha": ALPHA, "tau": TAU, "seed_oof": 20261004,
            "seed_aug": AUG_SEED, "seed_bootstrap": BSEED,
            "iters": int(iters), "limit_mols": limit,
            "dry_run": bool(args.dry_run),
            "wall_clock_total_s": float(total_elapsed),
            "per_ds": {k: {"n_mols": v["n_mols"], "elapsed_s": v["elapsed_s"]}
                       for k, v in all_out.items()},
            "inputs_sha256": input_shas,
            "device": "local CPU, no GPU, no Kaggle, no heldout",
            "command": " ".join(sys.argv),
            "notes": ("D2 固定单配置对照；训练/推理共用门控残差+clip+分组排序；"
                      "unscored 组内允许按 final 重排；α=0 须复现 A 全序。")}
    (OUTDIR / ("run_meta_D2%s.json" % suffix)).write_text(
        json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
    print("wrote %s in %.1fs" % (OUTDIR, total_elapsed), flush=True)
    print(json.dumps(meta, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

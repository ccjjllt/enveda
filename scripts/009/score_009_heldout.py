"""G1保留250一次性确认评分: 冻结009-conditional v1规则, CPU-only, 手动研究脚本, 非测试.

冻结口径逐字复用自 scripts/009/run_009_conditional.py:
  zscore (ddof1), v9_order (scored优先+unscored按base续后),
  full_order (order(-score,canon)), rr_of (RR@25),
  compare (seed 20261012, bootstrap 20000).

路由复用 artifacts/runs/009-heldout/routing.json 既有路由 (不支持重路由).
支持分支: frozen_top100 上 FP representative + 回传 split fused_merge
  (按canon匹配ICE JSON记录; 一canon多SMILES命中时按sorted(SMILES)取第一,
  记录条数, 不得挑分数) 做 v9_order.
不支持分支: all_mean 全窗 full_order. 不支持分支无ICE, 不补分.
observed-ICE 仅审计 mismatch, 不翻转路由.
truth_canon 仅本步一次性评分使用; 不得加阈值/改权重/改分组/重选all_mean;
不得合成隐藏比例总分; 不留第二轮确认接口 (输出已存在即拒绝覆盖).

手动运行:
  /opt/homebrew/bin/python3.11 scripts/009/score_009_heldout.py
"""

from __future__ import annotations

import collections
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261012
DRAWS = 20000


def sha_bytes(path: Path) -> dict:
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
            n += len(block)
    return {"bytes": n, "sha256": h.hexdigest()}


# --- 以下四函数逐字复用 scripts/009/run_009_conditional.py 冻结口径 ---
def zscore(values):
    values = np.array([np.nan if x is None else float(x) for x in values])
    ok = np.isfinite(values)
    result = np.zeros(values.shape)
    if ok.sum() >= 2:
        std = values[ok].std(ddof=1)
        if std > 0:
            result[ok] = (values[ok] - values[ok].mean()) / std
    return result


def compare(rows, a, b, seed, draws):
    x, y = (np.array([r[k] for r in rows], dtype=float) for k in (a, b))
    delta = y - x
    rng = np.random.default_rng(seed)
    boot = rng.integers(0, len(rows), size=(draws, len(rows)))
    means = delta[boot].mean(axis=1)
    ci = np.quantile(means, [0.025, 0.975])
    return {
        "n": len(rows),
        "mrr_a": float(x.mean()),
        "mrr_b": float(y.mean()),
        "delta_mrr": float(delta.mean()),
        "ci95": [float(ci[0]), float(ci[1])],
        "improved": int((delta > 1e-12).sum()),
        "worsened": int((delta < -1e-12).sum()),
        "tied": int((abs(delta) <= 1e-12).sum()),
        "top1_a": int((x == 1).sum()),
        "top1_b": int((y == 1).sum()),
        "top1_wrong_to_right": int(((x < 1) & (y == 1)).sum()),
        "top1_right_to_wrong": int(((x == 1) & (y < 1)).sum()),
        "top25_a": int((x > 0).sum()),
        "top25_b": int((y > 0).sum()),
    }


def rr_of(order, truth):
    return 1.0 / (order.index(truth) + 1) if truth in order[:25] else 0.0


def v9_order(short, fp_rep_map, ice_sims):
    base = np.asarray([float(fp_rep_map[k]) for k in short], dtype=float)
    sims = list(ice_sims)
    finite = np.asarray(
        [v is not None and np.isfinite(float(v)) for v in sims], dtype=bool
    )
    f = zscore(base) + zscore(sims)
    ix = np.arange(len(short))
    fused = sorted(ix[finite].tolist(), key=lambda i: (-float(f[i]), int(i)))
    fused += sorted(ix[~finite].tolist(), key=lambda i: (-float(base[i]), int(i)))
    return [short[i] for i in fused], bool(finite.any())


def full_order(full_keys, fp_map):
    return sorted(full_keys, key=lambda k: (-float(fp_map[k]), str(k)))


def summarize_final_vs_v9(per):
    rows = [{"v9": r["v9_rr"], "final": r["final_rr"]} for r in per]
    comp = compare(rows, "v9", "final", SEED, DRAWS)
    gained = [r["key"] for r in per if r["final_rr"] > 0 and r["v9_rr"] == 0]
    lost = [r["key"] for r in per if r["final_rr"] == 0 and r["v9_rr"] > 0]
    comp["top25_gained_keys"] = gained
    comp["top25_lost_keys"] = lost
    comp["top25_gained_n"] = len(gained)
    comp["top25_lost_n"] = lost and len(lost) or 0
    for label, subset in [
        ("supported", [r for r in per if r["route"] == "retain_v9"]),
        ("unsupported", [r for r in per if r["route"] != "retain_v9"]),
    ]:
        if subset:
            c = compare(
                [{"v9": x["v9_rr"], "final": x["final_rr"]} for x in subset],
                "v9",
                "final",
                SEED,
                DRAWS,
            )
            c["keys"] = [x["key"] for x in subset]
            comp["strata_" + label] = c
    return comp


def main():
    out_path = ROOT / "artifacts/runs/009-heldout/heldout_score.json"
    if out_path.exists():
        raise FileExistsError(
            f"{out_path} exists; 一次性确认已执行, 拒绝覆盖 (不留第二轮确认接口)"
        )
    t0 = time.monotonic()

    frozen_cfg = ROOT / "artifacts/runs/009-conditional/frozen_config.json"
    inputs_path = ROOT / "artifacts/runs/009-heldout/inputs.json"
    cand_path = ROOT / "artifacts/runs/009-heldout/heldout_candidate_scores.json"
    routing_path = ROOT / "artifacts/runs/009-heldout/routing.json"
    split_path = ROOT / "artifacts/runs/009-heldout/scores/scores/v9heldout_ice_top100_split.json"
    split_meta_path = split_path.with_suffix(".json.meta.json")
    mean_path = ROOT / "artifacts/runs/009-heldout/scores/scores/v9heldout_ice_top100_mean.json"
    mean_meta_path = mean_path.with_suffix(".json.meta.json")
    tar_path = ROOT / "artifacts/runs/009-heldout/scores/v9heldout-scores.tar.gz"

    # 1. 冻结核对: 不一致即停.
    cfg_sha = sha_bytes(frozen_cfg)
    assert cfg_sha["sha256"] == (
        "a25d45032c2bcc9ac822a74379f95836892625c1459cbd7fdb29d18274772ed3"
    ), f"frozen_config SHA mismatch: {cfg_sha['sha256']}"

    inputs = json.loads(inputs_path.read_text())
    molecules = {m["key"]: m for m in inputs["molecules"]}
    cand = json.loads(cand_path.read_text())
    routing = json.loads(routing_path.read_text())
    split = json.loads(split_path.read_text())
    mean = json.loads(mean_path.read_text())

    # 2. 路由复用核对 (不重算, 只核对计数).
    assert routing["n"] == 250, routing["n"]
    assert routing["supported_n"] == 232, routing["supported_n"]
    assert routing["unsupported_n"] == 18, routing["unsupported_n"]
    assert set(molecules.keys()) == set(cand.keys()), "inputs/cand key集不一致即停"
    assert len(molecules) == 250
    route_of = {}
    for r in routing["rows"]:
        route_of[r["key"]] = r
    assert set(route_of.keys()) == set(molecules.keys()), "routing key集不一致即停"
    assert set(split.keys()) == set(routing["supported_keys"]), (
        "split覆盖 != 支持集即停"
    )
    # mean 只有 fused (对照结构, 不用于评分).
    assert set(mean.keys()) == set(split.keys())
    assert not any(
        "fused_merge" in r for recs in mean.values() for r in recs.values()
    ), "mean 含 fused_merge 即停 (应只有 fused)"

    # split fused_merge 18644/18644 非空有限断言 (mismatch即停).
    n_tot = n_fin = 0
    for recs in split.values():
        for r in recs.values():
            n_tot += 1
            v = r.get("fused_merge")
            if v is not None and np.isfinite(float(v)):
                n_fin += 1
    assert (n_tot, n_fin) == (18644, 18644), f"split fused_merge {n_fin}/{n_tot}"

    per = []
    mismatches = []
    multi_hit_total = 0
    multi_hit_mols = 0
    miss_total = 0
    for key in sorted(molecules.keys()):
        m = molecules[key]
        entry = cand[key]
        rrow = route_of[key]
        full_keys = list(entry["full_keys"])
        fp_rep = dict(entry["fp"]["representative"])
        fp_all = dict(entry["fp"]["all_mean"])
        short = list(entry["frozen_top100"])
        # 输入一致性 (不一致即停).
        assert short == m["order_top100"], f"{key} frozen_top100与inputs不一致"
        assert set(full_keys) == set(m["base_scores_full_window"].keys()), key
        assert set(fp_rep.keys()) == set(full_keys), key
        assert set(fp_all.keys()) == set(full_keys), key
        # 路由只用既有值, 不重路由.
        meta_supported = bool(rrow["metadata_has_supported_record"])
        route = "v9" if meta_supported else "all_mean_full"

        if meta_supported:
            # 按canon匹配ICE记录; 多SMILES命中按sorted取第一并计数.
            recs = split[key]
            bycanon = collections.defaultdict(list)
            for smi, r in recs.items():
                bycanon[r["canon"]].append(smi)
            ice_sims = []
            nmulti = 0
            nmiss = 0
            for c in short:
                hits = bycanon.get(c, [])
                if not hits:
                    nmiss += 1
                    ice_sims.append(None)
                else:
                    if len(hits) > 1:
                        nmulti += 1
                    smi0 = sorted(hits)[0]
                    ice_sims.append(float(recs[smi0]["fused_merge"]))
            multi_hit_total += nmulti
            if nmulti:
                multi_hit_mols += 1
            miss_total += nmiss
        else:
            ice_sims = [None] * len(short)  # 不支持分支无ICE, 不补分.
            nmulti = 0
            nmiss = 0

        observed = bool(
            np.asarray(
                [v is not None and np.isfinite(float(v)) for v in ice_sims]
            ).any()
        )
        if meta_supported != observed:
            mismatches.append(
                {
                    "key": key,
                    "metadata_has_supported_record": meta_supported,
                    "observed_has_ice": observed,
                    "n_finite": int(
                        np.asarray(
                            [
                                v is not None and np.isfinite(float(v))
                                for v in ice_sims
                            ]
                        ).sum()
                    ),
                    "n_short": len(short),
                }
            )

        order_v9, _ = v9_order(short, fp_rep, ice_sims)
        order_full = full_order(full_keys, fp_all)
        final = order_v9 if route == "v9" else order_full

        _base_arr = np.asarray([float(fp_rep[k]) for k in short], dtype=float)
        _f_arr = zscore(_base_arr) + zscore(list(ice_sims))
        _finite_arr = np.asarray(
            [v is not None and np.isfinite(float(v)) for v in ice_sims],
            dtype=bool,
        )
        _pos = {k: i for i, k in enumerate(short)}
        v9_top25_scored = []
        for k in order_v9[:25]:
            i = _pos[k]
            v = ice_sims[i]
            v9_top25_scored.append(
                {
                    "canon": k,
                    "fp_representative": float(fp_rep[k]),
                    "ice_fused_merge": (None if v is None else float(v)),
                    "fused_zsum": (
                        float(_f_arr[i]) if bool(_finite_arr[i]) else None
                    ),
                }
            )
        all_mean_full_top25_scored = [
            {"canon": k, "fp_all_mean": float(fp_all[k])}
            for k in order_full[:25]
        ]
        final_top25_scored = (
            v9_top25_scored if route == "v9" else all_mean_full_top25_scored
        )

        # 唯一一次标签评分: truth仅本步用.
        truth = m["truth_canon"]
        rec = {
            "key": key,
            "n_spectra": rrow["n_spectra"],
            "modes": rrow["modes"],
            "adduct_counts": rrow["adduct_counts"],
            "instruments": rrow["instruments"],
            "n_window": len(full_keys),
            "n_frozen100": len(short),
            "metadata_has_supported_record": meta_supported,
            "observed_has_ice": observed,
            "route_match_observed": bool(meta_supported == observed),
            "route": "retain_v9" if route == "v9" else "all_mean_full_window",
            "ice_multi_hit_canon_n": nmulti,
            "ice_miss_canon_n": nmiss,
            "v9_top25": order_v9[:25],
            "v9_top25_scores": v9_top25_scored,
            "all_mean_full_top25": order_full[:25],
            "all_mean_full_top25_scores": all_mean_full_top25_scored,
            "final_top25": final[:25],
            "final_top25_scores": final_top25_scored,
            "final_full_len": len(final),
            "truth_canon": truth,
            "v9_rr": rr_of(order_v9, truth),
            "all_mean_full_rr": rr_of(order_full, truth),
            "final_rr": rr_of(final, truth),
            "v9_top1": order_v9[0] if order_v9 else None,
            "final_top1": final[0] if final else None,
        }
        per.append(rec)

    # 停损: mismatch>0 且原因未查明即停. 此处先如实记录; 若非空则抛错停.
    if mismatches:
        raise RuntimeError(
            f"metadata路由与observed-ICE mismatch {len(mismatches)} 例, 按协议停: "
            + json.dumps(mismatches[:5])
        )

    comp = summarize_final_vs_v9(per)

    shas = {}
    for p in [
        frozen_cfg,
        inputs_path,
        cand_path,
        routing_path,
        split_path,
        split_meta_path,
        mean_path,
        mean_meta_path,
        tar_path,
    ]:
        shas[str(p.relative_to(ROOT))] = sha_bytes(p)

    wall = datetime.now(timezone.utc).astimezone().isoformat()
    doc = {
        "bundle": "G1保留250一次性确认评分 (CPU-only; 该集唯一一次动用标签, 跑完即转已用验证)",
        "frozen_rule": "009-conditional frozen v1 (见 artifacts/runs/009-conditional/frozen_config.json)",
        "seed": SEED,
        "bootstrap_draws": DRAWS,
        "wall_clock_local": wall,
        "seconds_scoring": round(time.monotonic() - t0, 2),
        "environment": {"device": "local M5 CPU-only, 无GPU forward/训练, 无push/submit"},
        "input_sha256": shas,
        "routing_reused": {
            "n": 250,
            "supported_n": 232,
            "unsupported_n": 18,
            "note": "复用routing.json既有路由, 不重路由; 计数已核对",
        },
        "ice_assert": {
            "split_fused_merge_finite": f"{n_fin}/{n_tot}",
            "mean_only_fused": True,
            "canon_miss_total": miss_total,
            "canon_multi_hit_total": multi_hit_total,
            "canon_multi_hit_mols": multi_hit_mols,
        },
        "mismatch_n": len(mismatches),
        "mismatches": mismatches,
        "summary_final_vs_v9": comp,
        "per_molecule": per,
    }
    out_path.write_text(json.dumps(doc, ensure_ascii=False))
    print(
        json.dumps(
            {
                "n": len(per),
                "mismatch_n": len(mismatches),
                "mrr_v9": comp["mrr_a"],
                "mrr_final": comp["mrr_b"],
                "delta": comp["delta_mrr"],
                "improved/worsened/tied": [
                    comp["improved"],
                    comp["worsened"],
                    comp["tied"],
                ],
                "top1_v9->final": [comp["top1_a"], comp["top1_b"]],
                "top25_v9->final": [comp["top25_a"], comp["top25_b"]],
                "wrong_to_right": comp["top1_wrong_to_right"],
                "right_to_wrong": comp["top1_right_to_wrong"],
                "gained": comp["top25_gained_keys"],
                "lost": comp["top25_lost_keys"],
                "out": str(out_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

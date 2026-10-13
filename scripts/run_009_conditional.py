"""009-conditional frozen pipeline: label-blind routing, no training, CPU-only, no heldout.

Frozen rule (single, label-blind):
  SUPPORTED (metadata_has_supported_record True) -> v9 (representative FP + ICE fused_merge).
  UNSUPPORTED (metadata_has_supported_record False) -> all_mean FP over full original window.

metadata_has_supported_record = exists query record with mode positive
  AND adduct in {[M+H]+, [M+Na]+}. Uses only spectra metadata
  (mode/adduct/instrument/CE), never truth labels.

Choice all_mean vs adduct_balanced was fixed WITHOUT using development MRR:
  009 plan item 18 pre-fixed three ways (representative / all_mean / adduct_balanced).
  Freeze to all_mean because it is the minimal-assumption aggregator
  (equal weight per spectrum, no extra (mode,adduct)-group hyperparameter).
  adduct_balanced assumes equal information per adduct group, an extra inductive
  bias with no a priori justification. For single-spectrum molecules both coincide,
  so all_mean is the simpler generalization. This preserves continuity with the
  full-window diagnostic definition (which used all_mean) while the routing signal
  is now the ICE-support contract. Development MRR deltas were NOT used to pick.

Original conditional_diagnostic ([M+H]+/[M+Na]+ routing, all_mean full window)
  is POST-HOC (rule selected after seeing NP/G1 results) and is NOT promoted.
  This script freezes a new single rule by prior reasoning and reports NP/G1
  numbers as DIAGNOSTIC ONLY: no thresholds added, no rule changes from errors.

D2 line stays sealed: no residual training. D2 coverage summary is read-only
  for the metadata==observed equivalence note; no new fits.

CPU-only: reuses cached FPNet logits/scores and ICE caches. No model forward,
  no push/submit, no GPU. R3-50 routing is metadata-only; full conditional MRR
  on R3-50 is infeasible CPU-only (no ICE forward cache) and is NOT scored here.

G1 heldout 250: never read, never scored here. Standby protocol is written
  separately and execution waits for user approval.

Manual research script, not a test. Run manually:
  /opt/homebrew/bin/python3.11 scripts/run_009_conditional.py
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261012
DRAWS = 20000
SUPPORTED_ADDUCTS = {"[M+H]+", "[M+Na]+"}
POSITIVE_MODES = {"positive", "pos", "+", "1"}

FROZEN_RULE_TEXT = (
    "009-conditional frozen v1: per molecule, compute metadata_has_supported_record "
    "from query spectra metadata only (mode positive AND adduct in {[M+H]+,[M+Na]+}). "
    "If True -> SUPPORTED branch: v9 representative-FP + ICE fused_merge over frozen "
    "top100 (z(base)+z(sims), scored-first then unscored-by-base, Top25 output). "
    "If False -> UNSUPPORTED branch: all-spectra logits-mean FP (all_mean) ranking over "
    "full original mass window with original representative SMILES, order (-score, canon), "
    "Top25 output. No new candidates, no reshortlisting, no ICE in unsupported branch. "
    "Observed ICE finiteness is audit-only (mismatch logged, never flips route). "
    "Unsupported FP choice fixed to all_mean by prior parsimony (009 plan item 18), "
    "not by development MRR. No thresholds."
)


def sha_bytes(path: Path) -> dict:
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
            n += len(block)
    return {"bytes": n, "sha256": h.hexdigest()}


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


def metadata_supported_f1g1(spectra) -> bool:
    for s in spectra:
        mode = str(s.get("mode", "")).lower()
        if mode in POSITIVE_MODES and s.get("adduct") in SUPPORTED_ADDUCTS:
            return True
    return False


def metadata_supported_r3(spectra) -> bool:
    for s in spectra:
        mode = str(s.get("ionization_mode", "")).lower()
        if mode in POSITIVE_MODES and s.get("adduct") in SUPPORTED_ADDUCTS:
            return True
    return False


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


def fixed100_fp_order(short, fp_map):
    return sorted(short, key=lambda k: (-float(fp_map[k]), str(k)))


def analyze_domain(domain: str, inputs_path: Path, scores_path: Path):
    molecules = json.loads(inputs_path.read_text())["molecules"]
    cand = json.loads(scores_path.read_text())
    per = []
    for m in molecules:
        key = m["key"]
        # --- LABEL-BLIND ROUTING: spectra metadata only, truth never touched here.
        spectra = m["spectra"]
        meta_supported = metadata_supported_f1g1(spectra)
        entry = cand[key]
        full_keys = list(entry["full_keys"])
        fp_rep = dict(entry["fp"]["representative"])
        fp_all = dict(entry["fp"]["all_mean"])
        short = list(entry["frozen_top100"])
        ice_sims = list(entry["ice_sim"])
        observed = bool(
            np.asarray(
                [v is not None and np.isfinite(float(v)) for v in ice_sims]
            ).any()
        )
        route = "v9" if meta_supported else "all_mean_full"
        # --- Branch orders AND branch scores from cached scores only, no forward.
        order_v9, _ = v9_order(short, fp_rep, ice_sims)
        order_full = full_order(full_keys, fp_all)
        order_fixed100 = fixed100_fp_order(short, fp_all)
        final = order_v9 if route == "v9" else order_full
        # Branch scores for Top25 (cached values only, no recomputation).
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
        if route == "v9":
            final_top25_scored = v9_top25_scored
        else:
            final_top25_scored = all_mean_full_top25_scored
        # --- Audit fields (no labels).
        modes = sorted({str(s.get("mode")) for s in spectra})
        adducts = dict(collections.Counter(str(s.get("adduct")) for s in spectra))
        instruments = sorted({str(s.get("instrument")) for s in spectra})
        rec = {
            "key": key,
            "n_spectra": len(spectra),
            "modes": modes,
            "adduct_counts": adducts,
            "instruments": instruments,
            "n_window": len(full_keys),
            "n_frozen100": len(short),
            "metadata_has_supported_record": bool(meta_supported),
            "observed_has_ice": bool(observed),
            "route_match_observed": bool(meta_supported == observed),
            "route": "retain_v9" if route == "v9" else "all_mean_full_window",
            "v9_top25": order_v9[:25],
            "v9_top25_scores": v9_top25_scored,
            "all_mean_full_top25": order_full[:25],
            "all_mean_full_top25_scores": all_mean_full_top25_scored,
            "all_mean_fixed100_top25": order_fixed100[:25],
            "all_mean_full_vs_fixed100_top25_equal": bool(
                order_full[:25] == order_fixed100[:25]
                or (
                    len(full_keys) <= 100
                    and order_full == order_fixed100
                )
                or (
                    # When window >100, Top25 equality is the frozen-output criterion.
                    order_full[:25] == order_fixed100[:25]
                )
            ),
            "final_top25": final[:25],
            "final_top25_scores": final_top25_scored,
            "final_full_len": len(final),
            # Truth is attached BELOW for offline diagnostic scoring only.
            "truth_for_offline_scoring_only": m["truth_canon"],
        }
        # --- DIAGNOSTIC ONLY: use truth to score frozen outputs. Never routes.
        truth = m["truth_canon"]
        rec["v9_rr"] = rr_of(order_v9, truth)
        rec["all_mean_full_rr"] = rr_of(order_full, truth)
        rec["all_mean_fixed100_rr"] = rr_of(order_fixed100, truth)
        rec["final_rr"] = rr_of(final, truth)
        rec["v9_top1"] = order_v9[0] if order_v9 else None
        rec["final_top1"] = final[0] if final else None
        per.append(rec)
    return per


def summarize_final_vs_v9(per):
    rows = [
        {"v9": r["v9_rr"], "final": r["final_rr"]} for r in per
    ]
    comp = compare(
        [{"v9": x["v9"], "final": x["final"]} for x in rows],
        "v9",
        "final",
        SEED,
        DRAWS,
    )
    # Top25 gain/loss lists (diagnostic only, no rule change).
    gained = [r["key"] for r in per if r["final_rr"] > 0 and r["v9_rr"] == 0]
    lost = [r["key"] for r in per if r["final_rr"] == 0 and r["v9_rr"] > 0]
    comp["top25_gained_keys"] = gained
    comp["top25_lost_keys"] = lost
    comp["top25_gained_n"] = len(gained)
    comp["top25_lost_n"] = len(lost)
    # Route strata (diagnostic only).
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/009-conditional"
    )
    args = ap.parse_args()
    out = args.output_dir
    if (out / "summary.json").exists():
        raise FileExistsError(
            f"{out / 'summary.json'} exists; refuse overwrite (use a new --output-dir)"
        )
    t0 = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)

    inputs_np = ROOT / "artifacts/f1-bundle/inputs.json"
    inputs_g1 = ROOT / "artifacts/g1-validation/inputs.json"
    scores_np = ROOT / "artifacts/009-multispectrum/np_candidate_scores.json"
    scores_g1 = ROOT / "artifacts/009-multispectrum/g1_candidate_scores.json"
    d2_summary = ROOT / "artifacts/009-d2-coverage/summary.json"
    r3_obs = ROOT / "artifacts/007-r3/observations_50.json"
    r3_logits = ROOT / "artifacts/009-expansion/r3_50_logits.npy"
    r3_scores = ROOT / "artifacts/009-expansion/r3_50_full_candidate_scores.json"
    cond_diag = ROOT / "artifacts/009-multispectrum/conditional_diagnostic.json"

    shas = {}
    for p in [inputs_np, inputs_g1, scores_np, scores_g1, d2_summary, r3_obs]:
        shas[str(p.relative_to(ROOT))] = sha_bytes(p)
    # R3 FP cache existence (no GPU forward here; reuse only).
    r3_cache = {
        "r3_50_logits.npy_exists": r3_logits.exists(),
        "r3_50_full_candidate_scores.json_exists": r3_scores.exists(),
    }
    if r3_logits.exists():
        shas["artifacts/009-expansion/r3_50_logits.npy"] = sha_bytes(r3_logits)
    if r3_scores.exists():
        shas["artifacts/009-expansion/r3_50_full_candidate_scores.json"] = sha_bytes(
            r3_scores
        )
    shas["artifacts/009-multispectrum/conditional_diagnostic.json"] = sha_bytes(
        cond_diag
    )

    per_np = analyze_domain("np", inputs_np, scores_np)
    per_g1 = analyze_domain("g1", inputs_g1, scores_g1)

    # Consistency vs 009 post-hoc diagnostic (frozen rule coincides on dev
    # because metadata route == observed coverage; verify, do not tune).
    diag = json.loads(cond_diag.read_text())
    comp_np = summarize_final_vs_v9(per_np)
    comp_g1 = summarize_final_vs_v9(per_g1)

    def check(domain, comp, expected):
        return {
            "expected_conditional_mrr": float(expected["conditional_mrr"]),
            "recomputed_conditional_mrr": float(comp["mrr_b"]),
            "abs_diff": abs(float(comp["mrr_b"] - expected["conditional_mrr"])),
            "expected_A_mrr": float(expected["A_mrr"]),
            "recomputed_A_mrr": float(comp["mrr_a"]),
            "expected_delta": float(expected["delta_mrr"]),
            "recomputed_delta": float(comp["delta_mrr"]),
            "top1_match": bool(
                expected["conditional_top1_n"] == comp["top1_b"]
                and expected["A_top1_n"] == comp["top1_a"]
            ),
            "top25_match": bool(
                expected["conditional_top25_n"] == comp["top25_b"]
                and expected["A_top25_n"] == comp["top25_a"]
            ),
        }

    consistency = {
        "np": check("np", comp_np, diag["domains"]["np"]["summary"]),
        "g1": check("g1", comp_g1, diag["domains"]["g1"]["summary"]),
    }

    mismatch_np = [r["key"] for r in per_np if not r["route_match_observed"]]
    mismatch_g1 = [r["key"] for r in per_g1 if not r["route_match_observed"]]

    # R3-50 routing (metadata only; labels never used for routing).
    r3_obs_data = json.loads(r3_obs.read_text())["molecules"]
    r3_rows = []
    for m in r3_obs_data:
        supp = metadata_supported_r3(m["spectra"])
        r3_rows.append(
            {
                "canonical_key_for_audit_only": m["canonical_key"],
                "class_AB_for_audit_only": m["class_AB"],
                "n_spectra": int(m["n_spectra"]),
                "ion_modes": dict(
                    collections.Counter(
                        str(s.get("ionization_mode")) for s in m["spectra"]
                    )
                ),
                "adducts": dict(
                    collections.Counter(str(s.get("adduct")) for s in m["spectra"])
                ),
                "instrument_families": dict(
                    collections.Counter(
                        str(s.get("instrument_family")) for s in m["spectra"]
                    )
                ),
                "metadata_has_supported_record": bool(supp),
                "frozen_route": "retain_v9" if supp else "all_mean_full_window",
            }
        )
    r3_summary = {
        "n": len(r3_rows),
        "supported_n": sum(1 for r in r3_rows if r["frozen_route"] == "retain_v9"),
        "unsupported_n": sum(
            1 for r in r3_rows if r["frozen_route"] != "retain_v9"
        ),
        "by_class": {
            f"{cls}|{route}": n
            for (cls, route), n in collections.Counter(
                (r["class_AB_for_audit_only"], r["frozen_route"]) for r in r3_rows
            ).items()
        },
        "feasibility": (
            "INFEASIBLE for full conditional MRR CPU-only: R3-50 has FP logits "
            "cache (59 spectra) and FP full-window scores, but no ICE forward "
            "cache for the SUPPORTED branch (16 A + 7 B would need GL/ICE per-candidate "
            "GPU forward). 4 B have unknown-adduct mass-uncertain windows. "
            "This run outputs routing only; no R3 MRR is scored and no rule is changed."
        ),
        "existing_cache": r3_cache,
    }

    frozen_config = {
        "version": "009-conditional frozen v1",
        "rule": FROZEN_RULE_TEXT,
        "supported_adducts": sorted(SUPPORTED_ADDUCTS),
        "positive_modes": sorted(POSITIVE_MODES),
        "unsupported_fp": "all_mean (equal weight per spectrum; 009 plan item 18 one of three)",
        "unsupported_fp_choice_basis": (
            "Prior parsimony, NOT development MRR. all_mean has no adduct-group "
            "hyperparameter; adduct_balanced assumes equal information per "
            "(mode,adduct) group without a priori justification. Single-spectrum "
            "molecules coincide under both. Continuity with full-window diagnostic "
            "definition, but routing signal is now the ICE-support contract."
        ),
        "supported_branch": (
            "v9 representative-FP + ICE fused_merge over frozen top100; "
            "z(base)+z(sims) ddof1, scored-first then unscored-by-base; Top25."
        ),
        "unsupported_branch": (
            "all_mean FP over full original mass window with original rep SMILES; "
            "order (-score, canon); Top25. No ICE, no new candidates."
        ),
        "observed_ice_role": "audit-only mismatch check; never flips metadata route",
        "seed": SEED,
        "bootstrap_draws": DRAWS,
        "post_hoc_cut": (
            "artifacts/009-multispectrum/conditional_diagnostic.json is "
            "POST_HOC_DIAGNOSTIC_NOT_DEPLOYABLE (rule picked after seeing dev "
            "results). It is NOT promoted. This frozen v1 is a separate prior "
            "freeze; dev numbers below are diagnostic only."
        ),
    }

    (out / "frozen_config.json").write_text(
        json.dumps(frozen_config, ensure_ascii=False, indent=2) + "\n"
    )
    (out / "per_molecule_np.json").write_text(
        json.dumps(per_np, ensure_ascii=False, indent=2) + "\n"
    )
    (out / "per_molecule_g1.json").write_text(
        json.dumps(per_g1, ensure_ascii=False, indent=2) + "\n"
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "bootstrap_draws": DRAWS,
        "device": "local CPU; cached FPNet logits + ICE caches reused; no forward/training/GPU",
        "frozen_rule": FROZEN_RULE_TEXT,
        "domains": {
            "np": {
                "n": len(per_np),
                "routed_supported": sum(
                    1 for r in per_np if r["route"] == "retain_v9"
                ),
                "routed_unsupported": sum(
                    1 for r in per_np if r["route"] != "retain_v9"
                ),
                "route_observed_mismatch_n": len(mismatch_np),
                "route_observed_mismatched_keys": mismatch_np,
                "final_vs_v9": comp_np,
                "consistency_vs_conditional_diagnostic": consistency["np"],
                "all_mean_full_vs_fixed100_top25_equal_n": sum(
                    1
                    for r in per_np
                    if r["route"] != "retain_v9"
                    and r["all_mean_full_vs_fixed100_top25_equal"]
                ),
            },
            "g1": {
                "n": len(per_g1),
                "routed_supported": sum(
                    1 for r in per_g1 if r["route"] == "retain_v9"
                ),
                "routed_unsupported": sum(
                    1 for r in per_g1 if r["route"] != "retain_v9"
                ),
                "route_observed_mismatch_n": len(mismatch_g1),
                "route_observed_mismatched_keys": mismatch_g1,
                "final_vs_v9": comp_g1,
                "consistency_vs_conditional_diagnostic": consistency["g1"],
                "all_mean_full_vs_fixed100_top25_equal_n": sum(
                    1
                    for r in per_g1
                    if r["route"] != "retain_v9"
                    and r["all_mean_full_vs_fixed100_top25_equal"]
                ),
            },
        },
        "diagnostic_only_warning": (
            "NP250/G1-research250 numbers are DIAGNOSTIC ONLY. No thresholds added, "
            "no rule changes from errors. Do not average across domains into a "
            "hidden-proportion total."
        ),
        "d2_sealed_note": (
            "D2 residual training NOT reopened. Coverage conclusion cited read-only "
            "from artifacts/009-d2-coverage/summary.json: NP scored_n_zero 34 with "
            "pressure delta near zero and no Top1 loss; 5 NP pressure Top1 losses "
            "all in ICE-covered domain with same-formula cosine>=0.875 tops."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    (out / "r3_routing.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "summary": r3_summary,
                "rows": r3_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    wall = time.monotonic() - t0
    run_meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "bootstrap_draws": DRAWS,
        "device": "local CPU; no GPU forward/training/push/submit",
        "wall_seconds": wall,
        "inputs_sha256": shas,
        "outputs": [
            "frozen_config.json",
            "per_molecule_np.json",
            "per_molecule_g1.json",
            "summary.json",
            "r3_routing.json",
            "run_meta.json",
            "protocol_g1_heldout_standby.md",
        ],
        "script": "scripts/run_009_conditional.py",
    }
    run_meta["script_sha256"] = sha_bytes(Path(__file__))["sha256"]
    run_meta["frozen_config_sha256"] = sha_bytes(out / "frozen_config.json")[
        "sha256"
    ]
    (out / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n"
    )

    protocol = f"""# G1-heldout250 一次性确认待命协议（未执行，需用户批准）

状态：待命。本轮未读 heldout 任何标签/谱，未执行确认。

冻结配置：`artifacts/009-conditional/frozen_config.json`
SHA256：`{run_meta["frozen_config_sha256"]}`
脚本：`scripts/run_009_conditional.py`（本次已用于 NP250/G1-research250 诊断与 R3-50 路由）
种子：`{SEED}`，bootstrap：`{DRAWS}`（仅诊断 CI，不选规则）

输入清单（执行时逐项记录 bytes+SHA256，缺一即停）：
- heldout 查询包（谱 metadata：模式/adduct/仪器/CE；谱峰；质量窗输入），路径待执行时由用户指定并记录 SHA
- 候选池：`artifacts/pool-train/*` + `artifacts/005/coco/*`（union 712199 mmap，只读，不重建）
- FP 权重：`artifacts/001-baseline/weights/fp_single_s2.pt`（CPU 前向；如已有 heldout logits 缓存则复用，无缓存则 CPU 前向，禁 GPU 任务外的新训练）
- ICE 缓存：heldout 候选需完整 ICE forward（支持条件分子的 v9 分支必需）；无缓存则先补 forward 并记录成本，缺 ICE 不得分、不回退造 0 分
- 评分脚本：同一冻结脚本的 heldout 模式（路由仍只用 metadata+ICE 支持判定，标签仅一次性评分）

评分脚本行为（冻结）：
- 路由与分支定义与 `frozen_config.json` 逐字一致；不支持分支为 `all_mean` 全窗 Top25
- observed-ICE 仅审计 mismatch，不翻转 metadata 路由
- 输出逐分子路由、两分支 Top25、最终排序与 MRR/Top1/Top25/改对改错/Top25 得失
- 不得据 heldout 结果加阈值、改权重、改分组或重选 `all_mean`/`adduct_balanced`

停损条件（任一成立即停，不部署、不继续调参）：
- 输入 SHA 缺失或与冻结配置声明不一致
- metadata 路由与 observed-ICE mismatch > 0 且原因未查明（区分不支持条件 vs 解析/预测/预算失败）
- heldout 需新起 GPU 任务但用户未批准，或 9h/离线/无互联网的最终提交约束不满足
- 任何试图用 heldout 选择规则、合成隐藏比例总分、或动用保留集做第二轮确认

不可逆声明：动用保留 250 做一次性确认后，该集即转为已用验证，不再是保留集。
本次仅给出协议，不执行、不读标签，留待用户批准。
"""
    (out / "protocol_g1_heldout_standby.md").write_text(protocol)
    print(
        json.dumps(
            {
                "np": comp_np,
                "g1": comp_g1,
                "consistency": consistency,
                "r3": r3_summary,
                "wall_seconds": wall,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    print("saved", str(out), flush=True)


if __name__ == "__main__":
    main()

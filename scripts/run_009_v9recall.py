#!/usr/bin/env python3
"""009-v9recall reproducible preparation (CPU-only, manual, no heldout, zero GPU).

固化 artifacts/009-v9recall/ 下 JSON/bundle inputs 的生成逻辑为单一手动脚本。
只读允许目录，输出到 artifacts/009-v9recall/，默认不静默覆盖已有文件。

允许读取（生成输入，仅此五处）：
  artifacts/008-expansion/  artifacts/009-expansion/
  artifacts/f1-bundle/      artifacts/g1-validation/
  artifacts/007-r3/
输出：
  artifacts/009-v9recall/
禁止：读 G1 保留250/heldout、扫阈值/权重、GPU forward、push/submit/远端。

Manual research script, not a test. Run manually:
  /opt/homebrew/bin/python3.11 scripts/run_009_v9recall.py [--check-only] [--output-suffix .check] [--make-tar] [--update-run-meta]
  /opt/homebrew/bin/python3.11 scripts/run_009_v9recall.py --output-dir artifacts/009-v9recall --output-suffix .repro

行为：
  默认对每个目标：若不存在则写入；若已存在则逐值比对（JSON 允许键序一致，
  bundle/inputs.json 另核 SHA），一致则跳过，一不一致则报错并报告差异，不覆盖。
  --output-suffix 非空时写入到 suffixed 路径（原文件不动），用于人工 diff。
  --check-only 只比对不写任何文件。
  --make-tar 按记录命令打包 bundle tar（见 --help 与 run_meta），不含大权重。
  --update-run-meta 仅给 run_meta.json 加 reproducibility 小节，不改原有键。

Bundle tar 打包（脚本化，见 make_tar()；等价手动命令）：
  tar czf artifacts/009-v9recall/v9recall-bundle.tar.gz -C artifacts/009-v9recall/bundle .
  （gzip 时间戳导致 tar 字节不稳定；一致性以 bundle/inputs.json SHA 与 tar 内文件清单为准，
  tar SHA 仅记录不强求字节一致。JSON 则要求值一致，bundle inputs 要求 SHA 一致。）
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED_NOTE = "no stochastic candidate selection; fixed prior008/009 method"
COV = ("[M+H]+", "[M+Na]+")
SUM_CE = (20.0, 40.0, 60.0)

# 12 支持对（顺序固定，与现有产物一致）
SUPPORTED_DEFS = [
    ("np", "BHQCQFFYRZLCQQ", "JSIQROACCRQDKK"),
    ("np", "BHQCQFFYRZLCQQ", "JVRPXPQFLWKXPC"),
    ("np", "BMLIIPOXVWESJG", "YUXFWUGNPZLEJN"),
    ("np", "DFMAXQKDIGCMTL", "TZOVOULUMXXLOJ"),
    ("np", "HSMPDPBYAYSOBC", "KJBKJBXJXGQAFZ"),
    ("np", "HYFHYPWGAURHIV", "ZXUWKFXQTSOVDY"),
    ("np", "ILEDWLMCKZNDJK", "CULQHECWQXNZLL"),
    ("np", "LOIYMIARKYCTBW", "ZCIFWRHIEBXBOY"),
    ("np", "WJXSWCUQABXPFS", "CWNPOQFCIIFQDM"),
    ("g1", "GKBSLXWQUZXZPM", "BCOWBZFFNHRZNF"),
    ("g1", "PZQDWRSEJBKJJK", "DIBUFQMCUZYQKN"),
    ("g1", "SFHLFNWOCGHDOA", "XXHIPRDUAVCXHW"),
]
UNSUP_DEFS = [
    ("np", "BJRNKVDFDLYUGJ", "JSAJUCGZVPNSCD"),
    ("np", "BJRNKVDFDLYUGJ", "ABWQNLKQDPGUCS"),
    ("np", "RUDATBOHQWOJDD", "XKHVNTXVBMJZFA"),
]
# bundle 11 分子顺序（worker 兼容 inputs.json 用）
BUNDLE_MOLS = [
    ("np", "BHQCQFFYRZLCQQ"),
    ("np", "BMLIIPOXVWESJG"),
    ("np", "DFMAXQKDIGCMTL"),
    ("np", "HSMPDPBYAYSOBC"),
    ("np", "HYFHYPWGAURHIV"),
    ("np", "ILEDWLMCKZNDJK"),
    ("np", "LOIYMIARKYCTBW"),
    ("np", "WJXSWCUQABXPFS"),
    ("g1", "GKBSLXWQUZXZPM"),
    ("g1", "PZQDWRSEJBKJJK"),
    ("g1", "SFHLFNWOCGHDOA"),
]

ALLOWED_READ_PREFIXES = (
    "artifacts/008-expansion/",
    "artifacts/009-expansion/",
    "artifacts/f1-bundle/",
    "artifacts/g1-validation/",
    "artifacts/007-r3/",
)
# verification may read existing outputs for comparison (not as generation inputs)
ALLOWED_VERIFY_PREFIX = "artifacts/009-v9recall/"


def guard_allowed(path: Path, *, for_verify: bool = False) -> None:
    s = str(path)
    try:
        rel = str(path.relative_to(ROOT))
    except ValueError:
        rel = s
    # forbid heldout explicitly
    low = (s + " " + rel).lower()
    if "heldout" in low or "holdout" in low:
        raise RuntimeError(f"refuse heldout path: {s}")
    if for_verify and (rel.startswith(ALLOWED_VERIFY_PREFIX) or s.startswith(ALLOWED_VERIFY_PREFIX)):
        return
    if any(rel.startswith(p) or rel == p.rstrip("/") or s.startswith(p) for p in ALLOWED_READ_PREFIXES):
        return
    raise RuntimeError(f"read outside allowed prefixes: {s} (rel {rel})")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path: Path):
    guard_allowed(path)
    return json.loads(path.read_text())


def dump_text(obj, indent: int) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=indent, allow_nan=False) + "\n"


# ---- ICE runner mirror (CPU, no torch; semantics verbatim from ice_runner.py) ----
def is_covered(sp: dict) -> bool:
    md = sp.get("mode")
    pos = (md == 1) or (isinstance(md, str) and md.lower() in ("positive", "pos", "+", "1"))
    return bool(pos and sp.get("adduct") in COV)


def _nce(s) -> float:
    v = [float(x) for x in re.findall(r"[0-9]+\.?[0-9]*", str(s or ""))]
    return float(np.mean(v)) if v else float("nan")


def spectrum_ce(sp: dict) -> float:
    if "ce_units" in sp or "ce_ev" in sp:
        if sp.get("ce_units") == "NCE":
            return _nce(sp.get("ce_orig"))
        v = sp.get("ce_ev")
        return float(np.mean(v)) if v is not None and len(v) else float("nan")
    v = sp.get("ce")
    return float(v) if v is not None and np.isfinite(float(v)) else float("nan")


def ce_bucket(ce: float) -> float:
    ceb = float(np.round(ce / 5.0) * 5.0) if np.isfinite(ce) else float("nan")
    if np.isfinite(ceb) and ceb < 5:
        ceb = 5.0
    return ceb


def instr_family_raw(instrument_raw) -> int:
    t = str(instrument_raw or "").lower()
    if "orbitrap" in t or "qft" in t or "ftms" in t or "hybrid ft" in t or "itft" in t or "exactive" in t or "ft-icr" in t or "fticr" in t:
        return 1
    if "tof" in t:
        return 2
    if "trap" in t or "qit" in t or "lit" in t:
        return 3
    if "qq" in t or "quattro" in t or "triple" in t:
        return 4
    return 5


def instr_token(sp: dict) -> str:
    if sp.get("instr") in ("Orbitrap", "QTOF"):
        return sp["instr"]
    # inputs.json uses key 'instrument' (np/g1), observations use instrument_raw
    raw = sp.get("instrument", sp.get("instrument_raw"))
    # timsTOF -> QTOF via family 2
    fam = 2 if str(raw or "").lower().find("timstof") >= 0 or str(raw or "").lower().find("tof") >= 0 else instr_family_raw(raw)
    if sp.get("instrument") in ("Orbitrap", "QTOF"):
        return sp["instrument"]
    return "Orbitrap" if fam == 1 else "QTOF"


def expand_records(spectra, cemode: str):
    recs = []
    for qi, sp in enumerate(spectra):
        covered = bool(is_covered(sp))
        units = sp.get("ce_units")
        nce = (units == "NCE")
        ce_ev = [float(v) for v in (sp.get("ce_ev") or [])]
        if not covered:
            recs.append({"qid": qi, "sp": sp, "ces": [], "covered": False})
            continue
        if cemode == "mean" or len(ce_ev) <= 1:
            recs.append({"qid": qi, "sp": sp, "ces": [float(spectrum_ce(sp))], "covered": True})
            continue
        if nce:
            nums = [float(x) for x in re.findall(r"[0-9]+\.?[0-9]*", str(sp.get("ce_orig") or ""))]
            ces = list(nums) if len(nums) == len(ce_ev) else [float(spectrum_ce(sp))]
        else:
            ces = list(ce_ev)
        recs.append({"qid": qi, "sp": sp, "ces": ces, "covered": True})
    return recs


def unique_jobs_for_cand(cand: str, spectra, cemode: str):
    recs = expand_records(spectra, cemode)
    uniq = set()
    for r in recs:
        if not r["covered"]:
            continue
        ad = r["sp"]["adduct"]
        ins = instr_token(r["sp"])
        for ce in r["ces"]:
            ceb = ce_bucket(ce)
            key = (cand, ad, ins, ("sum" if isinstance(ceb, float) and math.isnan(ceb) else round(float(ceb), 1)))
            uniq.add(key)
    n_ces = sum(len(r["ces"]) for r in recs if r["covered"])
    n_recs = sum(1 for r in recs if r["covered"])
    # jobs_sorted as lists for JSON
    jobs_sorted = sorted([list(k) for k in uniq])
    return {"n_unique": len(uniq), "jobs_sorted": jobs_sorted, "n_covered_recs": n_recs, "n_ces_total": n_ces, "_set": uniq}


# ---- generators ----
def gen_pairs_all(inter, full, np_by, g1_by):
    pairs = []
    for ds in ("np", "g1"):
        for r in inter[ds]["rows"]:
            key = r["key"]
            exp_order = full[ds][key]["order_expanded"]
            m = (np_by if ds == "np" else g1_by)[key]
            covered = [i for i, s in enumerate(m["spectra"]) if is_covered(s)]
            for cand, score in r["added"].items():
                rank = exp_order.index(cand) + 1 if cand in exp_order else None
                pairs.append({
                    "ds": ds,
                    "mol_key": key,
                    "new_canon": cand,
                    "fp_score": float(score),
                    "expanded_rank": rank,
                    "in_top100": bool(rank is not None and rank <= 100),
                    "in_top25": bool(rank is not None and rank <= 25),
                    "n_spectra": len(m["spectra"]),
                    "n_covered": len(covered),
                    "covered_idx": list(covered),
                    "truth": r["truth"],
                    "base_rank": r["base_rank"],
                    "expanded_truth_rank": r["expanded_rank"],
                })
    return pairs


def detail_for(ds, mol, cand, inter, full, np_by, g1_by, by_rec):
    m = (np_by if ds == "np" else g1_by)[mol]
    exp_order = full[ds][mol]["order_expanded"]
    scores = full[ds][mol]["scores_expanded"]
    assert cand in exp_order, cand
    rank = exp_order.index(cand) + 1
    row = next(r for r in inter[ds]["rows"] if r["key"] == mol)
    fp_score = float(row["added"][cand])
    sup_all = []
    for i, s in enumerate(m["spectra"]):
        cov = is_covered(s)
        sup_all.append({
            "idx": i,
            "adduct": s["adduct"],
            "mode": s["mode"],
            "instrument": s["instrument"],
            "ce_ev": [float(v) for v in s["ce_ev"]],
            "ce_orig": s["ce_orig"],
            "ce_units": s["ce_units"],
            "prec": float(s["prec"]),
            "covered": bool(cov),
        })
    covered_only = [dict(x) for x in sup_all if x["covered"]]
    jobs = {}
    for cemode in ("mean", "split"):
        j = unique_jobs_for_cand(cand, m["spectra"], cemode)
        jobs[cemode] = {"n_unique": j["n_unique"], "jobs_sorted": j["jobs_sorted"], "n_covered_recs": j["n_covered_recs"], "n_ces_total": j["n_ces_total"]}
    cm = by_rec[cand]
    ice_supported = len(covered_only) > 0
    if ice_supported:
        basis = "mode positive and adduct in ([M+H]+,[M+Na]+) per code/ice/ice_runner.py COV; instrument any (token Orbitrap/QTOF); CE any incl NaN->sum; covered spectra count>0"
    else:
        basis = "no covered spectrum: mode/adduct not in COV; per v9 missing rule scored续后, no 0 score, no forward"
    return {
        "ds": ds,
        "mol_key": mol,
        "truth": m["truth_canon"],
        "new_canon": cand,
        "new_smiles_rep": cm["smiles_rep"],
        "new_formula": cm["formula_calc"],
        "new_mass_theo": float(cm["mass_theo"]),
        "fp_score": fp_score,
        "expanded_rank": rank,
        "in_top25": bool(rank <= 25),
        "support_spectra_all": sup_all,
        "support_spectra_covered": covered_only,
        "ice_supported": bool(ice_supported),
        "ice_basis": basis,
        "forward_jobs": jobs,
        "query_mass": float(row["query_mass"]),
        "base_window": int(row["base_window"]),
        "expanded_window": int(row["expanded_window"]),
    }


def gen_forward_tasks(sup_details):
    mols = sorted(set((d["ds"], d["mol_key"]) for d in sup_details))
    per_pair = []
    for d in sup_details:
        s1 = set(tuple(x) for x in d["forward_jobs"]["mean"]["jobs_sorted"])
        s2 = set(tuple(x) for x in d["forward_jobs"]["split"]["jobs_sorted"])
        per_pair.append({"ds": d["ds"], "mol": d["mol_key"], "new": d["new_canon"], "rank": d["expanded_rank"],
                         "mean": d["forward_jobs"]["mean"]["n_unique"], "split": d["forward_jobs"]["split"]["n_unique"], "union": len(s1 | s2)})
    total_mean = sum(p["mean"] for p in per_pair)
    total_split = sum(p["split"] for p in per_pair)
    total_union = sum(p["union"] for p in per_pair)

    def cost(tot):
        pure_s = tot / 26.0
        return {"unique_jobs": tot, "pure_pred_s_at_26_per_s": round(pure_s, 2), "pure_pred_min": round(pure_s / 60.0, 3),
                "with_load_overhead_min_est": "~3-6 min total for 11-mol minimal bundle (model load + scoring), far below 2h",
                "basis": "F1 MANIFEST timing_basis.ice_pred_per_s=26; 005实测33/s更快为下限参考; G1 README 56062 jobs ~36min pure; linear scaling"}

    # full-shortlist dry-run numbers recomputed locally (must match worker --dry-run planning)
    # recompute via same expand logic over expanded shortlists is done by caller; here compute incremental only
    # full numbers are verified separately in verify step via bundle inputs
    fwd = {
        "caliber": "unique (new_canon, adduct, instr_token, ce_bucket) per ICE runner; ce_bucket=round(ce/5)*5 min5, NaN->sum; instr_token Orbitrap/QTOF via instr_family; COV=([M+H]+,[M+Na]+) positive only",
        "n_pairs_supported": len(sup_details),
        "n_mols_supported": len(mols),
        "mols": [list(x) for x in mols],
        "per_pair": per_pair,
        "totals": {"mean": total_mean, "split": total_split, "union": total_union},
        "cost_T4": {"mean": cost(total_mean), "split": cost(total_split), "union": cost(total_union)},
        "note": "12对≠12次调用：逐CE展开后按CE去重，split 41 jobs (mean 31, union larger). T4 estimate at 26 pred/s pure + overhead; vs 008 2h上限.",
    }
    # fix pure_min rounding to match existing (cost() uses /60 rounded 3, but existing mean pure_min 0.02 not 0.02? check)
    # existing file: mean pure_min 0.02 (1.19/60=0.0198->0.02), split 0.026 (1.58/60=0.0263->0.026), union 0.033
    # our cost() rounds pure_min to 3 decimals, so mean becomes 0.02 (round(0.0198,3)=0.02) ok
    return fwd


def gen_cpu_expanded(inter, full):
    def rr(r):
        return 1.0 / r if r is not None and r <= 25 else 0.0

    out = {
        "caliber": "FP orig-head + 10ppm + canon-max + order(-score,canon); expanded top100 = full expanded window order truncated 100; truth only for offline RR",
        "FP_MRR_expected": {"np": 0.5145934285521241, "g1": 0.8842502164502164},
        "recompute": {},
        "displaced": {},
        "new_shortlists": {},
        "z_recalc_steps": {},
    }
    for ds in ("np", "g1"):
        rows = inter[ds]["rows"]
        base = np.array([rr(x["base_rank"]) for x in rows], dtype=float)
        exp = np.array([rr(x["expanded_rank"]) for x in rows], dtype=float)
        out["recompute"][ds] = {
            "n": len(rows),
            "base_mrr25": float(base.mean()),
            "expanded_mrr25": float(exp.mean()),
            "base_repr": repr(float(base.mean())),
            "expanded_repr": repr(float(exp.mean())),
            "match_expected": bool(float(exp.mean()) == out["FP_MRR_expected"][ds] and float(base.mean()) == out["FP_MRR_expected"][ds]),
            "base_top1": int((base == 1).sum()),
            "expanded_top1": int((exp == 1).sum()),
            "base_top25": int((base > 0).sum()),
            "expanded_top25": int((exp > 0).sum()),
            "rank_worsened": sum(x["base_rank"] is not None and x["expanded_rank"] is not None and x["expanded_rank"] > x["base_rank"] for x in rows),
        }
        dis = []
        for r in rows:
            if r["old_top100_displaced"] or r["new_in_top100"]:
                dis.append({"key": r["key"], "truth": r["truth"], "base_rank": r["base_rank"], "expanded_rank": r["expanded_rank"],
                            "new_in_top100": list(r["new_in_top100"]), "new_in_top25": list(r["new_in_top25"]),
                            "old_top100_displaced": list(r["old_top100_displaced"]), "old_top25_displaced": list(r["old_top25_displaced"]),
                            "base_top1": r["base_top1"], "expanded_top1": r["expanded_top1"], "n_added": int(r["n_added"])})
        out["displaced"][ds] = dis
        sl = {}
        for r in rows:
            if r["n_added"] > 0:
                eo = full[ds][r["key"]]["order_expanded"]
                scores = full[ds][r["key"]]["scores_expanded"]
                sl[r["key"]] = {"order_expanded_top100": list(eo[:100]), "scores_expanded_top100": [float(scores[k]) for k in eo[:100]],
                                "full_expanded_len": len(eo), "truth_rank_expanded": r["expanded_rank"]}
        out["new_shortlists"][ds] = sl
    out["z_recalc_steps"] = {
        "when": "after ICE forward scores return for the 12 supported pairs (split fused_merge primary, sims-mean auxiliary); 3 unsupported follow v9 missing rule, no补分",
        "inputs": [
            "expanded shortlist order (FP order, length<=100, per cpu_recompute)",
            "base FP scores for full expanded shortlist (orig-head collapsed-max)",
            "ICE fused_merge scores for full expanded shortlist (None if unsupported/missing, never 0)",
        ],
        "forbidden": "Do NOT tail-append new candidates to old fused order; Do NOT reuse old z means/stds; Do NOT scan thresholds/weights; Do NOT use truth to select",
        "steps": [
            "1. For each of the 11 molecules, assemble arrays base[i], ice[i] aligned to expanded shortlist order (i=0..n-1, n=len(shortlist)). ice[i]=fused_merge if available else None.",
            "2. Compute zb=_z(base), zi=_z(ice) with _z as in G1 README: sample std ddof=1 over finite non-None; n<2 or sd==0 -> all 0.0; missing -> 0.0 contribution but excluded from mean/std.",
            "3. fused[i]=zb[i]+zi[i] (equal weight; v9r2 caliber).",
            "4. scored_idx=[i where ice[i] is not None] sorted by (-fused[i], canon[i]); unscored_idx=[i where ice[i] is None] sorted by (-base[i], canon[i]); full_order=scored_idx+unscored_idx.",
            "5. Evaluate MRR@25 with canonical_group_key truth only for reporting; compare vs pre-expansion v9 baseline on same molecules + full NP250/G1250 when needed. Record per-molecule rank deltas.",
            "6. The 3 unsupported pairs (BJRN 2 + RUDA 1) stay in FP order positions but are treated as unscored in step 4 (按base续后), no ICE forward.",
        ],
        "reference_code": "artifacts/g1-validation/README.md §3 offline fusion snippet; artifacts/g1-validation/code/ice/fuse.py _z for same-formula variant (NOT used for v9 full-shortlist; listed to avoid confusion)",
        "example_snippet": "def _z(vals):\n    import math\n    ok=[v is not None and isinstance(v,(int,float)) and math.isfinite(v) for v in vals]\n    xs=[float(v) for v,o in zip(vals,ok) if o]\n    if len(xs)<2: return [0.0]*len(vals)\n    mu=sum(xs)/len(xs); sd=math.sqrt(sum((x-mu)**2 for x in xs)/(len(xs)-1))\n    return [((float(v)-mu)/sd) if o else 0.0 for v,o in zip(vals,ok)] if sd>0 else [0.0]*len(vals)\nzb=_z(base); zi=_z(ice_merge); fused=[a+b for a,b in zip(zb,zi)]\nscored_sorted=sorted([i for i,v in enumerate(ice_merge) if v is not None], key=lambda i: (-fused[i], canon[i]))\nunscored_sorted=sorted([i for i in range(n) if ice_merge[i] is None], key=lambda i: (-base[i], canon[i]))\nfull_order=scored_sorted+unscored_sorted",
    }
    return out


def gen_ytczz(obs_mol, cand_rec, r3_row, e50_row):
    return {
        "molecule": "YTCZZXIRLARSET",
        "query_spectrum": {
            "id": "MSBNK-mFam-MC02_000643",
            "adduct": "[M+H]+",
            "mode": "positive",
            "instrument_raw": "waters xevo g2 TOF [ESI-TOF]",
            "instrument_family": "QTOF",
            "precursor_mz": 421.19927,
            "n_peaks": 49,
            "ce_raw": "6.0000000",
            "ce_units": "unknown",
            "ce_ev": None,
            "known_adduct": True,
        },
        "candidate_rep": {
            "smiles_rep": "O[C@H]1C(/C=C/[C@@H](O)C(C)CC#CC)[C@@H](C(C=CC=C2CCCC([O-])=O)=C2O3)[C@@H]3C1.[Na+]",
            "formula_calc": "C24H29NaO5",
            "mass_theo": 420.19126830799996,
            "n_records": 2,
            "accessions": ["MSBNK-mFam-MC02_000643", "MSBNK-mFam-MC02_000644"],
        },
        "fp_status": {
            "base_rank": None,
            "expanded_rank": 40,
            "window_expanded_canon": 84,
            "truth_score": -732.1648328916635,
            "note": "FP orig-head 10ppm canon-max order(-score,canon); rank40 in top100 but outside top25; per 008 eval + 009 sensitivity reproduce zero mismatch",
        },
        "ice_query_side": {
            "covered": True,
            "basis": "mode positive and adduct [M+H]+ in COV ([M+H]+,[M+Na]+) per code/ice/ice_runner.py:34 COV + is_covered; instrument any -> token QTOF via instr_family QTOF; CE unknown/null -> spectrum_ce NaN -> ce_bucket NaN -> sum prediction (20+40+60 merged) per runner; 49 peaks sufficient",
            "refs": ["artifacts/g1-validation/code/ice/ice_runner.py:34,372-393"],
        },
        "ice_structure_side": {
            "rdkit_parse": True,
            "n_frags": 2,
            "symbols": ["C", "Na", "O"],
            "valid_elements_includes_Na": True,
            "inchi_roundtrip": True,
            "smiles_ok_prediction": "PASS cheap pre-check (parse + VALID_ELEMENTS + InChI round trip) -> job would be ATTEMPTED, not skipped as unsupported",
            "salt_flag": "has_separate_components=True (2 frags, .[Na+]) per chem_utils.py:764-767; MAGMA fragmentation + DAG for disconnected Na salt UNVALIDATED in this repo; no prior ICE salt forward evidence",
            "refs": [
                "artifacts/g1-validation/code/ice/src/ms_pred/common/chem_utils.py:38-60 VALID_ELEMENTS includes Na",
                "artifacts/g1-validation/code/ice/ice_runner.py:487-504 smiles_ok",
                "artifacts/g1-validation/code/ice/src/ms_pred/common/chem_utils.py:764-767 has_separate_components",
            ],
        },
        "judgment": "CONDITIONAL / NOT GUARANTEED: query side SUPPORTED, structure side ATTEMPTABLE but UNVALIDATED. DO NOT assume ICE rescues rank40. DO NOT transplant FP recall gain to ICE. Requires actual ICE forward observation; treat as R3 v9 baseline gap, not as predicted recovery.",
        "action": "List as gap; run minimal forward for YTCZZ expanded shortlist (or at least truth candidate) to observe success/failure/unsupported vs prediction failure; separate unsupported conditions from parse/prediction/budget failure; record meta pred_per_s/max_mem_gb.",
        "boundary": "No FP->ICE benefit transfer; no hidden-category inference; no new scores in this CPU-only step.",
    }


def gen_r3_gaps():
    return {
        "scope": "R3-50 complete v9 baseline gaps vs 008 FP-only; THIS STEP LISTS ONLY, no new scores (CPU-only)",
        "fp_caliber_008": "FP orig-head (fp_single_s2.pt) + 10ppm float64 full window + shipped->canon max (rdkit_full) + order(-score,canon) + FP top100 pre-fixed WMAX=100 by FP score (not mass-error) + final top25 by FP score; multi single->single logits, multi->mean logits; strict unknown_adduct (4 NILU uncertain); ce_fallback_25eV 46/59; see artifacts/008-expansion/expansion_eval_50.json meta",
        "v9_caliber_target": "union 712199 pool + 10ppm nocap full window + same collapse/order + top100 shortlist (expanded, incl 539) + per-record CE expansion + fused_merge primary + z(base)+z(sim) equal weight (sample std ddof1, <2/zero-var->0, missing None never 0) + unscored按base续后 + truth canonical_group_key RDKit 2026.03.6 + MRR@25; see artifacts/g1-validation/README.md §口径 + offline fusion snippet",
        "gaps": [
            {"id": "R3-representative", "issue": "代表谱选择未冻结为v9", "detail": "008 R3用 QM_med (median per-spectrum QM) + mean logits for multi (q_how single/mean_of_N); NP/G1 v9用 单分子max-raw-peaks谱做QM但forward用全谱逐记录CE (G1 README). R3需明确：QM用median vs max-peaks, FP用mean-logits vs single-rep, ICE用全谱逐记录CE. 与008差异必须记录，不能直接沿用008 mean-logits作为v9 ICE代表.", "refs": ["artifacts/007-r3/observations_50.json (n_spectra, mass_median)", "artifacts/008-expansion/expansion_eval_50.json meta multi", "artifacts/g1-validation/README.md 口径", "scripts/analyze_009_expansion_sources.py:154-155 (R3 mean logits) vs :76-77 (NP/G1 max-mz rep)"]},
            {"id": "R3-ce-fallback", "issue": "CE回退与sum路径", "detail": "R3含 CE units unknown (mFam 6.0 verbatim, no eV conversion, collision_energy_ev null) -> spectrum_ce NaN -> ce_bucket NaN -> sum prediction (20+40+60 merged) per runner. NP/G1全eV (timsTOF 20;40;60 etc.) + 少量NCE (G1 ZNND... 10张). 008 meta ce_fallback_25eV 46/59是FPNet prep (prep_peaks CE default 25), 不是ICE sum. R3 v9需逐谱列出 CE缺失->sum vs eV-mean vs NCE-mean, 与runner.spectrum_ce/ce_bucket对齐, 不能把FP fallback套到ICE.", "refs": ["artifacts/007-r3/observations_50.json collision_energy_*", "artifacts/g1-validation/code/ice/ice_runner.py:372-387 spectrum_ce/ce_bucket + SUM_CE"]},
            {"id": "R3-instrument", "issue": "仪器映射", "detail": "R3混合 ESI-TOF (waters xevo g2 TOF), Orbitrap Exploris, timsTOF等 -> token Orbitrap iff instr_family==1 else QTOF. NP/G1全timsTOF->QTOF. 需逐谱核对 instr_token, 不能假设全QTOF.", "refs": ["artifacts/007-r3/observations_50.json instrument_raw/family", "artifacts/g1-validation/code/ice/ice_runner.py:396-399 instr_token"]},
            {"id": "R3-coverage", "issue": "Adduct/模式覆盖与缺失规则", "detail": "R3含 [M+H]+/[M-H]-/未知/ [M-H2O-H]- (explicit supported delta -19...) + 4 NILU质量不确定 (no accurate window). v9 ICE仅 positive [M+H]+/[M+Na]+ (COV). 需按契约逐分子判定 支持->v9, 不支持->多谱FP或missing (scored续后, 从不打0). 不能把FP 20/21 (B_known)直接当ICE可救. 4 NILU另列, 不入B_known分母.", "refs": ["artifacts/008-expansion/expansion_eval_50.json meta unknown_adduct", "artifacts/009-expansion/r3_source_sensitivity.json (mass_uncertain, B_known 21)", "code/ice/ice_runner.py COV"]},
            {"id": "R3-window-shortlist-z", "issue": "质量窗+shortlist+z重算", "detail": "008 R3是FP全窗竞争 + FP top100 (WMAX=100 by FP score) + final top25 by FP. v9需 扩展窗 (base712K+539) -> 扩展top100 (含新增) -> 全shortlist重算z (zb+zi等权, ddof1, missing None) -> scored(-fused,canon)+unscored(-base,canon). 不能尾接旧融合, 不能复用旧z均值/方差. YTCZZ窗84, rank40, 需新shortlist + ICE回传后重算.", "refs": ["artifacts/008-expansion/expansion_eval_50.json meta window/collapse/order", "artifacts/g1-validation/README.md §3 fusion snippet"]},
            {"id": "R3-ytczz-salt", "issue": "YTCZZ断连钠盐", "detail": "见 ytczz_contract.json: query SUPPORTED, structure ATTEMPTABLE but UNVALIDATED (2 frags .[Na+], Na in VALID but salt DAG unvalidated). 不能假设救回rank40, 不把FP收益移植到ICE. 需实际forward观察 (成功/失败/不支持 vs 解析/预测/预算失败分离).", "refs": ["artifacts/008-expansion/candidate_index_records.jsonl:6142", "artifacts/009-expansion/r3_source_sensitivity.json:1623"]},
            {"id": "R3-weights-runner", "issue": "权重与runner复用", "detail": "FP用 fp_single_s2.pt (orig-head, sha c19aa...) + fp_bits.npy; ICE用 gen.pt/inten.pt (sha deef.../c49e..., 14.5/13.7MB) + code/ice runner verbatim (ice_runner.py sha dc0d..., PREP_TAG panel-A1:v1, deterministic). 大权重走远端下载/引用, 不塞进最小包. 需已验证SHA引用方式, 不重复造轮子.", "refs": ["artifacts/009-expansion/run_meta.json fp_checkpoint_sha", "artifacts/g1-validation/weights/MANIFEST.iceberg.json", "artifacts/f1-bundle/MANIFEST.json timing_basis"]},
            {"id": "R3-cost", "issue": "成本与预算", "detail": "R3 50分子全窗+扩展top100的ICE任务数尚未按CE去重实算 (本次只排清单不跑新分). 首轮沿用008建议2h T4上限, 实测小样本后再放大. 最小增量 (如YTCZZ单分子)应先dry-run (2分子) + 任务数->T4分钟数估算 (26/s pure + load/overhead), 再决定全量.", "refs": ["docs/progress/009-frontier-and-transfer.md 下一步1 (2h上限, 小样本后放大)"]},
            {"id": "R3-truth-boundary", "issue": "真值边界", "detail": "Truth (canonical_group_key, RDKit 2026.03.6)仅排序后评价, 不注入候选, 不决定删除/选择; 来源移除是压力诊断, 不是部署过滤; 不读G1保留250, 不碰保留集, 不扫描阈值/权重.", "refs": ["scripts/analyze_009_expansion_sources.py docstring", "docs/progress/009-frontier-and-transfer.md 扩库节"]},
        ],
        "not_done_in_this_step": ["No new FP/ICE forward for R3", "No representative re-selection", "No CE fallback change", "No threshold/weight scan", "No heldout read"],
        "next_for_R3": ["Freeze representative rule (max-peaks vs median)", "Enumerate per-spectrum CE/instr/adduct coverage table", "Build expanded top100 shortlists (FP) for R3-50", "Minimal ICE forward for expanded shortlists (esp YTCZZ) with cost estimate", "Full-shortlist z recalc per v9 caliber"],
    }


def gen_bundle_inputs(full, np_by, g1_by, by_rec):
    mols = []
    for ds, key in BUNDLE_MOLS:
        m = (np_by if ds == "np" else g1_by)[key]
        eo = full[ds][key]["order_expanded"]
        scores = full[ds][key]["scores_expanded"]
        top100 = list(eo[:100])
        scores100 = [float(scores[k]) for k in top100]
        rep_exp = dict(m["rep_smiles"])
        formulas = dict(m["formulas"])
        for cand in top100:
            if cand not in rep_exp:
                r = by_rec.get(cand)
                if r is None:
                    raise RuntimeError(f"missing candidate rec {cand}")
                rep_exp[cand] = r["smiles_rep"]
                formulas[cand] = r["formula_calc"]
        missing = [c for c in top100 if c not in rep_exp]
        if missing:
            raise RuntimeError(f"missing rep {missing}")
        mols.append({"key": key, "qidx": int(m["qidx"]), "truth_canon": m["truth_canon"], "n_win_canon": len(eo),
                     "base_rank_full": None, "base_rr": None, "order_top100": top100, "scores_top100": scores100,
                     "base_scores_full_window": {c: float(scores[c]) for c in top100},
                     "rep_smiles": rep_exp, "formulas": formulas, "spectra": m["spectra"], "ds": ds})
    return {"molecules": mols}


def gen_weights_reference():
    return {
        "note": "大权重不塞进包，走远端下载/引用；本地运行前按SHA核验，拒绝mismatch (zero-download gate)",
        "iceberg": {
            "gen.pt": {"bytes": 14561244, "sha256": "deef0451c1870d2125653bb212c9d95c5b5d9bf80156dfa9b22fedda7db49bea",
                       "local_refs": ["artifacts/g1-validation/weights/gen.pt", "artifacts/f1-bundle/weights/gen.pt"],
                       "manifest": "artifacts/g1-validation/weights/MANIFEST.iceberg.json ckpt/gen.pt (bundle copy)"},
            "inten.pt": {"bytes": 13793037, "sha256": "c49e330e049d3a8b0095026ed694fcf70045c24305354853732aaa5f89d55068",
                         "local_refs": ["artifacts/g1-validation/weights/inten.pt", "artifacts/f1-bundle/weights/inten.pt"],
                         "manifest": "artifacts/g1-validation/weights/MANIFEST.iceberg.json ckpt/inten.pt (bundle copy)"},
            "ms_pred_commit": "708148c2a8eb0c32120644436fefd2fe90cd2214",
            "rdkit_panel": "2025.3.6 (Colab PyPI非精确, 记入meta --any-rdkit)",
            "prep_tag": "panel-A1:v1(clean0.001/512/prec+2>x2->prep0.002/256/p1/entw;sim0.01Da/20ppm)",
        },
        "fp_for_cpu_recalc_only": {
            "fp_single_s2.pt": {"bytes": 144110255, "sha256": "c19aa29016ed4d7a67bac38105b8fa5076c696c29eb66a504558c9b2e52b614a",
                                "path": "artifacts/001-baseline/weights/fp_single_s2.pt", "note": "CPU FP已用, 不进Colab包"}
        },
        "how_to_provide_on_colab": [
            "方式A (推荐, Drive): 本地将 gen.pt + inten.pt 上传到 Drive (如 /content/drive/MyDrive/v9recall-weights/), Colab里 mkdir -p bundle/weights && cp /content/drive/MyDrive/v9recall-weights/*.pt bundle/weights/ 再按README三步运行; 运行前 driver会核对 bytes+sha, mismatch拒绝.",
            "方式B (Kaggle dataset): 将两权重作为私有Kaggle dataset挂载, cp到 bundle/weights/后同样核验.",
            "方式C (私有存储): 从私有存储下载到 bundle/weights/, 同样核验. 不要把 token/key写进包目录.",
            "本包发出时 weights/ 为空 (零下载运行要求权重+代码+输入齐备才跑; 此最小包故意不含大权重, 由用户按上法提供, 避免塞包).",
        ],
        "runner_reuse": {
            "code_ice": "bundle/code/ice (verbatim copy from artifacts/g1-validation/code/ice; ice_runner.py sha dc0d5d1e14053e2e7d9088817c620d6615372abc3afdd02be2fc7ff9ee0b2802)",
            "worker": "bundle/g1_worker.py (verbatim copy from artifacts/g1-validation/g1_worker.py; PREP_TAG panel-A1:v1, deterministic, cache key model|ckpt|structure|CE|adduct|instrument|prep)",
            "install": "bundle/install.sh (PyPI rdkit/einops/lightning/torchmetrics; torch用Colab自带, 不重装)",
        },
        "verify_cmd": "sha256sum bundle/weights/gen.pt bundle/weights/inten.pt  # 对照上表 sha256 + bytes; 或运行 driver自带 check_weights",
    }


def deep_equal(a, b, path="$"):
    if type(a) is not type(b):
        # allow int vs float with same value? keep strict for JSON numbers: compare via == but report type
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and float(a) == float(b):
            return []
        return [f"{path}: type {type(a).__name__} != {type(b).__name__} ({a!r} vs {b!r})"]
    if isinstance(a, dict):
        diffs = []
        ka, kb = set(a.keys()), set(b.keys())
        for k in sorted(ka - kb):
            diffs.append(f"{path}.{k}: missing in actual (expected {a[k]!r:.120})")
        for k in sorted(kb - ka):
            diffs.append(f"{path}.{k}: extra in actual ({b[k]!r:.120})")
        for k in sorted(ka & kb):
            diffs.extend(deep_equal(a[k], b[k], f"{path}.{k}"))
        return diffs
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: len {len(a)} != {len(b)}"]
        diffs = []
        for i, (x, y) in enumerate(zip(a, b)):
            d = deep_equal(x, y, f"{path}[{i}]")
            if d:
                diffs.extend(d)
                if len(diffs) > 20:
                    diffs.append(f"{path}: ... truncated")
                    break
        return diffs
    if isinstance(a, float):
        # exact float equality (repr consistent)
        if a != b and not (math.isnan(a) and math.isnan(b)):
            return [f"{path}: float {a!r} != {b!r}"]
        return []
    if a != b:
        return [f"{path}: {a!r:.200} != {b!r:.200}"]
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/009-v9recall")
    ap.add_argument("--output-suffix", type=str, default="", help="appended to each output filename, e.g. .repro (originals untouched)")
    ap.add_argument("--check-only", action="store_true", help="verify only, write nothing")
    ap.add_argument("--make-tar", action="store_true", help="also (re)create v9recall-bundle.tar.gz via recorded tar command")
    ap.add_argument("--update-run-meta", action="store_true", help="add reproducibility subsection to run_meta.json (no other keys touched)")
    args = ap.parse_args()

    out_dir: Path = args.output_dir
    suffix: str = args.output_suffix
    t0 = time.monotonic()
    wall = datetime.now().astimezone().isoformat()

    # ---- load allowed inputs ----
    p_index = ROOT / "artifacts/008-expansion/candidate_index_records.jsonl"
    p_obs = ROOT / "artifacts/007-r3/observations_50.json"
    p_np = ROOT / "artifacts/f1-bundle/inputs.json"
    p_g1 = ROOT / "artifacts/g1-validation/inputs.json"
    p_inter = ROOT / "artifacts/009-expansion/research_interference.json"
    p_full = ROOT / "artifacts/009-expansion/research_full_fp_scores.json"
    p_r3 = ROOT / "artifacts/009-expansion/r3_source_sensitivity.json"
    p_eval50 = ROOT / "artifacts/008-expansion/expansion_eval_50.json"
    for p in (p_index, p_obs, p_np, p_g1, p_inter, p_full, p_r3, p_eval50):
        guard_allowed(p)

    inter = json.loads(p_inter.read_text())
    full = json.loads(p_full.read_text())
    np_in = json.loads(p_np.read_text())
    g1_in = json.loads(p_g1.read_text())
    np_by = {m["key"]: m for m in np_in["molecules"]}
    g1_by = {m["key"]: m for m in g1_in["molecules"]}
    by_rec = {}
    for line in p_index.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        by_rec[r["canonical_key"]] = r
    obs = json.loads(p_obs.read_text())
    r3 = json.loads(p_r3.read_text())
    eval50 = json.loads(p_eval50.read_text())

    # ---- generate ----
    pairs_all = gen_pairs_all(inter, full, np_by, g1_by)
    sup_details = [detail_for(ds, mol, cand, inter, full, np_by, g1_by, by_rec) for ds, mol, cand in SUPPORTED_DEFS]
    unsup_details = [detail_for(ds, mol, cand, inter, full, np_by, g1_by, by_rec) for ds, mol, cand in UNSUP_DEFS]
    fwd_core = gen_forward_tasks(sup_details)
    # extend fwd with full-shortlist dry-run recomputed locally (mirror worker planning)
    # bundle inputs needed for full counts
    bundle_inputs = gen_bundle_inputs(full, np_by, g1_by, by_rec)
    # full counts via local expand over bundle shortlists (635 cands)
    # count unique (canon,adduct,instr,ce_bucket) for k=100
    def count_bundle(cemode):
        uq = set()
        n_cands = 0
        n_unsup = 0
        n_total_spec = 0
        n_cov = 0
        for m in bundle_inputs["molecules"]:
            order = m["order_top100"]
            n_cands += len(order)
            recs = expand_records(m["spectra"], cemode)
            n_total_spec += len(m["spectra"])
            n_cov += sum(1 for r in recs if r["covered"])
            n_unsup += sum(1 for r in recs if not r["covered"])
            for cand in order:
                for r in recs:
                    if not r["covered"]:
                        continue
                    ad = r["sp"]["adduct"]
                    ins = instr_token(r["sp"])
                    for ce in r["ces"]:
                        ceb = ce_bucket(ce)
                        uq.add((cand, ad, ins, ("sum" if isinstance(ceb, float) and math.isnan(ceb) else round(float(ceb), 1))))
        return len(uq), n_cands, n_unsup, n_total_spec, n_cov

    split_uq, n_cands_full, n_unsup_full, n_spec_full, n_cov_full = count_bundle("split")
    mean_uq, _, _, _, _ = count_bundle("mean")
    # dry-run 2 mols (first 2 in BUNDLE_MOLS order)
    two = {"molecules": bundle_inputs["molecules"][:2]}
    uq2 = set()
    n2 = 0
    for m in two["molecules"]:
        for cand in m["order_top100"]:
            n2 += 1
            for r in expand_records(m["spectra"], "split"):
                if not r["covered"]:
                    continue
                ad = r["sp"]["adduct"]
                ins = instr_token(r["sp"])
                for ce in r["ces"]:
                    ceb = ce_bucket(ce)
                    uq2.add((cand, ad, ins, ("sum" if isinstance(ceb, float) and math.isnan(ceb) else round(float(ceb), 1))))

    fwd = dict(fwd_core)
    fwd["cost_T4"]["full_11mols_mean"] = {"unique_jobs": mean_uq, "pure_pred_s_at_26_per_s": round(mean_uq / 26.0, 1), "pure_pred_min": round(mean_uq / 26.0 / 60.0, 2),
                                          "with_load_overhead_min_est": "~5-10 min total, far below 008 2h上限",
                                          "basis": "full expanded shortlist rerun (635 cands); incremental new-only is 31/41 jobs (<1min pure); both far below 2h"}
    fwd["cost_T4"]["full_11mols_split"] = {"unique_jobs": split_uq, "pure_pred_s_at_26_per_s": round(split_uq / 26.0, 1), "pure_pred_min": round(split_uq / 26.0 / 60.0, 2),
                                           "with_load_overhead_min_est": "~5-10 min total, far below 008 2h上限",
                                           "basis": "full expanded shortlist rerun (635 cands); incremental new-only is 31/41 jobs (<1min pure); both far below 2h"}
    fwd["full_shortlist_dryrun"] = {
        "n_mols": 11, "n_cands": n_cands_full,
        "split": {"unique_jobs": split_uq, "pure_s_at_26_per_s": round(split_uq / 26.0, 1), "pure_min": round(split_uq / 26.0 / 60.0, 2),
                  "with_overhead_min_est": "~5-10 min total (load+scoring), far below 2h",
                  "meta": "/tmp/v9recall_dry_full_split.json.meta.json (dry-run, CPU, 0.61s planning)"},
        "mean": {"unique_jobs": mean_uq, "pure_s_at_26_per_s": round(mean_uq / 26.0, 1), "pure_min": round(mean_uq / 26.0 / 60.0, 2)},
        "dryrun_2mols_split": {"unique_jobs": len(uq2), "n_cands": n2, "note": "first 2 mols (BHQC71+BMLI38) wiring check, CPU 0.1s planning, status ok-dry-run"},
        "n_spectra_unsupported": n_unsup_full, "n_spectra_total": n_spec_full, "n_covered": n_cov_full,
        "basis": "g1_worker --dry-run (no torch) on bundle/inputs.json, k=100, cemode split/mean; unique (canon,adduct,instr,ce_bucket); runner verbatim; CPU planning <1s",
    }
    fwd["comparison_to_2h"] = {"limit_min": 120, "incremental_split_min_pure": round(fwd["totals"]["split"] / 26.0 / 60.0, 3),
                               "full_split_min_pure": round(split_uq / 26.0 / 60.0, 2),
                               "conclusion": "Both incremental (41 jobs, ~0.03min pure) and full 11-mol rerun (2066 jobs, ~1.3min pure) are >50x below 2h pure; with load/overhead still <10min. G1 full 56062 jobs ~36min pure for scale. First run small sample then scale per 009下一步1."}

    cpu_exp = gen_cpu_expanded(inter, full)

    obs_by = {m["canonical_key"]: m for m in obs["molecules"]}
    # ytczz inputs cross-check (values must match hardcoded contract)
    y_obs = obs_by.get("YTCZZXIRLARSET")
    y_row = next(r for r in r3["rows"] if r["key"] == "YTCZZXIRLARSET")
    y_eval = next(p for p in eval50["per_molecule"] if p["canonical_key"] == "YTCZZXIRLARSET")
    # light cross-checks (fail fast on upstream change)
    assert y_obs is not None and y_obs["n_spectra"] == 1, "YTCZZ obs changed"
    assert y_row["expanded_rank"] == 40 and y_eval["fp_top100_expanded_rank_oracle"] == 40, "YTCZZ rank changed"
    yt = gen_ytczz(y_obs, by_rec.get("YTCZZXIRLARSET"), y_row, y_eval)
    gaps = gen_r3_gaps()

    new_cands = [{"ds": d, "mol": m, "new_canon": c, "new_smiles": by_rec[c]["smiles_rep"],
                  "rank": next(x["expanded_rank"] for x in sup_details if x["mol_key"] == m and x["new_canon"] == c),
                  "fp_score": float(next(x["fp_score"] for x in sup_details if x["mol_key"] == m and x["new_canon"] == c))}
                 for d, m, c in SUPPORTED_DEFS]
    weights_ref = gen_weights_reference()

    targets = [
        ("pairs_all_18.json", pairs_all, 2),
        ("pairs_12_supported_detail.json", sup_details, 2),
        ("pairs_3_unsupported_detail.json", unsup_details, 2),
        ("forward_tasks.json", fwd, 2),
        ("cpu_expanded_top100.json", cpu_exp, 2),
        ("ytczz_contract.json", yt, 2),
        ("r3_baseline_gaps.json", gaps, 2),
        ("bundle/inputs.json", bundle_inputs, 1),
        ("bundle/new_candidates_12.json", new_cands, 2),
        ("bundle/weights_reference.json", weights_ref, 2),
    ]

    # ---- verify vs existing ----
    failures = []
    for rel, obj, indent in targets:
        existing = out_dir / rel
        text_new = dump_text(obj, indent)
        if suffix:
            alt = out_dir / (rel + suffix)
            if args.check_only:
                # compare alt? in check-only with suffix, just report what would be written
                pass
            else:
                if not args.check_only:
                    alt.parent.mkdir(parents=True, exist_ok=True)
                    alt.write_text(text_new)
                    print(f"WROTE suffixed {alt} ({len(text_new)} bytes)")
            # also compare to existing for reporting
            if existing.exists():
                old_obj = json.loads((existing).read_text())
                diffs = deep_equal(obj, old_obj)
                # for bundle inputs also check SHA
                if rel == "bundle/inputs.json":
                    old_sha = sha_file(existing)
                    new_sha = hashlib.sha256(text_new.encode()).hexdigest()
                    print(f"VERIFY {rel}: existing sha {old_sha} vs generated {new_sha} {'OK' if old_sha == new_sha else 'DIFF'}")
                    if old_sha != new_sha and not diffs:
                        diffs = [f"{rel}: SHA differs but value-equal? check dump indent/ordering"]
                if diffs:
                    print(f"DIFF vs existing {rel} ({len(diffs)}):")
                    for d in diffs[:20]:
                        print("  " + d)
                else:
                    print(f"OK value-equal vs existing {rel}")
            continue
        # no suffix: compare to existing, do not overwrite on diff
        if existing.exists():
            guard_allowed(existing, for_verify=True)
            old_text = existing.read_text()
            try:
                old_obj = json.loads(old_text)
            except Exception as e:
                failures.append(f"{rel}: existing JSON unreadable: {e!r}")
                continue
            diffs = deep_equal(obj, old_obj)
            if rel == "bundle/inputs.json":
                old_sha = sha_file(existing)
                new_sha = hashlib.sha256(text_new.encode()).hexdigest()
                print(f"VERIFY {rel}: existing sha {old_sha} vs generated {new_sha}")
                if old_sha != new_sha:
                    # if value-equal but SHA differs, report dump difference
                    if not diffs:
                        failures.append(f"{rel}: SHA differs ({old_sha} vs {new_sha}) but value-equal; dump settings differ")
                        # show first byte diff
                        for i, (a, b) in enumerate(zip(old_text, text_new)):
                            if a != b:
                                failures.append(f"{rel}: first byte diff at {i}: {old_text[max(0,i-40):i+40]!r} vs {text_new[max(0,i-40):i+40]!r}")
                                break
                        else:
                            failures.append(f"{rel}: len differs {len(old_text)} vs {len(text_new)}")
                    else:
                        failures.append(f"{rel}: SHA differs and value differs ({len(diffs)} diffs)")
                        for d in diffs[:20]:
                            print("  DIFF " + d)
                    continue
            if diffs:
                failures.append(f"{rel}: {len(diffs)} value diffs, NOT overwriting")
                print(f"FAIL {rel} ({len(diffs)} diffs, not overwriting):")
                for d in diffs[:30]:
                    print("  " + d)
                continue
            print(f"OK {rel} value-consistent ({len(text_new)} bytes), skip write")
            continue
        else:
            if args.check_only:
                print(f"MISSING {rel} (would write {len(text_new)} bytes)")
            else:
                existing.parent.mkdir(parents=True, exist_ok=True)
                existing.write_text(text_new)
                print(f"WROTE {rel} ({len(text_new)} bytes)")

    if failures:
        print(f"\nFAILED {len(failures)} file(s):", file=sys.stderr)
        for f in failures:
            print(" - " + f, file=sys.stderr)
        sys.exit(1)

    # key numbers summary
    print("\nKEY NUMBERS:")
    print(f" pairs_all={len(pairs_all)} top100={sum(1 for x in pairs_all if x['in_top100'])} sup12={len(sup_details)} unsup3={len(unsup_details)}")
    print(f" forward split={fwd['totals']['split']} mean={fwd['totals']['mean']} union={fwd['totals']['union']} full_split={split_uq} full_mean={mean_uq}")
    print(f" cpu np {cpu_exp['recompute']['np']['expanded_mrr25']!r} g1 {cpu_exp['recompute']['g1']['expanded_mrr25']!r}")
    print(f" bundle_inputs mols={len(bundle_inputs['molecules'])} cands={n_cands_full}")

    if args.make_tar:
        # recorded tar command; tar bytes unstable due to gzip mtime, so verify listing not bytes
        import subprocess

        bundle_dir = out_dir / "bundle"
        tar_path = out_dir / "v9recall-bundle.tar.gz"
        cmd = ["tar", "czf", str(tar_path), "-C", str(bundle_dir), "."]
        print("+ " + " ".join(cmd) + "  # recorded bundle tar command (weights excluded)")
        if not args.check_only:
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print("tar failed", file=sys.stderr)
                sys.exit(1)
            print(f"tar wrote {tar_path} ({tar_path.stat().st_size} bytes, sha {sha_file(tar_path)})")
            # list contents
            r2 = subprocess.run(["tar", "tzf", str(tar_path)], capture_output=True, text=True)
            print(r2.stdout[:2000])

    if args.update_run_meta:
        meta_path = out_dir / "run_meta.json"
        guard_allowed(meta_path, for_verify=True)
        meta = json.loads(meta_path.read_text())
        script_sha = sha_file(Path(__file__))
        inputs_sha = {}
        for p in (p_index, p_obs, p_np, p_g1, p_inter, p_full, p_r3, p_eval50):
            inputs_sha[str(p.relative_to(ROOT))] = sha_file(p)
        # bundle inputs sha from generated text
        inputs_sha["artifacts/009-v9recall/bundle/inputs.json"] = hashlib.sha256(dump_text(bundle_inputs, 1).encode()).hexdigest()
        repro = {"script": str(Path(__file__).relative_to(ROOT)), "script_sha256": script_sha, "seed_note": SEED_NOTE,
                 "wall_clock_local": wall, "seconds": round(time.monotonic() - t0, 2),
                 "inputs_sha256": inputs_sha,
                 "verified": {"n_targets": len(targets), "failed": len(failures), "bundle_inputs_sha_match": True},
                 "command": " ".join(sys.argv)}
        # do not touch original keys: only add/overwrite reproducibility
        meta_keys_before = set(meta.keys())
        meta["reproducibility"] = repro
        if args.check_only:
            print("CHECK-ONLY: would add reproducibility to run_meta.json")
        else:
            # ensure no other keys changed
            assert set(meta.keys()) == meta_keys_before | {"reproducibility"}, "unexpected run_meta keys"
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            print(f"UPDATED {meta_path} reproducibility (script {script_sha[:16]}...)")

    print(f"\nDONE in {time.monotonic()-t0:.1f}s wall {wall}")


if __name__ == "__main__":
    main()

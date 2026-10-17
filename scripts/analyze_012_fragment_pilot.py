"""012 手动 CPU 碎片证据研究；固定参数、metadata 选样，不读新盲集。

手动运行：
  .venv-fp/bin/python scripts/analyze_012_fragment_pilot.py --probe
  .venv-fp/bin/python scripts/analyze_012_fragment_pilot.py

仅载入已审阅的 FRAG_SRC 纯计算模块，不执行第三方 notebook 入口。
源码由 Ahmed Berat Özer v7c 收录，作者归功于 Huseyin Emre Aksoy；
上游 notebook 声明 Apache-2.0。本地保留原文件及 SHA，不发布衍生资产。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import signal
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import rdkit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/012-fragment"
UPSTREAM = ROOT / "artifacts/010-frontier/ahmedberatozer__casmi26-v7c-e006-fraggap/embedded/FRAG_SRC.py"
UPSTREAM_SHA = "eeacdcf0afb173a8b438054d93c30f061df7c9c69d1ae32db3530b2916f2e6a7"
SEED = 20261017
LAMBDA = 0.5
TOP_K = 50
MAX_SPECTRA = 16
SECONDS_LIMIT = 3600
INPUTS = {"NP": "artifacts/f1-bundle/inputs.json", "G1": "artifacts/g1-validation/inputs.json"}
ICE_PATHS = {"NP": "artifacts/f1-scores/scores/f1e-4_ice_top100_split.json",
             "G1": "artifacts/g1-scores/scores/g1e-4_ice_top100_split.json"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_frag():
    if sha(UPSTREAM) != UPSTREAM_SHA:
        raise ValueError("Audited upstream fragment source changed; review it before running.")
    spec = importlib.util.spec_from_file_location("frozen_012_frag_rescore", UPSTREAM)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def supported(m):
    return any(s["mode"] == "positive" and s["adduct"] in {"[M+H]+", "[M+Na]+"}
               for s in m["spectra"])


def choose(molecules, dataset, tag, count):
    # Key serves only as an immutable identifier. No rank, truth membership,
    # score, formula ambiguity or spectrum peak property enters selection.
    def sortkey(m):
        return hashlib.sha256(f"{SEED}|{dataset}|{tag}|{m['key']}".encode()).hexdigest()
    if len(molecules) < count:
        raise ValueError((dataset, tag, "insufficient metadata pool", len(molecules), count))
    return [(dataset, tag, m) for m in sorted(molecules, key=sortkey)[:count]]


def select(inputs, full_existing=False):
    result = []
    pool_sizes = {}
    for ds, molecules in inputs.items():
        unsupported = [m for m in molecules if not supported(m)]
        negative = [m for m in unsupported if all(s["mode"] == "negative" for s in m["spectra"])]
        other = [m for m in unsupported if not all(s["mode"] == "negative" for s in m["spectra"])]
        h = [m for m in molecules if all(s["mode"] == "positive" and s["adduct"] == "[M+H]+"
                                       for s in m["spectra"])]
        na = [m for m in molecules if all(s["mode"] == "positive" and s["adduct"] == "[M+Na]+"
                                        for s in m["spectra"])]
        pool_sizes[ds] = dict(unsupported=len(unsupported), negative_only=len(negative),
                              other_or_mixed=len(other), positive_h=len(h), positive_na=len(na))
        result += choose(negative, ds, "unsupported_negative", len(negative) if full_existing else 6)
        result += choose(other, ds, "unsupported_other_or_mixed", len(other) if full_existing else 6)
        result += choose(h, ds, "control_H", 2)
        if ds == "G1":
            result += choose(na, ds, "control_Na", 2)
    return result, pool_sizes


def eligible_spectra(m, frag):
    used, omitted = [], []
    for index, s in enumerate(m["spectra"]):
        adduct = s["adduct"]
        expected_mode = "positive" if adduct.endswith("+") else "negative"
        # The upstream sign-only carrier fallback guesses dimer/unknown
        # chemistry. Deliberately bypass it by admitting only explicit entries.
        if adduct not in frag.CARRIERS or s["mode"] != expected_mode:
            omitted.append({"index": index, "adduct": adduct, "reason": "no_explicit_matching_carrier"})
            continue
        if len(used) >= MAX_SPECTRA:
            omitted.append({"index": index, "adduct": adduct, "reason": "fixed_16_spectrum_cap"})
            continue
        x = dict(s, mode=1 if s["mode"] == "positive" else -1)
        if frag._prep_spectrum(x, frag.DEFAULT_CFG) is None:
            omitted.append({"index": index, "adduct": adduct, "reason": "no_valid_positive_intensity"})
            continue
        used.append(x)
    return used, omitted


def z(values):
    values = np.asarray(values, dtype=float)
    sd = values.std(ddof=1) if len(values) > 1 else 0.0
    return (values - values.mean()) / sd if sd > 1e-12 else np.zeros(len(values))


def rerank(m, fragment):
    original = m["order_top100"]
    top = original[:TOP_K]
    groups = defaultdict(list)
    for i, key in enumerate(top):
        groups[m["formulas"].get(key) or f"unknown:{key}"].append(i)
    reranked = list(original)
    info = []
    for formula, slots in groups.items():
        finite = [i for i in slots if fragment.get(top[i]) is not None]
        base = [m["base_scores_full_window"][top[i]] for i in finite]
        scores = [fragment[top[i]] for i in finite]
        combined = z(base) + LAMBDA * z(scores) if finite else []
        order = sorted(range(len(finite)), key=lambda j: (-combined[j], finite[j]))
        for slot, j in zip(finite, order):
            reranked[slot] = top[finite[j]]
        info.append({"formula": formula, "size": len(slots), "finite": len(finite),
                     "fragment_distinct": len(set(round(v, 12) for v in scores)),
                     "changed_slots": sum(reranked[i] != original[i] for i in slots)})
    if any(m["formulas"].get(a) != m["formulas"].get(b) for a, b in zip(original, reranked)):
        raise ValueError("Formula slot invariant failed")
    if sorted(original) != sorted(reranked) or original[TOP_K:] != reranked[TOP_K:]:
        raise ValueError("Candidate pool or tail changed")
    return reranked, info


def truth_rank(order, truth):
    return order.index(truth) + 1 if truth in order else None


def rr(rank):
    return 1.0 / rank if rank is not None and rank <= 25 else 0.0


def v9_order(m, raw):
    order = m["order_top100"]
    iv = {r["canon"]: r.get("fused_merge") for r in raw.values()}
    values = [iv.get(k) for k in order]
    finite = [i for i, v in enumerate(values) if v is not None and math.isfinite(v)]
    base = [m["base_scores_full_window"][k] for k in order]
    zs = np.zeros(len(order))
    if finite:
        zs[finite] = z([values[i] for i in finite])
    combined = z(base) + zs
    covered = set(finite)
    ix = sorted(finite, key=lambda i: (-combined[i], i))
    ix += sorted((i for i in range(len(order)) if i not in covered), key=lambda i: (-base[i], i))
    return [order[i] for i in ix], len(finite)


def summarize(rows, after="fragment_rank"):
    if not rows:
        return {"n": 0}
    a = np.asarray([rr(r["base_rank"]) for r in rows])
    b = np.asarray([rr(r[after]) for r in rows])
    delta = b - a
    ci = np.quantile(np.random.default_rng(SEED).choice(delta, size=(20000, len(rows))).mean(1), [.025, .975])
    return {"n": len(rows), "base_mrr25": float(a.mean()), "fragment_mrr25": float(b.mean()),
            "delta_mrr25": float(delta.mean()), "paired_bootstrap_ci95": ci.tolist(),
            "base_top1": int(sum(a == 1)), "fragment_top1": int(sum(b == 1)),
            "base_top25": int(sum(a > 0)), "fragment_top25": int(sum(b > 0)),
            "improved": int(sum(delta > 1e-12)), "worsened": int(sum(delta < -1e-12)),
            "top1_gained": int(sum((a < 1) & (b == 1))), "top1_lost": int(sum((a == 1) & (b < 1))),
            "top25_gained": int(sum((a == 0) & (b > 0))), "top25_lost": int(sum((a > 0) & (b == 0))),
            "with_explicit_carrier": sum(r["n_spectra_used"] > 0 for r in rows),
            "with_group_ge3": sum(any(g["finite"] >= 3 for g in r["groups"]) for r in rows),
            "order_changed": sum(r["base_order"] != r["fragment_order"] for r in rows),
            "truth_top50": sum(r["truth_in_top50"] for r in rows),
            "n_candidates_top50": sum(r["n_candidates"] for r in rows),
            "n_candidates_finite_fragment": sum(r["n_finite_fragment"] for r in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true", help="Score 5 candidates in 2 selected molecules; report timing only.")
    parser.add_argument("--full-existing", action="store_true", help="Frozen-rule extension to all 55 existing unsupported research molecules, plus unchanged controls.")
    args = parser.parse_args()
    out = OUT / "full-existing" if args.full_existing else OUT
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    cpu_start = time.process_time()
    def timeout(_sig, _frame):
        # BaseException is intentional: upstream per-candidate exception
        # handling must not swallow this process-wide budget limit.
        raise SystemExit("Fixed one-hour wall-time maximum reached")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(SECONDS_LIMIT)
    frag = load_frag()
    inputs = {ds: json.loads((ROOT / path).read_text())["molecules"] for ds, path in INPUTS.items()}
    selected, pool_sizes = select(inputs, args.full_existing)
    manifest = {
        "created_beijing": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "seed": SEED, "source_version": 357736980, "source_sha256": UPSTREAM_SHA,
        "source_url": "https://www.kaggle.com/code/ahmedberatozer/casmi26-v7c-e006-fraggap?scriptVersionId=357736980",
        "source_inputs": {p: sha(ROOT / p) for p in INPUTS.values()},
        "script_sha256": sha(Path(__file__)), "fragment_cfg": frag.DEFAULT_CFG,
        "selection": ("All 55 unsupported existing NP/G1 research molecules, as a development extension after the fixed 24-molecule pilot; NOT an independent confirmation. " if args.full_existing else "24 unsupported: each NP/G1 6 all-negative and 6 other/mixed. ") + "6 unchanged controls: 2 H from each domain and 2 Na from G1 (NP has no Na-only). Salted SHA order, no labels/ranks/errors in selection.",
        "pool_sizes": pool_sizes, "top_k": TOP_K, "max_spectra": MAX_SPECTRA, "lambda": LAMBDA,
        "routing": "Only unsupported metadata receives treatment. Positive controls are forced FP+fragment diagnostics only. Known explicit carrier and matching polarity required; dimer/unknown spectra excluded, no admissible spectra means unchanged FP fallback.",
        "scoring": "Same-formula group z(FP)+0.5*z(fragment); original formula slots and top50 boundary preserved; original candidate order breaks ties; full top100 retained for rank evaluation.",
        "environment": {"python": sys.version, "numpy": np.__version__, "rdkit": rdkit.__version__,
                        "platform": platform.platform(), "device": "CPU", "numba": frag.HAVE_NUMBA},
        "selected": [{"dataset": ds, "stratum": tag, "key": m["key"],
                      "adducts": dict(Counter(s["adduct"] for s in m["spectra"]))}
                     for ds, tag, m in selected],
    }
    manifest_path = out / ("probe_manifest.json" if args.probe else "manifest.json")
    save(manifest_path, manifest)  # Freeze selection and parameters BEFORE any scoring or label evaluation.
    cache, rows = {}, []
    if args.probe:
        for ds, tag, m in selected[:2]:
            specs, _ = eligible_spectra(m, frag)
            frag.score_molecule(specs, [m["rep_smiles"][k] for k in m["order_top100"][:5]], cache=cache)
        result = {"n_molecules": 2, "n_candidates": 10, "wall_seconds": time.perf_counter()-start,
                  "cpu_seconds": time.process_time()-cpu_start, "numba": frag.HAVE_NUMBA,
                  "no_labels_evaluated": True}
        save(out / "probe.json", result)
        print(json.dumps(result), flush=True)
        return
    for ds, tag, m in selected:
        local_start = time.perf_counter()
        specs, omitted = eligible_spectra(m, frag)
        candidates = m["order_top100"][:TOP_K]
        values = frag.score_molecule(specs, [m["rep_smiles"][k] for k in candidates], cache=cache) if specs else [None] * len(candidates)
        scores = dict(zip(candidates, values))
        reordered, groups = rerank(m, scores)
        rows.append({"dataset": ds, "stratum": tag, "key": m["key"],
                     "n_spectra_input": len(m["spectra"]), "n_spectra_used": len(specs),
                     "used_adducts": dict(Counter(s["adduct"] for s in specs)), "omitted_spectra": omitted,
                     "n_candidates": len(candidates), "n_finite_fragment": sum(v is not None for v in values),
                     "fragment_scores": scores, "groups": groups,
                     "base_order": m["order_top100"], "fragment_order": reordered,
                     "wall_seconds": time.perf_counter() - local_start})
        save(out / "scored_rows.json", rows)
        print(f"scored {len(rows)}/{len(selected)} {ds}/{m['key']} candidates={len(candidates)} spectra={len(specs)} wall={rows[-1]['wall_seconds']:.2f}s", flush=True)
    # Labels are accessed only after the frozen scoring pass has finished.
    ice = {ds: json.loads((ROOT / path).read_text()) for ds, path in ICE_PATHS.items()}
    for row, (ds, tag, m) in zip(rows, selected):
        truth = m["truth_canon"]
        v9, n_ice = v9_order(m, ice[ds][m["key"]])
        row.update(truth=truth, truth_in_top50=truth in m["order_top100"][:TOP_K],
                   base_rank=truth_rank(row["base_order"], truth), fragment_rank=truth_rank(row["fragment_order"], truth),
                   v9_rank=truth_rank(v9, truth), v9_ice_finite=n_ice)
        if tag.startswith("unsupported") and v9 != row["base_order"]:
            raise ValueError((ds, m["key"], "unsupported FP baseline differs from cached v9"))
    unsupported_label = "unsupported55" if args.full_existing else "unsupported24"
    strata = {unsupported_label: [r for r in rows if r["stratum"].startswith("unsupported")],
              "positive_controls6_FP_only": [r for r in rows if r["stratum"].startswith("control")],
              "unsupported_with_carrier": [r for r in rows if r["stratum"].startswith("unsupported") and r["n_spectra_used"] > 0]}
    for ds in INPUTS:
        strata[f"{ds}_unsupported"] = [r for r in strata[unsupported_label] if r["dataset"] == ds]
    for tag in sorted({r["stratum"] for r in rows}):
        strata[tag] = [r for r in rows if r["stratum"] == tag]
    strata["unsupported_no_explicit_carrier_fallback"] = [r for r in strata[unsupported_label] if r["n_spectra_used"] == 0]
    strata["unsupported_only_negative_carriers_used"] = [r for r in strata[unsupported_label] if r["used_adducts"] and all(a.endswith("-") for a in r["used_adducts"])]
    strata["unsupported_has_positive_other_carrier_used"] = [r for r in strata[unsupported_label] if any(a.endswith("+") for a in r["used_adducts"])]
    full_metrics, full_rows = {}, []
    if args.full_existing:
        by_id = {(r["dataset"], r["key"]): r for r in strata[unsupported_label]}
        for ds, molecules in inputs.items():
            for m in molecules:
                base, _ = v9_order(m, ice[ds][m["key"]])
                treatment = by_id.get((ds, m["key"]))
                modified = treatment["fragment_order"] if treatment else base
                if treatment and base != treatment["base_order"]:
                    raise ValueError((ds, m["key"], "full v9 mismatch"))
                # Empty coverage fields on supported controls avoid implying
                # that fragment scoring was applied in the full-v9 policy.
                full_rows.append(dict(treatment or {}, dataset=ds, key=m["key"],
                    base_rank=truth_rank(base, m["truth_canon"]), fragment_rank=truth_rank(modified, m["truth_canon"]),
                    base_order=base, fragment_order=modified, n_spectra_used=treatment["n_spectra_used"] if treatment else 0,
                    n_candidates=treatment["n_candidates"] if treatment else 0,
                    n_finite_fragment=treatment["n_finite_fragment"] if treatment else 0,
                    groups=treatment["groups"] if treatment else [],
                    truth_in_top50=m["truth_canon"] in m["order_top100"][:TOP_K],
                    treatment_applied=treatment is not None))
        full_metrics = {ds: summarize([r for r in full_rows if r["dataset"] == ds]) for ds in INPUTS}
        full_metrics["pooled500_descriptive_only"] = summarize(full_rows)
        save(out / "full_v9_rows.json", full_rows)
    result = {"manifest_file": str(manifest_path.relative_to(ROOT)), "manifest_sha256": sha(manifest_path),
              "wall_seconds": time.perf_counter() - start, "cpu_seconds": time.process_time() - cpu_start,
              "fragment_cache_size": len(cache), "seed": SEED,
              "sources": {p: sha(ROOT / p) for p in ICE_PATHS.values()},
              "summary": {k: summarize(v) for k, v in strata.items()}, "full_v9_metrics": full_metrics, "rows": rows,
              "limits": ["Existing repeatedly used NP/G1 research sets; not unseen-structure or leaderboard proof.",
                         "Fixed metadata-only sample or all-55 existing-set extension and 6 diagnostic controls; no parameter scan.",
                         "Dimer/unknown adducts have no valid explicit carrier in this module and retain fallback.",
                         "Fragment explanation is an untrained heuristic, not probabilities or mechanistic chemistry.",
                         "Two-candidate non-tied groups cannot reverse under this lambda=0.5 standardized blend.",
                         "No electron-mass adjustment applied to these original cached spectra.",
                         "Positive controls deliberately compare FP vs FP+fragment; deployed v9 remains unchanged.",
                         "No model training, GPU run, Kaggle submission or new blind-set access."]}
    save(out / "result.json", result)
    print(json.dumps({"wall_seconds": result["wall_seconds"], "cpu_seconds": result["cpu_seconds"],
                      "summary": result["summary"], "full_v9_metrics": full_metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

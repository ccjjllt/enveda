"""手动 CPU：012 冻结碎片规则在已有 R3 纯 [M-H]- 结构组上的迁移检查。

不调参数、不读新盲集；以原 v9 代表谱 + 固定 538 资产为对照。
手动运行：/opt/homebrew/bin/python3.11 scripts/analyze_013_r3_fragment_transfer.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import platform
import signal
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pyarrow.parquet as pq
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_012_fragment_pilot as frozen

OUT = ROOT / "artifacts/013-r3-fragment"
LIMIT = 300
SOURCES = {}


def read(rel):
    p = ROOT / rel
    SOURCES[rel] = frozen.sha(p)
    return json.loads(p.read_text())


def run():
    start, cpu = time.monotonic(), time.process_time()
    signal.signal(signal.SIGALRM, lambda *_: sys.exit("Five-minute CPU research wall limit reached"))
    signal.alarm(LIMIT)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "result.json").exists():
        raise FileExistsError("Completed result already exists; preserve it")
    RDLogger.DisableLog("rdApp.warning")
    obs = read("artifacts/007-r3/observations_50.json")["molecules"]
    selected = [m for m in obs if all(s["ionization_mode"] == "negative" and s.get("adduct") == "[M-H]-" for s in m["spectra"])]
    selected.sort(key=lambda m: hashlib.sha256(f"20261017|013|{m['canonical_key']}".encode()).hexdigest())
    selected_keys = {m["canonical_key"] for m in selected}
    manifest = {
        "created_beijing": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "seed": 20261017, "device": "CPU", "wall_limit_seconds": LIMIT,
        "selection": "All R3 molecules with only negative [M-H]- spectra; SHA256(20261017|013|key) order; metadata only, no rank/truth membership filtering.",
        "selected_keys": [m["canonical_key"] for m in selected], "selected_n": len(selected),
        "frozen_012_script_sha256": frozen.sha(Path(frozen.__file__)),
        "fragment_source_sha256": frozen.UPSTREAM_SHA,
        "fragment_lambda": frozen.LAMBDA, "top_k": frozen.TOP_K,
        "max_spectra": frozen.MAX_SPECTRA, "script_sha256": frozen.sha(Path(__file__)),
        "scope": "Previously exposed R3 development set, separate from NP/G1 but NOT independent blind validation. No parameter tuning, new blind data, neural forward, GPU or submission.",
    }
    frozen.save(OUT / "frozen_selection.json", manifest)
    frag = frozen.load_frag()
    manifest["fragment_cfg"] = frag.DEFAULT_CFG

    tp, cp, xp = ROOT / "artifacts/pool-train", ROOT / "artifacts/005/coco", ROOT / "artifacts/011-expansion-asset"
    train_keys = np.load(tp / "keys.npy", allow_pickle=True).tolist()
    cm = pickle.loads((cp / "coco_meta.pkl").read_bytes())
    maps = []
    for domain in ("train", "coco"):
        rel = f"artifacts/005/structure_identity_{domain}.parquet"
        SOURCES[rel] = frozen.sha(ROOT / rel)
        tab = pq.read_table(ROOT / rel, columns=["raw_key", "canonical_group_key"]).to_pydict()
        maps.append(dict(zip(tab["raw_key"], tab["canonical_group_key"])))
    keys = [maps[0][k] for k in train_keys] + [maps[1][k] for k in cm["keys"]]
    smis = (tp / "smiles.txt").read_text().splitlines() + list(cm["smiles"])
    masses = np.concatenate([np.load(tp / "mass.npy"), np.load(cp / "coco_mass.npy")])
    fp = [np.load(tp / "fp.npy", mmap_mode="r"), np.load(cp / "coco_fp.npy", mmap_mode="r")]
    asset = read("artifacts/011-expansion-asset/mb539_manifest.json")
    xkeys = asset["canonical_keys"]
    xmass, xfp = np.load(xp / "mb539_masses.npy"), np.load(xp / "mb539_fp.npy")
    xsmis = (xp / "mb539_smiles.txt").read_text().splitlines()
    if len(xkeys) != 538 or len(keys) != 712199 or set(xkeys) & set(keys):
        raise ValueError("Unexpected pool version or overlapping expansion")
    for name in ("mb539_masses.npy", "mb539_fp.npy", "mb539_smiles.txt"):
        actual = frozen.sha(xp / name)
        if actual != asset["files"][name]["sha256"]:
            raise ValueError((name, "asset checksum changed"))
        SOURCES[str((xp / name).relative_to(ROOT))] = actual
    zpath = ROOT / "artifacts/009-expansion/r3_50_logits.npy"
    logits = np.load(zpath, mmap_mode="r")
    SOURCES[str(zpath.relative_to(ROOT))] = frozen.sha(zpath)
    notebook = read("notebooks/submission-v9/enveda-submission-v9.ipynb")
    mass_source = next("".join(c["source"]) for c in notebook["cells"] if "".join(c["source"]).startswith("MASS = dict"))
    env = {"np": np}
    exec(mass_source, env)
    frozen_queries = read("artifacts/012-pubchemlite/frozen_queries.json")
    qby = {r["case_id"]: r for r in frozen_queries["queries"]}
    pcl = read("artifacts/012-pubchemlite/frozen_fp_orders.json")
    old_orders = {r["case_id"]: r["orders"]["v10"] for r in pcl["rows"]}
    audit = read("artifacts/010-recall-audit/audit.json")
    old010 = {r["key"]: r for r in audit["r3_literal_v9"]["rows"]}
    old_verification = read("artifacts/010-recall-audit/verification.json")
    ice = read("artifacts/009-r3baseline/scores/scores/v9r3baseline_ice_top100_split.json")
    if any(r.get("fused_merge") is not None for d in ice.values() for r in d.values()):
        raise ValueError("R3 merge field no longer fully absent")
    cases, offset = [], 0
    for mi, mol in enumerate(obs):
        ss = mol["spectra"]
        chosen = max(range(len(ss)), key=lambda i: ss[i]["n_peaks"])
        spec = ss[chosen]
        q = np.asarray(logits[offset + chosen], dtype=float)
        qm = float(env["neutral_mass"](np.array([spec["precursor_mz"]]), [spec.get("adduct")])[0])
        case_id = f"r3:{mi:04d}"
        baseline, reps = {}, {}
        if math.isfinite(qm):
            ix = np.flatnonzero(np.abs(masses - qm) <= qm * 1e-5)
            if len(ix):
                packed = np.stack([fp[0][j] if j < len(train_keys) else fp[1][j-len(train_keys)] for j in ix])
                scores = np.unpackbits(packed, axis=1)[:, :6930].astype(float) @ q
                for j, score in zip(ix, scores):
                    k = keys[j]
                    if k not in baseline or score > baseline[k]:
                        baseline[k], reps[k] = float(score), smis[j]
            ix = np.flatnonzero(np.abs(xmass - qm) <= qm * 1e-5)
            if len(ix):
                scores = np.unpackbits(xfp[ix], axis=1)[:, :6930].astype(float) @ q
                for j, score in zip(ix, scores):
                    baseline[xkeys[j]], reps[xkeys[j]] = float(score), xsmis[j]
        order_full = sorted(baseline, key=lambda k: (-baseline[k], k))
        if order_full != old_orders[case_id]:
            raise ValueError((case_id, "012 v10 FP order mismatch"))
        key = mol["canonical_key"]
        if order_full[:25] != old010[key]["top25"]["expanded"] or spec["spectrum_id"] != old010[key]["selected_spectrum_id"]:
            raise ValueError((case_id, "010 literal-v9 baseline mismatch"))
        if offset + chosen != qby[case_id]["logit_index"]:
            raise ValueError((case_id, "logit row mismatch"))
        order = order_full[:100]
        specs = []
        if key in selected_keys:
            for s in ss:
                peak_path = ROOT / s["peak_file_ref"]
                actual = frozen.sha(peak_path)
                if actual != s["peak_file_sha256"]:
                    raise ValueError((key, "peak hash changed"))
                SOURCES[s["peak_file_ref"]] = actual
                peak = np.load(peak_path)
                # Same values used by 009 bundle: preserve raw m/z and it;
                # upstream 012 _prep_spectrum normalizes intensity itself.
                specs.append({"mz": peak["mz"].tolist(), "it": peak["it"].tolist(),
                              "prec": s["precursor_mz"], "adduct": s["adduct"],
                              "mode": s["ionization_mode"]})
        cases.append({"case_id": case_id, "key": key, "mass_uncertain": not math.isfinite(qm),
                      "query_mass": qm if math.isfinite(qm) else None,
                      "selected_spectrum_id": spec["spectrum_id"], "selected_local_index": chosen,
                      "order_top100": order, "base_scores_full_window": baseline,
                      "rep_smiles": {k: reps[k] for k in order},
                      "formulas": {k: CalcMolFormula(Chem.MolFromSmiles(reps[k])) for k in order},
                      "spectra": specs, "full_window_n": len(order_full), "treatment_applied": key in selected_keys})
        offset += len(ss)
    if offset != len(logits):
        raise ValueError("logit row count mismatch")
    frozen.save(OUT / "frozen_inputs.json", {"cases": cases, "all50_baseline_matches_010_top25_and_012_full_order": True})
    print("Frozen", len(selected), "pure-negative molecules; all50 baselines match 010/012", flush=True)
    by_key = {c["key"]: c for c in cases}
    rows, cache = [], {}
    for mol in selected:
        m = by_key[mol["canonical_key"]]
        local = time.monotonic()
        specs, omitted = frozen.eligible_spectra(m, frag)
        top = m["order_top100"][:frozen.TOP_K]
        vals = frag.score_molecule(specs, [m["rep_smiles"][k] for k in top], cache=cache) if specs else [None] * len(top)
        scores = dict(zip(top, vals))
        order, groups = frozen.rerank(m, scores)
        rows.append({"key": m["key"], "case_id": m["case_id"], "base_order": m["order_top100"],
                     "fragment_order": order, "fragment_scores": scores, "groups": groups,
                     "n_spectra_input": len(m["spectra"]), "n_spectra_used": len(specs),
                     "used_adducts": dict(Counter(s["adduct"] for s in specs)), "omitted_spectra": omitted,
                     "n_candidates": len(top), "n_finite_fragment": sum(v is not None for v in vals),
                     "mass_uncertain": m["mass_uncertain"], "wall_seconds": time.monotonic()-local})
        frozen.save(OUT / "frozen_scored_rows.json", rows)
        print("scored", len(rows), "/", len(selected), m["key"], "candidates", len(top), "seconds", round(time.monotonic()-local, 3), flush=True)
        if time.process_time() - cpu > LIMIT:
            raise TimeoutError("CPU budget reached")
    # Evaluate labels only after all selected cases have frozen scores/orders.
    selected_by = {r["key"]: r for r in rows}
    allrows = []
    for m in cases:
        row = selected_by.get(m["key"], dict(key=m["key"], case_id=m["case_id"],
            base_order=m["order_top100"], fragment_order=m["order_top100"], n_spectra_used=0,
            n_candidates=0, n_finite_fragment=0, groups=[], mass_uncertain=m["mass_uncertain"]))
        row.update(base_rank=frozen.truth_rank(row["base_order"], m["key"]),
                   fragment_rank=frozen.truth_rank(row["fragment_order"], m["key"]),
                   truth_in_top50=m["key"] in m["order_top100"][:50],
                   truth_in_full_window=m["key"] in m["base_scores_full_window"],
                   treatment_applied=m["treatment_applied"])
        allrows.append(row)
    npg1 = set()
    for rel in frozen.INPUTS.values():
        npg1.update(m["truth_canon"] for m in read(rel)["molecules"])
    overlap = sorted(selected_keys & npg1)
    if overlap:
        raise ValueError(("R3 selected/NP G1 canonical overlap", overlap))
    summary = {"selected_pure_negative22": frozen.summarize(rows),
               "all_R3_50_with_nonselected_unchanged": frozen.summarize(allrows),
               "known_mass46_with_nonselected_unchanged": frozen.summarize([r for r in allrows if not r["mass_uncertain"]])}
    report = {"manifest": manifest, "summary": summary, "rows": rows, "all_rows": allrows,
              "coverage": {"selected_n": len(rows), "selected_known_mass": sum(not r["mass_uncertain"] for r in rows),
                           "selected_truth_full_window": sum(r["truth_in_full_window"] for r in rows),
                           "all50_known_mass": sum(not r["mass_uncertain"] for r in allrows),
                           "all50_unknown_mass_keys": [r["key"] for r in allrows if r["mass_uncertain"]],
                           "selected_vs_NP_G1_canonical_overlap": overlap},
              "baseline": {"all50_matches_010_top25_and_012_full_order": True,
                           "R3_fused_merge_finite": 0, "total_logit_rows": offset,
                           "protocol": "Original maximum-raw-peak representative FP, original v9 neutral_mass, 10ppm, canonical max, fixed 538 asset; all merge ICE missing => unchanged representative FP fallback.",
                           "prior_forward_verification_file": "artifacts/010-recall-audit/verification.json"},
              "source_sha256": SOURCES, "fragment_cache_size": len(cache),
              "wall_seconds": time.monotonic()-start, "cpu_seconds": time.process_time()-cpu,
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "rdkit": rdkit.__version__, "numba": frag.HAVE_NUMBA},
              "limits": ["R3 has already been repeatedly evaluated, is source-selected, and shares MassBank with the fixed expansion; NOT a fresh independent blind set.",
                         "Canonical groups do not overlap the 012 NP/G1 development sample, but model-training exposure remains unknown.",
                         "This evaluates a pre-frozen pure-negative subgroup, not all 012 unsupported-adduct routing conditions.",
                         "Unselected R3 molecules are unchanged only for this transfer check; full-policy aggregate is not claimed.",
                         "CPU float64 cached FP and current RDKit may differ from deployed GPU float32; no new forward or remote run.",
                         "Candidate membership came only from base pool plus fixed 538 index, never truth insertion.",
                         "Four unknown-adduct/unknown-mass cases retain empty candidate fallback and are reported separately."]}
    frozen.save(OUT / "result.json", report)
    signal.alarm(0)
    print(json.dumps({"summary": summary, "coverage": report["coverage"], "wall_seconds": report["wall_seconds"], "cpu_seconds": report["cpu_seconds"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    run()

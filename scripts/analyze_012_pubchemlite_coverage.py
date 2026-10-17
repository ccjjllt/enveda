#!/usr/bin/env python3
"""手动研究：固定PubChemLite全源流式质量窗覆盖，不作全源canonical统计。

仅使用550个已有开发分子的观测谱字段、原v9代表谱/质量函数建立10ppm窗。
来源、窗口和canonical候选冻结到磁盘后才访问truth/class字段。没有新增训练、
自动测试、GPU或submission。可选FP干扰仅表示orig-head，不冒充ICE融合效果。
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import pickle
import platform
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
OUT = ROOT / "artifacts/012-pubchemlite"
CSV_NAME = "PubChemLite_exposomics_20260925.csv"
EXPECTED_MD5 = "3023e89b20e5b57bd3e2c70dc1366be1"
EXPECTED_BYTES = 200142750
EPS = 1e-5
INPUTS = {"np": "artifacts/f1-bundle/inputs.json",
          "g1": "artifacts/g1-validation/inputs.json",
          "r3": "artifacts/007-r3/observations_50.json"}
LOGITS = {"np": "artifacts/f1-bundle/inputs_e0_logits.npy",
          "g1": "artifacts/g1-validation/inputs_e0_logits.npy",
          "r3": "artifacts/009-expansion/r3_50_logits.npy"}


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def hashes(p):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with Path(p).open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            sha.update(data)
            md5.update(data)
    return {"bytes": Path(p).stat().st_size, "sha256": sha.hexdigest(), "md5": md5.hexdigest()}


def observed_queries():
    notebook = read(ROOT / "notebooks/submission-v9/enveda-submission-v9.ipynb")
    mass_src = next("".join(c["source"]) for c in notebook["cells"]
                    if "".join(c["source"]).startswith("MASS = dict"))
    env = {"np": np}
    exec(mass_src, env)
    queries = []
    for ds, rel in INPUTS.items():
        offset = 0
        for mi, mol in enumerate(read(ROOT / rel)["molecules"]):
            specs = mol["spectra"]
            selected = max(range(len(specs)), key=lambda i: specs[i]["n_peaks"] if ds == "r3" else len(specs[i]["mz"]))
            spec = specs[selected]
            precursor = float(spec["precursor_mz" if ds == "r3" else "prec"])
            adduct = spec.get("adduct")
            mass = float(env["neutral_mass"](np.array([precursor]), [adduct])[0])
            qidx = offset + selected
            if ds != "r3" and qidx != mol["qidx"]:
                raise RuntimeError("representative-index mismatch")
            queries.append({"case_id": f"{ds}:{mi:04d}", "dataset": ds, "molecule_index": mi,
                            "selected_local_index": selected, "logit_index": qidx,
                            "selected_spectrum_id": spec.get("spectrum_id"),
                            "precursor_mz": precursor, "adduct": adduct,
                            "mode": spec.get("ionization_mode" if ds == "r3" else "mode"),
                            "query_mass": mass if math.isfinite(mass) else None,
                            "mass_uncertain": not math.isfinite(mass)})
            offset += len(specs)
    write(OUT / "frozen_queries.json", {"queries": queries, "ppm": 10,
          "mass_contract": "Literal v9 notebook; unmodified observed precursor; no additional electron correction.",
          "selection": "max raw peak count, ties retain original order; no truth fields used",
          "mass_cell_sha256": hashlib.sha256(mass_src.encode()).hexdigest()})
    return queries


def base_pool():
    maps = []
    for domain in ("train", "coco"):
        tab = pq.read_table(ROOT / f"artifacts/005/structure_identity_{domain}.parquet",
                            columns=["raw_key", "canonical_group_key"]).to_pydict()
        maps.append(dict(zip(tab["raw_key"], tab["canonical_group_key"])))
    tp, cp = ROOT / "artifacts/pool-train", ROOT / "artifacts/005/coco"
    train_keys = np.load(tp / "keys.npy", allow_pickle=True).tolist()
    coco_meta = pickle.loads((cp / "coco_meta.pkl").read_bytes())
    keys = [maps[0][key] for key in train_keys] + [maps[1][key] for key in coco_meta["keys"]]
    masses = np.concatenate([np.load(tp / "mass.npy"), np.load(cp / "coco_mass.npy")]).astype(float)
    manifest = read(ROOT / "artifacts/011-expansion-asset/mb539_manifest.json")
    mbkeys = manifest["canonical_keys"]
    mbmass = np.load(ROOT / "artifacts/011-expansion-asset/mb539_masses.npy")
    if len(mbkeys) != 538 or len(masses) != 712199:
        raise RuntimeError("unexpected pool version")
    return keys, masses, mbkeys, mbmass, len(train_keys)


def stream_selected(queries, cpu_started, cpu_budget):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Descriptors import ExactMolWt
    from chemistry import canonical_group_key
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    ordered = sorted((q["query_mass"], i) for i, q in enumerate(queries) if q["query_mass"] is not None)
    centers = [x[0] for x in ordered]
    records, counts, cache = [], Counter(), {}
    max_mass_difference = 0.0
    with (OUT / CSV_NAME).open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        for rowno, row in enumerate(reader, start=2):
            counts["source_records"] += 1
            try:
                mass = float(row["MonoisotopicMass"])
                if not math.isfinite(mass) or mass <= 0:
                    raise ValueError()
            except (ValueError, KeyError):
                counts["invalid_source_mass"] += 1
                continue
            lower = bisect.bisect_left(centers, mass / (1 + EPS))
            upper = bisect.bisect_right(centers, mass / (1 - EPS))
            if lower == upper:
                continue
            counts["source_records_in_observed_10ppm_windows"] += 1
            if time.process_time() - cpu_started > cpu_budget:
                raise TimeoutError("CPU budget reached; no partial source treated as full")
            smi = row.get("SMILES", "")
            try:
                if smi not in cache:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None or any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
                        raise ValueError("invalid molecule or wildcard")
                    canon = canonical_group_key(smi)
                    if not re.fullmatch("[A-Z]{14}", canon):
                        raise ValueError("invalid canonical key")
                    cache[smi] = (canon, float(ExactMolWt(mol)))
                canon, rd_mass = cache[smi]
            except Exception as exc:
                counts["invalid_selected_structure"] += 1
                records.append({"source_row": rowno, "cid": row.get("Identifier"), "invalid": str(exc)[:160]})
                continue
            source_cases = [queries[ordered[j][1]]["case_id"] for j in range(lower, upper)]
            final_cases = [queries[ordered[j][1]]["case_id"] for j in range(lower, upper)
                           if abs(rd_mass - ordered[j][0]) <= ordered[j][0] * EPS]
            counts["source_case_pairs"] += len(source_cases)
            counts["rdkit_case_pairs"] += len(final_cases)
            counts["source_pairs_rejected_by_rdkit_mass"] += len(source_cases) - len(final_cases)
            max_mass_difference = max(max_mass_difference, abs(rd_mass - mass))
            records.append({"source_row": rowno, "cid": row["Identifier"], "source_firstblock": row["FirstBlock"],
                            "canonical_key": canon, "smiles": smi, "source_mass": mass, "rdkit_mass": rd_mass,
                            "source_case_ids": source_cases, "case_ids": final_cases})
            if len(records) % 1000 == 0:
                print("selected", len(records), "scanned", counts["source_records"], "cpu", round(time.process_time() - cpu_started, 2), flush=True)
    freeze = {"counts": dict(counts), "csv_fields": fields, "records": records,
              "scope": "full source mass scan; only query-window records canonicalized; not a global canonical index",
              "mass_contract": "source mass first 10ppm; selected records rechecked with RDKit ExactMolWt; outside first pass not canonicalized",
              "max_selected_source_vs_rdkit_mass_da": max_mass_difference}
    write(OUT / "frozen_window_candidates.json", freeze)
    return freeze


def score_without_truth(queries, records, pool, cpu_started, cpu_budget):
    from candidate_expansion import fp_bits_for_smiles
    keys, masses, mbkeys, mbmass, ntr = pool
    base_set = set(keys)
    v10_set = base_set | set(mbkeys)
    fp = [np.load(ROOT / "artifacts/pool-train/fp.npy", mmap_mode="r"),
          np.load(ROOT / "artifacts/005/coco/coco_fp.npy", mmap_mode="r")]
    mbfp = np.load(ROOT / "artifacts/011-expansion-asset/mb539_fp.npy", mmap_mode="r")
    logits = {ds: np.load(ROOT / path, mmap_mode="r") for ds, path in LOGITS.items()}
    bits = np.load(ROOT / "artifacts/005/coco/fp_bits.npy")
    ix = np.argsort(masses, kind="stable")
    sorted_mass = masses[ix]
    by_case = {q["case_id"]: [] for q in queries}
    for record in records:
        for case in record.get("case_ids", []):
            by_case[case].append(record)
    cache, rows = {}, []
    for q in queries:
        case, qm = q["case_id"], q["query_mass"]
        base, expanded, added, source_scores = {}, {}, {}, {}
        if qm is not None:
            selected = ix[np.searchsorted(sorted_mass, qm * (1 - EPS), side="left"):
                          np.searchsorted(sorted_mass, qm * (1 + EPS), side="right")]
            z = np.asarray(logits[q["dataset"]][q["logit_index"]], float)
            if len(selected):
                packed = np.stack([fp[0][i] if i < ntr else fp[1][i - ntr] for i in selected])
                scores = np.unpackbits(packed, axis=1)[:, :6930].astype(float) @ z
                for i, score in zip(selected, scores):
                    base[keys[i]] = max(base.get(keys[i], -np.inf), float(score))
            expanded = dict(base)
            selected_mb = np.flatnonzero(np.abs(mbmass - qm) <= qm * EPS)
            if len(selected_mb):
                scores = np.unpackbits(mbfp[selected_mb], axis=1)[:, :6930].astype(float) @ z
                for i, score in zip(selected_mb, scores):
                    expanded[mbkeys[i]] = max(expanded.get(mbkeys[i], -np.inf), float(score))
            for record in by_case[case]:
                canon = record["canonical_key"]
                if time.process_time() - cpu_started > cpu_budget:
                    raise TimeoutError("CPU budget reached during optional FP interference")
                smi = record["smiles"]
                if smi not in cache:
                    cache[smi] = np.asarray(fp_bits_for_smiles(smi, bits)[0], dtype=float)
                score = float(cache[smi] @ z)
                source_scores[canon] = max(source_scores.get(canon, -np.inf), score)
                # Single expansion experiment: existing candidate representatives stay fixed.
                if canon not in v10_set:
                    added[canon] = max(added.get(canon, -np.inf), score)
        pools = {"base": base, "v10": expanded, "v10_pubchemlite": {**expanded, **added}, "source_only": source_scores}
        orders = {name: sorted(sc, key=lambda k: (-sc[k], k)) for name, sc in pools.items()}
        rows.append({"case_id": case, "source_window_keys": sorted({r["canonical_key"] for r in by_case[case]}),
                     "source_window_records": len(by_case[case]), "n_new_vs_base": len({r["canonical_key"] for r in by_case[case]} - base_set),
                     "n_new_vs_v10": len(added), "orders": orders, "added_scores": added,
                     "new_top25": [k for k in orders["v10_pubchemlite"][:25] if k in added],
                     "old_top25_displaced": [k for k in orders["v10"][:25] if k not in orders["v10_pubchemlite"][:25]]})
    write(OUT / "frozen_fp_orders.json", {"caliber": "representative FP only; no ICE; source overlaps retain original representative", "rows": rows})
    return rows


def evaluate_after_freeze(queries, frozen, score_rows, pool):
    # First access to truth labels happens only after both candidate and order freezes exist.
    sources = {ds: read(ROOT / rel)["molecules"] for ds, rel in INPUTS.items()}
    records = frozen["records"]
    query_map = {q["case_id"]: q for q in queries}
    scored = {r["case_id"]: r for r in score_rows}
    base_set, v10_set = set(pool[0]), set(pool[0]) | set(pool[2])
    selected_keys = {r["canonical_key"] for r in records if r.get("case_ids")}
    rows = []
    for q in queries:
        m = sources[q["dataset"]][q["molecule_index"]]
        truth = m["canonical_key"] if q["dataset"] == "r3" else m["truth_canon"]
        result = scored[q["case_id"]]
        ranks = {name: order.index(truth) + 1 if truth in order else None for name, order in result["orders"].items()}
        rows.append({**q, "truth": truth, "class_AB": m.get("class_AB"),
                     "source_truth_in_own_window": truth in result["source_window_keys"],
                     "ranks": ranks, "window_counts": {name: len(order) for name, order in result["orders"].items()},
                     "n_new_vs_base": result["n_new_vs_base"], "n_new_vs_v10": result["n_new_vs_v10"],
                     "new_top25": result["new_top25"], "old_top25_displaced": result["old_top25_displaced"]})
    def stats(subset):
        known = [r for r in subset if not r["mass_uncertain"]]
        result = {"n": len(subset), "known_mass": len(known), "unknown_mass": len(subset) - len(known),
                  "source_truth_own_window": sum(r["source_truth_in_own_window"] for r in known),
                  "source_truth_own_window_rate": sum(r["source_truth_in_own_window"] for r in known) / len(known) if known else None,
                  "new_candidate_query_pairs_vs_base": sum(r["n_new_vs_base"] for r in known),
                  "new_candidate_query_pairs_vs_v10": sum(r["n_new_vs_v10"] for r in known),
                  "queries_with_new_candidates_vs_v10": sum(r["n_new_vs_v10"] > 0 for r in known),
                  "new_top25_pairs": sum(len(r["new_top25"]) for r in known),
                  "displaced_old_top25_pairs": sum(len(r["old_top25_displaced"]) for r in known)}
        for name in ("base", "v10", "v10_pubchemlite", "source_only"):
            rank = [r["ranks"][name] for r in known]
            result[name] = {"window_truth_coverage": sum(x is not None for x in rank),
                            "fp_top1": sum(x == 1 for x in rank),
                            "fp_top25": sum(x is not None and x <= 25 for x in rank),
                            "fp_top100": sum(x is not None and x <= 100 for x in rank),
                            "fp_mrr25": sum(1 / x if x is not None and x <= 25 else 0 for x in rank) / len(rank) if rank else None}
        result["new_truth_vs_base"] = [r["truth"] for r in known if r["ranks"]["base"] is None and r["source_truth_in_own_window"]]
        result["new_truth_vs_v10"] = [r["truth"] for r in known if r["ranks"]["v10"] is None and r["source_truth_in_own_window"]]
        return result
    summary = {ds: stats([r for r in rows if r["dataset"] == ds]) for ds in INPUTS}
    summary["r3_B"] = stats([r for r in rows if r["dataset"] == "r3" and r["class_AB"] == "B"])
    return {"counts": {**frozen["counts"], "selected_valid_window_canonical": len(selected_keys),
                       "selected_new_canonical_vs_base": len(selected_keys - base_set),
                       "selected_new_canonical_vs_v10": len(selected_keys - v10_set),
                       "base_global_canonical": len(base_set), "v10_global_canonical": len(v10_set)},
            "summary": summary, "rows": rows,
            "limitations": ["Only records in observed source-mass windows canonicalized; these are not whole-source canonical counts.",
                            "Source itself prefilters salts/disconnected/metal structures; lack of coverage is not PubChem-full lack of coverage.",
                            "Fixed old NP/G1/R3 development sets, not fresh heldout data; public/hidden gains cannot be inferred.",
                            "FP-only candidate interference; no ICE forward/fusion; no new learned model.",
                            "First pass uses official MonoisotopicMass; possible RDKit-only boundary entries outside this window are unexamined."]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu-budget-seconds", type=float, default=900)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "coverage_report.json").exists():
        raise FileExistsError("do not overwrite completed research")
    wall, cpu = time.monotonic(), time.process_time()
    meta = read(OUT / "zenodo-record-22953038.json")
    source = hashes(OUT / CSV_NAME)
    if source["md5"] != EXPECTED_MD5 or source["bytes"] != EXPECTED_BYTES:
        raise RuntimeError("official snapshot checksum/size mismatch")
    queries = observed_queries()
    print("queries", len(queries), "known", sum(not q["mass_uncertain"] for q in queries), flush=True)
    pool = base_pool()
    frozen = stream_selected(queries, cpu, args.cpu_budget_seconds)
    print("frozen coverage", frozen["counts"], flush=True)
    scores = score_without_truth(queries, frozen["records"], pool, cpu, args.cpu_budget_seconds)
    report = evaluate_after_freeze(queries, frozen, scores, pool)
    import rdkit
    report["meta"] = {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": 20261017,
                      "device": "CPU", "source_url": "https://zenodo.org/records/22953038", "version": meta["metadata"]["version"],
                      "license": meta["metadata"].get("license"), "source_file": source,
                      "seconds_wall": time.monotonic() - wall, "seconds_cpu": time.process_time() - cpu,
                      "environment": {"python": platform.python_version(), "numpy": np.__version__, "rdkit": rdkit.__version__},
                      "script": hashes(Path(__file__)), "input_hashes": {p: hashes(ROOT / p) for p in [*INPUTS.values(), *LOGITS.values(),
                         "artifacts/011-expansion-asset/mb539_manifest.json", "artifacts/005/coco/fp_bits.npy"]},
                      "freeze_hashes": {p: hashes(OUT / p) for p in ["frozen_queries.json", "frozen_window_candidates.json", "frozen_fp_orders.json"]}}
    write(OUT / "coverage_report.json", report)
    print(json.dumps({"counts": report["counts"], "summary": report["summary"], "seconds_wall": report["meta"]["seconds_wall"], "seconds_cpu": report["meta"]["seconds_cpu"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

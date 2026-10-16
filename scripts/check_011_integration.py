"""Manual 011 research check using actual notebook loaders and run_groups.

Uses existing representative logits only. Device and partial-output paths are
adapted to local CPU; no model forward, heldout selection, GPU, or submission.
This is a manually invoked research entry point, not a test suite.
"""
from __future__ import annotations

import ast
import contextlib
import functools
import hashlib
import importlib.util
import io
import json
import math
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rdkit
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/011-check"
SOURCES = {}


def read(rel):
    p = ROOT / rel
    raw = p.read_bytes()
    SOURCES[rel] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def lines(nb, index):
    return "".join(nb["cells"][index]["source"])


def get_queries(v9):
    mass_env = {"np": np}
    exec(lines(v9, 5), mass_env)
    tasks, zs = [], []
    for dataset, folder in [("np", "f1-bundle"), ("g1", "g1-validation")]:
        inp = read(f"artifacts/{folder}/inputs.json")
        p = ROOT / f"artifacts/{folder}/inputs_e0_logits.npy"
        z = np.load(p, mmap_mode="r")
        SOURCES[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        offset = 0
        for m in inp["molecules"]:
            chosen = max(range(len(m["spectra"])), key=lambda j: len(m["spectra"][j]["mz"]))
            assert m["qidx"] == offset + chosen
            s = m["spectra"][chosen]
            qm = float(mass_env["neutral_mass"](np.array([s["prec"]]), [s["adduct"]])[0])
            tasks.append({"mid": dataset + ":" + m["key"], "dataset": dataset, "key": m["key"],
                          "truth": m["truth_canon"], "query_mass": qm, "logit_row": m["qidx"],
                          "representative_local_index": chosen})
            zs.append(np.asarray(z[m["qidx"]]))
            offset += len(m["spectra"])
        assert offset == len(z)
    obs = read("artifacts/007-r3/observations_50.json")["molecules"]
    p = ROOT / "artifacts/009-expansion/r3_50_logits.npy"
    z = np.load(p, mmap_mode="r")
    SOURCES[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    offset = 0
    for m in obs:
        chosen = max(range(len(m["spectra"])), key=lambda j: m["spectra"][j]["n_peaks"])
        s = m["spectra"][chosen]
        qm = float(mass_env["neutral_mass"](np.array([s["precursor_mz"]]), [s["adduct"]])[0])
        tasks.append({"mid": "r3:" + m["canonical_key"], "dataset": "r3", "key": m["canonical_key"],
                      "truth": m["canonical_key"], "query_mass": qm, "logit_row": offset + chosen,
                      "representative_local_index": chosen})
        zs.append(np.asarray(z[offset + chosen]))
        offset += len(m["spectra"])
    assert offset == len(z) == 59
    return tasks, torch.as_tensor(np.stack(zs), dtype=torch.float32)


def run_notebook_functions(nb, label, tasks, logits):
    asset = ROOT / "artifacts/011-expansion-asset"
    train, coco = ROOT / "artifacts/pool-train", ROOT / "artifacts/005/coco"
    ids = [r["mid"] for r in tasks]
    fake_table = SimpleNamespace(to_pydict=lambda: {"molecule_id": ids})
    env = {"Path": Path, "np": np, "torch": torch, "time": time, "TEST": "cached-query-ids-only",
           "pq": SimpleNamespace(read_table=lambda _: fake_table),
           "T_FP": train / "fp.npy", "T_MASS": train / "mass.npy", "T_KEYS": train / "keys.npy",
           "T_SMI": train / "smiles.txt", "C_FP": coco / "coco_fp.npy",
           "C_MASS": coco / "coco_mass.npy", "C_META": coco / "coco_meta.pkl",
           "X_MANIFEST": asset / "mb539_manifest.json", "X_KEYS": asset / "mb539_keys.npy",
           "X_MASS": asset / "mb539_masses.npy", "X_FP": asset / "mb539_fp.npy",
           "X_SMI": asset / "mb539_smiles.txt"}
    # Execute the actual notebook's pool loader, including v10 manifest checks.
    exec(lines(nb, 3), env)
    env.update(NTR=275810, G=4, PPM=10.0, mids=ids, Z=logits,
               QM=np.asarray([r["query_mass"] for r in tasks]), PARTS={},
               stat={"empty_window": 0, "mass_fallback": 0, "cco": 0, "exception": 0})
    tree = ast.parse(lines(nb, 6))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_groups")
    source = ast.get_source_segment(lines(nb, 6), function)
    folder = OUT / label
    folder.mkdir(exist_ok=True)
    source = source.replace("device='cuda'", "device='cpu'").replace("/kaggle/working", str(folder))
    exec(source, env)
    env["run_groups"](0, math.ceil(len(tasks) / 4))
    records = env["PARTS"][0]
    # Execute native canonical max merge/top100. Cache only the pure canonical
    # helper to avoid repeating expensive tautomer enumeration for shared pools.
    cell = lines(nb, 32)
    canon_start, canon_end = cell.index("def _canon_key("), cell.index("stat_merge =")
    exec(cell[canon_start:canon_end], env)
    env["_canon_key"] = functools.lru_cache(maxsize=None)(env["_canon_key"])
    env.update(Chem=Chem, _te=rdMolStandardize.TautomerEnumerator(),
               rec_by_mid={r["mid"]: r for r in records})
    merge_end = cell.index("# --- build ICE items")
    exec(cell[canon_end:merge_end], env)
    return env, records


def add_full_reference_and_rounding(result):
    """Extend saved-order comparisons without repeating native canonicalization."""
    audit_path = ROOT / "scripts/analyze_010_recall_audit.py"
    spec = importlib.util.spec_from_file_location("audit010", audit_path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    captured, original = [], audit.fp_order

    def capture(scores):
        order = original(scores)
        captured.append({"order": order, "scores": scores})
        return order

    audit.fp_order = capture
    obs = read("artifacts/007-r3/observations_50.json")["molecules"]
    with contextlib.redirect_stdout(io.StringIO()):
        rr = audit.literal_v9_representative(obs, {"fused_merge": 0})
    assert len(captured) == 100
    ref = {r["key"]: {"base": captured[2 * i], "expanded": captured[2 * i + 1]}
           for i, r in enumerate(rr["rows"])}
    document = {"method": "010 literal_v9_representative with fp_order inputs/outputs captured; independent float64 reference, 539 entries, excluded Markush does not enter these windows",
                "rows": ref, "source_script_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest()}
    (OUT / "r3_full_reference.json").write_text(json.dumps(document, indent=2) + "\n")
    for r in result["rows"]:
        if r["dataset"] == "r3":
            r["reference_comparison_depth"] = 100
            for label in ("base", "expanded"):
                r[label + "_reference_order_equal"] = r[label + "_top100"] == ref[r["key"]][label]["order"]
            # This phase checks full order; no absent score-difference estimate
            # should be represented as a measured zero.
            r["max_abs_fp_score_diff_vs_float64_cache"] = None
    result["summary"]["r3"]["max_abs_fp_score_diff_vs_float64_cache"] = None
    for label in ("base", "expanded"):
        result["summary"]["r3"][label + "_reference_order_mismatches"] = sum(
            not r[label + "_reference_order_equal"] for r in result["rows"] if r["dataset"] == "r3")
    raw = []
    for label in ("v9", "v10"):
        with open(OUT / f"native-{label}-cpu/partial_0.pkl", "rb") as stream:
            raw.append(pickle.load(stream)["records"])
    differences = []
    for i, (old, new) in enumerate(zip(*raw)):
        vals = dict(zip(new["shipped"], new["base_scores"]))
        for key, score in zip(old["shipped"], old["base_scores"]):
            if score != vals[key]:
                g = i // 4 * 4
                differences.append({"mid": old["mid"], "shipped_key": key, "old_score": score,
                                    "new_score": vals[key], "delta": vals[key] - score,
                                    "float32_ulp": float(np.spacing(np.float32(score))),
                                    "old_group_window_lengths": [r["n_window"] for r in raw[0][g:g + 4]],
                                    "new_group_window_lengths": [r["n_window"] for r in raw[1][g:g + 4]]})
    result["retained_fp_rounding_differences"] = differences
    old_reference = read("artifacts/009-expansion/research_full_fp_scores.json")
    raw_by_mid = {r["mid"]: r for r in raw[0]}
    tie_details = []
    for row in result["rows"]:
        if row["dataset"] in old_reference and not row["base_reference_order_equal"]:
            expected = old_reference[row["dataset"]][row["key"]]
            native = raw_by_mid[row["mid"]]
            scores = dict(zip(native["shipped"], native["base_scores"]))
            positions = [i for i, (a, b) in enumerate(zip(row["base_top100"], expected["original_order"][:100])) if a != b]
            keys = sorted({row["base_top100"][i] for i in positions} | {expected["original_order"][i] for i in positions})
            tie_details.append({"mid": row["mid"], "positions_one_based": [i + 1 for i in positions],
                                "scores": {k: {"native_raw_score_if_same_key": scores.get(k), "float64_reference": expected["scores_expanded"][k]} for k in keys},
                                "top25_unchanged_vs_reference": row["base_top100"][:25] == expected["original_order"][:25],
                                "top100_set_unchanged_vs_reference": set(row["base_top100"]) == set(expected["original_order"][:100])})
    result["reference_order_numerical_details"] = tie_details
    result["metric_scope"] = "FP ranking before ICE. NP/G1 FP MRR values are not complete v9 MRR. R3 original merge field is all missing, so its FP ranking is also the reconstructed merge-only output."
    result["limits"] = [s for s in result["limits"] if not s.startswith("R3 historical")]
    result["limits"].append("R3 full top100 is independently reconstructed with 010 float64 logic; all 550 queries now have full top100 order comparisons.")
    result["numerical_interpretation"] = "Inspect retained_fp_rounding_differences for batch-width float32 changes, and reference_order_numerical_details for native float32 versus historical float64 ordering. Neither should be described as an expansion regression when v9/v10 retained orders are identical."
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def main():
    start = time.monotonic()
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    RDLogger.DisableLog("rdApp.warning")
    v9 = read("notebooks/submission-v9/enveda-submission-v9.ipynb")
    v10 = read("notebooks/submission-v10/enveda-submission-v10.ipynb")
    reference = read("artifacts/009-expansion/research_full_fp_scores.json")
    r3ref = {r["key"]: r for r in read("artifacts/010-recall-audit/audit.json")["r3_literal_v9"]["rows"]}
    changed = [i for i in range(len(v9["cells"])) if lines(v9, i) != lines(v10, i)]
    assert len(v9["cells"]) == len(v10["cells"]) and changed == [0, 2, 3, 6, 31, 33, 34]
    tasks, logits = get_queries(v9)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        base, br = run_notebook_functions(v9, "native-v9-cpu", tasks, logits)
        expanded, er = run_notebook_functions(v10, "native-v10-cpu", tasks, logits)
    (OUT / "run.log").write_text(log.getvalue())
    bmap, emap = {r["mid"]: r for r in br}, {r["mid"]: r for r in er}
    rows = []
    for q in tasks:
        mid, key, ds = q["mid"], q["key"], q["dataset"]
        b, e = bmap[mid], emap[mid]
        old = {k: (s, v) for k, s, v in zip(b["shipped"], b["smiles"], b["base_scores"])}
        retained = {k: (s, v) for k, s, v in zip(e["shipped"], e["smiles"], e["base_scores"]) if not k.startswith("MB539_")}
        base_order = base["CANON100"][mid]["canon"]
        exp_order = expanded["CANON100"][mid]["canon"]
        if ds in reference:
            ref = reference[ds][key]
            reference_base, reference_exp = ref["original_order"][:100], ref["order_expanded"][:100]
            comparison_depth = 100
            ref_scores = ref["scores_expanded"]
            errors = [abs(float(s) - ref_scores[c]) for c, s in zip(exp_order, expanded["CANON100"][mid]["base"]) if c in ref_scores]
        else:
            ref = r3ref[key]
            reference_base, reference_exp = ref["top25"]["base"], ref["top25"]["expanded"]
            comparison_depth, errors = 25, []
        kept_raw = [k for k in e["shipped"] if not k.startswith("MB539_")]
        row = {**q, "query_mass": q["query_mass"] if math.isfinite(q["query_mass"]) else None,
               "base_status": b["status"], "expanded_status": e["status"],
               "base_shipped_n": len(b["shipped"]), "expanded_shipped_n": len(e["shipped"]),
               "new_shipped_keys": [k for k in e["shipped"] if k.startswith("MB539_")],
               "original_shipped_scores_smiles_exact_equal": old == retained,
               "original_shipped_order_exact_equal": kept_raw == b["shipped"],
               "reference_comparison_depth": comparison_depth,
               "base_reference_order_equal": base_order[:comparison_depth] == reference_base,
               "expanded_reference_order_equal": exp_order[:comparison_depth] == reference_exp,
               "max_abs_fp_score_diff_vs_float64_cache": max(errors, default=0.0),
               "base_top100": base_order, "expanded_top100": exp_order,
               "base_truth_rank": base_order.index(q["truth"]) + 1 if q["truth"] in base_order else None,
               "expanded_truth_rank": exp_order.index(q["truth"]) + 1 if q["truth"] in exp_order else None}
        if not row["base_reference_order_equal"] or not row["expanded_reference_order_equal"]:
            row["reference_base_order"] = reference_base
            row["reference_expanded_order"] = reference_exp
        rows.append(row)
    summary = {}
    for ds in ["np", "g1", "r3"]:
        r = [r for r in rows if r["dataset"] == ds]
        summary[ds] = {"n": len(r), "old_scores_or_smiles_changed": sum(not x["original_shipped_scores_smiles_exact_equal"] for x in r),
                       "old_order_changed": sum(not x["original_shipped_order_exact_equal"] for x in r),
                       "base_reference_order_mismatches": sum(not x["base_reference_order_equal"] for x in r),
                       "expanded_reference_order_mismatches": sum(not x["expanded_reference_order_equal"] for x in r),
                       "max_abs_fp_score_diff_vs_float64_cache": max(x["max_abs_fp_score_diff_vs_float64_cache"] for x in r),
                       "queries_with_additions": sum(bool(x["new_shipped_keys"]) for x in r),
                       "new_candidate_query_pairs": sum(len(x["new_shipped_keys"]) for x in r),
                       "base_mrr25": float(np.mean([1 / x["base_truth_rank"] if x["base_truth_rank"] and x["base_truth_rank"] <= 25 else 0 for x in r])),
                       "expanded_mrr25": float(np.mean([1 / x["expanded_truth_rank"] if x["expanded_truth_rank"] and x["expanded_truth_rank"] <= 25 else 0 for x in r]))}
    result = {"scope": __doc__, "changed_notebook_cells": changed,
              "unchanged_model_mass_and_ice_cells": [4, 5, 32],
              "versions": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "rdkit": rdkit.__version__},
              "device": "cpu", "threads": 4, "n_reused_representative_logits": len(tasks),
              "native_v9_stat": base["stat"], "native_v10_stat": expanded["stat"],
              "native_v9_merge_stat": base["stat_merge"], "native_v10_merge_stat": expanded["stat_merge"],
              "summary": summary, "rows": rows, "source_sha256": SOURCES,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "limits": ["Native functions run on CPU float32 with cached logits. This is not a complete Kaggle/GPU/ICE execution.",
                         "R3 historical representative reference saves top25, so its direct saved-order comparison is top25; native top100 is retained here.",
                         "RDKit canonical identity is recomputed locally, and version is recorded; Kaggle version equivalence must be checked in its own log."],
              "seconds": time.monotonic() - start}
    result = add_full_reference_and_rounding(result)
    (OUT / "integration.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (OUT / "summary.json").write_text(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": summary, "v9_stat": base["stat"], "v10_stat": expanded["stat"], "seconds": result["seconds"]}, indent=2))


if __name__ == "__main__":
    main()

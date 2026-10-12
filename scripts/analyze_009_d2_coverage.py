"""Manual CPU audit of D2 by existing forward coverage; no fitting or search.

This is POST-HOC EXPLORATION of saved OOF predictions, not a deployment result.
The sole partition is scored_n == 0, determined without truth labels. Query
metadata are checked against the frozen ICE positive [M+H]+/[M+Na]+ contract.
No GPU, neural forward, source-library rebuild, or heldout evaluation occurs.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("seen", "nolib_all", "pressure_truth_off", "source_transfer")
SUPPORTED_ADDUCTS = {"[M+H]+", "[M+Na]+"}
SEED = 20261004
DRAWS = 20_000


def file_sha(path: Path) -> dict:
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def summarize(rows: list[dict], scenario: str) -> dict:
    if not rows:
        return {"n": 0}
    a = np.asarray([r["A_rr"] for r in rows], dtype=float)
    b = np.asarray([r["scenarios"][scenario]["rr"] for r in rows], dtype=float)
    delta = b - a
    rng = np.random.default_rng(SEED)
    means = delta[rng.integers(0, len(delta), size=(DRAWS, len(delta)))].mean(axis=1)
    ci = np.quantile(means, [0.025, 0.975])
    return {
        "n": len(rows), "A_mrr25": float(a.mean()), "D2_mrr25": float(b.mean()),
        "delta_mrr25": float(delta.mean()), "paired_bootstrap_ci95": ci.tolist(),
        "A_top1": float((a == 1).mean()), "D2_top1": float((b == 1).mean()),
        "A_top25": float((a > 0).mean()), "D2_top25": float((b > 0).mean()),
        "improved": int((delta > 1e-12).sum()),
        "worsened": int((delta < -1e-12).sum()),
        "tied": int((abs(delta) <= 1e-12).sum()),
        "top1_wrong_to_right": int(((a < 1) & (b == 1)).sum()),
        "top1_right_to_wrong": int(((a == 1) & (b < 1)).sum()),
    }


def condition_summary(spectra: list[dict], scored_n: int) -> dict:
    modes = collections.Counter(str(s.get("mode")) for s in spectra)
    adducts = collections.Counter(str(s.get("adduct")) for s in spectra)
    supported = [s for s in spectra
                 if str(s.get("mode")).lower() in {"positive", "pos", "+", "1"}
                 and s.get("adduct") in SUPPORTED_ADDUCTS]
    pos = modes.get("positive", 0)
    neg = modes.get("negative", 0)
    if scored_n:
        category = "has_forward"
    elif supported:
        category = "metadata_supported_but_zero_scored"
    elif neg == len(spectra):
        category = "negative_only"
    elif pos == len(spectra):
        category = "positive_only_unsupported_adduct"
    else:
        category = "mixed_polarity_or_other_unsupported"
    return {
        "n_query_spectra": len(spectra), "modes": dict(modes), "adducts": dict(adducts),
        "metadata_supported_records": len(supported),
        "metadata_has_supported_record": bool(supported),
        "observed_scored_n": int(scored_n), "category": category,
        "all_spectra_multimer_adduct": bool(spectra) and all(
            re.match(r"^\[[2-9]M", str(s.get("adduct"))) is not None for s in spectra),
    }


def analyze_dataset(ds: str, query_path: Path, shas: dict) -> dict:
    oof_path = ROOT / f"artifacts/008-d2/oof_scores_D2_{ds}.json"
    for path in (oof_path, query_path):
        shas[str(path.relative_to(ROOT))] = file_sha(path)
    oof = json.loads(oof_path.read_text())
    query = {m["key"]: m for m in json.loads(query_path.read_text())["molecules"]}
    if set(oof) != set(query):
        raise ValueError(f"{ds}: query/OOF key mismatch")
    rows = []
    mismatches = []
    for key, entry in oof.items():
        m = query[key]
        condition = condition_summary(m["spectra"], entry["scored_n"])
        if condition["metadata_has_supported_record"] != bool(entry["scored_n"]):
            mismatches.append(key)
        row = {"key": key, "truth_for_offline_scoring_only": entry["truth"],
               "fold": entry["fold"], "scored_n": entry["scored_n"],
               "A_rank": entry["A_rank"], "A_rr": entry["A_rr"],
               "A_top1": entry["A_ranked"][0], "query_conditions": condition,
               "scenarios": {}}
        for scenario in SCENARIOS:
            e = entry["scenarios"][scenario]
            truth, top = entry["truth"], e["ranked"][0]
            row["scenarios"][scenario] = {
                "rank": e["rank"], "rr": e["rr"],
                "delta_rr": float(e["rr"] - entry["A_rr"]), "top1": top,
                "truth_has_library": float(e["masks_has_lib"].get(truth, 0.0)),
                "top1_gated_residual": float(e["residuals_gated"][top]),
                "top1_same_formula_as_truth": (
                    m["formulas"].get(top) == m["formulas"].get(truth)
                    if m["formulas"].get(top) and m["formulas"].get(truth) else None),
            }
        rows.append(row)
    groups = {"all": rows, "scored_n_zero": [r for r in rows if r["scored_n"] == 0],
              "scored_n_positive": [r for r in rows if r["scored_n"] > 0]}
    return {
        "n": len(rows),
        "summary_by_group": {name: {s: summarize(values, s) for s in SCENARIOS}
                             for name, values in groups.items()},
        "metadata_route_audit": {
            "mismatch_n": len(mismatches), "mismatched_keys": mismatches,
            "uncovered_condition_counts": dict(collections.Counter(
                r["query_conditions"]["category"] for r in groups["scored_n_zero"])),
            "uncovered_all_multimer_n": sum(
                r["query_conditions"]["all_spectra_multimer_adduct"] for r in groups["scored_n_zero"]),
        },
        "all_worsened_by_scenario": {s: [r for r in rows if r["scenarios"][s]["delta_rr"] < -1e-12]
                                      for s in SCENARIOS},
        "per_molecule": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/009-d2-coverage")
    args = parser.parse_args()
    shas = {}
    report = {
        "status": "POST_HOC_EXPLORATION_NOT_DEPLOYABLE",
        "partition": "scored_n == 0 versus scored_n > 0; sole fixed partition, no threshold search",
        "seed": SEED, "bootstrap_draws": DRAWS,
        "uncertainty": "Paired molecule bootstrap is conditional on saved OOF fits; it does not refit the model, correct repeated development use, or establish unseen-structure generalization.",
        "routing_interpretation": "scored_n is observable after forward scoring. A metadata-only fallback route can use the frozen supported ion/adduct contract. In future runs separate unsupported conditions from valid conditions with parse/prediction/budget failure; do not assume all zero-scored cases are negative ions.",
        "deployment_limit": "NP and G1 use separately fitted OOF models; this audit supplies no validated model-selection rule for hidden queries and makes no public-score claim. Freeze a label-free rule and verify it on independent data before deployment.",
        "datasets": {},
    }
    for ds, path in (("np", ROOT / "artifacts/f1-bundle/inputs.json"),
                     ("g1", ROOT / "artifacts/g1-validation/inputs.json")):
        report["datasets"][ds] = analyze_dataset(ds, path, shas)
    report["inputs_sha256"] = shas
    report["script_sha256"] = file_sha(Path(__file__))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "coverage_audit.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    summary = {"status": report["status"], "partition": report["partition"],
               "datasets": {ds: {"summary_by_group": d["summary_by_group"],
                                  "metadata_route_audit": d["metadata_route_audit"]}
                            for ds, d in report["datasets"].items()},
               "full_report": str(target), "seed": SEED, "bootstrap_draws": DRAWS}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    for ds, d in report["datasets"].items():
        print(ds, json.dumps(d["metadata_route_audit"], ensure_ascii=False))
        for scenario, s in d["summary_by_group"]["scored_n_zero"].items():
            print(ds, "scored_n_zero", scenario,
                  json.dumps({k: s[k] for k in ("n", "delta_mrr25", "paired_bootstrap_ci95", "improved", "worsened", "top1_wrong_to_right", "top1_right_to_wrong")}, ensure_ascii=False))
    print("wrote", target)


if __name__ == "__main__":
    main()

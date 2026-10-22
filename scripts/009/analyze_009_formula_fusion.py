"""手动复算固定同式组 GL/ICE 融合；无训练、无模型前向、无参数扫描。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import time
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CONFIG = {
    "seed": 20261012,
    "bootstrap_draws": 20000,
    "shortlist": 100,
    "fp_weight": 1.0,
    "gl_weight": 0.5,
    "ice_weight": 0.5,
    "z_ddof": 1,
    "tie_grid": 1e-9,
    "group_min_candidates": 2,
    "require_full_both_simulator_coverage": True,
    "baseline": "F1 ICE100 split-merge: global z(FP)+z(ICE), scored first, unscored base tail",
    "treatment": "Within complete same-formula groups z(FP)+0.5z(GL)+0.5z(ICE); put back into original v9 formula slots; otherwise no change",
    "tie_rule": "Round fused group score to 1e-9 grid, then preserve v9 group order, matching the existing public gl_fuse convention",
    "scope": "NP250 repeatedly used development set; no heldout, no public-score estimate, not exact reproduction of public multi-engine pipeline",
}


def finite(v):
    return v is not None and isinstance(v, (int, float)) and bool(np.isfinite(v))


def zscore(values):
    a = np.asarray([float(v) if finite(v) else np.nan for v in values])
    ok = np.isfinite(a)
    out = np.zeros(len(values), dtype=np.float64)
    if ok.sum() >= 2:
        sd = a[ok].std(ddof=1)
        if sd > 0:
            out[ok] = (a[ok] - a[ok].mean()) / sd
    return out


def baseline_order(keys, fp, ice):
    fused = zscore(fp) + zscore(ice)
    covered = [i for i, value in enumerate(ice) if finite(value)]
    if not covered:
        return list(keys)
    missing = [i for i, value in enumerate(ice) if not finite(value)]
    head = sorted(covered, key=lambda i: (-fused[i], i))
    tail = sorted(missing, key=lambda i: (-fp[i], i))
    return [keys[i] for i in head + tail]


def formula_fusion(order, formulas, fp, gl, ice):
    """The treatment is label-free; truth is never passed to this function."""
    groups = defaultdict(list)
    for slot, key in enumerate(order):
        groups[formulas.get(key)].append(slot)
    revised = list(order)
    audits = []
    for formula, slots in groups.items():
        members = [order[i] for i in slots]
        eligible = bool(formula) and len(members) >= 2 and all(
            finite(gl.get(c)) and finite(ice.get(c)) for c in members
        )
        record = {
            "formula": formula,
            "slots_1based": [i + 1 for i in slots],
            "before": members,
            "eligible": eligible,
            "gl_covered": sum(finite(gl.get(c)) for c in members),
            "ice_covered": sum(finite(ice.get(c)) for c in members),
        }
        if eligible:
            scores = zscore([fp[c] for c in members])
            scores += CONFIG["gl_weight"] * zscore([gl[c] for c in members])
            scores += CONFIG["ice_weight"] * zscore([ice[c] for c in members])
            grid = [round(float(s) / CONFIG["tie_grid"]) for s in scores]
            ids = sorted(range(len(members)), key=lambda i: (-grid[i], i))
            after = [members[i] for i in ids]
            for slot, key in zip(slots, after):
                revised[slot] = key
            record.update(after=after, fused_scores=scores.tolist(), changed=members != after)
        else:
            record.update(after=members, changed=False)
        audits.append(record)
    return revised, audits


def rank(order, truth):
    return order.index(truth) + 1 if truth in order else None


def rr(r):
    return 1.0 / r if r is not None and r <= 25 else 0.0


def metrics(rows, method):
    ranks = [r[method + "_rank"] for r in rows]
    return {
        "n": len(ranks),
        "mrr25": float(np.mean([rr(r) for r in ranks])) if ranks else None,
        "top1": sum(r == 1 for r in ranks),
        "top25": sum(r is not None and r <= 25 for r in ranks),
    }


def compare(rows):
    b = np.array([rr(r["baseline_rank"]) for r in rows])
    t = np.array([rr(r["treatment_rank"]) for r in rows])
    delta = t - b
    rng = np.random.default_rng(CONFIG["seed"])
    draws = delta[rng.integers(0, len(delta), (CONFIG["bootstrap_draws"], len(delta)))].mean(1)
    return {
        "delta_mrr25": float(delta.mean()),
        "paired_molecule_bootstrap_95pct": np.quantile(draws, [0.025, 0.975]).tolist(),
        "improved": int((delta > 1e-12).sum()),
        "worsened": int((delta < -1e-12).sum()),
        "tied": int((np.abs(delta) <= 1e-12).sum()),
        "wrong_to_right_top1": int(((b != 1) & (t == 1)).sum()),
        "right_to_wrong_top1": int(((b == 1) & (t != 1)).sum()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/runs/009-formula-fusion")
    args = ap.parse_args()
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Persist the fixed settings before looking at any evaluation result.
    (args.output_dir / "config.json").write_text(json.dumps(CONFIG, ensure_ascii=False, indent=2))
    sources = {
        "inputs": "artifacts/legacy/f1-bundle/inputs.json",
        "gl": "artifacts/legacy/f1-scores/scores/f1d_gl_top100_split.json",
        "ice": "artifacts/legacy/f1-scores/scores/f1e-4_ice_top100_split.json",
        "filed": "artifacts/runs/006-f1/f1_per_molecule.json",
    }
    data, hashes = {}, {}
    for name, path in sources.items():
        blob = (ROOT / path).read_bytes()
        data[name] = json.loads(blob)
        hashes[name] = {"path": path, "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
    filed = {m["key"]: m for m in data["filed"]}
    rows, totals, source_omissions = [], Counter(), []
    alignment = {"canon_mismatches": [], "base_mismatches": [], "baseline_rr_mismatches": [], "formula_slot_mismatches": []}
    for molecule in data["inputs"]["molecules"]:
        mid = molecule["key"]
        keys = list(molecule["order_top100"])
        fp = dict(zip(keys, molecule["scores_top100"]))
        gl, ice = {}, {}
        for channel, target in (("gl", gl), ("ice", ice)):
            for c in keys:
                record = data[channel][mid].get(molecule["rep_smiles"][c])
                if record is None:
                    # Unsupported/failed structures may be absent from a cache;
                    # absence is missing evidence, never a fabricated zero score.
                    target[c] = None
                    source_omissions.append({"channel": channel, "key": mid, "candidate": c})
                    continue
                if record["canon"] != c:
                    alignment["canon_mismatches"].append([channel, mid, c])
                if abs(float(record["base"]) - fp[c]) > 1e-6:
                    alignment["base_mismatches"].append([channel, mid, c])
                target[c] = record.get("fused_merge")
        baseline = baseline_order(keys, [fp[c] for c in keys], [ice[c] for c in keys])
        treatment, group_audit = formula_fusion(baseline, molecule["formulas"], fp, gl, ice)
        if sorted(treatment) != sorted(baseline):
            raise RuntimeError("Candidate set changed: " + mid)
        if [molecule["formulas"].get(c) for c in treatment] != [molecule["formulas"].get(c) for c in baseline]:
            alignment["formula_slot_mismatches"].append(mid)
        # Labels enter only here, after both complete predicted orders are fixed.
        truth = molecule["truth_canon"]
        rb, rt = rank(baseline, truth), rank(treatment, truth)
        if abs(rr(rb) - filed[mid]["F1e_ice100_merge_rr"]) > 1e-12:
            alignment["baseline_rr_mismatches"].append(mid)
        eligible = [g for g in group_audit if g["eligible"]]
        changed = [g for g in eligible if g["changed"]]
        totals.update(molecules=1, candidates=len(keys), groups=len(group_audit),
                      multi_candidate_groups=sum(len(g["before"]) >= 2 for g in group_audit),
                      eligible_groups=len(eligible), eligible_candidates=sum(len(g["before"]) for g in eligible),
                      eligible_molecules=int(bool(eligible)), changed_groups=len(changed),
                      changed_molecules=int(baseline != treatment),
                      truth_in_eligible_group=int(any(truth in g["before"] for g in eligible)))
        rows.append({
            "key": mid, "truth": truth, "baseline_rank": rb, "treatment_rank": rt,
            "baseline_rr25": rr(rb), "treatment_rr25": rr(rt), "delta_rr25": rr(rt) - rr(rb),
            "eligible_groups": len(eligible), "changed_groups": len(changed),
            "baseline_top100": baseline, "treatment_top100": treatment, "group_audit": group_audit,
        })
    if any(alignment.values()):
        raise RuntimeError("Input/baseline mismatch: " + json.dumps(alignment))
    summary = {
        "config": CONFIG, "sources": hashes, "manual_checks": alignment,
        "source_candidate_omissions": source_omissions, "coverage": dict(totals),
        "baseline": metrics(rows, "baseline"), "treatment": metrics(rows, "treatment"),
        "paired": compare(rows),
        "all_worsened": [{k: r[k] for k in ("key", "baseline_rank", "treatment_rank", "delta_rr25")} for r in rows if r["delta_rr25"] < -1e-12],
        "all_improved": [{k: r[k] for k in ("key", "baseline_rank", "treatment_rank", "delta_rr25")} for r in rows if r["delta_rr25"] > 1e-12],
        "environment": {"device": "local CPU; no model forward", "python": platform.python_version(),
                        "numpy": np.__version__, "platform": platform.platform(),
                        "candidate_identity": "inherited input RDKit 2026.03.6 canonical; no recomputation here",
                        "finished_beijing": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                        "wall_seconds": round(time.monotonic() - started, 3)},
        "limitations": ["NP250 is repeatedly used development data", "Weights fixed before calculation; no search",
                        "This is a new rule inspired by public code, not public pipeline reproduction",
                        "No heldout, remote execution, training, or submission"],
    }
    (args.output_dir / "per_molecule.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: summary[k] for k in ("coverage", "baseline", "treatment", "paired", "manual_checks", "environment")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

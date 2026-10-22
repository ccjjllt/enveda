"""Manual, CPU-only repair of MB candidate identities in frozen 016 outputs.

Does not rerun FP/ICE, change candidate membership or scores, or overwrite 016.
The original ICE JSON remains SMILES-keyed; only its embedded canonical IDs
are updated to match the repaired input. Feature consumers must use SMILES or
the record's canon field, then reconstruct the actual v10 FP+ICE ordering.

Run: /opt/homebrew/bin/python3.11 scripts/017/repair_017_panel_inputs.py
"""
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

from rdkit import __version__ as rdkit_version
from rdkit.Chem.MolStandardize import rdMolStandardize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/016"))
from build_016_colab_inputs import canonical_key, sha_file
INPUT = ROOT / "artifacts/runs/016-colab/panel-016/inputs.json"
ICE = ROOT / "artifacts/runs/016-panel-scores/scores/panel016_ice_top100_split.json"
OUT = ROOT / "artifacts/runs/017-panel-inputs"


def main():
    original = json.loads(INPUT.read_text())
    original_ice = json.loads(ICE.read_text())
    repaired, scores = copy.deepcopy(original), copy.deepcopy(original_ice)
    te = rdMolStandardize.TautomerEnumerator()
    changes = []
    for mol, before in zip(repaired["molecules"], original["molecules"]):
        mapping = {}
        for old in before["order_top100"]:
            if old.startswith("MB539_"):
                new = canonical_key(before["rep_smiles"][old], old, te)
                assert new != old, (mol["key"], old, "canonicalization failed")
                mapping[old] = new
                changes.append({"query": mol["key"], "old": old, "new": new,
                                "position_1based": before["order_top100"].index(old) + 1})
        mol["order_top100"] = [mapping.get(k, k) for k in before["order_top100"]]
        assert len(set(mol["order_top100"])) == len(mol["order_top100"]), mol["key"]
        for field in ("rep_smiles", "formulas"):
            mol[field] = {mapping.get(k, k): v for k, v in before[field].items()}
        for old, new in mapping.items():
            sm = before["rep_smiles"][old]
            record = scores[mol["key"]][sm]
            assert record["canon"] == old
            record["canon"] = new
        assert mol["key"] == before["key"] and mol["truth_canon"] == before["truth_canon"]
        assert mol["scores_top100"] == before["scores_top100"]
        assert [mol["rep_smiles"][k] for k in mol["order_top100"]] == [
            before["rep_smiles"][k] for k in before["order_top100"]]
        assert [k == mol["truth_canon"] for k in mol["order_top100"]] == [
            k == before["truth_canon"] for k in before["order_top100"]]

    n_records, n_finite, n_covered, n_changed_records = 0, 0, 0, 0
    assert set(scores) == set(original_ice)
    for query, records in scores.items():
        assert set(records) == set(original_ice[query])
        covered = False
        for smiles, record in records.items():
            prior = original_ice[query][smiles]
            assert {k: v for k, v in record.items() if k != "canon"} == {
                k: v for k, v in prior.items() if k != "canon"}
            n_changed_records += record["canon"] != prior["canon"]
            value = record.get("fused_merge")
            if value is not None:
                assert math.isfinite(value)
                n_finite += 1
                covered = True
            n_records += 1
        n_covered += covered
    assert n_changed_records == len(changes)

    OUT.mkdir(parents=True, exist_ok=True)
    input_out, ice_out = OUT / "inputs.json", OUT / "ice_scores.json"
    input_out.write_text(json.dumps(repaired, allow_nan=False))
    ice_out.write_text(json.dumps(scores, allow_nan=False))
    nb = json.loads((ROOT / "notebooks/submissions/submission-v10/enveda-submission-v10.ipynb").read_text())
    c32 = "".join(nb["cells"][32]["source"])
    manifest = {
        "date_beijing": "2026-10-18", "operation": "candidate canonical ID repair only",
        "original": {str(INPUT.relative_to(ROOT)): sha_file(INPUT),
                     str(ICE.relative_to(ROOT)): sha_file(ICE)},
        "outputs": {str(input_out.relative_to(ROOT)): sha_file(input_out),
                    str(ice_out.relative_to(ROOT)): sha_file(ice_out)},
        "rdkit": rdkit_version,
        "identity_rule": "v10 cell32: MolFromSmiles, RemoveStereochemistry, default TautomerEnumerator.Canonicalize, InChIKey first block",
        "v10_cell32_sha256": hashlib.sha256(c32.encode()).hexdigest(),
        "changes": changes, "changed_query_count": len({c["query"] for c in changes}),
        "changed_candidate_count": len(changes), "duplicate_after_mapping": 0,
        "candidate_smiles_and_order_unchanged": True, "truth_match_vector_unchanged": True,
        "all_noncanon_ice_record_values_unchanged": True,
        "molecules": len(repaired["molecules"]), "candidate_records": n_records,
        "finite_fused_merge": n_finite, "molecules_with_fused_merge": n_covered,
        "gpu_runs": 0,
        "limitation": "Preserves original single-spectrum 016 inputs and Colab numerical environment; does not establish CPU/GPU or all-spectrum production parity.",
    }
    (OUT / "repair_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: manifest[k] for k in (
        "changed_query_count", "changed_candidate_count", "molecules", "candidate_records",
        "finite_fused_merge", "molecules_with_fused_merge", "gpu_runs")}, indent=2))
    print("outputs:", OUT)


if __name__ == "__main__":
    main()

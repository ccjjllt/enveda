"""Resume NPX and the four-arm comparison from the failed v1's upstream outputs."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path

from assemble import code_cell, pack


EARLIER_AUDITS = (
    "n035net_audit.json", "apex_extlib_audit.json", "p5tail_audit.json",
    "fr3to25_audit.json", "aprior_audit.json", "uniq_audit.json",
    "execution_receipt.json", "v43_execution_receipt.json",
    "four_inner_reference.json", "four_source_lists.json", "four_reference_audit.json",
)
V1_INPUTS = EARLIER_AUDITS + (
    "submission_pre_e180x.csv", "submission.csv", "npx_audit.json", "e180x_audit.json",
)


def ranks(path):
    with path.open(newline="") as stream:
        candidates = {row["molecule_id"]: row["smiles"].split(";") for row in csv.DictReader(stream)}
    return candidates


def checked_inputs(previous):
    def read(name):
        return json.loads((previous / name).read_text())

    npx, e180x = read("npx_audit.json"), read("e180x_audit.json")
    if not (npx["status"].startswith("failed closed") and "featurisation over budget" in npx["exception"]):
        raise ValueError("Expected the v1 NPX timeout with its input preserved")
    original = ranks(previous / "submission_pre_e180x.csv")
    current = ranks(previous / "submission.csv")
    changed = sum(v != current.get(k) for k, v in original.items())
    if not (e180x["status"] == "applied" and e180x["molecules"] == len(original)
            and changed == e180x["changed_lists"] and changed > 0
            and list(original) == list(current)
            and all(len(v) == 25 and Counter(v) == Counter(current[k]) for k, v in original.items())):
        raise ValueError("v1 candidate files do not match the E180X receipt")
    return {name: pack(previous / name) for name in V1_INPUTS}


def source(cell):
    return "".join(cell["source"])


def assemble_resume(launched, previous, output):
    metadata = json.loads((launched / "kernel-metadata.json").read_text())
    path = launched / metadata["code_file"]
    notebook = json.loads(path.read_text())
    previous_bytes = checked_inputs(previous)
    cells = notebook["cells"]
    fit, rank, score = cells[-3:]
    if [c["id"] for c in (fit, rank, score)] != ["four-arm-fit", "four-arm-rank", "four-arm-score"]:
        raise ValueError("Expected the launched four-arm notebook")
    prepare = source(cells[6])
    start, end = prepare.index("import os, sys, subprocess, importlib.metadata, time"), prepare.index("COMP = _four_root")
    prepare = prepare[start:end] + "COMP = _four_root\n"
    npx = source(cells[42])
    e180x, mark, export = source(cells[44]).partition("\nimport json\nimport shutil\nimport hashlib\n")
    if not mark:
        raise ValueError("Unable to identify the pinned baseline export part")
    e180x = e180x[:e180x.index('\nif "ROOT" in globals():')]
    npx = npx[:npx.index('\nif "ROOT" in globals():')]

    setup = code_cell("v1-recovery-setup", f"""
        import base64, hashlib, json, os, zlib
        from pathlib import Path
        ROOT = Path('/kaggle/working')
        INPUT = Path('/kaggle/input')
        IS_RERUN = PIPELINE_DONE = True
        ROOT.mkdir(parents=True, exist_ok=True)
        _recorded = ROOT / 'recovery_v1'
        _recorded.mkdir(exist_ok=True)
        _v1_inputs = {previous_bytes!r}
        for _name, (_packed, _expected) in _v1_inputs.items():
            _payload = zlib.decompress(base64.b64decode(_packed))
            if hashlib.sha256(_payload).hexdigest() != _expected:
                raise ValueError(f'v1 output digest changed: {{_name}}')
            (_recorded / _name).write_bytes(_payload)
        for _name in {EARLIER_AUDITS!r}:
            if _name != 'four_reference_audit.json':
                (ROOT / _name).write_bytes((_recorded / _name).read_bytes())
    """)
    run = code_cell("v1-recover-npx", r"""
        if _four_train_stats != json.loads((_recorded / 'four_reference_audit.json').read_text()):
            raise ValueError('Reference split differs from v1')
        _replay = Path('/tmp/replay_v1_e180x')
        _replay.mkdir(exist_ok=True)
        _pre = (_recorded / 'submission_pre_e180x.csv').read_bytes()
        (_replay / 'submission.csv').write_bytes(_pre)
        _old = ex_run(str(_replay), str(INPUT), True, cap_s=900.0)
        if _old['status'] != 'applied' or (_replay / 'submission.csv').read_bytes() != (_recorded / 'submission.csv').read_bytes():
            raise ValueError('Exact E180X replay did not reproduce v1')
        (ROOT / 'submission.csv').write_bytes(_pre)
        # NPX precedes the E180X definitions in the pinned notebook.
        _e180x_keyer = globals().pop('ex_keyer')
        try:
            _npx_audit = npx_run(str(ROOT), str(INPUT), True, cap_s=5400.0)
        finally:
            globals()['ex_keyer'] = _e180x_keyer
        print('[npx resume]', json.dumps(_npx_audit), flush=True)
        if not (_npx_audit['status'].startswith('applied') or _npx_audit['status'].startswith('gate failed')):
            raise RuntimeError(f'NPX recovery did not complete: {_npx_audit}')
        _ex_audit = ex_run(str(ROOT), str(INPUT), True, cap_s=900.0)
        print('[e180x resume]', json.dumps(_ex_audit), flush=True)
        (ROOT / 'four_arm_resume_receipt.json').write_text(json.dumps({
            'source': 'jiyujia/enveda-075-np-four-arm-0505/1',
            'source_output_sha256': {name: value[1] for name, value in _v1_inputs.items()},
            'v1_e180x_replay_matches_bytes': True,
            'v1_base_limits': {'npx_cap_s': 1200, 'npx_training_fraction': 0.7},
            'resume_limits': {'npx_cap_s': 5400, 'npx_training_fraction': 0.7},
        }, indent=2) + '\n')
    """)
    notebook["cells"] = [
        setup, code_cell("v1-reference-replay", prepare),
        code_cell("v1-replay-definitions", e180x + "\n\n" + npx),
        run, code_cell("v1-baseline-export", mark + export), fit, rank, score,
    ]
    for cell in notebook["cells"]:
        compile(source(cell), cell["id"], "exec")
    output.mkdir(parents=True, exist_ok=True)
    recovered = output / metadata["code_file"]
    recovered.write_text(json.dumps(notebook, ensure_ascii=False))
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"launched_sha256": sha256(path.read_bytes()).hexdigest(),
            "recovered_sha256": sha256(recovered.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("launched", "previous", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(assemble_resume(**vars(parser.parse_args())), indent=2))

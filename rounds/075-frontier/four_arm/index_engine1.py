"""Package the original Engine 1 ranking stages for a five-case forensic replay."""
import argparse
import ast
import base64
from hashlib import sha256
import json
from pathlib import Path
import textwrap
import zlib

import pandas as pd

from index_rank import TARGET_CHALLENGES, change


HERE = Path(__file__).parent


def extract(launched):
    metadata = json.loads((launched / "kernel-metadata.json").read_text())
    notebook = json.loads((launched / metadata["code_file"]).read_text())
    outer = ["".join(c["source"]) for c in notebook["cells"]]
    names = {"FROZEN_B64", "FROZEN_SHA256"}
    source = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(outer[12]).body
              if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id in names}
    inner = zlib.decompress(base64.b64decode(source["FROZEN_B64"]))
    if sha256(inner).hexdigest() != source["FROZEN_SHA256"]:
        raise ValueError("Original Engine 1 notebook digest mismatch")
    return outer, ["".join(c["source"]) for c in json.loads(inner)["cells"]], metadata


def build(launched, archive, panel, output, device="t4"):
    outer, inner, metadata = extract(launched)
    labels = pd.read_parquet(panel / "private/labels.parquet")
    cases = labels[labels.challenges.isin(TARGET_CHALLENGES)]
    scans = pd.read_parquet(archive / "effective_query_spectra.parquet")
    query = scans[scans.molecule_id.isin(cases.molecule_id)]
    if (set(cases.challenges) != TARGET_CHALLENGES or len(query) != 5
            or not cases.molecule_id.is_unique or not query.molecule_id.is_unique):
        raise ValueError("Expected the original five measured queries")
    archived = json.loads((archive / "e004_candidate_inputs.json").read_text())["base_tail"]
    audit = {
        "cases": {row.molecule_id: dict(challenge=row.challenges, truth_key=row.truth_key,
                                         truth_smiles=row.truth_smiles) for row in cases.itertuples(index=False)},
        "base_tail": {key: archived[key] for key in cases.molecule_id},
        "reference_mask": json.loads((archive / "four_reference_audit.json").read_text()),
        "extension": json.loads((archive / "four_inner_reference.json").read_text()),
    }
    reference_prep = outer[6][outer[6].index("import os, sys, subprocess, importlib.metadata, time"):
                              outer[6].index("COMP = _four_root")] + "COMP = str(_four_root)\n"
    setup = textwrap.dedent("""
        import base64, hashlib, json, os, time, zlib
        from pathlib import Path
        from hashlib import sha256
        ROOT = Path('/kaggle/working')
        ROOT.mkdir(parents=True, exist_ok=True)
        os.chdir(ROOT)
        _query = zlib.decompress(base64.b85decode(QUERY))
        if sha256(_query).hexdigest() != QUERY_SHA256:
            raise ValueError('Measured spectra failed checksum')
    """)
    check_reference = textwrap.dedent("""
        if _four_train_stats != AUDIT['reference_mask']:
            raise ValueError('Holdout mask differs from the v1 engine')
    """)
    check_extension = textwrap.dedent("""
        for field in ('added', 'keys', 'base_rows', 'validation_removed', 'error'):
            if _EXTERNAL_REFERENCE_STATE[field] != AUDIT['extension'][field]:
                raise ValueError(f'E1 spectral extension differs from v1: {field}')
        (_four_root / 'test.parquet').write_bytes(_query)
    """)
    rank = change(inner[39], "                order = np.argsort(-score, kind='stable')\n",
                  "                order = np.argsort(-score, kind='stable')\n"
                  "                trace_engine1(mid, target, C, score, order)\n")
    is_gpu = device == "t4"
    device_check = ("assert dev == 'cuda' and 'T4' in torch.cuda.get_device_name(0)"
                    if is_gpu else "assert dev == 'cpu'")
    all_code = [setup, reference_prep, check_reference, inner[5], inner[26], inner[36],
                check_extension, inner[37], (HERE / "index_engine1_trace.source").read_text(),
                device_check, "install_engine1_recording(E)",
                rank, "write_engine1_report()"]
    raw = query.to_parquet(index=False)
    prefix = "QUERY = " + repr(base64.b85encode(zlib.compress(raw, 9)).decode())
    prefix += "\nQUERY_SHA256 = " + repr(sha256(raw).hexdigest())
    prefix += "\nAUDIT = " + repr(audit)
    for cell in all_code:
        compile(cell, "engine1-audit-cell", "exec")
    replay = prefix + "\n\n" + "\n\n".join(all_code) + "\n"
    compile(replay, "engine1-replay.py", "exec")
    payload = base64.b85encode(zlib.compress(replay.encode(), 9)).decode()
    program = "REPLAY = " + repr(payload) + "\n"
    program += textwrap.dedent("""
        import base64
        import os
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        import zlib

        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / 'engine1-replay.py'
            script.write_bytes(zlib.decompress(base64.b85decode(REPLAY)))
            env = dict(os.environ, PYTHONHASHSEED='0', PYTHONUNBUFFERED='1')
            subprocess.run([sys.executable, str(script)], env=env, check=True)
    """)
    compile(program, "audit.py", "exec")
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.py").write_text(program)
    slug = "enveda-075-five-e1-rank-audit" + ("" if is_gpu else "-cpu")
    metadata.update(id="jiyujia/" + slug, title=slug, code_file="audit.py", kernel_type="script",
                    enable_gpu=is_gpu)
    if not is_gpu:
        metadata.pop("machine_shape", None)
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return dict(script_sha256=sha256(program.encode()).hexdigest(), query_sha256=sha256(raw).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("launched", "archive", "panel", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--device", choices=("t4", "cpu"), default="t4")
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))

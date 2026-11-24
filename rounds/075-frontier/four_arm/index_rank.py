"""Package an offline five-case C34 rank replay from the archived v1 runner."""
import argparse
import ast
import base64
from hashlib import sha256
import json
from pathlib import Path
import textwrap
import zlib

import pandas as pd


HERE = Path(__file__).parent
TARGET_CHALLENGES = {"challenge-011", "challenge-033", "challenge-034", "challenge-035", "challenge-178"}
ELECTRON_MASS_DA = 0.000548579909


def change(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Archived C34 source no longer matches {old!r}")
    return source.replace(old, new)


def native_sources(launched, archive):
    metadata = json.loads((launched / "kernel-metadata.json").read_text())
    notebook = json.loads((launched / metadata["code_file"]).read_text())
    assignments = ast.parse("".join(notebook["cells"][12]["source"])).body
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in assignments
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id in {"FROZEN_B64", "FROZEN_SHA256"}}
    payload = zlib.decompress(base64.b64decode(values["FROZEN_B64"]))
    if sha256(payload).hexdigest() != values["FROZEN_SHA256"]:
        raise ValueError("The original pipeline digest disagrees with the launch recipe")
    recovered = json.loads(payload)
    assignments = ast.parse("".join(recovered["cells"][34]["source"])).body
    code = {n.targets[0].id: ast.literal_eval(n.value) for n in assignments
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in {"CORE", "V17_PATCH", "RUNNER"}}
    original = {"probe_core2.py": code["CORE"] + code["V17_PATCH"], "pc_runner.py": code["RUNNER"]}
    if any(text.encode() != (archive / filename).read_bytes() for filename, text in original.items()):
        raise ValueError("Source files extracted from the v1 notebook differ from the remote v1 output")
    return code, original, metadata


def instrument_worker(code):
    patch = code["V17_PATCH"]
    patch = change(patch, "    if n == 0:\n",
                   "    diag.update(store_window_counts=dict(_W['last_window_counts']), truth_hits=[])\n"
                   "    if n == 0:\n")
    patch = change(patch, "    E = np.zeros((n, len(raw_e)), np.float32); ok = np.zeros(n, bool)\n",
                   "    truth_rows = []\n"
                   "    E = np.zeros((n, len(raw_e)), np.float32); ok = np.zeros(n, bool)\n")
    patch = change(patch, "        E[i] = gen.GetFingerprintAsNumPy(m)[raw_e]; ok[i] = True\n",
                   "        E[i] = gen.GetFingerprintAsNumPy(m)[raw_e]; ok[i] = True\n"
                   "        if _audit_is_truth(mid, m):\n"
                   "            truth_rows.append(i)\n")
    patch = change(patch, "    use_pop = ('ls' in _W) and _POP_LAM > 0\n",
                   "    partial_indices = top1.copy()\n"
                   "    prior_indices = np.empty(0, dtype=int)\n"
                   "    use_pop = ('ls' in _W) and _POP_LAM > 0\n")
    patch = change(patch, "        top1 = np.union1d(top1, np.argpartition(-pp, ku - 1)[:ku])\n",
                   "        prior_indices = np.argpartition(-pp, ku - 1)[:ku]\n"
                   "        top1 = np.union1d(top1, prior_indices)\n"
                   "    diag['n_prior_selected'] = len(prior_indices)\n"
                   "    diag['n_coarse_union'] = len(top1)\n")
    patch = change(patch, "    if not idx:\n        diag['secs'] = time.time() - t0\n",
                   "    if not idx:\n"
                   "        _audit_collect(diag, target, smis, ok, sc1, partial_indices, pop, prior_indices,\n"
                   "                       top1, np.asarray(idx, dtype=int), np.empty(0), np.empty(0),\n"
                   "                       np.empty(0, dtype=int), truth_rows)\n"
                   "        diag['secs'] = time.time() - t0\n")
    patch = change(patch, "    order = np.argsort(-sc, kind='stable')\n",
                   "    order = np.argsort(-sc, kind='stable')\n"
                   "    _audit_collect(diag, target, smis, ok, sc1, partial_indices, pop, prior_indices,\n"
                   "                   top1, idx, fz, sc, order, truth_rows)\n")
    return code["CORE"] + patch + "\n" + (HERE / "index_rank_trace.source").read_text()


def instrument_runner(source):
    source = change(source, "        tol = target * 10e-6\n",
                    "        cases = [(mid, target)]\n"
                    "        cases += [(mid + '@' + name, mass) for name, mass in\n"
                    "                  pc._AUDIT_REQUEST['cases'][mid]['alternative_centres'].items()]\n"
                    "        for run_id, target in cases:\n"
                    "            tol = target * 10e-6\n")
    begin = source.index("        a = np.searchsorted(pmass, target - tol, 'left')")
    end = source.index("    del bank; torch.cuda.empty_cache()", begin)
    part = source[begin:end].replace("extra[mid]", "extra[run_id]").replace("tasks.append((mid,", "tasks.append((run_id,")
    source = source[:begin] + textwrap.indent(part, "    ") + source[end:]
    marker = "            if (k + 1) % 25 == 0 or k + 1 == len(tasks):\n"
    return change(source, marker, "            res[mid]['diagnostic'] = {k: v for k, v in d.items() "
                  "if k != 'promotion_evidence'}\n" + marker)


def build(launched, archive, panel, output):
    code, original, metadata = native_sources(launched, archive)
    labels = pd.read_parquet(panel / "private/labels.parquet")
    cases = labels[labels.challenges.isin(TARGET_CHALLENGES)]
    if set(cases.challenges) != TARGET_CHALLENGES or not cases.molecule_id.is_unique:
        raise ValueError("Expected the original five omitted CASMI structures")
    scans = pd.read_parquet(archive / "effective_query_spectra.parquet")
    query = scans[scans.molecule_id.isin(cases.molecule_id)]
    archived = json.loads((archive / "pc_lists.json").read_text())
    audit = {}
    for row in cases.itertuples(index=False):
        centres = {}
        if row.challenges == "challenge-011":
            observed = float(query.loc[query.molecule_id.eq(row.molecule_id), "precursor_mz"].median())
            centres = {"measured_ion_mz": observed, "measured_mz_plus_electron": observed + ELECTRON_MASS_DA}
        audit[row.molecule_id] = dict(challenge=row.challenges, truth_smiles=row.truth_smiles,
                                      truth_key=row.truth_key, alternative_centres=centres)
    files = {
        "query.parquet": query.to_parquet(index=False),
        "archive.json": json.dumps({k: archived[k] for k in audit}).encode(),
        "request.json": json.dumps({"cases": audit}).encode(),
        "probe_core2.py": instrument_worker(code).encode(),
        "pc_runner.py": instrument_runner(code["RUNNER"]).encode(),
        "remote.py": (HERE / "index_rank_run.source").read_bytes(),
    }
    blobs = {name: (base64.b85encode(zlib.compress(raw, 9)).decode(), sha256(raw).hexdigest())
             for name, raw in files.items()}
    original_digests = {name: sha256(text.encode()).hexdigest() for name, text in original.items()}
    program = "BLOBS = " + repr(blobs) + "\nORIGINAL_DIGESTS = " + repr(original_digests) + "\n"
    program += textwrap.dedent("""
        import base64
        from hashlib import sha256
        from pathlib import Path
        import runpy
        import tempfile
        import zlib

        with tempfile.TemporaryDirectory() as tmp:
            for name, (data, expected) in BLOBS.items():
                raw = zlib.decompress(base64.b85decode(data))
                if sha256(raw).hexdigest() != expected:
                    raise ValueError(f'Mismatched embedded task file {name}')
                (Path(tmp) / name).write_bytes(raw)
            runpy.run_path(str(Path(tmp) / 'remote.py'))['run'](Path(tmp), ORIGINAL_DIGESTS)
    """)
    for filename in ("probe_core2.py", "pc_runner.py", "remote.py"):
        compile(files[filename], filename, "exec")
    compile(program, "audit.py", "exec")
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.py").write_text(program)
    slugs = {"casmi26-pubchem-popularity-prior", "casmi26-attributed-v4g-reproduction-assets",
             "casmi26-fold-safe-fpnet", "rdkit-2026-3-3-wheel"}
    metadata.update(
        id="jiyujia/enveda-075-five-c34-rank-audit", title="enveda-075-five-c34-rank-audit",
        code_file="audit.py", kernel_type="script", competition_sources=[], kernel_sources=[],
        dataset_sources=[s for s in metadata["dataset_sources"] if s.split("/")[-1] in slugs],
    )
    if {s.split("/")[-1] for s in metadata["dataset_sources"]} != slugs:
        raise ValueError("Missing frozen C34 input mount")
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"archive_source_sha256": original_digests, "molecule_ids": list(audit),
            "script_sha256": sha256(program.encode()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("launched", "archive", "panel", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))

"""Append locked training, ranking, and evaluation stages to the blind v7 baseline."""
import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import textwrap
import zlib


def pack(path):
    raw = path.read_bytes()
    return base64.b64encode(zlib.compress(raw, level=9)).decode(), sha256(raw).hexdigest()


def code_cell(name, source):
    return {"cell_type": "code", "id": name, "execution_count": None, "metadata": {},
            "outputs": [], "source": textwrap.dedent(source).strip().splitlines(keepends=True)}


def assemble(base, panel, output):
    metadata = json.loads((base / "kernel-metadata.json").read_text())
    notebook = json.loads((base / metadata["code_file"]).read_text())
    experiment = Path(__file__).parent / "models/experiment.py"
    source, source_hash = pack(experiment)
    exclude, exclude_hash = pack(panel / "private/holdout.json")
    labels, labels_hash = pack(panel / "private/labels.parquet")
    cells = notebook["cells"]
    if any(cell.get("id") == "four-arm-fit" for cell in cells):
        raise ValueError("Already assembled")
    cells.append(code_cell("four-arm-fit", f"""
        from pathlib import Path
        from hashlib import sha256
        import base64, json, os, subprocess, sys, zlib
        _out = Path('/kaggle/working/four_arm')
        _scratch = Path('/tmp/four_arm_experiment')
        _comp = Path('/tmp/four_arm_comp')
        _out.mkdir(parents=True, exist_ok=True)
        _scratch.mkdir(parents=True, exist_ok=True)
        _base = Path('/kaggle/working/four_arm_baseline.csv')
        _query = _comp / 'test.parquet'
        def _hash(path):
            return sha256(path.read_bytes()).hexdigest()
        def _restore(path, blob, expected):
            raw = zlib.decompress(base64.b64decode(blob))
            if sha256(raw).hexdigest() != expected:
                raise ValueError(f'Input hash mismatch: {{path.name}}')
            path.write_bytes(raw)
        _sealed = {{'baseline': _hash(_base), 'queries': _hash(_query)}}
        (_out / 'locked_inputs.json').write_text(json.dumps(_sealed, indent=2))
        _program = _scratch / 'experiment.py'
        _restore(_program, {source!r}, {source_hash!r})
        _exclude = _scratch / 'holdout.json'
        _restore(_exclude, {exclude!r}, {exclude_hash!r})
        _env = os.environ.copy()
        _check = [sys.executable, '-c', 'from rdkit import rdBase; print(rdBase.rdkitVersion)']
        _rdkit_check = subprocess.run(_check, text=True, env=_env, capture_output=True)
        if _rdkit_check.returncode or _rdkit_check.stdout.strip() != '2026.03.3':
            wheel_tag = f'cp{{sys.version_info.major}}{{sys.version_info.minor}}'
            wheels = sorted(Path('/kaggle/input').rglob(f'rdkit-2026.3.3-{{wheel_tag}}-*.whl'))
            if not wheels:
                raise RuntimeError('Competition RDKit wheel missing')
            site = _scratch / 'metric_site'
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index',
                            '--target', str(site), str(wheels[0])], check=True)
            _env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(site), _env.get('PYTHONPATH')]))
        def _stage(*args):
            subprocess.run([sys.executable, str(_program), *map(str, args)], env=_env, check=True)
        _stage('fit', '--train', _comp / 'train.parquet', '--exclude', _exclude, '--out', _out / 'model')
    """))
    cells.append(code_cell("four-arm-rank", """
        if (_hash(_base), _hash(_query)) != (_sealed['baseline'], _sealed['queries']):
            raise RuntimeError('Frozen candidate list or query spectra changed during training')
        _stage('rank', '--model', _out / 'model', '--queries', _query,
               '--baseline', _base, '--out', _out / 'rank')
        _sealed['ranks'] = _hash(_out / 'rank/ranks.parquet')
        (_out / 'locked_inputs.json').write_text(json.dumps(_sealed, indent=2))
    """))
    cells.append(code_cell("four-arm-score", f"""
        if _hash(_out / 'rank/ranks.parquet') != _sealed['ranks']:
            raise RuntimeError('Rankings changed before evaluation')
        _labels = _scratch / 'labels.parquet'
        _restore(_labels, {labels!r}, {labels_hash!r})
        _stage('score', '--ranks', _out / 'rank', '--labels', _labels, '--out', _out / 'score')
        print('FOUR_ARM_COMPLETE', (_out / 'score/metrics.json').read_text(), flush=True)
    """))
    for cell in cells[-3:]:
        compile("".join(cell["source"]), cell["id"], "exec")
    metadata.update(id="jiyujia/enveda-075-np-four-arm-0505",
                    title="enveda-075-np-four-arm-0505", code_file="four-arm.ipynb",
                    is_private=True, enable_gpu=True, enable_internet=False, machine_shape="NvidiaTeslaT4")
    output.mkdir(parents=True, exist_ok=True)
    (output / metadata["code_file"]).write_text(json.dumps(notebook, ensure_ascii=False))
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"fit_source_sha256": source_hash, "exclusions_sha256": exclude_hash,
            "labels_sha256": labels_hash, "notebook_sha256": sha256((output / metadata["code_file"]).read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "panel", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(assemble(**vars(parser.parse_args())), indent=2))

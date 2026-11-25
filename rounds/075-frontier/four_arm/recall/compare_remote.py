"""Build the private Kaggle CPU comparison after all four blind rankings are sealed."""

import argparse
import json
from pathlib import Path

from build_recall import FOUR_ARM, assemble, load_module


SLUG = 'enveda-075-np-recall-compare'
METRIC_BOOTSTRAP = load_module('recall_metric_wheel', FOUR_ARM.parent / 'virtual_np/build.py').METRIC_BOOTSTRAP


def build(manifest, labels, metadata, output):
    receipt = json.loads(manifest.read_text())
    jobs = receipt['jobs']
    replay = receipt['tail_replay']
    direct = {arm: jobs[arm]['slug'] for arm in ('late_dedupe', 'joint')}
    sources = [f"jiyujia/{replay['slug']}/{replay['version']}"]
    sources.extend(f"jiyujia/{slug}/{jobs[arm]['version']}" for arm, slug in direct.items())
    expected = receipt['panel']['output_sha256']['private/labels.parquet']
    label_blob = assemble.pack(labels)
    if label_blob[1] != expected:
        raise ValueError('The labels have changed since the panel was frozen')
    source = f'''
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zlib
from zipfile import ZipFile

inputs = Path('/kaggle/input')
root = Path('/kaggle/working')
runs = Path('/tmp/sealed-075')
runs.mkdir(exist_ok=True)
{METRIC_BOOTSTRAP}

def save(path, payload):
    blob, digest = payload
    contents = zlib.decompress(base64.b64decode(blob))
    if hashlib.sha256(contents).hexdigest() != digest:
        raise ValueError(f'Artifact digest differs: {{path}}')
    path.write_bytes(contents)

for arm in ('baseline', 'early_score'):
    archive, = (p for p in inputs.rglob(f'{{arm}}.zip') if {replay['slug']!r} in p.parts)
    with ZipFile(archive) as source:
        members = source.namelist()
        if any(Path(name).name != name for name in members):
            raise ValueError('The replay archive should contain only named evidence files')
        source.extractall(runs / arm)

for arm, slug in {direct!r}.items():
    report, = (p for p in inputs.rglob('pilot_arm_receipt.json') if slug in p.parts)
    target = runs / arm
    target.mkdir()
    shutil.copyfile(report, target / report.name)
    for name in json.loads(report.read_text())['hashes']:
        shutil.copyfile(report.parent / name, target / name)

program = Path('/tmp/scorer.py')
private = Path('/tmp/labels.parquet')
save(program, {assemble.pack(FOUR_ARM / 'retrieval_pilot_score.py')!r})
save(private, {label_blob!r})
subprocess.check_call([sys.executable, str(program), 'score', '--runs', str(runs),
                       '--out', str(root), '--labels', str(private)])
'''
    compile(source, SLUG, 'exec')
    book = {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
            'cells': [assemble.code_cell('compare-sealed-arms', source)]}
    options = json.loads(metadata.read_text())
    options.update(id=f'jiyujia/{SLUG}', title=SLUG, code_file='compare.ipynb',
                   is_private=True, enable_gpu=False, enable_internet=False, competition_sources=[],
                   dataset_sources=['metric/rdkit-2026-3-3-wheel'], kernel_sources=sources)
    options.pop('machine_shape', None)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'kernel-metadata.json').write_text(json.dumps(options, indent=2) + '\n')
    (output / 'compare.ipynb').write_text(json.dumps(book, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('manifest', 'labels', 'metadata', 'output'):
        parser.add_argument('--' + arg, required=True, type=Path)
    build(**vars(parser.parse_args()))

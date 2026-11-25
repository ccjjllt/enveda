"""Build the private Kaggle CPU comparison after the virtual candidate list is sealed."""

import argparse
import json
from pathlib import Path

from build import HERE, IMAGE, METRIC_BOOTSTRAP, SLUG, WHEEL, pack
from build_score import FULL_MAIN_SCORE_SLUG, score_slug_for


BASELINE = 'enveda-075-np-recall-reseal'


def build(panel, out, generation_version, baseline_version, score_version=None, generation_kernel=None,
          candidate_source='proposals'):
    known_main = candidate_source == 'known_main'
    if known_main and not score_version:
        raise ValueError('The known-main comparison requires a sealed known-main scoring version')
    manifest = json.loads((panel / 'panel_manifest.json').read_text())
    label_blob = pack((panel / 'private/labels.parquet').read_bytes())
    if label_blob[1] != manifest['output_sha256']['private/labels.parquet']:
        raise ValueError('Private labels are no longer the ones approved for the 60-query comparison')
    files = {
        'evaluate.py': pack((HERE / 'evaluate.py').read_bytes()),
        'retrieval_pilot_score.py': pack((HERE.parent / 'four_arm/retrieval_pilot_score.py').read_bytes()),
        'labels.parquet': label_blob,
    }
    dependency = generation_kernel or f'jiyujia/{SLUG}/{generation_version}'
    mount_slug = dependency.split('/')[1]
    fpnet_slug = score_slug_for(mount_slug, candidate_source) if score_version else None
    fpnet_kernel = f'jiyujia/{fpnet_slug}/{score_version}' if fpnet_slug else None
    compare_slug = f'{fpnet_slug if fpnet_kernel else mount_slug}-compare'
    if fpnet_kernel and candidate_source == 'known_main':
        compare_slug = f'{SLUG}-full-main-shortlist' if fpnet_slug == FULL_MAIN_SCORE_SLUG else f'{SLUG}-main-shortlist'
    branch_args = ['--candidate-source', 'known_main'] if known_main else []
    code = f'''
import base64
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib

inputs = Path('/kaggle/input')
seal, = (p for p in inputs.rglob('baseline.zip') if {BASELINE!r} in p.parts)
receipt, = (p for p in inputs.rglob('blind_virtual_np_receipt.json') if {mount_slug!r} in p.parts)
{METRIC_BOOTSTRAP}

with tempfile.TemporaryDirectory(prefix='virtual-coverage-') as temp:
    work = Path(temp)
    for name, (value, expected) in {files!r}.items():
        raw = zlib.decompress(base64.b64decode(value))
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f'Private evaluation artifact changed: {{name}}')
        (work / name).write_bytes(raw)
    command = [sys.executable, str(work / 'evaluate.py'), '--baseline', str(seal),
               '--run', str(receipt.parent), '--labels', str(work / 'labels.parquet'),
               '--labels-sha256', {label_blob[1]!r}, '--out', '/kaggle/working', *{branch_args!r}]
    if {bool(fpnet_kernel)!r}:
        fpnet, = (p for p in inputs.rglob('fpnet_a_score_receipt.json') if {fpnet_slug!r} in p.parts)
        command.extend(['--fpnet-run', str(fpnet.parent)])
    subprocess.run(command, check=True)
'''
    compile(code, compare_slug, 'exec')
    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                    cells=[dict(cell_type='code', id='score-virtual-coverage', execution_count=None,
                                metadata={}, outputs=[], source=code.strip().splitlines(keepends=True))])
    dependencies = [dependency, f'jiyujia/{BASELINE}/{baseline_version}']
    if fpnet_kernel:
        dependencies.append(fpnet_kernel)
    metadata = dict(id=f'jiyujia/{compare_slug}', title=compare_slug, code_file='compare.ipynb', language='python',
                    kernel_type='notebook', is_private=True, enable_gpu=False, enable_internet=False,
                    dataset_sources=[WHEEL], competition_sources=[], docker_image=IMAGE,
                    kernel_sources=dependencies)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'compare.ipynb').write_text(json.dumps(notebook, separators=(',', ':')) + '\n')
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    generation = parser.add_mutually_exclusive_group(required=True)
    generation.add_argument('--generation-version', type=int)
    generation.add_argument('--generation-kernel')
    parser.add_argument('--baseline-version', type=int, default=2)
    parser.add_argument('--score-version', type=int)
    parser.add_argument('--candidate-source', choices=('proposals', 'known_main'), default='proposals')
    build(**vars(parser.parse_args()))

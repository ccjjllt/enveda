"""Build isolated blind relisting and optional sealed private comparison notebooks."""

import argparse
import json
from pathlib import Path

from build import HERE, IMAGE, METRIC_BOOTSTRAP, WHEEL, pack


def mount_slug(reference):
    parts = reference.split('/')
    if len(parts) != 3 or not parts[0] or not parts[1] or not parts[2].isdigit() or int(parts[2]) < 1:
        raise ValueError('Use a versioned owner/notebook/version Kaggle reference')
    return parts[1]


def build(out, baseline_kernel, generation_kernel=None, score_kernel=None, relist_kernel=None, panel=None):
    baseline_slug = mount_slug(baseline_kernel)
    dependencies = [baseline_kernel]
    files = {
        'evaluate.py': pack((HERE / 'evaluate.py').read_bytes()),
        'retrieval_pilot_score.py': pack((HERE.parent / 'four_arm/retrieval_pilot_score.py').read_bytes()),
        'relist_unseen.py': pack((HERE / 'relist_unseen.py').read_bytes()),
    }
    if relist_kernel:
        if panel is None or generation_kernel is not None or score_kernel is not None:
            raise ValueError('Only the sealed relist, baseline and private panel supply the comparison')
        relist_slug = mount_slug(relist_kernel)
        dependencies.append(relist_kernel)
        slug = f'{relist_slug}-compare'
        digest = json.loads((panel / 'panel_manifest.json').read_bytes())['output_sha256']['private/labels.parquet']
        labels = pack((panel / 'private/labels.parquet').read_bytes())
        if labels[1] != digest:
            raise ValueError('Private labels differ from the frozen manifest')
        files['labels.parquet'] = labels
        files['compare_unseen_relist.py'] = pack((HERE / 'compare_unseen_relist.py').read_bytes())
        runner = f'''
    relist = mounted('unseen_relist_receipt.json', {relist_slug!r}).parent
    command = [sys.executable, str(scratch / 'compare_unseen_relist.py'), '--baseline-zip', str(seal),
               '--relist', str(relist), '--labels', str(scratch / 'labels.parquet'), '--labels-sha256', {digest!r},
               '--out', '/kaggle/working']
'''
    else:
        if panel is not None or generation_kernel is None or score_kernel is None:
            raise ValueError('Blind relisting requires the two sealed candidate inputs and no private panel')
        generation_slug, score_slug = mount_slug(generation_kernel), mount_slug(score_kernel)
        dependencies.extend((generation_kernel, score_kernel))
        scope = 'full-main' if generation_slug.endswith('-full-main-window') else 'rule-main'
        slug = f'enveda-075-{scope}-unseen'
        runner = f'''
    generation = mounted('blind_virtual_np_receipt.json', {generation_slug!r}).parent
    scores = mounted('fpnet_a_score_receipt.json', {score_slug!r}).parent
    command = [sys.executable, str(scratch / 'relist_unseen.py'), '--baseline-zip', str(seal),
               '--generation', str(generation), '--scores', str(scores), '--out', '/kaggle/working']
'''
    runtime = f'''
import base64
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib

inputs = Path('/kaggle/input')
{METRIC_BOOTSTRAP}

def mounted(name, slug):
    file, = (p for p in inputs.rglob(name) if slug in p.parts)
    return file

seal = mounted('baseline.zip', {baseline_slug!r})
with tempfile.TemporaryDirectory(prefix='virtual-unseen-relist-') as directory:
    scratch = Path(directory)
    for name, (blob, expected) in {files!r}.items():
        raw = zlib.decompress(base64.b64decode(blob))
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f'Relisting input changed: {{name}}')
        (scratch / name).write_bytes(raw)
{runner.rstrip()}
    subprocess.run(command, check=True)
'''
    compile(runtime, slug, 'exec')
    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                    cells=[dict(cell_type='code', id='unseen-relist', execution_count=None, metadata={},
                                outputs=[], source=runtime.strip().splitlines(keepends=True))])
    meta = dict(id=f'jiyujia/{slug}', title=slug, code_file='relist.ipynb', language='python',
                kernel_type='notebook', is_private=True, enable_gpu=False, enable_internet=False,
                dataset_sources=[WHEEL], kernel_sources=dependencies, competition_sources=[], docker_image=IMAGE)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'relist.ipynb').write_text(json.dumps(notebook, separators=(',', ':')) + '\n')
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline-kernel', default='jiyujia/enveda-075-np-recall-reseal/2')
    parser.add_argument('--generation-kernel')
    parser.add_argument('--score-kernel')
    parser.add_argument('--relist-kernel')
    parser.add_argument('--panel', type=Path)
    build(**vars(parser.parse_args()))

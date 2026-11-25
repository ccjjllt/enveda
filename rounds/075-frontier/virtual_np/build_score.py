"""Build an offline, label-blind FPNet A comparison for original and generated candidates."""

import argparse
import ast
import base64
from hashlib import sha256
import json
from pathlib import Path

from build import ARCHIVED_C34_SHA, ASSETS, HERE, IMAGE, METRIC_BOOTSTRAP, SLUG, WHEEL, pack
from build_main_window import FULL_MAIN_SCOPE, FULL_MAIN_WINDOW_SLUG


SCORE_SLUG = f'{SLUG}-fpnet-a'
KNOWN_MAIN_SCORE_SLUG = f'{SCORE_SLUG}-known-main'
FULL_MAIN_SCORE_SLUG = f'{SLUG}-full-main-fpnet-a'
PABLO_SHA = 'a183882abe0b131581109455e1b704ec6d93f27fb99e7b5c2f07d40063e9c5af'


def literals(source, names):
    values = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) in names:
            values[node.targets[0].id] = ast.literal_eval(node.value)
    if set(values) != set(names):
        raise ValueError('The published C34 sources have changed')
    return values


def baseline_core(published):
    data = published.read_bytes()
    if sha256(data).hexdigest() != PABLO_SHA:
        raise ValueError('Expected the public Pablo v29 notebook')
    outer = json.loads(data)['cells']
    inner = literals(''.join(outer[10]['source']), ('FROZEN_B64', 'FROZEN_SHA256'))
    decoded = base64.b64decode(inner['FROZEN_B64'])
    if sha256(decoded).hexdigest() != inner['FROZEN_SHA256']:
        raise ValueError('The exact native C34 notebook could not be verified')
    source = json.loads(decoded)['cells'][34]['source']
    pieces = literals(''.join(source), ('CORE', 'V17_PATCH'))
    return (pieces['CORE'] + pieces['V17_PATCH']).encode()


def score_slug_for(mount_slug, candidate_source):
    if mount_slug == FULL_MAIN_WINDOW_SLUG:
        if candidate_source != 'known_main':
            raise ValueError('The direct full-main scan only supplies the known-main artifact')
        return FULL_MAIN_SCORE_SLUG
    return KNOWN_MAIN_SCORE_SLUG if candidate_source == 'known_main' else SCORE_SLUG


def build(panel, pc, published, out, generation_version=None, generation_kernel=None, candidate_source='proposals'):
    blind = (panel / 'blind/test.parquet').read_bytes()
    manifest = json.loads((panel / 'panel_manifest.json').read_text())
    if sha256(blind).hexdigest() != manifest['output_sha256']['blind/test.parquet']:
        raise ValueError('Blind candidate inputs no longer match the validation manifest')
    native = pc.read_bytes()
    if sha256(native).hexdigest() != ARCHIVED_C34_SHA:
        raise ValueError('Use the verified archived C34 candidate order')
    fields = {mid: {'target': entry['target'], 'pc': entry['pc']} for mid, entry in json.loads(native).items()}
    if len(fields) != 60:
        raise ValueError('Expected 60 archived C34 candidate lists')
    dependency = generation_kernel or f'jiyujia/{SLUG}/{generation_version}'
    mount_slug = dependency.split('/')[1]
    candidate_artifact = f'{candidate_source}.parquet'
    known_main = candidate_source == 'known_main'
    score_slug = score_slug_for(mount_slug, candidate_source)
    branch_args = ['--candidate-source', 'known_main'] if known_main else []
    files = {
        'test.parquet': pack(blind),
        'c34.json': pack(json.dumps(dict(archive_sha256=ARCHIVED_C34_SHA, molecules=fields)).encode()),
        'score.py': pack((HERE / 'score.py').read_bytes()),
        'probe_core2.py': pack(baseline_core(published)),
    }
    code = f'''
import base64
from hashlib import file_digest, sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib

inputs = Path('/kaggle/input')
{METRIC_BOOTSTRAP}

def mounted(name, namespace, dataset=None):
    candidates = {{p.resolve() for p in inputs.rglob(name)
                  if namespace in p.parts and (dataset is None or dataset in p.parts)}}
    if len(candidates) != 1:
        raise ValueError(f'Expected one {{name}} in {{namespace}}, got {{len(candidates)}}')
    return candidates.pop()

receipt = mounted('blind_virtual_np_receipt.json', {mount_slug!r})
proposals = receipt.parent / {candidate_artifact!r}
published = json.loads(receipt.read_text())
if {mount_slug == FULL_MAIN_WINDOW_SLUG!r} and published.get('generation_scope') != {FULL_MAIN_SCOPE!r}:
    raise ValueError('The full-main-window arm expects independently scanned candidates')
with proposals.open('rb') as stream:
    if file_digest(stream, 'sha256').hexdigest() != published['output_sha256'][proposals.name]:
        raise ValueError('Sealed generated candidate list changed')
if published['query_count'] != 60 or published['archived_pc_lists_sha256'] != {ARCHIVED_C34_SHA!r}:
    raise ValueError('Original and generated candidates used different C34 targets')

assets = {ASSETS.split('/')[1]!r}
model = mounted('fpnet_0.pt', 'casmi26-v3-models', assets)
bits = mounted('fp_bits.npy', 'casmi26-v2-pool', assets)
source = mounted('fpnet.py', 'casmi26-v3-models', assets)
if source != model.parent / 'code/casmi/fpnet.py' or bits.parent.parent != model.parent.parent:
    raise ValueError('FPNet A, its C34 runtime, and the fingerprint order must come from the same public asset')

with tempfile.TemporaryDirectory(prefix='virtual-fpnet-a-') as temp:
    scratch = Path(temp)
    for name, (data, expected) in {files!r}.items():
        raw = zlib.decompress(base64.b64decode(data))
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f'Blind scorer input changed: {{name}}')
        (scratch / name).write_bytes(raw)
    subprocess.run([sys.executable, str(scratch / 'score.py'), '--queries', str(scratch / 'test.parquet'),
                    '--c34', str(scratch / 'c34.json'), '--proposals', str(proposals),
                    '--model-a', str(model), '--bits', str(bits), '--code-root', str(source.parent.parent),
                    '--core', str(scratch / 'probe_core2.py'), '--outdir', '/kaggle/working',
                    '--workers', '4', '--cap', '24', *{branch_args!r}], check=True)
'''
    compile(code, score_slug, 'exec')
    notebook = dict(nbformat=4, nbformat_minor=5, cells=[dict(
        cell_type='code', id='score-generated-and-native', execution_count=None,
        metadata={}, outputs=[], source=code.strip().splitlines(keepends=True))],
        metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    metadata = dict(id=f'jiyujia/{score_slug}', title=score_slug, code_file='score.ipynb',
                    language='python', kernel_type='notebook', is_private=True, enable_gpu=False,
                    enable_internet=False, dataset_sources=[ASSETS, WHEEL], competition_sources=[],
                    kernel_sources=[dependency], docker_image=IMAGE)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'score.ipynb').write_text(json.dumps(notebook, separators=(',', ':')) + '\n')
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('panel', 'pc', 'published', 'out'):
        parser.add_argument('--' + arg, type=Path, required=True)
    generation = parser.add_mutually_exclusive_group(required=True)
    generation.add_argument('--generation-version', type=int)
    generation.add_argument('--generation-kernel')
    parser.add_argument('--candidate-source', choices=('proposals', 'known_main'), default='proposals')
    build(**vars(parser.parse_args()))

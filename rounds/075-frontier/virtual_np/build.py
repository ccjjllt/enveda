"""Package blind reverse-mass virtual natural-product enumeration for Kaggle CPU."""

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import zlib


HERE = Path(__file__).resolve().parent
SLUG = 'enveda-075-np-virtual-window'
SUPPLEMENT = f'{SLUG}-oxidation'
GLYCAN_SUPPLEMENT = f'{SLUG}-glycan-loss'
COMBINED = f'{SLUG}-combined'
ASSETS = 'dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets'
WHEEL = 'metric/rdkit-2026-3-3-wheel'
IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f'
ARCHIVED_C34_SHA = 'd7b28154b1bdf132528300bcf8faa1211984c843d5340ac81191472a452f564b'
ARCHIVED_TARGETS_SHA = '3cc377c3dd4d5023c3c6d8d13d26ece73af76dd6e19c5ba556284bfe7b0e80f4'
ASSET_HASHES = {
    'pool_meta.parquet': '82b8c40315611289b9a4004b3af18d6d3402b3cd5baa4ee5723da07b7e2f6bb0',
    'derive.py': '7c1e8c3afb4c17a3ae58fbc6c36fd4cfc552da97b9129c9b46cca3cdc6ee4892',
    'chem.py': '99474f23d8a67040b0c0f8c70e73592a6f7f656f50103e350ca473451e2169f1',
    'LICENSE-NOTICE.txt': 'b9c26d77338c03cb4387ae61bf4f6524eb77ee59c8530bc9f8d33a742a913a3b',
}


METRIC_BOOTSTRAP = '''
version = subprocess.run([sys.executable, '-c', 'from rdkit import rdBase; print(rdBase.rdkitVersion)'],
                         text=True, capture_output=True)
if version.returncode or version.stdout.strip() != '2026.03.3':
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename

    tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
    preference = {tag: rank for rank, tag in enumerate(sys_tags())}
    options = {}
    for path in inputs.rglob(f'rdkit-2026.3.3-{tag}-*.whl'):
        if 'rdkit-2026-3-3-wheel' not in path.parts:
            continue
        supported = [preference[x] for x in parse_wheel_filename(path.name)[3] if x in preference]
        if supported:
            options[path] = min(supported)
    if not options:
        raise FileNotFoundError(f'No metric RDKit wheel compatible with {tag}')
    wheel = min(options, key=lambda path: (options[path], str(path)))
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '-q', str(wheel)], check=True)
'''


def pack(data):
    return base64.b64encode(zlib.compress(data, 9)).decode(), sha256(data).hexdigest()


def build(targets, out, workers, only_rule=None, glycan_pc_lists=None):
    raw_targets = targets.read_bytes()
    target_sha = sha256(raw_targets).hexdigest()
    archived = json.loads(raw_targets)
    if (len(archived) != 60 or any(row['error'] or not isinstance(row['target'], (int, float))
                                  or not 0 < row['target'] < 10000 for row in archived.values())):
        raise ValueError('Expected the 60 actually used C34 query masses')
    pairs = sorted((mid, row['target']) for mid, row in archived.items())
    payload = pack((HERE / 'generate.py').read_bytes())
    scope = None
    supplement = None
    slug = SUPPLEMENT if only_rule else SLUG
    if glycan_pc_lists is not None:
        if only_rule:
            raise ValueError('Choose either the one-rule correction or the separate O-glycan channel.')
        raw_pc_lists = glycan_pc_lists.read_bytes()
        if (sha256(raw_pc_lists).hexdigest() != ARCHIVED_C34_SHA
                or target_sha != ARCHIVED_TARGETS_SHA
                or pairs != sorted((mid, row['target']) for mid, row in json.loads(raw_pc_lists).items())):
            raise ValueError('O-glycan loss requires the same sealed 60 native C34 mass centers.')
        slug = GLYCAN_SUPPLEMENT
        supplement = pack((HERE / 'glycan_loss.py').read_bytes())
        scope = 'terminal-o-glycan-loss'
    code = f'''
import base64
from hashlib import file_digest, sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib

import pandas as pd

inputs = Path('/kaggle/input')
work = Path('/kaggle/working')
hashes = {ASSET_HASHES!r}

def mounted(filename, parent):
    path, = (p for p in inputs.rglob(filename) if parent in p.parts)
    with path.open('rb') as file:
        if file_digest(file, 'sha256').hexdigest() != hashes[filename]:
            raise ValueError(f'Published index asset changed: {{path}}')
    return path

pool = mounted('pool_meta.parquet', 'casmi26-v2-pool')
source = mounted('derive.py', 'casmi26-v4b-models')
mounted('chem.py', 'casmi26-v4b-models')
mounted('LICENSE-NOTICE.txt', 'casmi26-v4b-models')
{METRIC_BOOTSTRAP}

with tempfile.TemporaryDirectory(prefix='virtual-np-') as temp:
    scratch = Path(temp)
    blob, expected = {payload!r}
    contents = zlib.decompress(base64.b64decode(blob))
    if sha256(contents).hexdigest() != expected:
        raise ValueError('Virtual candidate generator source changed in transit')
    program = scratch / 'generate.py'
    program.write_bytes(contents)
    supplement = {supplement!r}
    if supplement is not None:
        encoded, loss_expected = supplement
        source_bytes = zlib.decompress(base64.b64decode(encoded))
        if sha256(source_bytes).hexdigest() != loss_expected:
            raise ValueError('O-glycan loss rules changed in transit')
        program = scratch / 'glycan_loss.py'
        program.write_bytes(source_bytes)
    queries = scratch / 'targets.parquet'
    pd.DataFrame({pairs!r}, columns=['molecule_id', 'target_mass']).to_parquet(queries, index=False)
    command = [sys.executable, str(program), '--queries', str(queries), '--pool-meta', str(pool),
               '--derive-source', str(source), '--outdir', str(work), '--workers', {str(workers)!r}]
    if {only_rule!r}:
        command.extend(['--only-rule', {only_rule!r}])
    subprocess.run(command, check=True)
    files = ('proposals.parquet', 'known_main.parquet', 'stats.parquet')
    checked = {{}}
    for name in files:
        with (work / name).open('rb') as stream:
            checked[name] = file_digest(stream, 'sha256').hexdigest()
    result = dict(published_data={ASSETS!r}, hashes=hashes, output_sha256=checked,
                  query_count=60, archived_c34='jiyujia/enveda-075-np-recall-base/2',
                  archived_pc_lists_sha256={ARCHIVED_C34_SHA!r}, target_receipt_sha256={target_sha!r},
                  pin='Pablo v29 common C34 mass centres', generator_sha256=expected, only_rule={only_rule!r})
    if supplement is not None:
        result.update(generation_scope={scope!r}, loss_rule_source_sha256=loss_expected,
                      detailed_receipt_sha256=sha256((work / 'receipt.json').read_bytes()).hexdigest())
    (work / 'blind_virtual_np_receipt.json').write_text(json.dumps(result, indent=2) + '\\n')
'''
    compile(code, slug, 'exec')
    cell = dict(cell_type='code', id='reverse-mass-virtual-np', execution_count=None, metadata={},
                outputs=[], source=code.strip().splitlines(keepends=True))
    book = dict(nbformat=4, nbformat_minor=5, cells=[cell], metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    metadata = dict(id=f'jiyujia/{slug}', title=slug, code_file='generate.ipynb',
                    language='python', kernel_type='notebook', is_private=True, enable_gpu=False,
                    enable_internet=False, dataset_sources=[ASSETS, WHEEL], competition_sources=[],
                    kernel_sources=[], docker_image=IMAGE)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'generate.ipynb').write_text(json.dumps(book, separators=(',', ':')) + '\n')
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--only-rule', choices=('carboxyl_CH3',))
    parser.add_argument('--glycan-pc-lists', type=Path, help='Use the pinned original pc_lists.json for a separate O-glycan loss run.')
    build(**vars(parser.parse_args()))

"""Build a private, offline Kaggle MSAlign scoring notebook from blinded inputs."""

import argparse
import importlib.util
import json
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
SLUG = 'enveda-075-msalign-morgan'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_module('msalign_prepare', HERE / 'assets/prepare.py')
assemble = load_module('msalign_cells', HERE.parent / 'four_arm/assemble.py')
virtual_builder = run_path(str(HERE.parent / 'virtual_np/build.py'))
METRIC_BOOTSTRAP = virtual_builder['METRIC_BOOTSTRAP']


def build(args):
    channels = ['bins_fingerprint']
    if args.dreams_kernel:
        channels.append('dreams_fingerprint')
    query = args.panel / 'blind/test.parquet'
    manifest = json.loads((args.panel / 'panel_manifest.json').read_text())
    query_data = assemble.pack(query)
    if query_data[1] != manifest['output_sha256']['blind/test.parquet']:
        raise ValueError('The blind spectra differ from the frozen panel manifest')
    payload = {}
    if not args.prepared_assets:
        with TemporaryDirectory() as scratch_dir:
            scratch = Path(scratch_dir)
            source = scratch / 'app'
            prepare.write_source(scratch, source, args.source_git)
            payload = {str(path.relative_to(source)): assemble.pack(path) for path in sorted(source.rglob('*'))
                       if path.is_file()}
    payload.update({
        'prepare.py': assemble.pack(HERE / 'assets/prepare.py'),
        'score.py': assemble.pack(HERE / 'scoring/score.py'),
        'test.parquet': query_data,
    })

    run = f'''
from pathlib import Path
from hashlib import sha256
import base64
import json
import subprocess
import sys
import zlib

inputs = Path('/kaggle/input')
scratch = Path('/tmp/msalign-pilot')
payload = {payload!r}
for name, (compressed, expected) in payload.items():
    path = scratch / name
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = zlib.decompress(base64.b64decode(compressed))
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f'Embedded file changed: {{name}}')
    path.write_bytes(raw)

from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
spec = spec_from_loader('msalign_prepare', SourceFileLoader('msalign_prepare', str(scratch / 'prepare.py')))
prepare = module_from_spec(spec)
spec.loader.exec_module(prepare)

def mounted(filename):
    found = sorted(inputs.rglob(filename))
    if len(found) != 1:
        raise FileNotFoundError(f'Expected one mounted {{filename}}; found {{found}}')
    return found[0]

channels = {channels!r}
if {args.prepared_assets!r}:
    receipt_file = mounted('asset_receipt.json')
    checkpoints = receipt_file.parent
    receipt = json.loads(receipt_file.read_text())
    if (
        receipt['source_revision'] != prepare.MSALIGN_REVISION
        or receipt['checkpoint_md5'] != prepare.RELEASE_MD5
        or not set(channels) <= set(receipt['channels'])
    ):
        raise ValueError('Mounted MSAlign asset receipt differs from the requested official release')
    required = set(channels)
    needed_files = {{'official_manifest.yaml', *('source/' + name for name in prepare.SOURCE_FILES),
                    'source/models/MSAlign/__init__.py',
                    *(f'checkpoints/{{name}}.ckpt' for name in required),
                    *(f'configs/{{name}}.yaml' for name in required)}}
    for name in sorted(needed_files):
        if prepare.digest(checkpoints / name) != receipt['files'][name]:
            raise ValueError(f'Prepared asset changed: {{name}}')
    source_root = checkpoints / 'source'
else:
    checkpoints = scratch / 'checkpoint'
    prepare.extract_morgan_channels(mounted({(prepare.RELEASE + '.zip')!r}), checkpoints, channels)
    source_root = scratch
command = [
    sys.executable, str(scratch / 'score.py'),
    '--baseline', str(mounted('baseline.zip')),
    '--queries', str(scratch / 'test.parquet'),
    '--source', str(source_root),
    '--checkpoints', str(checkpoints),
    '--out', '/kaggle/working',
    '--device', {args.device!r},
    '--channel', *channels,
]
if 'dreams_fingerprint' in channels:
    command.extend(['--dreams', str(mounted({prepare.DREAMS_NAME!r}))])
{METRIC_BOOTSTRAP}
subprocess.run(command, check=True)
'''
    compile(run, 'score-msalign', 'exec')
    book = dict(nbformat=4, nbformat_minor=5, metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        cells=[assemble.code_cell('score-msalign', run)])
    default_assets = ('jiyujia/enveda-075-msalign-bins-assets/1' if args.prepared_assets
                      else 'jiyujia/enveda-075-msalign-public-assets/1')
    kernels = [args.asset_kernel or default_assets, args.baseline_kernel]
    if args.dreams_kernel and args.dreams_kernel not in kernels:
        kernels.append(args.dreams_kernel)
    meta = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='score.ipynb',
                language='python', kernel_type='notebook', is_private=True,
                enable_gpu=args.device.startswith('cuda'), enable_tpu=False, enable_internet=False,
                dataset_sources=[virtual_builder['WHEEL']], kernel_sources=kernels, competition_sources=[],
                docker_image=virtual_builder['IMAGE'])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'score.ipynb').write_text(json.dumps(book, separators=(',', ':')) + '\n')
    (args.out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'Created private scoring notebook in {args.out} with {len(payload)} blind/code inputs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-git', type=Path)
    parser.add_argument('--asset-kernel')
    parser.add_argument('--prepared-assets', action='store_true')
    parser.add_argument('--baseline-kernel', default='jiyujia/enveda-075-np-recall-reseal/2')
    parser.add_argument('--dreams-kernel')
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    build(parser.parse_args())

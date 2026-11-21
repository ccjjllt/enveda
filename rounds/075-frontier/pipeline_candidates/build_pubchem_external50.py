"""Build the offline MassBank 202603 PubChem depth-50 candidate preparation."""

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path

from build_kernel import BASELINE, code_cell, literal_value, replace, source
from build_pubchem import candidate_setup


SLUG = 'enveda-075b-massbank-202603-pubchem-depth50'
PANEL = 'jiyujia/enveda-075b-external-massbank-202603/1'
ANCHOR = 'jiyujia/enveda-075b-massbank-202603-cross-formula/1 pc_lists.json'
ANCHOR_SHA256 = '12d30464e556175c55c4f6d7170def47551e8831656714d97dba7434f2aacfa3'


def build(destination, anchor_path):
    anchor = anchor_path.read_bytes()
    if hashlib.sha256(anchor).hexdigest() != ANCHOR_SHA256:
        raise ValueError('Unexpected frozen raw PubChem anchor archive')
    packed = base64.b64encode(gzip.compress(anchor, mtime=0)).decode('ascii')

    path, = BASELINE.glob('*.ipynb')
    original = json.loads(path.read_text())
    frozen = json.loads(base64.b64decode(literal_value(source(original, 12), 'FROZEN_B64')))
    setup = candidate_setup(frozen)
    native = source(frozen, 34)
    core = replace(literal_value(native, 'CORE'), 'K = 25', 'K = 50')
    prior = literal_value(native, 'V17_PATCH')
    runner = literal_value(native, 'RUNNER')
    run = f'''
import gzip
import base64
import tempfile

anchors = gzip.decompress(base64.b64decode({packed!r}))
if hashlib.sha256(anchors).hexdigest() != {ANCHOR_SHA256!r}:
    raise ValueError('Frozen PubChem anchor checksum mismatch')
original_pc = json.loads(anchors)

with tempfile.TemporaryDirectory(prefix='pubchem-depth50-') as temp:
    staging = Path(temp)
    (staging / 'probe_core2.py').write_text({(core + prior)!r})
    (staging / 'pc_runner.py').write_text({runner!r})
    raw = Path('/kaggle/working/pc50_raw.json')
    subprocess.run([sys.executable, '-u', str(staging / 'pc_runner.py'), V3CODE, V3MOD,
                    POOL_DIR, PC_DIR, TEST, str(raw), '4'],
                   check=True, env={{**os.environ, 'PYTHONHASHSEED': '0'}})

depth50 = json.loads(raw.read_bytes())
panel_source = {PANEL!r}
anchor_source = {ANCHOR!r}
anchor_sha256 = {ANCHOR_SHA256!r}
'''
    audit = Path(__file__).with_name('pubchem_external50.py').read_text()

    metadata = json.loads((BASELINE / 'kernel-metadata.json').read_text())
    metadata.update(id=f'jiyujia/{SLUG}', title=SLUG, code_file=f'{SLUG}.ipynb',
                    enable_gpu=False, keywords=['chemistry'],
                    dataset_sources=[asset for asset in metadata['dataset_sources'] if asset.endswith((
                        'fold-safe-fpnet', 'pubchem-popularity-prior', 'v4g-reproduction-assets',
                        'rdkit-2026-3-3-wheel'))], kernel_sources=[PANEL], competition_sources=[])
    metadata.pop('machine_shape', None)
    original['cells'] = [code_cell(text) for text in (setup, run, audit)]
    for i, cell in enumerate(original['cells']):
        compile(cell['source'], f'pubchem_external50_cell_{i}', 'exec')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / metadata['code_file']).write_text(json.dumps(original, ensure_ascii=False) + '\n')
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('anchor', type=Path)
    args = parser.parse_args()
    build(args.destination, args.anchor)

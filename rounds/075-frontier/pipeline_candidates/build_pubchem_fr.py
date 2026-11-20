"""Package the published FR scoring functions for the fixed depth-50 comparison."""

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path

from build_kernel import BASELINE, literal_value, source
from build_pubchem_forward import DEPTH, PANEL, PIPELINE, definition
from build_taeyang_pubchem50 import ORIGINAL_FROZEN_SHA, ORIGINAL_ID


HERE = Path(__file__).resolve().parent
SLUG = 'enveda-075b-massbank-pubchem-fr'


def build(original, destination):
    metadata = json.loads((original / 'kernel-metadata.json').read_text())
    notebook = json.loads((original / metadata['code_file']).read_text())
    frozen_bytes = base64.b64decode(literal_value(source(notebook, 13), 'FROZEN_B64'), validate=True)
    if metadata['id'] != ORIGINAL_ID or sha256(frozen_bytes).hexdigest() != ORIGINAL_FROZEN_SHA:
        raise ValueError('Use the verified Taeyang notebook v2')
    fr_cell, = [''.join(cell['source']) for cell in notebook['cells']
                if cell['cell_type'] == 'code' and 'def _f3_score_list(' in ''.join(cell['source'])]
    fr_scoring, _ = fr_cell.split('\ndef _f3_ens(', 1)

    baseline_path, = BASELINE.glob('*.ipynb')
    baseline = json.loads(baseline_path.read_text())
    frozen = json.loads(base64.b64decode(literal_value(source(baseline, 12), 'FROZEN_B64')))
    forward = (HERE / 'pubchem_forward.py').read_text()
    candidates, _ = forward.split('\ndef rank_group(', 1)
    comparison = (HERE / 'pubchem_fr.py').read_text()
    sections = (candidates, definition(forward, 'recorded_ranks'), definition(forward, 'summaries'),
                source(frozen, 2), fr_scoring, comparison)
    script = '\n\n'.join(sections)
    compile(script, 'pubchem_fr.py', 'exec')
    config = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='pubchem_fr.py', language='python',
                  kernel_type='script', is_private=True, enable_gpu=False, enable_internet=False,
                  dataset_sources=['taeyangg4/casmi26-fr3to25-clean', 'metric/rdkit-2026-3-3-wheel'],
                  kernel_sources=[PANEL, PIPELINE, DEPTH, 'jiyujia/enveda-075b-massbank-pubchem-forward/1',
                                  'jiyujia/enveda-075b-external-membership/2'],
                  competition_sources=[], model_sources=[])
    destination.mkdir(parents=True, exist_ok=True)
    (destination / config['code_file']).write_text(script)
    (destination / 'kernel-metadata.json').write_text(json.dumps(config, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    build(args.original, args.destination)

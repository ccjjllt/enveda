"""Package the existing published FR scorer for the fixed MassBank negative panel."""

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path

from build_kernel import BASELINE, literal_value, source
from build_pubchem_forward import definition
from build_taeyang_pubchem50 import ORIGINAL_FROZEN_SHA, ORIGINAL_ID


HERE = Path(__file__).resolve().parent
SLUG = 'enveda-075b-massbank-negative-fr'


def build(original, destination):
    metadata = json.loads((original / 'kernel-metadata.json').read_text())
    notebook = json.loads((original / metadata['code_file']).read_text())
    frozen = base64.b64decode(literal_value(source(notebook, 13), 'FROZEN_B64'), validate=True)
    if metadata['id'] != ORIGINAL_ID or sha256(frozen).hexdigest() != ORIGINAL_FROZEN_SHA:
        raise ValueError('Use the verified Taeyang notebook v2')
    fr_cell, = [''.join(c['source']) for c in notebook['cells']
                if c['cell_type'] == 'code' and 'def _f3_score_list(' in ''.join(c['source'])]
    scorer, _ = fr_cell.split('\ndef _f3_ens(', 1)

    path, = BASELINE.glob('*.ipynb')
    baseline = json.loads(path.read_text())
    baseline_frozen = json.loads(base64.b64decode(literal_value(source(baseline, 12), 'FROZEN_B64')))
    helpers = (HERE / 'pubchem_forward.py').read_text()
    script = '\n\n'.join((source(baseline_frozen, 2), scorer, definition(fr_cell, '_f3_blend'),
                          definition(helpers, 'artifact'), definition(helpers, 'summaries'),
                          (HERE / 'negative_fr.py').read_text()))
    compile(script, 'negative_fr.py', 'exec')
    config = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='negative_fr.py', language='python',
                  kernel_type='script', is_private=True, enable_gpu=False, enable_internet=False,
                  dataset_sources=['taeyangg4/casmi26-fr3to25-clean', 'metric/rdkit-2026-3-3-wheel'],
                  kernel_sources=['jiyujia/enveda-075b-external-massbank-adducts/1',
                                  'jiyujia/enveda-075b-massbank-adducts-pipeline/1'],
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

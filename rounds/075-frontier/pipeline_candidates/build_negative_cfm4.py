"""Package the fixed negative-ion comparison for the offline Kaggle CPU."""

import argparse
import json
from pathlib import Path

from build_pubchem_forward import definition


HERE = Path(__file__).resolve().parent
SLUG = 'enveda-075b-massbank-negative-cfm4'
RUNTIME = 'jiyujia/enveda-075b-cfm4-runtime/1'


def build(destination, asset_kernel):
    helpers = (HERE / 'pubchem_forward.py').read_text()
    peaks = (HERE.parent / 'ce_fragments' / 'score.py').read_text()
    script = '\n\n'.join((definition(helpers, 'artifact'), definition(helpers, 'summaries'),
                          *(definition(peaks, name) for name in ('clean_peaks', 'match_peaks', 'cosine')),
                          (HERE / 'negative_cfm4.py').read_text()))
    compile(script, 'negative_cfm4.py', 'exec')
    meta = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='negative_cfm4.py', language='python',
                kernel_type='script', is_private=True, enable_gpu=False, enable_internet=False, dataset_sources=[],
                kernel_sources=['jiyujia/enveda-075b-external-massbank-adducts/1',
                                'jiyujia/enveda-075b-massbank-adducts-pipeline/1', asset_kernel],
                competition_sources=[], model_sources=[])
    destination.mkdir(parents=True, exist_ok=True)
    (destination / meta['code_file']).write_text(script)
    (destination / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--asset-kernel', default=RUNTIME)
    args = parser.parse_args()
    build(args.destination, args.asset_kernel)

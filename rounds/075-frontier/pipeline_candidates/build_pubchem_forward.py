"""Package the existing simulator wrappers with the MassBank PubChem depth comparison."""

import argparse
import ast
import base64
import json
from pathlib import Path

from build_kernel import BASELINE, literal_value, source


HERE = Path(__file__).resolve().parent
SLUG = 'enveda-075b-massbank-pubchem-forward'
PANEL = 'jiyujia/enveda-075b-external-massbank-202611/1'
PIPELINE = 'jiyujia/enveda-075b-massbank-pipeline/1'
DEPTH = 'jiyujia/enveda-075b-massbank-pubchem-depth/2'


def definition(code, name):
    fn, = [node for node in ast.parse(code).body if isinstance(node, ast.FunctionDef) and node.name == name]
    return ast.get_source_segment(code, fn)


def build(destination, depth_kernel):
    path, = BASELINE.glob('*.ipynb')
    notebook = json.loads(path.read_text())
    frozen = json.loads(base64.b64decode(literal_value(source(notebook, 12), 'FROZEN_B64')))
    rdkit_setup = source(frozen, 2)
    wheel_selector = definition(source(frozen, 26), 'simulator_rdkit_wheel')
    simulator_setup = definition(source(frozen, 41), '_claw_preseed_site')
    script = (HERE / 'pubchem_forward.py').read_text()
    prefix, run = script.rsplit("\nif __name__ == '__main__':", 1)
    depth_slug = depth_kernel.split('/')[1]
    prefix = prefix.replace("DEPTH = 'enveda-075b-massbank-pubchem-depth'", f'DEPTH = {depth_slug!r}')
    lines = '\n\n'.join((prefix, wheel_selector, simulator_setup, rdkit_setup, "if __name__ == '__main__':" + run))
    compile(lines, 'pubchem_forward.py', 'exec')

    metadata = json.loads((BASELINE / 'kernel-metadata.json').read_text())
    metadata.update(id=f'jiyujia/{SLUG}', title=SLUG, code_file='pubchem_forward.py', kernel_type='script',
                    dataset_sources=[asset for asset in metadata['dataset_sources']
                                     if asset.endswith(('v4g-reproduction-assets', 'rdkit-2026-3-3-wheel'))],
                    kernel_sources=[PANEL, PIPELINE, depth_kernel], competition_sources=[])
    destination.mkdir(parents=True, exist_ok=True)
    (destination / metadata['code_file']).write_text(lines)
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--depth-kernel', default=DEPTH)
    args = parser.parse_args()
    build(args.destination, args.depth_kernel)

"""Package the structure coverage audit and the matched hard-negative pilot."""
import argparse
import ast
import base64
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OWNER = 'jiyujia'
PANEL = f'{OWNER}/enveda-075b-external-massbank-adducts/1'
ARCHIVE = f'{OWNER}/enveda-075b-massbank-adducts-pipeline/1'
PAIRED = f'{OWNER}/enveda-075-cross-polarity-pilot/1'
WINDOW = f'{OWNER}/enveda-075-negative-full-window/1'
WHEEL = 'metric/rdkit-2026-3-3-wheel'
POPULARITY = 'dmitriigluzdov/casmi26-pubchem-popularity-prior'
V4G = 'dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets'
FOLD = 'dmitriigluzdov/casmi26-fold-safe-fpnet'
CONTEST = 'enveda-CASMI26-molecule-id-mass-spectra'
FROZEN_NOTEBOOK_SHA = 'e2334fad6d57fcf4c6d6b05e8ef986b41954f4bc14d67f433d5ea6d8a945a346'


def literal(code, name):
    node, = [node for node in ast.parse(code).body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    return ast.literal_eval(node.value)


def c34(path):
    blob = path.read_bytes()
    if sha256(blob).hexdigest() != FROZEN_NOTEBOOK_SHA:
        raise ValueError('Expected the archived 068a MassBank notebook.')
    notebook = json.loads(blob)
    packed = literal(''.join(notebook['cells'][5]['source']), 'FROZEN_B64')
    frozen = json.loads(base64.b64decode(packed))
    return ''.join(frozen['cells'][34]['source'])


def sections(path, keep, imports=False):
    text = path.read_text()
    result = []
    for node in ast.parse(text).body:
        assignment = node.targets[0] if isinstance(node, ast.Assign) else None
        name = assignment.id if isinstance(assignment, ast.Name) else getattr(node, 'name', None)
        if name in keep or (imports and isinstance(node, (ast.Import, ast.ImportFrom))):
            result.append(ast.get_source_segment(text, node))
    return '\n\n'.join(result)


def build(arm, destination, original=None, mining_version=1):
    prefix = sections(HERE.parent / 'train.py', {'ROOT', 'OUTPUT', 'PROTON', 'BINS', 'find', 'chemistry',
                      'hash_order', 'spectrum_vector', 'Retriever'}, imports=True)
    prefix = prefix.replace("wheel, = ROOT.rglob(f'rdkit-2026.3.3-{tag}-*.whl')",
                            "wheel, = (p for p in ROOT.rglob(f'rdkit-2026.3.3-{tag}-*.whl')\n"
                            "                  if 'casmi26-attributed-v4g-reproduction-assets' not in p.parts)")
    assets = [WHEEL, POPULARITY, V4G]
    inputs = [PANEL, ARCHIVE]
    competitions = []
    if arm == 'audit':
        inputs.append(WINDOW)
        prefix += '\n\n' + sections(HERE.parent / 'full_window/retrieve.py', {'POPULARITY', 'index_files'})
    elif arm == 'mine':
        code = c34(original)
        assets.append(FOLD)
        competitions.append(CONTEST)
        prefix += f"\n\nC34_CORE = {literal(code, 'CORE') + literal(code, 'V17_PATCH')!r}"
        prefix += f"\n\nC34_RUNNER = {literal(code, 'RUNNER')!r}"
    elif arm == 'fit_retrieve':
        assets.remove(V4G)
        inputs.extend([PAIRED, f'{OWNER}/enveda-075-negative-hard-mine/{mining_version}',
                       f'{OWNER}/enveda-075-negative-window-control-cpu/2'])
        retrieve = HERE.parent / 'full_window/retrieve.py'
        definitions = {'POPULARITY', 'index_files', 'PPM', 'BLOCK', 'PROPOSALS', 'init_window', 'enumerate_window'}
        prefix += '\n\n' + sections(retrieve, definitions)
    if arm in ('mine', 'fit_retrieve'):
        prefix += '\n\n' + (HERE / 'groups.source').read_text()
    script = prefix + '\n\n' + (HERE / f'{arm}.py').read_text()
    compile(script, f'{arm}.py', 'exec')
    slug = {'audit':'enveda-075-negative-index-audit', 'mine':'enveda-075-negative-hard-mine',
            'fit_retrieve':'enveda-075-negative-hard-retrieve'}[arm]
    metadata = dict(id=f'{OWNER}/{slug}', title=slug, code_file=f'{arm}.py', language='python',
                    kernel_type='script', is_private=True, enable_gpu=False, enable_internet=False,
                    dataset_sources=assets, kernel_sources=inputs, competition_sources=competitions)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f'{arm}.py').write_text(script)
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm', choices=('audit', 'mine', 'fit_retrieve'))
    parser.add_argument('destination', type=Path)
    parser.add_argument('--original', type=Path)
    parser.add_argument('--mining-version', type=int, default=1)
    args = parser.parse_args()
    build(args.arm, args.destination, args.original, args.mining_version)

"""Package the blind MassBank 110 FR and C34 rank-5 request planner on Kaggle CPU."""

import argparse
from ast import Assign, FunctionDef, Name, literal_eval, parse
from hashlib import sha256
import json
from pathlib import Path

from build import ASSETS, HERE, WHEEL, pack


SLUG = 'enveda-075-mb110-fr-rank5-plan'
PANEL = 'jiyujia/enveda-075b-external-massbank-202611/1'
IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'
TEMPLATE = Path('/private/tmp/enveda-075b-pc-coarse-fr-110/enveda-075b-pc-coarse-fr-110.ipynb')
CELL_SHA = ('8ec39aef8aedf44a5687a4f83bdae7962f08bbd5029d81f2aff26891e56babb4',
            '5c7ece5ad6487747d5253d3044632f3c4790c55d04087eaddc0be5d3bca1c2b3')


def template_source(path):
    """Reuse the production C34 and FR source portions of the published wide-pool experiment."""
    template = json.loads(path.read_bytes())
    cells = [''.join(cell['source']) for cell in template['cells']]
    if tuple(sha256(cell.encode()).hexdigest() for cell in cells) != CELL_SHA:
        raise ValueError('The historical FR-widepool build template source does not match the frozen version')
    bootstrap, scoring = cells
    worker, runner, remainder = scoring.split('\n', 2)
    worker = literal_eval(worker.removeprefix('COARSE_WORKER = '))
    runner = literal_eval(runner.removeprefix('COARSE_RUNNER = '))
    old_picker = "formulas = {formula: [] for formula, count in config['quota'].items() if count >= 2}"
    if worker.count(old_picker) != 1:
        raise ValueError('Could not identify the archived per-formula selection block')
    worker = worker.replace(old_picker, "formulas = {config['formula']: []}")
    fr_scoring, _, _ = remainder.partition('\ndef _f3_z(')
    if not fr_scoring or 'def _f3_score_list(' not in fr_scoring:
        raise ValueError('Could not identify the exact FR model scoring code')
    cached_path = '/kaggle/working/numba_cache'
    if bootstrap.count(cached_path) != 2:
        raise ValueError('Could not point the archived Numba cache outside the notebook outputs')
    bootstrap = bootstrap.replace(cached_path, '/tmp/mb110-fr-rank5-numba-cache')
    for name, code in (('bootstrap', bootstrap), ('worker', worker), ('runner', runner), ('FR scorer', fr_scoring)):
        compile(code, name, 'exec')
    return bootstrap, worker, runner, fr_scoring


def function_source(filename, name):
    source = (HERE / filename).read_text()
    node, = (item for item in parse(source).body if isinstance(item, FunctionDef) and item.name == name)
    start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
    return '\n'.join(source.splitlines()[start - 1:node.end_lineno]) + '\n'


def build(template, out):
    bootstrap, worker, runner, fr_scoring = template_source(template)
    rule = ('from functools import lru_cache\nfrom rdkit import Chem\n'
            'from rdkit.Chem.rdMolDescriptors import CalcMolFormula\n\n'
            + function_source('final_native_top5.py', 'formula') + '\n\n'
            + function_source('build_final_native_outsider_np60.py', 'replace_at_rank'))
    spectra_source = function_source('build_final_native_outsider_mb110_depth50.py', 'spectral_array_bytes')
    modules = {'outsider_rule.py': rule, 'depth50_archive.py': 'import json\n\n' + spectra_source}
    modules.update({name: (HERE / name).read_text() for name in (
        'c39_request_cache.py', 'final_native_outsider_mb110.py', 'mb110_fr_rank5_plan.py')})
    assets = {}
    for name, source in modules.items():
        compile(source, name, 'exec')
        assets[name] = pack(source.encode())
    source_hashes = dict(template_c34_setup=CELL_SHA[0], template_coarse_fr=CELL_SHA[1],
                         worker_sha256=sha256(worker.encode()).hexdigest(),
                         runner_sha256=sha256(runner.encode()).hexdigest(),
                         fr_scoring_sha256=sha256(fr_scoring.encode()).hexdigest(),
                         planner_sha256=assets['mb110_fr_rank5_plan.py'][1])
    code = f'''{bootstrap}
COARSE_WORKER = {worker!r}
COARSE_RUNNER = {runner!r}
{fr_scoring}
import base64
from hashlib import sha256 as package_sha256
from tempfile import TemporaryDirectory
import zlib

with TemporaryDirectory(prefix='mb110-fr-rank5-code-') as directory:
    for name, (compressed, expected) in {assets!r}.items():
        raw = zlib.decompress(base64.b64decode(compressed))
        if package_sha256(raw).hexdigest() != expected:
            raise ValueError(f'Changed bundled source: {{name}}')
        (Path(directory) / name).write_bytes(raw)
    sys.path.insert(0, directory)
    from mb110_fr_rank5_plan import run
    run(Path('/kaggle/input'), Path('/kaggle/working'), globals(), {source_hashes!r})
'''
    compile(code, SLUG, 'exec')
    refs = {target.id: literal_eval(node.value)
            for node in parse(modules['final_native_outsider_mb110.py']).body if isinstance(node, Assign)
            for target in node.targets if isinstance(target, Name) and target.id in {'BASE', 'DEPTH'}}
    metadata = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='plan.py', language='python', kernel_type='script',
                    is_private=True, enable_gpu=False, enable_internet=False, docker_image=IMAGE,
                    dataset_sources=['dmitriigluzdov/casmi26-fold-safe-fpnet',
                                     'dmitriigluzdov/casmi26-pubchem-popularity-prior', ASSETS, WHEEL,
                                     'taeyangg4/casmi26-fr3to25-clean/1'],
                    kernel_sources=[PANEL, refs['BASE'], refs['DEPTH']], competition_sources=[])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'plan.py').write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return sha256(code.encode()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, default=TEMPLATE)
    parser.add_argument('--out', type=Path, required=True)
    print(build(**vars(parser.parse_args())))

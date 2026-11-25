"""Package the public WineDB source audit for a private Kaggle CPU run."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from runpy import run_path


HERE = Path(__file__).resolve().parent
SLUG = 'enveda-075-winedb-ms2-overlap'
COMPETITION = 'enveda-CASMI26-molecule-id-mass-spectra'
RD_WHEEL = 'metric/rdkit-2026-3-3-wheel'
METRIC_BOOTSTRAP = run_path(str(HERE.parent / 'virtual_np/build.py'))['METRIC_BOOTSTRAP']


def build(out, owner):
    audit = (HERE / 'audit.py').read_text()
    run = f'''
import subprocess
import sys

inputs = Path('/kaggle/input')
{METRIC_BOOTSTRAP}
competition = {COMPETITION.lower()!r}
matches = {{p.resolve() for p in inputs.rglob('train.parquet')
            if any(part.lower() == competition for part in p.parts)}}
if len(matches) != 1:
    raise FileNotFoundError('Expected one mounted official Kaggle competition training file')
run(matches.pop(), Path('/kaggle/working'))
'''
    source = audit + '\n' + run
    compile(source, 'winedb-availability', 'exec')
    book = dict(cells=[dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                            source=source.splitlines(keepends=True))],
                metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                nbformat=4, nbformat_minor=4)
    out.mkdir(parents=True, exist_ok=True)
    program = out / 'wine_db.ipynb'
    program.write_text(json.dumps(book, indent=1) + '\n')
    options = dict(id=f'{owner}/{SLUG}', title=SLUG, code_file=program.name,
                   language='python', kernel_type='notebook', is_private=True,
                   enable_internet=True, enable_gpu=False, enable_tpu=False,
                   competition_sources=[COMPETITION], dataset_sources=[RD_WHEEL], kernel_sources=[])
    (out / 'kernel-metadata.json').write_text(json.dumps(options, indent=2) + '\n')
    return {'path': str(out), 'notebook_sha256': sha256(program.read_bytes()).hexdigest(),
            'audit_source_sha256': sha256(audit.encode()).hexdigest(), 'blind_labels_embedded': False}


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--owner', default='jiyujia')
    args = parser.parse_args()
    print(json.dumps(build(args.out, args.owner), indent=2))

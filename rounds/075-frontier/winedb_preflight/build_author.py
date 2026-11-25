"""Build the independent author-release WineDB Kaggle CPU preflight."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

from build import COMPETITION, METRIC_BOOTSTRAP, RD_WHEEL


HERE = Path(__file__).resolve().parent
SLUG = 'enveda-075-winedb-author-overlap'


def build_author(out, owner):
    shared = (HERE / 'audit.py').read_text()
    author = (HERE / 'author_archive.py').read_text()
    run = f'''
import subprocess
import sys

inputs = Path('/kaggle/input')
{METRIC_BOOTSTRAP}
competition = {COMPETITION.lower()!r}
matches = {{path.resolve() for path in inputs.rglob('train.parquet')
            if any(part.lower() == competition for part in path.parts)}}
if len(matches) != 1:
    raise FileNotFoundError('Expected one mounted official Kaggle competition training file')
run_author(matches.pop(), Path('/kaggle/working'))
'''
    source = shared + '\n' + author + '\n' + run
    compile(source, 'winedb-author-archive', 'exec')
    book = dict(cells=[dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                            source=source.splitlines(keepends=True))],
                metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                nbformat=4, nbformat_minor=4)
    out.mkdir(parents=True, exist_ok=True)
    program = out / 'wine_db_author.ipynb'
    program.write_text(json.dumps(book, indent=1) + '\n')
    options = dict(id=f'{owner}/{SLUG}', title=SLUG, code_file=program.name, language='python', kernel_type='notebook',
                   is_private=True, enable_internet=True, enable_gpu=False, enable_tpu=False,
                   competition_sources=[COMPETITION], dataset_sources=[RD_WHEEL], kernel_sources=[])
    (out / 'kernel-metadata.json').write_text(json.dumps(options, indent=2) + '\n')
    return dict(path=str(out), notebook_sha256=sha256(program.read_bytes()).hexdigest(),
                author_source_sha256=sha256(author.encode()).hexdigest(), shared_source_sha256=sha256(shared.encode()).hexdigest(),
                blind_labels_embedded=False)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--owner', default='jiyujia')
    args = parser.parse_args()
    print(json.dumps(build_author(args.out, args.owner), indent=2))

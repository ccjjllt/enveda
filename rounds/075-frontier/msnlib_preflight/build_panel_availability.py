"""Build the private Kaggle CPU availability notebook from local exclusion inputs."""

from argparse import ArgumentParser
import base64
from hashlib import file_digest
import json
from pathlib import Path
import re
from runpy import run_path
import zlib

from panel_availability import INDEX_SHA, digest_text, groups


HERE = Path(__file__).resolve().parent
SLUG = 'enveda-075-targetmol-ms2-panel-audit'
STRUCTURES = 'dmitriigluzdov/casmi26-natural-product-panel-benchmark'
RD_WHEEL = 'metric/rdkit-2026-3-3-wheel'
KEY = re.compile(r'[A-Z]{14}')
E180_SHA = '0d493e7127bbc0424c8de121f18ce193b5fe7884d1d6484a5888c2a571409e9a'
METRIC_BOOTSTRAP = run_path(str(HERE.parent / 'virtual_np/build.py'))['METRIC_BOOTSTRAP']


def exclusions(prior, np250, index, e180=None):
    import pyarrow.parquet as pq
    from rdkit import Chem, RDLogger, rdBase
    from rdkit.Chem.Scaffolds import MurckoScaffold

    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError('Build exclusions with competition RDKit 2026.03.3')
    RDLogger.DisableLog('rdApp.*')
    with index.open('rb') as handle:
        if file_digest(handle, 'sha256').hexdigest() != INDEX_SHA:
            raise ValueError('Unexpected local competition structure index')
    union = {field: set() for field in ('identities', 'fine', 'generic')}
    audits = {}

    def add(source, keys, structures, reported_generic=(), reported_fine=()):
        keys = set(keys)
        fine, generic = set(reported_fine), set(reported_generic)
        for smiles in set(structures) - {None, ''}:
            result = groups(smiles)
            if not result:
                continue
            key, fg, gg = result
            keys.add(key)
            keys.add(Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))[:14])
            fine.add(fg)
            generic.add(gg)
        items = {'identities': {key for key in keys if isinstance(key, str) and KEY.fullmatch(key)},
                 'fine': fine - {''}, 'generic': generic - {''}}
        if not items['identities']:
            raise ValueError(f'Missing prior identities for {source}')
        for field, values in items.items():
            union[field].update(digest_text(value) for value in values)
        audits[source] = {field: len(values) for field, values in items.items()}

    index_rows = pq.read_table(index, columns=['inchikey14', 'scorer_key', 'smiles']).to_pylist()
    np_keys = set(pq.read_table(np250, columns=['molecule'])['molecule'].to_pylist())
    np_rows = [row for row in index_rows if np_keys.intersection((row['inchikey14'], row['scorer_key']))]
    np_resolved = {row[field] for row in np_rows for field in ('inchikey14', 'scorer_key')} & np_keys
    add('NP250', np_keys, [row['smiles'] for row in np_rows])
    audits['NP250']['structures_resolved_from_pin'] = len(np_resolved)
    if e180 is not None:
        with e180.open('rb') as handle:
            if file_digest(handle, 'sha256').hexdigest() != E180_SHA:
                raise ValueError('Unexpected E180 fresh-panel archive digest')
        panel = pq.read_table(e180, columns=['query', 'raw_key', 'murcko'])
        keys = [key for field in ('query', 'raw_key') for key in panel[field].to_pylist()]
        if panel.num_rows != 121 or any(not isinstance(key, str) or not KEY.fullmatch(key) for key in keys):
            raise ValueError('Expected 121 E180 fresh-panel identities and original keys')
        fine, generic = set(), set()
        for scaffold in panel['murcko'].unique().to_pylist():
            mol = Chem.MolFromSmiles(scaffold) if scaffold else None
            if mol is None or not mol.GetNumAtoms():
                raise ValueError('Invalid cyclic E180 fresh-panel scaffold')
            fine.add(Chem.MolToSmiles(mol, isomericSmiles=False))
            generic.add(Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(mol)))
        add('E180Fresh121', keys, [], reported_generic=generic, reported_fine=fine)
    for source, path_str in prior:
        if source in audits:
            raise ValueError(f'Repeated prior panel alias {source}')
        path = Path(path_str)
        if path.suffix == '.json':
            data = json.loads(path.read_text())
            add(source, data['blocked_keys'], [], data['blocked_scaffolds'])
            continue
        fields = set(pq.ParquetFile(path).schema_arrow.names)
        key_names = fields & {'truth_key', 'candidate_id', 'molecule', 'inchikey14', 'scorer_key'}
        smiles_names = fields & {'truth_smiles', 'normalized_smiles', 'smiles'}
        if path.suffix != '.parquet' or not (key_names | smiles_names):
            raise ValueError(f'Missing structure metadata for {source}')
        frame = pq.read_table(path, columns=sorted(key_names | smiles_names))
        keys = [value for field in key_names for value in frame[field].to_pylist()]
        structures = [value for field in smiles_names for value in frame[field].to_pylist()]
        add(source, keys, structures)
    return {**{field: sorted(values) for field, values in union.items()}, 'sources': audits}


def embedded(value):
    return base64.b64encode(zlib.compress(value, 9)).decode()


def build(args):
    mask = exclusions(args.prior, args.np250, args.index, args.e180)
    blob = json.dumps(mask, sort_keys=True).encode()
    files = {name: embedded((HERE / name).read_bytes()) for name in ('audit.py', 'panel_availability.py')}
    source = f'''import base64
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import zlib

inputs = Path('/kaggle/input')
scratch = Path('/tmp/enveda-targetmol-availability')
scratch.mkdir(exist_ok=True)
for name, blob in {files!r}.items():
    (scratch / name).write_bytes(zlib.decompress(base64.b64decode(blob)))

def only(paths, description):
    paths = {{p.resolve() for p in paths}}
    if len(paths) != 1:
        raise ValueError(f'Expected exactly one {{description}}, found {{len(paths)}}')
    return paths.pop()

{METRIC_BOOTSTRAP}

index = only((p for p in inputs.rglob('structures.parquet')
              if 'casmi26-identity-disjoint-validation-folds' in p.parts
              and 'casmi26-natural-product-panel-benchmark' in p.parts), 'competition structure index')
mask = zlib.decompress(base64.b64decode({embedded(blob)!r}))
if sha256(mask).hexdigest() != {digest_text(blob.decode())!r}:
    raise ValueError('Exclusion payload digest differs')
sys.path.insert(0, str(scratch))
from panel_availability import run

run(index, json.loads(mask), Path('/kaggle/working/targetmol_ms2_availability.json'))
'''
    compile(source, 'availability.py', 'exec')
    book = {
        'cells': [{'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                   'outputs': [], 'source': source.splitlines(keepends=True)}],
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'nbformat': 4, 'nbformat_minor': 4,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'availability.ipynb').write_text(json.dumps(book, indent=1) + '\n')
    metadata = dict(id=f'{args.owner}/{SLUG}', title=SLUG, code_file='availability.ipynb',
                    language='python', kernel_type='notebook', is_private=True,
                    enable_internet=True, enable_gpu=False, enable_tpu=False,
                    dataset_sources=[STRUCTURES, RD_WHEEL], kernel_sources=[], competition_sources=[])
    (args.out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'package': str(args.out), 'mask_counts': {k: len(mask[k]) for k in ('identities', 'fine', 'generic')},
                      'prior_panel_counts': mask['sources'], 'mask_sha256': digest_text(blob.decode())}, indent=2))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--owner', default='jiyujia')
    parser.add_argument('--index', required=True, type=Path)
    parser.add_argument('--np250', required=True, type=Path)
    parser.add_argument('--e180', type=Path, help='Pinned local E180 fresh 121-query preselection archive')
    parser.add_argument('--prior', nargs=2, action='append', required=True, metavar=('ALIAS', 'FILE'))
    build(parser.parse_args())

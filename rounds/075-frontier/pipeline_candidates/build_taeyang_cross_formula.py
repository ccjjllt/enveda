"""Package Taeyang v2 with the fixed cross-formula PubChem promotion."""

import argparse
import base64
from hashlib import sha256
from inspect import getsource
import json
from pathlib import Path

from build_kernel import code_cell, literal_value, source


ORIGINAL_ID = 'taeyangg4/casmi26-lb-0-495-network-fr-p5-tail'
ORIGINAL_FROZEN_SHA = 'd1e757c2b71b46e6e8a5777f14302a4737d0fdbcd56d8365093c039b93c5fd88'
SLUG = 'enveda-075b-taeyang-cross-formula'
APEX_CELL = 27


def promote_cross_formula(root, pipeline_done, output_path=None):
    import json
    from functools import lru_cache
    from math import isfinite

    if not pipeline_done:
        return {'status': 'no frozen pipeline output'}

    def read_json(path):
        return json.loads(path.read_text())

    required = ['submission.csv', 'e004_candidate_inputs.json', 'frozen_candidate_evidence.parquet']
    for model in ('ice', 'gl'):
        required.extend(f'{model}_work/{model}_{name}'
                        for name in ('in.json', 'out.json', 'out.json.meta.json'))
    missing = [path for path in required if not (root / path).is_file()]
    if missing:
        return {'status': 'missing original evidence', 'files': missing}

    def read_model(model):
        directory = root / f'{model}_work'
        meta = read_json(directory / f'{model}_out.json.meta.json')
        if meta.get('status') != 'ok':
            return None
        items = read_json(directory / f'{model}_in.json')
        contexts = {item['mid']: item for item in items}
        if len(contexts) != len(items):
            raise ValueError(f'Duplicate {model} query context')
        return contexts, read_json(directory / f'{model}_out.json')

    ice, gl = (read_model(model) for model in ('ice', 'gl'))
    if ice is None or gl is None:
        return {'status': 'original simulator incomplete'}
    (ice_inputs, ice_scores), (gl_inputs, gl_scores) = ice, gl

    import pandas as pd
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize

    if rdBase.rdkitVersion != '2026.03.3':
        return {'status': 'metric RDKit is unavailable', 'rdkit': rdBase.rdkitVersion}
    tautomer = rdMolStandardize.TautomerEnumerator()

    @lru_cache(maxsize=20000)
    def describe(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        Chem.RemoveStereochemistry(mol)
        mol = tautomer.Canonicalize(mol)
        key = Chem.MolToInchiKey(mol)[:14]
        return key or None, rdMolDescriptors.CalcMolFormula(mol)

    def strictly_wins(context, scores, mid, first, challenger):
        if first not in context['cands'] or challenger not in context['cands']:
            return False
        values = scores.get(mid, {})
        a, b = values.get(first), values.get(challenger)
        return a is not None and b is not None and isfinite(a) and isfinite(b) and b > a

    pc = read_json(root / 'e004_candidate_inputs.json')['pubchem']
    evidence = pd.read_parquet(root / 'frozen_candidate_evidence.parquet',
                               columns=['molecule_id', 'library_max'])
    library = evidence.groupby('molecule_id')['library_max']
    if not library.nunique().eq(1).all():
        raise ValueError('Inconsistent Engine 1 library scores')
    library = library.first().to_dict()

    path = root / 'submission.csv'
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != ['molecule_id', 'smiles'] or not frame['molecule_id'].is_unique:
        raise ValueError('Invalid incoming competition candidates')
    changed_ranks = []
    answers = frame['smiles'].tolist()
    for index, (mid, answer) in enumerate(frame.itertuples(index=False, name=None)):
        lib_max = library.get(mid)
        proposals = pc.get(mid, {}).get('pc', [])
        if lib_max is None or not isfinite(lib_max) or lib_max >= 0.9 or not proposals:
            continue
        ice_query, gl_query = ice_inputs.get(mid), gl_inputs.get(mid)
        if not ice_query or not gl_query or not ice_query['spectra'] or ice_query['spectra'] != gl_query['spectra']:
            continue
        candidates = answer.split(';')
        if not 2 <= len(candidates) <= 25 or any(not smiles for smiles in candidates):
            continue
        first, challenger = candidates[0], proposals[0]
        _, first_formula = describe(first)
        challenger_key, challenger_formula = describe(challenger)
        if not challenger_key or not first_formula or first_formula == challenger_formula:
            continue
        if not strictly_wins(ice_query, ice_scores, mid, first, challenger):
            continue
        if not strictly_wins(gl_query, gl_scores, mid, first, challenger):
            continue
        matches = [i for i, smiles in enumerate(candidates) if describe(smiles)[0] == challenger_key]
        if len(matches) != 1 or matches[0] == 0:
            continue
        rank, = matches
        candidates.insert(0, candidates.pop(rank))
        answers[index] = ';'.join(candidates)
        changed_ranks.append(rank + 1)

    if changed_ranks:
        frame['smiles'] = answers
    if changed_ranks or output_path is not None:
        output_path = output_path if output_path is not None else path
        temporary = output_path.with_suffix('.tmp.csv')
        frame.to_csv(temporary, index=False)
        temporary.replace(output_path)
    return {'status': 'ok', 'molecules': len(frame), 'moved_from': sorted(changed_ranks)}


def build(original, destination):
    metadata = json.loads((original / 'kernel-metadata.json').read_text())
    if metadata['id'] != ORIGINAL_ID or destination.resolve() == original.resolve():
        raise ValueError('Use the Taeyang v2 pull and a separate destination')
    notebook = json.loads((original / metadata['code_file']).read_text())
    frozen = base64.b64decode(literal_value(source(notebook, 13), 'FROZEN_B64'), validate=True)
    if sha256(frozen).hexdigest() != ORIGINAL_FROZEN_SHA:
        raise ValueError('The frozen source differs from verified Taeyang v2')
    if '_NNET_ARM = "D"' not in source(notebook, APEX_CELL - 1) or 'def _apex_extlib_overlay(' not in source(notebook, APEX_CELL):
        raise ValueError('Expected adjacent Taeyang v2 D-L1 and APEX cells')

    call = '''
try:
    print('PUBCHEM_CROSS_FORMULA', promote_cross_formula(ROOT, PIPELINE_DONE), flush=True)
except Exception as error:
    print('PUBCHEM_CROSS_FORMULA original candidates kept:', repr(error), flush=True)
'''
    notebook['cells'].insert(APEX_CELL, code_cell(getsource(promote_cross_formula) + call))
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            compile(source(notebook, index), f'taeyang_cell_{index}', 'exec')

    metadata.pop('id_no', None)
    metadata.update(id=f'jiyujia/{SLUG}', title=SLUG, code_file=f'{SLUG}.ipynb', is_private=True)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / metadata['code_file']).write_text(json.dumps(notebook, ensure_ascii=False) + '\n')
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original', type=Path, help='Directory containing the original v2 Kaggle pull')
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    build(args.original, args.destination)

"""Run the 068a PubChem channel on MassBank with deeper candidate lists."""

import argparse
import base64
import json
from pathlib import Path

from build_kernel import BASELINE, code_cell, literal_value, replace, source


SLUG = 'enveda-075b-massbank-pubchem-depth'
PANEL = 'jiyujia/enveda-075b-external-massbank-202611'
PIPELINE = 'jiyujia/enveda-075b-massbank-pipeline'


def build(destination, gpu):
    path, = BASELINE.glob('*.ipynb')
    original = json.loads(path.read_text())
    frozen = json.loads(base64.b64decode(literal_value(source(original, 12), 'FROZEN_B64')))
    chemical_setup = source(frozen, 26).split('V4CODE = ', 1)[0]
    chemical_setup = replace(chemical_setup, "FOLD0_PATH = str(FOLD / 'fpnet_fold0.pt')", '''FOLD = Path(find('fpnet_fold0.pt')).parent
POP = Path(find('pc_src_mask.npy')).parent
FOLD0_PATH = str(FOLD / 'fpnet_fold0.pt')''')
    setup = source(frozen, 2) + '\n' + chemical_setup + '''
V3CODE = os.path.dirname(os.path.dirname(find('casmi26-v3-models/**/casmi/engine.py')))
V3MOD = os.path.dirname(find('casmi26-v3-models/**/ranker_0.pkl'))
POOL_DIR = os.path.dirname(find('pool_meta.parquet'))
PC_DIR = os.path.dirname(find('pc_smiles.npy'))
TEST = find('pipeline_queries.parquet')
T0 = time.time()
'''

    block = source(frozen, 34)
    core = replace(literal_value(block, 'CORE'), 'K = 25', 'K = 100')
    prior = replace(literal_value(block, 'V17_PATCH'), "    diag['n_listed'] = len(out)", '''    raw_candidates, raw_keys = [], set()
    for j in np.argsort(-fz, kind='stable'):
        k = key(j)
        if k is None or k in raw_keys or k in _W['pool_keys']:
            continue
        raw_keys.add(k)
        raw_candidates.append(smis[idx[j]])
        if len(raw_candidates) == K:
            break
    diag['raw_candidates'] = raw_candidates
    diag['n_listed'] = len(out)''')
    runner = replace(literal_value(block, 'RUNNER'), 'promotion_evidence=d.get',
                     "raw_candidates=d.get('raw_candidates', []), promotion_evidence=d.get")

    run = f'''
work = Path('/kaggle/working')
(work / 'probe_core2.py').write_text({(core + prior)!r})
(work / 'pc_runner.py').write_text({runner!r})
subprocess.run([sys.executable, '-u', str(work / 'pc_runner.py'), V3CODE, V3MOD, POOL_DIR, PC_DIR,
                TEST, str(work / 'pc_depth.json'), '4'], check=True)
'''

    evaluate = '''
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from functools import lru_cache

RDLogger.DisableLog('rdApp.*')
tautomers = rdMolStandardize.TautomerEnumerator()

@lru_cache(maxsize=20000)
def identity(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    Chem.RemoveStereochemistry(molecule)
    return Chem.MolToInchiKey(tautomers.Canonicalize(molecule))[:14]

def rank(keys, truth):
    return next((i for i, key in enumerate(keys, 1) if key == truth), None)

labels = pd.read_parquet(Path(TEST).parent / 'labels.parquet')
previous = {mid: [identity(smiles) for smiles in record['pc']]
            for mid, record in json.loads(Path(find('pc_lists.json')).read_text()).items()}
reranked = pd.read_parquet(find('source_candidates.parquet'))
reranked = {mid: rows['candidate_id'].tolist()
            for mid, rows in reranked[reranked['source'].eq('pubchem')].sort_values('rank').groupby('molecule_id')}
depth = json.loads((work / 'pc_depth.json').read_text())
if set(depth) != set(labels['molecule_id']):
    raise ValueError('PubChem depth run did not return the full external panel')

results = []
for row in labels.itertuples(index=False):
    record = depth[row.molecule_id]
    keys = {arm: [identity(smiles) for smiles in record[name]]
            for arm, name in [('with_prior', 'pc'), ('fingerprint_only', 'raw_candidates')]}
    original = previous.get(row.molecule_id, [])
    reproduced = keys['with_prior'][:25]
    results.append(dict(molecule_id=row.molecule_id,
                        prior_head_matches=reproduced == original,
                        prior_top1_matches=reproduced[:1] == original[:1],
                        prior_top25_shared=len(set(reproduced) & set(original)),
                        previous_rank=rank(original, row.candidate_id),
                        previous_ice_gl_rank=rank(reranked.get(row.molecule_id, []), row.candidate_id),
                        **{arm: rank(items, row.candidate_id) for arm, items in keys.items()}))

results = pd.DataFrame(results)
results.to_csv(work / 'pc_depth_ranks.csv', index=False)
summary = {'molecules': len(results), 'agree_original_top25': int(results['prior_head_matches'].sum()),
           'agree_original_top1': int(results['prior_top1_matches'].sum()),
           'mean_shared_top25': float(results['prior_top25_shared'].mean()), 'arms': {}}
for arm in ('previous_rank', 'previous_ice_gl_rank', 'with_prior', 'fingerprint_only'):
    ranks = pd.to_numeric(results[arm], errors='coerce')
    summary['arms'][arm] = {
        'rank1': int(ranks.eq(1).sum()), 'mrr25': float((1 / ranks.where(ranks.le(25))).fillna(0).mean()),
        **{f'found_{cutoff}': int(ranks.le(cutoff).sum()) for cutoff in (25, 50, 100)},
    }
(work / 'pc_depth_summary.json').write_text(json.dumps(summary, indent=2) + '\\n')
print(json.dumps(summary), flush=True)
'''

    metadata = json.loads((BASELINE / 'kernel-metadata.json').read_text())
    metadata.update(id=f'jiyujia/{SLUG}', title=SLUG, code_file=f'{SLUG}.ipynb', enable_gpu=gpu,
                    dataset_sources=[asset for asset in metadata['dataset_sources'] if asset.endswith((
                        'fold-safe-fpnet', 'pubchem-popularity-prior', 'v4g-reproduction-assets',
                        'rdkit-2026-3-3-wheel'))], kernel_sources=[PANEL, PIPELINE], competition_sources=[])
    if not gpu:
        metadata.pop('machine_shape', None)
    original['cells'] = [code_cell(text) for text in (setup, run, evaluate)]
    for i, cell in enumerate(original['cells']):
        compile(cell['source'], f'pubchem_depth_cell_{i}', 'exec')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / metadata['code_file']).write_text(json.dumps(original, ensure_ascii=False) + '\n')
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    build(args.destination, args.gpu)

"""Compare PubChem fingerprint dot product and soft Tanimoto on the same candidates."""

import argparse
import base64
import json
from pathlib import Path

from build_kernel import BASELINE, code_cell, literal_value, replace, source
from build_pubchem import candidate_setup


SLUG = 'enveda-075b-massbank-pc-tanimoto'
PANEL = 'jiyujia/enveda-075b-external-massbank-202611/1'
MEMBERSHIP = 'jiyujia/enveda-075b-external-membership/2'
DEPTH = 'jiyujia/enveda-075b-massbank-pubchem-depth/2'


def soft_tanimoto_patch(prior):
    prior = replace(prior, 'fz, idx = [], []', '''fz, soft, idx = [], [], []
    # Undo the 1.5 scaling in molecule_logits before applying sigmoid.
    probability = 1 / (1 + np.exp(-np.clip(z / 1.5, -40, 40)))
    query_bits = float(probability.sum())''')
    prior = replace(prior, '        fz.append(float(fp[bits].astype(np.float32) @ z)); idx.append(int(i))', '''        candidate = fp[bits].astype(np.float32)
        intersection = float(candidate @ probability)
        soft.append(intersection / (query_bits + float(candidate.sum()) - intersection))
        fz.append(float(candidate @ z))
        idx.append(int(i))''')
    return replace(prior, "    diag['n_listed'] = len(out)", '''    soft = np.asarray(soft, np.float64)
    soft_z = (soft - soft.mean()) / (soft.std() + 1e-9)
    soft_score = soft_z + _lam * pop[idx]
    soft_best = []
    soft_seen = set()
    for j in np.argsort(-soft_score, kind='stable'):
        k = key(j)
        if k is None or k in soft_seen or k in _W['pool_keys']:
            continue
        soft_seen.add(k)
        soft_best.append((k, smis[idx[j]], float(soft_score[j])))
        if len(soft_best) == K:
            break
    diag['soft_tanimoto'] = soft_best
    diag['n_listed'] = len(out)''')


EVALUATE = """
import json
from pathlib import Path

import pandas as pd


def notebook_file(slug, filename):
    paths = [path for path in Path('/kaggle/input').rglob(filename) if slug in path.parts]
    if len(paths) != 1:
        raise FileNotFoundError(f'{slug}/{filename}: {paths}')
    return paths[0]


def native_rank(keys, truth):
    return next((rank for rank, key in enumerate(keys, 1) if key == truth), 0)


def metrics(old, new):
    report = {}
    for k in (5, 25, 50, 100):
        old_hit = old.between(1, k)
        new_hit = new.between(1, k)
        report[str(k)] = {
            'dot_hits': int(old_hit.sum()),
            'soft_hits': int(new_hit.sum()),
            'new_hits': int((new_hit & ~old_hit).sum()),
            'lost_hits': int((old_hit & ~new_hit).sum()),
        }
    report['dot_mrr_25'] = float((1 / old.where(old.between(1, 25))).fillna(0).mean())
    report['soft_mrr_25'] = float((1 / new.where(new.between(1, 25))).fillna(0).mean())
    return report


scores = json.loads((work / 'pc_tanimoto_candidates.json').read_text())
ranks = pd.read_parquet(Path(TEST).parent / 'labels.parquet', columns=['molecule_id', 'candidate_id'])
if not ranks.molecule_id.is_unique or ranks.empty or set(ranks.molecule_id) != scores.keys():
    raise ValueError('Label and scoring arms must cover the same nonempty set of unique structures')
ranks['dot'] = [native_rank(scores[mid]['pc_keys'], truth)
                for mid, truth in ranks[['molecule_id', 'candidate_id']].itertuples(index=False, name=None)]
ranks['soft'] = [native_rank([row[0] for row in scores[mid]['soft_tanimoto']], truth)
                 for mid, truth in ranks[['molecule_id', 'candidate_id']].itertuples(index=False, name=None)]
report = {'panel_size': len(ranks), 'all': metrics(ranks['dot'], ranks['soft'])}
if MASSBANK:
    membership = pd.read_parquet(notebook_file('enveda-075b-external-membership', 'structure_membership.parquet'),
                                 columns=['molecule_id', 'membership', 'pubchem_tier_observed_10ppm'])
    ranks = ranks.merge(membership, on='molecule_id', validate='one_to_one')
    ranks['pubchem_only'] = ranks.membership.eq('pubchem_store_found') & ranks.pubchem_tier_observed_10ppm
    prior = pd.read_csv(notebook_file('enveda-075b-massbank-pubchem-depth', 'pc_depth_ranks.csv'),
                        usecols=['molecule_id', 'with_prior'])
    ranks = ranks.merge(prior, on='molecule_id', validate='one_to_one')
    if len(ranks) != 110 or int(ranks.pubchem_only.sum()) != 99:
        raise ValueError('Expected the original 110 structures and 99 main-pool-absent structures')
    known = ranks['with_prior'].fillna(0)
    report.update(
        pubchem_only_size=int(ranks.pubchem_only.sum()),
        dot_truth_ranks_exact_vs_published_t4=int(ranks['dot'].eq(known).sum()),
        dot_membership_25_agrees_vs_published_t4=int(ranks['dot'].between(1, 25).eq(known.between(1, 25)).sum()),
        pubchem_only=metrics(ranks.loc[ranks.pubchem_only, 'dot'], ranks.loc[ranks.pubchem_only, 'soft']),
    )
ranks.to_parquet(work / 'pc_tanimoto_ranks.parquet', index=False)
(work / 'pc_tanimoto_summary.json').write_text(json.dumps(report, indent=2) + '\\n')
print(json.dumps(report), flush=True)
"""


def build(destination, panel=PANEL, slug=SLUG):
    notebook_path, = BASELINE.glob('*.ipynb')
    notebook = json.loads(notebook_path.read_text())
    frozen = json.loads(base64.b64decode(literal_value(source(notebook, 12), 'FROZEN_B64')))
    block = source(frozen, 34)
    core = replace(literal_value(block, 'CORE'), 'K = 25', 'K = 100')
    prior = soft_tanimoto_patch(literal_value(block, 'V17_PATCH'))
    runner = replace(literal_value(block, 'RUNNER'), 'promotion_evidence=d.get',
                     "soft_tanimoto=d.get('soft_tanimoto', []), promotion_evidence=d.get")
    run = f"""
work = Path('/kaggle/working')
(work / 'probe_core2.py').write_text({(core + prior)!r})
(work / 'pc_runner.py').write_text({runner!r})
subprocess.run([sys.executable, '-u', str(work / 'pc_runner.py'), V3CODE, V3MOD, POOL_DIR,
                PC_DIR, TEST, str(work / 'pc_tanimoto_candidates.json'), '4'], check=True)
"""
    known_massbank = panel == PANEL
    evaluate = f'MASSBANK = {known_massbank}\n' + EVALUATE
    notebook['cells'] = [code_cell(text) for text in (candidate_setup(frozen), run, evaluate)]
    for i, cell in enumerate(notebook['cells']):
        compile(cell['source'], f'pc_tanimoto_cell_{i}', 'exec')
    metadata = json.loads((BASELINE / 'kernel-metadata.json').read_text())
    metadata.update(
        id=f'jiyujia/{slug}', title=slug, code_file=f'{slug}.ipynb', enable_gpu=False, keywords=['chemistry'],
        dataset_sources=[asset for asset in metadata['dataset_sources'] if asset.endswith((
            'fold-safe-fpnet', 'pubchem-popularity-prior', 'v4g-reproduction-assets', 'rdkit-2026-3-3-wheel'))],
        kernel_sources=[panel, MEMBERSHIP, DEPTH] if known_massbank else [panel], competition_sources=[],
    )
    metadata.pop('machine_shape', None)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / metadata['code_file']).write_text(json.dumps(notebook, ensure_ascii=False) + '\n')
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--panel', default=PANEL)
    parser.add_argument('--slug', default=SLUG)
    args = parser.parse_args()
    if args.panel != PANEL and args.slug == SLUG:
        parser.error('Use a separate --slug for another panel')
    build(args.destination, args.panel, args.slug)

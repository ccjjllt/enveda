"""Replay the fixed insertion of a novel depth-50 PubChem leader."""

from collections import Counter
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
import json
from math import isclose, isfinite
from pathlib import Path
import subprocess
import sys

import pandas as pd


INPUT = Path('/kaggle/input')
OUTPUT = Path('/kaggle/working')
PANEL = 'enveda-075b-external-massbank-202611'
BASE = 'enveda-075b-massbank-pipeline'
DEPTH = 'enveda-075b-massbank-pubchem50'
FIELDS = ['molecule_id', 'candidate_id', 'smiles', 'formula', 'rank']


def artifact(run, name):
    direct = INPUT / 'notebooks' / 'jiyujia' / run / name
    if direct.is_file():
        return direct
    paths = [p for p in INPUT.rglob(Path(name).name) if run in p.parts and p.match(name)]
    if len(paths) != 1:
        raise FileNotFoundError(f'Expected one {run}/{name}, found {paths}')
    return paths[0]


def native(run, kind):
    prefix = f'{kind}_work/{kind}_'
    items, scores, meta = [json.loads(artifact(run, prefix + suffix).read_text())
                           for suffix in ('in.json', 'out.json', 'out.json.meta.json')]
    contexts = {item['mid']: item for item in items}
    expected = {mid: set(item['cands']) for mid, item in contexts.items()}
    n_scored = sum(value is not None and isfinite(value) for row in scores.values() for value in row.values())
    if (meta['status'], meta['n_mols'], meta['n_cands'], meta['n_scored']) != (
        'ok', len(items), sum(map(len, expected.values())), n_scored
    ) or len(contexts) != len(items) or scores.keys() != expected.keys() or any(
        row.keys() != expected[mid] for mid, row in scores.items()
    ):
        raise ValueError(f'{run} {kind} native receipt, input and scores are inconsistent')
    return contexts, scores, {name: meta[name] for name in ('status', 'n_mols', 'n_cands', 'n_scored')}


def identity_reader():
    try:
        installed = version('rdkit').replace('.03.', '.3.')
    except PackageNotFoundError:
        installed = ''
    if installed != '2026.3.3':
        abi = f'cp{sys.version_info.major}{sys.version_info.minor}'
        wheels = sorted(INPUT.rglob(f'rdkit-2026.3.3-{abi}-*.whl'))
        if not wheels:
            raise FileNotFoundError('The official metric RDKit wheel is unavailable')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--quiet',
                        str(wheels[0])], check=True)
    from rdkit import Chem, RDLogger, rdBase
    from rdkit.Chem.MolStandardize import rdMolStandardize

    if rdBase.rdkitVersion != '2026.03.3':
        raise RuntimeError(f'Expected the official RDKit version, got {rdBase.rdkitVersion}')
    RDLogger.DisableLog('rdApp.*')
    tautomer = rdMolStandardize.TautomerEnumerator()

    @lru_cache(maxsize=5000)
    def read(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError('The original retrieval contains invalid SMILES')
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToInchiKey(tautomer.Canonicalize(mol))[:14]

    return read


def score_pair(physical, mid, first, challenger):
    contexts, scores, _ = physical
    context = contexts.get(mid)
    if context is None or not set((first, challenger)) <= set(context['cands']):
        return None, None
    both = tuple(scores[mid].get(smiles) for smiles in (first, challenger))
    return both if all(x is not None and isfinite(x) for x in both) else (None, None)


def metrics(frame):
    old, new = frame['original_rank'], frame['new_rank']
    return dict(original_mrr=float((1 / old.where(old.gt(0))).fillna(0).mean()),
                new_mrr=float((1 / new.where(new.gt(0))).fillna(0).mean()),
                original_first=int(old.eq(1).sum()), new_first=int(new.eq(1).sum()),
                original_top25=int(old.gt(0).sum()), new_top25=int(new.gt(0).sum()),
                moved_to_first=int((old.ne(1) & new.eq(1)).sum()), lost_first=int((old.eq(1) & new.ne(1)).sum()),
                newly_found=int((old.eq(0) & new.gt(0)).sum()), lost_hit=int((old.gt(0) & new.eq(0)).sum()))


def main():
    models = {(run, kind): native(run, kind) for run in (BASE, DEPTH) for kind in ('ice', 'gl')}
    identity = identity_reader()
    final = pd.read_parquet(artifact(BASE, 'candidates.parquet'), columns=FIELDS)
    evidence = pd.read_parquet(artifact(BASE, 'frozen_candidate_evidence.parquet'),
                               columns=['molecule_id', 'library_max'])
    sources = pd.read_parquet(artifact(DEPTH, 'source_candidates.parquet'), columns=[*FIELDS, 'source'])
    leaders = sources.loc[sources['source'].eq('pubchem') & sources['rank'].eq(1)].set_index('molecule_id')
    raw = json.loads(artifact(BASE, 'pc_lists.json').read_text())
    groups = {mid: frame.sort_values('rank', kind='stable') for mid, frame in final.groupby('molecule_id')}
    library = evidence.groupby('molecule_id')['library_max']
    if (len(groups) != 110 or not leaders.index.is_unique or not set(leaders.index) <= groups.keys()
            or set(library.groups) != groups.keys() or not library.nunique().eq(1).all()):
        raise ValueError('The original and deeper files do not cover the fixed cohort')
    library = library.first().to_dict()
    coverage, ranks, output = Counter(), {}, []
    for mid, group in groups.items():
        old = group['candidate_id'].tolist()
        if len(old) != len(set(old)) or group['rank'].tolist() != list(range(1, len(old) + 1)) or len(old) > 25:
            raise ValueError(f'Invalid original ranks for {mid}')
        first = group.iloc[0]
        result = dict(molecule_id=mid, original_first=first.candidate_id, library_max=library[mid],
                      original_formula=first.formula, has_deeper_leader=mid in leaders.index,
                      baseline_and_deep_spectra_match=False, both_model_scores=False, inserted=False)
        contexts = [physical[0].get(mid) for physical in models.values()]
        spectra = [context['spectra'] if context else [] for context in contexts]
        spectra_match = bool(spectra[0]) and all(other == spectra[0] for other in spectra[1:])
        result['baseline_and_deep_spectra_match'] = spectra_match
        coverage['full_cohort_same_observed_spectra'] += int(spectra_match)
        new = old
        if mid not in leaders.index:
            result['status'] = 'no_deeper_leader'
        else:
            challenger = leaders.loc[mid]
            if challenger.candidate_id != identity(challenger.smiles):
                raise ValueError('Exported PubChem leader identity does not reproduce')
            no_final = challenger.candidate_id not in old
            raw_smiles = raw.get(mid, {}).get('pc', [])[:25]
            no_raw = challenger.smiles not in raw_smiles and not any(
                identity(smiles) == challenger.candidate_id for smiles in raw_smiles
            )
            weak = isfinite(library[mid]) and library[mid] < 0.90
            different = pd.notna(first.formula) and pd.notna(challenger.formula) and first.formula != challenger.formula
            result.update(challenger=challenger.candidate_id, challenger_formula=challenger.formula,
                          absent_original_final=no_final, absent_original_pubchem_raw=no_raw,
                          weak_e1=weak, different_formula=different)
            coverage.update(deeper_leader=1, absent_original_final=int(no_final), absent_original_pubchem_raw=int(no_raw),
                            absent_both_original_lists=int(no_final and no_raw),
                            also_weak_e1=int(no_final and no_raw and weak),
                            also_different_formula=int(no_final and no_raw and weak and different))
            if not no_final or not no_raw:
                result['status'] = 'present_in_original_final' if not no_final else 'present_in_original_pubchem_raw'
            elif not weak:
                result['status'] = 'strong_or_missing_e1'
            elif not different:
                result['status'] = 'same_or_missing_formula'
            elif not spectra_match:
                result['status'] = 'missing_or_different_native_spectra'
            else:
                coverage['eligible_same_observed_spectra'] += 1
                ice = score_pair(models[(DEPTH, 'ice')], mid, first.smiles, challenger.smiles)
                gl = score_pair(models[(DEPTH, 'gl')], mid, first.smiles, challenger.smiles)
                result.update(ice_first=ice[0], ice_challenger=ice[1], gl_first=gl[0], gl_challenger=gl[1])
                coverage.update(eligible_ice_pair=int(ice[0] is not None), eligible_gl_pair=int(gl[0] is not None))
                if ice[0] is None or gl[0] is None:
                    result['status'] = 'missing_native_pair_scores'
                else:
                    result['both_model_scores'] = True
                    win_ice, win_gl = ice[1] > ice[0], gl[1] > gl[0]
                    coverage.update(comparable=1, strict_ice_win=int(win_ice), strict_gl_win=int(win_gl))
                    if win_ice and win_gl:
                        result.update(status='inserted', inserted=True)
                        new = [challenger.candidate_id, *old][:25]
                    else:
                        result['status'] = 'no_strict_dual_win'
        ranks[mid] = old, new
        output.append(result)

    labels = pd.read_parquet(artifact(PANEL, 'labels.parquet'), columns=['molecule_id', 'candidate_id'])
    if len(labels) != 110 or set(labels['molecule_id']) != groups.keys():
        raise ValueError('Truth table does not match the fixed 110 queries')
    frame = pd.DataFrame(output).merge(labels, on='molecule_id', validate='one_to_one')
    for i, name in enumerate(('original_rank', 'new_rank')):
        frame[name] = [next((j for j, key in enumerate(ranks[mid][i], 1) if key == true), 0)
                       for mid, true in frame[['molecule_id', 'candidate_id']].itertuples(index=False, name=None)]
    scored = metrics(frame)
    if (scored['original_first'], scored['original_top25']) != (33, 58) or not isclose(
        scored['original_mrr'], 0.3368516738288762, abs_tol=1e-12
    ):
        raise ValueError('Original MassBank candidate order did not replay')
    report = dict(metrics=scored, mrr_delta=scored['new_mrr'] - scored['original_mrr'],
                  native={f'{run}/{kind}': physical[2] for (run, kind), physical in models.items()},
                  coverage=dict(coverage), statuses=dict(Counter(frame['status'])),
                  inserted=frame.loc[frame['inserted'], ['molecule_id', 'original_rank', 'new_rank']].to_dict('records'))
    frame.to_parquet(OUTPUT / 'novel_leader_ranks.parquet', index=False)
    (OUTPUT / 'novel_leader_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

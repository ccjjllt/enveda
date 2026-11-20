"""Compare PubChem depth using the frozen ICEBERG + GLACIER formula slots."""

from dataclasses import dataclass
from functools import lru_cache
import glob
import importlib.util
import json
import os
from pathlib import Path
import time

import pandas as pd


INPUT = Path('/kaggle/input')
OUTPUT = Path('/kaggle/working')
PANEL = 'enveda-075b-external-massbank-202611'
PIPELINE = 'enveda-075b-massbank-pipeline'
DEPTH = 'enveda-075b-massbank-pubchem-depth'
RUN_DIR = Path('/tmp/pubchem_forward')


def artifact(notebook, filename, optional=False):
    direct = INPUT / 'notebooks' / 'jiyujia' / notebook / filename
    if direct.is_file():
        return direct
    matches = [p for p in INPUT.rglob(Path(filename).name) if notebook in p.parts and p.match(filename)]
    if not matches and optional:
        return None
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one {notebook}/{filename}, found {matches}')
    return matches[0]


def read_json(path):
    return json.loads(Path(path).read_text())


def package(name, runner):
    paths = sorted((p for p in INPUT.rglob(runner) if name in p.parts), key=lambda p: len(str(p)))
    if not paths:
        raise FileNotFoundError(f'No {name}/{runner} mounted')
    return paths[0].parent


def import_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Candidate:
    smiles: str
    key: str
    score: float
    formula: str
    metric_key: str


@lru_cache(maxsize=20000)
def describe(smiles):
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Unparseable saved PubChem structure {smiles!r}')
    # The formula grouping precedes metric normalization in frozen cell 41.
    formula = CalcMolFormula(mol)
    Chem.RemoveStereochemistry(mol)
    identity = Chem.MolToInchiKey(TAUTOMERS.Canonicalize(mol))[:14]
    return formula, identity


def unpack(record, limit):
    return [Candidate(smi, key, float(score), *describe(smi))
            for smi, key, score in zip(record['pc'][:limit], record['pc_keys'][:limit],
                                       record['pc_fz'][:limit], strict=True)]


def eligible_tail(head, deep):
    formulas = {c.formula for c in head}
    keys = {c.metric_key for c in head}
    entries = []
    for raw_rank, candidate in enumerate(deep[25:], 26):
        if candidate.formula in formulas and candidate.metric_key not in keys:
            entries.append((raw_rank, candidate))
            keys.add(candidate.metric_key)
    return entries


def rank_group(cands, ice, gl, ice_fuse, gl_fuse):
    smis, keys, scores, formulas = ([getattr(c, name) for c in cands]
                                    for name in ('smiles', 'key', 'score', 'formula'))
    if not ice:
        return cands[:25]
    order = ice_fuse.rerank(smis, keys, scores, formulas, ice, lam=1.0, top_n=len(cands))
    if gl and any(value is not None for value in gl.values()):
        order = gl_fuse.rerank_multi(smis, keys, scores, formulas, [ice, gl], [1.0, 1.0], top_n=len(cands))
    return [cands[i] for i in order[:25]]


def recorded_ranks():
    frame = pd.read_parquet(artifact(PIPELINE, 'source_candidates.parquet'),
                            columns=['molecule_id', 'smiles', 'source', 'rank'])
    frame = frame[frame['source'].eq('pubchem')].sort_values('rank', kind='stable')
    return frame.groupby('molecule_id')['smiles'].agg(list).to_dict()


def native_inputs(ice_fuse, spectra, lists):
    candidates = {mid: ice_fuse.ice_candidates([c.smiles for c in cands], [c.formula for c in cands], len(cands))
                  for mid, cands in lists.items()}
    return ice_fuse.build_ice_input(spectra, {mid: smis for mid, smis in candidates.items() if smis})


def protonated(items):
    items = [dict(item, spectra=[s for s in item['spectra'] if s['adduct'] == '[M+H]+']) for item in items]
    return [item for item in items if item['spectra']]


def filter_scores(items, scores):
    return {i['mid']: {smi: scores.get(i['mid'], {}).get(smi) for smi in i['cands']} for i in items}


def cached_scores(kind, items):
    paths = [artifact(PIPELINE, f'{kind}_work/{kind}_{suffix}', optional=True)
             for suffix in ('in.json', 'out.json', 'out.json.meta.json')]
    if not all(paths):
        return None, 'saved input, scores or run metadata unavailable'
    original, scores, meta = map(read_json, paths)
    if meta['status'] != 'ok':
        return None, f'original simulator status {meta["status"]}'
    spectra = {item['mid']: item['spectra'] for item in items}
    reference = {item['mid']: item['spectra'] for item in original}
    bad = [mid for mid, values in spectra.items() if mid in reference and values != reference[mid]]
    if bad:
        return None, f'{len(bad)} changed query spectra'
    # New formula groups can require scoring an incumbent that the original run never requested.
    return {mid: values for mid, values in scores.items() if mid in reference}, 'same saved spectra'


def pending_jobs(items, cache):
    pending = []
    for item in items:
        scores = cache.get(item['mid'], {})
        remaining = [smi for smi in item['cands'] if smi not in scores]
        if remaining:
            pending.append(dict(item, cands=remaining))
    return pending


def check_model_output(kind, work, scores, items):
    meta = read_json(work / f'{kind}_out.json.meta.json')
    if meta.get('status') != 'ok' or not scores:
        raise RuntimeError(f'{kind} did not finish requested predictions: {meta}')
    missing = sum(smi not in scores.get(item['mid'], {}) for item in items for smi in item['cands'])
    if missing:
        raise RuntimeError(f'{kind} missed {missing} requested candidate entries')
    return meta


def merge_scores(saved, fresh):
    return {mid: {**saved.get(mid, {}), **fresh.get(mid, {})} for mid in saved.keys() | fresh.keys()}


def summaries(results, columns, control):
    ref = results[control]
    report = {}
    for name in columns:
        r = results[name]
        rr = (1 / r.where(r.gt(0))).fillna(0)
        report[name] = dict(found_at_25=int(r.gt(0).sum()), recall_at_25=float(r.gt(0).mean()),
                            mrr_at_25=float(rr.mean()), rank1=int(r.eq(1).sum()),
                            newly_found=int((ref.eq(0) & r.gt(0)).sum()),
                            lost_hit=int((ref.gt(0) & r.eq(0)).sum()),
                            moved_to_first=int((ref.ne(1) & r.eq(1)).sum()),
                            lost_first=int((ref.eq(1) & r.ne(1)).sum()))
    return report


def main():
    started = time.monotonic()
    from rdkit import RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize

    global TAUTOMERS
    RDLogger.DisableLog('rdApp.*')
    TAUTOMERS = rdMolStandardize.TautomerEnumerator()
    ice_pkg = package('casmi26-iceberg', 'ice_runner.py')
    gl_pkg = package('casmi26-glacier', 'gl_runner.py')
    ice_fuse = import_source('pubchem_ice_fuse', ice_pkg / 'fuse.py')
    gl_fuse = import_source('pubchem_gl_fuse', gl_pkg / 'gl_fuse.py')
    raw = read_json(artifact(PIPELINE, 'pc_lists.json'))
    deeper = read_json(artifact(DEPTH, 'pc_depth.json'))
    stored = recorded_ranks()
    if raw.keys() != deeper.keys() or raw.keys() != stored.keys():
        raise ValueError('Stored raw, stored rerank, and deep PubChem rows must cover the same query molecules')
    spectra = pd.read_parquet(artifact(PANEL, 'pipeline_queries.parquet'))

    heads = {mid: unpack(record, 25) for mid, record in raw.items()}
    deep = {mid: unpack(deeper[mid], 100) for mid in heads}
    tails = {mid: eligible_tail(heads[mid], deep[mid]) for mid in heads}
    lists = {width: {mid: heads[mid] + [cand for rank, cand in tails[mid] if rank <= width] for mid in heads}
             for width in (50, 100)}
    raw_match = {key: sum(test(heads[mid], deep[mid][:25]) for mid in heads) for key, test in (
        ('exact_smiles_order', lambda a, b: [x.smiles for x in a] == [x.smiles for x in b]),
        ('exact_metric_order', lambda a, b: [x.metric_key for x in a] == [x.metric_key for x in b]),
        ('same_metric_set', lambda a, b: {x.metric_key for x in a} == {x.metric_key for x in b}),
    )}
    if any(len(head) != 25 for head in heads.values()):
        raise ValueError('Expected 25 original PubChem candidates for every molecule')
    print('Deep top25 agreement with fixed saved head:', raw_match, flush=True)
    print('Additional same-formula candidates:',
          {width: sum(len(v) - 25 for v in mapping.values()) for width, mapping in lists.items()}, flush=True)
    original_items = native_inputs(ice_fuse, spectra, heads)
    items = native_inputs(ice_fuse, spectra, lists[100])
    gl_items = protonated(items)
    ice, ice_cache_status = cached_scores('ice', items)
    gl, gl_cache_status = cached_scores('gl', gl_items)
    reused = False
    if ice is not None and gl is not None:
        prior_replay = {mid: [c.smiles for c in rank_group(head, ice.get(mid, {}), gl.get(mid, {}), ice_fuse, gl_fuse)]
                        for mid, head in heads.items()}
        reused = all(prior_replay[mid] == stored[mid] for mid in heads)
        print(f'Saved scores replay {sum(prior_replay[mid] == stored[mid] for mid in heads)}/{len(heads)} exact lists', flush=True)
    print(f'ICE cache: {ice_cache_status}; GL cache: {gl_cache_status}; reuse={reused}', flush=True)
    ice, gl = (ice, gl) if reused else ({}, {})
    base_ice, base_gl = ice, gl
    ice_pending = pending_jobs(items, ice)
    gl_pending = pending_jobs(gl_items, gl)
    print(f'Fresh candidates ICE={sum(len(x["cands"]) for x in ice_pending)} '
          f'GL={sum(len(x["cands"]) for x in gl_pending)}', flush=True)
    native_meta = {}
    if ice_pending or gl_pending:
        site = RUN_DIR / 'ice_site'
        _claw_preseed_site(str(ice_pkg), str(site))
        if ice_pending:
            directory = RUN_DIR / 'ice_work'
            fresh = ice_fuse.run_ice(str(ice_pkg), ice_pending, workdir=str(directory), site=str(site),
                                     device='cuda', budget_s=900, grace_s=45, extra_args=['--any-rdkit'])
            native_meta['ice'] = check_model_output('ice', directory, fresh, ice_pending)
            ice = merge_scores(ice, fresh)
        if gl_pending:
            directory = RUN_DIR / 'gl_work'
            fresh = gl_fuse.run_gl(str(gl_pkg), gl_pending, workdir=str(directory), site=str(site), wheels=str(ice_pkg / 'wheels'),
                                  device='cuda', budget_s=900, grace_s=45, extra_args=['--any-rdkit'])
            native_meta['gl'] = check_model_output('gl', directory, fresh, gl_pending)
            gl = merge_scores(gl, fresh)
    if not reused:
        base_ice = filter_scores(original_items, ice)
        base_gl = filter_scores(protonated(original_items), gl)

    def rank_lists(mapping, ice_scores, gl_scores):
        return {mid: rank_group(cands, ice_scores.get(mid, {}), gl_scores.get(mid, {}), ice_fuse, gl_fuse)
                for mid, cands in mapping.items()}

    controls = rank_lists(heads, base_ice, base_gl)
    arms = {'raw_25': heads, 'replay_25': controls,
            'depth_50': rank_lists(lists[50], ice, gl), 'depth_100': rank_lists(lists[100], ice, gl)}
    for name, mapping in arms.items():
        if name.startswith('depth_') and any(
            [c.formula for c in mapping[mid]] != [c.formula for c in heads[mid]] for mid in heads
        ):
            raise RuntimeError(f'{name} changed the original top25 formula slots')
    submitted = {mid: [describe(smi)[1] for smi in smis] for mid, smis in stored.items()}
    predictions = {name: {mid: [c.metric_key for c in entries] for mid, entries in mapping.items()}
                   for name, mapping in arms.items()}
    predictions['stored_25'] = submitted
    # Read labels after all candidate construction and scoring are finished.
    labels = pd.read_parquet(artifact(PANEL, 'labels.parquet'), columns=['molecule_id', 'candidate_id'])
    results = labels[['molecule_id']].copy()
    for name, mapping in predictions.items():
        results[name] = [next((i for i, key in enumerate(mapping[mid], 1) if key == truth), 0)
                         for mid, truth in labels.itertuples(index=False, name=None)]
    result = dict(molecules=len(labels), raw_head_alignment=raw_match, reused_simulator_scores=reused,
                  replay_exact_smiles=sum([c.smiles for c in controls[mid]] == stored[mid] for mid in heads),
                  replay_exact_metric=sum(predictions['replay_25'][mid] == submitted[mid] for mid in heads),
                  total_same_formula_tail={w: sum(len(entries) - 25 for entries in mapping.values())
                                           for w, mapping in lists.items()},
                  tail_admitted={name: sum(len(set(mapping[mid]) - set(predictions['raw_25'][mid])) for mid in heads)
                                 for name, mapping in predictions.items() if name.startswith('depth_')},
                  relative_to_replay_25=summaries(results, predictions, 'replay_25'),
                  relative_to_stored_25=summaries(results, ('depth_50', 'depth_100'), 'stored_25'),
                  native_runs={k: {f: m.get(f) for f in ('status', 'n_cands', 'n_scored', 'seconds')}
                               for k, m in native_meta.items()}, seconds=round(time.monotonic() - started, 1))
    results.to_csv(OUTPUT / 'pc_forward_ranks.csv', index=False)
    (OUTPUT / 'pc_forward_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

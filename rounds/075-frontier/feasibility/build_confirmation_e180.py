"""Build the CPU confirmation of fixed absolute fragment coverage on the 121 archived Enveda-180 cases."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


SLUG = 'enveda-075-absolute-confirm-e180'
PREDICTION = 'enveda-075-first-formula-baseline-e180'
BLIND = 'enveda-075-e180-full-blind-input'
SCORED = 'enveda-075-e180-full-compare'
PINS = {
    (PREDICTION, 'submission.csv'): '2675b3cec0607a93b447c4bfbcac256812bc796cf8794b81fc8414c24fe1f55a',
    (PREDICTION, 'e004_candidate_inputs.json'): '7fc3f74d2f88c9952e11b931ef6624e3fd51c374203d1828b691bed94e1a39c7',
    (PREDICTION, 'frozen_candidate_evidence.parquet'): '3b0895aee03299edda67599e255ef77dbd2311d5e82823a2418165b000065f59',
    (PREDICTION, 'e180_ranked_input.json'): '4311f3d653ed29dd598a4c55297a73c49c1ee38f170ee238bb5b6e459dc14a9f',
    (BLIND, 'e180_blind/test.parquet'): '6fe782c57481b6fea95d1197da05ce4968c393881e3d962bcf8cdaed09b3bf7b',
    (SCORED, 'e180_pair_ranks.csv'): '46a7b2ff0d9a9f2778e0d099050ce36a5f69a5065c91e1c4d7d500c25d0afdf0',
    (SCORED, 'e180_pair_summary.json'): '13ad475c56f43e3c39773265a81fefd31f6bc8f665edcb7f4de5fef057bba921',
}


def mounted(inputs, slug, name, pins=PINS):
    suffix = Path(name)
    path, = (p for p in inputs.rglob(suffix.name)
             if slug in p.parts and p.parts[-len(suffix.parts):] == suffix.parts)
    if sha256(path.read_bytes()).hexdigest() != pins[slug, name]:
        raise ValueError(f'Archived input changed: {path}')
    return path


def source_heads(inputs, mids, prediction=PREDICTION, pins=PINS):
    import pandas as pd
    from rank import metric_identity

    raw = json.loads(mounted(inputs, prediction, 'e004_candidate_inputs.json', pins).read_text())
    frame = pd.read_parquet(mounted(inputs, prediction, 'frozen_candidate_evidence.parquet', pins),
                            columns=['molecule_id', 'smiles', 'metric_key', 'main_rank', 'engine2_rank', 'pubchem_rank'])
    if set(frame.molecule_id) != mids or any(set(raw[k]) != mids for k in ('base_tail', 'pubchem', 'engine2')):
        raise ValueError('Source archive differs from the complete query population')
    result = {}
    for mid, rows in frame.groupby('molecule_id', sort=False):
        # base_tail is captured before ICE/GL. The evidence map records the actual post-ICE/GL main rank.
        first, = rows.loc[rows.main_rank.eq(1)].itertuples(index=False)
        main_id = metric_identity(first.smiles)
        if main_id is None or main_id[0] != first.metric_key:
            raise ValueError(f'Inconsistent archived post-reranking main head for {mid}')
        main = raw['base_tail'][mid]
        if main[0] == first.smiles:
            main_first = first.smiles
        else:
            # An intermediate candidate can use a different SMILES representation of the same source hit.
            main_first, = [s for s in main if metric_identity(s) == main_id]
        lists = dict(main=[main_first], engine2=raw['engine2'][mid]['smiles'][:1],
                     pubchem=raw['pubchem'][mid].get('pc', [])[:1])
        for src in ('engine2', 'pubchem'):
            hits = rows.loc[rows[src + '_rank'].eq(1), 'smiles']
            if len(hits) > 1 or (len(hits) and (not lists[src] or metric_identity(hits.iloc[0]) != metric_identity(lists[src][0]))):
                raise ValueError(f'Archived {src} head disagrees with frozen source ranks for {mid}')
        result[mid] = lists
    return result


def rank_blind(inputs, out):
    import pandas as pd
    from rdkit import rdBase
    from rank import answers, digest, metric_identity, protects_first, rank_absolute, read_scans

    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError('Expected the competition metric RDKit')
    base_path = mounted(inputs, PREDICTION, 'submission.csv')
    base = answers(base_path)
    q = mounted(inputs, BLIND, 'e180_blind/test.parquet')
    receipt = json.loads(mounted(inputs, PREDICTION, 'e180_ranked_input.json').read_text())
    if (receipt['arm'] != 'baseline' or receipt['submission_sha256'] != digest(base_path)
            or receipt['inference']['sha256']['test.parquet'] != digest(q)
            or (receipt['inference']['molecules'], receipt['inference']['spectra']) != (121, 299)
            or len(base) != 121 or any(len(values) != 25 for values in base.values())):
        raise ValueError('Original 0.498 lists and measured query archive differ')
    deposited = pd.read_parquet(q, columns=['molecule_id', 'spectrum_id', 'adduct'])
    if (len(deposited) != 299 or set(deposited.molecule_id) != set(base) or not deposited.spectrum_id.is_unique
            or set(deposited.adduct) != {'[M+H]+'}):
        raise ValueError('The acquisition set differs from the complete archive')
    heads = source_heads(inputs, set(base))
    scans = read_scans(q)
    ranked, swaps = rank_absolute(base, heads, scans)
    if set(base) != set(ranked):
        raise ValueError('The intervention changed query population')
    for mid, values in base.items():
        if Counter(values) != Counter(ranked[mid]) or any(metric_identity(s) is None for s in values):
            raise ValueError(f'Candidate inventory changed or contains an invalid identity for {mid}')
    absolute = out / 'absolute.csv'
    pd.DataFrame([(mid, ';'.join(ranked[mid])) for mid in base], columns=['molecule_id','smiles']).to_csv(absolute,index=False)
    eligible = {s.mid for s in scans.values()}
    protected = {mid for mid in base if protects_first(base[mid][0], heads[mid])}
    seal = dict(panel='Enveda-180 reserved 121', source='goodpjw2008 v3 / 0.498', molecules=len(base),
                deposited_spectra=len(deposited), eligible_spectra=len(scans), eligible_molecules=len(eligible),
                candidate_slots=sum(map(len, base.values())), candidate_budget=25,
                stage_status=receipt['stage_status'], protected_leaders=len(protected),
                eligible_with_unprotected_leader=len(eligible - protected),
                nonempty_source_heads={src: sum(bool(v[src]) for v in heads.values()) for src in ('main','engine2','pubchem')},
                swaps=swaps, swaps_by_anchor=dict(Counter(s['anchor_rank'] for s in swaps)),
                baseline_sha256=digest(base_path), query_sha256=digest(q), absolute_sha256=digest(absolute),
                source_inputs_sha256=PINS[PREDICTION,'e004_candidate_inputs.json'],
                source_ranks_sha256=PINS[PREDICTION,'frozen_candidate_evidence.parquet'],
                frozen_rule_sha256=digest(Path(__file__).with_name('rank.py')))
    (out / 'ranking_receipt.json').write_text(json.dumps(seal,indent=2))
    print('Sealed', seal['absolute_sha256'], 'swaps', seal['swaps_by_anchor'], flush=True)


def score_sealed(inputs, out):
    import pandas as pd
    from rank import answers, digest, metric_identity
    from retrieval_pilot_score import reciprocal

    seal = json.loads((out / 'ranking_receipt.json').read_text())
    if digest(out / 'absolute.csv') != seal['absolute_sha256']:
        raise ValueError('Sealed ranking changed')
    ranked = answers(out / 'absolute.csv')
    base = answers(mounted(inputs, PREDICTION, 'submission.csv'))
    old_scores = pd.read_csv(mounted(inputs, SCORED, 'e180_pair_ranks.csv'), usecols=['molecule_id','control'],
                             dtype={'molecule_id': str, 'control': 'Int64'})
    previous = json.loads(mounted(inputs, SCORED, 'e180_pair_summary.json').read_text())
    if (set(old_scores.molecule_id) != set(base) or not old_scores.molecule_id.is_unique
            or previous['runs']['baseline']['submission_sha256'] != seal['baseline_sha256']):
        raise ValueError('Archived rank population or original ranking differs')
    rows = []
    for mid, old in old_scores.itertuples(index=False):
        before = 0 if pd.isna(old) else int(old)
        after = 0
        if before:
            if not 1 <= before <= 25:
                raise ValueError(f'Invalid historical rank for {mid}')
            # The archived Enveda-180 evaluator matched tautomer-normalized connectivity alone.
            # Re-evaluate duplicate aliases against that same retrieved identity after the permutation.
            truth_key = metric_identity(base[mid][before-1])[0]
            def first_hit(values):
                return next(i for i,s in enumerate(values,1) if metric_identity(s)[0] == truth_key)
            if first_hit(base[mid]) != before:
                raise ValueError(f'Original scorer disagrees with transferred identity for {mid}')
            after = first_hit(ranked[mid])
        rows.append(dict(molecule_id=mid, baseline_rank=before,absolute_rank=after,
                         delta_rr=reciprocal(after)-reciprocal(before)))
    def counts(field):
        vals = [r[field] for r in rows]
        return dict(top1=vals.count(1), top5=sum(0 < v <= 5 for v in vals), top25=sum(v>0 for v in vals),
                    mrr25=sum(reciprocal(v) for v in vals)/len(vals))
    old, new = counts('baseline_rank'), counts('absolute_rank')
    if (old['top1'] != previous['top1']['control'] or old['top5'] != previous['top5']['control']
            or old['top25'] != previous['hits25']['control'] or abs(old['mrr25'] - previous['mrr25']['control']) > 1e-12):
        raise ValueError('Historical control is inconsistent')
    metrics = dict(molecules=len(rows), baseline=old, absolute=new, delta_mrr25=new['mrr25']-old['mrr25'],
                   scored_rank_sha256=PINS[SCORED,'e180_pair_ranks.csv'], ranking_sha256=seal['absolute_sha256'],
                   rank_matching='Historical Enveda-180 RDKit 2026.03.3 tautomer-normalized InChIKey14')
    for k in (1,5):
        metrics[f'new_top{k}'] = [r['molecule_id'] for r in rows if 0 < r['absolute_rank'] <= k and not 0 < r['baseline_rank'] <= k]
        metrics[f'lost_top{k}'] = [r['molecule_id'] for r in rows if 0 < r['baseline_rank'] <= k and not 0 < r['absolute_rank'] <= k]
    pd.DataFrame(rows).to_csv(out / 'per_query.csv',index=False)
    (out / 'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
    print(json.dumps(metrics),flush=True)


def build(source, out, *, study=None, slug=SLUG, kernel_sources=None):
    from build import HERE, KAGGLE_IMAGE, METRIC_BOOTSTRAP, native_fragment_source, pack

    modules = {
        'frag_rescore.py':native_fragment_source(source), 'rank.py':(HERE / 'rank.py').read_bytes(),
        'retrieval_pilot_score.py':(HERE.parent / 'four_arm/retrieval_pilot_score.py').read_bytes(),
        'e180_confirmation.py':Path(__file__).read_bytes(),
    }
    entry = 'e180_confirmation'
    if study is not None:
        entry = 'confirmation_study'
        modules[entry+'.py'] = study.read_bytes()
    blobs = {name:pack(raw) for name,raw in modules.items()}
    setup = f"""
import base64, hashlib, subprocess, sys, zlib
from pathlib import Path
inputs, out, scratch = Path('/kaggle/input'), Path('/kaggle/working'), Path('/tmp/e180-absolute')
scratch.mkdir(exist_ok=True)
for name, (encoded, expected) in {blobs!r}.items():
    raw = zlib.decompress(base64.b64decode(encoded))
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f'Source changed: {{name}}')
    (scratch / name).write_bytes(raw)
{METRIC_BOOTSTRAP}
sys.path.insert(0, str(scratch))
from {entry} import rank_blind, score_sealed
"""
    cells = []
    for cid,code in [('sources',setup),('blind-ranking','rank_blind(inputs,out)'),('archived-ranks','score_sealed(inputs,out)')]:
        compile(code,cid,'exec')
        cells.append(dict(id=cid,cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True)))
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),cells=cells)
    if kernel_sources is None:
        kernel_sources=[f'jiyujia/{PREDICTION}/2',f'jiyujia/{BLIND}/1',f'jiyujia/{SCORED}/1']
    meta=dict(id='jiyujia/'+slug,title=slug,code_file='confirmation.ipynb',language='python',kernel_type='notebook',
              is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=False,dataset_sources=['metric/rdkit-2026-3-3-wheel'],
              kernel_sources=kernel_sources,competition_sources=[],
              docker_image=KAGGLE_IMAGE)
    out.mkdir(parents=True,exist_ok=True)
    (out / 'confirmation.ipynb').write_text(json.dumps(notebook,separators=(',',':'))+'\n')
    (out / 'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    build(**vars(parser.parse_args()))

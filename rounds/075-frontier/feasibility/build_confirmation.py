"""Build the private 0.505/CASMI-2017 confirmation of the locked absolute-coverage rule."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from build import HERE, KAGGLE_IMAGE, METRIC_BOOTSTRAP, native_fragment_source, pack


SLUG = 'enveda-075-absolute-confirm-casmi17'
ARCHIVE17 = 'enveda-075-np-four-arm-0505'
ARCHIVE60 = 'enveda-075-np-recall-reseal'
QUERY17 = '55fe6cb0344d10ee9a564e0bb3fcfea1c7cabedce4fa62bb82604795a947928f'
QUERY60 = '2a8bf747f50e6920fcd4f24e04302b721281f8794c947cc58fe545264b12d0d4'
SOURCE_LISTS = '7e30c2d7566156c15d4e5d0d855563ee662e374cac9975548c5d8d0f44457d60'
BASE17 = '684d5174e7933e9ac1450fca42fcf43ae922a7e1dbaa1deeb558eb9e758fdd33'
BASE60_ARCHIVE = '05509b32cfb9bcbe8ab563244f91c97e89f878eba3cc40263e3f48fd4a1f372d'
ABSOLUTE60 = '71f5df6faa8337150d6cffdcbed980bd938b685b4b93aae5a41c4d1fe9b79af0'
RANKS17 = '6f464fed14f4b0a45162d0bf0db5bb2a566b28eb88c0821546157b271679ae16'
SCORES17 = '4f6ede2b52622a8366e5f379ee49b30529af0dee6946226c24dfccd8fcc37127'


def pinned(path, expected):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f'Expected the pinned archive file {path.name}')
    return raw


def build(source, control_queries, queries, source_lists, out):
    blobs = {name: pack(raw) for name, raw in {
        'control.parquet': pinned(control_queries, QUERY60), 'queries.parquet': pinned(queries, QUERY17),
        'sources.json': pinned(source_lists, SOURCE_LISTS),
        'frag_rescore.py': native_fragment_source(source), 'rank.py': (HERE / 'rank.py').read_bytes(),
        'compare.py': (HERE / 'compare.py').read_bytes(),
        'retrieval_pilot_score.py': (HERE.parent / 'four_arm/retrieval_pilot_score.py').read_bytes(),
    }.items()}
    setup = f"""
import base64, hashlib, json, subprocess, sys, zlib
from pathlib import Path
from zipfile import ZipFile
inputs = Path('/kaggle/input')
scratch = Path('/tmp/absolute-confirmation')
scratch.mkdir(exist_ok=True)
out = Path('/kaggle/working')
def verify(path, digest):
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f'Unexpected source hash: {{path}}')
    return path
def mounted(slug, relative):
    suffix = Path(relative)
    path, = (p for p in inputs.rglob(suffix.name) if slug in p.parts and p.parts[-len(suffix.parts):] == suffix.parts)
    return path
for name, (payload, expected) in {blobs!r}.items():
    path = scratch / name
    path.write_bytes(zlib.decompress(base64.b64decode(payload)))
    verify(path, expected)
{METRIC_BOOTSTRAP}
sys.path.insert(0, str(scratch))
import pandas as pd
from rank import answers, digest, metric_identity, read_scans, rank_absolute
from rdkit import rdBase
assert rdBase.rdkitVersion == '2026.03.3'
def write_rank(lists, path):
    pd.DataFrame([(mid, ';'.join(values)) for mid, values in lists.items()],
                 columns=['molecule_id','smiles']).to_csv(path, index=False)
def same_candidates(original, ranked):
    if set(original) != set(ranked):
        raise ValueError('Query IDs changed')
    for mid, values in original.items():
        old = [metric_identity(s) for s in values]
        new = [metric_identity(s) for s in ranked[mid]]
        if (None in old or len(set(old)) != len(old) or sorted(values) != sorted(ranked[mid])
                or len(set(new)) != len(old) or set(new) != set(old)):
            raise ValueError(f'Candidate population changed: {{mid}}')
"""
    ranking = f"""
with ZipFile(verify(mounted({ARCHIVE60!r}, 'baseline.zip'), {BASE60_ARCHIVE!r})) as archive:
    (scratch / 'control.csv').write_bytes(archive.read('submission.csv'))
    control = answers(scratch / 'control.csv')
    control_sources = json.loads(archive.read('four_source_lists.json'))
    control_rank, _ = rank_absolute(control, control_sources, read_scans(scratch / 'control.parquet'))
    same_candidates(control, control_rank)
    write_rank(control_rank, scratch / 'control-absolute.csv')
    verify(scratch / 'control-absolute.csv', {ABSOLUTE60!r})

baseline_path = verify(mounted({ARCHIVE17!r}, 'four_arm_baseline.csv'), {BASE17!r})
selector = json.loads(mounted({ARCHIVE17!r}, 'four_arm_baseline_receipt.json').read_text())
if (selector['baseline_sha256'] != {BASE17!r} or selector['blind_test_sha256'] != {QUERY17!r}
        or (selector['molecules'],selector['slots']) != (17,425)
        or selector['stage_status']['npx'] != 'applied' or selector['stage_status']['e180x'] != 'applied'):
    raise ValueError('Incomplete or mismatched 0.505 reference')
base17 = answers(baseline_path)
scans = read_scans(scratch / 'queries.parquet')
deposited = pd.read_parquet(scratch / 'queries.parquet',columns=['molecule_id','spectrum_id'])
if len(deposited)!=19 or set(deposited.molecule_id)!=set(base17) or not deposited.spectrum_id.is_unique:
    raise ValueError('Independent experimental scans and baseline disagree')
ranks, swaps = rank_absolute(base17, json.loads((scratch / 'sources.json').read_text()), scans)
same_candidates(base17, ranks)
write_rank(ranks, out / 'absolute.csv')
seal = dict(panel='CASMI2017', original_pipeline='goodpjw2008 v7 0.505', molecules=len(ranks),
            candidate_slots=sum(map(len, ranks.values())), deposited_spectra=len(deposited),
            eligible_spectra=len(scans), eligible_molecules=len({{s.mid for s in scans.values()}}),
            baseline_sha256={BASE17!r}, query_sha256={QUERY17!r}, source_lists_sha256={SOURCE_LISTS!r},
            absolute_source_sha256=digest(scratch / 'rank.py'), absolute_sha256=digest(out / 'absolute.csv'),
            control60_absolute_sha256={ABSOLUTE60!r}, swaps=swaps)
(out / 'ranking_receipt.json').write_text(json.dumps(seal,indent=2))
print('Sealed independent rankings', seal['absolute_sha256'], flush=True)
"""
    scoring = f"""
# The source ranking and inference-only hashes have already been written. Read only archived rank and grouping columns.
from compare import compare
from retrieval_pilot_score import reciprocal
sealed = json.loads((out / 'ranking_receipt.json').read_text())
predictions = answers(verify(out / 'absolute.csv', sealed['absolute_sha256']))
index = json.loads(mounted({ARCHIVE17!r}, 'four_arm/rank/ranking.json').read_text())
previous = json.loads(mounted({ARCHIVE17!r}, 'four_arm/score/metrics.json').read_text())
if (index['baseline_sha256'] != {BASE17!r} or index['queries_sha256'] != {QUERY17!r}
        or index['ranks_sha256'] != {RANKS17!r} or previous['ranks_sha256'] != {RANKS17!r}):
    raise ValueError('Historical ranks do not refer to the frozen CASMI-2017 rankings')
frame = pd.read_parquet(verify(mounted({ARCHIVE17!r}, 'four_arm/rank/ranks.parquet'), {RANKS17!r}),
                        columns=['molecule_id','arm','rank','original_rank','smiles','score_key','heavy_signature'])
original = frame[frame.arm == 'original'].sort_values(['molecule_id','rank'])
prior = pd.read_csv(verify(mounted({ARCHIVE17!r}, 'four_arm/score/per_query.csv'), {SCORES17!r}),
                    usecols=['molecule_id','source','scaffold','original'], keep_default_na=False)
if len(prior) != 17 or not prior.molecule_id.is_unique or set(prior.molecule_id) != set(base17):
    raise ValueError('Historical 17 assessment rows or IDs differ')
groups = {{mid: group for mid, group in original.groupby('molecule_id',sort=False)}}
if set(groups) != set(base17):
    raise ValueError('Incomplete archived candidate population')
rows = []
for row in prior.sort_values('molecule_id').itertuples(index=False):
    mid = row.molecule_id
    stored = groups[mid]
    old_list = base17[mid]
    if (stored['rank'].tolist() != list(range(1,26)) or stored.original_rank.tolist() != list(range(1,26))
            or stored.smiles.tolist() != old_list or stored.duplicated(['score_key','heavy_signature']).any()):
        raise ValueError(f'Archived original candidate identities or order differ for {{mid}}')
    old = int(row.original)
    if not (0 <= old <= len(old_list)):
        raise ValueError('Invalid archived rank')
    new = predictions[mid].index(old_list[old-1]) + 1 if old else 0
    rows.append(dict(molecule_id=mid, source=row.source, scaffold=row.scaffold,
                     baseline_rank=old, absolute_rank=new, delta_rr=reciprocal(new)-reciprocal(old)))
metrics = compare(rows,'baseline','absolute')
if abs(metrics['control_mrr25'] - previous['arms']['original']['mrr25']) > 1e-12:
    raise ValueError('Historical baseline metric changed')
metrics.update(original=previous['arms']['original'], top1=sum(r['absolute_rank']==1 for r in rows),
               top5=sum(1 <= r['absolute_rank'] <= 5 for r in rows), top25=sum(r['absolute_rank']>0 for r in rows),
               by_source={{s:compare([r for r in rows if r['source']==s], 'baseline','absolute')
                          for s in sorted(set(r['source'] for r in rows))}},
               original_rank_artifact_sha256={SCORES17!r}, ranking_sha256=sealed['absolute_sha256'],
               scope='Permutation of a frozen 0.505 list, using historical per-query ranks. No challenge labels copied.')
pd.DataFrame(rows).to_csv(out / 'per_query.csv',index=False)
(out / 'metrics.json').write_text(json.dumps(metrics,indent=2, allow_nan=False))
print(json.dumps(metrics,indent=2),flush=True)
"""
    cells = []
    for cell_id, code in [('frozen-inputs',setup),('blind-control-and-confirmation',ranking),('sealed-rank-comparison',scoring)]:
        compile(code,cell_id,'exec')
        cells.append(dict(cell_type='code', id=cell_id, execution_count=None, metadata={}, outputs=[],source=code.splitlines(True)))
    book = dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),cells=cells)
    options = dict(id='jiyujia/'+SLUG,title=SLUG,code_file='confirmation.ipynb',language='python',kernel_type='notebook',
                   is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=False,dataset_sources=['metric/rdkit-2026-3-3-wheel'],
                   kernel_sources=[f'jiyujia/{ARCHIVE17}/2',f'jiyujia/{ARCHIVE60}/2'],competition_sources=[],docker_image=KAGGLE_IMAGE)
    out.mkdir(parents=True,exist_ok=True)
    (out / 'confirmation.ipynb').write_text(json.dumps(book,separators=(',',':'))+'\n')
    (out / 'kernel-metadata.json').write_text(json.dumps(options,indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','control-queries','queries','source-lists','out'):
        p.add_argument('--'+name,required=True,type=Path)
    build(**vars(p.parse_args()))

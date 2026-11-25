"""Build a label-blind C39-only replay and C41 forward-request comparison against original T4 v2."""
import argparse
import ast
import base64
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import zlib
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


HERE = Path(__file__).resolve().parent
CONTROL = 'enveda-075-np-recall-base'
CONTROL_DATASET = 'jiyujia/enveda-075-c39-blind-control'
CONTROL_BLOB = 'c39-control-v1.bin'
CONTROL_SHA = '08c6fd1393e8e49014aeb1d9b7ab35c1f761e2237614879666d5f94ec71272cc'
QUERY_SHA = '2a8bf747f50e6920fcd4f24e04302b721281f8794c947cc58fe545264b12d0d4'
PINS = {
    'four_reference_audit.json': '0a0e4e1961a855aab6f8ffbe870175f3bc6d4936041c3ea1c7b65618144999ba',
    'four_inner_reference.json': 'fccab3c5f2a9bcf662f18e5619e3673bf36d06535515de7420790273fa98ec80',
    'e004_verification.json': '929c0e7d2364dd3c665a03907222ff0916cf784fdb02c37bb069ef59e29862f4',
    'pilot_forward_budget.json': '0fdf62f6f401f2817de3688f1a7880f803b97bf3dc9dc7772acd4a9e6193e7c2',
    'pc_lists.json': 'd7b28154b1bdf132528300bcf8faa1211984c843d5340ac81191472a452f564b',
}
CONTROL_FILES = set(PINS) | {'e004_candidate_inputs.json', 'eng_lists.json', 'four_source_lists.json',
                             'pilot_shadow_scores.json', 'pilot_ice/ice_in.json', 'pilot_gl/gl_in.json',
                             *(f'{stage}_work/{stage}_{name}.json' for stage in ('ice','gl') for name in ('in','out')),
                             *(f'{stage}_work/{stage}_out.json.meta.json' for stage in ('ice','gl'))}


def control_archive(directory):
    doc = json.loads((directory/'manifest.json').read_bytes())
    if doc['source'] != f'jiyujia/{CONTROL}/2' or set(doc['files']) != CONTROL_FILES:
        raise ValueError('Unexpected fragment or input in private archived control')
    buf = BytesIO()
    with ZipFile(buf,'w') as archive:
        for name in sorted(CONTROL_FILES):
            data = (directory/'raw'/name).read_bytes()
            actual = sha256(data).hexdigest()
            if actual != doc['files'][name]['sha256'] or PINS.get(name, actual) != actual:
                raise ValueError(f'Local original T4 archive changed: {name}')
            entry = ZipInfo(name, date_time=(1980,1,1,0,0,0))
            entry.compress_type = ZIP_DEFLATED
            archive.writestr(entry,data,compresslevel=9)
    return buf.getvalue(), {k:v['sha256'] for k,v in doc['files'].items()}


def stage_control_dataset(path, raw, manifest):
    path.mkdir(parents=True, exist_ok=True)
    expected = {'dataset-metadata.json',CONTROL_BLOB,'c39_manifest.json'}
    if any(f.name not in expected or not f.is_file() for f in path.iterdir()):
        raise ValueError(f'Unexpected file in dedicated Kaggle Dataset staging directory: {path}')
    (path/CONTROL_BLOB).write_bytes(raw)
    receipt = dict(source=f'jiyujia/{CONTROL}/2', archive_sha256=sha256(raw).hexdigest(),
                   files=dict(sorted(manifest.items())))
    (path/'c39_manifest.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    meta = dict(id=CONTROL_DATASET,title='Enveda 075 C39 blind control',
                subtitle='Frozen inputs for the 60-query C39 validation replay',
                licenses=[dict(name='other')],
                description=('Private validation cache from the original T4 run of jiyujia/enveda-075-np-recall-base '
                             'version 2. The binary contains the 17 source files listed in the checksum manifest. '
                             'This cache retains the terms of its underlying source materials.'))
    (path/'dataset-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    return sha256((path/'c39_manifest.json').read_bytes()).hexdigest()


def extract(launched):
    meta = json.loads((launched / 'kernel-metadata.json').read_bytes())
    raw = (launched / meta['code_file']).read_bytes()
    if meta['id'] != 'jiyujia/' + CONTROL or sha256(raw).hexdigest() != CONTROL_SHA:
        raise ValueError('Expected the launched baseline notebook from T4 v2')
    cells = {c.get('id'): ''.join(c['source']) for c in json.loads(raw)['cells']}
    bundle = ast.parse(cells['cell-10']).body
    frozen = {n.targets[0].id: ast.literal_eval(n.value) for n in bundle
              if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') in ('FROZEN_SHA256','FROZEN_B64')}
    inner = zlib.decompress(base64.b64decode(frozen['FROZEN_B64']))
    if sha256(inner).hexdigest() != frozen['FROZEN_SHA256']:
        raise ValueError('Original inner notebook failed verification')
    original = [''.join(c['source']) for c in json.loads(inner)['cells']]
    outer = cells['cell-02']
    prep = outer[outer.index('import os, sys, subprocess, importlib.metadata, time'):outer.index('COMP = _four_root')]
    return original, prep, meta


SETUP = r"""
from pathlib import Path
from functools import lru_cache
import hashlib, json, os, sys, time
ROOT = Path('/kaggle/working')
ROOT.mkdir(exist_ok=True)
working = Path('/tmp/c39-snapshot-work')
working.mkdir(exist_ok=True)
os.chdir(working)
def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
@lru_cache(None)
def archived(name):
    path = control_root / name
    if name not in PINS or not path.is_file() or file_sha(path) != PINS[name]:
        raise ValueError(f'Unverified archived source: {name}')
    return path
def artifact(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, default=str, indent=2)+'\n')
"""


RECORD = """
import pyarrow as pa
import pyarrow.parquet as pq
raw_schema = pa.schema([('molecule_id',pa.string()), ('raw_rank',pa.int32()),
                        ('native_unique_rank',pa.int32()), ('first_representative',pa.bool_()),
                        ('pool_id',pa.int64()), ('smiles',pa.string()), ('native_key',pa.string()),
                        ('formula',pa.string()), ('raw_ranker_score',pa.float64()),
                        ('neutral_mass',pa.float64()), ('molecule_target',pa.float64())])
raw_path = ROOT / 'c39_full_slate.parquet'
raw_writer = pq.ParquetWriter(raw_path, raw_schema, compression='zstd')
row_counts = {}
def record_raw_slate(mid, candidates, score, order, target):
    identity_rank = {}
    entries = []
    for position, i in enumerate(order, 1):
        key = str(candidates.key[i])
        first = key not in identity_rank
        identity_rank.setdefault(key, len(identity_rank)+1)
        entries.append(dict(molecule_id=str(mid), raw_rank=position, native_unique_rank=identity_rank[key],
                            first_representative=first, pool_id=int(candidates.pid[i]),
                            smiles=str(candidates.smiles[i]), native_key=key, formula=str(candidates.formula[i]),
                            raw_ranker_score=float(score[i]), neutral_mass=float(candidates.mass[i]),
                            molecule_target=float(target)))
    raw_writer.write_table(pa.Table.from_pylist(entries, schema=raw_schema))
    row_counts[str(mid)] = len(entries)
"""


VERIFY = """
raw_writer.close()
verified = json.loads(archived('e004_verification.json').read_bytes())
receipt_path = archived('e004_candidate_inputs.json')
if (verified['hashes']['e004_candidate_inputs.json'] != file_sha(receipt_path)
        or verified['processed_molecules'] != 60 or verified['base_errors'] != 0
        or not verified['is_full_run'] or set(verified['processed_ids']) != set(BASE)):
    raise ValueError('Wrong original T4 stage or archived population')
reference = json.loads(receipt_path.read_bytes())
archived_ranks = reference['base_tail']
if set(archived_ranks) != set(BASE) or set(te.molecule_id) != set(BASE) or len(BASE) != 60:
    raise ValueError('Molecule IDs differ from the original archive')
if n_err:
    raise RuntimeError(f'C39 failed to produce {n_err} original candidate lists')
positions, slate = {}, {}
for mid, (smiles, keys, libmax, scores, formulas) in BASE.items():
    old = archived_ranks[mid]
    deep = PILOT_E1_DEEP.get(mid, [])
    if not (len(smiles)==len(keys)==len(scores)==len(formulas) and len(smiles)<=60
            and len(smiles)+len(deep)<=100):
        raise ValueError(f'Invalid native C39 channels for {mid}')
    slate[mid] = dict(library_max=float(libmax), main=[
        dict(smiles=s, native_key=k, source_score=float(v), native_formula=f)
        for s,k,v,f in zip(smiles,keys,scores,formulas)],
        shadow_pool=[dict(smiles=s,native_key=k,source_score=float(v),native_formula=f) for s,k,v,f in deep])
    positions[mid] = dict(raw_rows=row_counts.get(mid,0), main=len(smiles), deep=len(deep),
                          expected_main=len(old), exact=list(smiles)==old,
                          first_mismatch=next((i+1 for i,(a,b) in enumerate(zip(smiles,old)) if a!=b),
                                              min(len(smiles),len(old))+1 if len(smiles)!=len(old) else None))
artifact(ROOT / 'c39_native_ledgers.json', slate)
base_check = dict(original='jiyujia/enveda-075-np-recall-base/2', notebook_sha256=LAUNCHED_SHA,
                  blind_query_sha256=QUERY_SHA, device=dev, rdkit=__import__('rdkit').__version__,
                  cuda_device=torch.cuda.get_device_name(0) if dev=='cuda' else None,
                  eligible_original_lists=len(positions), exact_original_lists=sum(v['exact'] for v in positions.values()),
                  slate_rows=sum(row_counts.values()), native_ledger_sha256=file_sha(ROOT / 'c39_native_ledgers.json'),
                  raw_slate_sha256=file_sha(raw_path), cases=positions)
artifact(ROOT / 'c39_slate_verification.json', base_check)
print('Original C39 main lists', base_check['exact_original_lists'], '/', len(positions), flush=True)
# Use C34/C32 output from before C41. The post-C41 PubChem order in e004 is only for downstream control checks.
PC = json.loads(archived('pc_lists.json').read_bytes())
ENG = json.loads(archived('eng_lists.json').read_bytes())
if (not set(BASE) <= set(ENG) or any(ENG[mid]['smiles']!=reference['engine2'][mid]['smiles'] for mid in BASE)):
    raise ValueError('The archived Engine 2 input does not match the recorded source')
"""


def build(launched, out, device, archive_cache, dataset_dir=None):
    original, prep, meta = extract(launched)
    archive_bytes, manifest = control_archive(archive_cache)
    archive_sha = sha256(archive_bytes).hexdigest()
    data = f'PINS={dict(sorted(manifest.items()))!r}\nQUERY_SHA={QUERY_SHA!r}\nLAUNCHED_SHA={CONTROL_SHA!r}\n'
    extract_control = f"""
from zipfile import ZipFile
control_zip = ROOT / 'c39_original_control.zip'
if file_sha(control_zip) != {archive_sha!r}:
    raise ValueError('Original T4 input archive does not match the published snapshot')
control_root = working / 'original-control'
with ZipFile(control_zip) as control:
    if len(control.namelist()) != len(PINS) or set(control.namelist()) != set(PINS):
        raise ValueError('Unexpected control archive members')
    for member in sorted(PINS):
        dst = control_root / member
        dst.parent.mkdir(parents=True,exist_ok=True)
        dst.write_bytes(control.read(member))
        archived(member)
"""
    compare_input = """
if (file_sha(_four_root/'test.parquet') != QUERY_SHA or
        _four_train_stats != json.loads(archived('four_reference_audit.json').read_bytes())):
    raise ValueError('Frozen queries or excluded reference spectra differ from T4 v2')
COMP = str(_four_root)
"""
    compare_ext = """
old_extension = json.loads(archived('four_inner_reference.json').read_bytes())
for field in ('added','keys','base_rows','validation_removed','error'):
    if old_extension[field] != _EXTERNAL_REFERENCE_STATE[field]:
        raise ValueError(f'External spectrum library differs from T4 v2: {field}')
"""
    rank = original[39]
    anchor = "                order = np.argsort(-score, kind='stable')\n"
    if rank.count(anchor) != 1:
        raise ValueError('Expected the unique original C39 ranking operation')
    rank = rank.replace(anchor, anchor+'                record_raw_slate(mid, C, score, order, target)\n')
    # Keep the native queries, standardizer, v4 scalar features, generated candidates, and stable sorting.
    expected_device = ("dev=='cuda' and 'T4' in torch.cuda.get_device_name(0)" if device=='t4' else "dev=='cpu'")
    device_check = f"if not ({expected_device}):\n    raise RuntimeError('Expected {device} device')\n"
    request = (HERE / 'c39_verify_requests.source').read_text()
    finish = """
request_check = json.loads((ROOT/'c39_request_verification.json').read_bytes())
all_exact = base_check['exact_original_lists']==60 and request_check['exact']
artifact(ROOT/'c39_snapshot_receipt.json',dict(original=base_check['original'], device=dev,
         cache_reuse_supported=all_exact, status='exact' if all_exact else 'observed_difference',
         elapsed_seconds=round(time.monotonic()-NOTEBOOK_STARTED,2),
         hashes={name:file_sha(ROOT/name) for name in ('c39_slate_verification.json','c39_full_slate.parquet',
         'c39_native_ledgers.json','c39_request_verification.json','c39_shadow_provenance.json')}))
print('C39/C41 snapshot exact:', all_exact, flush=True)
"""
    parts = [data, SETUP, extract_control, prep, compare_input, original[5], original[26], original[36], compare_ext,
             original[37], device_check, RECORD, rank, VERIFY, request, finish]
    run = '\n\n'.join(parts)
    compile(run, 'c39-original-snapshot.py', 'exec')
    packed = base64.b85encode(zlib.compress(run.encode(),9)).decode()
    if dataset_dir is None:
        prepare_control = f"(Path('/kaggle/working')/'c39_original_control.zip').write_bytes(base64.b85decode({base64.b85encode(archive_bytes).decode()!r}))"
    else:
        manifest_sha = stage_control_dataset(dataset_dir, archive_bytes, manifest)
        prepare_control = f"""
from shutil import copyfile
slug = {CONTROL_DATASET.split('/')[-1]!r}
manifest, = (p for p in Path('/kaggle/input').rglob('c39_manifest.json') if slug in p.parts)
if hashlib.sha256(manifest.read_bytes()).hexdigest() != {manifest_sha!r}:
    raise ValueError('Wrong private C39 Dataset manifest')
copyfile(manifest.with_name({CONTROL_BLOB!r}),Path('/kaggle/working/c39_original_control.zip'))
"""
    parent = f"""import base64, hashlib, os, subprocess, sys, tempfile, zlib
from pathlib import Path
Path('/kaggle/working').mkdir(exist_ok=True)
{prepare_control}
raw = zlib.decompress(base64.b85decode({packed!r}))
if hashlib.sha256(raw).hexdigest()!={sha256(run.encode()).hexdigest()!r}:
    raise ValueError('Snapshot program changed')
with tempfile.TemporaryDirectory() as tmp:
    script = Path(tmp)/'c39_snapshot.py'
    script.write_bytes(raw)
    subprocess.run([sys.executable,str(script)], check=True, env=dict(os.environ,PYTHONHASHSEED='0',PYTHONUNBUFFERED='1'))
"""
    compile(parent, 'c39_snapshot.py', 'exec')
    slug = 'enveda-075-c39-snapshot-'+device
    meta.update(id='jiyujia/'+slug, title=slug, code_file='snapshot.py', kernel_type='script',
                is_private=True, enable_internet=False, enable_gpu=device=='t4',
                kernel_sources=meta['kernel_sources'])
    if device=='cpu':
        meta.pop('machine_shape',None)
    if dataset_dir is not None:
        meta['dataset_sources'] = [*meta['dataset_sources'],CONTROL_DATASET]
    out.mkdir(parents=True, exist_ok=True)
    script = out/'snapshot.py'
    script.write_text(parent)
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(dict(slug=slug, script_sha256=sha256(script.read_bytes()).hexdigest(),
                          control_zip_sha256=archive_sha, inner_sha256=sha256(run.encode()).hexdigest(), output=str(out))))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launched',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--archive-cache',required=True,type=Path)
    parser.add_argument('--dataset-dir',type=Path,help='Stage the 17 blind files as a private Dataset and reference it in kernel metadata.')
    parser.add_argument('--device',choices=['cpu','t4'],default='cpu')
    build(**vars(parser.parse_args()))

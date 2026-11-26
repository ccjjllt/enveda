"""Transfer the frozen final-five rule to the archived March 2026 MassBank cohort."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path
from runpy import run_path
import tarfile


ARCHIVE = 'jiyujia/enveda-075b-202603-final-ranking/1'
ORIGINAL = 'jiyujia/enveda-075b-massbank-202603-cpu-replay/3'
NATIVE = 'jiyujia/enveda-075b-massbank-202603-cross-formula/2'
SLUG = 'enveda-075-final-native-top5-mb101'
ARCHIVE_SHA = '4b7284049bc80cc8b9ae94e1ff9d0cb9252acdecc266fb894ae7f61a527762d7'
PINS = {
    'submission.csv':'864cf9f414e4ac2d8f634aaf50c6e76ef018c337739957c9fae56b2cc2f48bc4',
    'frozen_candidate_evidence.parquet':'6f58f76db65a54405b82f9354367ebdaf45f88e24ba44088cf6288ecc6b525d1',
    'execution_receipt.json':'4d5122a7cdc14cbddf33dc227dcf6d7dfd2b270683beb3c3700052fb48d77763',
    'ice_work/ice_in.json':'4ff0bc6b71e746781bc2b138620b8146c621bf8f3ed579b8e558b92b070e9571',
    'gl_work/gl_in.json':'4ff0bc6b71e746781bc2b138620b8146c621bf8f3ed579b8e558b92b070e9571',
    'ice_work/ice_out.json':'b0fbaaeea0f86d72eb9e58ff1b1dbdf194a714349d566902cf1f8f33878428a0',
    'gl_work/gl_out.json':'b8d576d0962a0bbb0b790a270a1dd842fdd981b1f2c3c7a4bdedad9b084c359d',
    'ice_work/ice_out.json.meta.json':'ac0afdadb58345e3d7f496959c01ae99104290d75827e129ff17d712bd76a82e',
    'gl_work/gl_out.json.meta.json':'d119bbc4b488528d8f77f3ba499787763f9ac5295069008bb92606a9dfa68188',
}


def replay(payload, out):
    import pyarrow.parquet as pq
    from rdkit import rdBase
    from frozen_top5 import apply, dump, hash_value, verify

    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError('The frozen final-five rule requires contest RDKit')
    verify(payload.read_bytes(), ARCHIVE_SHA)
    with tarfile.open(payload,'r:gz') as data:
        names = data.getnames()
        if len(names)!=len(set(names)) or set(PINS)-set(names):
            raise ValueError('Duplicate or missing frozen archive member')
        raw = {name:verify(data.extractfile(name).read(), digest) for name,digest in PINS.items()}
    run = json.loads(raw['execution_receipt.json'])
    if not run['is_full_run'] or run['processed_molecules']!=101:
        raise ValueError('Wrong original MassBank final run')
    rows = list(csv.DictReader(StringIO(raw['submission.csv'].decode())))
    ranks = {r['molecule_id']:r['smiles'].split(';') for r in rows}
    lengths = Counter(map(len,ranks.values()))
    if (len(rows)!=101 or len(ranks)!=101 or lengths!={25:98,1:2,2:1}
            or any(len(x)!=len(set(x)) for x in ranks.values())):
        raise ValueError('The original 2,454 slots did not survive')
    ledger = {}
    weak_table = pq.read_table(BytesIO(raw['frozen_candidate_evidence.parquet']),
                               columns=['molecule_id','processed','proxy_gl','library_max']).to_pylist()
    for row in weak_table:
        mid, strength = row['molecule_id'],row['library_max']
        if (not row['processed'] or row['proxy_gl'] is not False or strength is None
                or ledger.setdefault(mid,{'library_max':strength})['library_max']!=strength):
            raise ValueError('Missing or conflicting native E1 library strength, or proxy GL score')
    if ledger.keys()!=ranks.keys():
        raise ValueError('Original final cases and native E1 cases differ')
    request, scores, meta = {}, {}, {}
    budget={'cases':{m:{'predictor_budget':{}} for m in ranks}}
    for model in ('ice','gl'):
        def load(suffix):return json.loads(raw[f'{model}_work/{model}{suffix}'])
        items,score,native_meta = load('_in.json'),load('_out.json'),load('_out.json.meta.json')
        request[model],scores[model] = {r['mid']:r for r in items}, score
        if (len(items)!=len(request[model]) or set(request[model])-ranks.keys()
                or set(score)-request[model].keys() or native_meta['device']!='cuda' or native_meta['status']!='ok'):
            raise ValueError(f'Unexpected original {model} request or score population')
        for mid in ranks:
            item = request[model].get(mid,{})
            reps, scans = item.get('cands',[]),item.get('spectra',[])
            if len(reps)!=len(set(reps)) or set(score.get(mid,{}))-set(reps):
                raise ValueError(f'{model} output contains unrequested or duplicate representatives')
            budget['cases'][mid]['predictor_budget'][model+'_native_candidate_spectra']=len(reps)*len(scans)
        meta[model] = {k:native_meta[k] for k in ('status','device','n_mols','n_cands','n_scored')}
        meta[model]['historical_candidate_spectra'] = sum(
            case['predictor_budget'][model+'_native_candidate_spectra'] for case in budget['cases'].values())
    outputs,cases = apply(ranks,ledger,request,scores,budget)
    out.mkdir(exist_ok=True,parents=True)
    files={arm+'_final.json':dump(out/(arm+'_final.json'),mapping) for arm,mapping in outputs.items()}
    files['final_native_top5_mb101_cases.json']=dump(out/'final_native_top5_mb101_cases.json',cases)
    arms={}
    for arm in ('protected','open'):
        revised=outputs[arm]
        if any(Counter(ranks[m][:5])!=Counter(revised[m][:5]) or ranks[m][5:]!=revised[m][5:]
               or (arm=='protected' and ranks[m][:1]!=revised[m][:1]) for m in ranks):
            raise ValueError('The original ordered tail or top-five membership changed')
        arms[arm]=dict(eligible=[m for m,r in cases.items() if r['arms'][arm]['comparable_pairs']],
                       changed=[m for m in ranks if ranks[m]!=revised[m]],
                       first_changed=[m for m in ranks if ranks[m][:1]!=revised[m][:1]],
                       total_comparable_pairs=sum(r['arms'][arm]['comparable_pairs'] for r in cases.values()),
                       total_adjacent_swaps=sum(len(r['arms'][arm]['changes']) for r in cases.values()))
    seal=dict(status='sealed',arm_names=list(outputs),molecules=len(ranks),candidate_budget=25,
              actual_slots=sum(map(len,ranks.values())),original_lengths=dict(lengths),new_predictor_requests=0,
              baseline=ORIGINAL,native_source=NATIVE,archived_dataset=ARCHIVE,archive_sha256=ARCHIVE_SHA,
              base_submission_sha256=PINS['submission.csv'],source_library_threshold=0.9,
              source_60_rule_sha256='2a3b4ee7c93355eed8835ac3755a3d045601646e4ca80a58dd71b30368fcec8b',
              input_file_sha256=PINS,files=files,arms=arms,native_forward_archives=meta,
              historical_final_membership_sha256=hash_value({m:sorted(v) for m,v in ranks.items()}),
              weak_library_cases=sum(r['library_max']<.9 for r in cases.values()),
              common_nonempty_request_cases=sum(r['same_nonempty_native_spectra'] for r in cases.values()),
              rdkit=rdBase.rdkitVersion)
    dump(out/'final_native_top5_mb101_receipt.json',seal)
    print('MassBank original-final-five comparison',
          {a:{k:len(v) if isinstance(v,list) else v for k,v in d.items()} for a,d in arms.items()},flush=True)


def build(out):
    here=Path(__file__).resolve().parent
    common=run_path(str(here/'build.py'))
    shared=run_path(str(here/'build_final_native_top5_e180.py'))['frozen_module']()
    adapter=Path(__file__).read_text().split('\ndef build(out):',1)[0]
    payload={}
    for name,src in [('frozen_top5.py',shared),('mb101_final.py',adapter)]:
        compile(src,name,'exec')
        payload[name]=common['pack'](src.encode())
    runtime=f'''
import base64
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib
inputs=Path('/kaggle/input')
{common['METRIC_BOOTSTRAP']}
asset,=(p for p in inputs.rglob('final_ranking_payload.bin') if {ARCHIVE.split('/')[1]!r} in p.parts)
with tempfile.TemporaryDirectory(prefix='mb101-final-top5-') as directory:
    temp=Path(directory)
    for name,(blob,h) in {payload!r}.items():
        source=zlib.decompress(base64.b64decode(blob))
        if sha256(source).hexdigest()!=h:raise ValueError(f'Payload changed: {{name}}')
        (temp/name).write_bytes(source)
    sys.path.insert(0,str(temp))
    from mb101_final import replay
    replay(asset,Path('/kaggle/working'))
'''
    compile(runtime,SLUG,'exec')
    meta=dict(id='jiyujia/'+SLUG,title=SLUG,code_file='mb101.py',language='python',kernel_type='script',
              is_private=True,enable_gpu=False,enable_internet=False,docker_image=common['IMAGE'],
              dataset_sources=[ARCHIVE,common['WHEEL']],kernel_sources=[],competition_sources=[])
    out.mkdir(parents=True,exist_ok=True)
    (out/meta['code_file']).write_text(runtime)
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(sha256(runtime.encode()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    build(**vars(parser.parse_args()))

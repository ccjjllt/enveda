"""Build a private visible smoke preflight from the unmasked public Pablo v29 competition entry."""
import argparse
import ast
import base64
import copy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import textwrap
import zlib


HERE = Path(__file__).resolve().parent
PUBLIC_SHA = 'a183882abe0b131581109455e1b704ec6d93f27fb99e7b5c2f07d40063e9c5af'
PUBLIC_METADATA = 'ca306d7710f2b361eb182602368bc42b50afd299bd6061e9deed2ecdfe57a43c'
SLUG = 'enveda-075-old-slots-production-smoke'


def stage_tools():
    path = HERE.parent/'four_arm'/'pilot_stage.py'
    spec = importlib.util.spec_from_file_location('original_t4_retention_patch',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def text(book,idx):
    return ''.join(book['cells'][idx]['source'])


def set_text(book,idx,code):
    compile(code,f'old-slots-{idx}','exec')
    book['cells'][idx]['source']=code.splitlines(keepends=True)


def replace_one(s,a,b):
    if s.count(a)!=1:raise ValueError(f'Unknown public author source: {a[:70]}')
    return s.replace(a,b)


def patch_pc(original,helper):
    src=helper.patch_pubchem(original,100)
    worker,save=helper.literal_source(src,'V17_PATCH')
    start=worker.index('    # Export a wider evidence ledger')
    stop=worker.index("    diag['n_listed'] = len(out)",start)
    extra=worker[start:stop]
    worker=worker[:start]+"""    from time import perf_counter as old_slots_counter
    old_slots_start = old_slots_counter()
    pilot_original_keyed = diag.get('n_keyed', 0)
    try:
"""+textwrap.indent(extra,'    ')+"""    except Exception as pilot_error:
        diag.update(pilot_any=[], pilot_outside=[], pilot_error=type(pilot_error).__name__)
    finally:
        diag['n_keyed'] = pilot_original_keyed
        diag['old_slots_collect_seconds'] = old_slots_counter() - old_slots_start
"""+worker[stop:]
    compile(worker,'public-c34-worker','exec')
    src=save(worker)
    runner,save=helper.literal_source(src,'RUNNER')
    needle="            if (k + 1) % 25 == 0 or k + 1 == len(tasks):"
    runner=replace_one(runner,needle,
        "            res[mid]['old_slots_collect_seconds'] = d.get('old_slots_collect_seconds',0.)\n"
        "            if d.get('pilot_error'):\n                res[mid]['old_slots_error'] = d['pilot_error']\n"+needle)
    compile(runner,'public-c34-runner','exec')
    return save(runner)


def patch_e1(original,helper):
    patched=helper.patch_engine1(original,100)
    collect_start=patched.index('                for i in order:\n                    if len(deeper)')
    collect_end=patched.index('        except Exception as e:',collect_start)
    patched=(patched[:collect_start]+'                old_slots_start = time.perf_counter()\n                try:\n'+textwrap.indent(patched[collect_start:collect_end],'    ')+
             """                except Exception as old_slots_err:
                    deeper = []
                    OLD_SLOTS_E1_ERRORS[str(mid)] = type(old_slots_err).__name__
                finally:
                    OLD_SLOTS_E1_COLLECT[str(mid)] = time.perf_counter() - old_slots_start
"""+patched[collect_end:])
    normalize_start=patched.index('    if deeper:')
    normalize_end=patched.index('    if (gi + 1)',normalize_start)
    patched=(patched[:normalize_start]+'    old_slots_start = time.perf_counter()\n    try:\n'+textwrap.indent(patched[normalize_start:normalize_end],'    ')+
             """    except Exception as old_slots_err:
        PILOT_E1_DEEP.pop(mid,None)
        OLD_SLOTS_E1_ERRORS[str(mid)] = type(old_slots_err).__name__
    finally:
        OLD_SLOTS_E1_COLLECT[str(mid)] = OLD_SLOTS_E1_COLLECT.get(str(mid),0.) + time.perf_counter() - old_slots_start
"""+patched[normalize_end:])
    # P5 executes this same embedded cell in another isolated interpreter and deliberately skips the C34 producer.
    # Keep that independent P5 generator on the exact stock code path.
    return ("if 'PC' in globals():\n    OLD_SLOTS_E1_ERRORS, OLD_SLOTS_E1_COLLECT = {}, {}\n"+
            textwrap.indent(patched,'    ')+"\nelse:\n"+textwrap.indent(original,'    ')+'\n')


def rewrite_inner(book,shadow):
    bundle=text(book,10)
    nodes={n.targets[0].id:n for n in ast.parse(bundle).body if isinstance(n,ast.Assign)
           and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
    old_sha=ast.literal_eval(nodes['FROZEN_SHA256'].value)
    original=base64.b64decode(ast.literal_eval(nodes['FROZEN_B64'].value),validate=True)
    if sha256(original).hexdigest()!=old_sha:raise ValueError('Public inner checksum differs')
    inner=json.loads(original)
    helper=stage_tools()
    set_text(inner,34,patch_pc(text(inner,34),helper))
    set_text(inner,39,patch_e1(text(inner,39),helper))
    source=text(inner,41)
    anchor='for m in list(BASE):\n'
    source=replace_one(source,anchor,shadow+"\nold_slots_capture()\n"+anchor)
    set_text(inner,41,source+"""
old_slots_apply()
try:
    from pathlib import Path as _OldSlotsPath
    _OldSlotsPath('/kaggle/working/old_slots_retention_timing.json').write_text(json.dumps(dict(
        c34_sum_of_concurrent_worker_seconds=sum(float((p or {}).get('old_slots_collect_seconds',0.)) for p in PC.values()),
        c39_sequential_seconds=sum(OLD_SLOTS_E1_COLLECT.values())),indent=2))
except (TypeError,ValueError,OSError) as _old_slots_timing_error:
    print('[old slots] retention timing unavailable:', _old_slots_timing_error)
""")
    data=json.dumps(inner,ensure_ascii=False,separators=(',',':')).encode()
    lines=bundle.splitlines(keepends=True)
    values={'FROZEN_SHA256':sha256(data).hexdigest(),'FROZEN_B64':base64.b64encode(zlib.compress(data,9)).decode()}
    for name in sorted(values,key=lambda s:nodes[s].lineno,reverse=True):
        n=nodes[name];lines[n.lineno-1:n.end_lineno]=[f'{name} = {values[name]!r}\n']
    set_text(book,10,''.join(lines))
    loader=text(book,11)
    set_text(book,11,replace_one(loader,'    FROZEN.write_bytes(base64.b64decode(FROZEN_B64))',
                            '    import zlib\n    FROZEN.write_bytes(zlib.decompress(base64.b64decode(FROZEN_B64)))'))
    return old_sha,values['FROZEN_SHA256']


def final_audit():
    return """# Preserve the author's final submission. This cell records the result of the optional C41 intervention.
import hashlib, json, time
from pathlib import Path
import pandas as pd
_osl_root = Path('/kaggle/working')
_osl_report_path = _osl_root / 'old_slots_shadow_audit.json'
if not _osl_report_path.exists():
    raise RuntimeError('Public full pipeline did not reach the old-slot preflight at C41')
_osl_shadow = json.loads(_osl_report_path.read_bytes())
_osl_native = json.loads((_osl_root / 'e004_verification.json').read_bytes())
_osl_sub = _osl_root / 'submission.csv'
_osl_answers = pd.read_csv(_osl_sub, dtype=str, keep_default_na=False)
_osl_sample = pd.read_csv(Path(COMP) / 'sample_submission.csv', usecols=['molecule_id'], dtype=str)
if (list(_osl_answers.columns)!=['molecule_id','smiles'] or not _osl_answers.molecule_id.is_unique
        or set(_osl_answers.molecule_id)!=set(_osl_sample.molecule_id)
        or not _osl_answers.smiles.str.strip().ne('').all()
        or not _osl_answers.smiles.str.split(';').map(len).between(1,25).all()):
    raise ValueError('The public outer runner did not finish with the competition candidate budget')
_osl_summary = dict(author='pablodiegoo/enveda-casmi26-starter-notebook/29', smoke_or_hidden=MODE,
                    original_outer_full_pipeline=bool(PIPELINE_DONE), final_rows=len(_osl_answers),
                    source_query_count=_osl_native['processed_molecules'],
                    intervention_status=_osl_shadow.get('status'), elapsed_session_seconds=round(time.time()-NB_T0,2),
                    limit_seconds=32400, submission_sha256=hashlib.sha256(_osl_sub.read_bytes()).hexdigest(),
                    shadow_audit_sha256=hashlib.sha256(_osl_report_path.read_bytes()).hexdigest())
(_osl_root/'old_slots_production_receipt.json').write_text(json.dumps(_osl_summary,indent=2)+'\\n')
print('Public-entry old-slot preflight:',_osl_summary,flush=True)
"""


def build(source,metadata,out):
    raw=source.read_bytes()
    if sha256(raw).hexdigest()!=PUBLIC_SHA or sha256(metadata.read_bytes()).hexdigest()!=PUBLIC_METADATA:
        raise ValueError('Expected the published Pablo v29 source and metadata of the scored 0.506 reproduction')
    original=json.loads(raw);book=copy.deepcopy(original);meta=json.loads(metadata.read_bytes())
    before,after=rewrite_inner(book,(HERE/'shadow.source').read_text())
    setup=text(book,2)
    setup=replace_one(setup,'NB_T0 = time.time()','NB_T0 = time.time()\nos.environ["OSL_SESSION_STARTED"] = str(NB_T0)')
    set_text(book,2,setup)
    if "COMMIT_MODE = 'smoke'" not in text(book,1):
        raise ValueError('Native visible 12-molecule smoke configuration was changed')
    # Add this diagnostic after all original code cells; no competition ranking is calculated in the receipt cell.
    tail=final_audit()
    compile(tail,'production-smoke-receipt','exec')
    book['cells'].append(dict(id='original-slot-preflight-receipt',cell_type='code',metadata={},outputs=[],execution_count=None,
                              source=tail.splitlines(keepends=True)))
    for c in book['cells']:
        if c['cell_type']=='code':
            c['outputs']=[];c['execution_count']=None
            compile(''.join(c['source']),c.get('id') or 'source','exec')
    datasets=[v for v in meta['datasetDataSources'] if v!='nakhun/casmi26-extlib2']
    if "ENABLE_EXTLIB2 = False" not in text(book,1) or set(meta['kernelDataSources'])!={'alexchilton/casmi26-scorer-train-v80'}:
        raise ValueError('Unexpected public input configuration')
    options=dict(id='jiyujia/'+SLUG,title=SLUG,language='python',kernel_type='notebook',
                 code_file='old_slots.ipynb',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=False,
                 machine_shape='NvidiaTeslaT4',dataset_sources=datasets,kernel_sources=meta['kernelDataSources'],
                 competition_sources=meta['competitionDataSources'],model_sources=[],docker_image=meta['dockerImage'])
    out.mkdir(exist_ok=True,parents=True)
    target=out/options['code_file'];target.write_text(json.dumps(book,ensure_ascii=False,separators=(',',':'))+'\n')
    (out/'kernel-metadata.json').write_text(json.dumps(options,indent=2)+'\n')
    print(json.dumps(dict(id=options['id'],notebook_sha256=sha256(target.read_bytes()).hexdigest(),
                          original_inner=before,patched_inner=after,bytes=target.stat().st_size)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path);p.add_argument('--metadata',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);build(**vars(p.parse_args()))

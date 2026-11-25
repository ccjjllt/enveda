"""Package blind original-slot calibration or the paired T4 delta and native C41/C43 comparison."""
import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import re
from runpy import run_path

from build import HERE, METRIC_BOOTSTRAP, pack
from build_c39_snapshot import extract


IMAGE = run_path(str(HERE.parent/'feasibility/build.py'))['KAGGLE_IMAGE']
CONTROL_SHA = '7aa006825a0f8654c4239f25e89ddb490ea04575b9a195d6b744955688d5d7c5'
INVENTORY_SHA = '1b3797bb9465ef41f782a3e585bcf3d44ec7f7c925bfffc3d874973530887598'


def function(code, name):
    node, = (n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ast.get_source_segment(code,node)


def build(launched,inventory,snapshot,allocation,out,delta=False):
    if (not re.fullmatch(r'jiyujia/enveda-075-c39-snapshot-t4/[1-9][0-9]*',snapshot)
            or not re.fullmatch(r'jiyujia/enveda-075-c39-shadow-allocate/[1-9][0-9]*',allocation)):
        raise ValueError('Pin the validated T4 snapshot and 670-slot allocation')
    inner,_,_=extract(launched)
    if sha256(inventory.read_bytes()).hexdigest()!=INVENTORY_SHA:
        raise ValueError('Expected the original T4 mounted asset inventory')
    gate=inner[43].split("print('merge stats', stats)",1)[0]
    if not gate.endswith("    rows.append((mid, ';'.join(final[:25])))\n"):
        raise ValueError('Could not extract exactly the native C43 promotion and slot loop')
    setup_code='import glob, os, sys\nfrom pathlib import Path\n'+function(inner[26],'simulator_rdkit_wheel')+'\n'+function(inner[41],'_claw_preseed_site')
    blobs={n:pack(b) for n,b in {
        **{n:(HERE/n).read_bytes() for n in ('c41_native_panel.py','c41_delta_predict.py','c39_request_cache.py')},
        'original_c43.py':gate.encode(),'native_site.py':setup_code.encode(),
        'asset_inventory.json':inventory.read_bytes(),
    }.items()}
    program=f"""
import base64, hashlib, json, os, subprocess, sys, tempfile, zlib
from pathlib import Path
from zipfile import ZipFile
inputs, out = Path('/kaggle/input'), Path('/kaggle/working')
def mounted(slug, name):
    hit, = (p for p in inputs.rglob(name) if slug in p.parts)
    return hit
{METRIC_BOOTSTRAP}
def study(scratch):
    for name,(encoded,expected) in {blobs!r}.items():
        data = zlib.decompress(base64.b64decode(encoded))
        if hashlib.sha256(data).hexdigest()!=expected: raise ValueError(f'Program changed: {{name}}')
        (scratch/name).write_bytes(data)
    sys.path.insert(0,str(scratch))
    from c41_native_panel import calibrate, paired_main, fallback_slots, read, dump, pinned, checksum
    from c41_delta_predict import run_delta
    from c39_request_cache import plan_requests
    from native_site import _claw_preseed_site
    snapshot = mounted({snapshot.split('/')[1]!r},'c39_snapshot_receipt.json').parent
    allocated = mounted({allocation.split('/')[1]!r},'allocation_receipt.json').parent
    snapshot_check=read(snapshot/'c39_snapshot_receipt.json')
    if not snapshot_check['cache_reuse_supported'] or read(allocated/'allocation_receipt.json')['status']!='request_plan_only':
        dump(out/'c41_study_receipt.json',dict(status='requires_exact_snapshot_and_sealed_allocation',
                                               source_snapshot={snapshot!r},source_allocation={allocation!r}))
        return
    prior = mounted('casmi26-pubchem-popularity-prior','pool_lsid.npy').parent
    native, assignments, cal = calibrate(snapshot,allocated,prior,out)
    archive=scratch/'control'
    with ZipFile(pinned(snapshot/'c39_original_control.zip',{CONTROL_SHA!r})) as bundle:
        bundle.extractall(archive)
    gl_pkg = mounted('casmi26-glacier','gl_runner.py').parent
    ice_pkg = mounted('casmi26-iceberg','ice_runner.py').parent
    gate = (scratch/'original_c43.py').read_text()
    original = read(archive/'four_source_lists.json')
    pc = read(archive/'e004_candidate_inputs.json')['pubchem']
    shadow = read(archive/'pilot_shadow_scores.json')
    old_scores={{m:dict(native=read(archive/f'{{m}}_work/{{m}}_out.json'),shadow=shadow[m]) for m in ('ice','gl')}}
    old = paired_main(native,assignments,old_scores,pc,gl_pkg/'gl_fuse.py',gate,original,out,'original_slots')
    seal=read(allocated/'allocation_receipt.json')
    result=dict(source_snapshot={snapshot!r},source_allocation={allocation!r},original_slots=old,
                original_allocation_plan_sha256=seal['hashes']['requests/plan.json'],allocated=cal['allocated'],
                attempted=cal['applicable'],calibration_fallbacks=cal['allocated']-cal['applicable'],
                native_no_shadow_controls_exact=60,calibration_sha256=checksum(out/'c41_calibration.json'))
    if not cal['same_device_t4']:
        result['status']='requires_native_t4_snapshot'
    else:
        assigned={{mid:{{(stage['main']+stage['shadow_pool'])[pos]['smiles']:value['smiles']
                        for pos,value in assignments[mid].items()}} for mid,stage in native.items()}}
        attempted=out/'attempted_requests'
        requests=plan_requests(archive,assigned,attempted,snapshot=snapshot)
        if requests['changed_shadow_e1_slots']!=cal['applicable']:
            raise ValueError('The recertified forward plan differs from the native scalar ledger')
        result['plan_sha256']=checksum(attempted/'plan.json')
        result['status']='calibrated_waiting_for_delta'
        if {delta!r}:
            predicted=out/'forward_delta'
            report=run_delta(attempted,archive,predicted,ice_pkg=ice_pkg,gl_pkg=gl_pkg,
                             preseed_site=_claw_preseed_site,site=scratch/'simulator_site',
                             expected_plan_sha=result['plan_sha256'],asset_inventory=scratch/'asset_inventory.json',
                             execute=True)
            if not report.get('all_deltas_observed'):
                result['status']='issued_forward_request_or_runtime_unverified'
            else:
                scored={{m:read(pinned(predicted/f'{{m}}_treatment_scores.json',
                             report['files'][f'{{m}}_treatment_scores.json'])) for m in ('ice','gl')}}
                allocated_rows=read(pinned(allocated/'allocations.json',seal['hashes']['allocations.json']))
                measured=read(pinned(predicted/'delta_observed_by_case.json',report['files']['delta_observed_by_case.json']))
                effective,active,ledger,actual=fallback_slots(native,assignments,allocated_rows,old_scores,scored,measured)
                result['effective']=sum(map(len,active.values()))
                result['prediction_fallbacks']=cal['applicable']-result['effective']
                result['equal_finite_counts']=report['equal_finite_counts']
                result['unequal_finite_case_channels']=report['unequal_finite_case_channels']
                dump(out/'c41_slot_fallbacks.json',ledger)
                dump(out/'c41_forward_observed.json',actual)
                result['slot_fallbacks_sha256']=checksum(out/'c41_slot_fallbacks.json')
                result['effective_observed_sha256']=checksum(out/'c41_forward_observed.json')
                result['replacement']=paired_main(native,active,effective,pc,gl_pkg/'gl_fuse.py',gate,original,out,'replacement')
                result['status']='paired_c43_sealed'
                result['delta_receipt_sha256']=checksum(predicted/'delta_receipt.json')
    dump(out/'c41_study_receipt.json',result)
    print(result,flush=True)
with tempfile.TemporaryDirectory(prefix='c41-native-') as temporary:
    study(Path(temporary))
"""
    compile(program,'c41-native-study','exec')
    slug='enveda-075-c41-shadow-'+('paired' if delta else 'control')
    cell=dict(id='native-c41',cell_type='code',metadata={},outputs=[],execution_count=None,source=program.splitlines(True))
    book=dict(nbformat=4,nbformat_minor=5,cells=[cell],metadata={'kernelspec':dict(name='python3',language='python',display_name='Python 3')})
    meta=dict(id='jiyujia/'+slug,title=slug,language='python',kernel_type='notebook',code_file='native.ipynb',
              is_private=True,enable_gpu=delta,enable_internet=False,kernel_sources=[snapshot,allocation],
              dataset_sources=['dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets',
                               'dmitriigluzdov/casmi26-pubchem-popularity-prior','metric/rdkit-2026-3-3-wheel'],docker_image=IMAGE)
    if delta:meta['machine_shape']='NvidiaTeslaT4'
    out.mkdir(parents=True,exist_ok=True)
    (out/'native.ipynb').write_text(json.dumps(book,separators=(',',':'))+'\n')
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('launched','inventory','out'):p.add_argument('--'+n,required=True,type=Path)
    for n in ('snapshot','allocation'):p.add_argument('--'+n,required=True)
    p.add_argument('--delta',action='store_true')
    build(**vars(p.parse_args()))

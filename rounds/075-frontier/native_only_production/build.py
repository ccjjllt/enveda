"""Package the public Pablo v29 native-forward reranking preflight for Kaggle T4."""

import argparse
import ast
import base64
from hashlib import sha256
import json
from pathlib import Path
from runpy import run_path
import textwrap
import zlib


HERE = Path(__file__).resolve().parent
SLUG = 'enveda-075-native-only-production-smoke'
PUBLIC_SHA = 'a183882abe0b131581109455e1b704ec6d93f27fb99e7b5c2f07d40063e9c5af'
PUBLIC_METADATA = 'ca306d7710f2b361eb182602368bc42b50afd299bd6061e9deed2ecdfe57a43c'


def text(book, index):
    return ''.join(book['cells'][index]['source'])


def write(book, index, source):
    compile(source, f'native-only-preflight-{index}', 'exec')
    book['cells'][index]['source'] = source.splitlines(keepends=True)


def replace_once(source, anchor, replacement):
    if source.count(anchor) != 1:
        raise ValueError(f'Unknown public source at {anchor[:80]}')
    return source.replace(anchor, replacement)


def patch_inner(book):
    source = text(book, 10)
    nodes = {n.targets[0].id:n for n in ast.parse(source).body if isinstance(n,ast.Assign)
             and len(n.targets)==1 and getattr(n.targets[0],'id',None) in {'FROZEN_B64','FROZEN_SHA256'}}
    raw = base64.b64decode(ast.literal_eval(nodes['FROZEN_B64'].value), validate=True)
    original = sha256(raw).hexdigest()
    if original != ast.literal_eval(nodes['FROZEN_SHA256'].value):
        raise ValueError('Pablo v29 inner blob changed')
    inner = json.loads(raw)
    if [inner['cells'][i]['id'] for i in (34,39,41,43)] != ['97cb54ca','9a2b6163','3653cea7','0084a281']:
        raise ValueError('Pablo v29 inner pipeline layout changed')
    stage = run_path(str(HERE.parent/'four_arm'/'pilot_stage.py'))
    c39 = text(inner, 39)
    retained = stage['patch_engine1'](c39,100)
    # The outer P5 tail also executes this C39 cell, in a separate process without C34/PC.
    write(inner,39,"if 'PC' in globals():\n"+textwrap.indent(retained,'    ')+"\nelse:\n"+textwrap.indent(c39,'    ')+'\n')
    c41 = text(inner, 41)
    anchor = '# N1A complement: Seyit Kaan Gunes;'
    c41 = replace_once(c41, anchor, '_NATIVE_ONLY_GL = dict(GL_SCORES)\n' + anchor)
    anchor = 'for m in list(BASE):\n'
    c41 = replace_once(c41, anchor, '_NATIVE_ONLY_STOCK = dict(BASE)\n' + anchor)
    c41 += '\n' + (HERE/'route.source').read_text() + """

_NATIVE_ONLY_PROXIES = set(N1_MIDS) | set(N1_GL_MIDS) | {row['mid'] for row in globals().get('_n1_items',[])}
native_only_apply(_NATIVE_ONLY_STOCK, BASE, globals().get('PILOT_E1_DEEP',{}), ICE_SCORES, _NATIVE_ONLY_GL,
                  globals().get('items',[]), globals().get('gl_items',[]), _NATIVE_ONLY_PROXIES,
                  globals().get('gl_fuse'), LIB_TAU)
"""
    write(inner, 41, c41)
    rewritten = json.dumps(inner, ensure_ascii=False, separators=(',',':')).encode()
    values = {'FROZEN_SHA256':sha256(rewritten).hexdigest(),
              'FROZEN_B64':base64.b64encode(zlib.compress(rewritten,9)).decode()}
    split = source.splitlines(keepends=True)
    for name, node in sorted(nodes.items(), key=lambda item: item[1].lineno, reverse=True):
        split[node.lineno-1:node.end_lineno] = [f'{name} = {values[name]!r}\n']
    write(book, 10, ''.join(split))
    loader = text(book, 11)
    write(book, 11, replace_once(loader,'    FROZEN.write_bytes(base64.b64decode(FROZEN_B64))',
                                '    import zlib\n    FROZEN.write_bytes(zlib.decompress(base64.b64decode(FROZEN_B64)))'))
    return original, values['FROZEN_SHA256']


FINAL_AUDIT = """# Verify the visible format or final hidden run and retain the native-only provenance.
from hashlib import sha256
import json
from pathlib import Path
import time
import pandas as pd

native_root = Path('/kaggle/working')
native_output = native_root/'submission.csv'
native_audit = native_root/'native_only_c41_audit.json'
native_audit_row = json.loads(native_audit.read_bytes())
native_e004 = json.loads((native_root/'e004_verification.json').read_bytes())
native_answers = pd.read_csv(native_output,dtype=str,keep_default_na=False)
native_sample = pd.read_csv(Path(COMP)/'sample_submission.csv',usecols=['molecule_id'],dtype=str)
if (not PIPELINE_DONE or list(native_answers.columns) != ['molecule_id','smiles']
        or not native_answers.molecule_id.is_unique or set(native_answers.molecule_id) != set(native_sample.molecule_id)
        or not native_answers.smiles.str.strip().ne('').all() or not native_answers.smiles.str.split(';').map(len).between(1,25).all()
        or native_audit_row['added_model_requests'] != 0 or native_audit_row['queries'] != native_e004['processed_molecules']):
    raise ValueError('The native-only preflight did not preserve the public competition contract')
native_receipt = dict(author='pablodiegoo/enveda-casmi26-starter-notebook/29', mode=MODE,
                      final_rows=len(native_answers), source_query_count=native_e004['processed_molecules'],
                      elapsed_session_seconds=round(time.time()-NB_T0,2),
                      submission_sha256=sha256(native_output.read_bytes()).hexdigest(),
                      native_audit_sha256=sha256(native_audit.read_bytes()).hexdigest(),
                      intervention_status=native_audit_row['status'],
                      new_forward_model_requests=native_audit_row['added_model_requests'])
(native_root/'native_only_production_receipt.json').write_text(json.dumps(native_receipt,indent=2)+'\\n')
print('Public native-only preflight:',native_receipt,flush=True)
"""


def build(source, metadata, out):
    raw, meta_raw = source.read_bytes(), metadata.read_bytes()
    if sha256(raw).hexdigest()!=PUBLIC_SHA or sha256(meta_raw).hexdigest()!=PUBLIC_METADATA:
        raise ValueError('Expected the exact public Pablo v29 production source and metadata')
    book, meta = json.loads(raw), json.loads(meta_raw)
    setup = text(book,1)
    if "COMMIT_MODE = 'smoke'" not in setup or 'ENABLE_EXTLIB2 = False' not in setup:
        raise ValueError('Unexpected public preflight configuration')
    if set(meta['kernelDataSources']) != {'alexchilton/casmi26-scorer-train-v80'}:
        raise ValueError('Unexpected public kernel source')
    before, after = patch_inner(book)
    book['cells'].append(dict(id='native-only-production-receipt',cell_type='code',metadata={},outputs=[],execution_count=None,
                              source=FINAL_AUDIT.splitlines(keepends=True)))
    for cell in book['cells']:
        if cell['cell_type']=='code':
            compile(''.join(cell['source']),cell.get('id','public-c41-source'),'exec')
            cell.update(outputs=[],execution_count=None)
    options = dict(id='jiyujia/'+SLUG, title=SLUG, language='python', kernel_type='notebook', code_file='native_only.ipynb',
                   is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=False, machine_shape='NvidiaTeslaT4',
                   dataset_sources=[v for v in meta['datasetDataSources'] if v!='nakhun/casmi26-extlib2'],
                   kernel_sources=meta['kernelDataSources'], competition_sources=meta['competitionDataSources'],
                   model_sources=[],docker_image=meta['dockerImage'])
    out.mkdir(parents=True,exist_ok=True)
    target = out/options['code_file']
    target.write_text(json.dumps(book,ensure_ascii=False,separators=(',',':'))+'\n')
    (out/'kernel-metadata.json').write_text(json.dumps(options,indent=2)+'\n')
    print(json.dumps(dict(id=options['id'],notebook_sha256=sha256(target.read_bytes()).hexdigest(),
                          original_inner=before,patched_inner=after,bytes=target.stat().st_size)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    build(**vars(parser.parse_args()))

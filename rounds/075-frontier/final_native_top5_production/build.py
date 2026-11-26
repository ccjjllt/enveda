"""Package the frozen final-top-five rule on the public Pablo v29 competition entry."""

import argparse
import ast
import base64
from hashlib import sha256
import json
from pathlib import Path
import zlib


HERE = Path(__file__).resolve().parent
SLUG = 'enveda-075-final-native-top5-production-smoke'
PUBLIC_SHA = 'a183882abe0b131581109455e1b704ec6d93f27fb99e7b5c2f07d40063e9c5af'
PUBLIC_METADATA = 'ca306d7710f2b361eb182602368bc42b50afd299bd6061e9deed2ecdfe57a43c'
CAPTURE = '''

# Retain provenance for the optional post-fusion use of these exact stock scores.
from hashlib import sha256 as _top5_sha256
from pathlib import Path as _Top5Path
try:
    _top5_root = _Top5Path('/kaggle/working')
    _top5_files = {}
    for _top5_model in ('ice','gl'):
        for _top5_suffix in ('_in.json', '_out.json', '_out.json.meta.json'):
            _top5_name = f'{_top5_model}_work/{_top5_model}{_top5_suffix}'
            _top5_file = _top5_root/_top5_name
            if _top5_file.is_file():
                _top5_files[_top5_name] = _top5_sha256(_top5_file.read_bytes()).hexdigest()
    _top5_library = {str(m):float(v[2]) for m,v in BASE.items() if np.isfinite(v[2])}
    (_top5_root/'final_top5_c41_provenance.json').write_text(json.dumps(
        dict(library=_top5_library,file_sha256=_top5_files), separators=(',',':'), allow_nan=False)+'\\n')
except (OSError, ValueError, TypeError) as _top5_problem:
    print('Native top-five provenance unavailable:',type(_top5_problem).__name__)
'''
PRELUDE = '''from collections import Counter, defaultdict
import csv
from functools import lru_cache
from hashlib import sha256
from io import StringIO
import json
import math
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
'''


def text(book, index):
    return ''.join(book['cells'][index]['source'])


def write(book, index, source):
    compile(source, f'final-native-top5-{index}', 'exec')
    book['cells'][index]['source'] = source.splitlines(keepends=True)


def replace_once(source, anchor, replacement):
    if source.count(anchor) != 1:
        raise ValueError(f'Unknown public source at {anchor[:80]}')
    return source.replace(anchor, replacement)


def patch_inner(book):
    original = text(book,10)
    assignments = {node.targets[0].id:node for node in ast.parse(original).body
                   if isinstance(node,ast.Assign) and len(node.targets)==1
                   and getattr(node.targets[0],'id',None) in ('FROZEN_B64','FROZEN_SHA256')}
    raw = base64.b64decode(ast.literal_eval(assignments['FROZEN_B64'].value), validate=True)
    old_sha = sha256(raw).hexdigest()
    if old_sha!=ast.literal_eval(assignments['FROZEN_SHA256'].value):
        raise ValueError('Public inner notebook digest differs')
    body = json.loads(raw)
    if body['cells'][41]['id']!='3653cea7':
        raise ValueError('Expected the authored C41 prediction cell')
    write(body,41,text(body,41)+CAPTURE)
    revised = json.dumps(body,ensure_ascii=False,separators=(',',':')).encode()
    values = {'FROZEN_SHA256':sha256(revised).hexdigest(),
              'FROZEN_B64':base64.b64encode(zlib.compress(revised,9)).decode()}
    lines = original.splitlines(keepends=True)
    for name,node in sorted(assignments.items(),key=lambda item:item[1].lineno,reverse=True):
        lines[node.lineno-1:node.end_lineno] = [f'{name} = {values[name]!r}\n']
    write(book,10,''.join(lines))
    loader = text(book,11)
    write(book,11,replace_once(loader,'    FROZEN.write_bytes(base64.b64decode(FROZEN_B64))',
                                                    '    import zlib\n    FROZEN.write_bytes(zlib.decompress(base64.b64decode(FROZEN_B64)))'))
    return old_sha,values['FROZEN_SHA256']


def rule_source():
    source = (HERE.parent/'virtual_np'/'final_native_top5.py').read_text()
    lines = source.splitlines(keepends=True)
    needed = ('hash_value','formula','native_evidence','permutation')
    definitions = {}
    for node in ast.parse(source).body:
        if isinstance(node,ast.FunctionDef) and node.name in needed:
            start = min([node.lineno,*(item.lineno for item in node.decorator_list)])
            definitions[node.name] = ''.join(lines[start-1:node.end_lineno])
    return PRELUDE+'\n\n'+'\n\n'.join(definitions[name] for name in needed)+'\n\n'+(HERE/'route.source').read_text()


def build(source, metadata, out):
    raw, meta_raw = source.read_bytes(), metadata.read_bytes()
    if sha256(raw).hexdigest()!=PUBLIC_SHA or sha256(meta_raw).hexdigest()!=PUBLIC_METADATA:
        raise ValueError('Expected the exact public Pablo v29 base and metadata')
    book, meta = json.loads(raw), json.loads(meta_raw)
    setup = text(book,1)
    if "COMMIT_MODE = 'smoke'" not in setup or 'ENABLE_EXTLIB2 = False' not in setup:
        raise ValueError('Unexpected public commit configuration')
    if set(meta['kernelDataSources']) != {'alexchilton/casmi26-scorer-train-v80'}:
        raise ValueError('Unexpected public prediction asset')
    before, after = patch_inner(book)
    tail = rule_source()
    book['cells'].append(dict(id='final-native-top5-rerank',cell_type='code',metadata={},outputs=[],execution_count=None,
                              source=tail.splitlines(keepends=True)))
    for cell in book['cells']:
        if cell['cell_type']=='code':
            compile(''.join(cell['source']),cell.get('id','original-public-cell') or 'original-public-cell','exec')
            cell.update(outputs=[],execution_count=None)
    options = dict(id='jiyujia/'+SLUG,title=SLUG,language='python',kernel_type='notebook',code_file='final_native_top5.ipynb',
                   is_private=True,enable_gpu=True,enable_internet=False,machine_shape='NvidiaTeslaT4',
                   dataset_sources=[v for v in meta['datasetDataSources'] if v!='nakhun/casmi26-extlib2'],
                   kernel_sources=meta['kernelDataSources'],competition_sources=meta['competitionDataSources'],
                   model_sources=[],docker_image=meta['dockerImage'])
    out.mkdir(parents=True,exist_ok=True)
    target = out/options['code_file']
    target.write_text(json.dumps(book,ensure_ascii=False,separators=(',',':'))+'\n')
    (out/'kernel-metadata.json').write_text(json.dumps(options,indent=2)+'\n')
    print(json.dumps(dict(id=options['id'],notebook_sha256=sha256(target.read_bytes()).hexdigest(),
                          original_inner=before,patched_inner=after,bytes=target.stat().st_size)))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for option in ('source','metadata','out'):
        cli.add_argument('--'+option,type=Path,required=True)
    build(**vars(cli.parse_args()))

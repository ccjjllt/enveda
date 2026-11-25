"""Patch a copy of the launched 60-case baseline notebook for an archived C41-to-final replay."""
import ast
import base64
import copy
from hashlib import sha256
import json
import zlib


NPX_MODEL = '8c636d0d71ff72db7971eb6aa1a11be57b7797f85104b7b2e7f883d92f73dc64'
NPX_FIT = '716aa01e108e54162711548289e47f4f1c05b61154c611e505f8e547b6928c9e'
PINS = {
    'eng_lists.json':'f103a6db194f89ef6842335096c4e94970098914214ca3105f854176ae325035',
    'pc_lists.json':'d7b28154b1bdf132528300bcf8faa1211984c843d5340ac81191472a452f564b',
    'four_source_lists.json':'61c27cd182e26767f94a378c0170f1702ee8c84febe091fed571cead99645136',
    'pilot_forward_budget.json':'0fdf62f6f401f2817de3688f1a7880f803b97bf3dc9dc7772acd4a9e6193e7c2',
    'pilot_shadow_scores.json':'f0e0384f1428d571411bdc9f3875a272d31409ac56c4f022b1b4d5dc794e1bdc',
    'gl_work/gl_in.json':'44f9dbe123ec0e1242a9899460eca60dce69777037b97e37c1a8de583337395b',
    'gl_work/gl_out.json':'cc66cd418bf33292227b430d02ee6fe2f13600926ec970baeb78ce4a729c3f01',
    'gl_work/gl_out.json.meta.json':'7e07078db19ae8c1f3806acafdc39b540348eee2812a97ac53a126076012c485',
    'ice_work/ice_in.json':'3fbfd6d7c3bf45693525a27e520c52051f49d111388452ff43a795934585ac89',
    'ice_work/ice_out.json':'e483847fbba85e69ce93c0265afec6c2815c2a3a8db8442e1322ae4f036c56f2',
    'ice_work/ice_out.json.meta.json':'d80fdac0bf4e9287e8eea5f9e21c850096b0078dfa4a5be5bb4afd807bd48441',
}


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Unexpected original source for {old[:80]!r}')
    return source.replace(old,new)


def text(cell):
    return ''.join(cell['source'])


def write(cell, code):
    compile(code,cell.get('id') or 'patched-notebook-cell','exec')
    cell['source'] = code.splitlines(keepends=True)


def cell_by_id(book, name):
    found, = (c for c in book['cells'] if c.get('id')==name)
    return found


def ast_splice(source, node, replacement):
    lines = source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+replacement.rstrip()+'\n'+''.join(lines[node.end_lineno:])


def copy_branch(source, variable, filename):
    marker = f'{variable} = {{}}\n'
    if source.count(marker) != 1 or 'subprocess.run(' not in source[source.index(marker):]:
        raise ValueError(f'Unrecognized {variable} producer')
    # The original module and runner source files emitted above the subprocess are used by later stages.
    return source[:source.index(marker)] + f"""
from pathlib import Path as _RoutePath
import hashlib as _route_hashlib, json as _route_json
_route_blob = (_RoutePath('/tmp/c41_original_control') / {filename!r}).read_bytes()
if _route_hashlib.sha256(_route_blob).hexdigest() != {PINS[filename]!r}:
    raise ValueError('Wrong original T4 {filename}')
(_RoutePath('/kaggle/working') / {filename!r}).write_bytes(_route_blob)
{variable} = _route_json.loads(_route_blob)
"""


def routed_c41(source):
    prep = f"""
from pathlib import Path as _RoutePath
import hashlib as _route_hashlib, json as _route_json, os as _route_os
_route_root = _RoutePath('/tmp/c41_original_control')
_route_work = _RoutePath('/kaggle/working')
_route_pins = {PINS!r}
def _route_read(name):
    raw = (_route_root / name).read_bytes()
    if name in _route_pins and _route_hashlib.sha256(raw).hexdigest() != _route_pins[name]:
        raise ValueError('Wrong archived baseline file: '+name)
    return _route_json.loads(raw)
def _route_digest(value):
    return _route_hashlib.sha256(_route_json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
_route_budget = _route_read('pilot_forward_budget.json')
_route_shadow = _route_read('pilot_shadow_scores.json')
_route_os.environ['PILOT_SHADOW_SCORE_CACHE'] = str(_route_root/'pilot_shadow_scores.json')
_route_restored = set()
def _route_native(model, items):
    prefix = model+'_work/'+model
    expected = _route_read(prefix+'_in.json')
    if _route_digest(items) != _route_digest(expected):
        raise ValueError('Rebuilt native request differs from the original T4: '+model)
    by_id = {{str(item['mid']):item for item in items}}
    if len(by_id)!=len(items) or not set(by_id)<=set(_route_budget['cases']):
        raise ValueError('Native request query population differs: '+model)
    channel = model+'_native'
    meta = _route_read(prefix+'_out.json.meta.json')
    if meta != _route_budget['native_meta'][model] or meta['status']!='ok' or not meta['device'].startswith('cuda'):
        raise ValueError('Original native model metadata and budget disagree: '+model)
    scores = _route_read(prefix+'_out.json')
    finite_key = 'gl_native_and_proxy' if model=='gl' else 'ice_native'
    for mid,case in _route_budget['cases'].items():
        item = by_id.get(mid,{{}})
        n = len(item.get('cands',[]))*len(item.get('spectra',[]))
        if (_route_digest(item) != case['request_sha256'][channel]
                or n!=case['predictor_budget'][channel+'_candidate_spectra']
                or sum(v is not None and np.isfinite(v) for v in scores.get(mid,{{}}).values())!=case['finite_counts'][finite_key]):
            raise ValueError('Original predictor inputs or finite outputs disagree for '+model+'/'+mid)
    (_route_work/(model+'_work')).mkdir(exist_ok=True)
    for suffix in ('_in.json','_out.json','_out.json.meta.json'):
        rel = prefix+suffix
        (_route_work/rel).write_bytes((_route_root/rel).read_bytes())
    _route_restored.add(model)
    return scores
"""
    source = prep + source
    for model,name,method,arg in (('ice','ICE_SCORES','run_ice','items'),('gl','GL_SCORES','run_gl','gl_items')):
        node, = (n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Assign) and len(n.targets)==1
                 and getattr(n.targets[0],'id','')==name and isinstance(n.value,ast.Call)
                 and getattr(n.value.func,'attr','')==method)
        source = ast_splice(source,node,' '*node.col_offset+f"{name} = _route_native({model!r}, {arg})")
    node, = (n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='pilot_shadow_score')
    source = ast_splice(source,node,"""
def pilot_shadow_score(ice_items,gl_items):
    request_sha = pilot_digest(dict(ice=ice_items,gl=gl_items))
    if request_sha!=_route_shadow['request_sha256'] or request_sha!=_route_budget['shadow_input_sha256']:
        raise ValueError('Rebuilt shadow request differs from the original T4')
    return _route_shadow['ice'],_route_shadow['gl'],_route_shadow['meta'],request_sha
""")
    anchor = '    _n1_run = _n1_items + _n1_x_items\n'
    source = replace_once(source,anchor,anchor+"""    if _n1_run:
        raise RuntimeError('Unexpected proxy predictor request on the original no-proxy panel')
""")
    return source + """
if _route_restored!={'ice','gl'} or globals().get('_n1_run') or N1_DRV.get('status')!='nothing to score':
    raise ValueError('Native cache replay did not complete on the expected original no-proxy panel')
if _route_json.loads((_route_work/'pilot_forward_budget.json').read_bytes())['cases'] != _route_budget['cases']:
    raise ValueError('Native or shadow per-query budget, success count or branch ordering differs from baseline')
_route_original = _route_read('four_source_lists.json')
_route_path = _RoutePath('/tmp/c41_route/main.json')
_route_raw = _route_path.read_bytes()
_route_main = _route_json.loads(_route_raw)
if set(BASE)!=set(_route_original) or set(BASE)!=set(_route_main) or len(BASE)!=60:
    raise ValueError('Intervention and historical main lists disagree on queries')
_route_changed = []
for _mid,(_smiles,_keys,_lib) in BASE.items():
    if list(_smiles)!=_route_original[_mid]['main']:
        raise ValueError('Original T4 main branch does not replay before intervention: '+_mid)
    r_sm,r_keys,r_lib = _route_main[_mid]
    if (len(r_sm)!=len(_smiles) or len(r_sm)!=len(r_keys) or len(r_sm)>25
            or not all(isinstance(s,str) and s for s in r_sm+r_keys) or len(set(r_keys))!=len(r_keys)
            or not np.isfinite(r_lib) or r_lib!=_lib):
        raise ValueError('Post-C41 main route differs in budget, native keys or reference strength: '+_mid)
    if r_sm!=list(_smiles) or r_keys!=list(_keys):
        _route_changed.append(_mid)
    BASE[_mid] = (r_sm,r_keys,r_lib)
(_route_work/'c41_route_replay.json').write_text(_route_json.dumps(dict(
    source='jiyujia/enveda-075-np-recall-base/2',control_main_sha256=_route_pins['four_source_lists.json'],
    treatment_main_sha256=_route_hashlib.sha256(_route_raw).hexdigest(), native_requests_replayed=True,
    original_shadow_request_sha256=_route_budget['shadow_input_sha256'],queries=len(BASE),changed_queries=_route_changed),
    indent=2))
"""


def patch(launched_book:dict)->dict:
    result = copy.deepcopy(launched_book)
    bundle = cell_by_id(result,'cell-10')
    source = text(bundle)
    attrs = {n.targets[0].id:n for n in ast.parse(source).body if isinstance(n,ast.Assign)
             and len(n.targets)==1 and getattr(n.targets[0],'id','') in ('FROZEN_B64','FROZEN_SHA256')}
    raw = zlib.decompress(base64.b64decode(ast.literal_eval(attrs['FROZEN_B64'].value),validate=True))
    if sha256(raw).hexdigest()!=ast.literal_eval(attrs['FROZEN_SHA256'].value):
        raise ValueError('The launched notebook does not contain the recorded inner program')
    inner = json.loads(raw)
    for index,name,variable,filename in ((32,'8116570d','ENG','eng_lists.json'),(34,'97cb54ca','PC','pc_lists.json')):
        c=inner['cells'][index]
        if c['id']!=name:
            raise ValueError(f'Unexpected original inner cell {index}')
        write(c,copy_branch(text(c),variable,filename))
    c=inner['cells'][41]
    if c['id']!='3653cea7' or "PILOT_PARAMETERS = {'arm': 'baseline'," not in text(c):
        raise ValueError('Expected the authentic baseline C41 arm')
    write(c,routed_c41(text(c)))
    new = json.dumps(inner,separators=(',',':'),ensure_ascii=False).encode()
    updates = dict(FROZEN_B64=base64.b64encode(zlib.compress(new,9)).decode(),FROZEN_SHA256=sha256(new).hexdigest())
    for name,node in sorted(attrs.items(),key=lambda entry:entry[1].lineno,reverse=True):
        source = ast_splice(source,node,f'{name} = {updates[name]!r}')
    write(bundle,source)

    npx = cell_by_id(result,'3448b0b5')
    code = text(npx)
    node, = (n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='npx_train')
    replacement = f"""
def npx_train(*args,**kwargs):
    import hashlib,json,pickle
    from pathlib import Path
    path = Path('/tmp/c41_route')
    fit_file,model_file = path/'shared_npx_fit.json',path/'shared_npx_model.pickle'
    raw_fit,raw_model = fit_file.read_bytes(),model_file.read_bytes()
    if hashlib.sha256(raw_model).hexdigest()!={NPX_MODEL!r} or hashlib.sha256(raw_fit).hexdigest()!={NPX_FIT!r}:
        raise ValueError('Shared original NPX model or fit receipt changed')
    fit = json.loads(raw_fit)
    shared = pickle.loads(raw_model)
    if (shared['fit']!=fit or not fit['gate_ok']
            or shared['reference_counts']!=json.loads(Path('/kaggle/working/four_reference_audit.json').read_bytes())):
        raise ValueError('NPX mask or feature fit differs from the frozen T4 tail')
    return shared['model'],npx_featurizer(),shared['table'],shared['fit']
"""
    write(npx,ast_splice(code,node,replacement))
    final=cell_by_id(result,'seal-recall-arm')
    write(final,"""assert 'Tesla T4' in GPU and IS_RERUN and PIPELINE_DONE and NEURAL_OK
(ROOT/'pilot_provenance_audit.json').write_text(json.dumps(PROVENANCE_REPORT,indent=2))
if not (ROOT/'c41_route_replay.json').is_file():
    raise RuntimeError('The frozen C41 route was not executed')
""")
    return result

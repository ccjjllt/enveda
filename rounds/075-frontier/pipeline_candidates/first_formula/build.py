"""Build an original, within-coarse, or expanded PubChem arm from goodpjw v3."""
import argparse
import ast
import base64
from hashlib import sha256
import json
from pathlib import Path
import zlib


HERE = Path(__file__).resolve().parent
ORIGINAL_CODE_SHA256 = '9ad27be9f5661e91d1329131e71021512b46420a0de4e831f2076e0cfa36bcbd'
BUNDLE_CELL = 15
DRIVER_CELL = 16


def assignment(source, target):
    return next(node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                and any(ast.unparse(name) == target for name in node.targets))


def rewrite(source, original, updated):
    if source.count(original) != 1:
        raise ValueError(f'Expected one frozen source block starting {original.splitlines()[0]!r}')
    return source.replace(original, updated, 1)


def set_assignment(source, name, value):
    node = assignment(source, name)
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = [f'{name} = {value!r}\n']
    return ''.join(lines)


def set_cell(book, index, source):
    compile(source, f'cell_{index}', 'exec')
    book['cells'][index]['source'] = source.splitlines(keepends=True)


def cell(book, index):
    return ''.join(book['cells'][index]['source'])


def amend_c34(source):
    patch = ast.literal_eval(assignment(source, 'V17_PATCH').value)
    original_tail = "    diag['secs'] = time.time() - t0\n    return mid, out, out_s, out_k, diag\n"
    saved = ('    _pc_first_snapshot(mid, target, z, smis, ok, top1, idx, fz, sc, out, out_s, out_k)\n'
             + original_tail)
    begin, end = patch.index("    use_pop = ('ls' in _W)"), patch.index('    if use_pop and _POP_UNION > 0:')
    document = '\n\ndef _pc_first_document_prior(window_priors, n):\n' + patch[begin:end] + '    return pop\n'
    retrieval = (HERE / 'retrieval.source').read_text()
    patch = rewrite(patch, original_tail, saved) + document + retrieval
    core = ast.literal_eval(assignment(source, 'CORE').value)
    compile(core + patch, 'probe_core2.py', 'exec')
    source = set_assignment(source, 'V17_PATCH', patch)
    worker = (HERE / 'worker.source').read_text()
    compile(worker, 'pc_first_worker.py', 'exec')
    original = "Path('/kaggle/working/pc_lists.json').unlink(missing_ok=True)\n"
    source = rewrite(source, original, original + f"""_pc34_snapshot_env = {{}}
try:
    _pc34_snapshot = Path('/tmp/pc_first_formula_snapshot')
    _pc34_snapshot.mkdir(parents=True, exist_ok=True)
    Path('/kaggle/working/pc_first_worker.py').write_text({worker!r})
    _pc34_snapshot_env['CASMI_PC_FIRST_SNAPSHOT'] = str(_pc34_snapshot)
except Exception as _pc34_error:
    print('PC FIRST snapshot setup unavailable:', repr(_pc34_error), flush=True)
""")
    original = "capture_output=True, text=True, timeout=4 * 3600, env={**os.environ, 'PYTHONHASHSEED': '0'})"
    updated = "capture_output=True, text=True, timeout=4 * 3600, env={**os.environ, **_pc34_snapshot_env, 'PYTHONHASHSEED': '0'})"
    source = rewrite(source, original, updated)
    return source


def amend_c41(source):
    helpers = (HERE / 'physics.source').read_text()
    optional = """
_PC_FIRST_KEEP_HEAD = os.environ.get('CASMI_PC_FIRST_KEEP_HEAD', '0') == '1'
_pc_first_audit = dict(status='started', applied=0, incomplete_or_inconsistent=0, retained=0,
                       pc_head_changed=0, applied_queries={})
_pc_first_plans = {}
try:
    _pc_first_plans = _pc_first_plan(_PC_FIRST_KEEP_HEAD, _pc_first_audit)
except Exception as _pc_exception:
    _pc_first_audit.update(status='original_pc_kept', error=repr(_pc_exception)[:300])
    print('PC FIRST supplemental phase unavailable:', repr(_pc_exception), flush=True)

"""
    source = rewrite(source, 'def gl_of(m):', helpers + optional + 'def gl_of(m):')
    stamp = "print('ICE rerank stats', ice_stats, f'{time.time()-T0:.0f}s')"
    return rewrite(source, stamp, (HERE / 'commit.source').read_text() + stamp)


def digest(data):
    return sha256(data).hexdigest()


def add_first_formula(book, keep_head, arm):
    bundle = cell(book, BUNDLE_CELL)
    frozen_input = base64.b64decode(ast.literal_eval(assignment(bundle, 'FROZEN_B64').value))
    if digest(frozen_input) != ast.literal_eval(assignment(bundle, 'FROZEN_SHA256').value):
        raise ValueError('Original frozen notebook does not match its pinned source hash')
    inner = json.loads(frozen_input)
    set_cell(inner, 34, amend_c34(cell(inner, 34)))
    set_cell(inner, 41, amend_c41(cell(inner, 41)))
    emitted_input = json.dumps(inner, ensure_ascii=False, separators=(',', ':')).encode()
    bundle = set_assignment(bundle, 'FROZEN_B64', base64.b64encode(zlib.compress(emitted_input, 9)).decode())
    bundle = rewrite(bundle, 'FROZEN_B64: the public 0.465 pipeline notebook, byte for byte.',
                     'FROZEN_B64: compressed source of the pinned pipeline notebook.')
    bundle = rewrite(bundle, ' / 1e6:.2f} MB)', ' / 1e6:.2f} MB compressed)')
    bundle = set_assignment(bundle, 'FROZEN_SHA256', digest(emitted_input))
    label = 'preserve original first PubChem result' if keep_head else 'open first-formula slots'
    runner_name = "MODULES['runner.py']"
    runner = ast.literal_eval(assignment(bundle, runner_name).value)
    revised = rewrite(runner, 'source_adaptations=[]',
                      'source_adaptations=' + repr([f'Equal-budget PubChem first formula: {arm}, {label}']))
    compile(revised, 'runner.py', 'exec')
    bundle = set_assignment(bundle, runner_name, revised)
    bundle = rewrite(bundle, digest(runner.encode()), digest(revised.encode()))
    set_cell(book, BUNDLE_CELL, bundle)

    outer = cell(book, DRIVER_CELL)
    outer = rewrite(outer, 'import contextlib, importlib', 'import contextlib, importlib, zlib')
    outer = rewrite(outer, 'FROZEN.write_bytes(base64.b64decode(FROZEN_B64))',
                    'FROZEN.write_bytes(zlib.decompress(base64.b64decode(FROZEN_B64)))')
    marker = 'def run_stage(args, log, name, quiet=None):'
    outer = rewrite(outer, marker, '# First-formula rescue leader mode.\n'
                    f'PC_FIRST_KEEP_HEAD = {keep_head}\nPC_FIRST_ARM = {arm!r}\n\n' + marker)
    marker = "    env.pop('INC_PANEL', None)\n"
    outer = rewrite(outer, marker, marker + """    if name == 'Public 0.465 pipeline':
        env['CASMI_PC_FIRST_NB_T0'] = str(NB_T0)
        env['CASMI_PC_FIRST_KEEP_HEAD'] = '1' if PC_FIRST_KEEP_HEAD else '0'
        env['CASMI_PC_FIRST_ARM'] = PC_FIRST_ARM
""")
    set_cell(book, DRIVER_CELL, outer)


def add_blind_input(book, arm):
    outer = cell(book, 9)
    start = outer.index('COMP = next((p for p in (INPUT')
    end = outer.index("_ids = pd.read_parquet(COMP", start)
    set_cell(book, 9, rewrite(outer, outer[start:end], (HERE / 'blind_input.source').read_text() + '\n'))

    bundle = cell(book, BUNDLE_CELL)
    frozen = base64.b64decode(ast.literal_eval(assignment(bundle, 'FROZEN_B64').value))
    if digest(frozen) != ast.literal_eval(assignment(bundle, 'FROZEN_SHA256').value):
        raise ValueError('Original inner notebook digest mismatch')
    inner = json.loads(frozen)
    # Both the main runner and the P5 subprocess execute this instrumented inner notebook.
    set_cell(inner, 4, rewrite(cell(inner, 4), 'DATA = locate_data_dir()',
                               "DATA = Path(os.environ['E180_BLIND_ROOT'])"))
    set_cell(inner, 28, rewrite(cell(inner, 28),
                               "assert comp in {f'/kaggle/input/{slug}', f'/kaggle/input/competitions/{slug}'}",
                               "assert comp == os.path.realpath(os.environ['E180_BLIND_ROOT'])"))
    payload = json.dumps(inner, ensure_ascii=False, separators=(',', ':')).encode()
    bundle = set_assignment(bundle, 'FROZEN_B64', base64.b64encode(payload).decode())
    set_cell(book, BUNDLE_CELL, set_assignment(bundle, 'FROZEN_SHA256', digest(payload)))
    set_cell(book, 43, cell(book, 43) + f"\n_blind_arm = {arm!r}\n" + """
import torch
_blind_execution = json.loads((ROOT / 'execution_receipt.json').read_text())
if not PIPELINE_DONE or not _blind_execution['is_full_run'] or _blind_execution['processed_molecules'] != 121:
    raise RuntimeError('The complete 121-query pipeline did not finish.')
_blind_final = ROOT / 'submission.csv'
_blind_status = {}
for _name in ('pc_first_formula', 'n035net', 'fr3to25', 'p5tail', 'aprior', 'uniq'):
    _path = ROOT / f'{_name}_audit.json'
    _blind_status[_name] = json.loads(_path.read_text()).get('status') if _path.is_file() else None
(ROOT / 'e180_ranked_input.json').write_text(json.dumps(dict(
    inference=_blind_receipt, submission_sha256=hashlib.sha256(_blind_final.read_bytes()).hexdigest(),
    stage_status=_blind_status, arm=_blind_arm, elapsed_seconds=round(time.time()-NB_T0, 1),
    cuda_devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]), indent=2))
""")


def build(original, target, arm, keep_head, mode, panel_kernel=None, panel_dataset=None):
    if original.resolve() == target.resolve():
        raise ValueError('The output must be separate from the original source.')
    metadata = json.loads((original / 'kernel-metadata.json').read_text())
    book = json.loads((original / metadata['code_file']).read_text())
    cells = [''.join(c['source']) for c in book['cells'] if c.get('cell_type') == 'code']
    if digest(json.dumps(cells, ensure_ascii=False, separators=(',', ':')).encode()) != ORIGINAL_CODE_SHA256:
        raise ValueError('Expected the code cells of the verified goodpjw v3 reproduction.')
    if arm == 'baseline' and keep_head:
        raise ValueError('--keep-head applies to the control and expanded arms.')
    if panel_kernel or panel_dataset:
        add_blind_input(book, arm)
        metadata['competition_sources'] = []
        # The unused catalogue replay ships a second test.parquet for its own panel.
        metadata['dataset_sources'] = [source for source in metadata['dataset_sources']
                                       if source != 'sanpier/catalogue-experiment-saner']
        if panel_kernel:
            metadata['kernel_sources'].append(panel_kernel)
        else:
            metadata['dataset_sources'].append(panel_dataset)
    if arm != 'baseline':
        add_first_formula(book, keep_head, arm)
    set_cell(book, 8, rewrite(cell(book, 8), "COMMIT_MODE = 'fast'", f'COMMIT_MODE = {mode!r}'))
    slug = f'enveda-075-first-formula-{arm}' + ('-keep' if keep_head else '')
    if panel_kernel or panel_dataset:
        slug += '-e180'
    metadata.update(id=f'jiyujia/{slug}', title=slug, is_private=True, enable_internet=False)
    target.mkdir(parents=True, exist_ok=True)
    (target / metadata['code_file']).write_text(json.dumps(book, ensure_ascii=False, indent=1) + '\n')
    (target / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Private package', metadata['id'], mode, target, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Original notebook and kernel-metadata.json directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=('baseline', 'control', 'expanded'), required=True)
    parser.add_argument('--mode', choices=('smoke', 'fast'), default='smoke')
    parser.add_argument('--keep-head', action='store_true')
    panel = parser.add_mutually_exclusive_group()
    panel.add_argument('--panel-kernel', help='Kaggle kernel source with only prepared blind inference files')
    panel.add_argument('--panel-dataset', help='Private Kaggle dataset source with only prepared blind inference files')
    options = parser.parse_args()
    build(options.source, options.output, options.arm, options.keep_head, options.mode,
          options.panel_kernel, options.panel_dataset)

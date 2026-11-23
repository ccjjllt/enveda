"""Build the full-window GPU scorer and a narrowly changed matched 068a fusion run."""
import argparse
import ast
import base64
from hashlib import sha256
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL_SHA = 'e2334fad6d57fcf4c6d6b05e8ef986b41954f4bc14d67f433d5ea6d8a945a346'
OWNER = 'jiyujia'
RETRIEVE = 'enveda-075-negative-full-window'
FUSION = 'enveda-075-negative-window-fusion'
PANEL = f'{OWNER}/enveda-075b-external-massbank-adducts/1'
ORIGINAL = f'{OWNER}/enveda-075b-massbank-adducts-pipeline/1'
TRAINING = f'{OWNER}/enveda-075-cross-polarity-pilot/1'


def change(code, old, new):
    if code.count(old) != 1:
        raise ValueError(f'Expected one frozen patch site {old[:100]}')
    return code.replace(old, new, 1)


def assignment(code, name):
    node, = [node.value for node in ast.parse(code).body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
    return ast.get_source_segment(code, node), ast.literal_eval(node)


def write(destination, metadata, code):
    destination.mkdir(parents=True, exist_ok=True)
    (destination / metadata['code_file']).write_text(code)
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


def retrieval(destination):
    source = (HERE.parent / 'train.py').read_text()
    needed = {'PROTON', 'BINS', 'ROOT', 'OUTPUT', 'find', 'chemistry', 'spectrum_vector', 'Retriever'}
    lines = []
    for node in ast.parse(source).body:
        name = node.targets[0].id if isinstance(node, ast.Assign) else getattr(node, 'name', None)
        if isinstance(node, (ast.Import, ast.ImportFrom)) or name in needed:
            lines.append(ast.get_source_segment(source, node))
    code = '\n\n'.join(lines + [(HERE / 'retrieve.py').read_text()])
    compile(code, 'retrieve.py', 'exec')
    train_meta = json.loads((HERE.parent / 'kernel-metadata.json').read_text())
    train_meta.update(id=f'{OWNER}/{RETRIEVE}', title=RETRIEVE, code_file='retrieve.py',
                      kernel_sources=[PANEL, ORIGINAL, TRAINING], competition_sources=[],
                      dataset_sources=['metric/rdkit-2026-3-3-wheel', 'dmitriigluzdov/casmi26-pubchem-popularity-prior'])
    write(destination, train_meta, code)


def fusion(original, destination, version, retrieval_slug=RETRIEVE, fusion_slug=FUSION, cpu=False):
    metadata = json.loads((original / 'kernel-metadata.json').read_text())
    blob = (original / metadata['code_file']).read_bytes()
    if sha256(blob).hexdigest() != ORIGINAL_SHA:
        raise ValueError('Expected the archived 47-negative-plus-one-context 068a notebook.')
    notebook = json.loads(blob)
    outer = notebook['cells'][5]
    body = ''.join(outer['source'])
    encoded, payload = assignment(body, 'FROZEN_B64')
    inner = json.loads(base64.b64decode(payload))
    pc_cell = ''.join(inner['cells'][34]['source'])
    patch_text, patch = assignment(pc_cell, 'V17_PATCH')
    runner_text, runner = assignment(pc_cell, 'RUNNER')
    replacement = '''    diag['negative_window'] = _insert_negative_window(
        mid, target, smis, pop, fz, z, out, out_s, out_k, out_p, out_sc, promotion_evidence)
    diag['n_listed'] = len(out)'''
    patch = change(patch, "    diag['n_listed'] = len(out)", replacement)
    patch += '\n\n' + (HERE / 'insert.source').read_text()
    runner = change(runner, "promotion_evidence=d.get('promotion_evidence', {}), **extra[mid]",
                    "promotion_evidence=d.get('promotion_evidence', {}), negative_window=d.get('negative_window'), **extra[mid]")
    pc_cell = change(pc_cell, patch_text, repr(patch))
    pc_cell = change(pc_cell, runner_text, repr(runner))
    pc_cell = change(pc_cell, "    rr.check_returncode()",
                     "    if rr.returncode:\n        raise RuntimeError(rr.stderr[-3000:].strip())")
    pc_cell = change(pc_cell, "    print('PUBCHEM CHANNEL FAILED -> base lists only:', repr(e))",
                     "    raise RuntimeError(f'PubChem comparison failed: {e}') from e")
    compile(pc_cell, 'cell34', 'exec')
    compile(patch, 'pc_patch', 'exec')
    compile(runner, 'pc_runner', 'exec')
    inner['cells'][34]['source'] = pc_cell
    if cpu:
        preflight = ''.join(inner['cells'][28]['source'])
        probe = "    gpu = subprocess.check_output(\n        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()"
        inner['cells'][28]['source'] = change(preflight, probe, "    gpu = 'CPU'")
        physical = ''.join(inner['cells'][41]['source'])
        if physical.count("device='cuda'") != 2:
            raise ValueError('Expected both original forward model calls')
        physical = physical.replace("device='cuda'", "device='cpu'")
        inner['cells'][41]['source'] = change(physical, "'/kaggle/working/ice_site', 'cuda', []",
                                              "'/kaggle/working/ice_site', 'cpu', []")
        postflight = ''.join(inner['cells'][50]['source'])
        inner['cells'][50]['source'] = change(postflight, "meta['device'].startswith('cuda')",
                                              "meta['device'].startswith('cpu')")
        neural = ''.join(notebook['cells'][6]['source'])
        if neural.count(", 'cuda'],") != 2:
            raise ValueError('Expected both original outer neural calls')
        notebook['cells'][6]['source'] = neural.replace(", 'cuda'],", ", 'cpu'],")
        for i in (28, 41, 50):
            compile(inner['cells'][i]['source'], f'fusion_cell{i}', 'exec')
    expanded = json.dumps(inner, ensure_ascii=False).encode()
    body = change(body, encoded, repr(base64.b64encode(expanded).decode()))
    old_hash, _ = assignment(body, 'FROZEN_SHA256')
    outer['source'] = change(body, old_hash, repr(sha256(expanded).hexdigest()))
    notebook['cells'][0]['source'] = ''.join(notebook['cells'][0]['source']) + '''
_paths = list(Path('/kaggle/input').rglob('negative_window_proposals.json'))
if len(_paths) != 1:
    raise ValueError('Expected the frozen negative-ion proposals')
os.environ['NEGATIVE_WINDOW_PROPOSALS'] = str(_paths[0])
'''
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), 'fusion', 'exec')
    metadata.update(id=f'{OWNER}/{fusion_slug}', title=fusion_slug, code_file=f'{fusion_slug}.ipynb')
    metadata['kernel_sources'].append(f'{OWNER}/{retrieval_slug}/{version}')
    if cpu:
        metadata['enable_gpu'] = False
        metadata.pop('machine_shape', None)
    write(destination, metadata, json.dumps(notebook, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--original', type=Path)
    parser.add_argument('--retrieval-version', type=int, default=1)
    parser.add_argument('--retrieval-slug', default=RETRIEVE)
    parser.add_argument('--fusion-slug', default=FUSION)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    if args.original:
        fusion(args.original, args.destination, args.retrieval_version, args.retrieval_slug, args.fusion_slug, args.cpu)
    else:
        retrieval(args.destination)

"""Generate the self-contained Kaggle kernel: python build_kernel.py /tmp/enveda-075b-pipeline."""

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASELINE = HERE.parents[1] / '068-n1c-lam035' / '068a'
SLUG = 'enveda-075b-pipeline'
PANEL_SOURCE = 'jiyujia/enveda-075b-queries'


def replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Expected one upstream site for {old[:100]!r}')
    return source.replace(old, new)


def source(notebook, number):
    return ''.join(notebook['cells'][number]['source'])


def update(notebook, number, old, new):
    notebook['cells'][number]['source'] = replace(source(notebook, number), old, new)


def literal_value(text, name):
    assignment, = [node for node in ast.parse(text).body if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == name]
    return ast.literal_eval(assignment.value)


def change_engine2(notebook):
    engine_cell = source(notebook, 32)
    modules = literal_value(engine_cell, 'ENG_FILES')
    modules['casmi_engine.py'] = replace(
        modules['casmi_engine.py'], '    POOL = build_pool(L, workers)',
        '''    # Use spectrum-free columns from the original file so the full set of
    # training structures remains eligible after filtering the reference spectra.
    structure_table = pq.read_table(os.environ['PANEL_ORIGINAL_TRAIN'],
                                    columns=['inchikey14', 'normalized_smiles'])
    POOL = build_pool({'ik': structure_table['inchikey14'].to_pylist(),
                       'smi': structure_table['normalized_smiles'].to_pylist()}, workers)
    del structure_table''',
    )
    runner, = [node.args[0] for node in ast.walk(ast.parse(engine_cell))
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == 'write' and node.args and isinstance(node.args[0], ast.Constant)
               and isinstance(node.args[0].value, str) and node.args[0].value.startswith('"""Our engine')]
    runner_updated = replace(
        runner.value,
        '''        for c in r['cand'][order]:''',
        '''        blend, frag_ad = [], []
        for index in order:
            c = r['cand'][index]''',
    )
    runner_updated = replace(
        runner_updated,
        '''            seen.add(k); smis.append(s); keys.append(k)''',
        '''            seen.add(k); smis.append(s); keys.append(k)
            blend.append(float(bl[mid][index]))
            frag_ad.append(float(r['frag_ad'][index]))''',
    )
    runner_updated = replace(
        runner_updated, 'dict(smiles=smis, keys=keys)',
        'dict(smiles=smis, keys=keys, blend=blend, frag_ad=frag_ad)',
    )
    update(notebook, 32, ast.get_source_segment(engine_cell, runner), repr(runner_updated))
    engine_cell = source(notebook, 32)
    dictionary = next(node.value for node in ast.parse(engine_cell).body if isinstance(node, ast.Assign)
                      and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'ENG_FILES')
    update(notebook, 32, ast.get_source_segment(engine_cell, dictionary), repr(modules))


def update_frozen(notebook, edit_frozen=None):
    outer = source(notebook, 12)
    encoded = literal_value(outer, 'FROZEN_B64')
    frozen = json.loads(base64.b64decode(encoded))
    # Reuse 072d's explicit competition mount for both the outer and child notebooks.
    data_paths = '''DATA = Path(os.environ['INSTRUMENT_COMP'])
TRAIN_PATH = DATA / 'train.parquet'
TEST_PATH = DATA / 'test.parquet'
SAMPLE_PATH = DATA / 'sample_submission.csv'
'''
    frozen['cells'][4]['source'] = data_paths
    update(frozen, 26, "COMP = os.path.dirname(find('test.parquet'))", "COMP = os.environ['INSTRUMENT_COMP']")
    update(frozen, 28, "    assert comp in {f'/kaggle/input/{slug}', f'/kaggle/input/competitions/{slug}'}",
           "    assert comp == os.path.realpath(os.environ['INSTRUMENT_COMP'])")
    change_engine2(frozen)
    update(frozen, 36,
           "_EXTERNAL_REFERENCE_STATE = _build_external_reference_cache('work', TABLE, _EXTERNAL_REFERENCE_PINS)",
           "_EXTERNAL_REFERENCE_STATE = dict(path=None, added=0, keys=0, error='heldout panel')")
    frozen['cells'][39]['source'] = source(frozen, 39) + '''
with open(os.environ['PANEL_ENGINE1_SCORES'], 'w') as handle:
    panel_engine1 = {str(mid): {'smiles': entry[0], 'scs': entry[3]} for mid, entry in BASE.items()}
    panel_engine1['pool_meta_path'] = os.path.join(POOL_DIR, 'pool_meta.parquet')
    json.dump(panel_engine1, handle)
'''
    if edit_frozen is not None:
        edit_frozen(frozen)
    for index, cell in enumerate(frozen['cells']):
        if cell['cell_type'] == 'code':
            compile(source(frozen, index), f'frozen_cell_{index}', 'exec')
    contents = json.dumps(frozen, ensure_ascii=False).encode()
    update(notebook, 12, encoded, base64.b64encode(contents).decode())
    update(notebook, 12, literal_value(outer, 'FROZEN_SHA256'), hashlib.sha256(contents).hexdigest())
    return source(frozen, 2)


def update_outer(notebook):
    update(notebook, 6, '''COMP = next((p for p in (INPUT / 'competitions/enveda-CASMI26-molecule-id-mass-spectra',
                         INPUT / 'enveda-CASMI26-molecule-id-mass-spectra',
                         INPUT / 'enveda-casmi26-molecule-id-mass-spectra')
             if all((p / f).is_file() for f in ('train.parquet', 'test.parquet', 'sample_submission.csv'))), None)''',
           "COMP = Path(os.environ['INSTRUMENT_COMP'])")
    update(notebook, 6, "MODE = 'Scoring rerun' if IS_RERUN else", "MODE = '075b held-out panel' if IS_RERUN else")
    update(notebook, 6, 'NB_T0 = time.time()', 'NB_T0 = PANEL_STARTED')
    update(notebook, 6, "('Run mode', MODE, 'hidden test' if IS_RERUN else 'visible format example')",
           "('Run mode', MODE, '075b panel spectra' if IS_RERUN else 'visible format example')")
    update(notebook, 24,
           "    tp = sorted(glob.glob(inp + '/**/test.parquet', recursive=True), key=lambda p: ('mass-spectra' not in p.lower(), len(p)))",
           "    tp = [os.path.join(os.environ['INSTRUMENT_COMP'], 'test.parquet')]")
    update(notebook, 25,
           "    test_paths = sorted(glob.glob(inp + '/**/test.parquet', recursive=True),\n                       key=lambda p: ('mass-spectra' not in p.lower(), len(p)))",
           "    test_paths = [os.path.join(os.environ['INSTRUMENT_COMP'], 'test.parquet')]")
    update(notebook, 25, '    hits_count = 0', '''    apex_candidates = {}
    eligible_references, excluded_references = set(), set()
    hits_count = 0''')
    update(notebook, 25, '                    cos, n_matched = calc_cosine(', '''                    if ri not in eligible_references:
                        if ri in excluded_references or excluded_reference(ref[5], PANEL_IDENTITIES, PANEL_SCAFFOLDS):
                            excluded_references.add(ri)
                            continue
                        eligible_references.add(ri)
                    cos, n_matched = calc_cosine(''')
    update(notebook, 25, '                hits_count += 1', '''                hits_count += 1
                apex_candidates[mid] = {'smiles': match_smi, 'cosine': best_cos}''')
    update(notebook, 25, "    elapsed = round(time.monotonic() - t0, 1)", '''    with open(os.path.join(work, 'apex_external_candidates.json'), 'w') as handle:
        json.dump(apex_candidates, handle)
    PANEL_AUDIT['apex_external_compared'] = len(eligible_references)
    PANEL_AUDIT['apex_external_removed_from_query_windows'] = len(excluded_references)
    elapsed = round(time.monotonic() - t0, 1)''')
    update(notebook, 25, '    _apex_extlib_overlay()\nexcept Exception:',
           "    _apex_extlib_overlay()\nexcept Exception:\n    PANEL_AUDIT['apex_external_failed'] = True")

    physical = source(notebook, 14)
    post = literal_value(physical, '_post')
    updated = replace(post, '    frag_new,frag_old,energy=data', '''    frag_new,frag_old,energy=data
    with Path('/kaggle/working/physical_candidate_scores.jsonl').open('a') as handle:
        rows = [{'molecule_id': mid, 'smiles': smiles, 'frag_new': float(frag_new[index]),
                 'frag_old': float(frag_old[index]), 'energy_chunks': [
                     float(value) if np.isfinite(value) else None for value in energy[:, index]]}
                for index, smiles in enumerate(cand)]
        for row in rows:
            handle.write(json.dumps(row) + '\\n')''')
    post_literal, = [node.value for node in ast.parse(physical).body if isinstance(node, ast.Assign)
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == '_post']
    update(notebook, 14, ast.get_source_segment(physical, post_literal), repr(updated))
    update(notebook, 14, hashlib.sha256(post.encode()).hexdigest(), hashlib.sha256(updated.encode()).hexdigest())


def code_cell(text):
    return dict(cell_type='code', metadata={}, source=text, outputs=[], execution_count=None)


def build(destination, panel_source=PANEL_SOURCE, slug=SLUG, edit_frozen=None):
    notebook_path, = BASELINE.glob('*.ipynb')
    notebook = json.loads(notebook_path.read_text())
    bootstrap = update_frozen(notebook, edit_frozen)
    update_outer(notebook)
    runtime = (HERE / 'panel_io.py').read_text()
    initialization = (bootstrap + '\nPANEL_STARTED = time.time()\n' + runtime
                      + '\nPANEL_IDS, PANEL_IDENTITIES, PANEL_SCAFFOLDS, PANEL_AUDIT = prepare_panel()\n')
    export = 'export_candidates(ROOT, PANEL_IDS, PANEL_AUDIT)\n'
    notebook['cells'] = ([code_cell(initialization)]
                         + [code_cell(source(notebook, index)) for index in range(5, 26)
                            if notebook['cells'][index]['cell_type'] == 'code']
                         + [code_cell(export)])
    for index, cell in enumerate(notebook['cells']):
        compile(cell['source'], f'notebook_cell_{index}', 'exec')
    metadata = json.loads((BASELINE / 'kernel-metadata.json').read_text())
    metadata.update(id=f'jiyujia/{slug}', title=slug, code_file=f'{slug}.ipynb')
    metadata['kernel_sources'].append(panel_source)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (destination / metadata['code_file']).write_text(json.dumps(notebook, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--panel-source', default=PANEL_SOURCE)
    parser.add_argument('--slug', default=SLUG)
    args = parser.parse_args()
    build(args.destination, args.panel_source, args.slug)

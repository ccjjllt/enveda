"""Build the blinded validation edition of goodpjw2008's complete public 0.505 v7."""
import argparse
import ast
import base64
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import zlib


HERE = Path(__file__).resolve().parent
SOURCE_SHA = 'a7e9b7f609ed56a24e67d90eea23ed522e5fe9ec21da9de4fb04f73e1d45eddb'
OUTPUT_NAME = 'four_arm.ipynb'
sys.path.insert(0, str(HERE.parents[1] / 'pipeline_candidates' / 'first_formula'))
from build import assignment, cell, digest, rewrite, set_assignment, set_cell


def encoded(data):
    return base64.b85encode(zlib.compress(data, 9)).decode()


def prepare_source(blind, holdout, bootstrap):
    groups = json.loads(holdout.read_text())
    hashes = lambda values: sorted({digest(s.encode()) for s in values if s})
    patterns = {re.sub(r'[^a-z0-9]+', '', s.lower()) for s in groups['blocked_sources']}
    mask = dict(keys=hashes(groups['blocked_keys']), scaffolds=hashes(groups['blocked_scaffolds']),
                sources=sorted(patterns - {''}), block_acyclic=bool(groups.get('block_acyclic', False)))
    files = {name: (blind / name).read_bytes() for name in ['test.parquet', 'sample_submission.csv']}
    files['mask.json'] = json.dumps(mask).encode()
    files['four_reference_mask.py'] = (HERE / 'reference_mask.py').read_bytes()
    blobs = {name: (encoded(data), digest(data)) for name, data in files.items()}
    return bootstrap + '\n' + f'_four_blobs = {blobs!r}\n' + """
import json, os, zlib, base64, hashlib
from pathlib import Path
import sys
_four_root = Path('/tmp/four_arm_comp')
_four_root.mkdir(exist_ok=True)
for _name, (_blob, _sha) in _four_blobs.items():
    _bytes = zlib.decompress(base64.b85decode(_blob))
    assert hashlib.sha256(_bytes).hexdigest() == _sha
    (_four_root / _name).write_bytes(_bytes)
_four_raw, = {p.resolve() for p in Path('/kaggle/input').rglob('train.parquet')
              if any(s.lower() == 'enveda-casmi26-molecule-id-mass-spectra' for s in p.parts)}
_four_idx, = {p.resolve() for p in Path('/kaggle/input').rglob('structures.parquet')
              if 'casmi26-identity-disjoint-validation-folds' in p.parts and 'casmi26-natural-product-panel-benchmark' in p.parts}
assert hashlib.sha256(_four_idx.read_bytes()).hexdigest() == '7c16c4968a15956e94eba58e3ff94d4a6c2f6899b10a2f2824047601a46bd6c8'
os.environ['FOUR_ARM_COMP'] = str(_four_root)
os.environ['PANEL_ORIGINAL_TRAIN'] = str(_four_raw)
sys.path.insert(0, str(_four_root))
import four_reference_mask as _four_mask
_four_train_stats = _four_mask.filter_train(_four_raw, _four_idx)
(ROOT / 'four_reference_audit.json').write_text(json.dumps(_four_train_stats, indent=2))
print('Blinded input and group/source-masked reference spectra', _four_train_stats, flush=True)
COMP = _four_root
"""


def route_engine(book, *, bundle_index=12, driver_index=13, patch_inner=None,
                 allow_empty_extension=False):
    bundle = cell(book, bundle_index)
    raw = base64.b64decode(ast.literal_eval(assignment(bundle, 'FROZEN_B64').value))
    if digest(raw) != ast.literal_eval(assignment(bundle, 'FROZEN_SHA256').value):
        raise ValueError('The embedded v7 pipeline digest differs')
    inner = json.loads(raw)
    def edit(i, original, new):
        set_cell(inner, i, rewrite(cell(inner, i), original, new))
    edit(4, 'DATA = locate_data_dir()', "DATA = Path(os.environ['FOUR_ARM_COMP'])")
    edit(26, "COMP = os.path.dirname(find('test.parquet'))", "COMP = os.environ['FOUR_ARM_COMP']")
    edit(28, "assert comp in {f'/kaggle/input/{slug}', f'/kaggle/input/competitions/{slug}'}",
         "assert comp == os.path.realpath(os.environ['FOUR_ARM_COMP'])")
    engine = cell(inner, 32)
    modules = ast.literal_eval(assignment(engine, 'ENG_FILES').value)
    mods = modules['casmi_engine.py']
    modules['casmi_engine.py'] = rewrite(mods, '    POOL = build_pool(L, workers)', """
    structure_table = pq.read_table(os.environ['PANEL_ORIGINAL_TRAIN'],columns=['inchikey14','normalized_smiles'])
    POOL = build_pool({'ik': structure_table['inchikey14'].to_pylist(),
                       'smi': structure_table['normalized_smiles'].to_pylist()}, workers)
    del structure_table""")
    compile(modules['casmi_engine.py'], 'casmi_engine.py', 'exec')
    set_cell(inner, 32, set_assignment(engine, 'ENG_FILES', modules))

    h = "        extra_meta, extra_mz, extra_it, used_keys = [], [], [], set()"
    edit(36, h, h + """
        import sys, json
        sys.path.insert(0, os.environ['FOUR_ARM_COMP'])
        import four_reference_mask as reference_mask
        blocked_keys = set(json.loads((reference_mask.PANEL / 'blocked_scorer.json').read_text()))
        removed = 0
        group_keys_before_mask = set()""")
    h = "                extra_meta.append(d); extra_mz.append(x); extra_it.append(y); used_keys.add(r['scorer_key'])"
    edit(36, h, """                group_keys_before_mask.add(r['scorer_key'])
                if (r['scorer_key'] in blocked_keys or reference_mask.digest(r['scorer_key']) in reference_mask.KEYS
                    or reference_mask.blocked_source(r['record_id']) or reference_mask.blocked_source(r['source'])
                    or reference_mask.copied_spectrum(prec, x, y)):
                    removed += 1
                    continue
""" + h)
    edit(36, "len(extra_meta) == pins['added_rows'] and len(used_keys) == pins['added_keys']",
         "len(extra_meta) + removed == pins['added_rows'] and len(group_keys_before_mask) == pins['added_keys']")
    if allow_empty_extension:
        edit(36, "        check(len(extra_meta) > 0, 'no eligible external references')", """        if not extra_meta:
            return dict(path=None, added=0, keys=0, base_rows=n, validation_removed=removed, error=None)""")
    h = "_EXTERNAL_REFERENCE_STATE = _build_external_reference_cache('work', TABLE, _EXTERNAL_REFERENCE_PINS)"
    edit(36, h, h + "\nassert _EXTERNAL_REFERENCE_STATE['error'] is None, _EXTERNAL_REFERENCE_STATE\n")
    h = "return dict(path=str(target), added=len(extra_meta), keys=len(used_keys), base_rows=n, error=None)"
    edit(36, h, h.replace('base_rows=n', 'base_rows=n, validation_removed=removed'))
    set_cell(inner, 39, cell(inner, 39) + """
if os.path.abspath(os.getcwd()) == '/kaggle/working':
    with open('four_inner_reference.json', 'w') as handle:
        json.dump(_EXTERNAL_REFERENCE_STATE, handle, default=str)
""")
    set_cell(inner, 54, cell(inner, 54) + """
_four_sources = {str(mid):dict(main=list(BASE[mid][0]),
                              prefusion=list(globals().get('PREFUSION_LISTS',{}).get(mid, [])),
                              engine2=list(ENG.get(mid,{}).get('smiles', [])),
                              pubchem=list(PC.get(mid,{}).get('pc', []))) for mid in BASE}
Path('/kaggle/working/four_source_lists.json').write_text(json.dumps(_four_sources))
""")
    if patch_inner is not None:
        patch_inner(inner)
    payload = json.dumps(inner, ensure_ascii=False, separators=(',', ':')).encode()
    bundle = set_assignment(bundle, 'FROZEN_B64', base64.b64encode(zlib.compress(payload, 9)).decode())
    set_cell(book, bundle_index, set_assignment(bundle, 'FROZEN_SHA256', digest(payload)))
    driver = cell(book, driver_index)
    set_cell(book, driver_index, rewrite(driver, 'FROZEN.write_bytes(base64.b64decode(FROZEN_B64))',
                               'import zlib\n    FROZEN.write_bytes(zlib.decompress(base64.b64decode(FROZEN_B64)))'))
    return cell(inner, 2)


def adapt_outer(book, blind, holdout, bootstrap):
    def edit(i, before, after):
        set_cell(book, i, rewrite(cell(book, i), before, after))
    code = cell(book, 6)
    begin = code.index('COMP = next((p for p in (INPUT')
    end = code.index('_ids = pd.read_parquet(COMP', begin)
    edit(6, code[begin:end], prepare_source(blind, holdout, bootstrap))
    for f in ('test', 'train'):
        old = f'Path(inp).rglob("{f}.parquet")'
        edit(25, old, f'[Path("/tmp/four_arm_comp/{f}.parquet")]')
    edit(26, "glob.glob(inp + '/**/test.parquet', recursive=True)", "[str(COMP / 'test.parquet')]")
    edit(26, "                    smi = str(row['normalized_smiles'])",
         "                    if any(_four_mask.blocked_source(row.get(k, '')) for k in ('source_library', 'library', 'spectrum_id')):\n                        continue\n                    smi = str(row['normalized_smiles'])")
    edit(26, "    hits_count = 0", "    blocked_reference_indices = set()\n    checked_reference_indices = set()\n    hits_count = 0")
    site = "                    cos, n_matched = calc_cosine("
    edit(26, site, """                    if ri not in checked_reference_indices:
                        checked_reference_indices.add(ri)
                        if _four_mask.excluded_smiles(ref[5]) or _four_mask.copied_spectrum(ref[1], ref[2], ref[3]):
                            blocked_reference_indices.add(ri)
                    if ri in blocked_reference_indices:
                        continue
""" + site)
    edit(26, "    with open(os.path.join(work, 'apex_extlib_audit.json'), 'w') as f:",
         "    audit.update(heldout_or_copy_window_references=len(blocked_reference_indices), reference_rows=len(refs))\n    with open(os.path.join(work, 'apex_extlib_audit.json'), 'w') as f:")
    for i in (38, 40):
        edit(i, 'files = sorted(Path(inp).rglob(name),', 'files = sorted([Path("/tmp/four_arm_comp") / name],')
    for i in (42, 44):
        edit(i, 'sorted(Path(inp).rglob("train.parquet"),',
             'sorted([Path("/tmp/four_arm_comp/train.parquet")],')
    # Preserve openly documented structural priors; forbid held-out families from generating NPX training labels.
    edit(42, '        for s in group:\n            f = feats(s)', """        for s in group:
            if lab == 1 and _four_mask.excluded_smiles(s):
                continue
            f = feats(s)""")
    # Only evidence-free membership is read by the exclusion tier; membership remains that of original 0.505.
    edit(44, 'tier_a, tier_b, info = ex_sets(train, inp, deadline)',
         "tier_a, tier_b, info = ex_sets(os.environ['PANEL_ORIGINAL_TRAIN'], inp, deadline)")
    set_cell(book, 44, cell(book, 44) + '\n' + (HERE / 'export.source').read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('/private/tmp/enveda-075-public-0505-v7/source.ipynb'))
    p.add_argument('--panel', type=Path, default=Path('/private/tmp/enveda-four-panel'))
    p.add_argument('--out', type=Path, default=Path('/private/tmp/enveda-four-base'))
    args = p.parse_args()
    raw = args.source.read_bytes()
    assert digest(raw) == SOURCE_SHA
    book = json.loads(raw)
    bootstrap = route_engine(book)
    adapt_outer(book, args.panel / 'blind', args.panel / 'private/holdout.json', bootstrap)
    for c in book['cells']:
        if c['cell_type'] == 'code':
            c['outputs'] = []
            c['execution_count'] = None
    meta = json.loads((args.source.parent / 'metadata.json').read_text())
    slug = 'enveda-075-np-four-arm'
    metadata = dict(id='jiyujia/' + slug, title=slug, language='python', kernel_type='notebook',
                    code_file=OUTPUT_NAME, is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=False,
                    competition_sources=meta['competitionDataSources'], dataset_sources=meta['datasetDataSources'],
                    kernel_sources=meta['kernelDataSources'], model_sources=[], docker_image=meta['dockerImage'],
                    machine_shape='NvidiaTeslaT4')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / OUTPUT_NAME).write_text(json.dumps(book, ensure_ascii=False, indent=1)+'\n')
    (args.out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(args.out / OUTPUT_NAME)


if __name__ == '__main__':
    main()

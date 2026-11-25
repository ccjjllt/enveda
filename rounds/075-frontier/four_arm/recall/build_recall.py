"""Build a blinded Pablo v29 recall comparison on Kaggle T4."""
import argparse
import importlib.util
from hashlib import sha256
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
FOUR_ARM = HERE.parent
SOURCE_SHA = 'a183882abe0b131581109455e1b704ec6d93f27fb99e7b5c2f07d40063e9c5af'
ARMS = {'baseline': 'base', 'early_score': 'early', 'late_dedupe': 'dedup', 'joint': 'joint'}
RUN_SLUG = 'enveda-075-np-recall'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


masked = load_module('recall_masked_builder', FOUR_ARM / 'baseline/build.py')
assemble = load_module('recall_notebook_cells', FOUR_ARM / 'assemble.py')


def adapt_outer(book, blind, holdout, bootstrap):
    def edit(i, before, after):
        masked.set_cell(book, i, masked.rewrite(masked.cell(book, i), before, after))

    config = masked.cell(book, 2)
    begin = config.index('COMP = next((p for p in (INPUT')
    end = config.index("if (COMP / 'test.parquet').is_file():", begin)
    edit(2, config[begin:end], masked.prepare_source(blind, holdout, bootstrap))
    # Use the hidden-equivalent full path on the blinded validation input.
    edit(2, "RUN_PIPELINE = IS_RERUN or COMMIT_MODE != 'fast'", "assert IS_RERUN\nRUN_PIPELINE = True")
    for i, path_root in ((20, 'inp'), (23, 'inp_dir'), (25, 'inp')):
        edit(i, f"glob.glob({path_root} + '/**/test.parquet', recursive=True)", "[str(COMP / 'test.parquet')]")
    for name in ('test', 'train'):
        edit(22, f'inp.rglob("{name}.parquet")', f'[COMP / "{name}.parquet"]')

    edit(20, "    elapsed = round(time.time() - t0, 1)", """    elapsed = round(time.time() - t0, 1)
    from pathlib import Path
    import json
    (Path(work) / 'n1c_omni_audit.json').write_text(json.dumps(dict(
        status='applied', molecules=len(L), spectra=len(K), changed=nch, top1_changed=nt1,
        elapsed_s=elapsed)))""")
    edit(23, '    sub.to_csv(sp, index=False)', """    sub.to_csv(sp, index=False)
    from pathlib import Path
    import json
    (Path(work_dir) / 'dual_sync_audit.json').write_text(json.dumps(dict(
        status='applied', molecules=len(sub), consensus_molecules=len(dual_neutrals),
        adjusted=adjusted, demoted=demoted)))""")

    edit(25, '  refs = []', '''  refs = []
  _source_removed = 0
  _mask_meta_fields = ('source_library', 'library', 'spectrum_id', 'accession',
                       'record_id', 'source', 'contributor', 'authors', 'origin')''')
    edit(25, '      for idx, row in meta_df.iterrows():', '''      for idx, row in meta_df.iterrows():
        if any(_four_mask.blocked_source(row.get(key, '')) for key in _mask_meta_fields):
          _source_removed += 1
          continue''')
    edit(25, "  if mb_hits and 'massbank' in mb_hits[0].lower():", '''  if mb_hits and _four_mask.blocked_source('massbank'):
    import pyarrow.parquet as _mb_pq
    _source_removed += _mb_pq.ParquetFile(mb_hits[0]).metadata.num_rows
    mb_hits = []
  if mb_hits and 'massbank' in mb_hits[0].lower():''')
    edit(25, '''      mb_df = pd.read_parquet(
          mb_hits[0],
          columns=['smiles', 'exact_mass', 'ion_mode', 'mzs', 'intensities', 'is_casmi_subset'],
      )''', '''      import pyarrow.parquet as _mb_pq
      _base_columns = ['smiles', 'exact_mass', 'ion_mode', 'mzs', 'intensities', 'is_casmi_subset']
      _available = set(_mb_pq.read_schema(mb_hits[0]).names)
      _provenance = [key for key in _mask_meta_fields if key in _available]
      mb_df = pd.read_parquet(mb_hits[0], columns=_base_columns + _provenance)
      if _provenance:
        _by_source = mb_df[_provenance].apply(lambda column: column.map(_four_mask.blocked_source)).any(axis=1)
        _source_removed += int(_by_source.sum())
        mb_df = mb_df[~_by_source]''')
    edit(25, '  hits_count = 0', "  _checked_refs, _excluded_refs = set(), set()\n  hits_count = 0")
    edit(25, '          cos, n_matched = calc_cosine(q_mz, q_it, ref[4], ref[5])', '''          if ri not in _checked_refs:
            _checked_refs.add(ri)
            if _four_mask.excluded_smiles(ref[7]) or _four_mask.copied_spectrum(ref[1], ref[4], ref[5]):
              _excluded_refs.add(ri)
          if ri in _excluded_refs:
            continue
          cos, n_matched = calc_cosine(q_mz, q_it, ref[4], ref[5])''')
    edit(25, '      hits=hits_count,', """      status='applied',
      reference_rows=len(refs), source_rows_removed=_source_removed,
      window_reference_rows_removed=len(_excluded_refs),
      hits=hits_count,""")
    edit(25, "  if not refs:\n    print('[apex-extlib] No external reference spectra available; keeping submission untouched.')\n    return",
         """  if not refs:
    if not _source_removed:
      raise RuntimeError('APEX reference inventory unavailable')
    audit = dict(status='applied: no reference survives source holdout', molecules=len(sub),
                 reference_rows=0, source_rows_removed=_source_removed, window_reference_rows_removed=0,
                 hits=0, promoted=0, tier2_hits=0, tier2_moved=0, mass_rejected=0,
                 elapsed_s=round(time.monotonic() - t0, 1))
    with open(os.path.join(work, 'apex_extlib_audit.json'), 'w') as stream:
      json.dump(audit, stream, indent=2)
    print('[apex-extlib] No reference survives source holdout.')
    return""")
    adapt_tails(book)


def adapt_tails(book):
    def edit(i, before, after):
        masked.set_cell(book, i, masked.rewrite(masked.cell(book, i), before, after))

    # Group masking adds structure normalization to the original NPX featurization.
    edit(1, 'NPX_BUDGET_S = 1200.0', 'NPX_BUDGET_S = 4200.0')
    edit(35, '        for s in group:\n            f = feats(s)', '''        for s in group:
            if lab == 1:
                try:
                    blocked = _four_mask.excluded_smiles(s)
                except (ValueError, RuntimeError):
                    # An unnormalizable structure cannot establish disjointness.
                    continue
                if blocked:
                    continue
            f = feats(s)''')
    edit(35, "    target = Path(work) / 'submission.csv'", """    target = Path(work) / 'submission.csv'
    if target.is_file():
        (Path(work) / 'submission_npx_input.csv').write_bytes(target.read_bytes())""")
    # Static membership used by E180X is a structure prior, independent of measured reference spectra.
    edit(37, 'tier_a, tier_b, info = ex_sets(train, inp, deadline)',
         "tier_a, tier_b, info = ex_sets(os.environ['PANEL_ORIGINAL_TRAIN'], inp, deadline)")


def finish_source(arm, notifier):
    return f"""
assert 'Tesla T4' in GPU and IS_RERUN and PIPELINE_DONE and NEURAL_OK
(ROOT / 'pilot_provenance_audit.json').write_text(json.dumps(PROVENANCE_REPORT, indent=2))
_pilot_runtime = {notifier!r}
_pilot_program = Path('/tmp/pilot_fusion.py')
_pilot_program.write_bytes(zlib.decompress(base64.b64decode(_pilot_runtime[0])))
assert hashlib.sha256(_pilot_program.read_bytes()).hexdigest() == _pilot_runtime[1]
subprocess.run([sys.executable, str(_pilot_program), 'seal', '--arm', {arm!r},
                '--work', str(ROOT), '--queries', str(COMP / 'test.parquet')], check=True)
"""


def build(source, metadata, panel, output, arm, stage_path, fusion_path):
    raw = source.read_bytes()
    if sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError('Expected the official Pablo v29 source used for submission 58820476')
    book = json.loads(raw)
    options = json.loads(metadata.read_text())
    if options.get('kernel_type') != 'notebook' or not options.get('dataset_sources'):
        raise ValueError('Expected CLI kernel-metadata.json with the public Pablo data mounts')
    stage = load_module('recall_intervention', stage_path)

    def patch_inner(inner):
        stage.patch_inner(inner, arm)
        export = "pubchem=list(PC.get(mid,{}).get('pc', []))) for mid in BASE}"
        extended = """pubchem=list(PC.get(mid,{}).get('pc', [])),
                              fused_overflow=list(FUSION_OVERFLOW.get(str(mid), []))) for mid in BASE}"""
        masked.set_cell(inner, 54, masked.rewrite(masked.cell(inner, 54), export, extended))

    bootstrap = masked.route_engine(book, bundle_index=10, driver_index=11,
                                   patch_inner=patch_inner, allow_empty_extension=True)
    adapt_outer(book, panel / 'blind', panel / 'private/holdout.json', bootstrap)

    book['cells'] = [c for i, c in enumerate(book['cells']) if c['cell_type'] == 'code' and i <= 37]
    book['cells'].append(assemble.code_cell('seal-recall-arm', finish_source(arm, assemble.pack(fusion_path))))
    for i, c in enumerate(book['cells']):
        c['outputs'] = []
        c['execution_count'] = None
        compile(''.join(c['source']), f'outer-pablo-{i}', 'exec')
    slug = f'{RUN_SLUG}-{ARMS[arm]}'
    options.update(id=f'jiyujia/{slug}', title=slug, code_file='pilot.ipynb', is_private=True,
                   enable_internet=False, enable_gpu=True, machine_shape='NvidiaTeslaT4')
    output.mkdir(parents=True, exist_ok=True)
    path = output / options['code_file']
    path.write_text(json.dumps(book, ensure_ascii=False, separators=(',', ':')) + '\n')
    (output / 'kernel-metadata.json').write_text(json.dumps(options, indent=2) + '\n')
    return dict(arm=arm, slug=slug, notebook_sha256=sha256(path.read_bytes()).hexdigest(),
                query_sha256=sha256((panel / 'blind/test.parquet').read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('source', 'metadata', 'panel', 'output', 'stage-path', 'fusion-path'):
        parser.add_argument('--' + arg, type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))

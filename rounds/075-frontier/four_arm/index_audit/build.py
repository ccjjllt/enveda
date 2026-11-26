"""Build a private, offline CPU kernel for the frozen CASMI five-miss audit."""
import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
FRONTIER = HERE.parents[1]
V4G = 'dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets'
POPULARITY = 'dmitriigluzdov/casmi26-pubchem-popularity-prior'
WHEEL = 'metric/rdkit-2026-3-3-wheel'
CONTEST = 'enveda-CASMI26-molecule-id-mass-spectra'
SLUG = 'enveda-075-five-miss-full-index-audit'
IDS = {'challenge-'+x for x in ('011','033','034','035','178')}
EXPECTED = {
    'labels': '71124739e3fc24a2d32c5a08e029eec5325f8d433a3eddbf4aed69a850fad8a1',
    'spectra': '55fe6cb0344d10ee9a564e0bb3fcfea1c7cabedce4fa62bb82604795a947928f',
    'effective_spectra': '071218191b7bb59958d7288f76aa8f752714fe41d8216a5785fb179fbdee9a41',
}


def sections(path, names):
    source = path.read_text()
    output = []
    for node in ast.parse(source).body:
        assignment = node.targets[0] if isinstance(node,ast.Assign) else None
        name = assignment.id if isinstance(assignment,ast.Name) else getattr(node,'name',None)
        if name in names:
            output.append(ast.get_source_segment(source,node))
    return '\n\n'.join(output)


def build(output,panel,archive,include_main):
    labels,spectra,effective_spectra,pc_lists = (
        panel/'private/labels.parquet',panel/'blind/test.parquet',
        archive/'effective_query_spectra.parquet',archive/'pc_lists.json')
    paths = {'labels':labels, 'spectra':spectra, 'effective_spectra':effective_spectra, 'pc_lists':pc_lists}
    hashes = {name:sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}
    if any(hashes[name]!=value for name,value in EXPECTED.items()):
        raise ValueError(f'Frozen evaluation inputs changed: {hashes}')
    pc = json.loads(pc_lists.read_text())
    effective = pd.read_parquet(effective_spectra).set_index('molecule_id')
    selected = pd.read_parquet(labels).loc[lambda df:df.challenges.isin(IDS)]
    records = []
    for target in selected.sort_values('challenges').itertuples(index=False):
        row = effective.loc[target.molecule_id]
        if row.ndim!=1 or row.adduct!='[M+H]+':
            raise ValueError('Expected one positive archived scan per miss.')
        c34_target = float(pc[target.molecule_id]['target'])
        if float(row.nm) != c34_target:
            raise ValueError('C34 target differs from the archived query input.')
        records.append(dict(challenge=target.challenges,molecule_id=target.molecule_id,
                            precursor_mz=float(row.precursor_mz),original_adduct=row.adduct,
                            original_C34_target=c34_target,truth_smiles=target.truth_smiles,
                            truth_key=target.truth_key))
    if {row['challenge'] for row in records}!=IDS:
        raise ValueError('Frozen panel is missing a target compound.')
    prereqs = HERE / 'index_inputs.source'
    helpers = sections(prereqs, {'find', 'chemistry'})
    helpers = helpers.replace(
        "wheel, = ROOT.rglob(f'rdkit-2026.3.3-{tag}-*.whl')",
        "wheel, = (p for p in ROOT.rglob(f'rdkit-2026.3.3-{tag}-*.whl')\n"
        "                  if 'casmi26-attributed-v4g-reproduction-assets' not in p.parts)")
    helpers += '\n\n' + sections(prereqs, {'POPULARITY', 'index_files'})
    definitions = f'FROZEN_INPUT = {records!r}\nFROZEN_HASHES = {hashes!r}\nINCLUDE_MAIN = {include_main!r}\n'
    definitions += f'FASTSCAN_SOURCE = {(HERE/"fastscan.cpp").read_text()!r}'
    source = '\n\n'.join([helpers,definitions,(HERE/'audit.py').read_text()])
    compile(source,'audit.py','exec')
    metadata = dict(id=f'jiyujia/{SLUG}',title=SLUG,code_file='audit.py',language='python',
                    kernel_type='script',is_private=True,enable_gpu=False,enable_internet=False,
                    dataset_sources=[WHEEL,POPULARITY,V4G] if include_main else [WHEEL,POPULARITY],
                    kernel_sources=[],competition_sources=[CONTEST] if include_main else [])
    output.mkdir(parents=True,exist_ok=True)
    (output/'audit.py').write_text(source)
    (output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return (output/'audit.py').resolve()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--panel',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--include-main',action='store_true')
    print(build(**vars(parser.parse_args())))

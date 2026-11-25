"""Package a training-only dual-polarity counterfactual comparison on Kaggle CPU."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from runpy import run_path


HERE = Path(__file__).resolve().parent
COMMON = run_path(str(HERE.parent/'virtual_np'/'build.py'))
SUPPLY = 'enveda-075-dual-energy-pair-supply'


def build(out, source_version, pairs_per_fold, *, two_cut=False, learn_two_cut=False,
          forward_swap=False, glacier_forward_swap=False, glacier_cpu_forward_swap=False):
    if source_version < 1 or pairs_per_fold < 1:
        raise ValueError('Supply version and fold sample size must be positive')
    if sum((two_cut, learn_two_cut, forward_swap, glacier_forward_swap, glacier_cpu_forward_swap)) > 1:
        raise ValueError('Choose a single physical-scoring comparison')
    slug, entry = 'enveda-075-dual-branch-swap', 'fit_dual_energy_pairs.py'
    if two_cut:
        slug, entry = 'enveda-075-two-cut-hard-pairs', 'probe_two_cut_fragments.py'
    if learn_two_cut:
        slug, entry = 'enveda-075-learned-two-cut-hard', 'fit_two_cut_branches.py'
    if forward_swap:
        slug, entry = 'enveda-075-ice-forward-hard-pair', 'fit_forward_swap.py'
    glacier = glacier_forward_swap or glacier_cpu_forward_swap
    if glacier:
        slug, entry = 'enveda-075-gl-forward-hard-pair', 'fit_glacier_forward_swap.py'
        if glacier_cpu_forward_swap:
            slug += '-cpu'
    runtime_args = ['--device','cpu'] if glacier_cpu_forward_swap else []
    sample_arg = '--pairs-per-fold' if entry == 'fit_dual_energy_pairs.py' else '--per-fold'
    sources = {
        'fit_dual_energy_pairs.py': HERE/'fit_dual_energy_pairs.py',
        'read_dual_training_scans.py': HERE/'read_dual_training_scans.py',
        'cleavage_rank.py': HERE.parent/'cleavage_rank'/'train.py',
    }
    if two_cut or learn_two_cut:
        sources['probe_two_cut_fragments.py'] = HERE/'probe_two_cut_fragments.py'
    if learn_two_cut or forward_swap or glacier:
        sources[entry] = HERE/entry
    if forward_swap or glacier:
        sources['forward_ce.py'] = HERE.parent/'ce_fragments'/'score.py'
    if glacier:
        sources['fit_forward_swap.py'] = HERE/'fit_forward_swap.py'
        sources['np_glacier_reference.py'] = HERE.parent/'np_simulation'/'simulate.py'
    packaged = {}
    for name, path in sources.items():
        raw = path.read_bytes()
        compile(raw, str(path), 'exec')
        packaged[name] = COMMON['pack'](raw)
    bootstrap = '' if forward_swap or glacier else COMMON['METRIC_BOOTSTRAP']
    runtime = f'''
import base64
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib

inputs = Path('/kaggle/input')
{bootstrap}
train, = (p for p in inputs.rglob('train.parquet') if 'enveda-casmi26-molecule-id-mass-spectra' in
          (d.lower() for d in p.parts))
receipt, = (p for p in inputs.rglob('dual_energy_pair_supply.json') if {SUPPLY!r} in p.parts)
pairs = receipt.parent/'dual_energy_training_pairs.jsonl'
with tempfile.TemporaryDirectory(prefix='branch-swap-') as directory:
    root = Path(directory)
    for name, (blob, expected) in {packaged!r}.items():
        value = zlib.decompress(base64.b64decode(blob))
        if sha256(value).hexdigest() != expected:
            raise ValueError(f'Branch matcher payload changed: {{name}}')
        (root/name).write_bytes(value)
    settings = dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    subprocess.run([sys.executable, '-u', str(root/{entry!r}), '--train', str(train),
                    '--pairs', str(pairs), '--receipt', str(receipt), '--out', '/kaggle/working',
                    {sample_arg!r}, {str(pairs_per_fold)!r}, *{runtime_args!r}], env=settings, check=True)
'''
    compile(runtime, slug, 'exec')
    nb = dict(nbformat=4, nbformat_minor=5,
              metadata={'kernelspec': {'display_name':'Python 3', 'language':'python', 'name':'python3'}},
              cells=[dict(cell_type='code', id='dual-branch-swap', metadata={}, execution_count=None, outputs=[],
                          source=runtime.strip().splitlines(keepends=True))])
    datasets = ['ahmedberatozer/casmi26-iceberg', 'alexchilton/rdkit-2025-3-6-cp313'] if forward_swap else [COMMON['WHEEL']]
    meta = dict(id=f'jiyujia/{slug}', title=slug, code_file='branch_swap.ipynb', language='python', kernel_type='notebook',
                is_private=True, enable_gpu=False, enable_internet=False, docker_image=COMMON['IMAGE'],
                dataset_sources=datasets, kernel_sources=[f'jiyujia/{SUPPLY}/{source_version}'],
                competition_sources=['enveda-CASMI26-molecule-id-mass-spectra'])
    if glacier:
        reference = json.loads((HERE.parent/'np_simulation'/'kernel-metadata.json').read_text())
        meta.update(docker_image=reference['docker_image'],
                    dataset_sources=['ahmedberatozer/casmi26-glacier', 'ahmedberatozer/casmi26-iceberg'])
        if glacier_forward_swap:
            meta.update(enable_gpu=True,machine_shape=reference['machine_shape'])
    out.mkdir(parents=True, exist_ok=True)
    notebook = out/meta['code_file']
    notebook.write_text(json.dumps(nb, separators=(',',':'))+'\n')
    (out/'kernel-metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(sha256(notebook.read_bytes()).hexdigest())


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out', type=Path, required=True)
    cli.add_argument('--source-version', type=int, default=2)
    cli.add_argument('--pairs-per-fold', type=int)
    modes = cli.add_mutually_exclusive_group()
    modes.add_argument('--two-cut', action='store_true')
    modes.add_argument('--learn-two-cut', action='store_true')
    modes.add_argument('--forward-swap', action='store_true')
    modes.add_argument('--glacier-forward-swap', action='store_true')
    modes.add_argument('--glacier-cpu-forward-swap', action='store_true')
    args = cli.parse_args()
    if args.pairs_per_fold is None:
        args.pairs_per_fold = 48 if args.forward_swap or args.glacier_forward_swap or args.glacier_cpu_forward_swap else 400
    build(**vars(args))

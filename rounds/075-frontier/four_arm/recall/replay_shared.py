"""Package a blind CPU replay of the two remaining T4 tails using reseal/2's frozen model."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from build_recall import FOUR_ARM, SOURCE_SHA, assemble, load_module, masked
from replay_tails import INPUTS


SHARED = 'enveda-075-np-recall-reseal'
SLUG = 'enveda-075-np-recall-shared-tails'
MODEL_SHA = '8c636d0d71ff72db7971eb6aa1a11be57b7797f85104b7b2e7f883d92f73dc64'
FIT_SHA = '716aa01e108e54162711548289e47f4f1c05b61154c611e505f8e547b6928c9e'
BASE_ZIP_SHA = '05509b32cfb9bcbe8ab563244f91c97e89f878eba3cc40263e3f48fd4a1f372d'
CONTROL_SHA = '59532a7ff0d09e609b4c5ca7943ebeb6a43691014d2508cdd0864c99ec833450'
BOOTSTRAP = load_module('shared_tail_metric_wheel', FOUR_ARM.parent / 'virtual_np/build.py').METRIC_BOOTSTRAP


def build(source, metadata, manifest, queries, control_input, output, arms, spills=None):
    raw = source.read_bytes()
    plan = json.loads(manifest.read_text())
    if (sha256(raw).hexdigest() != SOURCE_SHA or sha256(control_input.read_bytes()).hexdigest() != CONTROL_SHA
            or assemble.pack(queries)[1] != plan['panel']['output_sha256']['blind/test.parquet']):
        raise ValueError('Expected the original Pablo v29 program and the pinned 60-query/control inputs')
    book = json.loads(raw)
    jobs = {arm: plan['jobs'][arm] for arm in dict.fromkeys(arms)}
    sources = [f'jiyujia/{SHARED}/2']
    spill_payloads = {}
    inputs = sorted(INPUTS - {'npx_audit.json', 'submission_pre_e180x.csv', 'pilot_provenance_audit.json'})
    for arm, job in jobs.items():
        directory = spills / arm if spills else None
        if directory is None or not directory.is_dir():
            sources.append(f"jiyujia/{job['slug']}/{job['version']}")
            continue
        version = json.loads((directory / 'version.json').read_text())
        expected = dict(owner='jiyujia', slug=job['slug'], version=job['version'])
        if version != expected:
            raise ValueError(f'{arm}: the spill is not the designated T4 version')
        files = set(inputs) | {'submission_npx_input.csv'}
        if json.loads((directory / 'n1c_dl1_audit.json').read_text())['changed_lists']:
            files.add('submission_pre_n1c_dl1.csv')
        if (directory / 'pilot_provenance_audit.json').exists():
            files.add('pilot_provenance_audit.json')
        spill_payloads[arm] = {name: assemble.pack(directory / name) for name in sorted(files)}
    replay = f"""
from pathlib import Path
import base64, hashlib, json, pickle, shutil, subprocess, sys, tempfile, zlib
from zipfile import ZIP_DEFLATED, ZipFile
inputs = Path('/kaggle/input')
root = Path('/kaggle/working')
{BOOTSTRAP}

def mounted(slug, name):
    path, = (p for p in inputs.rglob(name) if slug in p.parts)
    return path

def verify(path, expected):
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
            raise ValueError(f'Unexpected digest for {{path.name}}')
    return path

def restore(path, packed):
    payload, expected = packed
    path.write_bytes(zlib.decompress(base64.b64decode(payload)))
    return verify(path, expected)

def read(path):
    return json.loads(path.read_text())

fit = read(verify(mounted({SHARED!r}, 'shared_npx_fit.json'), {FIT_SHA!r}))
with verify(mounted({SHARED!r}, 'shared_npx_model.pickle'), {MODEL_SHA!r}).open('rb') as stream:
    shared = pickle.load(stream)
if shared['fit'] != fit or not fit['gate_ok']:
    raise ValueError('Unverified NPX fit')
def npx_train(*args, **kwargs):
    return shared['model'], npx_featurizer(), shared['table'], shared['fit']

COMP = Path(tempfile.mkdtemp(prefix='shared-tail-input-'))
raw, = (p for p in inputs.rglob('train.parquet')
         if any(s.lower() == 'enveda-casmi26-molecule-id-mass-spectra' for s in p.parts))
(COMP / 'train.parquet').symlink_to(raw)
for name, slug, checksum in [('pool_meta.parquet','casmi26-attributed-v4g-reproduction-assets',
                              '82b8c40315611289b9a4004b3af18d6d3402b3cd5baa4ee5723da07b7e2f6bb0'),
                             ('natural_products.parquet','casmi26-natural-product-knowledge-table',
                              '6ccb833e47d88715fca1d614b05cae92fa333fa388ae1d64226f2695a62081aa')]:
    matches = sorted((p for p in inputs.rglob(name) if slug in p.parts), key=lambda p: (len(p.parts), str(p)))
    for match in matches:
        try:
            verify(match, checksum)
        except ValueError:
            continue
        (COMP / name).symlink_to(match)
        break
    else:
        raise ValueError(f'No original {{name}} in {{slug}}')
query = restore(COMP / 'test.parquet', {assemble.pack(queries)!r})
scorer = restore(COMP / 'seal.py', {assemble.pack(FOUR_ARM / 'retrieval_pilot_score.py')!r})

def apply_tail(target):
    n = npx_run(str(target), str(COMP), True, enabled=True, cap_s=NPX_BUDGET_S)
    e = ex_run(str(target), str(COMP), True, enabled=True, cap_s=E180X_BUDGET_S)
    if n['status'] != 'applied' or e['status'] != 'applied' or n.get('feature_receipt') != fit['feature_receipt']:
        raise RuntimeError((n,e))

control = COMP / 'control'
control.mkdir()
restore(control / 'submission.csv', {assemble.pack(control_input)!r})
apply_tail(control)
with ZipFile(verify(mounted({SHARED!r}, 'baseline.zip'), {BASE_ZIP_SHA!r})) as archive:
    baseline = json.loads(archive.read('pilot_arm_receipt.json'))
    baseline_budget = json.loads(archive.read('pilot_forward_budget.json'))['cases']
    if (control / 'submission.csv').read_bytes() != archive.read('submission_unpadded.csv'):
        raise RuntimeError('Shared model failed exact replay of the baseline tail')

spills = {spill_payloads!r}
for arm, slug in { {arm:job['slug'] for arm,job in jobs.items()}!r}.items():
    if arm in spills:
        stage = COMP / (arm + '-spill')
        stage.mkdir()
        for name, packed in spills[arm].items():
            restore(stage / name, packed)
        checkpoint = stage / 'submission_npx_input.csv'
    else:
        checkpoint = mounted(slug, 'submission_npx_input.csv')
        stage = checkpoint.parent
    target = COMP / arm
    target.mkdir()
    for name in {inputs!r}:
        shutil.copyfile(stage / name, target / name)
    if read(target / 'four_reference_audit.json') != shared['reference_counts']:
        raise ValueError(f'{{arm}}: reference exclusion differs')
    if read(target / 'uniq_audit.json')['status'] != 'applied':
        raise RuntimeError(f'{{arm}}: incomplete UNIQ stage')
    if read(target / 'n1c_dl1_audit.json')['changed_lists']:
        name = 'submission_pre_n1c_dl1.csv'
        shutil.copyfile(stage / name, target / name)
    provenance = 'pilot_provenance_audit.json'
    restored_provenance = not (stage / provenance).is_file()
    if restored_provenance:
        if PROVENANCE is not False:
            raise ValueError('Active provenance stage requires its original audit')
        (target / provenance).write_text(json.dumps(dict(status='disabled in this experiment'), indent=2))
    else:
        shutil.copyfile(stage / provenance, target / provenance)
    shutil.copyfile(checkpoint, target / 'submission.csv')
    apply_tail(target)
    subprocess.run([sys.executable, str(scorer), 'seal', '--arm', arm, '--work', str(target),
                    '--queries', str(query)], check=True)
    receipt = read(target / 'pilot_arm_receipt.json')
    for name in ['predictor_budget','predictor_requests_sha256','predictor_parameters',
                 'queries_sha256','reference_mask_sha256','extension_counts','gpu']:
        if receipt[name] != baseline[name]:
            raise ValueError(f'{{arm}}: unpaired baseline {{name}}')
    arm_budget = read(target / 'pilot_forward_budget.json')['cases']
    if any(arm_budget[mid]['finite_counts'] != r['finite_counts'] for mid,r in baseline_budget.items()):
        raise ValueError(f'{{arm}}: unequal per-query successful predictor counts')
    with ZipFile(root / f'{{arm}}.zip','w',compression=ZIP_DEFLATED) as z:
        for name in ('pilot_arm_receipt.json', *receipt['hashes']):
            z.write(target / name, name)
    details = dict(source=slug, shared_model_sha256={MODEL_SHA!r}, exact_control_tail=True,
                   source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   restored_disabled_provenance=restored_provenance)
    (root / f'{{arm}}_replay.json').write_text(json.dumps(details, indent=2))
"""
    # Define the published inference functions without automatic notebook execution or training.
    program = '_NPX_NOTEBOOK_RUN = _EX_NOTEBOOK_RUN = False\n' + masked.cell(book, 35) + '\n' + masked.cell(book, 37)
    book['cells'] = [assemble.code_cell('config', masked.cell(book, 1)),
                     assemble.code_cell('tail-functions', program), assemble.code_cell('shared-replay', replay)]
    for cell in book['cells']:
        compile(''.join(cell['source']), cell['id'], 'exec')
    options = json.loads(metadata.read_text())
    needed = {'casmi26-natural-product-knowledge-table','casmi26-attributed-v4g-reproduction-assets',
              'rdkit-2026-3-3-wheel'}
    options.update(id=f'jiyujia/{SLUG}', title=SLUG, code_file='shared.ipynb',
                   is_private=True, enable_gpu=False, enable_internet=False, kernel_sources=sources,
                   dataset_sources=[s for s in options['dataset_sources'] if s.split('/')[-1] in needed])
    options.pop('machine_shape', None)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'shared.ipynb').write_text(json.dumps(book, ensure_ascii=False, separators=(',', ':'))+'\n')
    (output / 'kernel-metadata.json').write_text(json.dumps(options, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source','metadata','manifest','queries','control-input','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--arms', nargs='+', choices=('late_dedupe','joint'), default=['late_dedupe','joint'])
    parser.add_argument('--spills', type=Path, help='Optional directories <arm>/ from the official outputs with version.json containing owner, slug, version.')
    build(**vars(parser.parse_args()))

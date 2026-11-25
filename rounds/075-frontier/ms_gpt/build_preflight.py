"""Package a private offline check for the pinned MS-GPT MassSpecGym inference assets."""

import argparse
import json
from pathlib import Path


SLUG = 'enveda-075-ms-gpt-assets'
IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:fc720c15545eb096a510e1f085b17fc118387a007599424d953a4f327b71cacf'
SHARED = 'flyfish666/casmi26-ms-gpt-20261031-public-assets/1'
PROFILE = 'flyfish666/casmi26-ms-gpt-msg20261031-profile/1'
METRIC = 'metric/rdkit-2026-3-3-wheel/1'
SHARED_MANIFEST = 'bbadf84786bd6258f521286e61690ddacfa13bfd1f2725b264d911b09643bcc8'
PROFILE_MANIFEST = 'd6ebbe02b95e6349c738222194cb95d6c3f0eef9f465bcee06c4b47417e94a9a'
MS_GPT_REVISION = 'ec032f87dcaa54c6380176c2f959e11335440ae5'
MIST_REVISION = '48e339be6535546c87ba4f6b81a0f787c2e4ac40'

SHARED_FILES = (
    'ATTRIBUTION.md',
    'models/mlm/ms_gpt_mlm.pt',
    'models/safe_gpt_base/config.json',
    'models/safe_gpt_base/generation_config.json',
    'models/safe_gpt_base/tokenizer.json',
    'msgpt-source/LICENSE',
    'msgpt-source/configs/mlm_pretraining.yaml',
    'msgpt-source/density_calibration/mist_official/canonical_band.json',
    'msgpt-source/src/ms_gpt/data/mist_inputs.py',
    'msgpt-source/src/ms_gpt/data/mist_batch.py',
    'msgpt-source/src/ms_gpt/data/posterior_queries.py',
    'msgpt-source/src/ms_gpt/models/conditional_mlm.py',
    'msgpt-source/src/ms_gpt/models/mist_official_encoder.py',
    'msgpt-source/src/ms_gpt/models/posterior_aligned_adapter.py',
    'msgpt-source/src/ms_gpt/pipeline/postprocessing.py',
    'msgpt-source/src/ms_gpt/utils/mol_repair.py',
    'mist-source/LICENSE.md',
    'mist-source/NOTICE.md',
    'mist-source/src/mist/models/scaled_mist.py',
)
ADAPTER_DIR = 'posterior_aligned_adapters/ms_gpt_mlm/mist_official/massspecgym'
PROFILE_FILES = (
    'ATTRIBUTION.md', 'LICENSE.txt', 'mist/mist_official/mist_massspecgym.pt',
    *(f'{ADAPTER_DIR}/{name}' for name in (
        'adapter_model.safetensors', 'adapter_config.json',
        'inference_config.json', 'training_config.json',
    )),
)


def build(out):
    code = f'''
from hashlib import file_digest
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys

from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename


inputs = Path('/kaggle/input')


def locate_manifest(reference):
    slug = reference.split('/')[1]
    path, = (p for p in inputs.rglob('manifest.json') if slug in p.parts)
    return path


def sha256(path):
    with path.open('rb') as file:
        return file_digest(file, 'sha256').hexdigest()


def load_manifest(reference, expected):
    path = locate_manifest(reference)
    if sha256(path) != expected:
        raise ValueError(f'Changed upstream manifest: {{path}}')
    return path.parent, json.loads(path.read_text())


def verify_files(root, manifest, names):
    checked = {{}}
    for name in names:
        expected = manifest['files'][name]
        path = root / name
        size = path.stat().st_size
        checksum = sha256(path)
        if size != expected['bytes'] or checksum != expected['sha256']:
            raise ValueError(f'Changed mounted asset: {{path}}')
        checked[name] = dict(bytes=size, sha256=checksum)
    return checked


def installed_version(project):
    try:
        return version(project)
    except PackageNotFoundError:
        return None


shared_root, shared_manifest = load_manifest({SHARED!r}, {SHARED_MANIFEST!r})
profile_root, profile_manifest = load_manifest({PROFILE!r}, {PROFILE_MANIFEST!r})
if (shared_manifest['msgpt_revision'] != {MS_GPT_REVISION!r}
        or shared_manifest['mist_revision'] != {MIST_REVISION!r}
        or profile_manifest['profile'] != 'official-massspecgym'):
    raise ValueError('Expected the October 31 paired MS-GPT and official MSG MIST release.')

wheel_files = sorted(name for name in shared_manifest['files'] if name.startswith('wheels/'))
checked_shared = verify_files(shared_root, shared_manifest, (*{SHARED_FILES!r}, *wheel_files))
checked_profile = verify_files(profile_root, profile_manifest, {PROFILE_FILES!r})
runtime_tags = set(sys_tags())
wheels = {{}}
for name in wheel_files:
    package, release, _, tags = parse_wheel_filename(Path(name).name)
    if runtime_tags.intersection(tags):
        wheels[package] = dict(version=str(release), file=name)
    else:
        raise ValueError(f'Mirrored wheel does not support this environment: {{name}}')
for path in inputs.rglob('rdkit-*.whl'):
    if 'rdkit-2026-3-3-wheel' not in path.parts:
        continue
    package, release, _, tags = parse_wheel_filename(path.name)
    if str(release) == '2026.3.3' and runtime_tags.intersection(tags):
        wheels[package] = dict(version=str(release), file=path.name, sha256=sha256(path))

dependencies = (
    'torch', 'numpy', 'scipy', 'scikit-learn', 'pandas', 'pyyaml', 'packaging',
    'transformers', 'tokenizers', 'safetensors', 'accelerate', 'peft', 'safe-mol',
    'datamol', 'selfies', 'omegaconf', 'rdkit',
)
installed = {{package: installed_version(package) for package in dependencies}}
missing = [name for name in dependencies
           if installed[name] is None and canonicalize_name(name) not in wheels]
if 'rdkit' not in wheels and installed['rdkit'] not in ('2026.3.3', '2026.03.3'):
    missing.append('RDKit 2026.03.3')

adapter_config = json.loads((profile_root / {ADAPTER_DIR + '/adapter_config.json'!r}).read_text())
inference_config = json.loads((profile_root / {ADAPTER_DIR + '/inference_config.json'!r}).read_text())
calibration = json.loads((shared_root / inference_config['calibration_file'].replace(
    'density_calibration/', 'msgpt-source/density_calibration/', 1)).read_text())
band = calibration['benchmarks']['massspecgym']
if (inference_config['benchmark'] != 'massspecgym'
        or inference_config['mist_implementation'] != 'mist_official'
        or adapter_config['r'] != 16
        or inference_config['input_protocol'] != 'annotated_topn'
        or band['calibration_split'] != 'train'
        or band['max_fragment_peaks'] != inference_config['max_fragment_peaks']):
    raise ValueError('MSG MIST, LoRA and posterior band are not a matched inference profile.')

receipt = dict(
    status='assets_available' if not missing else 'missing_runtime_assets',
    python=sys.version,
    pinned_revisions=dict(ms_gpt={MS_GPT_REVISION!r}, mist={MIST_REVISION!r}),
    source_refs={{'shared': {SHARED!r}, 'profile': {PROFILE!r}, 'metric': {METRIC!r}}},
    manifest_sha256={{'shared': {SHARED_MANIFEST!r}, 'profile': {PROFILE_MANIFEST!r}}},
    checked_files={{'shared': checked_shared, 'profile': checked_profile}},
    compatible_wheels=wheels,
    installed_distributions=installed,
    missing_runtime_assets=missing,
    benchmark=inference_config['benchmark'],
    input_protocol=inference_config['input_protocol'],
    max_fragment_peaks=inference_config['max_fragment_peaks'],
    published_draws=inference_config['num_queries'] * inference_config['samples_per_query'],
)
(Path('/kaggle/working') / 'ms_gpt_assets_receipt.json').write_text(json.dumps(receipt, indent=2) + '\\n')
print(receipt['status'], 'checked files', len(checked_shared) + len(checked_profile),
      'missing runtime assets', missing, flush=True)
'''
    compile(code, SLUG, 'exec')
    notebook = dict(nbformat=4, nbformat_minor=5, cells=[
        dict(cell_type='code', id='ms-gpt-asset-preflight', execution_count=None,
             metadata={}, outputs=[], source=code.strip().splitlines(keepends=True)),
    ], metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    metadata = dict(
        id=f'jiyujia/{SLUG}', title=SLUG, code_file='preflight.ipynb', language='python',
        kernel_type='notebook', is_private=True, enable_gpu=False, enable_internet=False,
        dataset_sources=[SHARED, PROFILE, METRIC], competition_sources=[], kernel_sources=[],
        docker_image=IMAGE,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / 'preflight.ipynb').write_text(json.dumps(notebook, separators=(',', ':')) + '\n')
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    build(parser.parse_args().out)

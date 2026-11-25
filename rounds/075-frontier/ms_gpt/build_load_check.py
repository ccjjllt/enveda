"""Package a private Kaggle CPU check that imports and loads the pinned MS-GPT models."""

import argparse
import json
from pathlib import Path
from textwrap import indent

from build_preflight import (ADAPTER_DIR, METRIC, MIST_REVISION, MS_GPT_REVISION, PROFILE,
                             PROFILE_MANIFEST, SHARED, SHARED_MANIFEST)


SLUG = 'enveda-075-ms-gpt-load-check'
IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'
RUNTIME = 'jiyujia/enveda-075b-fragnnet-cp312-runtime/1'
RUNTIME_WHEELS = {
    'lightning-2.4.0-py3-none-any.whl': '560163af9711cf59055c448232c473150a299089efce0d2be3cc3288082d8768',
    'lightning_utilities-0.14.3-py3-none-any.whl': '4ab9066aa36cd7b93a05713808901909e96cc3f187ea6fd3052b2fd91313b468',
    'torchmetrics-1.6.3-py3-none-any.whl': '034640d4e1c30b7d3488551b1ce3fc1f363070811e6b97b9cb6b60a9ec15e9f4',
}


def model_runtime(use_gpu=False, after_load=''):
    code = f'''
from hashlib import file_digest
from importlib import invalidate_caches
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename


start = monotonic()
inputs = Path('/kaggle/input')
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                  OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
if not {use_gpu!r}:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
if sys.version_info[:2] != (3, 12):
    raise ValueError(f'The draft load path requires CPython 3.12, found {{sys.version}}.')


def find(name, source):
    path, = (p for p in inputs.rglob(name) if source.split('/')[1] in p.parts)
    return path


def checked(path, expected):
    with path.open('rb') as stream:
        if file_digest(stream, 'sha256').hexdigest() != expected:
            raise ValueError(f'Unexpected bytes in {{path}}')
    return path


def read_manifest(source, expected):
    path = checked(find('manifest.json', source), expected)
    return path.parent, json.loads(path.read_text())


def asset(root, manifest, name):
    return checked(root / name, manifest['files'][name]['sha256'])


def remove_optional_torchao():
    try:
        installed = version('torchao')
    except PackageNotFoundError:
        installed = None
    if installed is not None:
        command = [sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao']
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode and 'externally-managed-environment' in result.stderr:
            subprocess.run([*command, '--break-system-packages'], check=True)
        else:
            result.check_returncode()
        invalidate_caches()
    try:
        still_installed = version('torchao')
    except PackageNotFoundError:
        still_installed = None
    if still_installed is not None or find_spec('torchao') is not None:
        raise RuntimeError('The optional torchao backend remains visible to PEFT.')
    return installed


shared, shared_manifest = read_manifest({SHARED!r}, {SHARED_MANIFEST!r})
profile, profile_manifest = read_manifest({PROFILE!r}, {PROFILE_MANIFEST!r})
if (shared_manifest['msgpt_revision'] != {MS_GPT_REVISION!r}
        or shared_manifest['mist_revision'] != {MIST_REVISION!r}
        or profile_manifest['profile'] != 'official-massspecgym'):
    raise ValueError('Different paired MS-GPT release.')

path_config = asset(profile, profile_manifest, {ADAPTER_DIR + '/inference_config.json'!r})
profile_adapter = path_config.parent
for name in ('adapter_config.json', 'adapter_model.safetensors'):
    asset(profile, profile_manifest, {ADAPTER_DIR!r} + '/' + name)
config = json.loads(path_config.read_text())
if config['benchmark'] != 'massspecgym' or config['mist_implementation'] != 'mist_official':
    raise ValueError('Expected the official MSG pipeline profile.')

for name in sorted(shared_manifest['files']):
    if name.startswith(('msgpt-source/src/', 'mist-source/src/', 'models/safe_gpt_base/')):
        asset(shared, shared_manifest, name)
checkpoint = asset(shared, shared_manifest, 'models/mlm/ms_gpt_mlm.pt')
mist_checkpoint = asset(profile, profile_manifest, 'mist/mist_official/mist_massspecgym.pt')
mlm_config = asset(shared, shared_manifest, 'msgpt-source/' + config['mlm_config'])

preference = {{tag: order for order, tag in enumerate(sys_tags())}}
supported = {{}}
for path in inputs.rglob('rdkit-2026.3.3-cp312-*.whl'):
    if 'rdkit-2026-3-3-wheel' not in path.parts:
        continue
    compatible = [preference[tag] for tag in parse_wheel_filename(path.name)[3] if tag in preference]
    if compatible:
        supported[path] = min(compatible)
if not supported:
    raise FileNotFoundError('No official RDKit wheel supports this CPU image.')
rdkit_wheel = min(supported, key=lambda path: (supported[path], str(path)))
shared_wheels = [asset(shared, shared_manifest, name)
                 for name in sorted(shared_manifest['files']) if name.startswith('wheels/')]
runtime_wheels = [checked(find(name, {RUNTIME!r}), checksum)
                  for name, checksum in {RUNTIME_WHEELS!r}.items()]
wheel_files = [rdkit_wheel, *shared_wheels, *runtime_wheels]
torchao_removed = remove_optional_torchao()

with TemporaryDirectory(prefix='msgpt-load-check-') as temporary:
    imports = Path(temporary) / 'python'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                    '--target', str(imports), *map(str, wheel_files)], check=True)
    sys.path[:0] = [str(imports), str(shared / 'msgpt-source/src'), str(shared / 'mist-source/src')]
    # Reuse the draft's alias only where the base image lacks the old package name.
    try:
        import pytorch_lightning
        lightning_provider = 'pytorch_lightning'
    except ImportError:
        import lightning.pytorch as pytorch_lightning
        sys.modules['pytorch_lightning'] = pytorch_lightning
        lightning_provider = 'lightning.pytorch'

    import torch
    from omegaconf import OmegaConf
    from peft import PeftModel
    from rdkit import rdBase
    from ms_gpt.data.mist_inputs import ION_INDEX, prepare_mist_batch
    from ms_gpt.data.mlm_pretraining_dataset import load_molecule_tokenizer
    from ms_gpt.external.mist_official.chem_utils import instruments
    from ms_gpt.models.conditional_mlm import ConditionalMoleculeLanguageModel, normalize_mlm_checkpoint_state_dict
    from ms_gpt.models.mist_official_encoder import MIST
    from ms_gpt.pipeline.postprocessing import CandidateRanker

    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError(f'Unexpected RDKit import: {{rdBase.rdkitVersion}}.')
    torch.set_num_threads(2)
    device = torch.device('cuda:0' if {use_gpu!r} else 'cpu')
    if {use_gpu!r} and (not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0)):
        raise ValueError('Use the pinned Kaggle NVIDIA T4 for this blind run.')
    mist = MIST(mist_checkpoint).to(device).eval()
    training_dataset = mist.release_config.get('data', {{}}).get('dataset_name')
    if training_dataset != 'msg_all':
        raise ValueError(f'Expected MSG spectrum training, found {{training_dataset!r}}.')
    tokenizer = load_molecule_tokenizer(str(shared / 'models' / 'safe_gpt_base'))
    cfg = OmegaConf.load(mlm_config)
    model = ConditionalMoleculeLanguageModel.from_config(cfg, len(tokenizer), dropout=0.0)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(normalize_mlm_checkpoint_state_dict(state), strict=True)
    del state
    model = PeftModel.from_pretrained(model, str(profile_adapter), is_trainable=False).merge_and_unload().to(device).eval()
    if next(mist.parameters()).device.type != device.type or next(model.parameters()).device.type != device.type:
        raise ValueError('MS-GPT parameters loaded to an unexpected device.')
    receipt = dict(
        status='strict_model_load_complete', device=str(device), python=sys.version,
        versions={{name: version(name) for name in
                  ('torch', 'transformers', 'tokenizers', 'lightning', 'torchmetrics', 'peft', 'omegaconf')}},
        lightning_import=lightning_provider,
        optional_torchao_removed_version=torchao_removed, lora_backend='dense',
        rdkit=rdBase.rdkitVersion, spectrum_training_dataset=training_dataset,
        mist_checkpoint_step=int(mist.checkpoint_step), fingerprint_dim=int(mist.model_config['output_size']),
        mist_embeds_instrument=bool(mist.model_config.get('embed_instrument', False)),
        supported_instrument_codes=list(instruments), supported_adduct_codes=dict(ION_INDEX),
        timstof_mapping_applied=False,
        mlm_parameter_count=sum(p.numel() for p in model.parameters()), tokenizer_size=len(tokenizer),
        loaded_weights={{'mist': str(mist_checkpoint.relative_to(profile)),
                         'mlm': str(checkpoint.relative_to(shared)),
                         'adapter': str(profile_adapter.relative_to(profile))}},
        source_refs=[{SHARED!r}, {PROFILE!r}, {METRIC!r}, {RUNTIME!r}],
        input_helpers=[prepare_mist_batch.__name__, CandidateRanker.__name__],
        seconds=monotonic() - start,
    )
'''
    return code + indent(after_load.strip() + '\n', '    ')


def build(out):
    code = model_runtime(after_load='''
(Path('/kaggle/working') / 'ms_gpt_load_receipt.json').write_text(json.dumps(receipt, indent=2) + '\\n')
print('MS-GPT LOAD', json.dumps(receipt), flush=True)
''')
    compile(code, SLUG, 'exec')
    notebook = dict(nbformat=4, nbformat_minor=5, cells=[
        dict(cell_type='code', id='ms-gpt-load-check', execution_count=None,
             metadata={}, outputs=[], source=code.strip().splitlines(keepends=True)),
    ], metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    metadata = dict(
        id=f'jiyujia/{SLUG}', title=SLUG, code_file='load_check.ipynb', language='python',
        kernel_type='notebook', is_private=True, enable_gpu=False, enable_internet=False,
        dataset_sources=[SHARED, PROFILE, METRIC], competition_sources=[], kernel_sources=[RUNTIME],
        docker_image=IMAGE,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / 'load_check.ipynb').write_text(json.dumps(notebook, separators=(',', ':')) + '\n')
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    build(parser.parse_args().out)

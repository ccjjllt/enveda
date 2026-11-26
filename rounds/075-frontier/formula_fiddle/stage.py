"""Pin and exercise the author's timsTOF FIDDLE v2.4.0 release on Kaggle CPU."""

from collections import Counter
from hashlib import file_digest
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from urllib.request import Request, urlopen
import warnings
from zipfile import ZipFile


VERSION = '2.4.0'
TAG = f'v{VERSION}'
COMMIT = '805e2b363a73e8c70ebba7d7dca05cf46eb5c67f'
CONFIG_SHA = '6a21a3b025bea31c32a32917aa5b14491e56e15b86b36d78390ddb876e23b8e4'
REPO = 'https://github.com/JosieHong/FIDDLE'
PUBLISHED = '2026-11-22T06:02:12Z'
ASSETS = {
    'fiddle_timstof.zip': (f'{REPO}/releases/download/{TAG}/fiddle_timstof.zip', 80596255,
                          'f338b6531c34637917e7dd573cc83c2c6a60b15ae5f79493f57f79ee1d2d8e3a'),
    'msfiddle-2.4.0-py3-none-any.whl': (
        'https://files.pythonhosted.org/packages/af/4d/3f40dd307121294d1af43676562f2f71506fff9ee4186e531e47af6a4603/'
        'msfiddle-2.4.0-py3-none-any.whl', 100109,
        '829d2576db9b10c285058466e4c7de708e5668040dce1f313c14c7bd86a88d9b'),
    'molmass-2026.11.1-py3-none-any.whl': (
        'https://files.pythonhosted.org/packages/e2/66/5bd0fd74d2eb2e98ea12650eb21302e65d1b4d4706452f27a1e7321330b1/'
        'molmass-2026.11.1-py3-none-any.whl', 79195,
        'f16ba27aea36bb7b9ace10b8d77847946e8b4c9a6a0622c80de9a95bb517ac77'),
    'pyteomics-5.0.2-py3-none-any.whl': (
        'https://files.pythonhosted.org/packages/b4/3d/75b1235336ca11e09869efe009cef62082610317841b20748c658a4791b2/'
        'pyteomics-5.0.2-py3-none-any.whl', 264035,
        'ecd4c846762e8ba71d562e58ff8a0f47c7d249d0f28a98648ee455c7121a5aba'),
    'FIDDLE-LICENSE.txt': (f'https://raw.githubusercontent.com/josiehong/FIDDLE/{COMMIT}/LICENSE', 11340,
                           '6ac436add23560dca3d67b1721c9a467249e47228c21e8f5a0726b114cc9ddde'),
}
CHECKPOINTS = ('fiddle_tcn_timstof.pt', 'fiddle_rescore_timstof.pt')
RECEIPT = 'fiddle_v240_timstof_stage.json'
VERSIONS = ('numpy', 'pandas', 'pyyaml', 'scikit-learn', 'scipy', 'pyarrow', 'rdkit', 'torch', 'tqdm',
            'msfiddle', 'molmass', 'pyteomics')


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def check_asset(path):
    _, size, sha = ASSETS[path.name]
    if path.stat().st_size != size or digest(path) != sha:
        raise ValueError(f'Release asset content mismatch: {path.name}')


def download(destination):
    url, _, _ = ASSETS[destination.name]
    request = Request(url, headers={'User-Agent': 'Enveda-FIDDLE-reproducibility/1.0'})
    candidate = destination.with_name(destination.name + '.part')
    for attempt in range(3):
        try:
            with urlopen(request, timeout=75) as response, candidate.open('wb') as stream:
                shutil.copyfileobj(response, stream, length=1 << 20)
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
        else:
            candidate.replace(destination)
            check_asset(destination)
            return


def load_bundle(source, scratch):
    """Initialize pinned release assets on an offline classifier or in the preflight."""
    for name in ASSETS:
        check_asset(source / name)
    wheels = [str(source / name) for name in ASSETS if name.endswith('.whl')]
    installed = scratch / 'packages'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index', '--disable-pip-version-check',
                    '--target', str(installed), *wheels], check=True)
    sys.path.insert(0, str(installed))
    package_root = installed / 'msfiddle'
    config = package_root / 'config' / 'fiddle_tcn_timstof.yml'
    if metadata.version('msfiddle') != VERSION or digest(config) != CONFIG_SHA:
        raise ValueError('Unpinned FIDDLE package or timsTOF configuration')
    checkpoint_dir = scratch / 'checkpoints'
    checkpoint_dir.mkdir()
    with ZipFile(source / 'fiddle_timstof.zip') as archive:
        for name in CHECKPOINTS:
            member, = (item for item in archive.infolist() if item.filename == name and not item.is_dir())
            with archive.open(member) as origin, (checkpoint_dir / name).open('wb') as output:
                shutil.copyfileobj(origin, output, length=1 << 20)
    return package_root, config, checkpoint_dir


def make_predictor(config, checkpoint_dir):
    from msfiddle import MsFiddlePredictor

    return MsFiddlePredictor(instrument_type='timstof', device='cpu', no_cuda=True, config_path=config,
                             resume_path=checkpoint_dir / CHECKPOINTS[0],
                             rescore_resume_path=checkpoint_dir / CHECKPOINTS[1],
                             download_models=False, batch_size=16)


def reference_examples(predictor, package_root):
    """Use measured negative-ion examples shipped in the author wheel."""
    from pyteomics import mgf

    demo = package_root / 'demo' / 'input_msms.mgf'
    status = predictor.validate_mgf(demo)
    accepted = set(status.loc[status['status'].eq('ok'), 'id'])
    selected = []
    with mgf.MGF(str(demo), convert_arrays=1) as spectra:
        for spectrum in spectra:
            params = spectrum['params']
            native_id = params['title']
            adduct = params['precursor_type']
            if native_id not in accepted or adduct != '[M-H]-':
                continue
            selected.append(dict(id=native_id, mz_array=spectrum['m/z array'],
                                 intensity_array=spectrum['intensity array'],
                                 precursor_mz=float(params['precursor_mz']), adduct=adduct,
                                 collision_energy=params['collision_energy']))
            if len(selected) == 3:
                break
    if len(selected) != 3:
        raise ValueError('Official demo did not provide three accepted [M-H]- examples')
    return selected, dict(Counter(status['status'])), digest(demo)


def audit_predictions(predictor, examples):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        rows = predictor.predict_batch(examples, top_k=5)
    if any('different TCN' in str(item.message) for item in captured):
        raise ValueError('Author checkpoint pair has incompatible model hashes')
    if str(predictor.device) != 'cpu' or len(rows) != len(examples):
        raise ValueError('The timsTOF demo was not evaluated entirely on CPU')
    trace = []
    for source, row in zip(examples, rows, strict=True):
        candidates = row['candidates']
        observed = row['metadata']
        if row['id'] != source['id'] or observed['instrument_type'] != 'timstof':
            raise ValueError('Invalid author timsTOF prediction')
        raw_scores = []
        for item in candidates:
            score = item['metadata']['raw_rescore']
            if (not isinstance(item['formula'], str) or not item['formula'] or not math.isfinite(score)
                    or not math.isfinite(item['mass']) or item['score'] != min(1.0, max(0.0, score))):
                raise ValueError('Unexpected author formula candidate output')
            raw_scores.append(score)
        if raw_scores != sorted(raw_scores, reverse=True):
            raise ValueError('Author formula candidate order disagrees with its raw scores')
        trace.append(dict(id=row['id'], precursor_mz=source['precursor_mz'], adduct_input=source['adduct'],
                          adduct_encoded=observed['adduct'], ce_input=source['collision_energy'],
                          ce_encoded=observed['collision_energy'],
                          candidate_count=len(candidates), first_formula=candidates[0]['formula'] if candidates else None,
                          first_raw_rescore=raw_scores[0] if raw_scores else None))
    if not any(row['candidate_count'] for row in trace):
        raise ValueError('Author inference produced no candidate for any demo spectrum')
    return trace


def run(output):
    import torch

    torch.set_num_threads(2)
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='fiddle-v240-stage-') as directory:
        scratch = Path(directory)
        source = scratch / 'assets'
        source.mkdir()
        for name in ASSETS:
            print(f'Fetching pinned public release asset {name}', flush=True)
            download(source / name)
        package, config, checkpoints = load_bundle(source, scratch)
        predictor = make_predictor(config, checkpoints)
        examples, validation_counts, demo_sha = reference_examples(predictor, package)
        predictions = audit_predictions(predictor, examples)
        files = {}
        for name, (url, size, sha) in ASSETS.items():
            shutil.copyfile(source / name, output / name)
            files[name] = dict(source=url, bytes=size, sha256=sha)
        receipt = dict(status='sealed', upstream_commit=COMMIT, release_tag=TAG, published_utc=PUBLISHED,
                       code_license='Apache-2.0 (author repository LICENSE)', device='cpu', instrument='timstof',
                       config_sha256=CONFIG_SHA, checkpoints={name: dict(bytes=(checkpoints / name).stat().st_size,
                           sha256=digest(checkpoints / name)) for name in CHECKPOINTS}, files=files,
                       runtime_versions={dep: metadata.version(dep) for dep in VERSIONS},
                       call_contract=dict(inputs=['id', 'mz_array', 'intensity_array', 'precursor_mz', 'adduct',
                                                  'collision_energy'], score='candidates[].metadata.raw_rescore',
                                          observed_ce='Explicit measured numeric eV or unit-tagged string, observed '
                                          'stepped eV lists passed as comma-separated values (author encodes mean)',
                                          reject='Skip unsupported adducts/peaks or unspecified collision energy'),
                       demo_only=dict(source='Upstream redistributable smoke example', source_sha256=demo_sha,
                                      validation_status_counts=validation_counts, predictions=predictions,
                                      scientific_measurement=False), elapsed_seconds=round(time.monotonic()-started, 2))
        (output / RECEIPT).write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print('FIDDLE CPU Stage Sealed', json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run(Path('/kaggle/working'))

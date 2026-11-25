"""Prepare checksum-verified MSAlign inference assets in an online CPU notebook."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from tempfile import TemporaryDirectory
import time
import urllib.error
import urllib.request
import zipfile

import yaml


MSALIGN_SOURCE = 'https://github.com/KrzakalaPaul/MSAlign-NeurIPS2026.git'
MSALIGN_REVISION = '97e201b1dd61e792d9086815192f7518a0b62ed8'
RELEASE = 'msalign_star_massspecgym_mces_1'
RELEASE_URL = f'https://zenodo.org/api/records/23372807/files/{RELEASE}.zip/content'
RELEASE_MD5 = '8a096ae82cd1f856684df689ca2478fd'
RELEASE_BYTES = 288157301
DREAMS_NAME = 'DreaMS_embedding_model_torchscript.pt'
DREAMS_REVISION = 'c81a62766b10dd1d39fcda3edec5ef88623e5f6b'
DREAMS_URL = f'https://huggingface.co/roman-bushuiev/DreaMS/resolve/{DREAMS_REVISION}/{DREAMS_NAME}'
DREAMS_SHA256 = 'df69866990b194b27d5497e440f87f6ceca568ee512e7c22cb1f278721376f0c'
DREAMS_BYTES = 468475544
CHANNELS = {'bins_fingerprint': 'bins', 'dreams_fingerprint': 'dreams'}
SOURCE_FILES = (
    'LICENSE', 'models/__init__.py', 'models/MSAlign/model.py',
    'models/MSAlign/encoders.py', 'models/MSAlign/losses.py', 'models/MSAlign/modules.py',
    'playground/__init__.py', 'playground/bundle.py', 'playground/inference.py',
)


def digest(path, algorithm='sha256'):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def download(url, target, algorithm, expected, size):
    temporary = target.with_suffix('.part')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=90) as source, temporary.open('wb') as out:
                shutil.copyfileobj(source, out, 1024 * 1024)
            if temporary.stat().st_size != size or digest(temporary, algorithm) != expected:
                raise ValueError(f'Official download size or digest differs for {target.name}')
            temporary.replace(target)
            return
        except (OSError, urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(3 * (1 + attempt))


def write_source(scratch, output, git_dir=None):
    if git_dir is None:
        git_dir = scratch / 'source'
        git_dir.mkdir()
        subprocess.run(['git', 'init', '--quiet', str(git_dir)], check=True)
        subprocess.run(['git', '-C', str(git_dir), 'fetch', '--depth=1',
                        MSALIGN_SOURCE, MSALIGN_REVISION], check=True)
    revision = subprocess.check_output(['git', '-C', str(git_dir), 'rev-parse', MSALIGN_REVISION],
                                       text=True).strip()
    if revision != MSALIGN_REVISION:
        raise ValueError('MSAlign source revision changed')
    archive = scratch / 'source.tar'
    subprocess.run(['git', '-C', str(git_dir), 'archive', '--output', str(archive),
                    revision, *SOURCE_FILES], check=True)
    with tarfile.open(archive) as source:
        for name in SOURCE_FILES:
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('wb') as stream:
                shutil.copyfileobj(source.extractfile(name), stream)

    # Keep the official inference math and weights. Strip only Lightning's training
    # wrapper and training re-exports so the adapter uses Kaggle's native Torch stack.
    module = output / 'models/MSAlign/model.py'
    contents = module.read_text()
    hyperparams = (
        '        self.save_hyperparameters(\n'
        '            {"config": dict(config), "input_dimensions": dict(input_dimensions)}\n'
        '        )\n'
    )
    edits = {'import lightning.pytorch as pl\n': '',
             'class MSAlign(pl.LightningModule):': 'class MSAlign(nn.Module):',
             hyperparams: ''}
    for original, replacement in edits.items():
        if contents.count(original) != 1:
            raise ValueError('Pinned model source no longer matches its inference adapter')
        contents = contents.replace(original, replacement, 1)
    module.write_text(contents)
    (module.parent / '__init__.py').write_text('"""MSAlign inference modules."""\n')


def extract_morgan_channels(archive, output, channels):
    if archive.stat().st_size != RELEASE_BYTES or digest(archive, 'md5') != RELEASE_MD5:
        raise ValueError('MSAlign archive differs from the official Zenodo release')
    with zipfile.ZipFile(archive) as source:
        official = yaml.safe_load(source.read(f'{RELEASE}/manifest.yaml'))
        provenance = official['provenance']
        if (official['format_version'] != 1 or official['model_type'] != 'fusion'
                or provenance['dataset'] != 'massspecgym' or provenance['split'] != 'mces_1'):
            raise ValueError('Unexpected official checkpoint split or format')
        output.mkdir(parents=True, exist_ok=True)
        (output / 'official_manifest.yaml').write_bytes(source.read(f'{RELEASE}/manifest.yaml'))
        entries = {entry['name']: entry for entry in official['scorers']}
        for channel in channels:
            entry = entries[channel]
            if entry['type'] != 'msalign':
                raise ValueError(f'Unexpected scorer type for {channel}')
            for role, folder, suffix, hash_key in (
                ('checkpoint', 'checkpoints', '.ckpt', 'sha256'),
                ('config', 'configs', '.yaml', 'config_sha256'),
            ):
                relative = f'{folder}/{channel}{suffix}'
                if entry[role] != relative:
                    raise ValueError(f'Unexpected release path for {channel}')
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                with source.open(f'{RELEASE}/{relative}') as contents, path.open('wb') as stream:
                    shutil.copyfileobj(contents, stream)
                if digest(path) != entry[hash_key]:
                    raise ValueError(f'Official per-file checksum mismatch for {relative}')
            config = yaml.safe_load((output / entry['config']).read_text())
            expected = {'spectrum': CHANNELS[channel], 'molecule': 'morgan_2_4096'}
            if config['representations'] != expected:
                raise ValueError(f'Unexpected representations in {channel}: {config["representations"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('/kaggle/working/msalign_assets'))
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--with-dreams', action='store_true')
    args = parser.parse_args()
    channels = tuple(CHANNELS) if args.with_dreams else ('bins_fingerprint',)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as directory:
        scratch = Path(directory)
        staging = scratch / 'msalign_assets'
        staging.mkdir()
        write_source(scratch, staging / 'source')
        archive = args.archive
        if archive is None:
            archive = scratch / f'{RELEASE}.zip'
            download(RELEASE_URL, archive, 'md5', RELEASE_MD5, RELEASE_BYTES)
        extract_morgan_channels(archive, staging, channels)
        if args.with_dreams:
            download(DREAMS_URL, staging / DREAMS_NAME, 'sha256', DREAMS_SHA256, DREAMS_BYTES)
        files = {str(path.relative_to(staging)): digest(path) for path in sorted(staging.rglob('*'))
                 if path.is_file()}
        receipt = dict(channels=channels, source_revision=MSALIGN_REVISION, source_license='MIT',
                       checkpoint_url=RELEASE_URL, checkpoint_md5=RELEASE_MD5,
                       checkpoint_license='CC BY 4.0',
                       dreams_url=DREAMS_URL if args.with_dreams else None,
                       dreams_sha256=DREAMS_SHA256 if args.with_dreams else None, files=files)
        (staging / 'asset_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        if args.out.exists():
            raise FileExistsError(f'Output exists: {args.out}')
        staging.replace(args.out)
    print(f'Assets ready for {", ".join(channels)} at {args.out}', flush=True)


if __name__ == '__main__':
    main()

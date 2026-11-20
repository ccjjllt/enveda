"""Download the released model assets on Kaggle for subsequent offline inference."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import os
import subprocess
import sys


OUTPUT = Path('/kaggle/working')
FLARE_CODE = 'e7dfae2595b90703394527d985b5fab4b6a36806'
MIST_CODE = '48e339be6535546c87ba4f6b81a0f787c2e4ac40'


def download(filename, url, expected_sha256):
    path = OUTPUT / filename
    subprocess.run(['curl', '--fail', '--location', '--retry', '3', '--silent',
                    '--show-error', '--output', str(path), url], check=True)
    with path.open('rb') as data:
        actual = hashlib.file_digest(data, 'sha256').hexdigest()
    if actual != expected_sha256:
        raise ValueError(f'{filename} differs from the authors’ published SHA-256')
    print(filename, path.stat().st_size, flush=True)


def source(filename, url, revision, paths):
    checkout = Path('/tmp') / filename.removesuffix('.tgz')
    checkout.mkdir()
    env = {**os.environ, 'GIT_LFS_SKIP_SMUDGE': '1'}
    subprocess.run(['git', 'init', '-q'], cwd=checkout, check=True)
    subprocess.run(['git', 'fetch', '-q', '--depth', '1', url, revision],
                   cwd=checkout, env=env, check=True)
    with (OUTPUT / filename).open('wb') as archive:
        subprocess.run(['git', 'archive', '--format=tar.gz', 'FETCH_HEAD', *paths],
                       cwd=checkout, stdout=archive, check=True)


def wheels():
    print(sys.version, flush=True)
    path = OUTPUT / 'wheels'
    path.mkdir()
    subprocess.run([sys.executable, '-m', 'pip', 'download', '--no-deps',
                    '--only-binary=:all:', '-d', str(path), 'torch_geometric==2.6.1'], check=True)


def main():
    with ThreadPoolExecutor() as pool:
        jobs = [
            pool.submit(download, 'flare.ckpt', f'https://huggingface.co/spaces/HassounLab/FLARE/resolve/{FLARE_CODE}/pretrained_models/flare.ckpt',
                        '78d0be3cc488b2cccaa39cdcdf4fe1abe05aa22a76007f5ba7095a72fa93b9e0'),
            pool.submit(download, 'mist_msg_noaug.ckpt', 'https://zenodo.org/records/23303289/files/mist_msg_noaug.ckpt',
                        '6f8df582df36aa4a984f120efee14ee0a62f3a7723caa770b856a6e704236727'),
            pool.submit(source, 'flare.tgz', 'https://huggingface.co/spaces/HassounLab/FLARE',
                        FLARE_CODE, ['flare', 'massspecgym', 'README.md']),
            pool.submit(source, 'mist.tgz', 'https://github.com/coleygroup/MIST.git',
                        MIST_CODE, ['src', 'LICENSE.md', 'NOTICE.md', 'LICENSES']),
            pool.submit(wheels),
        ]
        for job in jobs:
            job.result()


if __name__ == '__main__':
    main()

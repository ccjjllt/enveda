"""Stage a public Linux CFM-ID 4 binary, libraries, and stock negative weights."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request


ROOT = Path('/tmp/enveda_cfm4_stage')
PREFIX = Path('/tmp/enveda_cfm4_runtime')
OUTPUT = Path('/kaggle/working')
MAMBA = ('https://conda.anaconda.org/conda-forge/linux-64/micromamba-2.10.0-0.tar.bz2',
         '9a97976244929bbae28fb1fae504e74a3e7e6d1932870e3a03321f01b7fbd07b')
BUILD = 'cfm-id=4.4.10=h8471819_0'
PACKAGE_SHA256 = 'a0c114eb94dbfa89ce1687f594202ae9773d3d65e4f8d6eca696f7ff2c3152b9'


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    manager = ROOT / 'micromamba.tar.bz2'
    with urllib.request.urlopen(MAMBA[0], timeout=60) as response, manager.open('wb') as handle:
        shutil.copyfileobj(response, handle)
    if sha256(manager) != MAMBA[1]:
        raise RuntimeError('Micromamba archive checksum mismatch')
    executable = ROOT / 'micromamba'
    with tarfile.open(manager) as archive:
        with archive.extractfile('bin/micromamba') as source, executable.open('wb') as handle:
            shutil.copyfileobj(source, handle)
    executable.chmod(0o755)
    env = {**os.environ, 'MAMBA_ROOT_PREFIX': str(ROOT / 'mamba')}
    subprocess.run([str(executable), '--no-rc', 'create', '-y', '-p', str(PREFIX), '--strict-channel-priority',
                    '-c', 'https://conda.anaconda.org/conda-forge', '-c', 'https://conda.anaconda.org/bioconda',
                    BUILD], check=True, env=env, timeout=900)
    packages = [json.loads(path.read_text()) for path in sorted((PREFIX / 'conda-meta').glob('*.json'))]
    cfm, = [entry for entry in packages if entry['name'] == 'cfm-id']
    if cfm['sha256'] != PACKAGE_SHA256:
        raise RuntimeError('Installed CFM-ID release package checksum mismatch')
    models = PREFIX / 'share/cfm-id-4.4.10/cfmid4'
    params, = [p for p in models.rglob('param_output.log') if '[M-H]-' in p.parts]
    config = params.with_name('param_config.txt')
    if not config.is_file():
        raise FileNotFoundError(config)
    cli = PREFIX / 'bin/cfm-predict'
    dependencies = subprocess.check_output(['ldd', str(cli)], text=True,
                                           env={**os.environ, 'LD_LIBRARY_PATH': str(PREFIX / 'lib')})
    if 'not found' in dependencies:
        raise RuntimeError(dependencies)
    archive = OUTPUT / 'cfm4_runtime.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=2) as bundle:
        bundle.add(PREFIX, arcname=PREFIX.name)
    receipt = dict(binary=str(cli.relative_to(PREFIX)), params=str(params.relative_to(PREFIX)),
                   config=str(config.relative_to(PREFIX)), binary_sha256=sha256(cli),
                   params_sha256=sha256(params), config_sha256=sha256(config),
                   archive_sha256=sha256(archive), archive_bytes=archive.stat().st_size,
                   source='https://bitbucket.org/wishartlab/cfm-id-code/src/cced92452e0f230acacc0427d19908b266811dc9/',
                   package=cfm['url'], package_sha256=cfm['sha256'], source_license='LGPL-2.1-only',
                   packages=[{key: entry.get(key) for key in ('name', 'version', 'build', 'url', 'sha256')}
                             for entry in packages])
    (OUTPUT / 'cfm4_runtime.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({key: value for key, value in receipt.items() if key != 'packages'}), flush=True)


if __name__ == '__main__':
    main()

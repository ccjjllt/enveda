"""Build the public-asset-only FIDDLE staging notebook for private Kaggle CPU."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from runpy import run_path


HERE = Path(__file__).resolve().parent
SLUG = 'enveda-075-fiddle-v240-timstof-stage'
IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'


def build(destination):
    common = run_path(str(HERE.parent / 'virtual_np' / 'build.py'))
    destination.mkdir(parents=True, exist_ok=True)
    source = (HERE / 'stage.py').read_text()
    entrypoint = "if __name__ == '__main__':\n    run(Path('/kaggle/working'))\n"
    if not source.endswith(entrypoint):
        raise ValueError('Expected the author asset stage entry point')
    bootstrap = "inputs = Path('/kaggle/input')\n" + common['METRIC_BOOTSTRAP']
    source = (source.removesuffix(entrypoint) + bootstrap + '\n' + entrypoint).encode()
    compile(source, 'stage.py', 'exec')
    (destination / 'stage.py').write_bytes(source)
    metadata = dict(id=f'jiyujia/{SLUG}', title=SLUG, language='python', kernel_type='script', code_file='stage.py',
                    is_private=True, enable_gpu=False, enable_internet=True, docker_image=IMAGE,
                    dataset_sources=[common['WHEEL']], kernel_sources=[], competition_sources=[])
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return sha256(source).hexdigest()


if __name__ == '__main__':
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument('--out', type=Path, required=True)
    print(build(arguments.parse_args().out))

"""Package the pinned NP-MRD data staging script for Kaggle CPU."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from build import IMAGE


HERE = Path(__file__).resolve().parent
SLUG = 'enveda-075-npmrd-2025-source'
PROGRAM = 'stage_npmrd_2025.py'


def build(out):
    source = (HERE / PROGRAM).read_bytes()
    compile(source, PROGRAM, 'exec')
    out.mkdir(parents=True, exist_ok=True)
    (out / PROGRAM).write_bytes(source)
    metadata = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file=PROGRAM, language='python',
                    kernel_type='script', is_private=True, enable_gpu=False, enable_internet=True,
                    dataset_sources=[], kernel_sources=[], competition_sources=[], docker_image=IMAGE)
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return sha256(source).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    print(build(parser.parse_args().out))

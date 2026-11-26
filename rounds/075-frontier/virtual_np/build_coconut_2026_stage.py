"""Package the pinned public COCONUT download for a private Kaggle CPU run."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SLUGS = {'lite': 'enveda-075-coconut-2026-source', 'full': 'enveda-075-coconut-2026-full-source'}
PROGRAM = 'stage_coconut_2026.py'


def build(out, variant='lite'):
    from build import IMAGE

    source = (HERE / PROGRAM).read_bytes()
    selector = b"DEFAULT_VARIANT = 'lite'"
    if source.count(selector) != 1:
        raise ValueError('The stage variant binding has changed')
    source = source.replace(selector, f'DEFAULT_VARIANT = {variant!r}'.encode(), 1)
    compile(source, PROGRAM, 'exec')
    out.mkdir(parents=True, exist_ok=True)
    (out / PROGRAM).write_bytes(source)
    slug = SLUGS[variant]
    metadata = dict(id=f'jiyujia/{slug}', title=slug, code_file=PROGRAM, language='python',
                    kernel_type='script', is_private=True, enable_gpu=False, enable_internet=True,
                    dataset_sources=[], kernel_sources=[], competition_sources=[], docker_image=IMAGE)
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return sha256(source).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--variant', choices=SLUGS, default='lite')
    print(build(**vars(parser.parse_args())))

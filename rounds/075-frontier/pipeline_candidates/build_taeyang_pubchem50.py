"""Package the public Taeyang v2 notebook with the PubChem depth-50 change."""

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path

from build_kernel import literal_value, source, update
from build_pubchem50 import extend_pubchem
from pubchem50_priority import prioritize_original_predictions


ORIGINAL_ID = 'taeyangg4/casmi26-lb-0-495-network-fr-p5-tail'
ORIGINAL_FROZEN_SHA = 'd1e757c2b71b46e6e8a5777f14302a4737d0fdbcd56d8365093c039b93c5fd88'
SLUG = 'enveda-075b-taeyang-pubchem50'


def build(original, destination):
    metadata = json.loads((original / 'kernel-metadata.json').read_text())
    if metadata['id'] != ORIGINAL_ID or destination.resolve() == original.resolve():
        raise ValueError('Use the original Taeyang v2 pull and a separate destination')

    notebook = json.loads((original / metadata['code_file']).read_text())
    pinned = source(notebook, 13)
    encoded = literal_value(pinned, 'FROZEN_B64')
    original_bytes = base64.b64decode(encoded, validate=True)
    if sha256(original_bytes).hexdigest() != ORIGINAL_FROZEN_SHA:
        raise ValueError('The embedded source differs from the verified Taeyang v2')

    frozen = json.loads(original_bytes)
    extend_pubchem(frozen)
    prioritize_original_predictions(frozen)
    updated = json.dumps(frozen, ensure_ascii=False).encode()
    update(notebook, 13, encoded, base64.b64encode(updated).decode())
    update(notebook, 13, ORIGINAL_FROZEN_SHA, sha256(updated).hexdigest())
    update(notebook, 6, "COMMIT_MODE = 'fast'", "COMMIT_MODE = 'smoke'")
    # P5 recomputes the full Engine 1 list only during the hidden rerun.
    update(notebook, 37, "    if globals().get('PIPELINE_DONE'):\n",
           "    if globals().get('PIPELINE_DONE') and globals().get('IS_RERUN'):\n")

    for label, book in (('frozen', frozen), ('outer', notebook)):
        for index, cell in enumerate(book['cells']):
            if cell['cell_type'] == 'code':
                compile(source(book, index), f'{label}_cell_{index}', 'exec')

    metadata.pop('id_no', None)
    metadata.update(id=f'jiyujia/{SLUG}', title=SLUG, code_file=f'{SLUG}.ipynb', is_private=True)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / metadata['code_file']).write_text(json.dumps(notebook, ensure_ascii=False) + '\n')
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original', type=Path, help='Directory containing the original v2 Kaggle pull')
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    build(args.original, args.destination)

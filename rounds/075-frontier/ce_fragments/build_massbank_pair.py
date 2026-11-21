"""Package the original ICEBERG and CE fragment functions with the fixed leader comparison."""

import argparse
import ast
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SLUG = 'enveda-075b-massbank-ce-pair'


def build(destination):
    functions = ('load_iceberg', 'clean_peaks', 'match_peaks', 'cosine', 'differences',
                 'measured_queries', 'measurement_groups', 'artifact', 'summaries')
    helpers = {}
    for path in (HERE / 'score.py', HERE.parent / 'pipeline_candidates' / 'pubchem_forward.py'):
        code = path.read_text()
        helpers.update((node.name, ast.get_source_segment(code, node)) for node in ast.parse(code).body
                       if isinstance(node, ast.FunctionDef) and node.name in functions)
    script = '\n\n'.join([*(helpers[name] for name in functions), (HERE / 'massbank_pair.py').read_text()])
    compile(script, 'massbank_pair.py', 'exec')
    metadata = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='massbank_pair.py', language='python',
                    kernel_type='script', is_private=True, enable_gpu=False, enable_internet=False,
                    dataset_sources=['ahmedberatozer/casmi26-iceberg', 'alexchilton/rdkit-2025-3-6-cp313'],
                    kernel_sources=['jiyujia/enveda-075b-external-massbank-202611/1',
                                    'jiyujia/enveda-075b-massbank-pipeline/1'],
                    competition_sources=[], model_sources=[])
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'massbank_pair.py').write_text(script)
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    build(parser.parse_args().destination)

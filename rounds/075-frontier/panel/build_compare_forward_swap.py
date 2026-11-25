"""Pack the read-only cross-model comparison on the two versioned feature exports."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
from runpy import run_path


HERE = Path(__file__).resolve().parent


def build(out, query_loss=False):
    shared = (HERE/'fit_dual_energy_pairs.py').read_text()
    definitions = {node.name: ast.get_source_segment(shared, node) for node in ast.parse(shared).body
                   if isinstance(node, ast.FunctionDef) and node.name in ('evaluate', 'fit')}
    helper = '\n\n\n'.join(
        definitions[name] for name in ('fit', 'evaluate'))
    source = (HERE/'compare_forward_swap.py').read_text()
    marker = 'from fit_dual_energy_pairs import evaluate, fit'
    if source.count(marker) != 1:
        raise ValueError('The paired comparison source import has changed')
    source = source.replace(marker, helper)
    slug = 'enveda-075-forward-hard-pair-blend'
    if query_loss:
        original = 'QUERY_LOSS = False'
        if source.count(original) != 1:
            raise ValueError('The query-loss setting has changed')
        source = source.replace(original,'QUERY_LOSS = True')
        slug += '-query-loss'
    compile(source, 'compare_forward_swap.py', 'exec')
    metadata = dict(id=f'jiyujia/{slug}', title=slug, code_file='compare_forward_swap.py', language='python',
                    kernel_type='script', is_private=True, enable_gpu=False, enable_internet=False,
                    docker_image=run_path(str(HERE.parent/'virtual_np'/'build.py'))['IMAGE'],
                    competition_sources=[], dataset_sources=[],
                    kernel_sources=['jiyujia/enveda-075-ice-forward-hard-pair/1',
                                    'jiyujia/enveda-075-gl-forward-hard-pair-cpu/1'])
    out.mkdir(parents=True, exist_ok=True)
    (out/metadata['code_file']).write_text(source)
    (out/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(sha256(source.encode()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--query-loss',action='store_true')
    build(**vars(parser.parse_args()))

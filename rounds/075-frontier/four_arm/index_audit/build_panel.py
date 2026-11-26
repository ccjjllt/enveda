"""Build the approved private CPU evaluation of post-seal index coverage."""

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
from runpy import run_path
from textwrap import indent
import zlib

from build import CONTEST, FRONTIER, HERE, POPULARITY, V4G, WHEEL, sections


SLUG = 'enveda-075-np-native-index-coverage'
EFFECTIVE_SHA = '1b1431a7ee13652fc037781dc05aa5284a0e52292ee57d5fa531764eb1511531'
SOURCE_PC_SHA = 'd7b28154b1bdf132528300bcf8faa1211984c843d5340ac81191472a452f564b'
METRIC_BOOTSTRAP = run_path(str(FRONTIER / 'virtual_np/build.py'))['METRIC_BOOTSTRAP']


def packed(path):
    data = path.read_bytes()
    return base64.b64encode(zlib.compress(data, 9)).decode(), sha256(data).hexdigest()


def build(panel, archive, output, baseline_kernel):
    manifest = json.loads((panel / 'panel_manifest.json').read_text())
    panel_files = {'queries': 'blind/test.parquet', 'labels': 'private/labels.parquet'}
    inputs = {name: packed(panel / path) for name, path in panel_files.items()}
    if any(inputs[name][1] != manifest['output_sha256'][path] for name, path in panel_files.items()):
        raise ValueError('Panel input hashes do not match the frozen manifest')
    inputs['effective_queries'] = packed(archive / 'effective_query_spectra.parquet')
    inputs['pc_targets'] = packed(archive / 'native_pc_targets.json')
    if inputs['effective_queries'][1] != EFFECTIVE_SHA:
        raise ValueError('The archived v2 Engine 1 spectra have changed')

    prerequisites = HERE / 'index_inputs.source'
    helpers = sections(prerequisites, {'find'})
    _, import_line, body = sections(prerequisites, {'chemistry'}).partition('    from rdkit import Chem, RDLogger, rdBase')
    if not import_line:
        raise ValueError('The shared chemistry runtime moved its RDKit import')
    helpers += ('\n\ndef chemistry():\n    import subprocess\n    import sys\n\n    inputs = ROOT\n'
                + indent(METRIC_BOOTSTRAP.strip(), '    ') + '\n' + import_line + body)
    helpers += '\n\n' + sections(prerequisites, {'POPULARITY', 'index_files'})
    raw, _, _ = (HERE / 'audit.py').read_text().rpartition("\nif __name__=='__main__':")
    if not raw:
        raise ValueError('The shared index audit runtime moved its entry point')
    prefix = (f'PANEL_INPUTS = {inputs!r}\nORIGINAL_PC_LISTS_SHA = {SOURCE_PC_SHA!r}\n'
              f'GENERAL_PROFILE_SOURCE = {(HERE / "fastscan_profiles.cpp").read_text()!r}\n')
    code = '\n\n'.join([prefix, helpers, raw, (HERE / 'coverage_panel.source').read_text()])
    compile(code, SLUG, 'exec')
    metadata = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='coverage.py', language='python',
                    kernel_type='script', is_private=True, enable_gpu=False, enable_internet=False,
                    dataset_sources=[WHEEL, POPULARITY, V4G], kernel_sources=[baseline_kernel],
                    competition_sources=[CONTEST])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'coverage.py').write_text(code)
    (output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('panel', 'archive', 'output'):
        parser.add_argument('--' + arg, type=Path, required=True)
    parser.add_argument('--baseline-kernel', default='jiyujia/enveda-075-np-recall-reseal/1')
    build(**vars(parser.parse_args()))

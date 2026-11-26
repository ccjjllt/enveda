"""Package the independent MassBank post-seal feasibility re-ranking as a blind CPU job."""

import argparse
from hashlib import sha256
from inspect import getsource
import json
from pathlib import Path

from build import HERE, IMAGE, METRIC_BOOTSTRAP, WHEEL, pack
from build_final_native_outsider_np60 import replace_at_rank


SLUG = 'enveda-075-final-native-outsider-mb110'
BASE = 'jiyujia/enveda-075b-massbank-pipeline/1'
DEPTH = 'jiyujia/enveda-075b-massbank-pubchem-depth/2'


def build(out):
    from ast import FunctionDef, parse

    top_five = (HERE / 'final_native_top5.py').read_text()
    lines = top_five.splitlines()
    node, = (item for item in parse(top_five).body if isinstance(item, FunctionDef) and item.name == 'formula')
    formula = '\n'.join(lines[node.decorator_list[0].lineno - 1:node.end_lineno])
    rule = ('from functools import lru_cache\nfrom rdkit import Chem\n'
            'from rdkit.Chem.rdMolDescriptors import CalcMolFormula\n\n'
            + formula + '\n\n' + getsource(replace_at_rank))
    rule_sha = sha256(rule.encode()).hexdigest()
    payload = {}
    modules = {'outsider_rule.py': rule,
               'c39_request_cache.py': (HERE / 'c39_request_cache.py').read_text(),
               'final_native_outsider_mb110.py': (HERE / 'final_native_outsider_mb110.py').read_text()}
    for name, source in modules.items():
        compile(source, name, 'exec')
        payload[name] = pack(source.encode())
    code = f'''
import base64
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib

inputs = Path('/kaggle/input')
{METRIC_BOOTSTRAP}
with tempfile.TemporaryDirectory(prefix='massbank110-final-outsider-') as directory:
    scratch = Path(directory)
    for name, (blob, expected) in {payload!r}.items():
        raw = zlib.decompress(base64.b64decode(blob))
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f'Embedded source changed: {{name}}')
        (scratch / name).write_bytes(raw)
    sys.path.insert(0, directory)
    from final_native_outsider_mb110 import run
    run(inputs, Path('/kaggle/working'), {rule_sha!r})
'''
    compile(code, SLUG, 'exec')
    meta = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='run_outsider.py', language='python',
                kernel_type='script', is_private=True, enable_gpu=False, enable_internet=False,
                docker_image=IMAGE, dataset_sources=[WHEEL], kernel_sources=[BASE, DEPTH], competition_sources=[])
    out.mkdir(parents=True, exist_ok=True)
    (out / meta['code_file']).write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return sha256(code.encode()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    print(build(**vars(parser.parse_args())))

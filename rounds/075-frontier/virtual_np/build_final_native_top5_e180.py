"""Freeze the original 60-panel swap rule and the six original model assets for the 121-panel CPU replay."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from runpy import run_path


HERE = Path(__file__).resolve().parent
COMMON = run_path(str(HERE/'build.py'))
RULE_SHA = '2a3b4ee7c93355eed8835ac3755a3d045601646e4ca80a58dd71b30368fcec8b'
SLUG = 'enveda-075-final-native-top5-e180'
REFERENCE = 'jiyujia/enveda-075-first-formula-baseline-e180/2'


def frozen_module():
    rule = (HERE/'final_native_top5.py').read_bytes()
    if sha256(rule).hexdigest()!=RULE_SHA:
        raise ValueError('The frozen rule used for the 60-case blind seal has changed')
    code = rule.decode()
    start = code.index("    outputs = {'historical':ranks, 'protected':{}, 'open':{}}")
    end = code.index('    out.mkdir(exist_ok=True, parents=True)', start)
    # Reuse the full scoring loop verbatim, including the library gate, request check and the finite-score barriers.
    shared = code[:code.index('def run(snapshot, baseline, out):')]
    shared += 'def apply(ranks, ledgers, request, raw_scores, budget):\n' + code[start:end] + '    return outputs, cases\n'
    return shared


def build(native, out):
    shared = frozen_module()
    names = [f'{m}_work/{m}{suffix}' for m in ('ice','gl')
             for suffix in ('_in.json','_out.json','_out.json.meta.json')]
    hashes = {name:sha256((native/name).read_bytes()).hexdigest() for name in names}
    adapter = (HERE/'final_native_top5_e180.py').read_text()
    compiled = {}
    for name,src in [('frozen_top5.py',shared),('e180_final.py',adapter)]:
        compile(src,name,'exec')
        compiled[name] = COMMON['pack'](src.encode())
    runtime = f'''
import base64
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib
inputs = Path('/kaggle/input')
{COMMON['METRIC_BOOTSTRAP']}
receipt, = (p for p in inputs.rglob('e180_ranked_input.json') if {REFERENCE.split('/')[1]!r} in p.parts)
with tempfile.TemporaryDirectory(prefix='e180-final-top5-') as temp:
    temp = Path(temp)
    for name,(blob,expected) in {compiled!r}.items():
        data = zlib.decompress(base64.b64decode(blob))
        if sha256(data).hexdigest() != expected:
            raise ValueError(f'Source changed: {{name}}')
        (temp/name).write_bytes(data)
    sys.path.insert(0,str(temp))
    from e180_final import run
    run(receipt.parent, Path('/kaggle/working'), {hashes!r})
'''
    compile(runtime,SLUG,'exec')
    meta = dict(id='jiyujia/'+SLUG, title=SLUG, code_file='transfer.py', language='python', kernel_type='script',
                is_private=True,enable_gpu=False,enable_internet=False,docker_image=COMMON['IMAGE'],
                kernel_sources=[REFERENCE],dataset_sources=[COMMON['WHEEL']],competition_sources=[])
    out.mkdir(parents=True,exist_ok=True)
    (out/meta['code_file']).write_text(runtime)
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(sha256(runtime.encode()).hexdigest())


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--native',type=Path,required=True)
    cli.add_argument('--out',type=Path,required=True)
    build(**vars(cli.parse_args()))

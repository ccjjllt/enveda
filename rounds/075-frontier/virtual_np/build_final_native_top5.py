"""Build the offline, label-blind CPU replay from the sealed historical final lists and native scores."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from runpy import run_path


HERE = Path(__file__).resolve().parent
COMMON = run_path(str(HERE/'build.py'))
BASE = 'jiyujia/enveda-075-np-recall-reseal/2'
SNAP = 'jiyujia/enveda-075-c39-snapshot-t4/1'
SLUG = 'enveda-075-final-native-top5'


def build(out):
    source = (HERE/'final_native_top5.py').read_text()
    bootstrap = """
from pathlib import Path
import subprocess
import sys
inputs = Path('/kaggle/input')
""" + COMMON['METRIC_BOOTSTRAP']
    entry = '''
def mounted(ref, name):
    slug = ref.split('/')[1]
    path, = (p for p in inputs.rglob(name) if slug in p.parts)
    return path
run(mounted(SNAP_REF,'c39_snapshot_receipt.json').parent, mounted(BASE_REF,'baseline.zip'), Path('/kaggle/working'))
'''
    code = bootstrap + '\n' + source + '\n' + entry
    compile(code, SLUG, 'exec')
    meta = dict(id='jiyujia/'+SLUG,title=SLUG,code_file='final_native_top5.py',language='python',
                kernel_type='script',is_private=True,enable_gpu=False,enable_internet=False,
                docker_image=COMMON['IMAGE'],kernel_sources=[BASE,SNAP],competition_sources=[],
                dataset_sources=[COMMON['WHEEL']])
    out.mkdir(parents=True,exist_ok=True)
    (out/meta['code_file']).write_text(code)
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(sha256(code.encode()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    build(**vars(parser.parse_args()))

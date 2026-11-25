"""Package the label-blind native-C34 MS-GPT comparison for one Kaggle T4."""

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import zlib

from build_load_check import IMAGE, RUNTIME, model_runtime
from build_preflight import METRIC, PROFILE, SHARED


SLUG = 'enveda-075-ms-gpt-blind-c34'
QUERY_SHA256 = '2a8bf747f50e6920fcd4f24e04302b721281f8794c947cc58fe545264b12d0d4'
ARCHIVE_SHA256 = 'd7b28154b1bdf132528300bcf8faa1211984c843d5340ac81191472a452f564b'
BLIND_SOURCE = 'jiyujia/enveda-075-np-recall-base/2'


def pack(value):
    return base64.b64encode(zlib.compress(value, 9)).decode(), sha256(value).hexdigest()


def build(queries, c34, out):
    spectra, archive = queries.read_bytes(), c34.read_bytes()
    if sha256(spectra).hexdigest() != QUERY_SHA256:
        raise ValueError('Expected the pinned 60 blind query spectra.')
    if sha256(archive).hexdigest() != ARCHIVE_SHA256:
        raise ValueError('Expected the unmodified archived native C34 candidate order.')
    native = {mid: {'target': entry['target'], 'pc': entry['pc']}
              for mid, entry in json.loads(archive).items()}
    if (len(native) != 60 or any(not 0 < r['target'] < 10000 or len(r['pc']) > 25
                                 for r in native.values())):
        raise ValueError('Expected the archived 60 native C34 centers and ranked lists.')
    compact = json.dumps({'molecules': native}, separators=(',', ':'), allow_nan=False).encode()
    program = Path(__file__).with_name('score_blind.py').read_bytes()
    compile(program, 'score_blind.py', 'exec')
    payload = {
        'queries.parquet': pack(spectra),
        'native_c34.json': pack(compact),
        'score_blind.py': pack(program),
    }
    sampler = f'''
import base64
import zlib

scratch = Path(temporary)
for filename, (blob, expected) in {payload!r}.items():
    raw = zlib.decompress(base64.b64decode(blob))
    path = scratch / filename
    path.write_bytes(raw)
    checked(path, expected)
sys.path.insert(0, str(scratch))
from score_blind import run

calibration_path = asset(shared, shared_manifest, 'msgpt-source/' + config['calibration_file'])
calibration = json.loads(calibration_path.read_text())['benchmarks']['massspecgym']
if calibration['calibration_split'] != 'train':
    raise ValueError('Expected a training-fold-only density band.')
band = (calibration['threshold_low'], calibration['threshold_high'])
provenance = dict(query_sha256={QUERY_SHA256!r}, c34_archive_sha256={ARCHIVE_SHA256!r},
                  c34_source={BLIND_SOURCE!r}, compact_c34_sha256={payload['native_c34.json'][1]!r},
                  source_sha256={payload['score_blind.py'][1]!r}, model_load=receipt)
run(scratch / 'queries.parquet', scratch / 'native_c34.json',
    Path('/kaggle/working'), model, mist, tokenizer, cfg, config, band, device, provenance)
'''
    code = model_runtime(use_gpu=True, after_load=sampler)
    compile(code, SLUG, 'exec')
    notebook = dict(nbformat=4, nbformat_minor=5, cells=[
        dict(cell_type='code', id='blind-native-c34-generation', execution_count=None,
             metadata={}, outputs=[], source=code.strip().splitlines(keepends=True)),
    ], metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    metadata = dict(
        id=f'jiyujia/{SLUG}', title=SLUG, code_file='predict.ipynb', language='python',
        kernel_type='notebook', is_private=True, enable_gpu=True, enable_internet=False,
        dataset_sources=[SHARED, PROFILE, METRIC], competition_sources=[], kernel_sources=[RUNTIME],
        docker_image=IMAGE, machine_shape='NvidiaTeslaT4',
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / 'predict.ipynb').write_text(json.dumps(notebook, separators=(',', ':')) + '\n')
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('queries', 'c34', 'out'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    build(**vars(parser.parse_args()))

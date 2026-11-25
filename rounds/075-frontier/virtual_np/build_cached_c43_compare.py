"""Package a private CPU comparison after the cached C43 arms are sealed."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from build import HERE, IMAGE, METRIC_BOOTSTRAP, WHEEL, pack


BASE = 'enveda-075-np-recall-reseal'


def build(panel, receipt, version, out, *, replacement=False, native_only=False):
    raw = receipt.read_bytes()
    sealed = json.loads(raw)
    if replacement and native_only:
        raise ValueError('Choose exactly one C43 scoring mode')
    if native_only:
        blind = 'enveda-075-c41-native-only-c43'
        slug = 'enveda-075-c41-native-only-compare'
        seal_name = 'c41_native_only_receipt.json'
        complete = (sealed['status']=='c43_sealed' and sealed['query_count']==60 and sealed['candidate_budget']==25
                    and sealed['historical_c43_exact']==sealed['original_slots_c43_exact']==sealed['original_slots_main_exact']==60
                    and sealed['new_predictor_requests_in_this_job']==0
                    and sealed['native_only_incremental_forward_requests_vs_public_v29']==0
                    and sealed['original_slots_reference_shadow_archived']
                    and sealed['files']['original_slots_c43.json']=='f9bddb8082ea263bbb29fbd95c4548a9de3d0216a6a15aae18011ae103c2940d')
    elif replacement:
        blind = 'enveda-075-c41-shadow-paired'
        slug = 'enveda-075-c41-shadow-paired-compare'
        seal_name = 'c41_study_receipt.json'
        complete = (sealed['status']=='paired_c43_sealed' and sealed['equal_finite_counts']
                    and sealed['original_slots']['molecules']==sealed['replacement']['molecules']==60)
    else:
        blind = 'enveda-075-c41-existing-c43'
        slug = 'enveda-075-c41-existing-compare'
        seal_name = 'c41_existing_receipt.json'
        complete = (sealed['status']=='c43_sealed' and sealed['molecules']==60
                    and sealed['original_forward_inputs_unchanged'] and sealed['new_predictor_requests']==0)
    if version < 1 or not complete:
        raise ValueError('Provide the completed blind C43 receipt')
    panel_hash = json.loads((panel/'panel_manifest.json').read_text())['output_sha256']
    label_sha, query_sha = (panel_hash[key] for key in ('private/labels.parquet','blind/test.parquet'))
    if native_only and sealed['queries_sha256']!=query_sha:
        raise ValueError('The sealed attribution uses a different query population')
    labels = pack((panel/'private/labels.parquet').read_bytes())
    if labels[1] != label_sha:
        raise ValueError('The authorized answers have changed since freezing')
    files = {
        'labels.parquet':labels,
        'score_cached_c43.py':pack((HERE/'score_cached_c43.py').read_bytes()),
        'evaluate.py':pack((HERE/'evaluate.py').read_bytes()),
        'retrieval_pilot_score.py':pack((HERE.parent/'four_arm/retrieval_pilot_score.py').read_bytes()),
    }
    runtime = f'''
import base64
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib

inputs = Path('/kaggle/input')
{METRIC_BOOTSTRAP}

def mounted(name, slug):
    path, = (p for p in inputs.rglob(name) if slug in p.parts)
    return path

with tempfile.TemporaryDirectory(prefix='cached-c43-score-') as directory:
    root = Path(directory)
    for name, (blob, expected) in {files!r}.items():
        raw = zlib.decompress(base64.b64decode(blob))
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f'Private scoring input drifted: {{name}}')
        (root/name).write_bytes(raw)
    subprocess.run([sys.executable, str(root/'score_cached_c43.py'),
                    '--baseline', str(mounted('baseline.zip', {BASE!r})),
                    '--blind', str(mounted({seal_name!r}, {blind!r}).parent),
                    '--labels', str(root/'labels.parquet'), '--seal-sha256', {sha256(raw).hexdigest()!r},
                    '--query-sha256', {query_sha!r}, '--label-sha256', {label_sha!r},
                    '--out', '/kaggle/working', *{['--native-only'] if native_only else ['--replacement'] if replacement else []!r}], check=True)
'''
    compile(runtime, slug, 'exec')
    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata={'kernelspec': {'display_name':'Python 3', 'language':'python', 'name':'python3'}},
                    cells=[dict(cell_type='code', id='cached-c43-compare', metadata={}, execution_count=None,
                                outputs=[], source=runtime.strip().splitlines(keepends=True))])
    metadata = dict(id=f'jiyujia/{slug}', title=slug, code_file='score.ipynb', language='python', kernel_type='notebook',
                    is_private=True, enable_gpu=False, enable_internet=False, competition_sources=[],
                    dataset_sources=[WHEEL], docker_image=IMAGE, kernel_sources=[f'jiyujia/{BASE}/2', f'jiyujia/{blind}/{version}'])
    out.mkdir(parents=True, exist_ok=True)
    (out/'score.ipynb').write_text(json.dumps(notebook, separators=(',',':'))+'\n')
    (out/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('panel','receipt','out'):
        cli.add_argument('--'+name, type=Path, required=True)
    cli.add_argument('--version', type=int, required=True)
    modes=cli.add_mutually_exclusive_group()
    modes.add_argument('--replacement', action='store_true')
    modes.add_argument('--native-only', action='store_true')
    build(**vars(cli.parse_args()))

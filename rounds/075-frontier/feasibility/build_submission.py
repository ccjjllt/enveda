"""Build the private offline Pablo v29 notebook with a final absolute reranking cell."""

import argparse
import ast
import base64
from hashlib import sha256
import json
from pathlib import Path

from build import HERE, SOURCE_SHA, native_fragment_source, pack


SLUG = 'enveda-075-pablo-0506-absolute'
PUBLIC_REF = 'pablodiegoo/enveda-casmi26-starter-notebook'
METADATA_SHA = 'ca306d7710f2b361eb182602368bc42b50afd299bd6061e9deed2ecdfe57a43c'
HEAD_EXPORT = '''
def _absolute_export_source_heads():
    import json
    from pathlib import Path

    heads = {
        mid: dict(main=list(BASE[mid][0][:1]),
                  engine2=list(ENG.get(mid, {}).get('smiles', [])[:1]),
                  pubchem=list(PC.get(mid, {}).get('pc', [])[:1]))
        for mid in V43_RECEIPT['processed_ids']
    }
    Path('/kaggle/working/absolute_source_heads.json').write_text(json.dumps(heads, sort_keys=True) + '\\n')

_absolute_export_source_heads()
del _absolute_export_source_heads
'''


def export_heads(book):
    bundle = book['cells'][10]
    code = ''.join(bundle['source'])
    nodes = {node.targets[0].id: node for node in ast.parse(code).body
             if isinstance(node, ast.Assign) and len(node.targets) == 1
             and isinstance(node.targets[0], ast.Name)}
    embedded = base64.b64decode(ast.literal_eval(nodes['FROZEN_B64'].value), validate=True)
    old_sha = sha256(embedded).hexdigest()
    if old_sha != ast.literal_eval(nodes['FROZEN_SHA256'].value):
        raise ValueError('The original Pablo embedded notebook digest differs')
    inner = json.loads(embedded)
    cell = inner['cells'][54]
    original = ''.join(cell['source'])
    if 'V43_RECEIPT = _v43_receipt(globals())' not in original:
        raise ValueError('The pinned inner final-receipt cell differs')
    exported = original.rstrip() + '\n\n' + HEAD_EXPORT
    compile(exported, 'inner-source-head-export', 'exec')
    cell['source'] = exported.splitlines(keepends=True)
    payload = json.dumps(inner, ensure_ascii=False, separators=(',', ':')).encode()
    new_sha = sha256(payload).hexdigest()
    values = {'FROZEN_B64': base64.b64encode(payload).decode(), 'FROZEN_SHA256': new_sha}
    lines = code.splitlines(keepends=True)
    for name, value in sorted(values.items(), key=lambda pair: -nodes[pair[0]].lineno):
        node = nodes[name]
        lines[node.lineno - 1:node.end_lineno] = [f'{name} = {value!r}\n']
    bundle['source'] = lines
    return old_sha, new_sha


def final_cell(payload, native_sha):
    return f'''
def _apply_frozen_absolute(root, comp, pipeline_done, started, session_hours):
    import base64
    from hashlib import sha256
    import json
    from pathlib import Path
    import subprocess
    import sys
    from tempfile import TemporaryDirectory
    import time
    import zlib

    root, comp = Path(root), Path(comp)
    target = root / 'submission.csv'
    audit_target = root / 'absolute_audit.json'
    remaining = session_hours * 3600 - (time.time() - started) - 90
    if not pipeline_done or remaining <= 0:
        reason = 'pipeline not run' if not pipeline_done else 'session time guard'
        audit_target.write_text(json.dumps(dict(status='skipped', reason=reason)) + '\\n')
        print('[absolute]', reason, flush=True)
        return
    with TemporaryDirectory(dir=root, prefix='.absolute-stage-') as name:
        scratch = Path(name)
        for filename, (blob, expected) in {payload!r}.items():
            raw = zlib.decompress(base64.b64decode(blob))
            if sha256(raw).hexdigest() != expected:
                raise ValueError(f'Absolute runtime changed: {{filename}}')
            (scratch / filename).write_bytes(raw)
        native = (root / 'frag_rescore.py').read_bytes()
        if sha256(native).hexdigest() != {native_sha!r}:
            raise ValueError('The native Pablo fragment source differs from the verified pilot')
        (scratch / 'frag_rescore.py').write_bytes(native)
        program = scratch / 'submission_tail.py'
        try:
            subprocess.run([sys.executable, str(program), '--work', str(root), '--comp', str(comp),
                            '--out', str(scratch)], timeout=remaining, check=True)
        except subprocess.TimeoutExpired:
            audit_target.write_text(json.dumps(dict(status='skipped', reason='session timeout')) + '\\n')
            print('[absolute] session timeout, retaining Pablo submission', flush=True)
            return
        receipt = scratch / 'absolute_audit.json'
        audit = json.loads(receipt.read_text())
        if sha256(target.read_bytes()).hexdigest() != audit['submission_before_sha256']:
            raise ValueError('Pablo submission changed while the absolute stage was running')
        if audit['changed_molecules']:
            staged = scratch / 'absolute_submission.csv'
            if sha256(staged.read_bytes()).hexdigest() != audit['submission_after_sha256']:
                raise ValueError('Staged absolute submission differs from its audit')
            staged.replace(target)
        receipt.replace(audit_target)
        print('[absolute]', audit['status'], 'changed_molecules', audit['changed_molecules'], flush=True)

_apply_frozen_absolute(ROOT, COMP, PIPELINE_DONE, NB_T0, TIME_GUARD_HOURS)
'''


def build(source, metadata, output):
    raw = source.read_bytes()
    meta_raw = metadata.read_bytes()
    if sha256(raw).hexdigest() != SOURCE_SHA or sha256(meta_raw).hexdigest() != METADATA_SHA:
        raise ValueError('Expected the pinned Pablo v29 notebook and metadata')
    original = json.loads(meta_raw)
    if original['ref'] != PUBLIC_REF or original['currentVersionNumber'] != 29:
        raise ValueError('Expected the Pablo v29 submission source')
    book = json.loads(raw)
    if book['cells'][37].get('id') != 'v29expy0':
        raise ValueError('Expected E180X to end the original Pablo v29 ranking')
    original_inner, adapted_inner = export_heads(book)
    files = (HERE / 'submission_tail.py', HERE / 'rank.py',
             HERE.parent / 'four_arm/retrieval_pilot_score.py')
    payload = {path.name: pack(path.read_bytes()) for path in files}
    native_sha = sha256(native_fragment_source(source)).hexdigest()
    book['cells'].insert(38, dict(id='physical-absolute-tail', cell_type='code', metadata={},
                                  execution_count=None, outputs=[],
                                  source=final_cell(payload, native_sha).splitlines(keepends=True)))
    for i, cell in enumerate(book['cells']):
        if cell['cell_type'] == 'code':
            cell.update(execution_count=None, outputs=[])
            compile(''.join(cell['source']), f'pablo-absolute-{i}', 'exec')
    options = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='submission.ipynb', language='python',
                   kernel_type='notebook', is_private=True, enable_internet=False,
                   enable_gpu=original['enableGpu'], enable_tpu=original['enableTpu'],
                   machine_shape=original['machineShape'], docker_image=original['dockerImage'],
                   dataset_sources=original['datasetDataSources'],
                   competition_sources=original['competitionDataSources'],
                   kernel_sources=original['kernelDataSources'])
    output.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(book, ensure_ascii=False, separators=(',', ':')).encode() + b'\n'
    (output / options['code_file']).write_bytes(rendered)
    (output / 'kernel-metadata.json').write_text(json.dumps(options, indent=2) + '\n')
    print(json.dumps(dict(path=str(output / options['code_file']), sha256=sha256(rendered).hexdigest(),
                          original_inner_sha256=original_inner, adapted_inner_sha256=adapted_inner,
                          absolute_runtime_sha256={name: value[1] for name, value in payload.items()})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'metadata', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    build(**vars(parser.parse_args()))

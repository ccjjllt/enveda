"""Package independent MassBank 110 coverage scoring for the sealed NP-MRD scan."""

import argparse
from collections import Counter
import csv
import json
from math import isclose
from pathlib import Path
import re


SLUG = 'enveda-075-npmrd-mb110-score'
SCAN_SLUG = 'enveda-075-npmrd-2025-mb110-window'
STAGE = 'jiyujia/enveda-075-npmrd-2025-source/1'
REPORT = 'npmrd_mb110_window_receipt.json'
WINDOW = 'npmrd_mb110_window.parquet'
OUTPUT = 'npmrd_mb110_window_score'
SCOPE = 'massbank110-npmrd-2025-exact-window-inventory'
COCONUT_RECEIPT_SHA = '78198393949cb494423d1446781f2699739e2226fcf477a6eb6e68dc9c47b825'
PC_SHA = '58092dc1f9db483f2681a7f90efb81d8f18fb6254c7982d90ca042d80c6d8f6c'
TARGETS_SHA = '4adca54b03c95f9e0f17e23a4ce0259da41fcf894da5d370143f20804bafde09'
NEW_INVENTORIES = ('npmrd', 'npmrd_added_to_coconut_rebuilt_union', 'union_with_npmrd')


def validate_seal(seal):
    expected = dict(status='complete', scope=SCOPE, query_count=110, exact_targets_sha256=TARGETS_SHA,
                    archived_pc_lists_sha256=PC_SHA, stage=STAGE, coconut_window_receipt_sha256=COCONUT_RECEIPT_SHA,
                    rdkit='2026.03.3', broad_window_ppm=10, author_window_ppm=[-3.0, 4.6])
    if (any(seal.get(key) != value for key, value in expected.items()) or len(seal['cases']) != 110
            or set(seal['outputs']) != {WINDOW}
            or not all(re.fullmatch(r'[0-9a-f]{64}', value) for value in
                       (seal['outputs'][WINDOW], seal['stage_receipt_sha256']))):
        raise ValueError('Expected the sealed NP-MRD inventory on the original 110 C34 windows')


def verify_npmrd(receipt, seal, targets, rebuilt, coconut, identity):
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from scan_npmrd_mb110_window import SCHEMA
    from score_coconut_mb110 import WINDOWS, read_candidates
    from score_outsider import pinned

    path = pinned(receipt.with_name(WINDOW), seal['outputs'][WINDOW])
    npmrd = read_candidates(path, SCHEMA, targets)
    if seal['cases'].keys() != targets.keys():
        raise ValueError('Sealed NP-MRD cases do not match the archived C34 queries')
    counts, occurrences = Counter(), 0
    for mid, rows in npmrd.items():
        for key, row in rows.items():
            normalized = identity(row['smiles'])
            if normalized is None or (normalized[0], ','.join(f'{a}:{n}' for a, n in normalized[1])) != key:
                raise ValueError(f'NP-MRD candidate identity changed at {mid}')
            mol = Chem.MolFromSmiles(row['smiles'])
            if (not isclose(rdMolDescriptors.CalcExactMolWt(mol), row['mass'], rel_tol=0, abs_tol=1e-9)
                    or rdMolDescriptors.CalcMolFormula(mol) != row['formula']
                    or Chem.GetFormalCharge(mol) != row['formal_charge'] or len(Chem.GetMolFrags(mol)) != 1
                    or row['source_row'] < 1 or row['source_records_in_window'] < 1
                    or not re.fullmatch(r'NP\d{7}', row['npmrd_id'])):
                raise ValueError(f'Invalid representative NP-MRD structure or source accession at {mid}')
            occurrences += row['source_records_in_window']
        summary = {}
        for win in WINDOWS:
            author = win == 'author'
            np_keys, old, new = ({key for key, row in group.items() if not author or row['in_author_window']}
                                 for group in (rows, rebuilt[mid], coconut[mid]))
            reference = old | new
            shared, residual = np_keys & reference, np_keys - reference
            summary.update({f'{name}_{win}': len(keys) for name, keys in (
                ('npmrd', np_keys), ('rebuilt', old), ('coconut_2026', new), ('reference_union', reference),
                ('shared_with_reference', shared), ('residual', residual),
            )})
            for key, row in rows.items():
                mask = rebuilt[mid][key]['source_mask'] if key in old else 0
                if (row[f'rebuilt_source_mask_in_{win}'] != mask or row[f'coconut_2026_in_{win}'] != (key in new)
                        or row[f'residual_in_{win}'] != (key in residual)):
                    raise ValueError(f'NP-MRD overlap flags differ from frozen COCONUT data at {mid}/{win}')
        if dict(target_mass=targets[mid], **summary) != seal['cases'][mid]:
            raise ValueError(f'NP-MRD blind case counts differ from the independent sets at {mid}')
        counts.update(summary)
    if (dict(counts) != seal['totals'] or occurrences != seal['source_counts']['window_source_query_occurrences']
            or occurrences - counts['npmrd_broad'] != seal['source_query_duplicates']):
        raise ValueError('NP-MRD blind overlap totals differ from the independent sets')
    return npmrd


def score(inputs, out, reference, receipt_sha):
    from rdkit import RDLogger
    from c39_request_cache import identity_function
    from scan_coconut_2026_mb110_window import OLD_WINDOW, WINDOW as COCO_WINDOW, frozen_targets
    from scan_npmrd_mb110_window import read_stage
    import score_coconut_mb110 as coco
    from score_outsider import BASE, LABELS, LABELS_SHA, PINS, artifact, checksum, pinned, read_json
    from stage_npmrd_2025 import RECEIPT as STAGE_REPORT

    RDLogger.DisableLog('rdApp.*')
    identity = identity_function()
    receipt = pinned(artifact(inputs, reference, REPORT), receipt_sha)
    seal = read_json(receipt)
    validate_seal(seal)
    _, release = read_stage(artifact(inputs, STAGE, STAGE_REPORT), seal['stage_receipt_sha256'])
    source_hashes = {item['filename']: item['sha256'] for item in release['artifacts']}
    if (source_hashes != seal['source_artifacts_sha256'] or release['release_date'] != seal['source_release_date']
            or release['license'] != seal['source_license']):
        raise ValueError('Source metadata in the NP-MRD blind receipt changed')

    coconut_receipt = pinned(artifact(inputs, coco.SCAN, coco.SCAN_REPORT), COCONUT_RECEIPT_SHA)
    coconut_seal = read_json(coconut_receipt)
    coco.validate_seal(coconut_seal)
    pc_lists = pinned(artifact(inputs, BASE, 'pc_lists.json'), PINS['pc_lists.json'])
    targets, targets_sha = frozen_targets(pc_lists)
    coco_hashes = {name: coconut_seal['outputs'][name] for name in (OLD_WINDOW, COCO_WINDOW)}
    if (targets_sha != TARGETS_SHA or targets_sha != coconut_seal['exact_targets_sha256']
            or seal['coconut_window'] != coco.SCAN or seal['coconut_window_outputs'] != coco_hashes):
        raise ValueError('Blind NP-MRD and COCONUT windows do not have the same original targets')
    new, old = coco.verified_inventory(coconut_receipt, coconut_seal, targets)
    npmrd = verify_npmrd(receipt, seal, targets, old, new, identity)
    known, source_counts = coco.read_source_union(inputs, pc_lists, targets, identity)

    # The benchmark truth and the historical miss audit are opened after verifying every blind source.
    rows = []
    for mid, key, missed, index_miss in coco.truth_cohorts(inputs, targets, identity, known):
        row = dict(molecule_id=mid, original_source_union_miss=missed, original_checked_window_index_miss=index_miss)
        match = npmrd[mid].get(key)
        row['npmrd_source_accession'] = match['npmrd_id'] if match is not None else ''
        for win in coco.WINDOWS:
            author = win == 'author'
            flags = coco.presence(new[mid].get(key), old[mid].get(key), author)
            present = match is not None and (not author or match['in_author_window'])
            flags.update(npmrd=present, npmrd_added_to_coconut_rebuilt_union=present and not flags['rebuilt_union'],
                         union_with_npmrd=present or flags['rebuilt_union'])
            row.update({f'{win}_{name}': flag for name, flag in flags.items()})
        rows.append(row)
    out.mkdir(exist_ok=True, parents=True)
    per_query = out / (OUTPUT + '_per_query.csv')
    with per_query.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(status='complete', scope='Exact structural coverage in fixed C34 windows before ranking',
                  query_count=len(rows), scan=reference, scan_receipt_sha256=receipt_sha,
                  stage=STAGE, stage_receipt_sha256=seal['stage_receipt_sha256'],
                  release_date=release['release_date'], source_license=release['license'],
                  coconut_scan=coco.SCAN, coconut_scan_receipt_sha256=COCONUT_RECEIPT_SHA,
                  labels=LABELS, labels_sha256=LABELS_SHA, original_index_audit=coco.INDEX_AUDIT,
                  original_index_audit_sha256=coco.INDEX_AUDIT_SHA, source_candidates=dict(source_counts),
                  exact_targets_sha256=targets_sha, broad_window_ppm=seal['broad_window_ppm'],
                  author_window_ppm=seal['author_window_ppm'], blind_inventory_totals=seal['totals'],
                  cohorts=coco.report_coverage(rows, (*coco.INVENTORIES, *NEW_INVENTORIES)),
                  per_query_sha256=checksum(per_query), new_predictor_requests=0)
    (out / (OUTPUT + '.json')).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report['cohorts'], allow_nan=False), flush=True)


def build(seal, out, version=1):
    from build import HERE, IMAGE, METRIC_BOOTSTRAP, WHEEL, pack
    from build_final_native_outsider_mb110_score import BASE, LABELS, checksum
    from build_score_coconut_2026_mb110_window import INDEX_AUDIT, SCAN as COCONUT_SCAN

    validate_seal(json.loads(seal.read_bytes()))
    reference = f'jiyujia/{SCAN_SLUG}/{version}'
    sources = {name: (HERE / name).read_bytes() for name in (
        'c39_request_cache.py', 'generate.py', 'rebuild_parents.py', 'scan_rebuilt_parent_window.py',
        'scan_coconut_2026_window.py', 'scan_coconut_2026_mb110_window.py', 'scan_npmrd_mb110_window.py',
        'stage_npmrd_2025.py',
    )}
    sources['score_coconut_mb110.py'] = (HERE / 'build_score_coconut_2026_mb110_window.py').read_bytes()
    sources['score_outsider.py'] = (HERE / 'build_final_native_outsider_mb110_score.py').read_bytes()
    sources['score_npmrd.py'] = Path(__file__).read_bytes()
    payload = {}
    for name, data in sources.items():
        compile(data, name, 'exec')
        payload[name] = pack(data)
    code = f'''
import base64, subprocess, sys, tempfile, zlib
from hashlib import sha256
from pathlib import Path
inputs = Path('/kaggle/input')
{METRIC_BOOTSTRAP}
with tempfile.TemporaryDirectory(prefix='npmrd-mb110-score-') as temp:
    for name, (blob, expected) in {payload!r}.items():
        data = zlib.decompress(base64.b64decode(blob))
        if sha256(data).hexdigest() != expected:
            raise ValueError(f'Packaged source changed: {{name}}')
        Path(temp, name).write_bytes(data)
    sys.path.insert(0, temp)
    from score_npmrd import score
    score(inputs, Path('/kaggle/working'), {reference!r}, {checksum(seal)!r})
'''
    compile(code, SLUG, 'exec')
    meta = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='run_score.py', language='python', kernel_type='script',
                is_private=True, enable_gpu=False, enable_internet=False, docker_image=IMAGE, dataset_sources=[WHEEL],
                competition_sources=[], kernel_sources=[reference, COCONUT_SCAN, STAGE, BASE, INDEX_AUDIT, LABELS])
    out.mkdir(exist_ok=True, parents=True)
    (out / 'run_score.py').write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return checksum(out / 'run_score.py')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--version', type=int, default=1)
    print(build(**vars(parser.parse_args())))

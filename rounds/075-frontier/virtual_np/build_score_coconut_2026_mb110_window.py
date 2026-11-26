"""Package a private, post-seal coverage score of the MassBank 110 COCONUT scan."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re


SLUG = 'enveda-075-coconut-mb110-window-score'
SCAN = 'jiyujia/enveda-075-coconut-2026-mb110-window/1'
INDEX_AUDIT = 'jiyujia/enveda-075b-massbank-union-misses/4'
INDEX_AUDIT_SHA = '358d043dfb9f9f6d13f3270d208cf9bead6063fbdcd73acc24d01433955b638d'
ABSENT_GROUP = 'not_found_in_e1_pubchem_or_checked_e2'
SCAN_REPORT = 'coconut_2026_mb110_window_receipt.json'
OUTPUT = 'coconut_2026_mb110_window_score'
STAGE_SHA = '53ceba25df0272f48a283d30fdec1e8dc05f79333fe5b7e3cfca1582cb15137c'
ARCHIVE_SHA = '81f3edc1731f01da289fe6b84d162667863e331d52041a781565e1af2eb79939'
REBUILT_RECEIPT_SHA = '96974cf1d5fcf789988791dd4d56d2dafdc39bcc442cc3bbcf348ebf1fa3ab96'
REBUILT_SHA = '9e3595946b9e417167e148f23bc1d275d3aa942642ad4f95d3dc4c22809fd88e'
INVENTORIES = ('coconut_2024', 'rebuilt', 'coconut_2026', 'coconut_union', 'rebuilt_union',
               'added_to_coconut_2024', 'added_to_rebuilt')
WINDOWS = ('broad', 'author')


def validate_seal(seal):
    from generate import AUTHOR_PPM, C34_PPM, RDKit_VERSION
    from scan_coconut_2026_mb110_window import BASE, OLD_WINDOW, PC_SHA, RECORDS, REBUILT, STAGE, WINDOW

    expected = dict(status='complete', scope='massbank110-coconut-2026-exact-window-inventory', query_count=110,
                    archived_c34=BASE, archived_pc_lists_sha256=PC_SHA, stage=STAGE, stage_receipt_sha256=STAGE_SHA,
                    archive_sha256=ARCHIVE_SHA, old_inventory=REBUILT, rebuilt_receipt_sha256=REBUILT_RECEIPT_SHA,
                    rebuilt_parent_sha256=REBUILT_SHA, rebuilt_parent_identities=716146,
                    rdkit=RDKit_VERSION, broad_window_ppm=C34_PPM, author_window_ppm=list(AUTHOR_PPM))
    if (any(seal.get(field) != value for field, value in expected.items())
            or len(seal['cases']) != 110 or set(seal['outputs']) != {WINDOW, OLD_WINDOW, RECORDS}
            or any(not re.fullmatch(r'[0-9a-f]{64}', digest) for digest in seal['outputs'].values())):
        raise ValueError('Expected the checksum-verified blind COCONUT MB110 v1 scan')


def read_candidates(path, schema, targets):
    import pyarrow.parquet as pq
    from generate import AUTHOR_PPM, C34_PPM

    handle = pq.ParquetFile(path)
    if not handle.schema_arrow.equals(schema, check_metadata=False):
        raise ValueError(f'Different frozen candidate schema: {path.name}')
    result = {mid: {} for mid in targets}
    for batch in handle.iter_batches():
        if any(column.null_count for column in batch.columns):
            raise ValueError(f'Null value in sealed candidate table: {path.name}')
        for row in batch.to_pylist():
            mid = row['molecule_id']
            mass = row['mass']
            if mid not in targets or not isfinite(mass) or mass <= 0:
                raise ValueError('Invalid source candidate population or mass')
            target = targets[mid]
            bound = target * C34_PPM * 1e-6
            key = row['score_key'], row['heavy_atoms']
            error = (target - mass) / mass * 1e6
            if (not re.fullmatch(r'[A-Z]{14}', key[0])
                    or not re.fullmatch(r'[A-Z][a-z]?:[1-9]\d*(,[A-Z][a-z]?:[1-9]\d*)*', key[1])
                    or key in result[mid] or not target - bound <= mass <= target + bound
                    or row['mass_error_ppm'] != error
                    or row['in_author_window'] != (AUTHOR_PPM[0] <= error <= AUTHOR_PPM[1])):
                raise ValueError(f'Sealed candidate does not reproduce exact C34 windows: {mid}')
            result[mid][key] = row
    return result


def verified_inventory(receipt, seal, targets):
    import pyarrow.parquet as pq
    from scan_coconut_2026_mb110_window import NEW_SCHEMA, OLD_SCHEMA, OLD_WINDOW, RECORDS, WINDOW, compare_window
    from score_outsider import pinned

    paths = {name: pinned(receipt.parent / name, digest) for name, digest in seal['outputs'].items()}
    new = read_candidates(paths[WINDOW], NEW_SCHEMA, targets)
    old = read_candidates(paths[OLD_WINDOW], OLD_SCHEMA, targets)
    if set(seal['cases']) != set(targets):
        raise ValueError('Sealed case roster differs from the archived baseline')
    counts, occurrences = Counter(), 0
    for mid, rows in new.items():
        if any(item['source_mask'] not in (1, 2, 3) for item in old[mid].values()):
            raise ValueError(f'Unexpected rebuilt source bit: {mid}')
        # Use the blind overlap routine to reproduce exact membership counters and source-mask flags.
        selected = {key: (None, item, item['source_records_in_window']) for key, item in rows.items()}
        expected_rows, summary = compare_window(selected, old[mid])
        if (expected_rows != [row for _, row in sorted(rows.items())] or any(
            item['source_records_in_window'] < 1 or item['source_row'] < 1 for item in rows.values()
        ) or dict(target_mass=targets[mid], **summary) != seal['cases'][mid]):
            raise ValueError(f'Sealed old/new COCONUT comparison changed: {mid}')
        counts.update(summary)
        occurrences += sum(item['source_records_in_window'] for item in rows.values())
    original_counts = seal['source_counts']
    if (dict(counts) != seal['totals'] or occurrences != original_counts['window_source_query_occurrences']
            or occurrences - counts['coconut_2026_broad'] != seal['source_query_duplicates']
            or pq.ParquetFile(paths[RECORDS]).metadata.num_rows != original_counts['valid_window_source_rows']):
        raise ValueError('Sealed candidate, provenance or deduplication counts changed')
    return new, old


def read_source_union(inputs, pc_lists, targets, identity):
    import pyarrow.parquet as pq
    from score_outsider import BASE, PINS, artifact, pinned, read_baseline, read_json

    final = read_baseline(pinned(artifact(inputs, BASE, 'submission.csv'), PINS['submission.csv']))
    original_pc = read_json(pc_lists)
    if set(final) != set(targets) or set(original_pc) != set(targets):
        raise ValueError('The pinned baseline query populations disagree')
    union = {mid: {identity(text) for text in (*final[mid], *original_pc[mid]['pc'])} for mid in final}
    source_candidates = pinned(artifact(inputs, BASE, 'source_candidates.parquet'), PINS['source_candidates.parquet'])
    required = {'engine1', 'engine2', 'prefusion', 'pubchem'}
    sources, positions = Counter(), set()
    for row in pq.read_table(source_candidates, columns=['molecule_id', 'source', 'smiles', 'rank', 'candidate_id']).to_pylist():
        mid, source, rank = row['molecule_id'], row['source'], row['rank']
        key = (mid, source, rank)
        normalized = identity(row['smiles'])
        if (mid not in targets or source not in required | {'fusion_overflow', 'apex_external'}
                or type(rank) is not int or rank <= 0 or key in positions or normalized is None
                or normalized[0] != row['candidate_id']):
            raise ValueError('Changed historical source candidate or query')
        positions.add(key)
        sources[source] += 1
        union[mid].add(normalized)
    if not required <= sources.keys() or any(None in group for group in union.values()):
        raise ValueError('Historical ground-candidate union is incomplete')
    return union, sources


def presence(new_row, old_row, author):
    old_available = old_row is not None and (not author or old_row['in_author_window'])
    coco_old = old_available and bool(old_row['source_mask'] & 2)
    coco_new = new_row is not None and (not author or new_row['in_author_window'])
    return dict(coconut_2024=coco_old, rebuilt=old_available, coconut_2026=coco_new,
                coconut_union=coco_old or coco_new, rebuilt_union=old_available or coco_new,
                added_to_coconut_2024=coco_new and not coco_old, added_to_rebuilt=coco_new and not old_available)


def report_coverage(rows, inventories=INVENTORIES):
    groupings = dict(all=rows,
                     original_source_union_misses=[row for row in rows if row['original_source_union_miss']],
                     original_checked_window_index_misses=[row for row in rows if row['original_checked_window_index_miss']])
    results = {}
    for name, group in groupings.items():
        results[name] = dict(queries=len(group))
        for win in WINDOWS:
            results[name][win] = {}
            for inventory in inventories:
                ids = [row['molecule_id'] for row in group if row[f'{win}_{inventory}']]
                results[name][win][inventory] = dict(count=len(ids), query_ids=ids)
    return results


def truth_cohorts(inputs, targets, identity, known):
    import pyarrow.parquet as pq
    from score_outsider import LABELS, LABELS_SHA, artifact, pinned, read_json

    audit_path = pinned(artifact(inputs, INDEX_AUDIT, 'massbank_union_misses.json'), INDEX_AUDIT_SHA)
    audit = read_json(audit_path)
    label_path = pinned(artifact(inputs, LABELS, 'labels.parquet'), LABELS_SHA)
    label_rows = pq.read_table(label_path, columns=['molecule_id', 'candidate_id', 'truth_smiles']).to_pylist()
    labels = {row['molecule_id']: row for row in label_rows}
    group = audit['groups'][ABSENT_GROUP]
    index_misses = set(group['ids'])
    audited_misses = set(audit['molecules'])
    if (len(label_rows) != 110 or labels.keys() != targets.keys() or audit['panel'] != 110
            or len(audited_misses) != 45 or audit['whole_pipeline_union_misses'] != 45
            or audit['misses_by_membership']['not_found_in_checked_windows'] != 7
            or len(index_misses) != group['count'] or len(index_misses) != 7 or not index_misses <= audited_misses
            or any(audit['molecules'][mid]['membership'] != 'not_found_in_checked_windows' for mid in index_misses)):
        raise ValueError('Archived query population or original miss strata changed')
    for mid, label in sorted(labels.items()):
        truth = identity(label['truth_smiles'])
        if truth is None or truth[0] != label['candidate_id']:
            raise ValueError(f'Archived molecular truth no longer normalizes: {mid}')
        missed = truth not in known[mid]
        if missed != (mid in audited_misses) or (missed and label['candidate_id'] != audit['molecules'][mid]['candidate_id']):
            raise ValueError(f'Original source union differs from the pinned miss audit: {mid}')
        key = truth[0], ','.join(f'{element}:{number}' for element, number in truth[1])
        yield mid, key, missed, mid in index_misses


def score(inputs, out, seal_sha):
    from rdkit import RDLogger
    from c39_request_cache import identity_function
    from scan_coconut_2026_mb110_window import frozen_targets
    from score_outsider import BASE, LABELS, LABELS_SHA, PINS, artifact, pinned, read_json

    RDLogger.DisableLog('rdApp.*')
    identity = identity_function()
    receipt = pinned(artifact(inputs, SCAN, SCAN_REPORT), seal_sha)
    seal = read_json(receipt)
    validate_seal(seal)
    pc_lists = pinned(artifact(inputs, BASE, 'pc_lists.json'), PINS['pc_lists.json'])
    targets, targets_sha = frozen_targets(pc_lists)
    if targets_sha != seal['exact_targets_sha256']:
        raise ValueError('The sealed target centres differ from historical C34')
    new, old = verified_inventory(receipt, seal, targets)
    known, source_counts = read_source_union(inputs, pc_lists, targets, identity)

    # Truth-bearing archives are opened only after all blind and candidate checks above.
    rows = []
    for mid, key, missed, index_miss in truth_cohorts(inputs, targets, identity, known):
        new_row, old_row = new[mid].get(key), old[mid].get(key)
        row = dict(molecule_id=mid, original_source_union_miss=missed, original_checked_window_index_miss=index_miss)
        for win in WINDOWS:
            row.update({f'{win}_{name}': flag for name, flag in presence(new_row, old_row, win == 'author').items()})
        rows.append(row)
    out.mkdir(exist_ok=True, parents=True)
    per_query = out / (OUTPUT + '_per_query.csv')
    with per_query.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(status='complete', scope='Frozen structural presence before ranking. No final candidate changes.',
                  query_count=len(rows), scan=SCAN, scan_receipt_sha256=seal_sha,
                  labels=LABELS, labels_sha256=LABELS_SHA, original_index_audit=INDEX_AUDIT,
                  original_index_audit_sha256=INDEX_AUDIT_SHA, source_candidates=dict(source_counts),
                  exact_targets_sha256=targets_sha, blind_inventory_totals=seal['totals'],
                  cohorts=report_coverage(rows), per_query_sha256=sha256(per_query.read_bytes()).hexdigest(),
                  new_predictor_requests=0)
    (out / (OUTPUT + '.json')).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({name: {win: {source: entry['count'] for source, entry in section[win].items()}
                             for win in WINDOWS} for name, section in report['cohorts'].items()}), flush=True)


def build(seal, out):
    from build import HERE, IMAGE, METRIC_BOOTSTRAP, WHEEL, pack
    from build_final_native_outsider_mb110_score import BASE, LABELS

    raw = seal.read_bytes()
    validate_seal(json.loads(raw))
    digest = sha256(raw).hexdigest()
    names = ('c39_request_cache.py', 'generate.py', 'rebuild_parents.py', 'scan_rebuilt_parent_window.py',
             'scan_coconut_2026_window.py', 'scan_coconut_2026_mb110_window.py')
    sources = {name: (HERE / name).read_bytes() for name in names}
    sources['score_coconut_mb110.py'] = Path(__file__).read_bytes()
    sources['score_outsider.py'] = (HERE / 'build_final_native_outsider_mb110_score.py').read_bytes()
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
with tempfile.TemporaryDirectory(prefix='score-coconut-mb110-') as temp:
    scratch = Path(temp)
    for name, (blob, expected) in {payload!r}.items():
        data = zlib.decompress(base64.b64decode(blob))
        if sha256(data).hexdigest() != expected:
            raise ValueError(f'Packaged source changed: {{name}}')
        (scratch / name).write_bytes(data)
    sys.path.insert(0, temp)
    from score_coconut_mb110 import score
    score(inputs, Path('/kaggle/working'), {digest!r})
'''
    compile(code, SLUG, 'exec')
    meta = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='run_score.py', language='python', kernel_type='script',
                is_private=True, enable_gpu=False, enable_internet=False, docker_image=IMAGE,
                dataset_sources=[WHEEL], competition_sources=[], kernel_sources=[SCAN, BASE, INDEX_AUDIT, LABELS])
    out.mkdir(exist_ok=True, parents=True)
    (out / 'run_score.py').write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return sha256(code.encode()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    print(build(**vars(parser.parse_args())))

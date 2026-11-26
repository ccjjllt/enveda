"""Score frozen old and rebuilt C34 windows using existing private assessment outputs."""

from collections import Counter, defaultdict
import csv
from hashlib import sha256
from io import StringIO
import json
from math import isclose, isfinite
from pathlib import Path
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

import generate
from scan_rebuilt_parent_window import FROZEN_TARGETS_SHA, SCHEMA, WINDOW, read_legacy


SCAN_RECEIPT = 'rebuilt_parent_window_receipt.json'
MERGE_RECEIPT = 'rebuilt_parent_receipt.json'
MERGE_TABLE = 'rebuilt_parent_meta.parquet'
SUMMARY = 'rebuilt_parent_window_score.json'
PER_QUERY = 'rebuilt_parent_window_score_per_query.csv'
LEGACY = 'jiyujia/enveda-075-np-virtual-window-full-main-window/1'
BASE = 'jiyujia/enveda-075-np-recall-reseal/2'
RANKS = 'jiyujia/enveda-075-np-recall-compare/1'
AUDIT = 'jiyujia/enveda-075-np-native-index-coverage/1'
PINS = {
    'baseline.zip': '05509b32cfb9bcbe8ab563244f91c97e89f878eba3cc40263e3f48fd4a1f372d',
    'submission.csv': '1dcb467d3c73a5aa983d9321e1193c7bd1fec6884ef277b7ee3486485ec153a4',
    'per_query.csv': '27ea12de0b59b2321a2e2d2ae27cc4620f6bcae6011c36a92a5bc8424224561d',
    'metrics.json': 'a839d5ce6260d44cd115636ea3488f85d1aab2287476e2ec3256587734f05c35',
    'native_window_coverage.json': '3d75c6791092b1d46718d3ac7247494807476379aefd246f7939353e53e2190f',
    'labels': 'ab7a98f7f5e35fd9029ebc905de279d25adde894f2ea49949066a72a1b2cb1fa',
    'queries': '2a8bf747f50e6920fcd4f24e04302b721281f8794c947cc58fe545264b12d0d4',
    'c34_pc': 'd7b28154b1bdf132528300bcf8faa1211984c843d5340ac81191472a452f564b',
    'c34_targets': '3cc377c3dd4d5023c3c6d8d13d26ece73af76dd6e19c5ba556284bfe7b0e80f4',
}
STAGES = ('main', 'engine2', 'pubchem', 'prefusion', 'fused_overflow')
AUDIT_CLASSES = Counter({'lost_after_reported_candidate': 2, 'native_window_not_shortlisted': 8,
                         'absent_from_scanned_indices': 2})


def mounted(inputs, ref, name):
    path, = (item for item in inputs.rglob(name) if ref.split('/')[1] in item.parts)
    return path


def verify(path, digest):
    if generate.file_hash(path) != digest:
        raise ValueError(f'Archive checksum changed: {path.name}')
    return path


def lines(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def by_mid(rows):
    lookup = {row['molecule_id']: row for row in rows}
    if len(rows) != 60 or len(lookup) != 60:
        raise ValueError('Expected 60 unique frozen query IDs')
    return lookup


def narrow(error):
    return generate.AUTHOR_PPM[0] <= error <= generate.AUTHOR_PPM[1]


def read_blind(inputs, scan_ref, merge_ref, scan_sha):
    scan_path = verify(mounted(inputs, scan_ref, SCAN_RECEIPT), scan_sha)
    seal = json.loads(scan_path.read_bytes())
    expected = dict(status='complete', scope='rebuilt-parent-direct-window', query_count=60,
                    rdkit=generate.RDKit_VERSION, archived_c34='jiyujia/enveda-075-np-recall-base/2',
                    archived_pc_lists_sha256=PINS['c34_pc'], exact_targets_sha256=FROZEN_TARGETS_SHA,
                    broad_window_ppm=generate.C34_PPM, author_window_ppm=list(generate.AUTHOR_PPM),
                    legacy_pool_sha256=generate.POOL_SHA256)
    if any(seal[field] != value for field, value in expected.items()):
        raise ValueError('Unexpected frozen C34 scan scope or input')
    panel = seal['cases']
    old = verify(mounted(inputs, LEGACY, 'blind_virtual_np_receipt.json'), seal['legacy_window_receipt_sha256'])
    if generate.file_hash(old.with_name('known_main.parquet')) != seal['legacy_window_sha256']:
        raise ValueError('Legacy exact-window membership changed')
    targets, old_broad, old_narrow = read_legacy(old.parent, PINS['c34_pc'], PINS['c34_targets'])
    if set(targets) != set(panel) or any(row['target_mass'] != targets[mid] for mid, row in panel.items()):
        raise ValueError('The frozen blind query centres or IDs changed')

    merge_path = verify(mounted(inputs, merge_ref, MERGE_RECEIPT), seal['rebuilt_receipt_sha256'])
    merge = json.loads(merge_path.read_bytes())
    if (merge['status'] != 'complete' or merge['version'] != 2 or merge['rdkit'] != generate.RDKit_VERSION
            or merge['identities'] != seal['rebuilt_inventory_rows'] or merge['source_bits'] != seal['source_bits']
            or merge['output_sha256'] != seal['rebuilt_inventory_sha256']
            or merge['legacy_pool_sha256'] != generate.POOL_SHA256):
        raise ValueError('Different rebuilt inventory from the blind scan')
    inventory = pq.ParquetFile(verify(merge_path.with_name(MERGE_TABLE), merge['output_sha256']))
    if inventory.metadata.num_rows != merge['identities']:
        raise ValueError('Rebuilt inventory row count changed')

    rebuilt = {mid: {} for mid in panel}
    window = pq.ParquetFile(verify(scan_path.with_name(WINDOW), seal['outputs'][WINDOW]))
    if not window.schema_arrow.equals(SCHEMA, check_metadata=False):
        raise ValueError('Unexpected rebuilt window identity schema')
    for batch in window.iter_batches(batch_size=32768):
        for row in batch.to_pylist():
            mid = row['molecule_id']
            identity = (row['score_key'], row['heavy_atoms'])
            mass, target = row['mass'], targets[mid]
            tolerance = target * generate.C34_PPM * 1e-6
            if (len(identity[0]) != 14 or not identity[1] or not isfinite(mass) or mass <= 0
                    or not target - tolerance <= mass <= target + tolerance):
                raise ValueError(f'Invalid rebuilt window row: {mid}')
            error = (target - mass) / mass * 1e6
            if (identity in rebuilt[mid] or str(row['src']) not in merge['source_bits']
                    or not isclose(row['mass_error_ppm'], error, rel_tol=0, abs_tol=1e-10)
                    or row['in_author_window'] != narrow(error)
                    or row['legacy_same_window'] != (identity in old_broad[mid])
                    or row['legacy_same_author_window'] != (identity in old_narrow[mid])):
                raise ValueError(f'Conflicting rebuilt window membership: {mid}')
            rebuilt[mid][identity] = row
    for mid, row in panel.items():
        wide = set(rebuilt[mid])
        tight = {identity for identity, item in rebuilt[mid].items() if item['in_author_window']}
        checks = dict(rebuilt_window=len(wide), legacy_window=len(old_broad[mid]),
                      shared_window=len(wide & old_broad[mid]), rebuilt_only_window=len(wide - old_broad[mid]),
                      legacy_only_window=len(old_broad[mid] - wide), rebuilt_author_window=len(tight),
                      legacy_author_window=len(old_narrow[mid]), shared_author_window=len(tight & old_narrow[mid]),
                      rebuilt_only_author_window=len(tight - old_narrow[mid]),
                      legacy_only_author_window=len(old_narrow[mid] - tight),
                      rebuilt_no_legacy_stored_key=sum(not item['legacy_stored_key_global'] for item in rebuilt[mid].values()),
                      rebuilt_extra_with_legacy_stored_key=sum(item['legacy_stored_key_global'] and
                                                               not item['legacy_same_window'] for item in rebuilt[mid].values()))
        by_src = Counter(str(item['src']) for item in rebuilt[mid].values())
        if any(row[field] != value for field, value in checks.items()) or row['rebuilt_window_by_src'] != dict(by_src):
            raise ValueError(f'Window candidate accounting differs from the blind receipt: {mid}')
    totals = Counter()
    for row in panel.values():
        totals.update({field: row[field] for field in seal['totals']})
    if dict(totals) != seal['totals']:
        raise ValueError('Full-window candidate totals differ from the blind receipt')
    return seal, inventory, targets, old_broad, old_narrow, rebuilt


def original_rankings(inputs):
    archive_path = verify(mounted(inputs, BASE, 'baseline.zip'), PINS['baseline.zip'])
    with ZipFile(archive_path) as archive:
        receipt = json.loads(archive.read('pilot_arm_receipt.json'))
        buffer = archive.read('submission.csv')
        if (receipt['arm'] != 'baseline' or receipt['molecules'] != 60 or receipt['candidate_slots'] != 25
                or receipt['queries_sha256'] != PINS['queries']
                or receipt['hashes']['submission.csv'] != PINS['submission.csv']
                or sha256(buffer).hexdigest() != PINS['submission.csv']):
            raise ValueError('Mismatch in original final-list archive')
    return by_mid(list(csv.DictReader(StringIO(buffer.decode()))))


def audit_composition(pairs):
    elements = {Chem.GetPeriodicTable().GetElementSymbol(number): count for number, count in pairs}
    if not elements or len(elements) != len(pairs) or any(number <= 1 or count <= 0 for number, count in pairs):
        raise ValueError('Malformed old normalized composition')
    return ','.join(f'{element}:{count}' for element, count in sorted(elements.items()))


def read_truth(inputs, targets):
    ranked = original_rankings(inputs)
    scored = by_mid(lines(verify(mounted(inputs, RANKS, 'per_query.csv'), PINS['per_query.csv'])))
    rank_metrics = json.loads(verify(mounted(inputs, RANKS, 'metrics.json'), PINS['metrics.json']).read_bytes())
    audit = json.loads(verify(mounted(inputs, AUDIT, 'native_window_coverage.json'),
                              PINS['native_window_coverage.json']).read_bytes())
    if (set(scored) != set(ranked) or set(scored) != set(targets) or audit['molecules'] != 60
            or rank_metrics['molecules'] != 60 or rank_metrics['queries_sha256'] != PINS['queries']
            or audit['query_sha256'] != PINS['queries'] or rank_metrics['labels_sha256'] != PINS['labels']
            or audit['labels_sha256'] != PINS['labels'] or tuple(audit['stages']) != STAGES
            or audit['baseline_raw_submission_sha256'] != PINS['submission.csv']
            or audit['c34_archived_pc_lists_sha256'] != PINS['c34_pc'] or audit['rdkit'] != generate.RDKit_VERSION):
        raise ValueError('Historical rank and coverage artifacts describe different panels')
    misses = {mid for mid, row in scored.items() if int(row['baseline_rank']) == 0}
    old_misses = audit['by_query']
    if (misses != set(old_misses) or audit['final_hits'] != 48 or audit['final_misses'] != 12
            or Counter(row['class'] for row in old_misses.values()) != AUDIT_CLASSES
            or audit['classification'] != dict(AUDIT_CLASSES)):
        raise ValueError('Historical final misses differ from the archived index audit')

    generate.TAUTOMER = rdMolStandardize.TautomerEnumerator()
    truth = {}
    for mid, row in scored.items():
        rank = int(row['baseline_rank'])
        candidates = ranked[mid]['smiles'].split(';') if ranked[mid]['smiles'] else []
        if not 0 <= rank <= len(candidates) <= 25 or not row['source'] or len(row['truth_key']) != 14:
            raise ValueError(f'Malformed historical assessment row: {mid}')
        if rank:
            identity = generate.normalize(candidates[rank - 1])
            group = 'original_final_hit'
            if identity is None or any(generate.normalize(text) == identity for text in candidates[:rank - 1]):
                raise ValueError(f'Historical exact identity fails at original hit position: {mid}')
        else:
            item = old_misses[mid]
            group = item['class']
            any_source_rank = any(item['source_ranks'].values())
            identity = item['normalized_key'], audit_composition(item['composition'])
            if (item['challenge'] != mid or item['source'] != row['source']
                    or not isclose(item['windows']['native_C34'], targets[mid], rel_tol=0, abs_tol=1e-10)
                    or set(item['source_ranks']) != set(STAGES)
                    or any_source_rank != (group == 'lost_after_reported_candidate')
                    or any(generate.normalize(text) == identity for text in candidates)):
                raise ValueError(f'Different archived scope, source or original final miss: {mid}')
            if group == 'absent_from_scanned_indices' and any(
                    record['indexed_matches'] for record in
                    [item['engine1'], *item['pubchem'].values(), *item['engine2_inventory'].values()]):
                raise ValueError(f'Whole-library absence is not confirmed by the archived scan: {mid}')
        if identity[0] != row['truth_key']:
            raise ValueError(f'Historical normalized answer keys disagree: {mid}')
        truth[mid] = dict(identity=identity, source=row['source'], original_final_rank=rank, group=group)
    if len({row['identity'] for row in truth.values()}) != 60:
        raise ValueError('Archived normalized truth identities are not distinct')
    return truth


def inventory_matches(inventory, identities):
    known = {}
    columns = ['key', 'heavy_atoms', 'mass', 'formula', 'formal_charge', 'src']
    keys = pa.array(sorted({identity[0] for identity in identities}))
    for batch in inventory.iter_batches(columns=columns, batch_size=32768):
        for row in batch.filter(pc.is_in(batch.column(0), value_set=keys)).to_pylist():
            identity = row['key'], row['heavy_atoms']
            if identity not in identities:
                continue
            if identity in known:
                raise ValueError('Exact assessed identity duplicated in the rebuilt inventory')
            known[identity] = row
    return known


def aggregate(rows):
    counts = dict(queries=len(rows), rebuilt_global=sum(row['rebuilt_global'] for row in rows),
                  rebuilt_global_outside_broad=sum(row['rebuilt_global'] and not row['rebuilt_broad'] for row in rows),
                  rebuilt_in_broad_outside_narrow=sum(row['rebuilt_broad'] and not row['rebuilt_narrow'] for row in rows),
                  rebuilt_global_only_train=sum(row['rebuilt_source_bits'] == 1 for row in rows),
                  rebuilt_global_only_coconut=sum(row['rebuilt_source_bits'] == 2 for row in rows),
                  rebuilt_global_train_and_coconut=sum(row['rebuilt_source_bits'] == 3 for row in rows))
    for window in ('broad', 'narrow'):
        old, new = f'legacy_{window}', f'rebuilt_{window}'
        gained = sorted(row['query'] for row in rows if not row[old] and row[new])
        lost = sorted(row['query'] for row in rows if row[old] and not row[new])
        counts[window] = dict(legacy=sum(row[old] for row in rows), rebuilt=sum(row[new] for row in rows),
                              both=sum(row[old] and row[new] for row in rows), gained=len(gained), lost=len(lost),
                              gained_queries=gained, lost_queries=lost,
                              legacy_candidate_slots=sum(row[f'legacy_{window}_candidates'] for row in rows),
                              rebuilt_candidate_slots=sum(row[f'rebuilt_{window}_candidates'] for row in rows))
    return counts


def score(inputs, out, scan_ref, merge_ref, scan_sha):
    if rdBase.rdkitVersion != generate.RDKit_VERSION:
        raise ValueError('Expected the official contest metric version')
    RDLogger.DisableLog('rdApp.*')
    seal, inventory, targets, old_broad, old_narrow, rebuilt = read_blind(inputs, scan_ref, merge_ref, scan_sha)
    truth = read_truth(inputs, targets)
    known = inventory_matches(inventory, {row['identity'] for row in truth.values()})
    rows = []
    for mid, record in sorted(truth.items()):
        identity = record['identity']
        hit, global_hit = rebuilt[mid].get(identity), known.get(identity)
        mass = global_hit['mass'] if global_hit is not None else None
        target = targets[mid]
        tolerance = target * generate.C34_PPM * 1e-6
        actual_in_window = mass is not None and target - tolerance <= mass <= target + tolerance
        if bool(hit) != actual_in_window or (hit and any(hit[field] != global_hit[field] for field in
                                                          ('mass', 'formula', 'formal_charge', 'src'))):
            raise ValueError(f'Exact truth identity differs between inventory and blind window: {mid}')
        counts = seal['cases'][mid]
        rows.append(dict(query=mid, source=record['source'], original_final_rank=record['original_final_rank'],
                         historical_group=record['group'], rebuilt_global=global_hit is not None,
                         rebuilt_source_bits=0 if global_hit is None else global_hit['src'],
                         rebuilt_mass_error_ppm=None if mass is None else (target - mass) / mass * 1e6,
                         rebuilt_legacy_stored_key_present=None if hit is None else hit['legacy_stored_key_global'],
                         legacy_broad=identity in old_broad[mid], legacy_narrow=identity in old_narrow[mid],
                         rebuilt_broad=bool(hit), rebuilt_narrow=bool(hit and hit['in_author_window']),
                         legacy_broad_candidates=counts['legacy_window'], legacy_narrow_candidates=counts['legacy_author_window'],
                         rebuilt_broad_candidates=counts['rebuilt_window'], rebuilt_narrow_candidates=counts['rebuilt_author_window']))

    groups = defaultdict(list)
    sources = defaultdict(list)
    for row in rows:
        groups[row['historical_group']].append(row)
        sources[row['source']].append(row)
    groups['original_final_miss'] = [row for row in rows if not row['original_final_rank']]
    groups['all_original_sources_missed'] = groups['absent_from_scanned_indices'] + groups['native_window_not_shortlisted']
    out.mkdir(parents=True, exist_ok=True)
    per = out / PER_QUERY
    with per.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(status='complete', scope='direct index coverage; no candidate ranking or fusion',
                  scan_ref=scan_ref, merge_ref=merge_ref, scan_seal_sha256=scan_sha,
                  legacy_window_ref=LEGACY, legacy_window_sha256=seal['legacy_window_sha256'],
                  rebuilt_inventory_sha256=seal['rebuilt_inventory_sha256'], historical_scores_ref=RANKS,
                  historical_catalogue_audit_ref=AUDIT, historical_baseline_ref=BASE,
                  broad_window_ppm=generate.C34_PPM, author_window_ppm=generate.AUTHOR_PPM,
                  identity='RDKit 2026.03.3 normalized K14 and exact heavy-atom composition',
                  all=aggregate(rows), by_group={group: aggregate(items) for group, items in sorted(groups.items())},
                  by_source={source: aggregate(items) for source, items in sorted(sources.items())},
                  output_sha256={PER_QUERY: generate.file_hash(per)})
    (out / SUMMARY).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('REBUILT_PARENT_WINDOW_SCORE', json.dumps(result['all'], allow_nan=False), flush=True)

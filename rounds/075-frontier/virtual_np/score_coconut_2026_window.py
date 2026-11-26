"""Independently compare the frozen C34 windows before scoring exact identities."""

from collections import Counter, defaultdict
import csv
import json
from math import isclose, isfinite
import re


SCAN_2026 = 'jiyujia/enveda-075-coconut-2026-window/1'
SCAN_2024 = 'jiyujia/enveda-075-rebuilt-parent-direct-window/1'
REPORT_2026 = 'coconut_2026_window_receipt.json'
REPORT_2024 = 'rebuilt_parent_window_receipt.json'
SCOPE_2026 = 'coconut-2026-exact-direct-structure-window'
SCOPE_2024 = 'rebuilt-parent-direct-window'
LITE_STAGE_SHA = '53ceba25df0272f48a283d30fdec1e8dc05f79333fe5b7e3cfca1582cb15137c'
LITE_ARCHIVE_SHA = '81f3edc1731f01da289fe6b84d162667863e331d52041a781565e1af2eb79939'
SUMMARY = 'coconut_2026_window_score.json'
PER_QUERY = 'coconut_2026_window_score_per_query.csv'
CHANNELS = ('legacy', 'rebuilt_2024', 'coconut_2026')
WINDOWS = ('broad', 'narrow')


def window_seal(path, scope):
    seal = json.loads(path.read_bytes())
    fields = dict(status='complete', scope=scope, query_count=60, rdkit='2026.03.3',
                  broad_window_ppm=10, author_window_ppm=[-3.0, 4.6],
                  archived_c34='jiyujia/enveda-075-np-recall-base/2')
    if scope == SCOPE_2026:
        fields.update(stage_receipt_sha256=LITE_STAGE_SHA, archive_sha256=LITE_ARCHIVE_SHA,
                      csv_member='coconut_csv_lite-11-2026.csv')
    if any(seal.get(name) != value for name, value in fields.items()) or len(seal['cases']) != 60:
        raise ValueError(f'Unexpected scope or population for {path.name}')
    named = ({'coconut_2026_window.parquet', 'coconut_2026_window_sources.parquet'} if scope == SCOPE_2026
             else {'rebuilt_parent_window.parquet'})
    if set(seal['outputs']) != named or any(not re.fullmatch(r'[0-9a-f]{64}', h) for h in seal['outputs'].values()):
        raise ValueError(f'Invalid output checksum manifest in {path.name}')
    return seal


def read_window(path, schema, targets, old_broad, old_narrow):
    from pyarrow.parquet import ParquetFile
    from generate import C34_PPM
    from score_rebuilt_parent_window import narrow

    table = ParquetFile(path)
    if not table.schema_arrow.equals(schema, check_metadata=False):
        raise ValueError(f'Unexpected window schema: {path.name}')
    by_query = {mid: {} for mid in targets}
    for batch in table.iter_batches(batch_size=32768):
        for row in batch.to_pylist():
            mid = row['molecule_id']
            ident = row['score_key'], row['heavy_atoms']
            mass, centre = row['mass'], targets[mid]
            tolerance = centre * C34_PPM * 1e-6
            if (not re.fullmatch(r'[A-Z]{14}', ident[0]) or not ident[1]
                    or ident in by_query[mid] or not isfinite(mass) or mass <= 0
                    or not centre - tolerance <= mass <= centre + tolerance):
                raise ValueError(f'Invalid or duplicated exact window identity in {path.name}: {mid}')
            error = (centre - mass) / mass * 1e6
            if (not isclose(row['mass_error_ppm'], error, rel_tol=0, abs_tol=1e-10)
                    or row['in_author_window'] != narrow(error)
                    or row['legacy_same_window'] != (ident in old_broad[mid])
                    or row['legacy_same_author_window'] != (ident in old_narrow[mid])):
                raise ValueError(f'Conflicting window geometry or legacy overlap in {path.name}: {mid}')
            by_query[mid][ident] = row
    return by_query


def check_2026_sources(path, schema, rows, targets, seal):
    from pyarrow.parquet import ParquetFile
    from generate import C34_PPM
    from score_rebuilt_parent_window import narrow

    table = ParquetFile(path)
    if not table.schema_arrow.equals(schema, check_metadata=False):
        raise ValueError('Unexpected COCONUT 2026 source-row schema')
    names = ['source_row', 'score_key', 'heavy_atoms', 'mass', 'query_ids', 'author_window_query_ids']
    expected_reps = {(mid, identity): row for mid, mapping in rows.items() for identity, row in mapping.items()}
    occurrences, representatives = Counter(), set()
    seen_rows = set()
    for batch in table.iter_batches(columns=names, batch_size=32768):
        for item in batch.to_pylist():
            row_no = item['source_row']
            ident = item['score_key'], item['heavy_atoms']
            mass = item['mass']
            query_ids, narrow_ids = item['query_ids'], item['author_window_query_ids']
            if (row_no <= 0 or row_no in seen_rows or not isfinite(mass) or mass <= 0 or not query_ids
                    or len(query_ids) != len(set(query_ids)) or len(narrow_ids) != len(set(narrow_ids))
                    or not set(narrow_ids) <= set(query_ids) <= targets.keys()):
                raise ValueError('Malformed or repeated COCONUT 2026 source assignment')
            seen_rows.add(row_no)
            for mid in query_ids:
                target = targets[mid]
                tolerance = target * C34_PPM * 1e-6
                lookup = mid, ident
                representative = expected_reps.get(lookup)
                if (representative is None or not target - tolerance <= mass <= target + tolerance
                        or ((mid in narrow_ids) != narrow((target - mass) / mass * 1e6))
                        or (mid in narrow_ids and not representative['in_author_window'])):
                    raise ValueError(f'COCONUT 2026 source and representative disagree: {mid}')
                if representative['source_row'] == row_no:
                    if representative['mass'] != mass or representative['in_author_window'] != (mid in narrow_ids):
                        raise ValueError(f'COCONUT 2026 representative mass changed: {mid}')
                    representatives.add(lookup)
                occurrences[lookup] += 1
    source_counts = seal['source_counts']
    total = sum(occurrences.values())
    if (len(seen_rows) != source_counts['valid_window_source_rows']
            or total != source_counts['window_source_query_occurrences']
            or total - len(expected_reps) != seal['source_query_duplicates']
            or representatives != expected_reps.keys()
            or any(row['source_records_in_window'] != occurrences[key] for key, row in expected_reps.items())):
        raise ValueError('COCONUT 2026 source occurrence counts do not match the blind receipt and windows')


def geometry(rows):
    return {mid: (set(mapping), {ident for ident, row in mapping.items() if row['in_author_window']})
            for mid, mapping in rows.items()}


def check_counts(rebuilt, coconut, targets, old_broad, old_narrow, seal24, seal26):
    modern, earlier = geometry(coconut), geometry(rebuilt)
    total24, total26, sources24 = Counter(), Counter(), Counter()
    for mid, mass in targets.items():
        broad_legacy, narrow_legacy = old_broad[mid], old_narrow[mid]
        broad_24, narrow_24 = earlier[mid]
        broad_26, narrow_26 = modern[mid]
        s24 = dict(rebuilt_window=len(broad_24), legacy_window=len(broad_legacy),
                   shared_window=len(broad_24 & broad_legacy), rebuilt_only_window=len(broad_24 - broad_legacy),
                   legacy_only_window=len(broad_legacy - broad_24), rebuilt_author_window=len(narrow_24),
                   legacy_author_window=len(narrow_legacy), shared_author_window=len(narrow_24 & narrow_legacy),
                   rebuilt_only_author_window=len(narrow_24 - narrow_legacy),
                   legacy_only_author_window=len(narrow_legacy - narrow_24),
                   rebuilt_no_legacy_stored_key=sum(not row['legacy_stored_key_global'] for row in rebuilt[mid].values()),
                   rebuilt_extra_with_legacy_stored_key=sum(row['legacy_stored_key_global'] and not row['legacy_same_window']
                                                            for row in rebuilt[mid].values()))
        s26 = dict(coconut_broad=len(broad_26), legacy_broad=len(broad_legacy),
                   shared_broad=len(broad_26 & broad_legacy), added_to_legacy_broad=len(broad_26 - broad_legacy),
                   coconut_author=len(narrow_26), legacy_author=len(narrow_legacy),
                   shared_author=len(narrow_26 & narrow_legacy), added_to_legacy_author=len(narrow_26 - narrow_legacy))
        by_src = Counter(str(row['src']) for row in rebuilt[mid].values())
        if (not set(by_src) <= {'1', '2', '3'}
                or seal24['cases'][mid] != dict(target_mass=mass, rebuilt_window_by_src=dict(by_src), **s24)
                or seal26['cases'][mid] != dict(target_mass=mass, **s26)):
            raise ValueError(f'Window candidate counts differ from the emitted blind receipts: {mid}')
        total24.update(s24)
        total26.update(s26)
        sources24.update(by_src)
    if (dict(total24) != seal24['totals'] or dict(total26) != seal26['totals']
            or dict(sources24) != seal24['rebuilt_window_by_src']):
        raise ValueError('Full candidate counts differ from the two blind receipts')
    return earlier, modern


def aggregate(rows):
    result = {'queries': len(rows)}
    for window in WINDOWS:
        channels = {name: {row['query'] for row in rows if row[f'{name}_{window}_hit']} for name in CHANNELS}
        hits26 = channels['coconut_2026']
        union24 = channels['legacy'] | channels['rebuilt_2024']
        union26 = union24 | hits26
        final_slots = sum(row[f'union_{window}_candidates'] for row in rows)
        extra_2026 = sum(row[f'added_to_union_2024_{window}_candidates'] for row in rows)
        result[window] = dict(hits={name: len(values) for name, values in channels.items()},
                              candidate_slots={name: sum(row[f'{name}_{window}_candidates'] for row in rows)
                                               for name in CHANNELS},
                              added_candidates_vs={name: sum(row[f'added_to_{name}_{window}_candidates'] for row in rows)
                                                   for name in CHANNELS[:-1]}, comparisons={})
        result[window]['additive_union'] = dict(
            legacy_and_2024_hits=len(union24), legacy_and_2024_candidate_slots=final_slots - extra_2026,
            all_three_hits=len(union26), all_three_candidate_slots=final_slots,
            candidate_slots_added_vs_legacy=final_slots - result[window]['candidate_slots']['legacy'],
            gained_vs_legacy=len(union26 - channels['legacy']), lost_vs_legacy=0,
            gained_vs_legacy_queries=sorted(union26 - channels['legacy']),
            candidate_slots_increment_from_2026=extra_2026,
            gained_from_2026=len(hits26 - union24), gained_from_2026_queries=sorted(hits26 - union24))
        for name in CHANNELS[:-1]:
            before = channels[name]
            gained, lost = sorted(hits26 - before), sorted(before - hits26)
            result[window]['comparisons'][f'coconut_2026_vs_{name}'] = dict(
                shared=len(before & hits26), gained=len(gained), lost=len(lost),
                gained_queries=gained, lost_queries=lost)
    return result


def score(inputs, out, scan26, scan24, sha26, sha24):
    from rdkit import RDLogger, rdBase
    from generate import AUTHOR_PPM, C34_PPM, POOL_SHA256, RDKit_VERSION, file_hash
    from scan_coconut_2026_window import RECORDS, RECORD_SCHEMA, WINDOW as WINDOW26, WINDOW_SCHEMA
    from scan_rebuilt_parent_window import FROZEN_TARGETS_SHA, SCHEMA as SCHEMA24, WINDOW as WINDOW24
    from score_rebuilt_parent_window import AUDIT, BASE, LEGACY, PINS, RANKS, mounted, read_legacy, read_truth, verify

    if rdBase.rdkitVersion != RDKit_VERSION:
        raise ValueError('Expected the official contest metric version')
    RDLogger.DisableLog('rdApp.*')
    path26 = verify(mounted(inputs, scan26, REPORT_2026), sha26)
    path24 = verify(mounted(inputs, scan24, REPORT_2024), sha24)
    seal26, seal24 = window_seal(path26, SCOPE_2026), window_seal(path24, SCOPE_2024)
    common = dict(archived_pc_lists_sha256=PINS['c34_pc'], exact_targets_sha256=FROZEN_TARGETS_SHA,
                  legacy_window_receipt_sha256=seal26['legacy_window_receipt_sha256'],
                  legacy_window_sha256=seal26['legacy_window_sha256'])
    if (any(seal[field] != value for seal in (seal24, seal26) for field, value in common.items())
            or seal24['legacy_pool_sha256'] != POOL_SHA256):
        raise ValueError('The two snapshots do not describe the same historical input')
    legacy_path = verify(mounted(inputs, LEGACY, 'blind_virtual_np_receipt.json'), common['legacy_window_receipt_sha256'])
    verify(legacy_path.with_name('known_main.parquet'), common['legacy_window_sha256'])
    targets, old_broad, old_narrow = read_legacy(legacy_path.parent, PINS['c34_pc'], PINS['c34_targets'])
    if any(set(seal['cases']) != targets.keys() for seal in (seal24, seal26)):
        raise ValueError('The two snapshots do not contain the frozen legacy query IDs')
    coconut = read_window(verify(path26.with_name(WINDOW26), seal26['outputs'][WINDOW26]),
                          WINDOW_SCHEMA, targets, old_broad, old_narrow)
    rebuilt = read_window(verify(path24.with_name(WINDOW24), seal24['outputs'][WINDOW24]),
                          SCHEMA24, targets, old_broad, old_narrow)
    check_2026_sources(verify(path26.with_name(RECORDS), seal26['outputs'][RECORDS]), RECORD_SCHEMA, coconut, targets, seal26)
    earlier, modern = check_counts(rebuilt, coconut, targets, old_broad, old_narrow, seal24, seal26)

    # Original private assessment is first read after verifying both independent scans and their invariant geometry.
    truth = read_truth(inputs, targets)
    rows, groups, sources = [], defaultdict(list), defaultdict(list)
    for mid, item in sorted(truth.items()):
        row = dict(query=mid, source=item['source'], original_final_rank=item['original_final_rank'],
                   historical_group=item['group'])
        for window, legacy, prev, current in zip(WINDOWS, (old_broad[mid], old_narrow[mid]),
                                                 earlier[mid], modern[mid], strict=True):
            for channel, candidates in zip(CHANNELS, (legacy, prev, current), strict=True):
                row[f'{channel}_{window}_hit'] = item['identity'] in candidates
                row[f'{channel}_{window}_candidates'] = len(candidates)
            row[f'added_to_legacy_{window}_candidates'] = len(current - legacy)
            row[f'added_to_rebuilt_2024_{window}_candidates'] = len(current - prev)
            row[f'added_to_union_2024_{window}_candidates'] = len(current - (legacy | prev))
            row[f'union_{window}_candidates'] = len(legacy | prev | current)
        rows.append(row)
        groups[item['group']].append(row)
        sources[item['source']].append(row)
    groups['original_final_miss'] = [row for row in rows if not row['original_final_rank']]
    groups['other_original_final_miss'] = [row for row in groups['original_final_miss']
                                           if row['historical_group'] != 'absent_from_scanned_indices']
    groups['all_original_sources_missed'] = groups['absent_from_scanned_indices'] + groups['native_window_not_shortlisted']
    out.mkdir(parents=True, exist_ok=True)
    per_query = out / PER_QUERY
    with per_query.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(status='complete', scope='direct exact window coverage and candidate counts',
                  limitations='2024 rebuilt includes training identities. Off-window absence and 2026 provenance are not inferred.',
                  baseline_ref=BASE, historical_scores_ref=RANKS, historical_catalogue_audit_ref=AUDIT,
                  legacy_window_ref=LEGACY, legacy_window_sha256=common['legacy_window_sha256'],
                  scan_2026_ref=scan26, scan_2026_receipt_sha256=sha26, scan_2026_artifacts=seal26['outputs'],
                  coconut_2026_lite_archive_sha256=seal26['archive_sha256'],
                  scan_2024_ref=scan24, scan_2024_receipt_sha256=sha24, scan_2024_artifacts=seal24['outputs'],
                  broad_window_ppm=C34_PPM, author_window_ppm=AUTHOR_PPM, exact_identity='tautomer K14 + heavy composition',
                  original_final_candidate_cap=25, assigned_candidate_slots=0, new_predictor_requests=0,
                  all=aggregate(rows), by_group={key: aggregate(group) for key, group in sorted(groups.items())},
                  by_source={key: aggregate(group) for key, group in sorted(sources.items())},
                  output_sha256={PER_QUERY: file_hash(per_query)})
    (out / SUMMARY).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('COCONUT_2026_WINDOW_SCORE', json.dumps(result['all'], allow_nan=False), flush=True)

"""Unify the original windows and the corrected single-edit pass without reranking."""

from collections import Counter
from hashlib import file_digest, sha256
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


FIXED_EDIT = 'carboxyl_CH3'
BRANCHES = ('proposals', 'known_main')


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def checked(path):
    primary = json.loads((path / 'blind_virtual_np_receipt.json').read_text())
    exact = json.loads((path / 'receipt.json').read_text())
    for name in ('proposals.parquet', 'known_main.parquet', 'stats.parquet'):
        observed = digest(path / name)
        if primary['output_sha256'][name] != observed or exact['outputs'][name] != observed:
            raise ValueError(f'Input candidate artifact changed: {name}')
    return primary, exact


def ordering(row):
    return not row['in_author_window'], abs(row['mass_error_ppm']), row['smiles']


def merge_list(untouched, fixed):
    output = {}
    for rows, supplement in ((untouched, False), (fixed, True)):
        seen = set()
        for row in rows:
            key = row['score_key'], row['heavy_atoms']
            if key in seen:
                raise ValueError('Duplicate normalization identity in an input list')
            seen.add(key)
            if (FIXED_EDIT in row['transforms']) != supplement:
                raise ValueError('Unexpected successful oxidation rows in the original list')
            if supplement and list(row['transforms']) != [FIXED_EDIT]:
                raise ValueError('The supplemental package ran more than one edit')
            prior = output.get(key)
            if prior is None:
                output[key] = row
                continue
            if prior['in_main_pool'] != row['in_main_pool']:
                raise ValueError('Conflicting main-library membership for one metric identity')
            winner = min(prior, row, key=ordering)
            output[key] = dict(winner,
                               n_routes=prior['n_routes'] + row['n_routes'],
                               n_parents=max(prior['n_parents'], row['n_parents']),
                               transforms=sorted(set(prior['transforms']) | {FIXED_EDIT}))
    return [output[key] for key in sorted(output)], len(output) - len(untouched)


def merge(original, supplement, out):
    base, base_detail = checked(original)
    fixed, fixed_detail = checked(supplement)
    if any(base[field] != fixed[field] for field in
           ('query_count', 'archived_c34', 'archived_pc_lists_sha256', 'target_receipt_sha256', 'hashes')):
        raise ValueError('Candidate packages did not use the same published pool and 60 mass centers')
    if (base['query_count'] != 60 or fixed.get('only_rule') != FIXED_EDIT or
            set(fixed_detail['transforms']) != {FIXED_EDIT} or
            set(fixed_detail['transform_delta_corrections']) != {FIXED_EDIT}):
        raise ValueError('Expected the verified original and the isolated corrected oxidation rule')
    if any(base_detail[key] != fixed_detail[key] for key in
           ('rdkit', 'pool_sha256', 'source_files', 'c34_ppm', 'author_ppm', 'query_sha256')):
        raise ValueError('Candidate packages did not use the same raw source snapshots')

    original_stats = pq.read_table(original / 'stats.parquet')
    supplement_stats = pq.read_table(supplement / 'stats.parquet')
    indexes = [{row['molecule_id']: row for row in table.to_pylist()}
               for table in (original_stats, supplement_stats)]
    if (len(indexes[0]) != 60 or len(indexes[1]) != 60 or len(original_stats) != len(supplement_stats)
            or set(indexes[0]) != set(indexes[1])):
        raise ValueError('The two packages must contain the same full 60-query panel')
    filenames = [f'{branch}.parquet' for branch in BRANCHES]
    schema = pq.read_schema(original / filenames[0])
    out.mkdir(parents=True, exist_ok=True)
    stats, totals = [], Counter()
    with (pq.ParquetWriter(out / filenames[0], schema, compression='zstd') as novel_writer,
          pq.ParquetWriter(out / filenames[1], schema, compression='zstd') as main_writer):
        for mid, source in indexes[0].items():
            repair = indexes[1][mid]
            if source['target_mass'] != repair['target_mass'] or source['status'] != repair['status']:
                raise ValueError(f'Different window for {mid}')
            pooled = {}
            for branch, name, writer in zip(BRANCHES, filenames, (novel_writer, main_writer), strict=True):
                groups = [pq.read_table(root / name, filters=[('molecule_id', '=', mid)]).to_pylist()
                          for root in (original, supplement)]
                rows, added = merge_list(*groups)
                totals[branch] += added
                pooled[branch] = len(rows), sum(row['in_author_window'] for row in rows)
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
            novel, narrow_novel = pooled['proposals']
            known, narrow_known = pooled['known_main']
            summed = {name: source[name] + repair[name] for name in
                      ('candidate_parent_transform_pairs', 'scanned_parent_transform_pairs', 'raw_products',
                       'invalid_products', 'out_of_window_products', 'valid_product_occurrences', 'seconds')}
            stats.append({**source, **summed, 'parent_rows': max(source['parent_rows'], repair['parent_rows']),
                          'invalid_parent_rows': max(source['invalid_parent_rows'], repair['invalid_parent_rows']),
                          'identities': novel + known, 'novel': novel, 'known_main': known,
                          'author_window_identities': narrow_novel + narrow_known,
                          'author_window_novel': narrow_novel, 'author_window_known_main': narrow_known})
    pq.write_table(pa.Table.from_pylist(stats, schema=original_stats.schema), out / 'stats.parquet')
    hashes = {filename: digest(out / filename) for filename in (*filenames, 'stats.parquet')}
    lineage = [base['generator_sha256'], fixed['generator_sha256'], digest(Path(__file__))]
    result = dict(base, generator_sha256=sha256(json.dumps(lineage).encode()).hexdigest(),
                  output_sha256=hashes, generator_code_lineage=lineage,
                  upstream_receipts=[digest(root / 'blind_virtual_np_receipt.json') for root in (original, supplement)],
                  transform_delta_corrections=fixed_detail['transform_delta_corrections'],
                  added_unique_identities=dict(totals),
                  counting_note='Combined distinct-parent counts are lower bounds for overlapping inputs.')
    (out / 'blind_virtual_np_receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print('CORRECTED_BLIND_CANDIDATES', dict(totals), flush=True)

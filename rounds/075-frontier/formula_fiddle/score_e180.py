"""Score sealed FIDDLE and C34 against five distinct E180 formulas per query."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import file_digest
import json
from pathlib import Path
import re
from runpy import run_path

from blind_pilot import FILES, RECEIPT, SLUG as PILOT_SLUG, SOURCES, STAGE, mounted


SLUG = 'enveda-075-fiddle-v240-e180-score'
OUTPUT = 'fiddle_e180_formula_score'
BASE = 'jiyujia/enveda-075-first-formula-baseline-e180/2'
RESERVE = 'jiyujia/enveda-075b-e180-d3-reserve-panel/1'
COMPACT = 'aryanawasthiteykik/casmi26-e180-not-in-train/2'
PINS = {'pc_lists.json': 'f9404c9276c1208e82c4dda93329f3fffd8096c8cedf98338feb76776765422f',
        'e004_candidate_inputs.json': '7fc3f74d2f88c9952e11b931ef6624e3fd51c374203d1828b691bed94e1a39c7',
        'e180_fresh_d3_panel.parquet': '0d493e7127bbc0424c8de121f18ce193b5fe7884d1d6484a5888c2a571409e9a',
        'e180_scan_pipeline_queries_ce.parquet': '6b30f38cc7172d8895599502e461546471a368c99b230c27d65147517dcb692d'}
TRAIN_SHA = '9423f90ec08b941a9646b535bf0606183e86ff0c5a297978f5a70693d41bcd98'
STAGE_SHA = '95aa42bbfa7537579ea46365848c6601390f5ea37b5f9b5ee4e68efa72aad32a'
AUTHOR_ELEMENTS = set('C O N H P S F Cl B Br I Na K'.split())
ATOM = re.compile(r'([A-Z][a-z]?)([1-9][0-9]*)?')


def checksum(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def pinned(path, digest):
    if checksum(path) != digest:
        raise ValueError(f'Archived input changed: {path.name}')
    return path


def validate_seal(seal):
    if (seal.get('status') != 'sealed' or seal['formula_budget_per_scan'] != 5
            or seal['original_precursor'] is not True or seal['explicit_eV_only'] is not True
            or seal['site_instrument'] != 'timstof' or set(seal['files']) != set(FILES)
            or (seal['stage'], seal['stage_receipt_sha256']) != (STAGE, STAGE_SHA)):
        raise ValueError('Unexpected blind FIDDLE experiment scope')
    if 'e180' not in seal['panels'] or not set(seal['panels']) <= SOURCES.keys():
        raise ValueError('Unexpected FIDDLE panel roster')
    for panel in seal['panels']:
        _, _, digest, molecules, scans = SOURCES[panel]
        item = seal['panels'][panel]
        if (item['input_sha256'], item['molecules'], item['scans']) != (digest, molecules, scans):
            raise ValueError(f'Frozen source panel differs at {panel}')
    for panel in set(seal['panels']) - {'e180'}:
        if seal['panels'][panel]['eligible_instrument_and_ce_queries'] != 0:
            raise ValueError('Direct FIDDLE transfer should not guess instrument or collision energy')
    if seal['panels']['e180']['eligible_instrument_and_ce_queries'] != 121:
        raise ValueError('Original E180 panel has measured timsTOF energy for every molecule')


def formula(text):
    atoms = {}
    end = 0
    for item in ATOM.finditer(text):
        element, count = item.groups()
        if item.start() != end or element in atoms:
            return None
        atoms[element] = int(count or '1')
        end = item.end()
    if not atoms or end != len(text):
        return None
    order = (['C', 'H'] if 'C' in atoms else []) + sorted(atoms.keys() - ({'C', 'H'} if 'C' in atoms else set()))
    display = ''.join(element + (str(atoms[element]) if atoms[element] > 1 else '') for element in order if element in atoms)
    return display, atoms


def ranked_unique(values):
    parsed = (formula(text) for text in values)
    return list(dict.fromkeys(entry[0] for entry in parsed if entry is not None))


def verify_blind(case_rows, predictions, archive_path, pc, e1, panels):
    import pyarrow.parquet as pq
    from blind_pilot import measured_scan

    raw = pq.read_table(archive_path).to_pylist()
    roster = {row['molecule_id'] for row in raw}
    accepted, groups, authorized = {}, defaultdict(list), set()
    e180 = [row for row in case_rows if row['panel'] == 'e180']
    if (roster != pc.keys() or roster != e1.keys() or len(e180) != len(raw) or len(roster) != 121
            or Counter(row['panel'] for row in case_rows) != {name: item['scans'] for name, item in panels.items()}):
        raise ValueError('E180 blind lineage, C34 and E1 populations differ')
    for ordinal, (entry, scan) in enumerate(zip(e180, raw, strict=True), 1):
        mid = scan['molecule_id']
        alias = f'formula_e180_{ordinal:04d}'
        eligible, _ = measured_scan(scan, alias)
        if eligible is None or (entry['id'], entry['molecule_id'], entry['spectrum_id']) != (alias, mid, scan['spectrum_id']):
            raise ValueError(f'Sealed scan identity differs at {alias}')
        checks = {'precursor_mz': 'precursor_mz', 'adduct': 'adduct', 'instrument': 'instrument_type',
                  'energy_ev': 'collision_energy_ev', 'energy_original': 'collision_energy_orig',
                  'energy_original_units': 'collision_energy_orig_units'}
        if any(entry[col] != scan[origin] for col, origin in checks.items()):
            raise ValueError(f'Sealed scan metadata differs at {alias}')
        groups[mid].append(alias)
        status = entry['status']
        validation = entry['author_validation_status']
        if status == 'author_rejected' and validation == 'rejected':
            continue
        if status not in ('predicted', 'no_formula_candidates') or validation not in ('ok', 'warning'):
            raise ValueError(f'Unexpected author validation result at {alias}')
        accepted[alias] = (entry, eligible)
        authorized.add(mid)
    if len(predictions) != len(accepted) or {row['id'] for row in predictions} != accepted.keys():
        raise ValueError('Sealed predicted scan population differs from author accepted scans')
    hypotheses = {}
    for row in predictions:
        entry, eligible = accepted[row['id']]
        candidates = row['candidates']
        if (row['instrument'] != 'timstof' or row['adduct'] != eligible['adduct']
                or not 0 <= len(candidates) <= 5 or bool(candidates) != (entry['status'] == 'predicted')
                or float(row['ce_encoded']) != entry['energy_ev'][0]):
            raise ValueError('Sealed candidate output has inconsistent measured conditions')
        if any(formula(item['formula']) is None for item in candidates):
            raise ValueError('Malformed neutral formula in author response')
        hypotheses[row['id']] = ranked_unique(item['formula'] for item in candidates)
    fiddle = {}
    for mid, aliases in groups.items():
        votes = Counter()
        for alias in aliases:
            for rank, key in enumerate(hypotheses.get(alias, ()), 1):
                votes[key] += 60 // rank
        fiddle[mid] = sorted(votes, key=lambda key: (-votes[key], key))[:5]
    return fiddle, authorized, groups


def truth_and_exposure(inputs, blind_path, calc_formula, metric_key):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from e180_labels import blind_labels

    raw, panel = (pinned(mounted(inputs, RESERVE, name), PINS[name])
                  for name in ('e180_scan_pipeline_queries_ce.parquet', 'e180_fresh_d3_panel.parquet'))
    blind_to_key = blind_labels(blind_path, raw).to_dict()
    selected = {row['query']: row for row in pq.read_table(panel).to_pylist()}
    if len(selected) != 121 or set(blind_to_key.values()) != selected.keys():
        raise ValueError('Archive truth lineage should cover 121 measured identities exactly')
    keys = sorted(selected)
    ordinals = [selected[key]['selected_compact_row'] for key in keys]
    compact = mounted(inputs, COMPACT, 'e180_not_in_train.parquet')
    source_rows = pq.read_table(compact, columns=['smiles', 'ik14']).take(pa.array(ordinals)).to_pylist()
    truth = {}
    for key, row in zip(keys, source_rows, strict=True):
        identity = metric_key(row['smiles'])
        if identity is None or key != identity[0] or row['ik14'] != selected[key]['raw_key']:
            raise ValueError('Selected E180 compact row does not reproduce the archived truth identity')
        text = calc_formula(row['smiles'])
        parsed = formula(text)
        truth[key] = dict(formula=None if parsed is None else parsed[0], identity=identity,
                          stratum=selected[key]['murcko_stratum'], atoms={} if parsed is None else parsed[1])

    train = pq.ParquetFile(pinned(mounted(inputs, SOURCES['e180'][0], 'train.parquet'), TRAIN_SHA))
    keys_and_aliases = {alias: key for key, record in selected.items() for alias in (key, record['raw_key'])}
    targets = {value['identity']: key for key, value in truth.items()}
    forms = {value['formula'] for value in truth.values() if value['formula'] is not None}
    exposure = defaultdict(set)
    data = ['ingest_lib', 'instrument_type', 'inchikey14', 'molecular_formula', 'normalized_smiles']
    for batch in train.iter_batches(batch_size=65536, columns=data):
        for source, instrument, raw_key, text, smiles in zip(*(column.to_pylist() for column in batch.columns), strict=True):
            scope = 'author_timsTOF_training_libraries' if (instrument == 'timsTOF' and source in ('enveda-180', 'enveda-np-examples')) else 'other_competition_training'
            if raw_key in keys_and_aliases:
                exposure[keys_and_aliases[raw_key]].add(scope)
            if text in forms:
                identity = metric_key(smiles)
                if identity in targets:
                    exposure[targets[identity]].add(scope)
    return {mid: truth[key] | {'training_overlap': sorted(exposure[key])} for mid, key in blind_to_key.items()}, train.metadata.num_rows


def score(inputs, out, reference, digest):
    from functools import lru_cache

    from rdkit import Chem, RDLogger, rdBase
    from rdkit.Chem import rdMolDescriptors
    from c39_request_cache import identity_function

    RDLogger.DisableLog('rdApp.*')
    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError('The official metric RDKit build is required')
    receipt = pinned(mounted(inputs, reference, RECEIPT), digest)
    seal = json.loads(receipt.read_bytes())
    validate_seal(seal)
    paths = {name: pinned(receipt.with_name(name), sha) for name, sha in seal['files'].items()}
    scan_rows, predictions = ([json.loads(line) for line in paths[name].read_text().splitlines()] for name in FILES[1:])
    blind = pinned(mounted(inputs, SOURCES['e180'][0], SOURCES['e180'][1]), SOURCES['e180'][2])
    pc, e1_export = (json.loads(pinned(mounted(inputs, BASE, name), PINS[name]).read_bytes())
                     for name in ('pc_lists.json', 'e004_candidate_inputs.json'))
    e1 = e1_export['base_tail']
    if any(len(value['pc']) != 25 for value in pc.values()):
        raise ValueError('Expected 25 raw C34 candidates per E180 query')
    fiddle, author_valid, group_scans = verify_blind(scan_rows, predictions, blind, pc, e1, seal['panels'])

    @lru_cache(maxsize=32768)
    def calc_formula(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError('Archived candidate or truth SMILES failed RDKit')
        return rdMolDescriptors.CalcMolFormula(mol)

    c34_all = {mid: ranked_unique(map(calc_formula, row['pc'])) for mid, row in pc.items()}
    prior = {mid: set(c34_all[mid]) | set(ranked_unique(map(calc_formula, e1[mid]))) for mid in pc}
    labels, n_train = truth_and_exposure(inputs, blind, calc_formula, identity_function())
    rows = []
    for mid, truth in sorted(labels.items()):
        key = truth['formula']
        c, f = c34_all[mid][:5], fiddle[mid]
        row = dict(molecule_id=mid, author_valid_scan=mid in author_valid, measured_scans=len(group_scans[mid]),
                   author_formula_scope=bool(key and set(truth['atoms']) <= AUTHOR_ELEMENTS and 10 <= sum(truth['atoms'].values()) <= 300),
                   murcko_stratum=truth['stratum'], any_training_overlap=bool(truth['training_overlap']),
                   author_library_overlap='author_timsTOF_training_libraries' in truth['training_overlap'],
                   fiddle_formulas=';'.join(f), c34_formulas=';'.join(c),
                   c34_truth_rank=c.index(key) + 1 if key in c else 0,
                   fiddle_truth_rank=f.index(key) + 1 if key in f else 0,
                   truth_in_e1_or_raw_c34=key in prior[mid], fiddle_lead_has_e1_or_c34_support=bool(f and f[0] in prior[mid]))
        rows.append(row)

    def metrics(group):
        counts = {arm: {f'top{k}': sum(0 < row[f'{arm}_truth_rank'] <= k for row in group) for k in (1, 5)}
                  for arm in ('c34', 'fiddle')}
        return dict(molecules=len(group), **counts,
                    original_full_e1_c34_formula_covered=sum(row['truth_in_e1_or_raw_c34'] for row in group),
                    fiddle_gained_top1=[r['molecule_id'] for r in group if r['fiddle_truth_rank'] == 1 and r['c34_truth_rank'] != 1],
                    fiddle_lost_top1=[r['molecule_id'] for r in group if r['c34_truth_rank'] == 1 and r['fiddle_truth_rank'] != 1],
                    fiddle_gained_top5=[r['molecule_id'] for r in group if r['fiddle_truth_rank'] and not r['c34_truth_rank']],
                    fiddle_lost_top5=[r['molecule_id'] for r in group if r['c34_truth_rank'] and not r['fiddle_truth_rank']],
                    fiddle_added_beyond_e1_and_c34=[r['molecule_id'] for r in group if r['fiddle_truth_rank'] and not r['truth_in_e1_or_raw_c34']],
                    fiddle_lead_supported_by_e1_or_c34=sum(row['fiddle_lead_has_e1_or_c34_support'] for row in group))

    cohorts = dict(all=metrics(rows), author_valid=metrics([r for r in rows if r['author_valid_scan']]),
                   author_formula_scope=metrics([r for r in rows if r['author_formula_scope']]),
                   author_valid_and_formula_scope=metrics([r for r in rows if r['author_valid_scan'] and r['author_formula_scope']]),
                   no_detected_competition_training_overlap=metrics([r for r in rows if not r['any_training_overlap']]))
    out.mkdir(exist_ok=True, parents=True)
    per_query = out / (OUTPUT + '_per_query.csv')
    with per_query.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(status='scored', pilot=reference, pilot_sha256=digest, sources=PINS,
                  scope='E180 timsTOF instrument transfer on the frozen external panel; candidate formula accuracy',
                  formula_budget=5, all_panel_input_counts=seal['panels'], cohorts=cohorts,
                  training_overlap_audit=dict(train_sha256=TRAIN_SHA, train_rows=n_train,
                                           any_competition_training_overlap=sum(r['any_training_overlap'] for r in rows),
                                           author_timsTOF_library_overlap=sum(r['author_library_overlap'] for r in rows),
                                           exact_identity_method='Stored raw aliases plus official metric normalization on matching neutral formula rows'),
                  author_formula_scope='Author timsTOF training template elements and 10–300 total atoms; measured input validity separately checked by native validator',
                  fiddle_query_aggregation='Sum reciprocals of distinct scan formula ranks, lexical formula ties, keep five',
                  c34_query_aggregation='First five distinct molecular formulae in frozen raw 25',
                  per_query_sha256=checksum(per_query))
    (out / (OUTPUT + '.json')).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(cohorts, allow_nan=False), flush=True)


def build(seal, out, version=1):
    here = Path(__file__).resolve().parent
    common = run_path(str(here.parent / 'virtual_np/build.py'))
    validate_seal(json.loads(seal.read_bytes()))
    reference = f'jiyujia/{PILOT_SLUG}/{version}'
    files = {'score_e180.py': Path(__file__), 'blind_pilot.py': here / 'blind_pilot.py',
             'e180_labels.py': here.parent / 'pipeline_candidates/first_formula/score_blind_pair.py',
             'c39_request_cache.py': here.parent / 'virtual_np/c39_request_cache.py'}
    sources = {name: path.read_bytes() for name, path in files.items()}
    for name, raw in sources.items():
        compile(raw, name, 'exec')
    payload = {name: common['pack'](raw) for name, raw in sources.items()}
    code = f'''
import base64, subprocess, sys, tempfile, zlib
from hashlib import sha256
from pathlib import Path
inputs = Path('/kaggle/input')
{common['METRIC_BOOTSTRAP']}
with tempfile.TemporaryDirectory(prefix='fiddle-e180-score-') as temp:
    for name, (blob, expected) in {payload!r}.items():
        data = zlib.decompress(base64.b64decode(blob))
        if sha256(data).hexdigest() != expected:
            raise ValueError(f'Embedded scoring source changed: {{name}}')
        Path(temp, name).write_bytes(data)
    sys.path.insert(0, temp)
    from score_e180 import score
    score(inputs, Path('/kaggle/working'), {reference!r}, {checksum(seal)!r})
'''
    compile(code, SLUG, 'exec')
    metadata = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='score.py', language='python', kernel_type='script',
                    is_private=True, enable_gpu=False, enable_internet=False, docker_image=common['IMAGE'],
                    dataset_sources=[common['WHEEL'], COMPACT], competition_sources=[],
                    kernel_sources=[reference, SOURCES['e180'][0], BASE, RESERVE])
    out.mkdir(exist_ok=True, parents=True)
    (out / 'score.py').write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return checksum(out / 'score.py')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--version', type=int, default=1)
    print(build(**vars(parser.parse_args())))

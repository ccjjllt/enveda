"""Audit public WineDB spectral records against official competition structures."""

from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from hashlib import file_digest, sha256
import json
from math import isfinite
from pathlib import Path
import re
from urllib.request import Request, urlopen


GNPS_MIRROR = 'https://external.gnps2.org/gnpslibrary/'
BUNDLES = {'WINE-DB-ORBITRAP': ('orbitrap', 1247711), 'WINE-DB-QTOF': ('qtof', 4573590)}
TRAIN_SHA = '9423f90ec08b941a9646b535bf0606183e86ff0c5a297978f5a70693d41bcd98'
TRAIN_SIZE = 2539608
PROTON = 1.00727645217
ADDUCTS = {'M+H': '[M+H]+', '[M+H]+': '[M+H]+', 'M-H': '[M-H]-', '[M-H]-': '[M-H]-'}
ELEMENT = re.compile(r'([A-Z][a-z]?)([0-9]*)')
SCAN_FIELDS = ('SOURCE_SCAN_USI', 'USI', 'SOURCE_SCAN_ID', 'RAW_FILE', 'retention_time')
ENERGY_FIELDS = ('Collision_Energy', 'CollisionEnergy', 'COLLISION_ENERGY', 'collision_energy', 'collision_energy_ev')
DATA_GUIDE = 'https://ccms-ucsd.github.io/GNPSDocumentation/gnpslibraries/#library-list'
LICENSE_RECORD = 'https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?task=71642757606c40738edd1c8354779989'


def download(name, expected_size):
    url = GNPS_MIRROR + name + '.json'
    req = Request(url, headers={'Accept-Encoding': 'identity', 'User-Agent': 'CASMI-research/1.0'})
    with urlopen(req, timeout=45) as response:
        raw = response.read(expected_size + 1)
        source = dict(url=url, dataset=name, bytes=len(raw), sha256=sha256(raw).hexdigest(),
                      last_modified=response.headers.get('Last-Modified'), etag=response.headers.get('ETag'))
    if len(raw) != expected_size:
        raise ValueError(f'{name}: GNPS mirror content changed size; expected {expected_size}, received {len(raw)}')
    rows = json.loads(raw)
    if not rows or any(record.get('library_membership') != name for record in rows):
        raise ValueError(f'{name}: response is empty or has a different library membership')
    if len({row['spectrum_id'] for row in rows}) != len(rows):
        raise ValueError(f'{name}: duplicated original spectrum accession')
    return source, rows


@lru_cache(maxsize=1024)
def composition(formula):
    if not isinstance(formula, str) or not formula:
        return None
    plain = re.sub(r'[+-]\d*$', '', formula)
    matches = ELEMENT.findall(plain)
    if not matches or ''.join(symbol + count for symbol, count in matches) != plain:
        return None
    counts = Counter()
    for symbol, count in matches:
        if symbol != 'H':
            counts[symbol] += int(count or '1')
    return tuple(sorted(counts.items()))


def identity(smiles, scaffold=False):
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None or not mol.GetNumHeavyAtoms():
        return None
    raw = Chem.MolToInchiKey(mol)[:14]
    if scaffold and (len(Chem.GetMolFrags(mol)) != 1 or mol.GetNumHeavyAtoms() < 5 or Chem.GetFormalCharge(mol)):
        return None
    heavy = tuple(sorted(Counter(atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1).items()))
    mass = rdMolDescriptors.CalcExactMolWt(mol)
    Chem.RemoveStereochemistry(mol)
    measured = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    key = Chem.MolToInchiKey(measured)[:14]
    if len(key) != 14 or len(raw) != 14:
        return None
    if not scaffold:
        return key, heavy
    core = MurckoScaffold.GetScaffoldForMol(measured)
    generic = MurckoScaffold.MakeScaffoldGeneric(core)
    return dict(key=key, raw_key=raw, heavy=heavy, mass=mass,
                fine=Chem.MolToSmiles(core), generic=Chem.MolToSmiles(generic))


def energy_scope(row):
    fields = {field: str(row[field]).strip() for field in ENERGY_FIELDS if row.get(field) not in (None, '')}
    sample_file = str(row.get('source_file') or '')
    if re.search(r'merged', sample_file, re.I):
        return 'merge_declared_in_author_filename', fields
    if len(fields) == 1:
        field, value = next(iter(fields.items()))
        if re.fullmatch(r'\d+(?:\.\d+)?\s*(?:eV|NCE|%)', value, flags=re.I):
            return 'single_reported_with_unit', fields
        if field.endswith('_ev') and re.fullmatch(r'\d+(?:\.\d+)?', value):
            return 'single_reported_with_unit', fields
    return ('reported_value_or_scale_unresolved' if fields else 'not_exposed_in_gnps_record'), fields


def parsed_record(record, instrument):
    status = str(record.get('spectrum_status'))
    if str(record.get('ms_level')) != '2' or status != '1':
        return 'not_active_ms2', None
    if instrument not in str(record.get('Instrument', '')).replace('-', '').lower():
        return 'unexpected_instrument', None
    provenance = str(record.get('Compound_Source') or '').strip().lower()
    if provenance not in {'commercial', 'isolated', 'synthetic'}:
        return 'explicit_standard_provenance_missing', None
    smiles = str(record.get('Smiles') or '').strip()
    info = identity(smiles, scaffold=True)
    if info is None:
        return 'missing_permitted_structure', None
    declared = str(record.get('InChIKey_smiles') or record.get('InChIKey_inchi') or '')[:14]
    if declared not in {info['key'], info['raw_key']}:
        return 'unexpected_author_identity_key', None
    formula = record.get('Formula_smiles') or record.get('Formula_inchi')
    if formula and composition(formula) != info['heavy']:
        return 'author_formula_conflict', None
    adduct = ADDUCTS.get(str(record.get('Adduct') or '').replace(' ', ''))
    mode = str(record.get('Ion_Mode') or '').strip().lower()
    if not adduct or mode != ('positive' if adduct.endswith('+') else 'negative'):
        return 'not_an_explicit_protonated_or_deprotonated_adduct', None
    text_mz = str(record.get('Precursor_MZ') or '')
    try:
        decimal = Decimal(text_mz)
        precursor = float(decimal)
        pairs = json.loads(record.get('peaks_json') or '[]')
        fragments = {float(mz) for mz, intensity in pairs if isfinite(mz) and isfinite(intensity)
                     and 0 < mz < precursor - 0.5 and intensity > 0}
        if precursor <= 0 or not isfinite(precursor) or len(fragments) < 4:
            raise ValueError
    except (ValueError, TypeError, InvalidOperation):
        return 'invalid_precursor_or_fewer_than_four_fragments', None
    expected_mz = info['mass'] + (PROTON if mode == 'positive' else -PROTON)
    ppm = (precursor - expected_mz) / expected_mz * 1e6
    ce, ce_raw = energy_scope(record)
    # A GNPS export scan refers to the deposited library input, which can itself be a merged MGF.
    # Count only explicit raw provenance; a difference from theoretical mass cannot prove its origin.
    explicit_raw = {field: str(record[field]) for field in SCAN_FIELDS if record.get(field) not in (None, '', 'N/A')}
    return None, dict(accession=record['spectrum_id'], url=record.get('url'), name=record.get('Compound_Name'),
                      instrument=instrument, standard_source=provenance, smiles=smiles, **info, adduct=adduct,
                      polarity=mode, author_formula=formula, precursor_mz=precursor, expected_mz=expected_mz,
                      precursor_error_ppm=ppm, reported_decimal_places=max(0, -decimal.as_tuple().exponent),
                      fragment_count=len(fragments), ce_scope=ce, ce_raw=ce_raw, source_file=record.get('source_file'),
                      source_task=record.get('task'), raw_provenance=explicit_raw)


def training_overlap(train, public):
    import pyarrow.parquet as pq

    with train.open('rb') as handle:
        fingerprint = file_digest(handle, 'sha256').hexdigest()
    pf = pq.ParquetFile(train)
    if fingerprint != TRAIN_SHA or pf.metadata.num_rows != TRAIN_SIZE:
        raise ValueError('Different competition training input')
    targets = {(row['key'], row['heavy']) for row in public}
    hard_keys = {row[key] for row in public for key in ('key', 'raw_key')}
    compositions = {row['heavy'] for row in public}
    records = Counter()
    raw_hits = defaultdict(set)
    structures = defaultdict(set)
    screening_rows = 0
    no_formula_rows = 0
    for batch in pf.iter_batches(batch_size=65536, columns=['inchikey14', 'ingest_lib', 'molecular_formula', 'normalized_smiles']):
        values = [batch.column(i).to_pylist() for i in range(4)]
        for cell, library, formula, smi in zip(*values):
            records[library or ''] += 1
            if cell in hard_keys:
                raw_hits[cell].add(library or '')
            elements = composition(formula)
            if elements is None:
                no_formula_rows += 1
            if smi and (elements in compositions or elements is None or cell in hard_keys):
                screening_rows += 1
                structures[smi].add(library or '')
    exact_hits = defaultdict(set)
    for smiles, libraries in structures.items():
        candidate = identity(smiles)
        if candidate in targets:
            exact_hits[candidate].update(libraries)
    receipt = dict(sha256=fingerprint, rows=pf.metadata.num_rows, training_source_counts=dict(records),
                   same_composition_or_unclassified_rows=screening_rows, unclassified_formula_rows=no_formula_rows,
                   normalized_competition_structures=len(structures), direct_keys_hit=len(raw_hits),
                   exact_metric_identity_hits=len(exact_hits))
    return raw_hits, exact_hits, receipt


def summarize(rows):
    return dict(spectra=len(rows), identities=len({r['key'] for r in rows}),
                fine_scaffolds=len({r['fine'] for r in rows} - {''}),
                generic_scaffolds=len({r['generic'] for r in rows} - {''}),
                positive_identities=len({r['key'] for r in rows if r['polarity'] == 'positive'}),
                negative_identities=len({r['key'] for r in rows if r['polarity'] == 'negative'}),
                single_ce_with_unit=sum(r['ce_scope'] == 'single_reported_with_unit' for r in rows),
                explicit_raw_provenance=sum(bool(r['raw_provenance']) for r in rows),
                energy_scope=dict(Counter(r['ce_scope'] for r in rows)),
                reported_mz_decimal_places=dict(Counter(r['reported_decimal_places'] for r in rows)))


def run(train, output):
    from rdkit import RDLogger, rdBase

    if rdBase.rdkitVersion != '2026.03.3':
        raise RuntimeError('Run with the competition RDKit wheel')
    RDLogger.DisableLog('rdApp.*')
    output.mkdir(parents=True, exist_ok=True)
    all_public = []
    sources = {}
    for name, (instrument, size) in BUNDLES.items():
        archive, records = download(name, size)
        summaries = defaultdict(Counter)
        failures = Counter()
        good = []
        for record in records:
            for field in ('Compound_Source', 'Adduct', 'Ion_Mode', 'Instrument'):
                summaries[field][str(record.get(field))] += 1
            reason, parsed = parsed_record(record, instrument)
            if parsed is None:
                failures[reason] += 1
                continue
            good.append(parsed)
        sources[name] = dict(archive=archive, all_public_spectra=len(records), source_fields=summaries,
                             rejections=dict(failures), parsed=summarize(good))
        all_public.extend(good)
    if not all_public:
        raise ValueError('No public WineDB MS2 standards passed structural and provenance parsing')
    raw, exact, exposure = training_overlap(train, all_public)
    for row in all_public:
        known = set(exact.get((row['key'], row['heavy']), ()))
        known.update(raw.get(row['key'], ()))
        known.update(raw.get(row['raw_key'], ()))
        row['training_sources_matching_raw_or_metric_identity'] = sorted(known)
    novel = [r for r in all_public if not r['training_sources_matching_raw_or_metric_identity']]
    in_window = [r for r in novel if abs(r['precursor_error_ppm']) <= 10]
    candidate_path = output / 'wine_db_public_rows.json'
    candidate_path.write_text(json.dumps(all_public, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    receipt = dict(rdkit=rdBase.rdkitVersion, public_library_documentation=DATA_GUIDE,
                   original_license_record=LICENSE_RECORD, original_license='CC0 1.0 Universal',
                   provider_scope='One WineDB study spanning two acquisition instruments.',
                   data_sources=sources, training_overlap=exposure, all_parsed=summarize(all_public),
                   training_identity_unseen=summarize(novel), training_identity_unseen_within_10ppm=summarize(in_window),
                   per_instrument_training_unseen_within_10ppm={instrument: summarize([r for r in in_window if r['instrument'] == instrument])
                                                                 for instrument, _ in BUNDLES.values()},
                   rows_file=candidate_path.name, rows_sha256=sha256(candidate_path.read_bytes()).hexdigest(),
                   scope='This first audit checks WineDB public library peak records, compound identities and official training exposure. An author-exported precursor is retained exactly; raw centroid and single-CE provenance are counted only if explicitly available. No prior blind labels or private exclusion files are used.')
    (output / 'wine_db_availability.json').write_text(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + '\n')
    print(json.dumps({k: receipt[k] for k in ('all_parsed', 'training_identity_unseen', 'training_identity_unseen_within_10ppm')}, sort_keys=True), flush=True)

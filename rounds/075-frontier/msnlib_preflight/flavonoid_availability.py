"""Aggregate eligibility of the Rypar et al. flavonoid standards library."""

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from audit import download, scan_sources, values
from panel_availability import PROTON, bins, digest_text, identify, number, train_exposure


RELEASE = 16762591
ASSET = ('in-house_MSMS_library.mgf', 18770115, 'cfd02785d4c7400fdd06f59fb4580112')
LICENSE = 'CC BY 4.0'
PAPER = 'https://doi.org/10.3390/metabo15090616'
DATASET = 'https://doi.org/10.5281/zenodo.16762591'
RAWDATA = 'https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?task=f4420d487b344b9eabe6fa3f65695fee'
RMASSBANK = 'https://rdrr.io/bioc/RMassBank/src/R/buildRecord.R'
ADDUCTS = {'[M+H]+': ('positive', PROTON), '[M-H]-': ('negative', -PROTON)}
ADDUCT_IN_TEXT = re.compile(r'\[M[+-]H\](?:1)?[+-]', re.I)
ORIGINAL_FILE = re.compile(r'\.(?:raw|mzml|mzxml)$', re.I)
KEY14 = re.compile(r'^[A-Z]{14}(?:-|$)')
MGF_FIELD = re.compile(r'[A-Za-z][A-Za-z0-9$_./ -]*')
COLUMN_FIELD = re.compile(r'^([A-Za-z][A-Za-z0-9$_./ -]*?)\s*(?:[:;\t]|,\s*)\s*(.+)$')
NUMERIC = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
EXPLICIT_PEAK_COLUMNS = re.compile(rf'^\s*({NUMERIC})\s*([,;:])\s*({NUMERIC})(?:\s+.*)?$')
SCIENTIFIC_SIGNS = str.maketrans({'−': '-', '–': '-', '＋': '+', '－': '-'})
CHARGE_NUMBER = re.compile(r'^([+-]?)(\d+(?:\.0+)?)([+-]?)$')
KNOWN_FIELDS = {
    'TITLE', 'NAME', 'SMILES', 'CANONICAL_SMILES', 'INCHI', 'INCHIKEY', 'INCHI_KEY', 'INCHIAUX',
    'PEPMASS', 'PRECURSOR_MZ', 'PRECURSOR_TYPE', 'ION_TYPE', 'ADDUCT', 'CHARGE', 'IONMODE', 'ION_MODE',
    'POLARITY', 'MSLEVEL', 'MS_LEVEL', 'MS_TYPE', 'FRAGMENTATION_METHOD', 'FRAGMENTATION_MODE',
    'ACTIVATION', 'ACTIVATION_METHOD', 'SPECTYPE', 'SPECTRUM_TYPE', 'SCAN_TYPE', 'SOURCE_SCAN_USI',
    'SCANS', 'SCAN_NUMBER', 'FILENAME', 'RAW_FILE', 'SOURCE_FILE', 'RAW_FILENAME', 'ACQUISITIONNUM',
    'FORMULA', 'RTINSECONDS', 'RTINMINUTES', 'RETENTION_TIME', 'COLLISION_ENERGY', 'INSTRUMENT',
    'INSTRUMENT_TYPE', 'RESOLUTION', 'AUTHOR', 'AUTHORS', 'ID', 'CAS', 'IONIZATION',
    'MERGED_ACROSS_N_SAMPLES', 'MERGED_SCAN_COUNT', 'NUMBER_OF_MERGED_SCANS', 'N_MERGED_SCANS',
}
OBSERVED_FIELDS = ('MEASURED_PRECURSOR_MZ', 'OBSERVED_PRECURSOR_MZ', 'EXPERIMENTAL_PRECURSOR_MZ',
                   'OBSERVED_MS1_MZ', 'MS1_FEATURE_MZ', 'MEASURED_MS1_MZ')
ORIGIN_FIELDS = ('PEPMASS_ORIGIN', 'PRECURSOR_MZ_ORIGIN', 'PEPMASS_TYPE', 'PRECURSOR_MZ_TYPE')
MEASURED = {'measured', 'observed', 'experimental', 'measured ms1', 'observed ms1', 'ms1 centroid'}
CALCULATED = {'calculated', 'theoretical', 'theoretical monoisotopic', 'calculated exact'}
KNOWN_FIELDS.update((*OBSERVED_FIELDS, *ORIGIN_FIELDS))


@dataclass(frozen=True)
class Evidence:
    polarity: str
    activation: str
    scans: str
    precursor: str
    observed_field: str
    ppm_exported: float
    ppm_observed: float | None
    fragment_peaks: int
    theoretical_round_match: bool


def first(header, *fields):
    return next((header[name] for name in fields if header.get(name)), '')


def decode_mgf(path, expected_sha):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected_sha:
        raise ValueError('The archived author MGF changed between download and parsing')
    bom = raw.startswith(b'\xef\xbb\xbf')
    try:
        return raw.decode('utf-8-sig'), dict(decoder='utf-8-sig', leading_utf8_bom=bom)
    except UnicodeDecodeError as error:
        details = dict(leading_utf8_bom=bom, first_utf8_error_offset=error.start + 3 * bom,
                       first_utf8_error_hex=error.object[error.start:error.end].hex())
    legacy = raw[3:] if bom else raw
    try:
        return legacy.decode('cp1252'), dict(details, decoder='cp1252')
    except UnicodeDecodeError as error:
        return legacy.decode('latin-1'), dict(details, decoder='latin-1',
                                             first_cp1252_error_offset=error.start + 3 * bom)


def field_name(text):
    return re.sub(r'[\s-]+', '_', text.strip().upper())


def line_shape(text):
    punctuation = '=,.+-:;\t|"\'\\[](){}<>'
    return ''.join(char if char in punctuation else ('A' if char.isalpha() else '9' if char.isdecimal()
                   else ' ' if char.isspace() else '?') for char in text[:20])


def peak_columns(line):
    fields = line.split(maxsplit=2)
    if len(fields) >= 2 and (mass := number(fields[0])) is not None and (intensity := number(fields[1])) is not None:
        return mass, intensity, 'whitespace'
    match = EXPLICIT_PEAK_COLUMNS.fullmatch(line)
    if match:
        return number(match[1]), number(match[3]), {',': 'comma', ';': 'semicolon', ':': 'colon'}[match[2]]
    return None, None, None


def author_records(text, diagnostics, shapes, extra_fields, peak_columns_seen):
    header, peaks = None, []
    for line in StringIO(text, newline=None):
        line = line.strip()
        if line.startswith(('\ufeff', 'ï»¿')):
            line = line.removeprefix('\ufeff').removeprefix('ï»¿').lstrip()
            diagnostics['line_bom_prefixes'] += 1
        if line.startswith('"') and line.endswith('"'):
            try:
                pieces = next(csv.reader([line], strict=True))
            except csv.Error:
                pieces = []
            if len(pieces) == 1:
                line = pieces[0].strip()
                diagnostics['framed_csv_lines'] += 1
        marker = line.upper()
        if marker == 'BEGIN IONS':
            if header is not None:
                diagnostics['unterminated_blocks'] += 1
            header, peaks = {}, []
            diagnostics['begin_ions'] += 1
        elif marker == 'END IONS':
            if header is None:
                diagnostics['unmatched_end_ions'] += 1
            else:
                yield header, peaks
                header = None
        elif header is not None and line and not line.startswith(('#', ';')):
            key, separator, value = line.partition('=')
            name = field_name(key) if separator and MGF_FIELD.fullmatch(key.strip()) else ''
            if not name and (token := COLUMN_FIELD.fullmatch(line)):
                candidate = field_name(token[1])
                if candidate in KNOWN_FIELDS:
                    name, value = candidate, token[2]
                    diagnostics['non_equals_metadata_fields'] += 1
                elif ':' in line and len(candidate) <= 80:
                    # The descriptor has no value; aggregate format only.
                    extra_fields[line_shape(candidate)] += 1
                    diagnostics['unrecognized_colon_metadata_lines'] += 1
                    continue
            if name:
                if name in header and header[name] != value.strip():
                    diagnostics['conflicting_repeated_fields'] += 1
                header[name] = value.strip()
                continue
            mass, intensity, separator = peak_columns(line)
            if mass is None or intensity is None:
                diagnostics['unrecognized_in_block_lines'] += 1
                shapes[line_shape(line)] += 1
            elif mass <= 0 or intensity <= 0:
                diagnostics['nonpositive_peaks'] += 1
            else:
                peaks.append(mass)
                peak_columns_seen[separator] += 1
                diagnostics['parsed_positive_peak_lines'] += 1
    if header is not None:
        diagnostics['unterminated_blocks'] += 1


def scan_class(header):
    mode = first(header, 'SPECTYPE', 'SPECTRUM_TYPE', 'SCAN_TYPE').upper().replace(' ', '_')
    if mode in {'MERGED', 'CONSENSUS', 'MERGED_SCAN', 'SAME_ENERGY', 'AVERAGED'}:
        return 'explicitly_merged'
    counts = [number(header.get(name)) for name in ('MERGED_ACROSS_N_SAMPLES', 'MERGED_SCAN_COUNT',
                                                     'NUMBER_OF_MERGED_SCANS', 'N_MERGED_SCANS')]
    scans = scan_sources(header)
    source_usis = {item for item in values(header.get('SOURCE_SCAN_USI')) if isinstance(item, str)}
    explicit = first(header, 'SCANS', 'SCAN_NUMBER')
    if (len(source_usis) > 1 or len(scans) > 1 or any(not scan.isdigit() for scan in scans.values())
            or any(value is not None and value > 1 for value in counts)
            or re.fullmatch(r'\d+(?:[-,]\d+)+', explicit)):
        return 'explicitly_merged'
    raw_file = first(header, 'FILENAME', 'RAW_FILE', 'SOURCE_FILE', 'RAW_FILENAME')
    source_scan = (len(scans) == 1 and next(iter(scans.values())).isdigit()) or (
        bool(ORIGINAL_FILE.search(raw_file)) and explicit.isdigit())
    if mode in {'SINGLE_SCAN', 'SINGLE_BEST_SCAN', 'NATIVE_SINGLE_SCAN'}:
        return 'source_traced_single' if source_scan else 'declared_single_without_source'
    return 'processed_library_record_scan_count_unverified'


def precursor_evidence(header, token, exact):
    exported = number(token)
    provenance = first(header, *ORIGIN_FIELDS).casefold().replace('_', ' ').strip()
    fields = [(name, m) for name in OBSERVED_FIELDS
              if (m := number(header.get(name))) is not None and m > 0]
    observed_field, observed = fields[0] if fields else ('', None)
    if any(abs(m - observed) > max(0.0005, observed * 1e-6) for _, m in fields):
        observed_field, observed = 'conflicting_observed_fields', None
    try:
        printed = Decimal(token)
        round_match = abs(Decimal(str(exact)) - printed) <= (Decimal(10) ** printed.as_tuple().exponent) / 2
    except (InvalidOperation, ValueError, OverflowError):
        round_match = False
    if provenance in CALCULATED:
        origin = 'explicitly_calculated'
    elif provenance in MEASURED:
        if observed_field == 'conflicting_observed_fields' or (
                observed is not None and abs(exported - observed) > max(0.0005, observed * 1e-6)):
            origin, observed_field, observed = 'conflicting_origin', 'conflicting_observed_fields', None
        else:
            origin = 'explicitly_measured'
            if not fields:
                observed_field, observed = 'PEPMASS_with_explicit_origin', exported
    elif observed is not None and abs(exported - observed) <= max(0.0005, observed * 1e-6):
        origin = 'matches_explicit_observed_field'
    else:
        origin = 'theoretical_rounding_compatible_unverified' if round_match else 'origin_unverified'
    return (origin, observed_field, (exported - exact) / exact * 1e6,
            None if observed is None else (observed - exact) / exact * 1e6, round_match)


def charge_value(text):
    compact = re.sub(r'\s+', '', text.translate(SCIENTIFIC_SIGNS)).casefold()
    compact = re.sub(r'^(?:charge|z)[:=]?', '', compact).removesuffix(';')
    for left, right in (('[', ']'), ('(', ')')):
        if compact.startswith(left) and compact.endswith(right):
            compact = compact[1:-1]
            break
    if (named := re.fullmatch(r'(\d+(?:\.0+)?)(positive|negative|pos|neg)', compact)):
        return (float(named[1]) == 1, '-' if named[2] in {'negative', 'neg'} else '+')
    if (match := CHARGE_NUMBER.fullmatch(compact)) and not (match[1] and match[3]):
        return (float(match[2]) == 1, match[1] or match[3])
    return None


def charge_descriptor(value):
    if not value:
        return 'missing'
    safe = value.translate(SCIENTIFIC_SIGNS)
    if (len(safe) <= 40 and re.fullmatch(r'(?:[0-9.+\-;\[\]() \t]+|(?:charge|z)[:=0-9.+\-;\[\]() \t]+|[0-9. \t]+(?:pos|neg|positive|negative))',
                                          safe, flags=re.I)):
        return safe
    return 'shape=' + line_shape(value)


def header_identity(header, chemistry):
    smiles = first(header, 'SMILES', 'CANONICAL_SMILES')
    if smiles not in chemistry:
        chemistry[smiles] = identify(smiles)
    info = chemistry[smiles]
    published = re.sub(r'^inchikey\s*=\s*', '', first(header, 'INCHIKEY', 'INCHI_KEY', 'INCHIAUX'), flags=re.I)
    alias = KEY14.match(published)
    if info is None or alias is None or alias[0][:14] not in {info.raw, info.score}:
        return None
    return info, alias[0][:14]


def entry(header, peaks, assigned):
    title = header.get('TITLE', '')
    level = first(header, 'MSLEVEL', 'MS_LEVEL', 'MS_TYPE').upper().replace(' ', '')
    if level not in {'2', '2.0', 'MS2', 'MS/MS'} and (level or not re.search(r'\bMS\s*2\b', title, re.I)):
        return 'ms2_not_confirmed', None
    raw_adduct = first(header, 'ADDUCT', 'PRECURSOR_TYPE', 'ION_TYPE')
    if not raw_adduct:
        matches = set(ADDUCT_IN_TEXT.findall(title))
        raw_adduct = next(iter(matches)) if len(matches) == 1 else ''
    adduct = raw_adduct.upper().replace(' ', '').replace(']1', ']')
    if adduct not in ADDUCTS:
        return 'unsupported_or_missing_adduct', None
    polarity, shift = ADDUCTS[adduct]
    explicit_polarity = first(header, 'IONMODE', 'ION_MODE', 'POLARITY').casefold()
    if explicit_polarity and explicit_polarity not in ({'positive', 'pos', '+'} if shift > 0 else {'negative', 'neg', '-'}):
        return 'polarity_adduct_conflict', None
    charge = charge_value(first(header, 'CHARGE'))
    if charge is None:
        return 'unparsed_charge', None
    if not charge[0] or charge[1] not in ('', '+' if shift > 0 else '-'):
        return 'charge_adduct_conflict', None

    if assigned is None:
        return 'missing_or_inconsistent_structure', None
    info, alias = assigned
    value = first(header, 'PEPMASS').split()
    if not value or (mz := number(value[0])) is None or mz <= 0:
        return 'missing_exported_precursor', None
    activation = first(header, 'FRAGMENTATION_METHOD', 'FRAGMENTATION_MODE', 'ACTIVATION', 'ACTIVATION_METHOD') or title
    modes = set(re.findall(r'\b(?:HCD|CID)\b', activation.upper()))
    method = next(iter(modes)) if len(modes) == 1 else 'unresolved'
    provenance, field, pep_ppm, observed_ppm, rounded = precursor_evidence(header, value[0], info.mass + shift)
    valid_peaks = sum(0 < m < mz - 0.5 for m in set(peaks))
    if not valid_peaks:
        return 'no_fragment_peak_below_precursor', None
    return None, (info, alias, Evidence(polarity, method, scan_class(header), provenance,
                                       field, pep_ppm, observed_ppm, valid_peaks, rounded))


def families(pool, identities, field):
    groups = defaultdict(list)
    for key in pool:
        groups[getattr(identities[key], field)].append(key)
    rows = []
    for scaffold, keys in groups.items():
        observations = [obs for key in keys for obs in pool[key]]
        rows.append(dict(acyclic=not bool(scaffold), identities=len(keys), exported_records=len(observations),
                         fine_scaffolds=len({identities[key].fine for key in keys if identities[key].fine}),
                         both_polarities=sum(len({obs.polarity for obs in pool[key]}) == 2 for key in keys),
                         source_traced_single_records=sum(obs.scans == 'source_traced_single' for obs in observations),
                         observed_precursor_records=sum(obs.ppm_observed is not None for obs in observations)))
    return sorted(rows, key=lambda row: tuple(row[name] for name in row), reverse=True)


def describe(pool, identities):
    records = [obs for values in pool.values() for obs in values]
    observed = [obs.ppm_observed for obs in records if obs.ppm_observed is not None]
    try_spectra = [obs for obs in records if obs.scans == 'source_traced_single' and obs.activation == 'HCD'
                   and obs.fragment_peaks >= 4 and obs.ppm_observed is not None and abs(obs.ppm_observed) <= 10]
    return dict(identities=len(pool), exported_records=len(records),
                polarities={name: dict(records=sum(obs.polarity == name for obs in records),
                                       identities=sum(any(obs.polarity == name for obs in v) for v in pool.values()))
                            for name in ('positive', 'negative')},
                identities_with_both_polarities=sum(len({obs.polarity for obs in values}) == 2
                                                    for values in pool.values()),
                activation_counts=dict(Counter(obs.activation for obs in records)),
                acquisition_provenance_counts=dict(Counter(obs.scans for obs in records)),
                pepmass_provenance_counts=dict(Counter(obs.precursor for obs in records)),
                explicit_precursor_fields=dict(Counter(obs.observed_field for obs in records if obs.observed_field)),
                exported_pepmass_error_ppm=bins(obs.ppm_exported for obs in records),
                measured_precursor_error_ppm=bins(observed),
                measured_records=len(observed),
                theoretical_rounding_compatible_records=sum(obs.theoretical_round_match for obs in records),
                records_with_at_least_four_fragment_peaks=sum(obs.fragment_peaks >= 4 for obs in records),
                strict_native_measured_hcd_records=len(try_spectra),
                strict_native_measured_hcd_identities=sum(
                    any(obs.scans == 'source_traced_single' and obs.activation == 'HCD' and obs.fragment_peaks >= 4
                        and obs.ppm_observed is not None and abs(obs.ppm_observed) <= 10 for obs in values)
                    for values in pool.values()),
                cyclic_fine_murcko_groups=len({identities[key].fine for key in pool if identities[key].fine}),
                cyclic_generic_murcko_groups=len({identities[key].generic for key in pool if identities[key].generic}),
                fine_families=families(pool, identities, 'fine'), generic_families=families(pool, identities, 'generic'))


def describe_structures(records, identities):
    result = dict(identities=len(records), exported_records=sum(records.values()),
                  acyclic_identities=sum(not identities[key].generic for key in records))
    for level in ('fine', 'generic'):
        groups = defaultdict(Counter)
        for key, number_of_rows in records.items():
            scaffold = getattr(identities[key], level)
            groups[scaffold].update(identities=1, exported_records=number_of_rows)
        result[f'cyclic_{level}_murcko_groups'] = len(groups) - int('' in groups)
        result[f'{level}_families'] = sorted(
            (dict(**counts, acyclic=not scaffold) for scaffold, counts in groups.items()),
            key=lambda row: (row['identities'], row['exported_records'], row['acyclic']), reverse=True)
    return result


def run(index, exclusion, output):
    from rdkit import RDLogger, rdBase

    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError('Expected competition RDKit 2026.03.3')
    RDLogger.DisableLog('rdApp.*')
    chemistry, identities = {}, {}
    counts, failures, fields, parsing, shapes = Counter(), Counter(), Counter(), Counter(), Counter()
    structure_rows, extra_fields, peak_separators, charges = Counter(), Counter(), Counter(), Counter()
    charge_pairings = Counter()
    exported, aliases = defaultdict(list), defaultdict(set)
    with TemporaryDirectory(prefix='rypar-flavonoids-') as folder:
        path, provenance = download(RELEASE, ASSET, Path(folder))
        text, encoding = decode_mgf(path, provenance['sha256'])
        print('AUTHOR_MGF_DECODER', encoding, flush=True)
        for header, peaks in author_records(text, parsing, shapes, extra_fields, peak_separators):
            counts['mgf_records'] += 1
            counts['records_with_nonempty_peak_table'] += bool(peaks)
            fields.update(header.keys())
            charge = charge_descriptor(first(header, 'CHARGE'))
            charges[charge] += 1
            raw_adduct = first(header, 'ADDUCT', 'PRECURSOR_TYPE', 'ION_TYPE').upper().replace(' ', '').replace(']1', ']')
            adduct_class = raw_adduct if raw_adduct in ADDUCTS else 'other_or_missing'
            assigned = header_identity(header, chemistry)
            if assigned is None:
                counts['records_with_unresolved_structure_header'] += 1
            else:
                info, alias = assigned
                key = (info.score, info.heavy)
                identities[key] = info
                aliases[key].update((info.raw, info.score, alias))
                structure_rows[key] += 1
            reason, valid = entry(header, peaks, assigned)
            charge_pairings[(adduct_class, reason or 'spectrum_metadata_valid', charge)] += 1
            if reason:
                failures[reason] += 1
                continue
            info, alias, observation = valid
            key = (info.score, info.heavy)
            exported[key].append(observation)
    if not counts['mgf_records']:
        raise ValueError('No parsed spectra in the pinned author MGF export')
    train_keys, exact_train, train_report = train_exposure(index, identities)
    old_keys, excluded_fine, excluded_generic = (set(exclusion[item]) for item in ('identities', 'fine', 'generic'))
    drops = Counter()
    structure_unseen, structure_eligible = {}, {}
    for key, row_count in structure_rows.items():
        info = identities[key]
        if key in exact_train or aliases[key] & train_keys:
            drops['training_identity'] += 1
            continue
        structure_unseen[key] = row_count
        if any(digest_text(alias) in old_keys for alias in aliases[key]):
            drops['previous_panel_identity'] += 1
        elif ((info.fine and digest_text(info.fine) in excluded_fine)
              or (info.generic and digest_text(info.generic) in excluded_generic)):
            drops['previous_panel_scaffold'] += 1
        else:
            structure_eligible[key] = row_count
    unseen = {key: rows for key, rows in exported.items() if key in structure_unseen}
    eligible = {key: rows for key, rows in exported.items() if key in structure_eligible}
    report = dict(author='Rypar et al.', license=LICENSE, dataset=DATASET, original_raw_reference=RAWDATA,
                  paper=PAPER, rmassbank_precursor_reference=RMASSBANK, archive=provenance,
                  acquisition='Previously isolated geranylated flavonoids and flavonoid reference standards; targeted LC-HRMS/MS in both polarities, HCD/CID.',
                  export_provenance='Author workflow collected at least five original fragment scans per adduct/energy for processing. An unspecified MGF record is a processed export; acquisition IDs alone cannot certify native single-scan peaks.',
                  precursor_provenance='The usual RMassBank exporter rounds calculated precursor m/z. PEPMASS-only values remain provenance-unverified even when they match rounded theory. Only an explicit measured MS1/precursor field or PEPMASS origin is counted as observed.',
                  counts=dict(counts), mgf_decoding=encoding, mgf_parse_diagnostics=dict(parsing),
                  unrecognized_line_shapes=dict(shapes.most_common(24)),
                  unrecognized_colon_field_shapes=dict(extra_fields.most_common(24)),
                  positive_peak_separators=dict(peak_separators),
                  charge_field_fingerprint=dict(charges.most_common(30)),
                  charge_adduct_fingerprint=[dict(adduct=name, outcome=outcome, charge=charge, records=count)
                                             for (name, outcome, charge), count in charge_pairings.most_common(30)],
                  input_fields=dict(fields), spectrum_rejections=dict(failures),
                  train_index=train_report, rdkit=rdBase.rdkitVersion, prior_panel_scope=exclusion['sources'],
                  identity_exclusions=dict(drops), structure_ledger=dict(
                      status='evaluated' if structure_rows else 'no_resolvable_structure_headers',
                      scope='Neutral connected SMILES and concordant author InChIKey before any spectral eligibility check',
                      author_headers=describe_structures(structure_rows, identities),
                      training_unseen=describe_structures(structure_unseen, identities),
                      after_train_and_prior_panels=describe_structures(structure_eligible, identities)),
                  valid_export=describe(exported, identities),
                  training_unseen=describe(unseen, identities), after_train_and_prior_panels=describe(eligible, identities))
    report['coverage_status'] = 'evaluated' if exported else 'no_valid_export_to_evaluate'
    output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print('FLAVONOID_AVAILABILITY', counts['mgf_records'], 'exported records',
          len(exported), 'valid identities', len(unseen), 'train-unseen', len(eligible), 'fully held out', flush=True)

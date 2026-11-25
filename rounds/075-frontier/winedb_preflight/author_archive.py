"""Acquire and verify the original WineDB MassIVE library and metadata."""

from io import BytesIO
from time import monotonic, sleep
from urllib.error import HTTPError, URLError
from xml.etree import ElementTree
from zipfile import ZipFile


WINE_ACCESSION = 'MSV000104069'
WINE_TASK = '3edfe5c92cba405f942978f0f3fe6e56'
WINE_LANDING = f'https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?task={WINE_TASK}'
WINE_FILE_INDEX = f'https://massive.ucsd.edu/ProteoSAFe/dataset_files.jsp?task={WINE_TASK}'
WINE_DOWNLOAD = f'https://massiveproxy.gnps2.org/massiveproxy/{WINE_ACCESSION}/'
WINE_WORKBOOK = 'metadata/Supplementary-Info-WDB_V9.xlsx'
WINE_WORKBOOK_SHA = 'ab9e45bc6b3fac63abaee48fee99013044a5a7cd6ddf5e564dc899e0f1398d3a'
WINE_WORKBOOK_HEADER = (
    'FILENAME', 'COMPOUND_NAME', 'PEPMASS', 'INSTRUMENT', 'IONSOURCE', 'SMILES', 'INCHI',
    'IONMODE', 'ACQUISITION', 'EXACTMASS', 'ADDUCT', 'LIBQUALITY', 'CASNUMBER',
    'COLLISION ENERGIES', 'COLLISION ENERGY UNITS', 'FRAGMENTATION METHOD',
)
WINE_LIBRARY = {
    'library/WineDB_Orbitrap_NCE10-30-50.mgf': (364758, 'orbitrap', (10, 30, 50), 'NCE', 'HCD'),
    'library/WineDB_QTOF_20eV.mgf': (285717, 'qtof', (20,), 'eV', 'CID'),
    'library/WineDB_QTOF_30eV.mgf': (415891, 'qtof', (30,), 'eV', 'CID'),
    'library/WineDB_QTOF_40eV.mgf': (338123, 'qtof', (40,), 'eV', 'CID'),
    'library/WineDB_QTOF_50eV.mgf': (157802, 'qtof', (50,), 'eV', 'CID'),
    'library/WineDB_QTOF_merged.mgf': (359922, 'qtof', None, 'eV', 'CID'),
}
XML = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
CE_VALUE = re.compile(r'\d+(?:\.\d+)?(?:\s*[,;/+-]\s*\d+(?:\.\d+)?)*')


def author_download(files):
    # The official public MassIVE mirror allows one download request per minute.
    last_start = None
    for path, expected_size in files.items():
        url = WINE_DOWNLOAD + path
        for attempt in range(3):
            if last_start is not None:
                sleep(max(0, 62 - (monotonic() - last_start)))
            last_start = monotonic()
            request = Request(url, headers={'Accept-Encoding': 'identity', 'User-Agent': 'CASMI-research/1.0'})
            try:
                with urlopen(request, timeout=35) as response:
                    data = response.read(expected_size + 1)
                    receipt = dict(path=path, url=url, actual_url=response.geturl(), bytes=len(data),
                                   sha256=sha256(data).hexdigest(), etag=response.headers.get('ETag'),
                                   last_modified=response.headers.get('Last-Modified'))
            except (HTTPError, URLError, TimeoutError) as error:
                if attempt == 2 or isinstance(error, HTTPError) and error.code not in {429, 500, 502, 503, 504}:
                    raise
                print(f'Retrying {path}: {error}', flush=True)
                continue
            if len(data) != expected_size:
                raise ValueError(f'{path}: expected {expected_size} bytes, received {len(data)}')
            print(f'Acquired {path} {receipt["sha256"]}', flush=True)
            yield path, data, receipt
            break


def author_workbook(data):
    if sha256(data).hexdigest() != WINE_WORKBOOK_SHA:
        raise ValueError('Unexpected WineDB supplementary workbook digest')
    with ZipFile(BytesIO(data)) as workbook:
        shared = [''.join(node.itertext()) for node in ElementTree.fromstring(
            workbook.read('xl/sharedStrings.xml')).findall('x:si', XML)]
        main = ElementTree.fromstring(workbook.read('xl/worksheets/sheet1.xml'))
        rows = []
        for row in main.findall('x:sheetData/x:row', XML):
            values = {}
            for cell in row:
                col = cell.get('r', '').rstrip('0123456789')
                value = cell.findtext('x:v', namespaces=XML)
                if value is not None:
                    values[col] = shared[int(value)] if cell.get('t') == 's' else value
            rows.append((int(row.get('r')), tuple(values.get(col) for col in 'ABCDEFGHIJKLMNOP')))
    if not rows or rows[0] != (1, WINE_WORKBOOK_HEADER) or len(rows) != 1577:
        raise ValueError('Unexpected WineDB supplementary workbook schema or row count')
    records = {}
    for number, values in rows[1:]:
        row = dict(zip(WINE_WORKBOOK_HEADER, values, strict=True))
        filename = row['FILENAME']
        if not filename or filename in records:
            raise ValueError(f'Empty or duplicate author workbook filename at row {number}')
        row['sheet'], row['sheet_row'] = 'Feuil1', number
        records[filename] = row
    return records


def mgf_blocks(data, path):
    """Retain exact byte-level spectrum locations plus unmodified MGF header values."""
    start = None
    offset = 0
    number = 0
    for lineno, raw in enumerate(data.splitlines(keepends=True), start=1):
        line = raw.strip().decode('utf-8-sig')
        if line == 'BEGIN IONS':
            if start is not None:
                raise ValueError(f'{path}:{lineno}: nested spectrum block')
            start, first_line, fields, peaks = offset, lineno, {}, []
        elif line == 'END IONS':
            if start is None:
                raise ValueError(f'{path}:{lineno}: unmatched end of spectrum block')
            number += 1
            end = offset + len(raw)
            yield dict(block=number, first_line=first_line, last_line=lineno, byte_offset=start,
                       byte_length=end - start, sha256=sha256(data[start:end]).hexdigest()), fields, peaks
            start = None
        elif start is not None and line and not line.startswith(('#', '!', ';')):
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip().upper()
                if key in fields:
                    raise ValueError(f'{path}:{lineno}: duplicate MGF field {key}')
                fields[key] = value.strip()
            else:
                mz, abundance, *_ = line.split()
                peaks.append((float(mz), float(abundance)))
        offset += len(raw)
    if not number or start is not None:
        raise ValueError(f'{path}: empty MGF or unterminated spectrum block')


def spectrum_crosswalk():
    exports = {}
    index = {}
    for name, (instrument, size) in BUNDLES.items():
        desc, rows = download(name, size)
        exports[name] = desc
        for record in rows:
            source_file = str(record.get('source_file') or '')
            if source_file:
                index[instrument, record['spectrum_id']] = (source_file, record.get('url'), record.get('task'))
    return index, exports


def author_filename(fields, workbook, crosswalk, instrument):
    title, accession = fields.get('TITLE'), fields.get('SPECTRUMID')
    if not title or title != accession:
        return None, 'conflicting_or_missing_spectrum_accession', {}
    usi = f'mzspec:GNPS:WINE-DB-{instrument.upper()}:accession:{accession}'
    if fields.get('USI') != usi:
        return None, 'unexpected_spectrum_usi', {}
    crossed = crosswalk.get((instrument, accession))
    pending = f'{accession.removeprefix("PENDING_")}.mgf' if accession.startswith('PENDING_') else None
    if pending and crossed and pending != crossed[0]:
        return None, 'conflicting_author_filename_keys', {}
    key = pending or (crossed[0] if crossed else None)
    if key not in workbook:
        return None, 'no_exact_author_filename_match', {}
    if pending:
        return key, 'author_pending_accession_exact_filename', {}
    return key, 'gnps_accession_to_exact_author_filename', dict(url=crossed[1], task=crossed[2])


def energy_values(text):
    value = re.sub(r'\b(?:eV|NCE)\b|%', '', text or '', flags=re.I).strip()
    if not CE_VALUE.fullmatch(value):
        return None
    return tuple(Decimal(item) for item in re.split(r'\s*[,;/+-]\s*', value))


@lru_cache(maxsize=1024)
def checked_structure(smiles, inchi):
    from rdkit.Chem import InchiToInchiKey

    info = identity(smiles, scaffold=True)
    author_key = InchiToInchiKey(inchi) if inchi else None
    if not info or not author_key or author_key[:14] not in {info['raw_key'], info['key']}:
        return None
    return info


def author_record(fields, peaks, workbook, library_info):
    _, instrument, expected_ce, unit, activation = library_info
    declared_instrument = re.sub(r'[^a-z]', '', workbook['INSTRUMENT'].lower())
    exported_instrument = re.sub(r'[^a-z]', '', fields.get('SOURCE_INSTRUMENT', '').lower()).removeprefix('esi')
    polarity = workbook['IONMODE'].lower()
    mode = fields.get('IONMODE', '').lower()
    if (declared_instrument != instrument or exported_instrument != instrument or mode != polarity
            or fields.get('MSLEVEL') != '2' or polarity not in {'positive', 'negative'}):
        return 'instrument_ionisation_or_ms_level_conflict', None
    mgf_ce_text = fields.get('COLLISION_ENERGY')
    mgf_ce = energy_values(mgf_ce_text)
    author_ce = energy_values(workbook['COLLISION ENERGIES'])
    if (not author_ce or workbook['COLLISION ENERGY UNITS'].lower() != unit.lower()
            or workbook['FRAGMENTATION METHOD'] != activation or fields.get('FRAGMENTATION_METHOD') != activation):
        return 'author_energy_unit_or_activation_conflict', None
    if (expected_ce is not None and author_ce != expected_ce
            or expected_ce is None and (len(author_ce) < 2 or not set(author_ce) <= {20, 30, 40, 50})
            or mgf_ce is not None and mgf_ce != author_ce):
        return 'author_collision_energy_conflict', None
    mgf_unit = re.search(r'\b(?:eV|NCE)\b|%', mgf_ce_text or '', re.I)
    if mgf_unit and mgf_unit.group().lower() not in {unit.lower(), '%' if unit == 'NCE' else unit.lower()}:
        return 'mgf_energy_unit_conflict', None
    if workbook['LIBQUALITY'] != '1' or fields.get('LIBRARYQUALITY') != '1':
        return 'non_reference_quality_spectrum', None
    standard = workbook['ACQUISITION'].lower()
    if standard not in {'commercial', 'isolated'}:
        return 'explicit_standard_provenance_missing', None
    raw_adduct = workbook['ADDUCT']
    adduct = ADDUCTS.get(raw_adduct)
    if not adduct or polarity != ('positive' if adduct.endswith('+') else 'negative'):
        return 'not_an_explicit_protonated_or_deprotonated_adduct', None
    mgf_adduct = re.search(r'\[[^\[\]]+\]\d?[+-]$', fields.get('NAME', ''))
    if mgf_adduct and mgf_adduct.group() != adduct:
        return 'adduct_conflict', None
    smiles = workbook['SMILES']
    info = checked_structure(smiles, workbook['INCHI'])
    exported_structure = checked_structure(fields.get('SMILES'), fields.get('INCHI'))
    if not info or not exported_structure or any(info[k] != exported_structure[k] for k in ('key', 'heavy')):
        return 'author_structure_invalid_or_conflicting', None
    try:
        mz_text = fields['PEPMASS'].split()[0]
        decimal = Decimal(mz_text)
        precursor = float(decimal)
        workbook_mz = Decimal(workbook['PEPMASS'])
        workbook_exact_mass = float(workbook['EXACTMASS'])
        if (not isfinite(precursor) or precursor <= 0 or not isfinite(workbook_exact_mass)
                or abs(decimal - workbook_mz) > Decimal('0.000001')):
            raise ValueError
    except (KeyError, ValueError, TypeError, InvalidOperation, IndexError):
        return 'no_valid_matching_reported_precursor', None
    fragments = {mass for mass, intensity in peaks if isfinite(mass) and isfinite(intensity)
                 and 0 < mass < precursor - 0.5 and intensity > 0}
    if len(fragments) < 4:
        return 'fewer_than_four_product_fragments', None
    expected_mz = info['mass'] + (PROTON if polarity == 'positive' else -PROTON)
    raw_usi = fields.get('SOURCE_SCAN_USI')
    validated_raw_reference = bool(raw_usi and re.match(r'mzspec:MSV\d+:raw/.+:(?:scan|index):\d+$', raw_usi))
    native_status = 'raw_scan_referenced_centroid_unverified' if validated_raw_reference else 'library_export_native_centroid_unresolved'
    return None, dict(name=workbook['COMPOUND_NAME'], standard_source=standard, instrument=instrument,
                      polarity=polarity, smiles=smiles, **info, adduct=adduct, author_adduct=raw_adduct,
                      precursor_mz=precursor, expected_mz=expected_mz,
                      precursor_error_ppm=(precursor - expected_mz) / expected_mz * 1e6,
                      reported_decimal_places=max(0, -decimal.as_tuple().exponent), fragment_count=len(fragments),
                      author_mass_source=native_status, author_mgf_pepmass=fields['PEPMASS'],
                      author_workbook_pepmass=workbook['PEPMASS'], author_workbook_exactmass=workbook['EXACTMASS'],
                      ce_scope='single_reported_with_unit' if len(author_ce) == 1 else 'merged_reported_with_unit',
                      ce_values=[float(energy) for energy in author_ce], ce_unit=unit,
                      ce_mgf_verified=mgf_ce == author_ce, ce_mgf_report=mgf_ce_text, activation=activation,
                      raw_provenance={'source_scan_usi': raw_usi} if validated_raw_reference else {})


def author_summary(rows):
    return dict(**summarize(rows), isolated_identities=len({r['key'] for r in rows if r['standard_source'] == 'isolated'}),
                qtof_single_ce_identities=len({r['key'] for r in rows if r['instrument'] == 'qtof'
                                               and r['ce_scope'] == 'single_reported_with_unit'}),
                verified_mgf_ce=sum(r['ce_mgf_verified'] for r in rows),
                precursor_origin=dict(Counter(r['author_mass_source'] for r in rows)))


def run_author(train, output):
    from rdkit import RDLogger, rdBase

    if rdBase.rdkitVersion != '2026.03.3':
        raise RuntimeError('Run with the competition RDKit wheel')
    RDLogger.DisableLog('rdApp.*')
    output.mkdir(parents=True, exist_ok=True)
    crosswalk, gnps_exports = spectrum_crosswalk()
    collected = {}
    public = []
    files = {WINE_WORKBOOK: 164567} | {path: row[0] for path, row in WINE_LIBRARY.items()}
    seen_keys = set()
    workbook = None
    for path, data, fetched in author_download(files):
        stash = output / 'winedb_author_release' / path
        stash.parent.mkdir(parents=True, exist_ok=True)
        if path == WINE_WORKBOOK:
            workbook = author_workbook(data)
            collected[path] = dict(**fetched, registry_rows=len(workbook))
            stash.write_bytes(data)
            continue
        accepted = []
        rejected = Counter()
        join_modes = Counter()
        bundle_headers = Counter()
        unresolved = []
        info = WINE_LIBRARY[path]
        for block, fields, peaks in mgf_blocks(data, path):
            bundle_headers.update(fields.keys())
            filename, method, gnps_meta = author_filename(fields, workbook, crosswalk, info[1])
            if filename is None:
                rejected[method] += 1
                unresolved.append(dict(**block, title=fields.get('TITLE'), accession=fields.get('SPECTRUMID'),
                                       usi=fields.get('USI'), reason=method))
                continue
            if filename in seen_keys:
                rejected['duplicate_workbook_key_in_author_release'] += 1
                unresolved.append(dict(**block, title=fields['TITLE'], accession=fields['SPECTRUMID'],
                                       usi=fields['USI'], author_filename=filename,
                                       reason='duplicate_workbook_key_in_author_release'))
                continue
            seen_keys.add(filename)
            join_modes[method] += 1
            reason, parsed = author_record(fields, peaks, workbook[filename], info)
            if parsed is None:
                rejected[reason] += 1
                unresolved.append(dict(**block, title=fields['TITLE'], accession=fields['SPECTRUMID'],
                                       usi=fields['USI'], author_filename=filename, reason=reason))
                continue
            meta = workbook[filename]
            accepted.append(dict(**parsed, accession=fields['SPECTRUMID'], usi=fields['USI'], source_path=path,
                                 source_accession=WINE_ACCESSION, source_block=block, join_method=method,
                                 author_filename=filename, workbook_sheet=meta['sheet'], workbook_row=meta['sheet_row'],
                                 original_export_filename_header=fields.get('FILENAME'),
                                 original_export_scan_header=fields.get('SCANS'), accession_crosswalk=gnps_meta))
        collected[path] = dict(**fetched, original_blocks=len(accepted) + sum(rejected.values()),
                               workbook_joins=dict(join_modes), mgf_header_counts=dict(bundle_headers),
                               parsed=author_summary(accepted), rejections=dict(rejected), unresolved=unresolved)
        stash.write_bytes(data)
        public.extend(accepted)
        print(json.dumps(dict(path=path, parsed=len(accepted), joins=dict(join_modes), rejected=dict(rejected)),
                         sort_keys=True), flush=True)
    if not public:
        raise ValueError('No author records passed exact source and structure checks')
    raw, metric, train_receipt = training_overlap(train, public)
    for row in public:
        training_sources = set(metric.get((row['key'], row['heavy']), ()))
        training_sources.update(raw.get(row['key'], ()))
        training_sources.update(raw.get(row['raw_key'], ()))
        row['training_sources_matching_raw_or_metric_identity'] = sorted(training_sources)
    novel = [row for row in public if not row['training_sources_matching_raw_or_metric_identity']]
    novel_10ppm = [row for row in novel if abs(row['precursor_error_ppm']) <= 10]
    rows_path = output / 'wine_db_author_public_rows.json'
    rows_path.write_text(json.dumps(public, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    receipt = dict(accession=WINE_ACCESSION, license_record=WINE_LANDING, license='CC0 1.0 Universal',
                   file_index=WINE_FILE_INDEX, download_service='https://massiveproxy.gnps2.org/',
                   author_files=collected, author_workbook_rows=len(workbook), author_workbook_keys_linked=len(seen_keys),
                   author_workbook_keys_unlinked=sorted(workbook.keys() - seen_keys),
                   gnps_accession_crosswalk_sources=gnps_exports, rdkit=rdBase.rdkitVersion, training_overlap=train_receipt,
                   all_parsed=author_summary(public), training_identity_unseen=author_summary(novel),
                   training_identity_unseen_within_10ppm=author_summary(novel_10ppm),
                   per_bundle_training_identity_unseen_within_10ppm={
                       path: author_summary([r for r in novel_10ppm if r['source_path'] == path]) for path in WINE_LIBRARY},
                   rows_file=rows_path.name, rows_sha256=sha256(rows_path.read_bytes()).hexdigest(),
                   precursor_scope='Verbatim MGF PEPMASS. Workbook EXACTMASS is kept explicitly separate. Library SCANS and GNPS USIs identify an exported record. The raw centroid remains unverified.',
                   independence_scope='All six instrument/energy bundles are one WineDB study. This audit checks competition train identities only. Prior blind labels and private exclusions are absent.')
    (output / 'wine_db_author_availability.json').write_text(json.dumps(receipt, indent=2, sort_keys=True,
                                                                       ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({k: receipt[k] for k in ('all_parsed', 'training_identity_unseen',
                                            'training_identity_unseen_within_10ppm')}, sort_keys=True), flush=True)

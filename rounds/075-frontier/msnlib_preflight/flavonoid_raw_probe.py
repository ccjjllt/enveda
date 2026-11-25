"""Audit scan and precursor provenance in one public MassIVE confirmation mzML."""

from collections import Counter
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


DATASET = 'MSV000098757'
DATASET_FILES = 'https://massive.ucsd.edu/ProteoSAFe/dataset_files.jsp?task=f4420d487b344b9eabe6fa3f65695fee'
DATASET_DOI = 'https://doi.org/10.25345/C5B56DH7P'
FILE = 'raw/confirmation/PT24CH-fraction7-1-ii.mzML'
FILE_BYTES = 53_811_487
MASSIVE_PROXY = f'https://massiveproxy.gnps2.org/massiveproxy/{DATASET}'
MASSIVE_ARCHIVE = f'https://massive-ftp.ucsd.edu/v10/{DATASET}'


def download(path, relative_path, file_bytes):
    errors = []
    for root in (MASSIVE_PROXY, MASSIVE_ARCHIVE):
        url = f'{root}/{relative_path}'
        digest = sha256()
        try:
            request = Request(url, headers={'User-Agent': 'Enveda-Raw-Metadata-Probe/1.0'})
            with urlopen(request, timeout=90) as response, path.open('wb') as output:
                received = 0
                while block := response.read(1 << 20):
                    received += len(block)
                    if received > file_bytes:
                        raise ValueError('Artifact exceeded its official size')
                    output.write(block)
                    digest.update(block)
            if received != file_bytes:
                raise ValueError('Artifact size differs from the official file table')
            return dict(relative_path=relative_path, bytes=received, sha256=digest.hexdigest(), download_url=url)
        except (OSError, ValueError) as error:
            errors.append(f'{root.split("/")[2]}: {type(error).__name__}')
    raise RuntimeError('The single pinned MassIVE file could not be retrieved. ' + ', '.join(errors))


def cv(node, groups):
    if node is None:
        return {}
    fields = {}
    for reference in node.findall('{*}referenceableParamGroupRef'):
        fields.update(groups.get(reference.get('ref'), {}))
    fields.update({child.get('accession'): child.attrib for child in node.findall('{*}cvParam')})
    return fields


def numeric(fields, accession):
    try:
        value = float(fields[accession]['value'])
    except (KeyError, TypeError, ValueError):
        return None
    return value if isfinite(value) else None


def positive(fields, accession):
    value = numeric(fields, accession)
    return value is not None and value > 0


def polarity(fields):
    pos, neg = 'MS:1000130' in fields, 'MS:1000129' in fields
    return 'positive' if pos and not neg else 'negative' if neg and not pos else 'missing_or_conflicting'


def precursor_fields(element, groups):
    selected = [cv(ion, groups) for ion in element.findall('./{*}selectedIonList/{*}selectedIon')]
    activation = cv(element.find('{*}activation'), groups)
    isolation = cv(element.find('{*}isolationWindow'), groups)
    charges = [numeric(ion, 'MS:1000041') for ion in selected]
    return {
        'spectrum_ref': element.get('spectrumRef'),
        'external_reference': bool(element.get('sourceFileRef') or element.get('externalSpectrumID')),
        'has_selected_ion_mz': any(positive(ion, 'MS:1000744') for ion in selected),
        'has_charge_state': any(charge is not None and charge != 0 and charge.is_integer() for charge in charges),
        'explicit_charge_states': sorted({int(charge) for charge in charges
                                         if charge is not None and charge != 0 and charge.is_integer()}),
        'has_possible_charge_state': any('MS:1000633' in ion for ion in selected),
        'has_selected_ion_mz_and_charge': any(positive(ion, 'MS:1000744') and charge is not None
                                             and charge != 0 and charge.is_integer()
                                             for ion, charge in zip(selected, charges, strict=True)),
        'one_selected_ion_mz_and_unit_charge': (len(selected) == 1
                                               and positive(selected[0], 'MS:1000744')
                                               and charges[0] is not None and abs(charges[0]) == 1),
        'isolation_target_mz': positive(isolation, 'MS:1000827'),
        'collision_energy': numeric(activation, 'MS:1000045') is not None,
        'hcd': 'MS:1000422' in activation or any(
            'higher energy collisional dissociation' in field.get('name', '').casefold().replace('-', ' ')
            for field in activation.values()),
    }


def inspect(path):
    groups, scans, ms2 = {}, {}, {}
    counts, levels, polar, charge_histogram = Counter(), Counter(), Counter(), Counter()
    for _, element in ET.iterparse(path, events=('end',)):
        name = element.tag.rsplit('}', 1)[-1]
        if name == 'referenceableParamGroup':
            groups[element.get('id')] = cv(element, groups)
            element.clear()
        elif name == 'spectrum':
            identity = element.get('id')
            if not identity or identity in scans:
                raise ValueError('A spectrum native ID is missing or repeated')
            fields = cv(element, groups)
            level = numeric(fields, 'MS:1000511')
            level_name = str(int(level)) if level is not None and level.is_integer() and level > 0 else 'missing_or_invalid'
            mode = polarity(fields)
            scans[identity] = (level_name, mode)
            levels[level_name] += 1
            polar[f'MS{level_name}/{mode}'] += 1
            if level_name == '2':
                scan = element.find('./{*}scanList/{*}scan')
                retention = numeric(cv(scan, groups), 'MS:1000016')
                try:
                    length = int(element.get('defaultArrayLength', '0'))
                except ValueError:
                    length = 0
                arrays = [cv(array, groups) for array in element.findall('./{*}binaryDataArrayList/{*}binaryDataArray')]
                array_types = set().union(*(value.keys() for value in arrays))
                ms2[identity] = (mode, retention is not None and retention >= 0,
                                 length >= 4 and {'MS:1000514', 'MS:1000515'} <= array_types,
                                 [precursor_fields(item, groups)
                                  for item in element.findall('./{*}precursorList/{*}precursor')])
            element.clear()
    for mode, retention, arrays, precursors in ms2.values():
        counts['ms2_with_retention_time'] += retention
        counts['ms2_with_declared_mz_intensity_arrays_ge4'] += arrays
        counts['ms2_with_precursor_entry'] += bool(precursors)
        for field in ('has_selected_ion_mz', 'has_charge_state', 'has_possible_charge_state',
                      'has_selected_ion_mz_and_charge', 'isolation_target_mz', 'collision_energy', 'hcd'):
            counts[f'ms2_{field}'] += any(row[field] for row in precursors)
        charge_histogram.update({str(charge) for row in precursors for charge in row['explicit_charge_states']})
        local_refs = [row for row in precursors if row['spectrum_ref'] and not row['external_reference']]
        counts['ms2_with_local_spectrum_ref'] += bool(local_refs)
        counts['ms2_with_external_precursor_reference'] += any(row['external_reference'] for row in precursors)
        counts['ms2_with_unresolved_local_ref'] += any(row['spectrum_ref'] not in scans for row in local_refs)
        parent_ms1 = [row for row in local_refs if scans.get(row['spectrum_ref'], (None,))[0] == '1']
        same_polarity = [row for row in parent_ms1 if mode != 'missing_or_conflicting'
                         and scans[row['spectrum_ref']][1] == mode]
        counts['ms2_with_resolved_ms1_parent'] += bool(parent_ms1)
        counts['ms2_with_same_polarity_ms1_parent'] += bool(same_polarity)
        usable = (retention and arrays and len(precursors) == len(same_polarity) == 1
                  and same_polarity[0]['one_selected_ion_mz_and_unit_charge']
                  and same_polarity[0]['hcd'])
        counts['ms2_strict_native_metadata_ready'] += usable
        if usable:
            counts[f'ms2_strict_native_metadata_ready_{mode}'] += 1
    return dict(spectra=len(scans), ms_levels=dict(levels), ms_levels_and_polarity=dict(polar),
                ms2_metadata=dict(counts), has_ms2=bool(ms2),
                ms2_explicit_charge_state_spectra=dict(charge_histogram),
                native_hcd_metadata_candidates=counts['ms2_strict_native_metadata_ready'])


def run(output, fallback, *, relative_path=FILE, file_bytes=FILE_BYTES, acquisition_scope='isolated_confirmation'):
    with TemporaryDirectory(prefix='paulownia-single-mzml-') as directory:
        file = Path(directory) / Path(relative_path).name
        provenance = download(file, relative_path, file_bytes)
        audit = inspect(file)
    empty = 'no_ms2_in_confirmation' if acquisition_scope == 'isolated_confirmation' else 'no_ms2_in_numbered_dda'
    status = ('metadata_ready' if audit['native_hcd_metadata_candidates'] else
              'ms2_present_but_no_strict_metadata_candidate' if audit['has_ms2'] else empty)
    report = dict(dataset=DATASET, dataset_doi=DATASET_DOI, license='CC0 1.0',
                  official_file_table=DATASET_FILES, raw_file=provenance, coverage_status=status,
                  acquisition_scope=acquisition_scope,
                  interpretation='Selected-ion m/z is instrument metadata. The parent MS1 centroid and compound identity were not independently verified. Spectrum IDs, peak arrays and structures were not exported.',
                  **audit)
    if fallback is not None:
        report['smallest_same_fraction_manuscript_file'] = fallback
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print('FLAVONOID_SINGLE_RAW_METADATA', status, audit['ms_levels'], flush=True)

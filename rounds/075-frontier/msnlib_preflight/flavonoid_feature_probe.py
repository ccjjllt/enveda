"""Locate two author's chromatographic features in one pinned MassIVE DDA file."""

from array import array
import base64
from collections import Counter
import json
from math import fsum, isfinite
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET
import zlib

from flavonoid_raw_probe import DATASET, DATASET_DOI, DATASET_FILES, cv, download, numeric, polarity, precursor_fields


FILE = 'raw/manuscript/PT24-CH-fraction7_ID1.mzML.mzML'
FILE_BYTES = 122_239_367
FILE_SHA256 = 'bff008ead50830bdece0dca40d04bbeb82576ce9cdc1b3897f710b4c702e4a34'
CONFIRMATION = 'raw/confirmation/PT24CH-fraction7-1-ii.mzML'
ARTICLE = 'https://www.mdpi.com/2218-1989/15/9/616'
SUPPLEMENT = 'https://pub.mdpi-res.com/d_attachment/metabolites/metabolites-15-00616/article_deploy/metabolites-15-00616-s001.zip'
REFERENCE = 'datafile:Samp1_neg:fraction7_ID1.mzML:area'
RT_MINUTES = 0.07
MASS_PPM = 8.0
FEATURES = {
    '8830': dict(role='author_target', mz=385.12935, rt_min=15.368500000000001,
                 table_row=1024, observed_adduct_in_table=None),
    '8590': dict(role='same_mass_rt_control', mz=385.12949, rt_min=15.159666666666668,
                 table_row=966, observed_adduct_in_table='[M-H]-'),
}


def spectra(path):
    groups = {}
    for _, element in ET.iterparse(path, events=('end',)):
        element_name = element.tag.rsplit('}', 1)[-1]
        if element_name == 'referenceableParamGroup':
            groups[element.get('id')] = cv(element, groups)
            element.clear()
        elif element_name == 'spectrum':
            yield element, groups
            element.clear()


def minutes(scan, groups):
    fields = cv(scan.find('./{*}scanList/{*}scan'), groups)
    value = numeric(fields, 'MS:1000016')
    if value is None or value < 0:
        return None
    field = fields['MS:1000016']
    accession, unit = field.get('unitAccession'), field.get('unitName', '').casefold()
    if accession == 'UO:0000031' or unit in {'minute', 'minutes', 'min'}:
        return value
    if accession == 'UO:0000010' or unit in {'second', 'seconds', 's'}:
        return value / 60
    return None


def within_mz(mz, reference):
    return abs(mz - reference) <= MASS_PPM * 1e-6 * reference


def match_ms2(path):
    counts = {feature_id: Counter() for feature_id in FEATURES}
    matches = {feature_id: [] for feature_id in FEATURES}
    first_pass = Counter()
    for scan, groups in spectra(path):
        fields = cv(scan, groups)
        if numeric(fields, 'MS:1000511') != 2 or polarity(fields) != 'negative':
            continue
        first_pass['negative_ms2'] += 1
        rt = minutes(scan, groups)
        if rt is None:
            first_pass['ms2_missing_supported_rt_units'] += 1
            continue
        for feature_id, feature in FEATURES.items():
            if abs(rt - feature['rt_min']) > RT_MINUTES:
                continue
            counter = counts[feature_id]
            counter['negative_ms2_in_rt_window'] += 1
            precursors = scan.findall('./{*}precursorList/{*}precursor')
            if len(precursors) != 1:
                counter['not_one_precursor'] += 1
                continue
            precursor, = precursors
            ions = precursor.findall('./{*}selectedIonList/{*}selectedIon')
            if len(ions) != 1:
                counter['not_one_selected_ion'] += 1
                continue
            ion = cv(ions[0], groups)
            selected_mz = numeric(ion, 'MS:1000744')
            if selected_mz is None or not within_mz(selected_mz, feature['mz']):
                continue
            counter['negative_ms2_in_rt_and_mz_window'] += 1
            info = precursor_fields(precursor, groups)
            charge = numeric(ion, 'MS:1000041')
            activation = cv(precursor.find('{*}activation'), groups)
            collision = numeric(activation, 'MS:1000045')
            isolation = cv(precursor.find('{*}isolationWindow'), groups)
            target = numeric(isolation, 'MS:1000827')
            lower = numeric(isolation, 'MS:1000828')
            upper = numeric(isolation, 'MS:1000829')
            bounds = None
            if target is not None and target > 0 and lower is not None and upper is not None and lower >= 0 and upper >= 0 and lower + upper > 0:
                bounds = (target - lower, target + upper)
            flags = dict(one_unit_charge_state=charge is not None and abs(charge) == 1,
                         local_parent_reference=bool(info['spectrum_ref'] and not info['external_reference']),
                         recorded_hcd_energy=info['hcd'] and collision is not None and collision > 0,
                         selected_mz_in_explicit_isolation=bool(bounds and bounds[0] <= selected_mz <= bounds[1]))
            counter.update({name: passed for name, passed in flags.items()})
            if not all(flags.values()):
                continue
            own_id = scan.get('id')
            if not own_id:
                raise ValueError('Missing native ID for a selected HCD scan')
            counter['negative_hcd_with_native_metadata'] += 1
            matches[feature_id].append(dict(native_scan_id=own_id, parent_native_scan_id=info['spectrum_ref'],
                                            rt_min=rt, selected_mz=selected_mz, charge_state_as_recorded=int(charge),
                                            hcd_energy=collision,
                                            hcd_energy_unit=activation['MS:1000045'].get('unitAccession'),
                                            hcd_energy_unit_name=activation['MS:1000045'].get('unitName'),
                                            isolation_target_mz=target, isolation_lower_mz_offset=lower,
                                            isolation_upper_mz_offset=upper, isolation_width_mz=lower+upper,
                                            isolation_bounds_mz=bounds))
    return matches, counts, first_pass


def raw_arrays(scan, groups):
    marker = cv(scan, groups)
    profile, centroid = 'MS:1000128' in marker, 'MS:1000127' in marker
    if profile == centroid:
        raise ValueError('missing_or_conflicting_ms1_array_representation')
    values = {}
    encodings = []
    for item in scan.findall('./{*}binaryDataArrayList/{*}binaryDataArray'):
        fields = cv(item, groups)
        has_mz, has_intensity = 'MS:1000514' in fields, 'MS:1000515' in fields
        if not has_mz and not has_intensity:
            continue
        kind = 'mz' if has_mz else 'intensity'
        if (has_mz and has_intensity) or kind in values:
            raise ValueError('duplicate_or_ambiguous_ms1_array')
        try:
            binary = item.find('{*}binary')
            if binary is None or not binary.text:
                raise ValueError('missing_ms1_binary_data')
            raw = base64.b64decode(binary.text)
            if 'MS:1000574' in fields:
                raw = zlib.decompress(raw)
                encoding = 'zlib'
            elif 'MS:1000576' in fields:
                encoding = 'none'
            else:
                raise ValueError('unsupported_ms1_array_compression')
            if 'MS:1000523' in fields:
                typecode = 'd'
            elif 'MS:1000521' in fields:
                typecode = 'f'
            else:
                raise ValueError('unsupported_ms1_float_precision')
            data = array(typecode)
            data.frombytes(raw)
            if sys.byteorder != 'little':
                data.byteswap()
        except (ValueError, zlib.error) as error:
            if str(error).startswith(('missing_ms1_', 'unsupported_ms1_')):
                raise
            raise ValueError('invalid_ms1_binary_data') from error
        values[kind] = data
        encodings.append(f'{kind}/{encoding}/float{data.itemsize * 8}')
    mz, intensity = values.get('mz', ()), values.get('intensity', ())
    try:
        declared = int(scan.get('defaultArrayLength', '0'))
    except ValueError as error:
        raise ValueError('invalid_ms1_array_declared_length') from error
    if len(mz) != len(intensity) or len(mz) != declared or len(mz) < 3:
        raise ValueError('invalid_ms1_array_length')
    if any(not isfinite(value) or value <= 0 for value in mz) or any(not isfinite(value) for value in intensity):
        raise ValueError('invalid_ms1_mass_or_intensity_values')
    if any(left > right for left, right in zip(mz, mz[1:])):
        raise ValueError('unsorted_ms1_mz_array')
    return mz, intensity, 'profile_local_apices' if profile else 'native_centroids', encodings


def parent_evidence(row, feature, mode, rt, decoded):
    if mode != 'negative':
        return 'resolved_parent_is_not_negative', None
    if rt is None or rt > row['rt_min'] + 1e-8:
        return 'missing_or_out_of_order_parent_rt', None
    if abs(rt - feature['rt_min']) > RT_MINUTES:
        return 'parent_rt_outside_same_feature_window', None
    if isinstance(decoded, str):
        return decoded, None
    mz, intensity, representation, encodings = decoded
    lower, upper = row['isolation_bounds_mz']
    signal = []
    for i, (mass, abundance) in enumerate(zip(mz, intensity, strict=True)):
        if not lower <= mass <= upper or abundance <= 0:
            continue
        if representation == 'profile_local_apices' and not (0 < i < len(mz) - 1 and intensity[i-1] < abundance >= intensity[i+1]):
            continue
        signal.append((mass, abundance))
    targets = [(mass, abundance) for mass, abundance in signal if within_mz(mass, feature['mz'])]
    if not targets:
        return 'no_target_peak_in_native_parent_isolation_window', None
    mass, abundance = max(targets, key=lambda peak: (peak[1], -abs(peak[0] - feature['mz'])))
    other = [(m, a) for m, a in signal if not within_mz(m, feature['mz'])]
    strongest = max(other, key=lambda peak: peak[1]) if other else None
    return 'supported_native_parent_peak', dict(parent_rt_min=rt, parent_representation=representation,
                                               parent_array_encodings=encodings, observed_mz=mass,
                                               mass_error_ppm=(mass-feature['mz']) / feature['mz'] * 1e6,
                                               observed_intensity=abundance, target_peak_count=len(targets),
                                               isolation_peak_count=len(signal), strongest_other_peak=strongest,
                                               target_peak_fraction_of_isolation_apex_signal=abundance / fsum(a for _, a in signal))


def verify_parents(path, matches, counts):
    wanted = {}
    for feature_id, rows in matches.items():
        for row in rows:
            wanted.setdefault(row['parent_native_scan_id'], []).append((feature_id, row))
    seen = set()
    for scan, groups in spectra(path):
        parent_id = scan.get('id')
        if parent_id not in wanted:
            continue
        if parent_id in seen:
            raise ValueError('A selected MS1 parent native ID is duplicated')
        seen.add(parent_id)
        fields = cv(scan, groups)
        level, mode = numeric(fields, 'MS:1000511'), polarity(fields)
        rt = minutes(scan, groups)
        decoded = 'missing_ms1_parent_arrays'
        if level == 1 and mode == 'negative':
            try:
                decoded = raw_arrays(scan, groups)
            except ValueError as error:
                decoded = str(error)
        for feature_id, row in wanted[parent_id]:
            status, proof = ('resolved_parent_level_is_not_ms1', None) if level != 1 else parent_evidence(row, FEATURES[feature_id], mode, rt, decoded)
            counts[feature_id][status] += 1
            row.update(parent_status=status, native_parent_peak=proof)
    for parent_id in wanted.keys() - seen:
        for feature_id, row in wanted[parent_id]:
            counts[feature_id]['unresolved_local_parent_reference'] += 1
            row.update(parent_status='unresolved_local_parent_reference', native_parent_peak=None)


def run(output):
    with TemporaryDirectory(prefix='paulownia-author-feature-') as directory:
        path = Path(directory) / Path(FILE).name
        raw = download(path, FILE, FILE_BYTES)
        if raw['sha256'] != FILE_SHA256:
            raise ValueError('Pinned MassIVE DDA file SHA-256 does not match the prior completed receipt')
        matches, counts, first_pass = match_ms2(path)
        if first_pass['negative_ms2'] != 4449:
            raise ValueError('Pinned DDA negative MS2 count differs from the prior official metadata audit')
        verify_parents(path, matches, counts)
    report = dict(dataset=DATASET, dataset_doi=DATASET_DOI, license='CC0 1.0', official_file_table=DATASET_FILES,
                  raw_file=raw, author_article=ARTICLE, author_supplement=SUPPLEMENT,
                  sample_column=REFERENCE, mass_window_ppm=MASS_PPM, rt_window_min=RT_MINUTES,
                  author_target_evidence='S3 Table S2 A1024, G1024, H1024 and DT1024; S2 Figures S3 and S9 connect fraction7 feature 8830 to the isolated compound.',
                  author_control_evidence='S3 Table S2 A966, G966, H966 and DT966; the article discusses the same-mass different-RT isomer in section 3.3.',
                  confirmation=dict(relative_path=CONFIRMATION, verified_ms1=2790, verified_ms2=0,
                                    explicit_compound_to_raw_basename_mapping=False,
                                    interpretation='The article and Figure S9 link the isolated compound to feature 8830; the confirmation mzML basename has no explicit author sample-to-structure mapping.'),
                  first_pass=dict(first_pass), profiles_use_recorded_local_apices=True,
                  distinct_target_candidates_at_most=1, independent_structures_admitted=0,
                  interpretation='This is a candidate evidence audit. The target and control are separated by author-assigned retention time. Parent-isolation signal fractions are descriptive and cannot resolve coeluting exact isobars. No compound identities, fragment peak lists or evaluation labels are exported.',
                  features={key: {**feature, 'counts': dict(counts[key]), 'selected_hcd_scans': matches[key]}
                            for key, feature in FEATURES.items()})
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    print('FLAVONOID_AUTHOR_FEATURE_PROBE',
          {key: dict(counts[key]) for key in FEATURES}, 'independent_structures_admitted=0', flush=True)

"""Audit public MSnLib native fragmentation trees before allocating training time."""

from __future__ import annotations

from argparse import ArgumentParser
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import md5, sha256
from itertools import combinations
from math import isfinite
from pathlib import Path
import json
import re
import tempfile
from urllib.parse import quote
from urllib.request import Request, urlopen
from xml.etree.ElementTree import iterparse
from zipfile import ZipFile


RELEASE = 21105617
RAW_RELEASE = 15683784
FILES = {
    'targetmol_pos': ('20250828_targetmol_hts_np_pos_msn.mgf', 108944735, '954ff7dba644b12954724bc3d7be029d'),
    'targetmol_neg': ('20250828_targetmol_hts_np_neg_msn.mgf', 87793533, '2891f986185c202756de454f3d1d4cb4'),
    'nih_pos': ('20241003_nihnp_pos_msn.mgf', 215152005, '225e72c9a70489a279762f20448cbfc2'),
    'nih_neg': ('20241003_nihnp_neg_msn.mgf', 146191719, '5efe6a20725fa6f8d5940e73e5a8c05a'),
}
RAW_TARGETMOL_NEG = ('mzml_20241120_pluskal_targetmolnphts_MSn_negative.zip', 175235275,
                     '5d185be5694f481ffd06da3f03d1d737')
NATIVE = {'SINGLE_SCAN', 'SINGLE_BEST_SCAN', 'SAME_ENERGY'}
ADDUCTS = {'[M+H]+', '[M-H]-'}
SCANS = re.compile(r'\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*')
NATIVE_ID = re.compile(r'(?:^|\s)scan=(\d+)(?:\s|$)')


def download(record, asset, scratch):
    name, size, checksum = asset
    path = scratch / name
    link = f'https://zenodo.org/api/records/{record}/files/{quote(name)}/content'
    for attempt in range(2):
        digest_shas = (md5(usedforsecurity=False), sha256())
        try:
            request = Request(link, headers={'User-Agent': 'MSnLib-data-preflight/1.0'})
            with urlopen(request, timeout=90) as source, path.open('wb') as out:
                while block := source.read(2 ** 20):
                    out.write(block)
                    for digest in digest_shas:
                        digest.update(block)
            if path.stat().st_size == size and digest_shas[0].hexdigest() == checksum:
                return path, dict(url=link, bytes=size, md5=checksum, sha256=digest_shas[1].hexdigest())
        except OSError:
            if attempt:
                raise
    raise ValueError(f'Public archive checksum mismatch for {name}')


def records(path):
    header, peaks = {}, []
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            line = line.strip()
            if line == 'BEGIN IONS':
                header, peaks = {}, []
            elif line == 'END IONS':
                yield header, tuple(sorted(peaks))
            elif '=' in line:
                key, value = line.split('=', 1)
                header[key.upper()] = value.strip()
            elif line and line[0].isdigit() and len(fields := line.split()) >= 2:
                mass, intensity = map(float, fields[:2])
                if isfinite(mass) and isfinite(intensity) and mass > 0 and intensity > 0:
                    peaks.append(mass)


def values(text):
    if not text:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else [value]


def scan_sources(header):
    sources = {}
    usis = values(header.get('SOURCE_SCAN_USI'))
    for usi in usis:
        if not isinstance(usi, str):
            continue
        prefix, _, selectors = usi.rpartition(':')
        kind, dataset, filename, *_ = (prefix.split(':') + ['', '', ''])
        if kind == 'mzspec' and dataset and filename and SCANS.fullmatch(selectors):
            key = f'{dataset}:{filename.removesuffix(".mzML")}'
            sources[key] = selectors
    return sources


@dataclass
class Entry:
    label: str
    identity: str
    formula: str
    smiles: str
    adduct: str
    instrument: str
    spectrum: str
    root_mass: float
    child_mass: float | None
    root_ce: float
    ce: float
    peaks: tuple
    files: dict
    raw_purity: float | None


def observed_match(peaks, desired):
    tolerance = max(0.002, desired * 10e-6)
    n = bisect_left(peaks, desired - tolerance)
    return n < len(peaks) and peaks[n] <= desired + tolerance


def inventory(path):
    parents, children = defaultdict(list), defaultdict(list)
    counts, record_types, fields, energies = Counter(), Counter(), Counter(), Counter()
    coverage = defaultdict(set)
    for header, peaks in records(path):
        level = header.get('MSLEVEL', 'unknown')
        mode = header.get('SPECTYPE', 'SINGLE_BEST_SCAN')
        record_types[f'MS{level}/{mode}'] += 1
        fields.update(header.keys())
        if level not in ('2', '3') or mode not in NATIVE:
            continue
        adduct = header.get('ADDUCT')
        if adduct not in ADDUCTS:
            continue
        identity = header.get('INCHIAUX', '').split('-')[0]
        feature = header.get('FEATURELIST_FEATURE_ID') or header.get('FEATURE_ID')
        ce = values(header.get('COLLISION_ENERGY'))
        if len(identity) != 14 or not feature or len(ce) != 1 or not isinstance(ce[0], (float, int)):
            counts['native_missing_identity_or_feature_or_single_ce'] += 1
            continue
        if header.get('QUALITY_CHIMERIC', '').casefold() not in ('', 'passed'):
            counts['export_chimeric_flag'] += 1
            continue
        try:
            mass = float(header['PEPMASS'].split()[0])
            purity = float(header['PRECURSOR_PURITY']) if 'PRECURSOR_PURITY' in header else None
        except (KeyError, ValueError):
            counts['invalid_precursor_or_purity'] += 1
            continue
        if purity is not None and (not isfinite(purity) or purity < .90):
            counts['export_purity_below_0p90'] += 1
            continue
        if not isfinite(mass) or mass <= 0 or not isfinite(ce[0]):
            counts['invalid_precursor_or_ce'] += 1
            continue
        instrument = header.get('SOURCE_INSTRUMENT', header.get('INSTRUMENT_TYPE', ''))
        child_mass, root_ce = None, ce[0]
        if level == '3':
            chain = values(header.get('MSN_PRECURSOR_MZS'))
            channels = values(header.get('MSN_COLLISION_ENERGIES'))
            if len(channels) == 2 and all(isinstance(value, (float, int)) for value in channels):
                channels = [channels]
            if (len(chain) != 2 or any(not isinstance(x, (float, int)) or not isfinite(x) for x in chain)
                    or len(channels) != 1 or not isinstance(channels[0], list) or len(channels[0]) != 2
                    or not isinstance(channels[0][0], (float, int)) or not isfinite(channels[0][0])
                    or channels[0][1] != ce[0] or abs(chain[0] - mass) > max(.002, mass * 10e-6)
                    or not 0 < chain[1] < chain[0] - 0.01):
                counts['ms3_has_no_unambiguous_mass_ce_chain'] += 1
                continue
            child_mass, root_ce = chain[1], channels[0][0]
        limit = child_mass if child_mass is not None else mass
        peaks = tuple(peak for peak in peaks if 25 <= peak < limit - 0.01)
        if len(peaks) < (4 if level == '2' else 3):
            counts[f'ms{level}_too_few_peaks'] += 1
            continue
        files = scan_sources(header)
        counts[f'usable_ms{level}_rows'] += 1
        counts[f'usable_ms{level}_with_raw_scan_map'] += bool(files)
        energies[f'MS{level}/{adduct}/{instrument}/root={root_ce}/stage={ce[0]}'] += 1
        formula = header.get('FORMULA', '')
        entry = Entry(header.get('NAME', ''), identity, formula, header.get('SMILES', ''), adduct,
                      instrument, header.get('USI', ''),
                      mass, child_mass, root_ce, ce[0], peaks, files, purity)
        key = (identity, adduct, feature, instrument, root_ce)
        (parents if level == '2' else children)[key].append(entry)
        if level == '2' and formula and instrument and files:
            coverage[(identity, formula, adduct, instrument, root_ce)].update(files)
    return parents, children, counts, record_types, fields, energies, coverage


def scan_numbers(numbers):
    found = set()
    for selector in numbers.split(','):
        first, _, last = selector.partition('-')
        start, end = int(first), int(last or first)
        if 0 <= end - start < 500:
            found.update(range(start, end + 1))
    return found


def matching_parents(roots, child):
    return [root for root in roots
            if abs(root.root_mass - child.root_mass) <= max(.002, child.root_mass * 10e-6)
            and observed_match(root.peaks, child.child_mass)]


def links(path):
    parents, children, counts, types, fields, energies, coverage = inventory(path)
    examples, export_only_examples, raw_checks = [], [], []
    identities = defaultdict(set)
    unique = set()
    for key, records_for_key in children.items():
        for child in records_for_key:
            roots = parents.get(key, ())
            if not roots:
                counts['ms3_no_native_same_tree_same_ce_root'] += 1
                continue
            matched = matching_parents(roots, child)
            if not matched:
                counts['ms3_has_root_without_matching_isolated_product'] += 1
                continue
            counts['ms3_native_root_and_selected_observed_peak'] += 1
            identities[child.adduct].add(child.identity)
            unique.add((key, round(child.child_mass, 4)))
            shared = sorted(((name, root) for root in matched for name in root.files.keys() & child.files.keys()),
                            key=lambda entry: (entry[0], abs(entry[1].root_mass - child.root_mass), entry[1].spectrum))
            if not shared:
                if len(export_only_examples) < 12 and child.identity not in {
                        value['identity'] for value in export_only_examples}:
                    export_only_examples.append(dict(name=child.label, identity=child.identity,
                                                     adduct=child.adduct, parent_export_usi=matched[0].spectrum,
                                                     child_export_usi=child.spectrum, root_ce_raw=child.root_ce,
                                                     child_ce_raw=child.ce, selected_product_mz=child.child_mass))
                continue
            counts['ms3_selected_peak_and_shared_raw_file'] += 1
            raw_name, root = shared[0]
            if root.raw_purity is not None and child.raw_purity is not None:
                counts['same_file_link_with_both_exported_purities'] += 1
            row = dict(name=child.label, identity=child.identity, adduct=child.adduct, formula=child.formula,
                       raw_file=raw_name, parent_export_usi=root.spectrum, child_export_usi=child.spectrum,
                       parent_scans=root.files[raw_name], child_scans=child.files[raw_name],
                       root_ce_raw=child.root_ce, child_ce_raw=child.ce, root_mz=child.root_mass,
                       selected_product_mz=child.child_mass)
            if len(examples) < 12 and child.identity not in {value['identity'] for value in examples}:
                examples.append(row)
            if len(raw_checks) < 36 and sum(item['raw_file'] == raw_name for item in raw_checks) < 3:
                raw_checks.append(row)
    for (_, formula, adduct, instrument, energy), files in coverage.items():
        counts['native_ms2_identity_conditions_with_raw_file'] += 1
        counts['native_ms2_identity_conditions_with_two_raw_files'] += len(files) >= 2
    pairs = defaultdict(list)
    for (identity, formula, adduct, instrument, energy), files in coverage.items():
        pairs[(formula, adduct, instrument, energy)].append((identity, files))
    group_counts = dict(conditions_with_same_formula_isomers=sum(len(value) >= 2 for value in pairs.values()),
                        conditions_with_same_formula_isomers_both_multiple_files=sum(
                            sum(len(files) >= 2 for _, files in value) >= 2 for value in pairs.values()),
                        conditions_with_two_shared_files_for_an_isomer_pair=sum(
                            any(len(a_files & b_files) >= 2 for (_, a_files), (_, b_files) in combinations(value, 2))
                            for value in pairs.values()))
    result = dict(counts=counts, types=types, source_fields=fields, same_formula_condition_counts=group_counts,
                  native_energy_channels=energies,
                  unique_tree_products=len(unique), linked_raw_author_identities={k: len(v) for k, v in identities.items()},
                  same_file_examples=examples, export_only_examples=export_only_examples)
    return result, coverage, raw_checks


def verify_raw(path, links_to_check):
    checks, counts = [], Counter()
    with ZipFile(path) as archive:
        manifest = {Path(name).name.removesuffix('.mzML'): name for name in archive.namelist()
                    if name.endswith('.mzML')}
        by_file = defaultdict(list)
        for link in links_to_check:
            by_file[link['raw_file'].split(':', 1)[1]].append(link)
        for raw_name, links_in_file in list(by_file.items())[:6]:
            if raw_name not in manifest:
                counts['raw_filename_missing_from_release'] += 1
                continue
            child_selectors = set().union(*(scan_numbers(row['child_scans']) for row in links_in_file))
            parent_selectors = set().union(*(scan_numbers(row['parent_scans']) for row in links_in_file))
            selectors = child_selectors | parent_selectors
            observed, levels = {}, {}
            with archive.open(manifest[raw_name]) as stream:
                for _, node in iterparse(stream, events=('end',)):
                    if node.tag.rsplit('}', 1)[-1] != 'spectrum':
                        continue
                    scan = NATIVE_ID.search(node.attrib.get('id', ''))
                    if scan and (number := int(scan[1])) in selectors:
                        levels[number] = next((entry.attrib.get('value') for entry in node
                                               if entry.attrib.get('accession') == 'MS:1000511'), None)
                        refs = [entry.attrib['spectrumRef'] for entry in node.iter()
                                if entry.tag.rsplit('}', 1)[-1] == 'precursor' and 'spectrumRef' in entry.attrib]
                        observed[number] = [int(value[1]) for ref in refs if (value := NATIVE_ID.search(ref))]
                    node.clear()
                    if selectors <= observed.keys():
                        break
            for link in links_in_file:
                expected_parents = scan_numbers(link['parent_scans'])
                observed_children = {scan: observed[scan] for scan in sorted(scan_numbers(link['child_scans']))
                                     if levels.get(scan) == '3'}
                direct = {scan: verified for scan, parents in observed_children.items()
                          if (verified := [parent for parent in parents if parent in expected_parents
                                           and levels.get(parent) == '2'])}
                checks.append(dict(**link, observed_child_scan_refs=observed_children, direct_parent_refs=direct))
                counts['candidate_links_checked'] += 1
                counts['candidate_links_with_observed_child'] += bool(observed_children)
                counts['candidate_links_with_precursor_ref'] += any(observed_children.values())
                counts['candidate_links_directly_verified'] += bool(direct)
    return dict(counts=counts, examples=checks)


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(author_release=RELEASE, raw_release=RAW_RELEASE, release_license='CC BY 4.0',
                  strict_export_pool='native MS2/MS3 at individual reported energies for [M+H]+ and [M-H]-',
                  scope='Raw InChIKey14 and same-formula counts precede scorer normalization and benchmark exclusions.',
                  assets={})
    all_coverage = defaultdict(set)
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        for name in args.source:
            path, provenance = download(RELEASE, FILES[name], scratch)
            results, coverage, checks = links(path)
            report['assets'][name] = dict(provenance=provenance, **results)
            for key, files in coverage.items():
                all_coverage[key].update(files)
            path.unlink()
            if name == 'targetmol_neg' and args.verify_raw and checks:
                raw_path, raw_provenance = download(RAW_RELEASE, RAW_TARGETMOL_NEG, scratch)
                report['raw_direct_reference_audit'] = dict(provenance=raw_provenance, **verify_raw(raw_path, checks))
                raw_path.unlink()
            print(name, dict(results['counts']), flush=True)
    groups = defaultdict(set)
    for (identity, formula, adduct, instrument, ce), raw_files in all_coverage.items():
        if len(raw_files) >= 2:
            groups[(formula, adduct, instrument, ce)].add(identity)
    report['combined_multiple_file_isomer_conditions'] = sum(len(keys) >= 2 for keys in groups.values())
    report['combined_multiple_file_isomer_identities'] = len({identity for keys in groups.values()
                                                                if len(keys) >= 2 for identity in keys})
    report['unambiguous_direct_parent_links_require_raw_precursor_reference'] = True
    (args.out / 'msnlib_preflight.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--source', nargs='+', choices=tuple(FILES), default=['targetmol_pos', 'targetmol_neg'])
    parser.add_argument('--verify-raw', action='store_true')
    parser.add_argument('--out', required=True, type=Path)
    run(parser.parse_args())

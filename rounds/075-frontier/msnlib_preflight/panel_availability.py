"""Count independent native TargetMol MS2 queries without emitting structures or spectra."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import file_digest, sha256
import json
from math import isfinite
from pathlib import Path
import re
import tempfile

from audit import download, records, scan_sources, values


RELEASE = 21105617
LICENSE = 'CC BY 4.0'
FILES = {
    'positive': ('20250828_targetmol_hts_np_pos_ms2.mgf', 17756464, '15ef1e68450e1ffb1f96fb4a2f58ad49'),
    'negative': ('20250828_targetmol_hts_np_neg_ms2.mgf', 16552003, '1ec1b270f154da5e94bd66d91e8b1799'),
}
INDEX_SHA = '7c16c4968a15956e94eba58e3ff94d4a6c2f6899b10a2f2824047601a46bd6c8'
PROTON = 1.007276466621
FEATURE_MZ = re.compile(r'(?:^|_)mz(\d+(?:\.\d+)?)(?=_|$)', re.I)
PLATE = re.compile(r'targetmol_(?:hts_np|np_hts)_([A-Z0-9-]+)_[A-Z]\d{1,2}_', re.I)
PEAK_ADDUCTS = {'positive': {'[M+H]+', '[M+H]1+'}, 'negative': {'[M-H]-', '[M-H]1-'}}
SCANSET = 'MSV000094528'
MIN_GROUPS = 40


def digest_text(value):
    return sha256(value.encode()).hexdigest()


def normalized_mol(smiles):
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return rdMolStandardize.TautomerEnumerator().Canonicalize(mol)


def groups(smiles):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = normalized_mol(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_key = Chem.MolToSmiles(scaffold)
    generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    return Chem.MolToInchiKey(mol)[:14], scaffold_key, Chem.MolToSmiles(generic)


def heavy_elements(mol):
    return tuple(sorted(Counter(atom.GetSymbol() for atom in mol.GetAtoms()
                                if atom.GetAtomicNum() > 1).items()))


def formula_elements(formula):
    return tuple(sorted((atom, int(count or 1)) for atom, count in re.findall(r'([A-Z][a-z]?)(\d*)', formula)
                        if atom != 'H'))


@dataclass(frozen=True)
class Identity:
    raw: str
    score: str
    heavy: tuple
    fine: str
    generic: str
    mass: float


def identify(smiles):
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if (mol is None or len(Chem.GetMolFrags(mol)) != 1 or mol.GetNumHeavyAtoms() < 5
            or Chem.GetFormalCharge(mol) != 0):
        return None
    score, fine, generic = groups(smiles)
    raw = Chem.MolToInchiKey(mol)[:14]
    if len(score) != 14 or len(raw) != 14:
        return None
    return Identity(raw, score, heavy_elements(mol), fine, generic, Descriptors.ExactMolWt(mol))


def number(text):
    try:
        result = float(text)
    except (ValueError, TypeError):
        return None
    return result if isfinite(result) else None


def major_fields(header, peaks, polarity, info):
    if header.get('MSLEVEL') not in {'2', '2.0'} or header.get('SPECTYPE', '').upper() != 'SINGLE_SCAN':
        return 'not_native_single_ms2', None
    if header.get('MERGED_ACROSS_N_SAMPLES') != '1':
        return 'sample_merge_not_one', None
    sign = '+' if polarity == 'positive' else '-'
    charge = header.get('CHARGE', '').replace(' ', '')
    adduct = header.get('ADDUCT', '').replace(' ', '')
    if (adduct not in PEAK_ADDUCTS[polarity] or header.get('IONMODE', '').lower() != polarity
            or charge not in {'1', f'1{sign}', f'{sign}1'}):
        return 'adduct_polarity_charge_inconsistent', None
    sources = scan_sources(header)
    if len(values(header.get('SOURCE_SCAN_USI'))) != 1 or len(sources) != 1:
        return 'no_single_original_scan', None
    acquisition, scan = next(iter(sources.items()))
    raw_name = header.get('FILENAME', '').removesuffix('.mzML')
    if (acquisition != f'{SCANSET}:{raw_name}' or header.get('DATASET_ID') != SCANSET
            or not scan.isdigit() or scan != header.get('SCANS')):
        return 'original_scan_metadata_conflict', None
    if (header.get('INSTRUMENT_TYPE', '').casefold() != 'orbitrap'
            or header.get('SOURCE_INSTRUMENT', '').casefold() != 'orbitrap id-x'
            or header.get('ION_SOURCE', '').upper() != 'ESI'
            or header.get('FRAGMENTATION_METHOD', '').upper() != 'HCD'):
        return 'instrument_method_outside_target', None
    if header.get('ACQUISITION', '').casefold() != 'commercial':
        return 'missing_commercial_standard_provenance', None
    if info is None:
        return 'invalid_or_disconnected_neutral_structure', None
    published = header.get('INCHIAUX', '')[:14]
    if published not in {info.raw, info.score}:
        return 'structure_and_author_identity_conflict', None
    precursor = number((header.get('PEPMASS') or 'nan').split()[0])
    centroid_match = FEATURE_MZ.search(header.get('FEATURE_FULL_ID', ''))
    observed = number(centroid_match[1]) if centroid_match else None
    if precursor is None or observed is None or min(precursor, observed) <= 0:
        return 'missing_measured_centroid_or_isolation_mz', None
    shift = PROTON if polarity == 'positive' else -PROTON
    centroid_mass, isolation_mass = observed - shift, precursor - shift
    centroid_ppm = (centroid_mass - info.mass) / centroid_mass * 1e6
    isolation_ppm = (isolation_mass - info.mass) / isolation_mass * 1e6
    if abs(info.mass - centroid_mass) > centroid_mass * 1e-5:
        return 'observed_centroid_outside_10ppm', None
    if abs(info.mass - isolation_mass) > isolation_mass * 1e-5:
        return 'pepmass_outside_10ppm', None
    if sum(mz < precursor - 0.5 for mz in set(peaks)) < 4:
        return 'fewer_than_four_fragment_peaks', None
    energy = values(header.get('COLLISION_ENERGY'))
    ce = number(energy[0]) if len(energy) == 1 else None
    if ce is None or ce <= 0:
        return 'missing_single_collision_energy', None
    plate = PLATE.search(raw_name)
    plate_id = plate[1] if plate else None
    return None, (acquisition, scan, plate_id, ce, centroid_ppm, isolation_ppm)


def train_exposure(index, identities):
    import pyarrow.parquet as pq
    from rdkit import Chem

    with index.open('rb') as handle:
        if file_digest(handle, 'sha256').hexdigest() != INDEX_SHA:
            raise ValueError('Different competition train structure index')
    target_compositions = defaultdict(set)
    for info in identities.values():
        target_compositions[info.heavy].add(info.score)
    frame = pq.read_table(index, columns=['formula', 'smiles', 'inchikey14', 'scorer_key'])
    recorded = set(frame['inchikey14'].to_pylist()) | set(frame['scorer_key'].to_pylist())
    signature = {formula: formula_elements(formula) for formula in set(frame['formula'].to_pylist()) if formula}
    relevant = {formula: comp for formula, comp in signature.items() if comp in target_compositions}
    seen, examined = set(), set()
    unparseable = 0
    for batch in frame.to_batches(max_chunksize=32768):
        for formula, smiles in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            if formula not in relevant or not smiles or smiles in examined:
                continue
            examined.add(smiles)
            molecule = normalized_mol(smiles)
            if molecule is None:
                unparseable += 1
                continue
            same = target_compositions[relevant[formula]]
            key = Chem.MolToInchiKey(molecule)[:14]
            if key in same and heavy_elements(molecule) == relevant[formula]:
                seen.add((key, relevant[formula]))
    return recorded, seen, {'rows': frame.num_rows, 'same_heavy_composition_smiles_recomputed': len(examined),
                            'invalid_recomputed_smiles': unparseable, 'sha256': INDEX_SHA}


def bins(values):
    values = [abs(value) for value in values]
    return {f'abs_le_{bound}_ppm': sum(value <= bound for value in values) for bound in (0.1, 1, 3, 5, 10)}


def summarize(pool, identities):
    scoped = [identities[key] for key in pool]
    fine = {info.fine for info in scoped if info.fine}
    group_sizes = Counter(info.generic for info in scoped if info.generic)
    scans = [scan for blocks in pool.values() for scan in blocks]
    return {
        'identities': len(scoped), 'single_scans': len(scans), 'fine_murcko_groups': len(fine),
        'generic_murcko_groups': len(group_sizes), 'acyclic_identities': sum(not info.generic for info in scoped),
        'largest_generic_group_size': max(group_sizes.values(), default=0),
        'acquisition_files': len({scan[0] for scan in scans}),
        'acquisition_plates': len({scan[2] for scan in scans if scan[2]}),
        'scans_without_parsed_plate': sum(scan[2] is None for scan in scans),
        'collision_energy_nce_counts': dict(Counter(scan[3] for scan in scans)),
        'observed_feature_error': bins(scan[4] for scan in scans),
        'exported_pepmass_error': bins(scan[5] for scan in scans),
    }


def run(index, exclusion, output):
    from rdkit import RDLogger, rdBase

    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError(f'Expected competition RDKit 2026.03.3, found {rdBase.rdkitVersion}')
    RDLogger.DisableLog('rdApp.*')
    identities, aliases, chemistry = {}, defaultdict(set), {}
    provenance, rejected, scanned, methods = {}, {}, {}, {}
    candidates = {polarity: defaultdict(list) for polarity in FILES}
    scans_seen = set()
    with tempfile.TemporaryDirectory(prefix='enveda-targetmol-ms2-') as directory:
        for polarity, asset in FILES.items():
            path, provenance[polarity] = download(RELEASE, asset, Path(directory))
            rejected[polarity] = Counter()
            scanned[polarity] = Counter()
            methods[polarity] = {name: Counter() for name in ('INSTRUMENT_TYPE', 'SOURCE_INSTRUMENT',
                                                           'ION_SOURCE', 'FRAGMENTATION_METHOD', 'ACQUISITION')}
            for header, peaks in records(path):
                scanned[polarity][header.get('SPECTYPE') or 'missing'] += 1
                if header.get('SPECTYPE', '').upper() != 'SINGLE_SCAN':
                    rejected[polarity]['not_native_single_ms2'] += 1
                    continue
                for field, counts in methods[polarity].items():
                    counts[header.get(field) or 'missing'] += 1
                smiles = header.get('SMILES')
                if smiles not in chemistry:
                    chemistry[smiles] = identify(smiles)
                info = chemistry[smiles]
                reason, fields = major_fields(header, peaks, polarity, info)
                if reason:
                    rejected[polarity][reason] += 1
                    continue
                if fields[:2] in scans_seen:
                    rejected[polarity]['duplicate_original_scan'] += 1
                    continue
                scans_seen.add(fields[:2])
                key = (info.score, info.heavy)
                identities[key] = info
                aliases[key].update((info.raw, info.score, header['INCHIAUX'][:14]))
                candidates[polarity][key].append(fields)
            print(f'{polarity}: {sum(scanned[polarity].values())} export blocks, '
                  f'{sum(map(len, candidates[polarity].values()))} valid native scans', flush=True)

    known, exact, train = train_exposure(index, identities)
    prior_identity, exclusions_fine, exclusions_generic = (set(exclusion[name]) for name in ('identities', 'fine', 'generic'))
    aggregate = {'holdout_scope': exclusion['sources'], 'source_arrays': provenance, 'train_index': train,
                 'rdkit': rdBase.rdkitVersion, 'author_release': RELEASE, 'license': LICENSE,
                 'acquisition': 'One TargetMol study, ESI HCD Orbitrap ID-X, native single scan',
                 'mass_evidence': 'Observed author MS1 feature centroid and exported PEPMASS, independently compared against physical monoisotopic mass; neither replaced with theoretical m/z.',
                 'summary': {}, 'minimum_combined_generic_murcko_groups': MIN_GROUPS}
    accepted = {}
    for polarity, pool in candidates.items():
        drops = Counter()
        kept = {}
        for key, blocks in pool.items():
            info = identities[key]
            keys = {digest_text(alias) for alias in aliases[key]}
            if key in exact:
                drops['same_composition_scorer_identity_in_train'] += 1
            elif aliases[key] & known:
                drops['train_index_alias'] += 1
            elif keys & prior_identity:
                drops['previous_panel_identity'] += 1
            elif (info.fine and digest_text(info.fine) in exclusions_fine) or (
                    info.generic and digest_text(info.generic) in exclusions_generic):
                drops['previous_panel_scaffold'] += 1
            else:
                kept[key] = blocks
        accepted[polarity] = kept
        aggregate['summary'][polarity] = {
            'author_export_type_counts': scanned[polarity], 'single_scan_method_metadata': methods[polarity],
            'spectrum_exclusions': rejected[polarity],
            'valid_before_exposure': summarize(pool, identities), 'identity_exclusions': drops,
            'eligible': summarize(kept, identities),
        }
    positive, negative = accepted['positive'], accepted['negative']
    by_key = {key: positive.get(key, []) + negative.get(key, []) for key in positive.keys() | negative.keys()}
    aggregate['eligible_combined'] = summarize(by_key, identities)
    aggregate['eligible_combined']['measured_both_polarities_identities'] = len(positive.keys() & negative.keys())
    aggregate['eligible_combined']['has_both_polarities'] = bool(positive and negative)
    aggregate['sufficient_groups'] = aggregate['eligible_combined']['generic_murcko_groups'] >= MIN_GROUPS
    output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + '\n')
    print(f"Availability complete. Eligible combined {len(by_key)} identities and "
          f"{aggregate['eligible_combined']['generic_murcko_groups']} cyclic generic Murcko groups", flush=True)

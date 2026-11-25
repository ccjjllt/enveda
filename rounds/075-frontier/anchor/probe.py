"""Blind measured-anchor reachability and fragment-site scoring on a fixed candidate budget."""

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
from hashlib import file_digest, sha256
import csv
import io
import json
from pathlib import Path
import time
from zipfile import ZipFile

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rdkit import Chem, rdBase
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt

import four_reference_mask as mask
from retrieval_pilot_score import metric_identity


CAP = 16
FINAL_CORE = 8
PARENT_CAP = 64
ANCHOR_CAP = 3
PERMUTATIONS = 8
PROTON = 1.007276466621
ADDUCTS = {'[M+H]+': PROTON, '[M-H]-': -PROTON}
SOURCE_ORDER = ('fused_overflow', 'main', 'pubchem', 'engine2', 'prefusion')
EDIT = {
    'O-methyl': (14.015650064, Counter(C=1)),
    'O-acetyl': (42.010564684, Counter(C=2, O=1)),
    'O-sulfate': (79.95681486, Counter(S=1, O=3)),
    'O-hexosyl': (162.05282342, Counter(C=6, O=5)),
}
STRUCTURE_SHA = '7c16c4968a15956e94eba58e3ff94d4a6c2f6899b10a2f2824047601a46bd6c8'
BASELINE_SOURCE = 'pablodiegoo/enveda-casmi26-starter-notebook/29'
PEAKS = ('ms2_mzs', 'ms2_normalized_intensities')
TRAIN_METADATA = ('inchikey14', 'ingest_lib', 'adduct', 'precursor_mz')


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def blind_candidates(baseline, queries, mask_file):
    with ZipFile(baseline) as source:
        receipt = json.loads(source.read('pilot_arm_receipt.json'))
        if (receipt['arm'] != 'baseline' or receipt['source'] != BASELINE_SOURCE
                or receipt['queries_sha256'] != digest(queries)
                or receipt['candidate_slots'] != 25):
            raise ValueError('Expected the sealed Pablo v29 baseline for these blind spectra')
        raw = {name: source.read(name) for name in ('submission.csv', 'four_source_lists.json',
                                                   'four_reference_audit.json')}
        for name, data in raw.items():
            if sha256(data).hexdigest() != receipt['hashes'][name]:
                raise ValueError(f'Sealed input changed: {name}')
        if ('reference_groups_sha256' in receipt
                and receipt['reference_groups_sha256'] != digest(mask_file)):
            raise ValueError('Anchor mask differs from baseline')
    ledgers = json.loads(raw['four_source_lists.json'])
    result = {}
    for row in csv.DictReader(io.StringIO(raw['submission.csv'].decode())):
        mid = row['molecule_id']
        final = row['smiles'].split(';') if row['smiles'] else []
        neighbors = [list(ledgers[mid][name]) for name in SOURCE_ORDER]
        fixed, seen = [], set()

        def add(smiles, origin):
            identity = metric_identity(smiles)
            if len(fixed) < CAP and identity and identity not in seen:
                fixed.append(dict(smiles=smiles, origin=origin))
                seen.add(identity)

        for smiles in final[:FINAL_CORE]:
            add(smiles, 'final_core')
        for i in range(max(map(len, neighbors), default=0)):
            for name, branch in zip(SOURCE_ORDER, neighbors):
                if i < len(branch):
                    add(branch[i], name)
            if len(fixed) == CAP:
                break
        for smiles in final[FINAL_CORE:]:
            add(smiles, 'final_tail')
        result[mid] = fixed
    if len(result) != receipt['molecules'] or set(result) != set(ledgers):
        raise ValueError('Frozen candidates and query ledger disagree')
    return result


def normalized_peaks(mz, intensity, precursor):
    x, y = mask.peaks(mz, intensity, precursor)
    top = np.argsort(-y, kind='stable')[:50]
    order = top[np.argsort(x[top], kind='stable')]
    x, y = x[order], y[order]
    if len(y):
        y = y / np.linalg.norm(y)
    return x, y


def align(query, reference, delta):
    qmz, qi = query
    rmz, ri = reference
    offsets = qmz[:, None] - rmz[None, :]
    direct = np.abs(offsets) <= .02
    shifted = np.abs(offsets - delta) <= .02 if delta else np.zeros_like(direct)
    q, r = np.where(direct | shifted)
    weights = qi[q] * ri[r]
    used_q, used_r, pairs = set(), set(), []
    for i in np.argsort(-weights, kind='stable'):
        qidx, ridx = int(q[i]), int(r[i])
        if qidx not in used_q and ridx not in used_r:
            pairs.append((ridx, bool(shifted[qidx, ridx]), float(weights[i])))
            used_q.add(qidx)
            used_r.add(ridx)
    return sum(row[2] for row in pairs) if len(pairs) >= 3 else 0., pairs


@lru_cache(maxsize=10000)
def molecule(smiles):
    return Chem.MolFromSmiles(smiles)


def attach_sites(anchor, candidate, expected_elements):
    if anchor.GetNumAtoms() + sum(expected_elements.values()) != candidate.GetNumAtoms():
        return ()
    sites = set()
    for mapping in candidate.GetSubstructMatches(anchor, useChirality=False, maxMatches=32):
        known = set(mapping)
        added = set(range(candidate.GetNumAtoms())) - known
        if Counter(candidate.GetAtomWithIdx(i).GetSymbol() for i in added) != expected_elements:
            continue
        edges = [(i, atom.GetIdx()) for i in added for atom in candidate.GetAtomWithIdx(i).GetNeighbors()
                 if atom.GetIdx() in known]
        if len(edges) != 1:
            continue
        added_atom, parent_atom = edges[0]
        site = mapping.index(parent_atom)
        oxygen = anchor.GetAtomWithIdx(site)
        bond = candidate.GetBondBetweenAtoms(added_atom, parent_atom)
        if (oxygen.GetSymbol() == 'O' and oxygen.GetTotalNumHs() == 1
                and oxygen.GetFormalCharge() == 0 and bond.GetBondType() == Chem.BondType.SINGLE):
            sites.add(site)
    return tuple(sorted(sites))


def reachability(structures, candidates):
    fields = ['inchikey14', 'scorer_key', 'alias', 'smiles', 'scaffold', 'mono_mass', 'n_heavy']
    by_key = {}
    for row in pq.read_table(structures, columns=fields).to_pylist():
        key = row['inchikey14']
        if not key or not row['smiles'] or row['mono_mass'] is None or not np.isfinite(row['mono_mass']):
            continue
        previous = by_key.get(key)
        # Site indices are for exactly one molecular representation of each anchor.
        if previous is None or (bool(row['alias']), row['smiles']) < (bool(previous['alias']), previous['smiles']):
            by_key[key] = row
    rows = sorted(by_key.values(), key=lambda row: row['mono_mass'])
    masses = np.array([row['mono_mass'] for row in rows])
    linked, lookup = {}, {}
    for mid, current in candidates.items():
        choices = defaultdict(dict)
        for index, item in enumerate(current):
            mol = molecule(item['smiles'])
            if mol is None or len(Chem.GetMolFrags(mol)) != 1 or Chem.GetFormalCharge(mol) != 0:
                continue
            target = CalcExactMolWt(mol)
            for name, (delta, heavy) in EDIT.items():
                expected = target - delta
                tolerance = max(.004, target * 10e-6)
                lower, upper = np.searchsorted(masses, [expected - tolerance, expected + tolerance])
                for row in rows[lower:upper]:
                    key = row['inchikey14']
                    if (not key or not row['scorer_key'] or row['n_heavy'] != mol.GetNumHeavyAtoms() - sum(heavy.values())
                            or mask.digest(row['scorer_key']) in mask.KEYS
                            or mask.blocked_scaffold(row['scaffold']) or mask.excluded_smiles(row['smiles'])):
                        continue
                    anchor = molecule(row['smiles'])
                    sites = attach_sites(anchor, mol, heavy)
                    if sites:
                        lookup[key] = row['smiles']
                        choices[key][index] = (name, sites)
        ordered = sorted(choices, key=lambda key: (-len(choices[key]), key))
        linked[mid] = {key: choices[key] for key in ordered[:PARENT_CAP]}
    return linked, lookup


@lru_cache(maxsize=2048)
def fragment_ions(smiles, adduct):
    mol = molecule(smiles)
    if mol is None or mol.GetNumAtoms() > 96:
        return [], []
    ions, atoms = [], []
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        broken = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=False)
        original = []
        try:
            fragments = Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=True,
                                         fragsMolAtomMapping=original)
        except (ValueError, RuntimeError):
            continue
        for fragment, indices in zip(fragments, original):
            if len(indices) >= 2 and Chem.GetFormalCharge(fragment) == 0:
                ions.append(CalcExactMolWt(fragment) + ADDUCTS[adduct])
                atoms.append(set(indices))
    return np.asarray(ions), atoms


def site_fit(site, annotations):
    weight, support, shifted, direct = 0., 0., 0, 0
    for is_shifted, strength, options in annotations:
        inside = sum(site in fragment for fragment in options) / len(options)
        support += strength * (inside if is_shifted else 1. - inside)
        weight += strength
        shifted += int(is_shifted)
        direct += int(not is_shifted)
    # Separate ordinary spectral score from conditional evidence for the site.
    if shifted == 0 or direct == 0 or weight == 0.:
        return None
    return float(2. * support / weight - 1.)


def site_scores(key, smiles, adduct, peaks, pairs, sites):
    ions, fragments = fragment_ions(smiles, adduct)
    if not len(ions):
        return None, []
    x, _ = peaks
    annotations = []
    for reference_idx, shifted, weight in pairs:
        mz = x[reference_idx]
        assigned = np.flatnonzero(np.abs(ions - mz) <= max(.01, mz * 20e-6))
        # Equivalent atom sets from different cuts count once.
        options = {tuple(sorted(fragments[i])) for i in assigned}
        if options:
            annotations.append((shifted, weight, options))
    original = [site_fit(site, annotations) for site in sites]
    if not original or original[0] is None:
        return None, []
    available = [a.GetIdx() for a in molecule(smiles).GetAtoms()
                 if a.GetSymbol() == 'O' and a.GetFormalCharge() == 0 and a.GetTotalNumHs() == 1]
    controls = []
    for i in range(PERMUTATIONS):
        seed = int.from_bytes(sha256(f'{key}/anchor-site-control/{i}'.encode()).digest()[:8], 'little')
        permuted = np.random.default_rng(seed).permutation(available)
        control_sites = {site: int(other) for site, other in zip(available, permuted)}
        controls.append(max(site_fit(control_sites[site], annotations) for site in sites))
    return max(original), controls


def rank_anchors(train, queries, linked, lookup):
    invited = defaultdict(list)
    for i, q in enumerate(queries):
        if q['adduct'] not in ADDUCTS or len(q['peaks'][0]) < 3:
            continue
        for key, options in linked[q['molecule_id']].items():
            invited[key, q['adduct']].append((i, options))
    if not invited:
        return {}, Counter(), {}
    wanted = pa.array(sorted({key for key, _ in invited}))
    parquet = pq.ParquetFile(train)
    anchors, counts, measured = defaultdict(dict), Counter(), defaultdict(set)
    for group in range(parquet.num_row_groups):
        metadata = parquet.read_row_group(group, columns=list(TRAIN_METADATA))
        indices = pc.indices_nonzero(pc.is_in(metadata['inchikey14'], value_set=wanted))
        selected = metadata.take(indices).to_pylist()
        remaining = []
        for idx, row in zip(indices.to_pylist(), selected):
            key, adduct = row['inchikey14'], row['adduct']
            counts['metadata_anchor_matches'] += 1
            if not row['ingest_lib'] or mask.blocked_source(row['ingest_lib']) or (key, adduct) not in invited:
                continue
            precursor = row['precursor_mz']
            if precursor is None or not np.isfinite(precursor):
                continue
            usable = []
            for i, options in invited[key, adduct]:
                qmz = queries[i]['precursor_mz']
                delta = qmz - precursor
                tol = max(.02, qmz * 25e-6)
                matches = {rank: (name, sites) for rank, (name, sites) in options.items()
                           if abs(delta - EDIT[name][0]) <= tol}
                if matches:
                    usable.append((i, matches, delta))
            if usable:
                remaining.append((idx, row, usable))
        if not remaining:
            continue
        wide = parquet.read_row_group(group, columns=['normalized_smiles', *PEAKS])
        fragments = wide.take(pa.array([r[0] for r in remaining], type=pa.int64())).to_pylist()
        for (_, row, usable), raw in zip(remaining, fragments):
            key, precursor, adduct = row['inchikey14'], row['precursor_mz'], row['adduct']
            if mask.excluded_smiles(raw['normalized_smiles']) or (
                    metric_identity(raw['normalized_smiles']) != metric_identity(lookup[key])):
                continue
            mz, intensity = raw[PEAKS[0]], raw[PEAKS[1]]
            if mz is None or intensity is None or len(mz) != len(intensity):
                continue
            if mask.copied_spectrum(precursor, mz, intensity):
                counts['excluded_spectral_copy'] += 1
                continue
            peaks = normalized_peaks(mz, intensity, precursor)
            if len(peaks[0]) < 3:
                continue
            counts['measured_rows_in_compatible_window'] += 1
            for i, options, delta in usable:
                measured[i].update(options)
                modified, pairs = align(queries[i]['peaks'], peaks, delta)
                counts['spectrum_anchor_pairs'] += 1
                if modified < .15 or modified <= anchors[i].get(key, {}).get('modified', 0.):
                    continue
                ordinary, _ = align(queries[i]['peaks'], peaks, 0.)
                anchors[i][key] = dict(modified=modified, ordinary=ordinary, pairs=pairs,
                                        peaks=peaks, options=options, source=row['ingest_lib'],
                                        precursor_mz=precursor)
    return anchors, counts, measured


def main(args):
    start = time.monotonic()
    if digest(args.structures) != STRUCTURE_SHA:
        raise ValueError('Training structure index is not the released panel-mask asset')
    candidates = blind_candidates(args.baseline, args.queries, args.mask)
    query_columns = ['molecule_id', 'spectrum_id', 'adduct', 'precursor_mz', *PEAKS]
    queries = pq.read_table(args.queries, columns=query_columns).to_pylist()
    if set(candidates) != {q['molecule_id'] for q in queries}:
        raise ValueError('Queries differ from the sealed candidate ledger')
    for row in queries:
        row['peaks'] = normalized_peaks(row[PEAKS[0]], row[PEAKS[1]], row['precursor_mz'])
    linked, lookup = reachability(args.structures, candidates)
    print('Finished structural reachability', len(lookup), round(time.monotonic() - start, 2), flush=True)
    anchors, counters, measured = rank_anchors(args.train, queries, linked, lookup)
    details = []
    reachability_queries, measured_queries, site_queries, discriminating_queries = set(), set(), set(), set()
    for i, query in enumerate(queries):
        mid, adduct = query['molecule_id'], query['adduct']
        best = sorted(anchors.get(i, {}).items(), key=lambda item: (-item[1]['modified'], item[0]))
        if linked[mid]:
            reachability_queries.add(mid)
        if measured.get(i):
            measured_queries.add(mid)
        for key, record in best[:ANCHOR_CAP]:
            lines = []
            for candidate, (edit, sites) in sorted(record['options'].items()):
                fit, controls = site_scores(key, lookup[key], adduct, record['peaks'], record['pairs'], sites)
                lines.append(dict(rank=candidate + 1, edit=edit, plausible_sites=len(sites),
                                  site_score=fit, shuffled_site_scores=controls))
            scores = {line['site_score'] for line in lines if line['site_score'] is not None}
            if scores:
                site_queries.add(mid)
            if len(scores) > 1:
                discriminating_queries.add(mid)
            details.append(dict(molecule_id=mid, spectrum_id=query['spectrum_id'], adduct=adduct,
                                anchor_key=key, anchor_source=record['source'],
                                ordinary_cosine=record['ordinary'], modified_cosine=record['modified'],
                                eligible_candidates=lines))
    report = dict(queries=len(candidates), spectra=len(queries), candidate_cap=CAP, final_core=FINAL_CORE,
                  parent_cap=PARENT_CAP, anchor_cap_per_spectrum=ANCHOR_CAP, permutations=PERMUTATIONS,
                  candidate_slots=sum(map(len, candidates.values())), candidate_map_reachable=len(reachability_queries),
                  same_adduct_measured_reachable=len(measured_queries),
                  molecules_with_selected_anchor=len({row['molecule_id'] for row in details}),
                  molecules_with_scored_site=len(site_queries),
                  molecules_with_discriminating_shared_anchor=len(discriminating_queries),
                  scanned=dict(counters), queries_sha256=digest(args.queries), mask_sha256=digest(args.mask),
                  structures_sha256=STRUCTURE_SHA, baseline_sha256=digest(args.baseline),
                  rdkit=rdBase.rdkitVersion, elapsed_seconds=round(time.monotonic() - start, 2))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'anchor_probe_pairs.json').write_text(json.dumps(details, indent=2, allow_nan=False)+'\n')
    (args.out / 'anchor_probe_candidates.json').write_text(json.dumps(candidates, indent=2)+'\n')
    report['evidence_sha256'] = digest(args.out / 'anchor_probe_pairs.json')
    report['candidates_sha256'] = digest(args.out / 'anchor_probe_candidates.json')
    (args.out / 'anchor_probe_receipt.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'queries', 'mask', 'structures', 'train', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    main(parser.parse_args())

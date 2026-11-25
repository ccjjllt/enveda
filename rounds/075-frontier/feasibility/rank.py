"""Blind candidate-relative fragment feasibility on a sealed Pablo candidate list."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import time
from zipfile import ZipFile

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt, CalcMolFormula

import frag_rescore as native
from retrieval_pilot_score import CAP, SOURCE, answers, digest, metric_identity


ADDUCT = {'[M+H]+': native.PROTON, '[M-H]-': -native.PROTON}
MAX_LEADERS = 8
MAX_FRAGMENTS = 128
MAX_PEAKS = 80
MAX_DONOR_GAP_DA = 5.0
TIGHT_DONOR_GAP_DA = 0.01
MAX_DONOR_ENERGY_GAP = 15.0
COLUMNS = ('molecule_id', 'spectrum_id', 'instrument_type', 'adduct', 'precursor_mz',
           'collision_energy_ev', 'ms2_mzs', 'ms2_normalized_intensities')
ARMS = ('baseline', 'absolute', 'relative', 'matched_real', 'matched_donor',
        'tight_real', 'tight_donor')


@dataclass
class Scan:
    mid: str
    sid: str
    instrument: str
    adduct: str
    precursor: float
    energy: float | None
    mz: np.ndarray
    intensity: np.ndarray

    def as_native(self):
        return dict(mz=self.mz, it=self.intensity, adduct=self.adduct,
                    prec=self.precursor, mode=1 if ADDUCT[self.adduct] > 0 else -1)


def read_scans(path):
    scans = {}
    for row in pd.read_parquet(path, columns=list(COLUMNS)).itertuples(index=False):
        if row.adduct not in ADDUCT or not np.isfinite(row.precursor_mz) or row.precursor_mz <= 0:
            continue
        mz = np.asarray(row.ms2_mzs, dtype=np.float64)
        intensity = np.asarray(row.ms2_normalized_intensities, dtype=np.float64)
        if mz.ndim != 1 or mz.shape != intensity.shape:
            continue
        take = (np.isfinite(mz) & np.isfinite(intensity) & (intensity > 0)
                & (mz >= 35.0) & (mz < row.precursor_mz - 0.5))
        mz, intensity = mz[take], intensity[take]
        if len(mz) < 4:
            continue
        indices = np.argsort(-intensity, kind='stable')[:MAX_PEAKS]
        energies = list(row.collision_energy_ev) if row.collision_energy_ev is not None else []
        energy = float(energies[0]) if len(energies) == 1 and np.isfinite(energies[0]) else None
        scans[row.spectrum_id] = Scan(row.molecule_id, row.spectrum_id,
                                     str(row.instrument_type or '').strip().casefold(),
                                     row.adduct, float(row.precursor_mz), energy,
                                     mz[indices], intensity[indices])
    return scans


def pair_donors(scans, max_gap):
    edges = []
    for i, left in enumerate(scans):
        for right in scans[i + 1:]:
            if (left.mid == right.mid or left.adduct != right.adduct or not left.instrument
                    or left.instrument != right.instrument):
                continue
            gap = abs(left.precursor - right.precursor)
            if gap > max_gap or (left.energy is None) != (right.energy is None):
                continue
            energy_gap = abs(left.energy - right.energy) if left.energy is not None else 0.0
            if energy_gap <= MAX_DONOR_ENERGY_GAP:
                edges.append((gap, energy_gap, left.sid, right.sid))
    pairs = {}
    for _, _, left, right in sorted(edges):
        if left not in pairs and right not in pairs:
            pairs[left], pairs[right] = right, left
    return pairs


def bounded_masses(smiles, cache):
    levels = native.candidate_levels(smiles, cache=cache)
    parts = []
    remaining = MAX_FRAGMENTS
    for masses in levels[1:]:
        if remaining and len(masses):
            # Limit proposals before observing any peaks. Take cheaper cleavages first.
            indices = np.linspace(0, len(masses) - 1, min(remaining, len(masses)), dtype=int)
            parts.append(masses[indices])
            remaining -= len(indices)
    return np.sort(np.concatenate(parts)) if parts else np.empty(0)


def neutral_formula(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if (molecule is None or len(Chem.GetMolFrags(molecule)) != 1
            or any(atom.GetFormalCharge() or atom.GetIsotope() for atom in molecule.GetAtoms())):
        return None
    return CalcMolFormula(molecule), CalcExactMolWt(molecule)


def feasibility(scan, peak_scan, proposed):
    masses = peak_scan.mz
    use = masses < scan.precursor - 0.5
    masses, raw_intensity = masses[use], peak_scan.intensity[use]
    shifts = ADDUCT[scan.adduct] + native.H * np.arange(-2, 4)
    neutral = masses[None, :] - shifts[:, None]
    tolerance = np.maximum(0.002, masses * 10e-6)
    match = []
    for fragments in proposed:
        lo = np.searchsorted(fragments, neutral - tolerance, 'left')
        hi = np.searchsorted(fragments, neutral + tolerance, 'right')
        match.append(np.any(hi > lo, axis=0))
    support = np.asarray(match, dtype=bool)
    weights = np.sqrt(raw_intensity)
    # The same normalization makes an unmatched or cropped donor peak count against coverage.
    weights /= np.sqrt(peak_scan.intensity).sum()
    return masses, weights, support


def compare_scans(scans, parent_mass, proposed, donors=None):
    absolute, relative, unique = [], [], defaultdict(list)
    for receiver, peaks in scans:
        expected = parent_mass + ADDUCT[receiver.adduct]
        if abs(expected - receiver.precursor) > max(0.005, expected * 15e-6):
            continue
        peak_scan = donors[receiver.sid] if donors is not None else peaks
        mz, weights, matched = feasibility(receiver, peak_scan, proposed)
        if len(mz) < 4:
            continue
        absolute.append(matched @ weights)
        rarity = (len(proposed) - matched.sum(axis=0)) / (len(proposed) - 1)
        relative.append(matched @ (weights * rarity))
        individual = matched.sum(axis=0) == 1
        for i, row in enumerate(matched):
            unique[i].extend(mz[row & individual])
    if not absolute:
        return None
    # A repeatedly observed ion is one distinguishing peak, regardless of scan count.
    counts = []
    for i in range(len(proposed)):
        merged = []
        for mz in sorted(unique[i]):
            if not merged or mz - merged[-1] > 0.02:
                merged.append(mz)
        counts.append(len(merged))
    return dict(absolute=np.mean(absolute, axis=0), relative=np.mean(relative, axis=0),
                exclusive=counts, scans=len(absolute))


def choose(indices, anchor, evidence, scheme):
    pos = indices.index(anchor)
    winners = []
    for i, old_rank in enumerate(indices):
        if not anchor < old_rank <= anchor + 2:
            continue
        absolute_margin = evidence['absolute'][i] - evidence['absolute'][pos]
        new_margin = evidence[scheme][i] - evidence[scheme][pos]
        if (evidence['absolute'][i] >= 0.10 and absolute_margin >= 0.0
                and new_margin >= 0.035 and evidence['exclusive'][i] >= 2):
            winners.append((-new_margin, old_rank))
    return min(winners)[1] if winners else None


def formula_windows(candidates, cache):
    leaders = candidates[:MAX_LEADERS]
    composition = [neutral_formula(smiles) for smiles in leaders]
    for anchor in (0, 4):
        if len(leaders) <= anchor or composition[anchor] is None:
            continue
        formula, mass = composition[anchor]
        positions = [i for i, item in enumerate(composition) if item is not None and item[0] == formula]
        if len(positions) < 3 or not any(anchor < i <= anchor + 2 for i in positions):
            continue
        group = [leaders[i] for i in positions]
        yield anchor, formula, mass, positions, group, [bounded_masses(smiles, cache) for smiles in group]


def protects_first(leader, sources):
    key = metric_identity(leader)
    tops = (sources[name][0] for name in ('main', 'engine2', 'pubchem') if sources[name])
    return sum(key == metric_identity(smiles) for smiles in tops) >= 3


def rank_absolute(baseline, source_lists, scans):
    """Apply the frozen absolute swap rule to complete original candidate lists."""
    by_id = defaultdict(list)
    for scan in scans.values():
        by_id[scan.mid].append((scan, scan))
    if set(source_lists) != set(baseline) or set(by_id) - set(baseline):
        raise ValueError('Absolute comparison inputs disagree on query identities')
    cache, outputs, swaps = {}, {}, []
    native.warmup()
    for mid, candidates in baseline.items():
        values = list(candidates)
        outputs[mid] = values
        if mid not in by_id:
            continue
        for anchor, _, mass, positions, _, proposed in formula_windows(candidates, cache):
            if anchor == 0 and protects_first(candidates[0], source_lists[mid]):
                continue
            evidence = compare_scans(by_id[mid], mass, proposed)
            winner = choose(positions, anchor, evidence, 'absolute') if evidence is not None else None
            if winner is not None:
                values[anchor], values[winner] = values[winner], values[anchor]
                swaps.append(dict(molecule_id=mid, anchor_rank=anchor + 1, chosen_rank=winner + 1))
    return outputs, swaps


def sealed_inputs(path, queries, out):
    with ZipFile(path) as files:
        receipt = json.loads(files.read('pilot_arm_receipt.json'))
        if (receipt['arm'] != 'baseline' or receipt['source'] != SOURCE
                or receipt['queries_sha256'] != digest(queries) or receipt['candidate_slots'] != CAP
                or receipt['metric_rdkit'] != rdBase.rdkitVersion):
            raise ValueError('Sealed Pablo baseline or metric version differs')
        for stage in ('npx', 'e180x'):
            if not receipt['stage_status'][stage].startswith(('applied', 'gate failed')):
                raise ValueError(f'Sealed baseline stage {stage} did not complete')
        names = ('submission.csv', 'four_source_lists.json', 'spectral_parallax_report.json')
        frozen = {name: files.read(name) for name in names}
        if any(sha256(raw).hexdigest() != receipt['hashes'][name] for name, raw in frozen.items()):
            raise ValueError('Baseline outputs differ from the sealed receipt')
    output = out / 'baseline.csv'
    output.write_bytes(frozen['submission.csv'])
    guesses = answers(output)
    vacant = {mid: CAP - len(values) for mid, values in guesses.items() if len(values) < CAP}
    if vacant != receipt['vacant_slots_by_query']:
        raise ValueError('Sealed baseline vacancy accounting differs')
    return receipt, guesses, json.loads(frozen['four_source_lists.json']), json.loads(frozen['spectral_parallax_report.json'])


def rank(args):
    start = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    receipt, baseline, source_lists, parallax = sealed_inputs(args.baseline, args.queries, args.out)
    scans = read_scans(args.queries)
    by_id = defaultdict(list)
    for scan in scans.values():
        by_id[scan.mid].append(scan)
    if set(by_id) != set(baseline) or set(source_lists) != set(baseline):
        raise ValueError('Panel query IDs differ from the sealed baseline')
    all_scans = sorted(scans.values(), key=lambda scan: scan.sid)
    tight = pair_donors(all_scans, TIGHT_DONOR_GAP_DA)
    near = {**tight, **pair_donors([s for s in all_scans if s.sid not in tight], MAX_DONOR_GAP_DA)}
    mappings = {'matched': near, 'tight': tight}
    outputs = {arm: {mid: list(values) for mid, values in baseline.items()} for arm in ARMS[1:]}
    evidence_rows, events = [], []
    native_cache = {}
    native.warmup()
    for mid, candidates in baseline.items():
        leaders = candidates[:MAX_LEADERS]
        real = by_id[mid]
        for anchor, formula, mass, indices, same_formula, proposals in formula_windows(candidates, native_cache):
            paired = [(scan, scan) for scan in real]
            evidence = compare_scans(paired, mass, proposals)
            if evidence is None:
                continue
            native_scores = native.score_molecule([s.as_native() for s in real], same_formula, cache=native_cache)
            data = {'absolute': (evidence, 'absolute'), 'relative': (evidence, 'relative')}
            for name, mapping in mappings.items():
                matched = [(receiver, donor) for receiver, donor in paired if receiver.sid in mapping
                           and np.count_nonzero(scans[mapping[receiver.sid]].mz < receiver.precursor - 0.5) >= 4]
                donor_scans = {receiver.sid: scans[mapping[receiver.sid]] for receiver, _ in matched}
                observed = compare_scans(matched, mass, proposals)
                donor = compare_scans(matched, mass, proposals, donor_scans)
                if observed is not None and donor is not None and observed['scans'] == donor['scans']:
                    data[f'{name}_real'] = observed, 'relative'
                    data[f'{name}_donor'] = donor, 'relative'
            protected = anchor == 0 and protects_first(leaders[0], source_lists[mid])
            for arm, (values, scheme) in data.items():
                winner = None if protected else choose(indices, anchor, values, scheme)
                events.append(dict(molecule_id=mid, anchor_rank=anchor + 1, formula=formula,
                                   arm=arm, protected=protected, scans=values['scans'],
                                   chosen_rank=winner + 1 if winner is not None else None))
                if winner is not None:
                    values_out = outputs[arm][mid]
                    values_out[anchor], values_out[winner] = values_out[winner], values_out[anchor]
                for pos, i in enumerate(indices):
                    evidence_rows.append(dict(molecule_id=mid, anchor_rank=anchor + 1, formula=formula,
                                              original_rank=i + 1, smiles=leaders[i], arm=arm,
                                              proposed_masses=len(proposals[pos]), spectrum_count=values['scans'],
                                              absolute=values['absolute'][pos], relative=values['relative'][pos],
                                              exclusive_peaks=values['exclusive'][pos], native_coverage=native_scores[pos]))
    for arm, lists in outputs.items():
        if any(Counter(values) != Counter(baseline[mid]) for mid, values in lists.items()):
            raise ValueError(f'Candidate population changed in {arm}')
        pd.DataFrame([(mid, ';'.join(values)) for mid, values in lists.items()],
                     columns=['molecule_id', 'smiles']).to_csv(args.out / f'{arm}.csv', index=False)
    pairs = [dict(recipient_scan_id=receiver.sid, recipient_id=receiver.mid, donor_scan_id=donor.sid,
                  donor_id=donor.mid, precursor_gap_da=abs(receiver.precursor - donor.precursor),
                  tight=receiver.sid in tight, same_known_energy=receiver.energy is not None)
             for sid, other in sorted(near.items()) for receiver, donor in [(scans[sid], scans[other])]]
    details = dict(pairing=pairs, events=events)
    (args.out / 'audit.json').write_text(json.dumps(details, indent=2, allow_nan=False) + '\n')
    pd.DataFrame(evidence_rows).to_csv(args.out / 'candidate_evidence.csv', index=False)
    report = dict(baseline_sha256=digest(args.baseline), blind_queries_sha256=digest(args.queries),
                  upstream_parallax=parallax, native_library_max_available=False,
                  molecule_count=len(baseline), deposited_spectra=receipt['spectra'],
                  eligible_spectra=len(scans), per_query_cap=CAP, vacant_slots=receipt['vacant_slots_by_query'],
                  max_leaders=MAX_LEADERS, fragment_masses_per_candidate=MAX_FRAGMENTS, peaks_per_scan=MAX_PEAKS,
                  donor_gaps_da=dict(tight=TIGHT_DONOR_GAP_DA, near=MAX_DONOR_GAP_DA),
                  matched_scan_counts={name: len(mapping) for name, mapping in mappings.items()},
                  evaluated_events=dict(Counter(row['arm'] for row in events)),
                  changed_queries={arm: sum(lists[mid] != baseline[mid] for mid in baseline)
                                   for arm, lists in outputs.items()},
                  elapsed_seconds=round(time.monotonic() - start, 2))
    report['hashes'] = {name: digest(args.out / name) for name in
                        ('audit.json', 'candidate_evidence.csv', *(f'{arm}.csv' for arm in ARMS))}
    (args.out / 'receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('elapsed_seconds', 'evaluated_events', 'changed_queries',
                                                    'matched_scan_counts')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'queries', 'out'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    rank(parser.parse_args())

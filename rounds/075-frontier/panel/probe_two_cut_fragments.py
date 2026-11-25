"""Measure whether two-cut graph fragments distinguish observed same-formula pairs."""

import argparse
from collections import Counter
from functools import lru_cache
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from time import monotonic

import numpy as np
from rdkit import Chem

import cleavage_rank
from fit_dual_energy_pairs import ADDUCTS, CHANNELS, ENERGIES, VIEWS, evaluate, observed
from read_dual_training_scans import load_pairs


ARMS = ('single', 'bridge', 'ring', 'two_cut')


def connected_component(start, adjacent, removed):
    mask = 1 << start
    pending = [start]
    while pending:
        atom = pending.pop()
        for other, edge in adjacent[atom]:
            bit = 1 << other
            if edge not in removed and not mask & bit:
                mask |= bit
                pending.append(other)
    return mask


def mass_of(mask, masses):
    total = 0.0
    while mask:
        bit = mask & -mask
        total += masses[bit.bit_length() - 1]
        mask ^= bit
    return total


def ion_hypotheses(masks, masses):
    return np.asarray(sorted({mass_of(mask, masses) + cleavage_rank.PROTON + offset * cleavage_rank.HYDROGEN
                              for mask in masks for offset in (-1, 0, 1)}))


@lru_cache(maxsize=5000)
def collect_hypotheses(smiles):
    single, _ = cleavage_rank.fragments(smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1 or any(atom.GetFormalCharge() for atom in mol.GetAtoms()):
        return (np.sort(single),) * len(ARMS), 0, 0
    bonds = [(edge.GetBeginAtomIdx(), edge.GetEndAtomIdx()) for edge in mol.GetBonds()]
    adjacent = [[] for _ in mol.GetAtoms()]
    for i, (a, b) in enumerate(bonds):
        adjacent[a].append((b, i))
        adjacent[b].append((a, i))
    whole = (1 << mol.GetNumAtoms()) - 1
    bridges = {edge.GetIdx(): connected_component(edge.GetBeginAtomIdx(), adjacent, {edge.GetIdx()})
               for edge in mol.GetBonds() if not edge.IsInRing() and edge.GetBondType() == Chem.BondType.SINGLE}
    bridge_fragments = set()
    for i, j in combinations(bridges, 2):
        a = bridges[i] if not bridges[i] & (1 << bonds[j][0]) else whole ^ bridges[i]
        b = bridges[j] if not bridges[j] & (1 << bonds[i][0]) else whole ^ bridges[j]
        middle = whole ^ (a | b)
        if middle:
            bridge_fragments.add(middle)
    ring_pairs = {tuple(sorted(pair)) for ring in mol.GetRingInfo().BondRings() for pair in combinations(ring, 2)}
    ring_fragments = set()
    for i, j in sorted(ring_pairs):
        a = connected_component(bonds[i][0], adjacent, {i, j})
        if a != whole:
            ring_fragments.update((a, whole ^ a))

    table = Chem.GetPeriodicTable()
    atom_masses = [(table.GetMassForIsotope(a.GetAtomicNum(), a.GetIsotope())
                    if a.GetIsotope() else table.GetMostCommonIsotopeMass(a.GetAtomicNum()))
                   + a.GetTotalNumHs() * cleavage_rank.HYDROGEN for a in mol.GetAtoms()]
    by_bridge = ion_hypotheses(bridge_fragments, atom_masses)
    by_ring = ion_hypotheses(ring_fragments, atom_masses)
    both = np.unique(np.concatenate((single, by_bridge, by_ring)))
    return (np.unique(single), np.union1d(single, by_bridge), np.union1d(single, by_ring), both), len(bridge_fragments), len(ring_fragments)


def support(query, hypotheses, positive):
    mz, weights = query
    adjusted = mz if positive else mz + 2 * cleavage_rank.PROTON
    tolerance = np.maximum(0.003, mz * 10e-6)
    low = np.searchsorted(hypotheses, adjusted - tolerance, side='left')
    high = np.searchsorted(hypotheses, adjusted + tolerance, side='right')
    return float(weights @ (high > low))


def spectral_views(pair, scans):
    candidates = (pair['left'], pair['right'])
    left, right = [collect_hypotheses(candidate['smiles']) for candidate in candidates]
    margins = np.zeros((len(CHANNELS), 2, len(ARMS)), dtype=float)
    present = np.zeros(len(CHANNELS), dtype=bool)
    for channel, condition in enumerate(CHANNELS):
        adduct, energy = condition.split('/')
        if int(energy.removesuffix('eV')) not in pair['matched_dual_energies']:
            continue
        queries = [observed(scans[side['train_row_0based'][condition]]) for side in candidates]
        if any(query is None for query in queries):
            continue
        present[channel] = True
        for side, query in enumerate(queries):
            for i, (a, b) in enumerate(zip(left[0], right[0], strict=True)):
                margins[channel, side, i] = support(query, a, adduct == ADDUCTS[0]) - support(query, b, adduct == ADDUCTS[0])
    for i in range(len(ENERGIES)):
        if not present[i] or not present[i + len(ENERGIES)]:
            present[[i, i + len(ENERGIES)]] = False
    return margins, present, dict(bridge=left[1] + right[1], ring=left[2] + right[2])


def describe_pair(pair, scans):
    margins, present, counts = spectral_views(pair, scans)
    candidates = (pair['left'], pair['right'])
    key = '/'.join(sorted(candidate['key'] for candidate in candidates))
    output = dict(fold=pair['fold'], pair_sha256=sha256(key.encode()).hexdigest(), similarity=pair['morgan_tanimoto'],
                  added_fragment_components=counts, views={})
    for name, channels in VIEWS.items():
        kept = [i for i in channels if present[i]]
        if kept:
            score = margins[kept].mean(axis=0)
            output['views'][name] = {arm:score[:,i].tolist() for i,arm in enumerate(ARMS)}
    return output


def compare(rows, name):
    present = [row['views'][name] for row in rows if name in row['views']]
    if not present:
        return {'models':{arm:evaluate([]) for arm in ARMS}, 'sign_transitions':{}}
    margins = {arm:[data[arm] for data in present] for arm in ARMS}
    label = lambda value: 'win' if value[0] - value[1] > 1e-10 else ('loss' if value[0] - value[1] < -1e-10 else 'tie')
    transitions = {arm:dict(Counter(f'{label(a)}_to_{label(b)}' for a,b in zip(margins['single'],margins[arm],strict=True)))
                   for arm in ARMS[1:]}
    return dict(models={arm:evaluate(values) for arm,values in margins.items()}, sign_transitions=transitions)


def run(train, pairs, receipt, out, per_fold):
    start = monotonic()
    cleavage_rank.load_chemistry()
    chosen, scans = load_pairs(pairs, receipt, train, per_fold=per_fold, minimum_similarity=0.6)
    rows = [describe_pair(pair, scans) for pair in chosen]
    out.mkdir(parents=True, exist_ok=True)
    with (out/'two_cut_pair_predictions.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(',',':'))+'\n')
    summary = dict(scope='Observed competition training spectra. Same formula and same experimental condition on each pair.',
                   selected_pairs=len(chosen), available_scans=len(scans), sample_per_fold=per_fold, selected_similarity_min=0.6,
                   pairs_with_added_fragments={arm:sum(row['added_fragment_components'][arm]>0 for row in rows)
                                               for arm in ('bridge','ring')},
                   seconds=monotonic()-start, views={name:compare(rows,name) for name in VIEWS})
    (out/'two_cut_hard_pair_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train','pairs','receipt','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--per-fold',type=int,default=160)
    run(**vars(parser.parse_args()))

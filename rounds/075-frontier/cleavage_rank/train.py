"""Learn CE-specific single-cut evidence from measured formula siblings, evaluate real lists."""
from collections import defaultdict
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import importlib.metadata
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path('/kaggle/input')
OUT = Path('/kaggle/working')
SEED = 75129
HYDROGEN = 1.00782503223
PROTON = 1.00727646662
ENERGIES = (20, 40, 60)
# 5 retained atom types x 5 lost atom types, H transfer, ring context and mass fraction.
PER_ENERGY = 25 + 3 + 4 + 4 + 1
WIDTH = PER_ENERGY * len(ENERGIES)
ARMS = ('baseline', 'uniform', 'learned')
FROZEN_SHA = 'f9e684409c06ae479bf9c4de4c9243500160eabcb24d98d81d1ae827041d4c0c'


def artifact(name):
    matches = list(ROOT.rglob(name))
    if len(matches) != 1:
        raise ValueError(f'Expected one {name}, got {matches}')
    return matches[0]


def load_chemistry():
    try:
        version = importlib.metadata.version('rdkit')
    except importlib.metadata.PackageNotFoundError:
        version = ''
    if version.replace('.03.', '.3.') != '2026.3.3':
        tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
        wheel = next(ROOT.rglob(f'rdkit-2026.3.3-{tag}-*.whl'))
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                        '--quiet', str(wheel)], check=True)
    global Chem
    from rdkit import Chem, RDLogger, rdBase
    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError('Unexpected RDKit runtime')
    RDLogger.DisableLog('rdApp.*')


def atom_kind(atom):
    return {6: 0, 7: 1, 8: 2, 16: 3}.get(atom.GetAtomicNum(), 4)


@lru_cache(maxsize=None)
def fragments(smiles):
    """Hypotheses for protonated pieces from a single acyclic single-bond cut."""
    mol = Chem.MolFromSmiles(smiles)
    masses, codes = [], []
    if mol and len(Chem.GetMolFrags(mol)) == 1 and all(a.GetFormalCharge() == 0 for a in mol.GetAtoms()):
        table = Chem.GetPeriodicTable()
        atom_mass = np.array([(table.GetMassForIsotope(a.GetAtomicNum(), a.GetIsotope())
                               if a.GetIsotope() else table.GetMostCommonIsotopeMass(a.GetAtomicNum()))
                              + a.GetTotalNumHs() * HYDROGEN for a in mol.GetAtoms()])
        total = atom_mass.sum()
        adj = [[n.GetIdx() for n in a.GetNeighbors()] for a in mol.GetAtoms()]
        for bond in mol.GetBonds():
            if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            a, b = bond.GetBeginAtom(), bond.GetEndAtom()
            stack, seen = [a.GetIdx()], {b.GetIdx(), a.GetIdx()}
            side = 0.
            while stack:
                index = stack.pop()
                side += atom_mass[index]
                for neighbor in adj[index]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
            for kept, lost, mass in ((a, b, side), (b, a, total - side)):
                pair = 5 * atom_kind(kept) + atom_kind(lost)
                ring = 2 * int(kept.IsInRing()) + int(lost.IsInRing())
                quartile = min(int(4 * mass / total), 3)
                for ix, offset in enumerate((-1, 0, 1)):
                    masses.append(mass + PROTON + offset * HYDROGEN)
                    codes.append((pair, 25 + ix, 28 + ring, 32 + quartile, 36))
    return np.asarray(masses), np.asarray(codes, dtype=np.int16).reshape(-1, 5)


def scan(mzs, intensities, precursor, energy):
    mzs, intensities = np.asarray(mzs), np.asarray(intensities)
    keep = (mzs > 20) & (mzs < precursor - 1.5) & (intensities > 0)
    mzs, intensity = mzs[keep], np.sqrt(intensities[keep])
    order = np.argsort(mzs)
    return mzs[order], intensity[order] / max(1e-10, intensity.sum()), ENERGIES.index(int(energy))


def evidence(scans, smiles):
    masses, codes = fragments(smiles)
    output = np.zeros((3, PER_ENERGY), dtype=np.float32)
    if not len(masses):
        return output.ravel()
    for mz, weight, energy in scans:
        if not len(mz):
            continue
        right = np.searchsorted(mz, masses)
        tolerance = np.maximum(0.003, masses * 10e-6)
        indices = []
        descriptors = []
        for bucket in (right - 1, right):
            valid = (bucket >= 0) & (bucket < len(mz))
            indices_in = np.flatnonzero(valid)
            matched = indices_in[abs(mz[bucket[valid]] - masses[valid]) <= tolerance[valid]]
            if len(matched):
                indices.append(bucket[matched])
                descriptors.append(codes[matched])
        if indices:
            peaks = np.concatenate(indices)
            feature_ids = np.concatenate(descriptors)
            # A peak supported by several equivalent cuts counts once per descriptor.
            unique = np.unique((peaks[:, None] * PER_ENERGY + feature_ids).ravel())
            output[energy] += np.bincount(unique % PER_ENERGY, weights=weight[unique // PER_ENERGY],
                                         minlength=PER_ENERGY)
    return output.ravel() / len(scans)


def training_lists(frame):
    examples = []
    for _, group in frame.groupby('formula'):
        structures = list(group.groupby('metric_key', sort=True))
        if len(structures) < 2:
            continue
        smiles = [part.iloc[0].smiles for _, part in structures]
        for truth, (_, part) in enumerate(structures):
            observations = [scan(np.fromstring(row.mzs, sep=','), np.fromstring(row.intensities, sep=','),
                                 row.precursor_mz, row.collision_energy) for row in part.itertuples()]
            features = np.stack([evidence(observations, candidate) for candidate in smiles])
            examples.append((features, truth))
    return examples


def pad(lists, device):
    n = max(len(x) for x, _ in lists)
    pairs = np.zeros((len(lists), n, WIDTH), dtype='float32')
    masks = np.zeros(pairs.shape[:2], dtype=bool)
    for i, (features, _) in enumerate(lists):
        pairs[i, :len(features)] = features
        masks[i, :len(features)] = True
    return (torch.from_numpy(pairs).to(device), torch.from_numpy(masks).to(device),
            torch.tensor([y for _, y in lists], device=device))


def fit(train, validation, device):
    x, mask, truth = pad(train, device)
    v, vm, y = pad(validation, device)
    model = nn.Sequential(nn.Linear(WIDTH, 48), nn.SiLU(), nn.Linear(48, 1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    best = float('inf')
    history = []
    for epoch in range(1, 31):
        model.train()
        for ids in torch.randperm(len(x), device=device).split(48):
            scores = model(x[ids]).squeeze(-1).masked_fill(~mask[ids], -torch.inf)
            loss = nn.functional.cross_entropy(scores, truth[ids])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            model.eval()
            scores = model(v).squeeze(-1).masked_fill(~vm, -torch.inf)
            val = nn.functional.cross_entropy(scores, y).item()
        history.append(val)
        if val < best:
            best, selected = val, epoch
            torch.save(model.state_dict(), OUT / 'cleavage_rank.pt')
    model.load_state_dict(torch.load(OUT / 'cleavage_rank.pt', map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        scores = model(v).squeeze(-1).masked_fill(~vm, -torch.inf)
        uniform = v[:, :, PER_ENERGY-1::PER_ENERGY].sum(dim=2).masked_fill(~vm, -torch.inf)
        # Every formula's candidate vector uses the same deterministic structure order across its queries.
        # Selecting the next true identity cyclically pairs each score vector with its sibling's query.
        swapped = (y + 1) % vm.sum(dim=1)

        def expected_rr(values, labels):
            chosen = values.gather(1, labels[:, None])
            greater = (values > chosen).sum(1)
            tied = (values == chosen).sum(1)
            ranks = torch.arange(1, values.size(1) + 1, device=device)[None, :]
            tie_interval = (ranks > greater[:, None]) & (ranks <= (greater + tied)[:, None])
            return float(((tie_interval / ranks).sum(1) / tied).mean())

        diagnostics = dict(uniform=expected_rr(uniform, y), learned=expected_rr(scores, y),
                           same_formula_query_swap=expected_rr(scores, swapped))
    return model.cpu().eval(), dict(best_epoch=selected, val_cross_entropy=best, history=history,
                                     train_queries=len(train), validation_queries=len(validation),
                                     controlled_same_formula_expected_rr=diagnostics)


def order(candidates, scores):
    result = list(candidates)
    formula_slots = defaultdict(list)
    for i, c in enumerate(candidates):
        formula_slots[c['formula']].append(i)
    for slots in formula_slots.values():
        if len(slots) < 2:
            continue
        values = scores[slots]
        # Freeze list position prior and bound how strongly a model can replace it.
        residual = np.clip(values - np.median(values), -4, 4)
        total = -np.log(np.arange(1, len(slots) + 1)) + 0.5 * residual
        ranked = np.argsort(-total, kind='stable')
        for slot, j in zip(slots, ranked):
            result[slot] = candidates[slots[j]]
    return result


@torch.no_grad()
def external(model, locked, spectra):
    predictions = []
    for row in locked:
        key = row['query']
        candidates = row['ranks']['expanded_open']
        qs = spectra.get_group(key)
        scans = [scan(s.ms2_mzs, s.ms2_normalized_intensities, s.precursor_mz, s.collision_energy_ev[0])
                 for s in qs.itertuples()]
        features = np.stack([evidence(scans, c['smiles']) for c in candidates])
        learned = model(torch.from_numpy(features)).flatten().numpy()
        # Uniform logit-scale control uses 4 x observed fragment-intensity coverage [0, 1].
        uniform = 4 * features[:, PER_ENERGY-1::PER_ENERGY].sum(axis=1)
        rankings = [candidates, order(candidates, uniform), order(candidates, learned)]
        predictions.append(dict(query=key, stratum=row['murcko'],
                                orders={arm: [c['metric_key'] for c in ranks] for arm, ranks in zip(ARMS, rankings)}))
    frozen = OUT / 'cleavage_rank_orders.jsonl'
    frozen.write_text(''.join(json.dumps(row) + '\n' for row in predictions))
    results = []
    for line in frozen.read_text().splitlines():
        row = json.loads(line)
        item = dict(query=row['query'], stratum=row['stratum'])
        for arm, keys in row['orders'].items():
            item[arm] = keys.index(row['query']) + 1 if row['query'] in keys else 0
        results.append(item)
    return pd.DataFrame(results)


def metrics(frame):
    base = frame.baseline.to_numpy()
    def measure(arm):
        ranks = frame[arm].to_numpy()
        rr = np.divide(1., ranks, out=np.zeros(len(ranks)), where=ranks > 0)
        return dict(mrr25=float(rr.mean()), first=int((ranks == 1).sum()), top25=int((ranks > 0).sum()),
                    first_gains=int(((ranks == 1) & (base != 1)).sum()), first_losses=int(((ranks != 1) & (base == 1)).sum()))
    return {arm: measure(arm) for arm in ARMS}


def main():
    started = time.monotonic()
    load_chemistry()
    torch.manual_seed(SEED)
    tsv = artifact('verified_e180_d3_input.tsv')
    identities = pd.read_parquet(artifact('e180_fragnnet_d3_structure_selection.parquet'))
    training = pd.read_csv(tsv, sep='\t')
    training['raw_key'] = training.inchikey.str[:14]
    training = training.merge(identities[['raw_key', 'metric_key']], on='raw_key', validate='many_to_one')
    locked_path = artifact('e180_reserve_first_formula_c41_locked.jsonl')
    if sha256(locked_path.read_bytes()).hexdigest() != FROZEN_SHA:
        raise ValueError('Independent C41 ranked lists changed')
    locked = [json.loads(line) for line in locked_path.read_text().splitlines()]
    external_ids = {row['query'] for row in locked}
    if len(external_ids) != 121 or external_ids.intersection(training.metric_key):
        raise ValueError('External panel changed or overlaps training/validation identities')
    if training.groupby('generic_murcko').fold.nunique().gt(1).any():
        raise ValueError('A nonempty scaffold crossed the training/validation boundary')
    print('Preparing training fields', training.fold.value_counts().to_dict(), flush=True)
    lists = {fold: training_lists(group) for fold, group in training.groupby('fold')}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, training_report = fit(lists['train'], lists['val'], device)
    print('Internal fit', training_report['best_epoch'], flush=True)
    spectra = pd.read_parquet(artifact('e180_scan_pipeline_queries_ce.parquet'))
    results = external(model, locked, spectra.groupby('molecule_id'))
    report = dict(source_sha256=sha256(tsv.read_bytes()).hexdigest(), fixed_list_sha256=FROZEN_SHA,
                  model_device=str(device), training=training_report, all121=metrics(results),
                  by_murcko={k: metrics(g) for k, g in results.groupby('stratum')},
                  spectra=len(spectra), seconds=round(time.monotonic() - started, 2))
    original = report['all121']['baseline']
    if (original['first'], original['top25']) != (71, 108) or abs(original['mrr25'] - .7073553719) > 1e-8:
        raise ValueError('Existing natural-list reference score did not reproduce')
    results.to_parquet(OUT / 'cleavage_rank_per_query.parquet', index=False)
    (OUT / 'cleavage_rank_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

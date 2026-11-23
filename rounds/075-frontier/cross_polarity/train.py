"""Compare paired and unpaired training at fixed spectra, structure pools and optimizer steps."""
from hashlib import sha256
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F

PROTON = 1.00727646662
MAX_RAW = 12000
LIMIT = 6000
BINS = 2048
EPOCHS = 18
SEED = 7523
ADDUCTS = ('[M+H]+', '[M-H]-')
ROOT = Path('/kaggle/input')
OUTPUT = Path('/kaggle/working')


def find(slug, name):
    matches = [path for path in ROOT.rglob(name) if slug.casefold() in
               {part.casefold() for part in path.parts}]
    if len(matches) != 1:
        raise ValueError(f'Expected one {name} under {slug}, got {len(matches)}')
    return matches[0]


def hash_order(value):
    return sha256(f'casmi-polarity-v1/{value}'.encode()).digest()


def chemistry():
    import importlib.metadata
    import subprocess
    import sys

    try:
        version = importlib.metadata.version('rdkit')
    except importlib.metadata.PackageNotFoundError:
        version = ''
    if version.replace('.03.', '.3.') != '2026.3.3':
        tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
        wheel, = ROOT.rglob(f'rdkit-2026.3.3-{tag}-*.whl')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-index', '--no-deps', str(wheel)], check=True)
    from rdkit import Chem, RDLogger, rdBase
    from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.Scaffolds import MurckoScaffold

    if rdBase.rdkitVersion != '2026.03.3':
        raise RuntimeError(f'Expected competition RDKit, got {rdBase.rdkitVersion}')
    RDLogger.DisableLog('rdApp.*')
    tautomer = rdMolStandardize.TautomerEnumerator()
    fingerprints = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def describe(smiles, scaffold=False):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        canonical = tautomer.Canonicalize(mol)
        key = Chem.MolToInchiKey(canonical)[:14]
        group = ''
        if scaffold:
            framework = MurckoScaffold.GetScaffoldForMol(canonical)
            if framework.GetNumAtoms():
                group = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(framework))
        return key, group, rdMolDescriptors.CalcExactMolWt(canonical), fingerprints.GetFingerprintAsNumPy(canonical)

    return describe


def spectrum_vector(masses, intensities, precursor, view):
    masses, intensities = np.asarray(masses), np.asarray(intensities)
    keep = np.isfinite(masses) & np.isfinite(intensities) & (intensities > 0)
    keep &= (masses >= 15) & (masses < precursor - 1.5)
    weight = np.sqrt(intensities[keep])
    peaks = masses[keep]
    vector = np.zeros(4 * BINS + 2, dtype=np.float32)
    for index, coordinates in enumerate((peaks, precursor - peaks)):
        for resolution, step in enumerate((1., 20.)):
            slots = np.rint(coordinates * step).astype(int) % BINS
            np.maximum.at(vector[(index * 2 + resolution) * BINS:(index * 2 + resolution + 1) * BINS], slots, weight)
    vector[:-2] /= max(float(np.linalg.norm(vector[:-2])), 1e-10)
    vector[-2 + view] = 0.25
    return vector


def paired_training(source, describe, excluded):
    rows = pq.ParquetFile(source)
    meta = ['inchikey14', 'normalized_smiles', 'precursor_mz', 'num_peaks', 'ingest_lib',
            'instrument_type', 'adduct', 'collision_energy_ev', 'collision_energy_orig_units',
            'collision_energy_orig']
    scans = {}
    for group in range(rows.num_row_groups):
        frame = rows.read_row_group(group, columns=meta).to_pandas()
        select = (frame.ingest_lib.eq('enveda-180') & frame.instrument_type.eq('timsTOF') &
                  frame.adduct.isin(ADDUCTS) & frame.collision_energy_orig_units.eq('eV') &
                  frame.collision_energy_orig.isin(('40', '-40')) & frame.num_peaks.ge(4) &
                  frame.precursor_mz.gt(30) & frame.inchikey14.notna() & frame.normalized_smiles.notna())
        for rec in frame.loc[select].itertuples():
            if rec.collision_energy_ev is None or list(rec.collision_energy_ev) != [40.]:
                continue
            view = ADDUCTS.index(rec.adduct)
            entry = scans.setdefault(rec.inchikey14, {})
            if view not in entry or rec.num_peaks > entry[view][0]:
                entry[view] = (rec.num_peaks, group, rec.Index, rec.precursor_mz, rec.normalized_smiles)
    paired = sorted((key for key, modes in scans.items() if len(modes) == 2), key=hash_order)
    print('Raw paired structure keys', len(paired), flush=True)
    identities, vectors, masses, observed, partitions, requests = set(), [], [], [], [], {}
    banned_keys, banned_scaffolds = excluded
    for raw in paired[:MAX_RAW]:
        modes = scans[raw]
        descriptions = [describe(modes[view][4], scaffold=True) for view in range(2)]
        if None in descriptions or descriptions[0][0] != descriptions[1][0]:
            continue
        key, scaffold, exact_mass, fingerprint = descriptions[0]
        if not key or key in identities or key in banned_keys or scaffold in banned_scaffolds:
            continue
        identities.add(key)
        idx = len(vectors)
        partitions.append(hash_order(scaffold or key)[0] < 51)
        vectors.append(fingerprint)
        masses.append(exact_mass)
        observed.append([modes[v][3] + (-PROTON if v == 0 else PROTON) for v in range(2)])
        for view in range(2):
            _, group, row, precursor, _ = modes[view]
            requests.setdefault(group, []).append((row, idx, view, precursor))
        if len(vectors) >= LIMIT:
            break
    if len(vectors) < 2000:
        raise ValueError(f'Too few disjoint qualifying pairs: {len(vectors)}')
    spectra = np.empty((len(vectors), 2, 4 * BINS + 2), dtype=np.float16)
    for group, order in requests.items():
        chunk = rows.read_row_group(group, columns=['ms2_mzs', 'ms2_normalized_intensities'])
        selected = chunk.take([row for row, _, _, _ in order]).to_pylist()
        for peaks, (_, idx, view, precursor) in zip(selected, order, strict=True):
            spectra[idx, view] = spectrum_vector(peaks['ms2_mzs'], peaks['ms2_normalized_intensities'], precursor, view)
    return spectra, np.stack(vectors), np.array(masses), np.array(observed), np.array(partitions)


class Retriever(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Sequential(nn.Linear(4 * BINS + 2, 384), nn.LayerNorm(384), nn.GELU(), nn.Linear(384, 128))
        self.candidate = nn.Sequential(nn.Linear(2048, 192), nn.GELU(), nn.Linear(192, 128))

    def encode_query(self, spectra):
        return F.normalize(self.query(spectra.float()), dim=-1)

    def encode_candidate(self, fingerprints):
        return F.normalize(self.candidate(fingerprints.float()), dim=-1)


def fit(spectra, fingerprints, mass, train_ids, alignment):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = Retriever().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    history = []
    order = train_ids[np.argsort(mass[train_ids], kind='stable')]
    for _ in range(EPOCHS):
        rotated = np.roll(order, int(rng.integers(0, 48)))
        batches = list(np.array_split(rotated, max(1, len(rotated) // 48)))
        losses = []
        for batch_index in rng.permutation(len(batches)):
            ids = batches[batch_index]
            query = model.encode_query(spectra[ids])
            candidates = model.encode_candidate(fingerprints[ids])
            truth = torch.arange(len(ids), device='cuda')
            logits = (query @ candidates.T) * 16
            structure_loss = (F.cross_entropy(logits[:, 0], truth) + F.cross_entropy(logits[:, 1], truth)) / 2
            opposite = (query[:, 0] @ query[:, 1].T) * 16
            pair_loss = (F.cross_entropy(opposite, truth) + F.cross_entropy(opposite.T, truth)) / 2
            loss = structure_loss + alignment * pair_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(structure_loss.detach()))
        history.append(float(np.mean(losses)))
    return model.eval(), history


def metrics(ranks):
    array = np.asarray(ranks, dtype=int)
    rr = np.divide(1., array, out=np.zeros(len(array)), where=(0 < array) & (array <= 25))
    return dict(n=len(array), mrr25=float(rr.mean()) if len(rr) else None,
                top1=int((array == 1).sum()), top5=int(((array > 0) & (array <= 5)).sum()),
                top25=int(((array > 0) & (array <= 25)).sum()))


@torch.inference_mode()
def measured_pool(model, spectra, fingerprints, mass, observed, dev):
    c = model.encode_candidate(fingerprints)
    ids = np.flatnonzero(dev)
    ranks = [[], []]
    sizes = [[], []]
    for batch in np.array_split(ids, max(1, len(ids) // 48)):
        values = (model.encode_query(spectra[batch]) @ c.T).cpu().numpy()
        for row, idx in enumerate(batch):
            for view in range(2):
                eligible = np.abs(mass - observed[idx, view]) <= observed[idx, view] * 1e-5
                order = np.argsort(-values[row, view], kind='stable')
                eligible_order = order[eligible[order]]
                place = np.flatnonzero(eligible_order == idx)
                ranks[view].append(int(place[0]) + 1 if place.size else 0)
                sizes[view].append(len(eligible_order))
    return {adduct: dict(all=metrics(rank), with_competition=metrics(
        [r for r, count in zip(rank, size) if count > 1]), mass_window_hits=sum(r > 0 for r in rank),
        candidate_count_median=float(np.median(size))) for adduct, rank, size in zip(ADDUCTS, ranks, sizes)}


@torch.inference_mode()
def archived_pool(model, queries, candidates, describe):
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def bits(smiles):
        item = describe(smiles)
        return item[-1] if item is not None else None

    result = {}
    for rec in queries.itertuples():
        items = candidates.get_group(rec.molecule_id)
        entries = [(row.candidate_id, bits(row.smiles)) for row in items.itertuples()]
        entries = [(key, fp) for key, fp in entries if fp is not None]
        if not entries:
            result[rec.molecule_id] = []
            continue
        view = 1
        spec = spectrum_vector(rec.ms2_mzs, rec.ms2_normalized_intensities, rec.precursor_mz, view)
        q = model.encode_query(torch.as_tensor(spec, device='cuda'))
        c = model.encode_candidate(torch.as_tensor(np.stack([fp for _, fp in entries]), device='cuda'))
        order = np.argsort(-(c @ q).cpu().numpy(), kind='stable')[:25]
        result[rec.molecule_id] = [entries[i][0] for i in order]
    return result


def main():
    start = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError('The paired training comparison requires Kaggle GPU.')
    describe = chemistry()
    tag = 'enveda-075b-external-massbank-adducts'
    panel = pd.read_parquet(find(tag, 'pipeline_queries.parquet'))
    panel = panel[panel.adduct.eq(ADDUCTS[1])]
    evaluation = pd.read_parquet(find(tag, 'labels.parquet'))
    evaluation = evaluation[evaluation.molecule_id.isin(panel.molecule_id)]
    source = pd.read_parquet(find('enveda-075b-massbank-adducts-pipeline', 'source_candidates.parquet'))
    source = source[source.molecule_id.isin(panel.molecule_id)].drop_duplicates(['molecule_id', 'candidate_id'])
    previous = pd.read_parquet(find('enveda-075b-massbank-adducts-pipeline', 'candidates.parquet'))
    if len(panel) != 47 or not panel.molecule_id.is_unique or set(evaluation.molecule_id) != set(panel.molecule_id):
        raise ValueError('Expected the fixed 47 external negative-ion queries.')
    external = [describe(row.truth_smiles, scaffold=True) for row in evaluation.itertuples()]
    if any(item is None for item in external):
        raise ValueError('Could not verify all external identities for training exclusion.')
    excluded = (set(evaluation.candidate_id) | {x[0] for x in external}, {x[1] for x in external} - {''})
    raw = find('enveda-casmi26-molecule-id-mass-spectra', 'train.parquet')
    spectra, fp, masses, observed, dev = paired_training(raw, describe, excluded)
    print('Paired data', len(spectra), 'training', int((~dev).sum()), 'dev', int(dev.sum()), flush=True)
    spectra, fp = torch.from_numpy(spectra).cuda(), torch.from_numpy(fp).cuda()
    history, internal, external_predictions = {}, {}, {}
    for name, alignment in [('unpaired', 0.), ('paired', 0.2)]:
        model, history[name] = fit(spectra, fp, masses, np.flatnonzero(~dev), alignment)
        torch.save(model.state_dict(), OUTPUT / f'{name}.pt')
        internal[name] = measured_pool(model, spectra, fp, masses, observed, dev)
        external_predictions[name] = archived_pool(model, panel, source.groupby('molecule_id'), describe)
        print(name, internal[name], flush=True)
    outcome = OUTPUT / 'external_predictions.json'
    outcome.write_text(json.dumps(external_predictions) + '\n')
    external_metrics = {}
    previous = previous[previous.molecule_id.isin(panel.molecule_id)].sort_values('rank')
    baseline = []
    recorded = {(r.molecule_id, r.candidate_id): r.rank for r in previous.itertuples()}
    for row in evaluation.itertuples():
        baseline.append(int(recorded.get((row.molecule_id, row.candidate_id), 0)))
    for name, predictions in json.loads(outcome.read_text()).items():
        ranks = []
        for row in evaluation.itertuples():
            order = predictions[row.molecule_id]
            ranks.append(order.index(row.candidate_id) + 1 if row.candidate_id in order else 0)
        external_metrics[name] = metrics(ranks)
    report = dict(device=torch.cuda.get_device_name(), seconds=round(time.monotonic() - start, 1),
                  train=int((~dev).sum()), heldout=int(dev.sum()), internal=internal, external=external_metrics,
                  external_baseline=metrics(baseline), external_candidate_rows=len(source), history=history)
    (OUTPUT / 'pair_training_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

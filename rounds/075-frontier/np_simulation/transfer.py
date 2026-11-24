"""Compare equal-step real-only and simulated→real fingerprint training on frozen candidate lists."""

import argparse
from collections import defaultdict
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch
from torch import nn

SEEDS = (2411, 2412, 2413)
STEPS = 128
BATCH = 64
FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
TAUTOMERS = rdMolStandardize.TautomerEnumerator()


def hash_list(values):
    return sha256('\n'.join(sorted(values)).encode()).hexdigest()


@lru_cache(maxsize=12000)
def structure(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Invalid frozen or training structure')
    Chem.RemoveStereochemistry(mol)
    mol = TAUTOMERS.Canonicalize(mol)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    generic = (
        Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(scaffold))
        if scaffold.GetNumAtoms()
        else ''
    )
    key = Chem.MolToInchiKey(mol).split('-')[0]
    return key, generic, FP.GetFingerprintAsNumPy(mol).astype(np.float32)


def spectrum(mz, intensity, precursor, simulated=False):
    mz = np.asarray(mz, dtype=np.float64)
    v = np.asarray(intensity, dtype=np.float64)
    ok = np.isfinite(mz) & np.isfinite(v) & (mz > 0) & (v > 0) & (mz < precursor - 0.5)
    mz, v = mz[ok], v[ok]
    if simulated:
        v = np.square(v)
    if not len(v):
        return None
    keep = v >= 0.01 * max(v)
    mz, v = mz[keep], np.sqrt(v[keep])
    if len(v) < 3:
        return None
    feat = np.zeros((2, 1024), dtype=np.float32)
    for i, axis in enumerate((mz, precursor - mz)):
        bins = np.rint(axis).astype(int)
        valid = (bins >= 0) & (bins < 1024)
        np.maximum.at(feat[i], bins[valid], v[valid])
        n = np.linalg.norm(feat[i])
        if n:
            feat[i] /= n
    return feat.ravel()


def blocked(molecules, truth_keys, scaffolds):
    return {
        k
        for k, (identity, scaffold, _) in molecules.items()
        if identity in truth_keys or (scaffold and scaffold in scaffolds)
    }


def collect_real(train, structures_file, truth_keys, scaffolds):
    table = pd.read_parquet(
        structures_file, columns=['inchikey14', 'smiles', 'libs', 'n_pos']
    )
    table = table[
        table.libs.str.split(';').map(lambda libs: 'riken' in libs) & table.n_pos.gt(0)
    ]
    table = table.assign(
        order=table.inchikey14.map(
            lambda k: sha256(('np-sim-transfer-v1/' + k).encode()).hexdigest()
        )
    )
    chosen = table.sort_values('order').head(3600).drop_duplicates('inchikey14')
    rows = {r.inchikey14: structure(r.smiles) for r in chosen.itertuples(index=False)}
    rejected = blocked(rows, truth_keys, scaffolds)
    wanted = pa.array(sorted(set(rows) - rejected))
    views = defaultdict(list)
    columns = [
        'inchikey14',
        'ingest_lib',
        'adduct',
        'precursor_mz',
        'ms2_mzs',
        'ms2_normalized_intensities',
    ]
    for b in pq.ParquetFile(train).iter_batches(batch_size=32768, columns=columns):
        keep = pc.and_(
            pc.and_(pc.equal(b[1], 'riken'), pc.equal(b[2], '[M+H]+')),
            pc.is_in(b[0], value_set=wanted),
        )
        if not pc.any(keep).as_py():
            continue
        for raw, _, _, mz, ms, iv in zip(*b.filter(keep).to_pydict().values()):
            identity = rows[raw][0]
            if len(views.get(identity, ())) >= 3:
                continue
            x = spectrum(ms, iv, mz)
            if x is not None:
                views[identity].append(x)
    chosen_ids = list(
        dict.fromkeys(
            rows[k][0]
            for k in chosen.inchikey14
            if k not in rejected and rows[k][0] in views
        )
    )[:2000]
    selected = {k: views[k] for k in chosen_ids}
    features = {r[0]: r for r in rows.values() if r[0] in selected}
    return (
        selected,
        features,
        dict(
            source='riken; [M+H]+', raw_examined=len(rows), raw_rejected=len(rejected)
        ),
    )


def collect_sim(directory, truth_keys, scaffolds):
    sel = pd.read_parquet(directory / 'selected.parquet')
    rows = {r.scorer_key: structure(r.smiles) for r in sel.itertuples(index=False)}
    mass = dict(zip(sel.scorer_key, sel.mono_mass))
    rejected = blocked(rows, truth_keys, scaffolds)
    views = defaultdict(list)
    for path in sorted(directory.glob('spectra_*.parquet')):
        for r in pd.read_parquet(path).itertuples(index=False):
            if r.scorer_key in rejected:
                continue
            x = spectrum(
                r.mz, r.intensity, mass[r.scorer_key] + 1.007276466621, simulated=True
            )
            if x is not None:
                views[rows[r.scorer_key][0]].append(x)
    features = {r[0]: r for r in rows.values() if r[0] in views}
    return (
        views,
        features,
        dict(
            source='LOTUS + MSG GLACIER; [M+H]+',
            raw_examined=len(rows),
            raw_rejected=len(rejected),
        ),
    )


def pack(views, features, truth, scaffolds, meta):
    ids = sorted(views)
    unique_scaffolds = {features[k][1] for k in ids if features[k][1]}
    assert not set(ids) & truth and not unique_scaffolds & scaffolds
    offsets = np.cumsum([0] + [len(views[k]) for k in ids])
    x = torch.from_numpy(np.stack([v for k in ids for v in views[k]]))
    y = torch.from_numpy(np.stack([features[k][2] for k in ids]))
    meta.update(
        identities=len(ids),
        spectra=len(x),
        nonempty_generic_scaffolds=len(unique_scaffolds),
        identity_hash=hash_list(ids),
        overlap_truth_keys=0,
        overlap_nonempty_generic_scaffolds=0,
    )
    return (x, y, offsets), meta


def fit(seed, first, real):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(2048, 160), nn.ReLU(), nn.Linear(160, 512))
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    prevalence = real[1].mean(0)
    importance = ((1 - prevalence) / (prevalence + 0.001)).sqrt().clamp(1, 6)
    loss = nn.BCEWithLogitsLoss(pos_weight=importance)
    for phase, data in enumerate((first, real)):
        x, y, offsets = data
        rng = np.random.default_rng(seed + 100 * phase)
        for _ in range(STEPS):
            groups = rng.integers(0, len(y), size=BATCH)
            samples = rng.integers(offsets[groups], offsets[groups + 1])
            pred = model(x[samples])
            value = loss(pred, y[groups])
            opt.zero_grad()
            value.backward()
            opt.step()
    return model.eval()


def score_predictions(model, queries, cands):
    result = {}
    with torch.inference_mode():
        for q, x in queries.items():
            probs = model(torch.from_numpy(np.stack(x))).sigmoid().mean(0).numpy()
            fp = cands[q]
            dot = fp @ probs
            result[q] = dot / (fp.sum(1) + np.square(probs).sum() - dot + 1e-8)
    return result


def ranked(metric_keys, predictions, truth):
    arms = {'model': {}, 'fusion': {}}
    for q, keys in metric_keys.items():
        base = np.arange(len(keys))
        score = predictions.get(q)
        orders = {a: base for a in arms}
        if score is not None:
            by = np.argsort(-score, kind='stable')
            model_rank = np.empty(len(by), int)
            model_rank[by] = base + 1
            mix = 0.65 * np.log1p(base) + 0.35 * np.log(model_rank)
            orders = {'model': by, 'fusion': np.argsort(mix, kind='stable')}
        for arm, order in orders.items():
            found = np.flatnonzero(keys[order] == truth[q])
            arms[arm][q] = int(found[0]) + 1 if len(found) else None
    return arms


def metrics(ranks):
    return {
        'n': len(ranks),
        'top1': sum(v == 1 for v in ranks.values()),
        'top5': sum(v is not None and v <= 5 for v in ranks.values()),
        'top25': sum(v is not None and v <= 25 for v in ranks.values()),
        'mrr25': sum(1 / v for v in ranks.values() if v) / len(ranks),
    }


def compare(a, b):
    best = lambda x: x is not None and x <= 5
    return {
        'new_top5': [q for q in a if not best(a[q]) and best(b[q])],
        'lost_top5': [q for q in a if best(a[q]) and not best(b[q])],
        'new_top1': [q for q in a if a[q] != 1 and b[q] == 1],
        'lost_top1': [q for q in a if a[q] == 1 and b[q] != 1],
        'delta_mrr25': metrics(b)['mrr25'] - metrics(a)['mrr25'],
    }


def available_swaps(scans):
    # A diagnostic donor must pass the same preprocessing and preserve the mass-window setting.
    frame = pd.DataFrame(scans, columns=['molecule', 'instrument', 'adduct', 'mz'])
    groups = frame.groupby(
        ['molecule', 'instrument', 'adduct'], as_index=False
    ).mz.median()
    pairs = set()
    for q in groups.itertuples(index=False):
        other = groups[
            (groups.instrument == q.instrument)
            & (groups.adduct == q.adduct)
            & (groups.molecule != q.molecule)
            & ((groups.mz - q.mz).abs() <= q.mz * 1e-5)
        ]
        pairs.update(tuple(sorted((q.molecule, k))) for k in other.molecule)
    return {
        'ppm': 10,
        'same_adduct_and_instrument': True,
        'eligible_queries': frame.molecule.nunique(),
        'pairs': sorted(pairs),
        'model_wrong_spectrum_diagnostic_run': False,
    }


def cluster_interval(original, modified, scaffolds):
    # Resample whole nonempty generic Murcko groups. Each acyclic target is a singleton.
    groups = defaultdict(list)
    for key in original:
        a, b = original[key], modified[key]
        groups[scaffolds[key] or key].append(
            (1 / b if b else 0.0) - (1 / a if a else 0.0)
        )
    totals = np.array([sum(x) for x in groups.values()])
    counts = np.array([len(x) for x in groups.values()])
    draws = np.random.default_rng(24110).integers(len(groups), size=(5000, len(groups)))
    differences = totals[draws].sum(1) / counts[draws].sum(1)
    return {
        'paired_generic_scaffold_groups': len(groups),
        'bootstrap_replicates': 5000,
        'delta_mrr25_percentile_95': np.quantile(differences, [0.025, 0.975]).tolist(),
    }


def run(args):
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    RDLogger.DisableLog('rdApp.*')
    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError('Use competition RDKit 2026.03.3')
    labels = pd.read_parquet(args.labels)
    structures = {
        r.molecule_id: structure(r.truth_smiles) for r in labels.itertuples(index=False)
    }
    truth = {k: v[0] for k, v in structures.items()}
    identity = set(truth.values())
    scaffolds = {v[1] for v in structures.values() if v[1]}
    assert all(
        truth[r.molecule_id] == r.candidate_id for r in labels.itertuples(index=False)
    )
    candidates = pd.read_parquet(args.candidates).sort_values(['molecule_id', 'rank'])
    cands = {}
    keys = {}
    for q, group in candidates.groupby('molecule_id', observed=True):
        assert (
            group['rank'].tolist() == list(range(1, len(group) + 1))
            and len(group) <= 25
        )
        rows = [structure(s) for s in group.smiles]
        keys[q] = np.array([r[0] for r in rows])
        cands[q] = np.stack([r[2] for r in rows])
        assert (keys[q] == group.candidate_id).all()
    assert set(keys) == set(truth)
    qry = pd.read_parquet(args.queries)
    queries = defaultdict(list)
    eligible_scans = []
    for r in qry.itertuples(index=False):
        if r.adduct == '[M+H]+':
            x = spectrum(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz)
            if x is not None:
                queries[r.molecule_id].append(x)
                eligible_scans.append(
                    (r.molecule_id, r.instrument_type, r.adduct, r.precursor_mz)
                )
    print('Heldout loaded', len(keys), 'eligible', len(queries), flush=True)
    real, rf, rm = collect_real(args.train, args.structures, identity, scaffolds)
    sim, sf, sm = collect_sim(args.simulation, identity, scaffolds)
    real, rm = pack(real, rf, identity, scaffolds, rm)
    sim, sm = pack(sim, sf, identity, scaffolds, sm)
    print('Training', rm['identities'], sm['identities'], flush=True)
    assert not set(rf) & set(sf), (
        'The simulated source was selected outside competition train.'
    )
    base = ranked(keys, {}, truth)['fusion']
    runs = {}
    means = {}
    for arm, first in [('real_only', real), ('sim_then_real', sim)]:
        preds = []
        runs[arm] = {}
        for seed in SEEDS:
            p = score_predictions(fit(seed, first, real), queries, cands)
            preds.append(p)
            ranks = ranked(keys, p, truth)
            runs[arm][str(seed)] = {k: metrics(r) for k, r in ranks.items()}
        means[arm] = ranked(
            keys,
            {k: np.mean([p[k] for p in preds], axis=0) for k in queries},
            truth,
        )
    simfusion = means['sim_then_real']['fusion']
    realfusion = means['real_only']['fusion']
    stats = {
        k: {kind: metrics(ranks) for kind, ranks in arms.items()}
        for k, arms in means.items()
    }
    return {
        'date_beijing': '2026-11-24',
        'rdkit': rdBase.rdkitVersion,
        'device': 'CPU',
        'elapsed_seconds': round(time.monotonic() - started, 2),
        'query_source': 'jiyujia/enveda-075b-external-massbank-202611/1',
        'candidate_source': 'jiyujia/enveda-075b-massbank-pipeline/1',
        'simulation_source': 'GLACIER LOTUS throughput pilot; see simulation summary hash',
        'sha256': {
            k: sha256(p.read_bytes()).hexdigest()
            for k, p in [
                ('candidates', args.candidates),
                ('labels', args.labels),
                ('queries', args.queries),
                ('structure_table', args.structures),
                ('simulation_summary', args.simulation / 'simulation_summary.json'),
            ]
        },
        'training': {
            'real': rm,
            'simulation': sm,
            'real_and_simulated_sources_shared_identity_count': len(set(rf) & set(sf)),
            'phase_steps': [STEPS, STEPS],
            'batch': BATCH,
            'seeds': SEEDS,
            'per_arm_training_samples_per_seed': 2 * STEPS * BATCH,
            'optimizer': 'AdamW, lr=.002, wd=.01, BCE real-prevalence-weighted',
            'phase2': 'Identical measured batches and order for each matched seed. Uniform identities then a uniform usable scan.',
        },
        'evaluation': {
            'molecules': len(truth),
            'applicable_molecules': len(queries),
            'raw_spectra': len(qry),
            'eligible_spectra': sum(map(len, queries.values())),
            'frozen_candidate_slots': len(candidates),
            'matched_spectral_swap_supply': available_swaps(eligible_scans),
            'unique_target_keys': len(identity),
            'unique_nonempty_generic_scaffolds': len(scaffolds),
            'target_mapped_to_MSG_train_or_val': int(labels.msg_train_val_match.sum()),
            'teacher_training_exposure': 'Existing external panel removed MassSpecGym train/validation identity matches. Individual GLACIER training manifest and scaffold exposure were not independently audited. Baseline weights are historical.',
        },
        'method': 'Relative 1% fragment threshold, at least three fragments below precursor−0.5 Da, nearest nominal mass bins for product m/z and neutral loss. Simulated intensity is squared before shared preprocessing. Uniform scans of training structures, fingerprint probability averages per query over available measured spectra, continuous Tanimoto against frozen fingerprints. Same ranking and slots in both arms. Fusion is 65% log original rank and 35% log model rank, stable ties. Queries without eligible spectra keep baseline for both arms. No hyperparameter selection on this run.',
        'baseline': metrics(base),
        'ensemble': stats,
        'per_seed': runs,
        'sim_fusion_vs_baseline': compare(base, simfusion),
        'real_fusion_vs_baseline': compare(base, realfusion),
        'sim_fusion_vs_real_fusion': compare(realfusion, simfusion),
        'cluster_bootstrap': {
            name: cluster_interval(
                reference, simfusion, {k: v[1] for k, v in structures.items()}
            )
            for name, reference in [
                ('sim_vs_real', realfusion),
                ('sim_vs_baseline', base),
            ]
        },
        'per_query': {
            q: {
                'eligible_scans': len(queries.get(q, [])),
                'baseline': v,
                'real_fusion': realfusion[q],
                'sim_fusion': simfusion[q],
                'real_model': means['real_only']['model'][q],
                'sim_model': means['sim_then_real']['model'][q],
            }
            for q, v in base.items()
        },
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in [
        'train',
        'structures',
        'simulation',
        'queries',
        'candidates',
        'labels',
        'output',
    ]:
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a), ensure_ascii=False, indent=2) + '\n')

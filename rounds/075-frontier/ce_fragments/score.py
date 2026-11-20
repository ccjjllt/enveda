import argparse
from collections import Counter
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename


DELTA_WEIGHT = 0.15


def load_iceberg(inputs, device, batch_size):
    packages = [path.parent for path in inputs.rglob("ice_runner.py") if "casmi26-iceberg" in path.parts]
    if len(packages) != 1:
        raise FileNotFoundError(f"Expected one MassSpecGym ICEBERG package, found {packages}")
    pkg = packages[0]
    compatible = set(sys_tags())
    wheels = {}
    paths = sorted(pkg.glob("wheels/*.whl*")) + sorted(inputs.rglob("rdkit-2025.3.6-*.whl*"))
    with tempfile.TemporaryDirectory(prefix="iceberg-wheels-") as temp:
        for path in paths:
            filename = path.name.removesuffix(".ice")
            name, _, _, tags = parse_wheel_filename(filename)
            if compatible.isdisjoint(tags) or name in wheels:
                continue
            wheels[name] = shutil.copyfile(path, Path(temp) / filename)
        if "rdkit" not in wheels:
            raise RuntimeError(f"Missing RDKit 2025.3.6 wheel for {sys.version.split()[0]}")
        site = tempfile.mkdtemp(prefix="iceberg-site-")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
             "--target", site, *wheels.values()],
            check=True,
        )
    sys.path[:0] = [str(pkg), site]
    import ice_runner as ice

    log = ice.Log()
    ice.setup_env(SimpleNamespace(pkg=str(pkg), site="none", any_rdkit=False), log)
    import torch

    torch.set_num_threads(2)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but the Kaggle notebook has no GPU")
    model = ice.load_model(str(pkg), device, log)
    print(f"ICEBERG device={device} torch={torch.__version__}", flush=True)
    return ice, ice.Predictor(model, device, batch_size, log), device


def clean_peaks(mz, intensity, precursor):
    mz = np.asarray(mz, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    keep = (
        np.isfinite(mz) & np.isfinite(intensity) &
        (mz > 0) & (intensity > 0) & (mz < precursor - 0.5)
    )
    mz, intensity = mz[keep], intensity[keep]
    if len(mz) == 0:
        return mz, intensity
    keep = intensity >= 0.002 * intensity.max()
    mz, intensity = mz[keep], intensity[keep]
    if len(mz) > 256:
        strongest = np.argsort(intensity)[-256:]
        mz, intensity = mz[strongest], intensity[strongest]
    order = np.argsort(mz)
    return mz[order], intensity[order] / intensity.sum()


def match_peaks(reference, spectrum):
    masses, intensities = spectrum
    if not len(masses):
        return np.zeros(len(reference))
    high = np.minimum(np.searchsorted(masses, reference), len(masses) - 1)
    low = np.maximum(high - 1, 0)
    best = np.where(np.abs(masses[high] - reference) < np.abs(masses[low] - reference), high, low)
    error = np.abs(masses[best] - reference)
    within = np.flatnonzero(error <= np.maximum(0.01, reference * 20e-6))
    # A predicted peak can support at most one measured peak.
    nearest_first = within[np.argsort(error[within], kind="stable")]
    _, nearest = np.unique(best[nearest_first], return_index=True)
    matched = nearest_first[nearest]
    result = np.zeros(len(reference))
    result[matched] = intensities[best[matched]]
    return result


def entropy(ice, query, prediction):
    qm, qi = query
    pm, pi = prediction
    weighted_query = ice._weights(qi, 1.0, True)
    weighted_prediction = ice._weights(pi, 1.0, True)
    return ice.entropy_sim(qm, weighted_query, pm, weighted_prediction, 0.01, 20.0)


def cosine(query, prediction):
    return np.sqrt(query[1]) @ np.sqrt(match_peaks(query[0], prediction))


def differences(rows, queries, predictions):
    by_energy = {}
    for row in rows:
        by_energy.setdefault(float(row.collision_energy), []).append(row.query_id)
    energy = sorted(by_energy)
    names = list(predictions)
    result = {name: [] for name in names}
    for low, high in zip(energy, energy[1:]):
        low_ids, high_ids = by_energy[low], by_energy[high]
        query_ids = low_ids + high_ids
        all_masses = np.sort(np.concatenate([queries[q][0] for q in query_ids]))
        if not len(all_masses):
            continue
        anchors = [all_masses[0]]
        for mass in all_masses[1:]:
            if mass - anchors[-1] > max(0.01, mass * 20e-6):
                anchors.append(mass)
        anchors = np.asarray(anchors)
        observed = np.sqrt(np.stack([match_peaks(anchors, queries[q]) for q in query_ids]))
        observed_delta = observed[len(low_ids):].mean(axis=0) - observed[:len(low_ids)].mean(axis=0)
        valid_names, deltas = [], []
        for name in names:
            p_low, p_high = predictions[name].get(low), predictions[name].get(high)
            if p_low is None or p_high is None:
                continue
            valid_names.append(name)
            deltas.append(np.sqrt(match_peaks(anchors, p_high)) - np.sqrt(match_peaks(anchors, p_low)))
        if len(deltas) < 2:
            continue
        delta = np.stack(deltas)
        # Emphasize measured fragment masses for which candidates predict different responses.
        variation = delta.std(axis=0)
        scale = np.quantile(variation[variation > 0], 0.75) if variation.max() else 1.0
        weight = 1.0 + np.clip(variation / scale, 0.0, 2.0)
        delta *= np.sqrt(weight)
        observed_delta *= np.sqrt(weight)
        denom = np.linalg.norm(delta, axis=1) * np.linalg.norm(observed_delta)
        score = np.divide(delta @ observed_delta, denom, out=np.zeros(len(delta)), where=denom > 1e-12)
        for name, value in zip(valid_names, score):
            result[name].append(float(value))
    return result


def measured_queries(queries, adducts, require_reported_ev):
    queries = queries.assign(collision_energy=pd.to_numeric(queries["collision_energy"], errors="coerce"))
    eligible = queries["adduct"].isin(adducts) & queries["collision_energy"].between(0, 100)
    if require_reported_ev:
        eligible &= queries["collision_energy_orig_units"].eq("eV")
    return queries[eligible]


def measurement_groups(queries):
    groups = ["adduct", "instrument_type"]
    groups.extend(field for field in ("ingest_lib", "source_library", "instrument_name") if field in queries)
    return groups


def rank_opportunities(queries, candidates, labels):
    measured = measured_queries(queries, ("[M+H]+", "[M+Na]+"), require_reported_ev=True)
    affected = set(measured["molecule_id"])
    energy_counts = measured.groupby(["molecule_id", *measurement_groups(measured)])["collision_energy"].nunique()
    has_multi = set(energy_counts[energy_counts.ge(2)].index.get_level_values("molecule_id"))
    top25 = candidates[candidates["rank"].between(1, 25)].sort_values("rank", kind="stable")
    unique = top25.drop_duplicates(["molecule_id", "candidate_id"])
    first_same_formula = unique.groupby(["molecule_id", "formula"])["rank"].agg(["min", "size"])
    result = labels.merge(unique[["molecule_id", "candidate_id", "formula", "rank"]],
                          on=["molecule_id", "candidate_id"], how="left")
    result = result.join(first_same_formula, on=["molecule_id", "formula"])
    reported = result["molecule_id"].isin(affected)
    result["can_improve"] = reported & result["min"].lt(result["rank"])
    result["can_reach_first"] = result["can_improve"] & result["min"].eq(1)
    result["can_lose_first"] = reported & result["rank"].eq(1) & result["size"].gt(1)
    result["has_reported_ev"] = reported
    result["has_distinct_ev"] = result["molecule_id"].isin(has_multi)
    result["rr"] = (1 / result["rank"]).fillna(0)
    result["upper_delta_rr"] = (1 / result["min"] - result["rr"]).where(result["can_improve"], 0)
    cohorts = [("all", result)]
    if "spectrum_group" in result:
        cohorts.append(("same_source_multi_ce", result[result["spectrum_group"].eq("same_source_multi_ce")]))
    return pd.DataFrame([
        dict(cohort=name, molecules=len(group), measured_ev=int(group["has_reported_ev"].sum()),
             distinct_measured_ev=int(group["has_distinct_ev"].sum()),
             can_improve=int(group["can_improve"].sum()), can_reach_first=int(group["can_reach_first"].sum()),
             can_lose_first=int(group["can_lose_first"].sum()), mrr=group["rr"].mean(),
             upper_delta_mrr=group["upper_delta_rr"].mean())
        for name, group in cohorts
    ])


def paired_ce_summary(measured):
    results = []
    for cohort, group in measured.groupby("cohort", sort=False):
        ranks = group.pivot(index="molecule_id", columns="model", values="after")
        fixed, observed = ranks["ice_fixed_ce_cosine"], ranks["ice_observed_ce_cosine"]
        fixed_rr, observed_rr = (1 / fixed).fillna(0), (1 / observed).fillna(0)
        results.append(dict(cohort=cohort, molecules=len(ranks), fixed_ce_mrr=fixed_rr.mean(),
                            observed_ce_mrr=observed_rr.mean(), delta_mrr=(observed_rr - fixed_rr).mean(),
                            moved_to_first=int((fixed.ne(1) & observed.eq(1)).sum()),
                            lost_first=int((fixed.eq(1) & observed.ne(1)).sum())))
    return pd.DataFrame(results)


def score_panel(args, ice, predictor, device):
    queries = pd.read_parquet(args.queries)
    candidates = pd.read_parquet(args.candidates).sort_values("rank", kind="stable")
    queries = measured_queries(queries, ice.COV, args.measured_ev_only)
    if args.limit:
        queries = queries[queries["molecule_id"].isin(queries["molecule_id"].unique()[:args.limit])]
    candidate_groups = dict(tuple(candidates.groupby("molecule_id", sort=False)))
    scores = []
    coverage = Counter()
    start = time.monotonic()
    groups = measurement_groups(queries)
    eligible = []
    jobs = set()
    smiles_supported = {}
    for molecule_id, group in queries.groupby("molecule_id", sort=False):
        shortlist = candidate_groups.get(molecule_id)
        if shortlist is None:
            continue
        formula_groups = []
        for _, alternatives in shortlist.head(args.top_k).groupby("formula", sort=False):
            if len(alternatives) < 2:
                continue
            items = list(alternatives.itertuples(index=False))
            for c in items:
                if c.smiles not in smiles_supported:
                    smiles_supported[c.smiles] = ice.smiles_ok(c.smiles)[0]
            if all(smiles_supported[c.smiles] for c in items):
                formula_groups.append(items)
        if not formula_groups:
            continue
        kept = [candidate for alternatives in formula_groups for candidate in alternatives]
        for group_key, spectra in group.groupby(groups, sort=False, dropna=False):
            adduct, raw_instrument = group_key[:2]
            instrument = ice.instr_token({"instrument": raw_instrument})
            rows = list(spectra.itertuples(index=False))
            actual_energy = set(float(row.collision_energy) for row in rows)
            fixed_energy = float(np.median(sorted(actual_energy)))
            # Use the same candidates and measured spectra for the fixed-CE and observed-CE scores.
            energies = actual_energy | {fixed_energy}
            jobs.update((c.smiles, adduct, instrument, e) for c in kept for e in energies)
            eligible.append((molecule_id, adduct, instrument, rows, formula_groups, energies, fixed_energy))

    print(f"Predicting {len(jobs)} candidate conditions for {len(eligible)} measurement groups on {device}", flush=True)
    predictor.predict(jobs, float("inf"))
    for molecule_id, adduct, instrument, rows, formula_groups, energies, fixed_energy in eligible:
        query_peaks = {
            row.query_id: clean_peaks(row.mz, row.intensity, float(row.precursor_mz)) for row in rows
        }
        # Across-energy comparisons use a common precursor exclusion for the entire group.
        precursor = min(float(row.precursor_mz) for row in rows)
        for alternatives in formula_groups:
            predicted = {}
            for c in alternatives:
                predicted[c.candidate_id] = {}
                for energy in energies:
                    peaks = predictor.cache.get((c.smiles, adduct, instrument, energy))
                    if peaks is not None:
                        peaks = clean_peaks(*peaks, precursor)
                        if len(peaks[0]):
                            predicted[c.candidate_id][energy] = peaks
            delta = {} if args.cosine_only else differences(rows, query_peaks, predicted)
            for c in alternatives:
                changes = delta.get(c.candidate_id, ())
                delta_score = float(np.mean(changes)) if changes else None
                if delta_score is not None:
                    scores.append((molecule_id, None, c.candidate_id, "ice_ce_fragment_delta", delta_score))
                for row in rows:
                    query = query_peaks[row.query_id]
                    p = predicted[c.candidate_id].get(float(row.collision_energy))
                    fixed = predicted[c.candidate_id].get(fixed_energy)
                    if not len(query[0]) or p is None or fixed is None:
                        continue
                    context = molecule_id, row.query_id, c.candidate_id
                    scores.append((*context, "ice_observed_ce_cosine", cosine(query, p)))
                    if args.cosine_only:
                        scores.append((*context, "ice_fixed_ce_cosine", cosine(query, fixed)))
                    else:
                        observed_entropy = entropy(ice, query, p)
                        scores.append((*context, "ice_fixed_ce_entropy", entropy(ice, query, fixed)))
                        scores.append((*context, "ice_observed_ce_entropy", observed_entropy))
                        scores.append((*context, "ice_ce_entropy_plus_delta", observed_entropy + DELTA_WEIGHT * (delta_score or 0.0)))
                    coverage[f"{instrument}/{adduct}"] += 1

    frame = pd.DataFrame(scores, columns=["molecule_id", "query_id", "candidate_id", "model", "score"])
    if len(frame):
        frame = frame.groupby(["molecule_id", "query_id", "candidate_id", "model"], dropna=False, sort=False, as_index=False)["score"].mean()
    frame.to_parquet(args.output, index=False)
    print(f"Prediction coverage by instrument/adduct {dict(coverage)}", flush=True)
    print(frame.groupby("model").agg(scores=("score", "size"), molecules=("molecule_id", "nunique")), flush=True)
    print(f"device={device} predictions={predictor.n_pred} failures={predictor.n_fail} seconds={time.monotonic() - start:.1f}", flush=True)
    if frame.empty:
        raise RuntimeError("ICEBERG produced no usable candidate scores")


def measure_energy_response(ice, predictor):
    smiles = ["CC(=O)Oc1ccccc1C(=O)O", "COc1ccc(C=CC(=O)O)cc1O", "CC(=O)Nc1ccc(O)cc1"]
    energies = (15.0, 35.0, 55.0)
    jobs = {(s, "[M+H]+", "QTOF", ce) for s in smiles for ce in energies}
    predictor.predict(jobs, float("inf"))
    response = []
    for s in smiles:
        peaks = [predictor.cache.get((s, "[M+H]+", "QTOF", ce)) for ce in energies]
        if any(p is None for p in peaks):
            raise RuntimeError(f"ICEBERG did not predict all three collision energies for {s}")
        print(f"{s}: peaks at 15/35/55 eV {[len(p[0]) for p in peaks]}", flush=True)
        comparison = [entropy(ice, peaks[i], peaks[j]) for i, j in [(0, 0), (0, 1), (0, 2), (1, 2)]]
        response.append(comparison)
    print("Spectral entropy similarity for 15/15, 15/35, 15/55, 35/55 eV", flush=True)
    print(np.asarray(response).round(6), flush=True)
    pd.DataFrame(response, index=smiles, columns=["15_15", "15_35", "15_55", "35_55"]).to_csv("iceberg_energy_response.csv", index_label="smiles")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score candidate fragments with matched collision energies")
    parser.add_argument("--inputs", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=Path("scores.parquet"))
    parser.add_argument("--cosine-only", action="store_true")
    parser.add_argument("--measured-ev-only", action="store_true")
    parser.add_argument("--measure-response", action="store_true")
    args, _ = parser.parse_known_args(argv)
    if not args.measure_response:
        for name in ("queries", "candidates"):
            if getattr(args, name) is None:
                paths = list(args.inputs.rglob(f"{name}.parquet"))
                if len(paths) != 1:
                    raise FileNotFoundError(f"Select --{name} explicitly. Mounted {name}.parquet files: {paths}")
                setattr(args, name, paths[0])
    ice, predictor, device = load_iceberg(args.inputs, args.device, args.batch_size)
    if args.measure_response:
        measure_energy_response(ice, predictor)
        return
    score_panel(args, ice, predictor, device)


if __name__ == "__main__":
    main()

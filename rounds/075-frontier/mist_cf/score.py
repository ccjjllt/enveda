"""Compare candidate formula rankings from observed spectra on Kaggle CPU."""

import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from time import monotonic
from urllib.request import urlopen

import numpy as np
import pandas as pd
import torch


INPUT = Path('/kaggle/input')
OUTPUT = Path('/kaggle/working')
CHECKPOINT_URL = 'https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/resolve/main/models/mist_cf_msg.ckpt'
CHECKPOINT_SHA = '53345744860a92ed2c3cb4a9326313d1d50d38011e178f33e02a12ac0dbea729'
SOURCE_URL = 'https://codeload.github.com/samgoldman97/mist-cf/tar.gz/refs/tags/v1.0.0'
ADDUCT_ALIASES = {'M+H': '[M+H]+', '[M+NH4]+': '[M+H3N+H]+',
                 '[M+H-H2O]+': '[M-H2O+H]+', '[M-2H2O+H]+': '[M-H4O2+H]+'}


def mounted(notebook, filename):
    paths = [p for p in INPUT.rglob(filename) if notebook in p.parts]
    if len(paths) != 1:
        raise FileNotFoundError(f'Expected {notebook}/{filename}, found {paths}')
    return paths[0]


def download(url, destination):
    with urlopen(url, timeout=60) as response:
        destination.write_bytes(response.read())


def load_models(work):
    upstream = work / 'mist-cf.tgz'
    checkpoint = work / 'mist_cf_msg.ckpt'
    download(SOURCE_URL, upstream)
    download(CHECKPOINT_URL, checkpoint)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA:
        raise RuntimeError('Unexpected MassSpecGym MIST-CF weights')
    with tarfile.open(upstream) as archive:
        archive.extractall(work, (m for m in archive.getmembers() if '/src/mist_cf/' in m.name), filter='data')
    src, = work.glob('mist-cf-*/src')
    # The two chemistry helpers suffice for inference. The original package also imports training utilities.
    (src / 'mist_cf/common/__init__.py').write_text('from .chem_utils import *\nfrom .parse_utils import *\n')
    sys.path.insert(0, str(src))
    with tarfile.open(mounted('enveda-075b-flare-mist-assets', 'mist.tgz')) as archive:
        flare_helpers = work / 'flare_helpers'
        archive.extractall(flare_helpers, filter='data')
    sys.path.insert(0, str(flare_helpers / 'src'))
    abi = f'cp{sys.version_info.major}{sys.version_info.minor}'
    wheels = list(INPUT.rglob(f'rdkit-2026.3.3-{abi}-*.whl'))
    if not wheels:
        raise FileNotFoundError(f'RDKit 2026.03.3 wheel is missing for {abi}')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-index',
                    '--no-deps', str(wheels[0])], check=True)
    try:
        import pytorch_lightning
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'pytorch-lightning==2.6.6'], check=True)
    from mist_cf.mist_cf_score.mist_cf_model import MistNet

    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model_args = dict(hidden_size=128, layers=2, dropout=0.1, max_subpeak=20,
                      ion_info=True, instrument_info=True, cls_mass_diff=False, form_encoder='abs-sines')
    if 'state_dict' in state:
        model_args.update({k: v for k, v in state.get('hyper_parameters', {}).items() if k in model_args})
        state = state['state_dict']
    model = MistNet(**model_args)
    model.load_state_dict(state, strict=True)
    print(f'MIST-CF checkpoint SHA256 {CHECKPOINT_SHA}, settings {model_args}', flush=True)
    return model.eval()


@lru_cache(maxsize=4096)
def formula_mass(formula):
    from mist.utils.chem_utils import formula_to_dense, VALID_MONO_MASSES
    from mist_cf.common import formula_to_dense as cf_dense

    try:
        counts = formula_to_dense(formula)
        cf_dense(formula)
    except (ValueError, KeyError):
        return None
    # FLARE's formula product is fully materialized in the existing annotator.
    count = 1
    for element in counts:
        count *= int(element) + 1
    return float(counts @ VALID_MONO_MASSES) if count <= 200_000 else None


def eligible_formulas(query, formulas):
    from mist.utils.chem_utils import ion_to_mass
    from mist_cf.common import ION_LST

    ion = ADDUCT_ALIASES.get(query.adduct, query.adduct)
    precursor = float(query.precursor_mz)
    if ion not in ion_to_mass or ion not in ION_LST or not 0 < precursor <= 1000:
        return ion, []
    hypotheses = []
    for formula in formulas:
        mass = formula_mass(formula)
        if mass is not None:
            ppm = abs(mass + ion_to_mass[ion] - precursor) * 1e6 / precursor
            if ppm <= 20.0:
                hypotheses.append((ppm, formula))
    return ion, sorted(hypotheses)[:8]


def peaks_for_flare(query):
    mz, intens = np.asarray(query.mz, dtype=float), np.asarray(query.intensity, dtype=float)
    valid = np.isfinite(mz) & np.isfinite(intens) & (mz > 0) & (intens > 0)
    valid &= mz <= float(query.precursor_mz) + 1.0
    mz, intens = mz[valid], intens[valid]
    if not len(mz):
        return np.empty((0, 2))
    order = np.argsort(-intens, kind='stable')
    mz, intens = mz[order], intens[order]
    _, first = np.unique(np.round(mz, 4), return_index=True)
    selected = np.sort(first)[:60]
    peaks = np.column_stack([mz[selected], np.sqrt(intens[selected] / intens[0])])
    return peaks[peaks[:, 1] > 0.001]


def flare_formula_scores(peaks, ion, hypotheses):
    from mist.data.featurizers import assign_subformulae

    peak_table = {'mz': peaks[:, 0], 'ms2_inten': peaks[:, 1]}
    explained_denominator = float(peaks[:, 1].sum())
    scores = {}
    for ppm, formula in hypotheses:
        assigned = assign_subformulae({'cand_form': formula, 'cand_ion': ion, 'output_tbl': peak_table})
        if assigned['output_tbl'] is not None:
            explained = sum(assigned['output_tbl']['ms2_inten']) / explained_denominator
            scores[formula] = explained - 0.20 * (ppm / 20.0) ** 2
    return scores


def instrument_name(raw, known):
    if raw in known:
        return raw
    value = str(raw).lower()
    if 'orbitrap' in value or 'exactive' in value:
        return 'Orbitrap (LCMS)'
    if 'tof' in value:
        return 'Q-ToF (LCMS)'
    if 'fticr' in value or 'ft-icr' in value:
        return 'FTICR (LCMS)'
    if 'trap' in value:
        return 'Ion Trap (LCMS)'
    return 'Unknown (LCMS)'


@torch.inference_mode()
def mist_cf_formula_scores(model, extractor, query, peaks, ion, hypotheses):
    from mist_cf import common
    from mist_cf.mist_cf_score.mist_cf_data import PredDataset

    # The published MIST-CF preprocessing uses relative raw intensities and its strongest 20 peaks.
    spectrum = common.max_thresh_spec(np.column_stack([peaks[:, 0], peaks[:, 1] ** 2]),
                                      max_peaks=model.max_subpeak, inten_thresh=0.003)
    instrument = instrument_name(query.instrument_type, common.instrument_to_type)
    batch = defaultdict(list)
    for _, formula in hypotheses:
        spec = {'cand_ion': ion, 'cand_tbl': None}
        if len(spectrum):
            assigned = common.assign_subforms(formula, spectrum, ion, common.get_instr_tol(instrument))
            spec['cand_tbl'] = assigned['output_tbl']
        features = extractor.extract_json(spec, formula, ion, float(query.precursor_mz), instrument,
                                          ablate_cls_error=not model.cls_mass_diff)
        batch['name'].append(query.query_id)
        batch['formula'].append(formula)
        batch['ion'].append(ion)
        batch['instrument'].append(instrument)
        batch['parentmass'].append(float(query.precursor_mz))
        batch['peak_type'].append(features.pop('peak_types'))
        for key, value in features.items():
            batch[key].append(value)
    batch = PredDataset.collate_fn([batch])
    cols = ['num_peaks', 'types', 'form_vec', 'ion_vec', 'instrument_vec', 'intens', 'rel_mass_diffs']
    scores = model(*(batch[key] for key in cols)).reshape(-1).numpy()
    return dict(zip(batch['str_forms'], map(float, scores)))


def predict_formulas(model, queries, candidates):
    from mist_cf.mist_cf_score.mist_cf_data import JsonExtractor

    extractor = JsonExtractor(max_subpeak=model.max_subpeak)
    formulas = candidates.groupby('molecule_id')['formula'].agg(lambda s: s.dropna().unique()).to_dict()
    rows, hypotheses_by_query = [], []
    for query in queries.itertuples(index=False):
        peaks = peaks_for_flare(query)
        ion, hypotheses = eligible_formulas(query, formulas.get(query.molecule_id, ()))
        if not len(peaks) or not hypotheses:
            continue
        ppm_by_formula = {f: ppm for ppm, f in hypotheses}
        hypotheses_by_query.append((query.query_id, tuple(ppm_by_formula)))
        flare = flare_formula_scores(peaks, ion, hypotheses)
        mist_cf = mist_cf_formula_scores(model, extractor, query, peaks, ion, hypotheses)
        for method, scores in [('flare_fragment_coverage', flare), ('mist_cf', mist_cf)]:
            ordered = sorted(scores, key=lambda formula: (-scores[formula], ppm_by_formula[formula], formula))
            rows.extend((query.query_id, query.molecule_id, method, formula, rank, scores[formula])
                        for rank, formula in enumerate(ordered, 1))
    rankings = pd.DataFrame(rows, columns=['query_id', 'molecule_id', 'method', 'formula', 'rank', 'score'])
    shortlist = pd.DataFrame(hypotheses_by_query, columns=['query_id', 'shortlist'])
    return rankings, shortlist


def evaluate(queries, candidates, rankings, shortlist, labels):
    truth = labels.merge(candidates[['molecule_id', 'candidate_id', 'formula']], on=['molecule_id', 'candidate_id'])
    truth = truth[['molecule_id', 'formula']].drop_duplicates('molecule_id')
    labeled = queries[['query_id', 'molecule_id']].merge(truth, on='molecule_id').merge(shortlist, on='query_id', how='left')
    truth_ranks = rankings.merge(labeled[['query_id', 'formula']], on=['query_id', 'formula'])
    methods = ['flare_fragment_coverage', 'mist_cf']
    predicted = {m: set(rankings.loc[rankings.method.eq(m), 'query_id']) for m in methods}
    paired = set.intersection(*(predicted[m] for m in methods))
    contested = labeled.loc[labeled.shortlist.map(lambda x: len(x) > 1 if isinstance(x, tuple) else False), 'query_id']
    candidate_oracle = sum(isinstance(r.shortlist, tuple) and r.formula in r.shortlist for r in labeled.itertuples())
    print(f'{len(queries)} input spectra, {len(labeled)} spectra with truth identity in candidates, '
          f'{candidate_oracle} with truth formula in the shared shortlist', flush=True)
    results = []
    for cohort, ids in [('truth_in_pool', set(labeled.query_id)), ('paired', set(labeled.query_id) & paired),
                        ('paired_contested', set(contested) & paired)]:
        for method in methods:
            ranks = truth_ranks.loc[truth_ranks.method.eq(method) & truth_ranks.query_id.isin(ids), 'rank']
            results.append((cohort, method, len(ids), len(ids & predicted[method]), int(ranks.eq(1).sum()), int(ranks.le(5).sum())))
    return pd.DataFrame(results, columns=['cohort', 'method', 'spectra', 'predicted', 'top1', 'top5'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path)
    parser.add_argument('--candidates', type=Path)
    parser.add_argument('--labels', type=Path)
    args, _ = parser.parse_known_args()
    queries = pd.read_parquet(args.queries or mounted('enveda-075b-queries', 'queries.parquet'))
    candidates = pd.read_parquet(args.candidates or mounted('enveda-075b-pipeline', 'candidates.parquet'))
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='mist-cf-') as work:
        model = load_models(Path(work))
        started = monotonic()
        rankings, shortlist = predict_formulas(model, queries, candidates)
        print(f'Scored {len(shortlist)} spectra in {monotonic() - started:.2f}s on CPU', flush=True)
        labels = pd.read_parquet(args.labels or mounted('enveda-075b-queries', 'labels.parquet'))
        comparison = evaluate(queries, candidates, rankings, shortlist, labels)
    rankings.to_parquet(OUTPUT / 'formula_scores.parquet', index=False)
    comparison.to_csv(OUTPUT / 'formula_comparison.csv', index=False)
    print(comparison.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()

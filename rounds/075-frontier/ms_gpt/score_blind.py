"""Run two equal-budget MS-GPT arms on blind spectra and native C34 formula hypotheses."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
from itertools import zip_longest
from pathlib import Path
from time import monotonic

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize


SPECTRUM_COLUMNS = ('molecule_id', 'spectrum_id', 'precursor_mz', 'adduct', 'ionization_mode',
                    'instrument_type', 'ms2_mzs', 'ms2_normalized_intensities')
FORMULA_LIMIT = 5
QUERY_BUDGET = 20
DRAWS_PER_QUERY = 5
CANDIDATE_LIMIT = 25
MASS_PPM = 10
FRAGMENT_PPM = 20.0


def select_observations(path, supported):
    """Use one unchanged measured spectrum per molecule, consistently in both arms."""
    all_rows = pq.read_table(path, columns=list(SPECTRUM_COLUMNS)).to_pylist()
    groups = defaultdict(list)
    for row in all_rows:
        groups[row['molecule_id']].append(row)
    selected = {}
    for mid, spectra in groups.items():
        eligible = []
        for row in spectra:
            if row['ionization_mode'] != 'positive' or row['adduct'] not in supported:
                continue
            mz = np.asarray(row['ms2_mzs'], dtype='<f8')
            intensity = np.asarray(row['ms2_normalized_intensities'], dtype='<f8')
            precursor = row['precursor_mz']
            if not (np.isfinite(precursor) and precursor > 0 and mz.shape == intensity.shape):
                raise ValueError(f'Malformed supported spectrum in blind query {mid}.')
            valid = (np.isfinite(mz) & np.isfinite(intensity) & (mz > 0)
                     & (mz <= precursor + 1) & (intensity > 0))
            peaks = np.column_stack((mz[valid], intensity[valid]))
            if not len(peaks):
                continue
            peaks[:, 1] /= peaks[:, 1].max()
            peaks = peaks[np.argsort(-peaks[:, 1], kind='stable')[:60]]
            spectrum_sha = sha256(np.asarray(precursor, dtype='<f8').tobytes()
                                  + mz.tobytes() + intensity.tobytes()).hexdigest()
            eligible.append((int(np.count_nonzero(valid)), row, peaks, spectrum_sha))
        selected[mid] = max(eligible, key=lambda entry: entry[0], default=None)
    return selected


def mass_window(target):
    tolerance = target * MASS_PPM * 1e-6
    return target - tolerance, target + tolerance


def c34_formulas(record, elements):
    """Preserve the first native candidate rank of each modeled neutral formula."""
    lo, hi = mass_window(record['target'])
    seen, accepted = set(), []
    for position, text in enumerate(record['pc'], 1):
        mol = Chem.MolFromSmiles(text)
        if mol is None or Chem.GetFormalCharge(mol) != 0 or len(Chem.GetMolFrags(mol)) != 1:
            continue
        if any(atom.GetIsotope() or atom.GetSymbol() not in elements for atom in mol.GetAtoms()):
            continue
        mass = rdMolDescriptors.CalcExactMolWt(mol)
        if not lo <= mass <= hi:
            continue
        formula = rdMolDescriptors.CalcMolFormula(mol)
        if formula in seen:
            continue
        seen.add(formula)
        accepted.append(dict(formula=formula, c34_position=position, candidate_mass=mass))
        if len(accepted) == FORMULA_LIMIT:
            break
    return accepted


def assign_fragments(peaks, formula, adduct):
    from mist.data.featurizers import assign_subformulae

    result = assign_subformulae(
        {'cand_form': formula, 'cand_ion': adduct,
         'output_tbl': {'mz': peaks[:, 0], 'ms2_inten': peaks[:, 1]}},
        mass_diff_thresh=FRAGMENT_PPM,
    )['output_tbl']
    return {'formula': result['formula'], 'inten': result['ms2_inten']} if result else {'formula': [], 'inten': []}


def molecule_identity(mol, tautomer):
    canonical = Chem.Mol(mol)
    Chem.RemoveStereochemistry(canonical)
    canonical = tautomer.Canonicalize(canonical)
    key = Chem.MolToInchiKey(canonical)[:14]
    atoms = Counter(atom.GetSymbol() for atom in canonical.GetAtoms() if atom.GetAtomicNum() > 1)
    return key, ','.join(f'{symbol}:{n}' for symbol, n in sorted(atoms.items()))


def generation_arm(mid, forms, posteriors, context):
    import torch
    from ms_gpt.data.mlm_pretraining_dataset import formula_str_to_vec
    from ms_gpt.data.posterior_queries import construct_queries
    from ms_gpt.pipeline.postprocessing import CandidateRanker, candidate_record

    per_formula = []
    base, remainder = divmod(QUERY_BUDGET, len(forms))
    for index, (entry, posterior) in enumerate(zip(forms, posteriors, strict=True)):
        form = entry['formula']
        query_count = base + (index < remainder)
        queries = construct_queries('density_calibrated_band', posterior, num_queries=query_count,
                                    threshold_low=context['band'][0], threshold_high=context['band'][1])
        if len(queries) != query_count:
            raise RuntimeError('Posterior querying changed the fixed draw count.')
        formula_vectors = np.repeat(np.stack([formula_str_to_vec(form)]), len(queries), axis=0)
        seed = int.from_bytes(sha256(f'{context["config"]["seed"]}:{mid}:{form}'.encode()).digest()[:8], 'big')
        torch.manual_seed(seed)
        sequences = []
        tokenizer = context['tokenizer']
        for offset in range(0, query_count, 4):
            limit = offset + 4
            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
                sampled = context['model'].generate(
                    fingerprint=torch.as_tensor(queries[offset:limit], device=context['device']),
                    formula_vec=torch.as_tensor(formula_vectors[offset:limit], device=context['device']),
                    max_len=context['cfg'].model.max_seq_len, temperature=context['config']['temperature'],
                    top_k=context['config']['top_k'], top_p=context['config']['top_p'],
                    bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                    num_samples=DRAWS_PER_QUERY,
                ).cpu().tolist()
            for item in sampled:
                end = item.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in item else len(item)
                sequences.append(tokenizer.decode(item[:end], skip_special_tokens=True))
        if len(sequences) != query_count * DRAWS_PER_QUERY:
            raise RuntimeError('Decoder did not return every requested draw.')
        ranked = CandidateRanker(posterior, context['band']).rank(sequences, form, top_k=CANDIDATE_LIMIT)
        proposals = []
        for mol in ranked:
            if Chem.GetFormalCharge(mol) != 0 or len(Chem.GetMolFrags(mol)) != 1:
                continue
            mass = rdMolDescriptors.CalcExactMolWt(mol)
            if not context['window'][0] <= mass <= context['window'][1]:
                continue
            key, heavy = molecule_identity(mol, context['tautomer'])
            if key:
                proposals.append(dict(score_key=key, heavy_atoms=heavy, mass=mass,
                                      formula=form, **candidate_record(mol)))
        per_formula.append(dict(**entry, draws=len(sequences), retained_before_merge=len(proposals),
                                proposals=proposals))
    if sum(row['draws'] for row in per_formula) != QUERY_BUDGET * DRAWS_PER_QUERY:
        raise RuntimeError('Arms did not share the same 100-draw budget.')
    return per_formula


def merge_formulas(predictions):
    seen, selected = set(), []
    for block in zip_longest(*(f['proposals'] for f in predictions)):
        for row in block:
            if row is None:
                continue
            identity = (row['score_key'], row['heavy_atoms'])
            if identity not in seen:
                selected.append(row)
                seen.add(identity)
            if len(selected) == CANDIDATE_LIMIT:
                return selected
    return selected


def run(queries, c34, out, model, mist, tokenizer, cfg, config, band, device, provenance):
    import torch
    from ms_gpt.data.mist_inputs import ELEMENTS as MIST_ELEMENTS, ION_INDEX, prepare_mist_batch
    from ms_gpt.data.mlm_pretraining_dataset import ELEMENTS as DECODER_ELEMENTS
    from ms_gpt.external.mist_official.chem_utils import instrument_to_type, instruments

    RDLogger.DisableLog('rdApp.*')
    tautomer = rdMolStandardize.TautomerEnumerator()
    observed = select_observations(queries, ION_INDEX)
    native = json.loads(c34.read_text())['molecules']
    if len(observed) != 60 or set(observed) != set(native):
        raise ValueError('Blind spectra and native lists must cover the same 60 queries.')
    if config['max_fragment_peaks'] != 10 or config['num_queries'] * config['samples_per_query'] != 100:
        raise ValueError('Unexpected released MSG sampling protocol.')
    allowed_elements = set(MIST_ELEMENTS) & set(DECODER_ELEMENTS)
    result = out / 'ms_gpt_blind_predictions.jsonl'
    tmp = result.with_suffix('.tmp')
    counts, available = Counter(), Counter()
    start = monotonic()
    with tmp.open('w') as writer:
        for order, (mid, selection) in enumerate(observed.items(), 1):
            record = dict(molecule_id=mid, top1=[], top5=[])
            if selection is None:
                record['status'] = 'no_supported_spectrum'
            else:
                forms = c34_formulas(native[mid], allowed_elements)
                available[len(forms)] += 1
                if not forms:
                    record['status'] = 'no_eligible_c34_formula'
                else:
                    _, row, peaks, spectrum_sha = selection
                    instrument = instrument_to_type.get(row['instrument_type'], 'unknown')
                    features = [dict(formula=form['formula'], adduct=row['adduct'],
                                     peak_formula=assign_fragments(peaks, form['formula'], row['adduct']))
                                for form in forms]
                    for entry, feature in zip(forms, features, strict=True):
                        entry['annotated_peaks'] = len(feature['peak_formula']['formula'])
                    with torch.inference_mode():
                        inp = prepare_mist_batch(features, max_peaks=config['max_fragment_peaks'], device=device)
                        inp['batch_spec_instrument'] = torch.full(
                            (len(features),), instruments.index(instrument), dtype=torch.long, device=device,
                        )
                        posteriors = mist.predict(inp).cpu().numpy()
                    if posteriors.shape != (len(forms), 4096) or not np.isfinite(posteriors).all():
                        raise ValueError(f'Invalid fingerprint posterior for {mid}.')
                    context = dict(model=model, tokenizer=tokenizer, cfg=cfg, config=config,
                                   band=band, device=device, tautomer=tautomer,
                                   window=mass_window(native[mid]['target']))
                    one = generation_arm(mid, forms[:1], posteriors[:1], context)
                    five = generation_arm(mid, forms, posteriors, context) if len(forms) > 1 else one
                    counts['identical_single_formula_arms'] += len(forms) == 1
                    record.update(status='complete', spectrum_sha256=spectrum_sha, adduct=row['adduct'],
                                  instrument_type=row['instrument_type'], model_instrument=instrument)
                    for name, forecasts in (('top1', one), ('top5', five)):
                        record[name] = merge_formulas(forecasts)
                        record[name + '_formulas'] = [
                            {key: value for key, value in branch.items() if key != 'proposals'}
                            for branch in forecasts]
                        counts[name + '_draws'] += sum(f['draws'] for f in forecasts)
                        counts[name + '_candidates'] += len(record[name])
            counts[record['status']] += 1
            writer.write(json.dumps(record, allow_nan=False) + '\n')
            print(f'MS-GPT {order}/60 {mid} {record["status"]} '
                  f'final {len(record["top1"])}/{len(record["top5"])}', flush=True)
    expected = counts['complete'] * QUERY_BUDGET * DRAWS_PER_QUERY
    if not expected or counts['top1_draws'] != expected or counts['top5_draws'] != expected:
        raise ValueError('No predictions or unequal decoding totals.')
    tmp.replace(result)
    seal = dict(**provenance, all_queries=60, active_queries=counts['complete'],
                per_active_query_draws=QUERY_BUDGET * DRAWS_PER_QUERY, candidate_cap=CANDIDATE_LIMIT,
                formula_supply=dict(available), counts=dict(counts), elapsed_s=monotonic() - start,
                output_sha256=sha256(result.read_bytes()).hexdigest())
    (out / 'ms_gpt_blind_receipt.json').write_text(json.dumps(seal, indent=2) + '\n')

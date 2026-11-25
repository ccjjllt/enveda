"""Score a sealed Pablo 0.506 candidate list with the released MSAlign Morgan heads."""

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from zipfile import ZipFile

import numpy as np
import pyarrow.parquet as pq
import torch
from rdkit import rdBase
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import yaml


BASE_SOURCE = 'pablodiegoo/enveda-casmi26-starter-notebook/29'
CAP = 25
TABLE_COLUMNS = ('molecule_id', 'spectrum_id', 'ms2_mzs', 'ms2_normalized_intensities',
                 'precursor_mz', 'adduct', 'collision_energy_ev')
HEADS = {'bins_fingerprint': 'bins', 'dreams_fingerprint': 'dreams'}
DREAMS_SHA256 = 'df69866990b194b27d5497e440f87f6ceca568ee512e7c22cb1f278721376f0c'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def frozen_baseline(path, queries):
    with ZipFile(path) as files:
        receipt = json.loads(files.read('pilot_arm_receipt.json'))
        if (receipt['arm'] != 'baseline' or receipt['source'] != BASE_SOURCE
                or receipt['queries_sha256'] != digest(queries)
                or receipt['candidate_slots'] != CAP):
            raise ValueError('This is not the sealed Pablo baseline for these exact spectra')
        required = ('submission.csv', 'pilot_final_slots.json', 'npx_audit.json', 'e180x_audit.json')
        frozen = {name: files.read(name) for name in required}
        for name, raw in frozen.items():
            if hashlib.sha256(raw).hexdigest() != receipt['hashes'][name]:
                raise ValueError(f'Baseline file changed after sealing: {name}')
        for name in ('npx', 'e180x'):
            status = json.loads(frozen[f'{name}_audit.json'])['status']
            if receipt['stage_status'][name] != status or status.lower().startswith('failed'):
                raise ValueError(f'{name} did not complete the baseline replay: {status}')
    guesses = {}
    for row in csv.DictReader(io.StringIO(frozen['submission.csv'].decode())):
        mid, smiles = row['molecule_id'], row['smiles']
        candidates = smiles.split(';') if smiles else []
        if mid in guesses or len(candidates) > CAP or any(not candidate for candidate in candidates):
            raise ValueError(f'Invalid frozen candidate list for {mid}')
        guesses[mid] = candidates
    vacant = {mid: CAP - len(candidates) for mid, candidates in guesses.items() if len(candidates) < CAP}
    if vacant != receipt['vacant_slots_by_query']:
        raise ValueError('Final candidate slot accounting differs from the sealed audit')
    return receipt, guesses


def observed_spectrum(row):
    peaks = np.column_stack((row['ms2_mzs'], row['ms2_normalized_intensities']))
    peaks = peaks[np.argsort(peaks[:, 0], kind='stable')]
    energies = row['collision_energy_ev'] or []
    energy = energies[0] if len(energies) == 1 else None
    if energy is not None and not np.isfinite(energy):
        energy = None
    return dict(peaks=peaks, precursor_mz=row['precursor_mz'], adduct=row['adduct'],
                collision_energy=energy)


def pretrain_embedding_input(peaks, precursor):
    # pluskal-lab/DreaMS dbec3a0b SpectrumPreprocessor with default options.
    keep = np.sort(np.argsort(peaks[:, 1])[-100:])
    result = np.zeros((101, 2), dtype=np.float32)
    result[0] = (precursor, 1.1)
    selected = peaks[keep]
    result[1:len(keep) + 1] = selected
    result[1:len(keep) + 1, 1] /= selected[:, 1].max()
    return result


def load_heads(root, channels, device, implementation):
    manifest = yaml.safe_load((root / 'official_manifest.yaml').read_text())
    entries = {entry['name']: entry for entry in manifest['scorers']}
    result = {}
    for name in channels:
        entry = entries[name]
        path = root / entry['checkpoint']
        if digest(path) != entry['sha256']:
            raise ValueError(f'Released checkpoint checksum mismatch: {name}')
        model, config, dimensions = implementation._alignment_checkpoint(path, device)
        representations = config['representations']
        if (representations['spectrum'] != HEADS[name] or representations['molecule'] != 'morgan_2_4096'
                or dimensions['d_mol'] != 4096):
            raise ValueError(f'Wrong released checkpoint configuration: {name}')
        if name == 'dreams_fingerprint':
            width = 1024
        else:
            params = config['spectrum_bins']
            width = int(np.ceil(params['max_mz'] / params['bin_width']))
        if dimensions['d_ms'] != width:
            raise ValueError(f'Spectrum width for {name} is {dimensions["d_ms"]}, expected {width}')
        result[name] = model, config, dimensions
    return result


def write_csv(path, rows, columns):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def score(args):
    clock = time.monotonic()
    receipt, candidates = frozen_baseline(args.baseline, args.queries)
    rows = pq.read_table(args.queries, columns=list(TABLE_COLUMNS)).to_pylist()
    by_id = defaultdict(list)
    for row in rows:
        by_id[row['molecule_id']].append(row)
    if set(by_id) != set(candidates) or len(rows) != receipt['spectra']:
        raise ValueError('The sealed queries differ from the candidate lists')
    if len({row['spectrum_id'] for row in rows}) != len(rows):
        raise ValueError('Query spectrum IDs are not unique')

    sys.path.insert(0, str(args.source.resolve()))
    from playground import inference as implementation

    device = torch.device(args.device)
    heads = load_heads(args.checkpoints, args.channel, device, implementation)
    dreams = None
    if 'dreams_fingerprint' in heads:
        if args.dreams is None or digest(args.dreams) != DREAMS_SHA256:
            raise ValueError('DreaMS channel requires the authors’ checksum-verified TorchScript artifact')
        dreams = torch.jit.load(str(args.dreams), map_location=device).eval()
    aggregate, per_spectrum = [], []
    skipped = {name: {} for name in heads}
    usable = {name: {mid: 0 for mid in by_id} for name in heads}
    for mid, guesses in candidates.items():
        if not guesses:
            for name in heads:
                for row in by_id[mid]:
                    skipped[name][row['spectrum_id']] = dict(molecule_id=mid, reason='empty candidate list')
            continue
        _, molecules = implementation._canonical_smiles(guesses)
        morgan = torch.from_numpy(implementation._morgan_fingerprints(molecules)).to(device)
        repeated = defaultdict(list)
        mask = torch.ones((1, len(guesses)), dtype=torch.bool, device=device)
        formulas = [CalcMolFormula(molecule) for molecule in molecules]
        for row in by_id[mid]:
            spectrum = observed_spectrum(row)
            try:
                peaks, precursor, adduct, energy = implementation._validate_spectrum(spectrum)
            except ValueError as exc:
                for name in heads:
                    skipped[name][row['spectrum_id']] = dict(molecule_id=mid, reason=str(exc))
                continue
            values = {}
            for name, (model, config, dimensions) in heads.items():
                if name == 'dreams_fingerprint':
                    prepared = pretrain_embedding_input(peaks, precursor)
                    vector = dreams(torch.from_numpy(prepared).to(device)[None]).float()
                else:
                    params = config['spectrum_bins']
                    if not np.any((peaks[:, 0] <= params['max_mz']) & (peaks[:, 1] > 0)):
                        skipped[name][row['spectrum_id']] = dict(
                            molecule_id=mid, reason='no positive-intensity peak within native bins',
                            max_mz_inclusive=float(params['max_mz']))
                        continue
                    prepared = implementation._binned_spectrum(peaks, **params)
                    vector = torch.from_numpy(prepared).to(device)[None]
                if tuple(vector.shape) != (1, dimensions['d_ms']):
                    raise ValueError(f'{mid} {name} produced an incompatible spectral embedding')
                metadata = implementation.MSAlign._metadata(dimensions, adduct, energy, device)
                value = model(vector, morgan[None], metadata, mask)[0].cpu().numpy()
                if value.shape != (len(guesses),) or not np.isfinite(value).all():
                    raise ValueError(f'{mid} {name} produced nonfinite or misaligned scores')
                values[name] = value
                repeated[name].append(value)
                usable[name][mid] += 1
            if not values:
                continue
            for i, guess in enumerate(guesses):
                per_spectrum.append(dict(molecule_id=mid, spectrum_id=row['spectrum_id'],
                                         baseline_rank=i + 1, smiles=guess, adduct=adduct,
                                         collision_energy_ev=None if np.isnan(energy) else energy,
                                         **{name: scores[i] for name, scores in values.items()}))
        mean = {name: np.mean(scores, axis=0) for name, scores in repeated.items() if scores}
        for i, guess in enumerate(guesses):
            aggregate.append(dict(molecule_id=mid, baseline_rank=i + 1, smiles=guess,
                                  formula=formulas[i], spectra=len(by_id[mid]),
                                  **{name: scores[i] for name, scores in mean.items()}))

    args.out.mkdir(parents=True, exist_ok=True)
    aggregate_path = args.out / 'msalign_scores.csv'
    spectra_path = args.out / 'msalign_spectrum_scores.csv'
    write_csv(aggregate_path, aggregate,
              ('molecule_id', 'baseline_rank', 'smiles', 'formula', 'spectra', *heads))
    write_csv(spectra_path, per_spectrum,
              ('molecule_id', 'spectrum_id', 'baseline_rank', 'smiles', 'adduct',
               'collision_energy_ev', *heads))
    provenance = dict(queries_sha256=receipt['queries_sha256'],
                      baseline_sha256=receipt['hashes']['submission.csv'],
                      official_manifest_sha256=digest(args.checkpoints / 'official_manifest.yaml'),
                      checkpoint_sha256={name: digest(args.checkpoints / 'checkpoints' / f'{name}.ckpt')
                                         for name in heads},
                      channels=list(heads), device=str(device),
                      cuda_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                      torch_version=torch.__version__, rdkit_version=rdBase.rdkitVersion,
                      molecules=len(by_id), spectra=len(rows), candidate_slots_per_query=CAP,
                      candidate_rows=len(aggregate), candidates_scored=sum(
                          any(row.get(name) is not None for name in heads) for row in aggregate),
                      spectrum_candidate_pairs=len(per_spectrum),
                      scored_scans_by_query=usable, skipped_scans_by_channel=skipped,
                      scored_scan_count_by_channel={name: sum(counts.values()) for name, counts in usable.items()},
                      scored_molecule_count_by_channel={name: sum(n > 0 for n in counts.values())
                                                        for name, counts in usable.items()},
                      adducts=sorted({row['adduct'] or 'unknown' for row in rows}),
                      candidate_pool='sealed final Pablo baseline', uploaded_truth=False,
                      vacant_slots_by_query=receipt['vacant_slots_by_query'],
                      scores_sha256=digest(aggregate_path), spectra_scores_sha256=digest(spectra_path),
                      elapsed_seconds=round(time.monotonic() - clock, 2))
    (args.out / 'msalign_receipt.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('MSALIGN_SCORED', provenance['candidates_scored'], 'candidate slots;',
          provenance['scored_scan_count_by_channel'], 'scans;', len(by_id), 'lists', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'queries', 'source', 'checkpoints', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--dreams', type=Path)
    parser.add_argument('--channel', choices=HEADS, nargs='+', default=['bins_fingerprint'])
    parser.add_argument('--device', default='cuda:0')
    score(parser.parse_args())


if __name__ == '__main__':
    main()

"""Prepare the 121-structure E180 panel. Mount only the inference directory on both ranking jobs."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def file_hash(path):
    with path.open('rb') as stream:
        digest = sha256()
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(scans_path, competition, inference, scoring):
    inference, scoring = inference.resolve(), scoring.resolve()
    if inference == scoring or inference in scoring.parents or scoring in inference.parents:
        raise ValueError('Use separate, non-nested inference and scoring directories.')
    for destination in (inference, scoring):
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

    schema = pq.read_schema(competition / 'test.parquet').remove_metadata()
    expected = {'molecule_id', 'spectrum_id', 'ms2_mzs', 'ms2_normalized_intensities',
                'base_peak_intensity', 'adduct', 'ionization_mode', 'instrument_type',
                'precursor_mz', 'collision_energy_orig', 'collision_energy_ev', 'collision_energy_orig_units'}
    if set(schema.names) != expected:
        raise ValueError('Unexpected competition query schema.')
    frame = pd.read_parquet(scans_path, columns=schema.names + ['scan_ordinal'])
    truth = sorted(frame.molecule_id.unique())
    if (len(frame) != 299 or len(truth) != 121 or not frame.spectrum_id.is_unique
            or frame[['molecule_id', 'scan_ordinal']].duplicated().any()
            or not all(isinstance(key, str) and re.fullmatch('[A-Z]{14}', key) for key in truth)):
        raise ValueError('Expected the frozen 121 structures and 299 spectra with E180 identity keys.')
    if (not frame.adduct.eq('[M+H]+').all() or not frame.ionization_mode.eq('positive').all()
            or not frame.collision_energy_orig_units.eq('eV').all()
            or not all(len(ev) == 1 and ev[0] in (20., 40., 60.) for ev in frame.collision_energy_ev)):
        raise ValueError('Expected measured, single-energy E180 protonated spectra.')

    mapping = {key: 'm_' + uuid4().hex for key in truth}
    labels = pd.DataFrame({'molecule_id': list(mapping.values()), 'truth_key': truth})
    frame['molecule_id'] = frame['molecule_id'].map(mapping)
    frame['spectrum_id'] = ['s_' + uuid4().hex for _ in range(len(frame))]
    frame = frame.sort_values(['molecule_id', 'scan_ordinal'])
    table = pa.Table.from_pandas(frame[schema.names], schema=schema, preserve_index=False).replace_schema_metadata(None)
    with TemporaryDirectory(dir=inference.parent) as temp_input, TemporaryDirectory(dir=scoring.parent) as temp_score:
        staging, private = Path(temp_input) / 'inference', Path(temp_score) / 'scoring'
        staging.mkdir()
        private.mkdir()
        pq.write_table(table, staging / 'test.parquet')
        ids = sorted(mapping.values())
        pd.DataFrame({'molecule_id': ids, 'smiles': 'CCO'}).to_csv(staging / 'sample_submission.csv', index=False)
        shutil.copyfile(competition / 'train.parquet', staging / 'train.parquet')
        receipt = {'molecules': len(ids), 'spectra': len(frame),
                   'sha256': {name: file_hash(staging / name) for name in
                              ('train.parquet', 'test.parquet', 'sample_submission.csv')}}
        (staging / 'e180_blind_input.json').write_text(json.dumps(receipt, indent=2)+'\n')
        labels.sort_values('molecule_id').to_parquet(private / 'blind_labels.parquet', index=False)
        (private / 'e180_scoring_input.json').write_text(json.dumps(dict(
            scans_sha256=file_hash(scans_path), inference=receipt), indent=2)+'\n')
        private.rename(scoring)
        staging.rename(inference)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scans', type=Path)
    parser.add_argument('--competition', type=Path)
    parser.add_argument('--inference', type=Path)
    parser.add_argument('--scoring', type=Path)
    args = parser.parse_args()
    paths = (args.scans, args.competition, args.inference, args.scoring)
    if all(paths):
        receipt = prepare(*paths)
    elif any(paths):
        parser.error('Supply all four paths for local execution.')
    else:
        inputs = Path('/kaggle/input')
        scans, = inputs.rglob('e180_scan_pipeline_queries_ce.parquet')
        competition, = {p.parent for p in inputs.rglob('test.parquet')
                         if (p.parent / 'train.parquet').exists()}
        with TemporaryDirectory() as separate:
            receipt = prepare(scans, competition, Path('/kaggle/working/e180_blind'),
                              Path(separate) / 'scoring')
    print(json.dumps(receipt, indent=2))

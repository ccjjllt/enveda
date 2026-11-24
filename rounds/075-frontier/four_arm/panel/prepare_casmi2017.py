"""Build blind CASMI 2017 inputs from organizer spectra and the previously audited train-unseen identities."""
import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold


HOST = 'http://www.casmi-contest.org/2017/'
SOLUTIONS = 'CASMI-solutions.csv'
SUMMARY = 'Chal1to45Summary.csv'
MS1 = 'challenges-001-045-ms-mgf-20170908.zip'
MS2 = 'challenges-001-045-msms-mgf-20170908.zip'
MSE = 'challenges-046-243-msms-mgf-20170522.zip'
CORRECTED = 'challenges-046-243-msms-tsv-20170707.zip'
DIGESTS = {
    SOLUTIONS: 'd1aed33a27746999fcf60ad0ff93389a515a8e012c7f66b028a34aa5528542a6',
    SUMMARY: '0d39cb0e191079f6e87af99d4aa67cb4b5faf7aff2a2607356ce26dfcf1e935d',
    MS1: 'd3d4342e80d144eecdff7c9c5d8c9725555373d2bc14826c20b7e45046d5102a',
    MS2: 'fe1732ba6b24724d3d24cc5745be5a58fd23c6bdae9556d01f9a0ed4060f9deb',
    MSE: '86f38e71b92c0971b7275e1f25b25a44e30d0f245dc4dc9899af4f1ac1b10813',
    CORRECTED: '5a76a4dbca6929cd548c799b6f4de28547b9f4e7d674da682db2e31c3f4f8540',
}
URLS = {n: HOST + ('' if n == SOLUTIONS else 'challenges/') + n for n in DIGESTS}
SOURCES = {1: 'CASMI2017_001_045', 2: 'CASMI2017_046_243'}
ALIASES = ['casmi', 'casmi2017', 'casmi-2017', 'casmi_2017', 'casmi 2017']
TAUT = rdMolStandardize.TautomerEnumerator()


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def load_inputs(cache):
    cache.mkdir(parents=True, exist_ok=True)
    for name, expected in DIGESTS.items():
        path = cache / name
        if not path.exists():
            with urlopen(URLS[name], timeout=60) as response:
                path.write_bytes(response.read())
        if digest(path) != expected:
            raise ValueError(f'Unexpected digest for {name}')


def parse_mgf(archive, member):
    lines = archive.read('casmi2017/' + member).decode().splitlines()
    meta = dict(line.partition('=')[::2] for line in lines if '=' in line and not line.startswith('#'))
    peaks = np.asarray([list(map(float, line.split())) for line in lines if re.match(r'^[0-9]', line)])
    if peaks.ndim != 2 or peaks.shape[1] != 2 or not np.all(np.isfinite(peaks)) or np.any(peaks <= 0):
        raise ValueError(f'Unexpected experimental peak list: {member}')
    return meta, peaks


def identity(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    Chem.RemoveStereochemistry(molecule)
    molecule = TAUT.Canonicalize(molecule)
    key = Chem.MolToInchiKey(molecule).split('-')[0]
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    generic = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(scaffold)) if scaffold.GetNumAtoms() else ''
    return key, generic


def opaque(prefix, name, digits):
    return prefix + sha256(('casmi-2017-independent-v1/' + name).encode()).hexdigest()[:digits]


def measured_record(cache, number, meta):
    early = number <= 45
    name = f'challenge-{number:03d}'
    if early and 'unreported NP or synthetic analog' in meta['Description']:
        return None, 'Organizer cannot confirm natural origin.'
    with ZipFile(cache / (MS2 if early else MSE)) as archive:
        head, peaks = parse_mgf(archive, name + '-msms.mgf')
    polarity = 'positive' if '+' in head['CHARGE'] else 'negative'
    if early:
        with ZipFile(cache / MS1) as archive:
            parent, isotopes = parse_mgf(archive, name + '-ms.mgf')
        stated = {'Pos': 'positive', 'Neg': 'negative'}[meta['ESI Mode'].strip()]
        parent_polarity = 'positive' if '+' in parent['CHARGE'] else 'negative'
        if polarity != stated or parent_polarity != stated:
            return None, 'Published MS/MS charge disagrees with the acquisition summary or paired MS1 metadata.'
        # These particular MS1 files contain the isolated ion isotope cluster; its first peak is monoisotopic.
        if (not np.all(np.diff(isotopes[:, 0]) > 0)
                or not (0.97 <= isotopes[1, 0] - isotopes[0, 0] <= 1.04)
                or isotopes[0,1] != isotopes[:,1].max()):
            raise ValueError(f'{name}: unexpected annotated MS1 isotope cluster')
        precursor = isotopes[0, 0]
        energy = meta['CollisionEnergy(eV)']
        instrument = 'Waters SYNAPT QTOF'
        natural = meta['Description']
        method = 'Method ' + meta['Method']
        origin = 'MS1 measured monoisotopic peak'
    else:
        if polarity != ('negative' if number <= 131 else 'positive'):
            raise ValueError(f'{name}: inconsistent official polarity')
        precursor = float(head['PEPMASS'])
        energy = '15-35 ramp' if polarity == 'negative' else '10-30 ramp'
        instrument = 'Waters SYNAPT HDMS QTOF'
        natural = 'Natural product as declared for challenges 46–243 by organizer'
        method = 'UPLC-MSE'
        origin = 'MS/MS MGF PEPMASS'
        with ZipFile(cache / CORRECTED) as archive:
            with archive.open('casmi2017/' + name + '-msms.txt') as stream:
                tsv = np.loadtxt(stream)
        if peaks.shape != tsv.shape or not np.array_equal(peaks, tsv):
            raise ValueError(f'{name}: MGF disagrees with organizer corrected peak table')
    values = [float(n) for n in re.findall(r'\d+(?:\.\d+)?', energy)]
    if not values or len(values) > 2 or precursor <= 0:
        raise ValueError(f'{name}: invalid energy or precursor')
    row = dict(spectrum_id=opaque('s_', name, 8), ms2_mzs=peaks[:,0].tolist(),
               ms2_normalized_intensities=(peaks[:,1] / peaks[:,1].max()).tolist(),
               base_peak_intensity=peaks[:,1].max(), precursor_mz=float(precursor),
               adduct='[M+H]+' if polarity == 'positive' else '[M-H]-',
               instrument_type=instrument, ionization_mode=polarity,
               collision_energy_orig=energy, collision_energy_ev=values, collision_energy_orig_units='eV')
    study = dict(challenge=name, source=SOURCES[1 if early else 2], study_method=method,
                 natural_origin=natural, precursor_origin=origin,
                 adduct_status='polarity-based assumed protonation/deprotonation')
    return (row, study), None


def prepare(args):
    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError('Build with the competition scoring RDKit 2026.03.3')
    cache, out = args.cache, args.out
    load_inputs(cache)
    frozen = json.loads(args.eligible.read_text())['scoring_identity']['unseen_query_ids']
    with (cache / SUMMARY).open() as handle:
        acquisition = {int(r['ChalNo']): r for r in csv.DictReader(handle)}
    with (cache / SOLUTIONS).open() as handle:
        truths = {r['Challenge']: r['SMILES'].strip() for r in csv.DictReader(handle)}
    observations, exclusions = defaultdict(list), {}
    for challenge in frozen:
        number = int(challenge[-3:])
        record, reason = measured_record(cache, number, acquisition.get(number))
        if reason:
            exclusions[challenge] = reason
            continue
        key, scaffold = identity(truths[challenge])
        observations[key].append((challenge, scaffold, *record))
    scans, labels, lineage = [], [], []
    for key, members in sorted(observations.items()):
        challenge, scaffold, _, provenance = members[0]
        mid = opaque('m_', '/'.join(m[0] for m in members), 6)
        labels.append(dict(molecule_id=mid, truth_key=key, truth_smiles=truths[challenge],
                           source=provenance['source'], scaffold=scaffold,
                           challenges='|'.join(m[0] for m in members),
                           natural_origin=provenance['natural_origin']))
        for _, _, observation, source in members:
            if source['source'] != provenance['source']:
                raise ValueError('One identity crosses collection series; merge source groups before use')
            scans.append(dict(molecule_id=mid, **observation))
            lineage.append(dict(molecule_id=mid, spectrum_id=observation['spectrum_id'], **source))
    blind, private = out / 'blind', out / 'private'
    for folder in [blind, private]:
        folder.mkdir(parents=True, exist_ok=True)
    schema = pq.read_schema(args.schema).remove_metadata()
    frame = pd.DataFrame(scans).sort_values(['molecule_id','spectrum_id'])
    labels = pd.DataFrame(labels).sort_values('molecule_id')
    if labels.truth_key.duplicated().any() or labels.molecule_id.duplicated().any() or frame.spectrum_id.duplicated().any():
        raise ValueError('Duplicate blind id, spectrum or normalized structure')
    pq.write_table(pa.Table.from_pandas(frame[schema.names], schema=schema, preserve_index=False)
                   .replace_schema_metadata(None), blind / 'test.parquet')
    pd.DataFrame({'molecule_id':labels.molecule_id, 'smiles':'CCO'}).to_csv(blind / 'sample_submission.csv',index=False)
    labels.to_parquet(private / 'labels.parquet', index=False)
    pd.DataFrame(lineage).to_parquet(private / 'spectra_provenance.parquet', index=False)
    holdout = dict(blocked_keys=sorted(observations), blocked_scaffolds=sorted(set(labels.scaffold) - {''}),
                   block_acyclic=bool(labels.scaffold.eq('').any()),
                   blocked_sources=sorted(set(ALIASES) | set(SOURCES.values())))
    (private/'holdout.json').write_text(json.dumps(holdout, indent=2)+'\n')
    counts = frame.groupby('molecule_id').size().value_counts()
    report = dict(source_urls=URLS, inputs_sha256=DIGESTS,
                  acquisition_pages=[HOST+'challenges_1-45.shtml',HOST+'challenges_46-243.shtml'],
                  excluded_challenges=exclusions, eligible_queries=len(frozen), eligible_identities=21,
                  molecules=len(labels), spectra=len(scans),
                  origin_categories=labels.natural_origin.value_counts().to_dict(),
                  molecules_by_source=labels.source.value_counts().to_dict(),
                  spectra_by_source=dict(Counter(s['source'] for s in lineage)),
                  spectra_by_adduct=frame.adduct.value_counts().to_dict(), multiplicity=counts.to_dict(),
                  trust='The organizer provides measured polarity, not ion species. For all scans use the same label-independent direct proton assumption; no reference to true mass. Ramps are one experimental spectrum with endpoints in eV.',
                  blocking='Source is the entire CASMI 2017 benchmark; the two independent acquisition series are additionally reported. LC methods within the first series are not treated as independent institutions. Normalize tautomers and remove stereochemistry before generic Murcko and identity filtering. Acyclic compounds have no scaffold.',
                  output_sha256={str(p.relative_to(out)):digest(p) for p in [blind/'test.parquet',blind/'sample_submission.csv',private/'labels.parquet',private/'holdout.json',private/'spectra_provenance.parquet']})
    (out/'panel_manifest.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k:report[k] for k in ['molecules','spectra','molecules_by_source','spectra_by_adduct','excluded_challenges']}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--schema', type=Path, default=Path('data/raw/test.parquet'))
    ap.add_argument('--eligible', type=Path, default=Path('data/references/075-casmi2017-overlap-2026-11-24.json'))
    prepare(ap.parse_args())

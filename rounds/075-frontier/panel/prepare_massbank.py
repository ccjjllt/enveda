"""Stage the official MassBank 2026.11 experimental MS2 accession additions."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from zipfile import ZipFile

import numpy as np
import pandas as pd


NEW = ('24082491', '2026.11', 'd82fe03140b1d86f1cddf59577ef5bad')
OLD = ('19590880', '2026.03', '1de9018361f7ffc689aa327f92e97e0f')
MAY_2025 = ('15341877', '2025.05.1', '7f684f9f39df50943ef38573c104e572')
OUTPUT = Path('/kaggle/working')
PERMITTED_LICENSE = re.compile(r'^(?:CC0(?: 1\.0)?|CC BY(?: NC)?(?: SA)?(?: [0-9]+\.[0-9]+)?)$', re.I)
SINGLE_CE = re.compile(r'^(\d+(?:\.\d+)?)\s*(eV)?$', re.I)
PROTONATED = {'[M+H]+': 'positive'}
EXTRA_ADDUCTS = {'[M-H]-': 'negative', '[M+Na]+': 'positive', '[M+NH4]+': 'positive'}
AMMONIUM_ALIASES = {'[(M+NH3)+H]+', '[M+NH3+H]+', '[M+H+NH3]+', '[M+H3N+H]+', '[M+H4N]+'}


def get_zip(release, directory):
    record, tag, checksum = release
    path = directory / f'massbank-{tag}.zip'
    filename = f'MassBank-data-{tag}.zip'
    url = f'https://zenodo.org/records/{record}/files/MassBank/{filename}?download=1'
    subprocess.run(['curl', '--fail', '--location', '--retry', '3', '--silent', '--show-error',
                    '--output', str(path), url], check=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'md5').hexdigest()
    if digest != checksum:
        raise ValueError(f'MassBank {tag} did not match the pinned archive MD5 ({digest})')
    print(f'MassBank {tag}: {path.stat().st_size} bytes', flush=True)
    return path


def accession_files(archive):
    entries = {}
    for name in archive.namelist():
        member = PurePosixPath(name)
        if member.suffix.lower() == '.txt' and not member.stem.startswith('.') and len(member.parts) >= 2:
            entries[member.stem] = name
    return entries


def ms2_record(archive, member, allowed_adducts=PROTONATED, *, allow_ion_type=False):
    lines = archive.read(member).decode('utf-8-sig', errors='replace').splitlines()
    fields = {}
    mzs, intensities = [], []
    in_peaks = False
    for line in lines:
        if line == '//':
            break
        if in_peaks:
            columns = line.split()
            if len(columns) >= 2:
                try:
                    mz, intensity = float(columns[0]), float(columns[1])
                except ValueError:
                    continue
                if np.isfinite(mz) and np.isfinite(intensity) and mz > 0 and intensity > 0:
                    mzs.append(mz)
                    intensities.append(intensity)
            continue
        if ': ' in line:
            name, value = line.split(': ', 1)
            if name in ('AC$MASS_SPECTROMETRY', 'MS$FOCUSED_ION', 'CH$LINK') and ' ' in value:
                detail, value = value.split(' ', 1)
                name = f'{name}: {detail}'
            fields.setdefault(name, value.strip())
        if line.startswith('PK$PEAK:'):
            in_peaks = True

    if not fields.get('ACCESSION'):
        return 'missing_accession', None
    license_text = fields.get('LICENSE', '')
    license_code = ' '.join(license_text.replace('-', ' ').split())
    if not PERMITTED_LICENSE.fullmatch(license_code):
        return 'license_unavailable_or_restricted', None
    polarity = fields.get('AC$MASS_SPECTROMETRY: ION_MODE', '').lower()
    precursor_type = fields.get('MS$FOCUSED_ION: PRECURSOR_TYPE', '')
    if not precursor_type and (allow_ion_type or '[M+NH4]+' in allowed_adducts):
        precursor_type = fields.get('MS$FOCUSED_ION: ION_TYPE', '')
    raw_adduct = ''.join(precursor_type.split())
    adduct = '[M+NH4]+' if raw_adduct in AMMONIUM_ALIASES else raw_adduct
    if (fields.get('AC$MASS_SPECTROMETRY: MS_TYPE', '').upper() != 'MS2'
            or polarity not in {'positive', 'negative'} or allowed_adducts.get(adduct) != polarity):
        return 'outside_requested_adduct_ms2', None
    smiles = fields.get('CH$SMILES', '')
    if not smiles or smiles in ('N/A', 'NA', 'Unknown'):
        return 'missing_smiles', None
    instrument_name = fields.get('AC$INSTRUMENT', '')
    instrument = fields.get('AC$INSTRUMENT_TYPE', '')
    title = fields.get('RECORD_TITLE', '')
    if not instrument_name or not instrument or re.search(r'in.silico|simulat|predict', f'{instrument_name} {instrument} {title}', re.I):
        return 'no_experimental_instrument', None
    precursor_text = fields.get('MS$FOCUSED_ION: PRECURSOR_M/Z', '')
    try:
        precursor_mz = float(precursor_text)
        neutral_mass = float(fields.get('CH$EXACT_MASS', 'nan'))
    except ValueError:
        return 'invalid_mass', None
    if not np.isfinite(precursor_mz) or precursor_mz <= 0 or len(mzs) < 3:
        return 'unusable_spectrum', None
    ce_original = fields.get('AC$MASS_SPECTROMETRY: COLLISION_ENERGY', '')
    ce_match = SINGLE_CE.fullmatch(ce_original)
    ce_ev = float(ce_match.group(1)) if ce_match and ce_match.group(2) else None
    raw = fields.get('CH$LINK: INCHIKEY', '').split()
    return 'retained', dict(identifier=fields['ACCESSION'], smiles=smiles,
                            inchikey=raw[0] if raw else None, adduct=adduct, raw_adduct=raw_adduct, polarity=polarity,
                            mzs=mzs, intensities=intensities, precursor_mz=precursor_mz,
                            parent_mass=neutral_mass if np.isfinite(neutral_mass) else None,
                            instrument_type=instrument, instrument_name=instrument_name,
                            source_library=PurePosixPath(member).parts[-2],
                            collision_energy=ce_ev, collision_energy_orig=ce_original,
                            collision_energy_orig_units='eV' if ce_ev is not None else 'unknown',
                            license=license_text, citation=fields.get('PUBLICATION'),
                            record_date=fields.get('DATE'))


def main(allowed_adducts=PROTONATED, filename='massbank_added_positive.parquet', new_release=NEW, old_release=OLD):
    with tempfile.TemporaryDirectory(prefix='massbank-releases-') as temporary:
        directory = Path(temporary)
        with ThreadPoolExecutor(max_workers=2) as pool:
            old_future = pool.submit(get_zip, old_release, directory)
            new_future = pool.submit(get_zip, new_release, directory)
            old, new = old_future.result(), new_future.result()
        with ZipFile(old) as prior, ZipFile(new) as current:
            reference = accession_files(prior)
            catalogue = accession_files(current)
            novel_accessions = sorted(catalogue.keys() - reference.keys())
            counts = Counter()
            rows = []
            for accession in novel_accessions:
                outcome, row = ms2_record(current, catalogue[accession], allowed_adducts)
                counts[outcome] += 1
                if row is not None:
                    rows.append(row)
    frame = pd.DataFrame(rows).drop_duplicates('identifier') if rows else pd.DataFrame()
    report = dict(release=f'MassBank {new_release[1]}', old_release=f'MassBank {old_release[1]}',
                  new_zenodo_record=new_release[0], old_zenodo_record=old_release[0],
                  new_md5=new_release[2], old_md5=old_release[2], new_accessions=len(novel_accessions),
                  filter_counts=dict(counts), retained=len(frame),
                  requested_adducts=allowed_adducts)
    if not frame.empty:
        frame.to_parquet(OUTPUT / filename, index=False)
        report.update(unique_smiles=int(frame.smiles.nunique()),
                      licenses=frame.license.value_counts().to_dict(),
                      by_source=frame.source_library.value_counts().to_dict(),
                      by_adduct=frame.adduct.value_counts().to_dict(),
                      by_reported_adduct=frame.raw_adduct.value_counts().to_dict(),
                      single_ev_reported=int(frame.collision_energy.notna().sum()))
    (OUTPUT / 'massbank_source.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

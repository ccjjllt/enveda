"""Freeze measured natural-product queries and keep labels outside inference inputs."""

import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
from hashlib import file_digest, sha256
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdMolDescriptors

from prepare_casmi2017 import ALIASES, DIGESTS, SOLUTIONS, SUMMARY, URLS, identity, load_inputs, measured_record

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'panel'))
from prepare_massbank import accession_files, ms2_record


MASSBANK_MD5 = 'd82fe03140b1d86f1cddf59577ef5bad'
MASSBANK_URL = 'https://zenodo.org/records/24082491'
MASSBANK_ADDUCTS = {'[M+H]+': 'positive', '[M-H]-': 'negative'}
NATURAL_COLLECTIONS = {'RIKEN_ReSpect': 'ReSpect phytochemicals from published primary plant MS/MS studies'}
NATURAL_TRAIN_SOURCE = 'enveda-np-examples'
SEED = 'casmi26-pilot-retrieval-2026-11-24-v1'
GNPS_LITERATURE = {
    'ELIXDB-LICHEN-DATABASE': 'https://doi.org/10.1038/s41597-025-04488-w',
    'LEAFBOT': 'https://bpb-us-e1.wpmucdn.com/sites.psu.edu/dist/4/104057/files/2026/04/2025-LEAFBot-Botanical-MS-Database-JAmericanSocietyMassSpectrometry.pdf',
    'MIADB': 'https://doi.org/10.1186/s13321-025-01009-0',
    'TUEBINGEN-NATURAL-PRODUCT-COLLECTION': 'https://gnps.ucsd.edu/ProteoSAFe/gnpslibrary.jsp?library=TUEBINGEN-NATURAL-PRODUCT-COLLECTION',
    'GNPS-XENORHABDUS-PHOTORHABDUS-COMPOUNDS-LIBRARY': 'https://gnps.ucsd.edu/ProteoSAFe/gnpslibrary.jsp?library=GNPS-XENORHABDUS-PHOTORHABDUS-COMPOUNDS-LIBRARY',
}
MYCO_URL = 'https://mycomsbase.helmholtz-hzi.de/MycoMSBase/'


def digest(path, scheme='sha256'):
    with path.open('rb') as source:
        return file_digest(source, scheme).hexdigest()


def stable_id(kind, key, length):
    return kind + '_' + sha256((SEED + '/' + key).encode()).hexdigest()[:length]


def spectrum_hash(row):
    payload = (row['precursor_mz'], row['adduct'], row['ms2_mzs'], row['ms2_normalized_intensities'])
    return sha256(json.dumps(payload, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@lru_cache(100000)
def metric(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    if mol.GetNumHeavyAtoms() < 5:
        return None
    key, scaffold = identity(smiles)
    if len(key) != 14 or not scaffold:
        return None
    return key, scaffold, Chem.MolToInchiKey(mol)[:14]


def dev_keys(paths, train):
    removed, identities_by_file = set(), {}
    for path in paths:
        frame = pd.read_parquet(path)
        keys = set()
        for col in ['truth_key', 'candidate_id', 'molecule', 'inchikey14']:
            if col in frame:
                keys.update(v for v in frame[col].dropna().astype(str) if len(v) == 14 and v.isalpha())
        if 'truth_smiles' in frame:
            for smiles in frame.truth_smiles.dropna().unique():
                result = metric(smiles)
                if result:
                    keys.update((result[0], result[2]))
        if not keys:
            raise ValueError(f'Could not read normalized development identity from {path.name}')
        removed.update(keys)
        identities_by_file[path.parent.name + '/' + path.name] = len(keys)
    from_train = pq.read_table(train, columns=['normalized_smiles'],
                               filters=[('ingest_lib', '==', NATURAL_TRAIN_SOURCE)])
    extra = set()
    for smiles in pc.unique(from_train['normalized_smiles']).to_pylist():
        result = metric(smiles)
        if result:
            extra.update((result[0], result[2]))
    removed.update(extra)
    identities_by_file[NATURAL_TRAIN_SOURCE] = len(extra)
    return removed, identities_by_file


def casmi_rows(cache):
    load_inputs(cache)
    with (cache / SOLUTIONS).open() as handle:
        truths = {row['Challenge']: row['SMILES'].strip() for row in csv.DictReader(handle)}
    with (cache / SUMMARY).open() as handle:
        acquisition = {int(row['ChalNo']): row for row in csv.DictReader(handle)}
    for label, smiles in sorted(truths.items()):
        number = int(label[-3:])
        if not 1 <= number <= 243 or not smiles:
            continue
        try:
            record, reason = measured_record(cache, number, acquisition.get(number))
        except (KeyError, ValueError) as error:
            yield None, f'CASMI2017: invalid or absent organizer spectrum ({type(error).__name__})'
            continue
        if reason:
            yield None, f'CASMI2017: {reason}'
            continue
        row, lineage = record
        info = dict(lineage, archive='CASMI 2017 original MS/MS and acquisition sheets',
                    accession=label, natural_evidence=lineage['natural_origin'])
        yield (smiles, row, info), None


def massbank_rows(path, natural_identities, excluded_raw):
    if digest(path, 'md5') != MASSBANK_MD5:
        raise ValueError('MassBank 2026.11 archive does not match the Zenodo digest')
    with ZipFile(path) as archive:
        for accession, member in sorted(accession_files(archive).items()):
            # Skip most entries using their untouched published connectivity key. Exact membership
            # is checked on the RDKit-normalized record before accepting it.
            content = archive.read(member)
            collection = Path(member).parent.name
            marker = content.find(b'CH$LINK: INCHIKEY ')
            if marker < 0:
                yield None, 'MassBank: absent published InChIKey'
                continue
            key = content[marker + 18:marker + 32].decode('ascii', errors='ignore')
            if key not in natural_identities and collection not in NATURAL_COLLECTIONS:
                yield None, 'MassBank: no catalogued natural origin'
                continue
            if key in excluded_raw:
                yield None, 'MassBank: known raw training or development identity'
                continue
            state, record = ms2_record(archive, member, MASSBANK_ADDUCTS, allow_ion_type=True)
            if record is None:
                yield None, f'MassBank: {state}'
                continue
            if 'casmi' in (record['source_library'] + str(record['citation'])).lower():
                yield None, 'MassBank: imported competition spectra'
                continue
            if 'nist' in record['source_library'].lower():
                yield None, 'MassBank: NIST source excluded'
                continue
            normalized = metric(record['smiles'])
            if normalized is None or (normalized[0] not in natural_identities and normalized[2] != key):
                yield None, 'MassBank: parsed structure differs from catalog evidence'
                continue
            inten = np.asarray(record['intensities'], dtype=float)
            base = float(inten.max())
            energy = record['collision_energy']
            row = dict(ms2_mzs=record['mzs'], ms2_normalized_intensities=(inten/base).tolist(),
                       base_peak_intensity=base, precursor_mz=record['precursor_mz'],
                       adduct=record['adduct'], instrument_type=record['instrument_type'],
                       ionization_mode=record['polarity'], collision_energy_orig=record['collision_energy_orig'],
                       collision_energy_ev=[] if energy is None else [energy],
                       collision_energy_orig_units=record['collision_energy_orig_units'])
            source = record['source_library']
            evidence = natural_identities.get(key) or NATURAL_COLLECTIONS.get(collection)
            if collection in NATURAL_COLLECTIONS and key not in natural_identities:
                if not record['citation'] or not re.search(r'\bTOF\b|QTOF|ITFT|QFT|Orbitrap|Fourier',
                                                            record['instrument_type'], re.I):
                    yield None, 'MassBank: ReSpect only with study metadata and high-resolution instrument'
                    continue
            info = dict(source=f'MassBank_{source}', source_collection=source, archive='MassBank 2026.11',
                        accession=accession, natural_evidence=evidence,
                        citation=record['citation'], license=record['license'],
                        adduct_status='MassBank record precursor type')
            yield (record['smiles'], row, info), None


def natural_catalog(path, theobroma=None):
    df = pd.read_parquet(path, columns=['ik14', 'scorer_key', 'in_lotus', 'in_npatlas'])
    evidence = {}
    for row in df.itertuples(index=False):
        tags = '+'.join(x for x, present in (('LOTUS', row.in_lotus), ('NPAtlas', row.in_npatlas)) if present)
        if tags:
            for key in (row.ik14, row.scorer_key):
                if isinstance(key, str) and len(key) == 14:
                    evidence[key] = tags
    if theobroma:
        material = pq.read_table(theobroma, columns=['inchikey','source_db','source_organism_curated','reference_doi'],
                                  filters=[('license_tier','in',['CC0','CC BY 4.0','CC BY-NC 4.0'])]).to_pandas()
        material = material.dropna(subset=['inchikey','source_db','source_organism_curated','reference_doi'])
        material = material.loc[material.reference_doi.str.contains('10.', regex=False)
                                & material.source_organism_curated.str.len().ge(4)]
        for row in material.itertuples(index=False):
            evidence.setdefault(row.inchikey[:14], f'{row.source_db}: {row.source_organism_curated}; {row.reference_doi}')
    return evidence


def gnps_rows(path, known_raw):
    original = json.loads(path.read_text())
    for rec in original:
        library = rec['library_membership']
        if library not in GNPS_LITERATURE:
            raise ValueError(f'Undocumented GNPS library {library}')
        key = (rec.get('InChIKey_smiles') or rec.get('InChIKey_inchi') or '')[:14]
        if key in known_raw:
            yield None, 'GNPS: known raw training or development identity'
            continue
        compound_source = rec.get('Compound_Source', '').strip().lower()
        admissible = {'isolated', 'commercial'} if library == 'LEAFBOT' else {'isolated'}
        if (compound_source not in admissible or rec.get('ms_level') != '2'
                or rec.get('spectrum_status') != '1'):
            yield None, 'GNPS: not an active MS/MS scan of an isolated or botanical standard'
            continue
        smiles = rec.get('Smiles','').strip().strip('"')
        if smiles in ('','N/A','NA','None'):
            yield None, 'GNPS: no public SMILES label in original record'
            continue
        # A primary high-resolution reference and a directly reported precursor are necessary.
        instrument = rec.get('Instrument','')
        if not re.search(r'Orbitrap|Q.?ToF|FT|ZenoTOF|TripleTOF', instrument, re.I):
            yield None, 'GNPS: instrument does not identify a high-resolution reference scan'
            continue
        adduct_tag = rec.get('Adduct','').strip().replace(' ', '')
        adduct = {'M+H':'[M+H]+','M-H':'[M-H]-','[M+H]+':'[M+H]+','[M-H]-':'[M-H]-'}.get(adduct_tag)
        if not adduct or MASSBANK_ADDUCTS[adduct] != rec.get('Ion_Mode','').lower():
            yield None, 'GNPS: unsupported or conflicting adduct'
            continue
        try:
            precursor = float(rec['Precursor_MZ'])
            pairs = np.asarray(json.loads(rec['peaks_json']), dtype=float)
            pairs = pairs[np.all(np.isfinite(pairs),axis=1) & (pairs > 0).all(axis=1)]
            if pairs.shape[0] < 4 or not np.isfinite(precursor) or precursor <= 0:
                raise ValueError('invalid precursor or peaks')
        except (TypeError, ValueError, IndexError):
            yield None, 'GNPS: invalid observed precursor or peak list'
            continue
        base = float(pairs[:,1].max())
        energy_text = str(rec.get('Collision_Energy', '')).strip()
        energy = re.fullmatch(r'(\d+(?:\.\d+)?)\s*eV',energy_text,re.I)
        row = dict(ms2_mzs=pairs[:,0].tolist(), ms2_normalized_intensities=(pairs[:,1]/base).tolist(),
                   base_peak_intensity=base, precursor_mz=precursor, adduct=adduct,
                   instrument_type=instrument, ionization_mode=MASSBANK_ADDUCTS[adduct],
                   collision_energy_orig=energy_text, collision_energy_ev=[float(energy.group(1))] if energy else [],
                   collision_energy_orig_units='eV' if energy else 'unknown')
        info = dict(source='GNPS_'+library, source_collection=library, archive='GNPS original reference library',
                    accession=rec['spectrum_id'], natural_evidence=f'{library}: {compound_source} compound',
                    citation=GNPS_LITERATURE[library], instrument_data=rec.get('source_file'),
                    accession_url=rec.get('url'), adduct_status='GNPS original metadata')
        yield (smiles,row,info), None


def mycomsb_rows(path, known_raw, audited_ids=None):
    for accession_file in sorted(path.glob('MycoMSB-*.json')):
        if audited_ids is not None and accession_file.stem not in audited_ids:
            yield None, 'MycoMSBase: excluded by independent accession audit'
            continue
        rec = json.loads(accession_file.read_text())
        if rec.get('deprecated') or rec.get('license') != 'CC BY' or not rec.get('publication'):
            yield None, 'MycoMSBase: deprecated, no primary reference, or unsupported license'
            continue
        compound = rec.get('compound') or {}
        key = next((x['identifier'][:14] for x in compound.get('link', []) if x['database'] == 'INCHIKEY'), '')
        if key in known_raw:
            yield None, 'MycoMSBase: known raw training or development identity'
            continue
        smiles = compound.get('smiles')
        chem = metric(smiles) if isinstance(smiles,str) and smiles else None
        species = (rec.get('species') or {}).get('name')
        if not chem or not key or key not in (chem[0],chem[2]) or not species:
            yield None, 'MycoMSBase: label or published organism cannot be verified'
            continue
        acquisition = rec['acquisition']
        settings = acquisition['mass_spectrometry']
        metadata = {x['subtag']:x['value'] for x in rec['mass_spectrometry']['focused_ion']}
        adduct = metadata.get('PRECURSOR_TYPE')
        polarity = settings.get('ion_mode','').lower()
        if settings.get('ms_type') != 'MS2' or MASSBANK_ADDUCTS.get(adduct) != polarity:
            yield None, 'MycoMSBase: unsupported MS/MS adduct'
            continue
        try:
            precursor = float(metadata['PRECURSOR_M/Z'])
            pairs = np.asarray([(p['mz'],p['intensity']) for p in rec['peak']['peak']['values']],dtype=float)
            pairs = pairs[np.isfinite(pairs).all(axis=1) & (pairs > 0).all(axis=1)]
            if len(pairs) < 4 or not np.isfinite(precursor) or precursor <= 0:
                raise ValueError('invalid observed precursor or peaks')
        except (KeyError, TypeError, ValueError, IndexError):
            yield None, 'MycoMSBase: invalid observed precursor or peak list'
            continue
        base = float(pairs[:,1].max())
        energy_text = next((x['value'] for x in settings.get('subtags',[]) if x['subtag'] == 'COLLISION_ENERGY'),'')
        energy = re.fullmatch(r'(\d+(?:\.\d+)?)\s*eV',str(energy_text),re.I)
        instrument = acquisition['instrument']
        instrument_type = 'timsTOF' if 'timstof' in instrument.lower() else instrument
        row = dict(ms2_mzs=pairs[:,0].tolist(), ms2_normalized_intensities=(pairs[:,1]/base).tolist(),
                   base_peak_intensity=base, precursor_mz=precursor, adduct=adduct,
                   instrument_type=instrument_type, ionization_mode=polarity,
                   collision_energy_orig=str(energy_text), collision_energy_ev=[float(energy.group(1))] if energy else [],
                   collision_energy_orig_units='eV' if energy else 'unknown')
        accession = rec['accession']
        info = dict(source='MycoMSBase', source_collection='MycoMSBase', archive='MycoMSBase original CC BY records',
                    accession=accession, natural_evidence=species, citation=rec['publication'], license=rec['license'],
                    measured_instrument=instrument,
                    accession_url=f'https://mycomsbase.helmholtz-hzi.de/MycoMSBase-api/records/{accession}',
                    adduct_status='MycoMSBase original focused-ion metadata', source_sha256=digest(accession_file))
        yield (smiles,row,info), None


def collect(rows, blocked):
    groups, counts = defaultdict(list), Counter()
    for record, reason in rows:
        if reason:
            counts[reason] += 1
            continue
        smiles, row, info = record
        chem = metric(smiles)
        if chem is None:
            counts['invalid, acyclic or disconnected'] += 1
            continue
        key, scaffold, raw = chem
        if key in blocked or raw in blocked:
            counts['previous development identity'] += 1
            continue
        if len([mz for mz in row['ms2_mzs'] if mz < row['precursor_mz'] - .5]) < 4:
            counts['fewer than four fragment peaks'] += 1
            continue
        groups[(info['source'], key)].append((smiles, row, info, scaffold, raw))
    return groups, counts


def known_training_identities(path):
    frame = pd.read_parquet(path, columns=['inchikey14', 'scorer_key'])
    return {key for column in frame for key in frame[column].dropna().unique()
            if isinstance(key, str) and len(key) == 14 and key.isalpha()}


@lru_cache(100000)
def composition(formula):
    return tuple(sorted((atom, int(count or 1)) for atom, count in re.findall(r'([A-Z][a-z]?)(\d*)', formula)
                        if atom != 'H'))


def exact_training_collisions(path, groups):
    targets = defaultdict(set)
    for (_, key), rows in groups.items():
        mol = Chem.MolFromSmiles(rows[0][0])
        targets[composition(rdMolDescriptors.CalcMolFormula(mol))].add(key)
    frame = pd.read_parquet(path, columns=['formula','smiles']).dropna()
    needed = {f:composition(f) for f in frame.formula.unique()}
    needed = {f:g for f,g in needed.items() if g in targets}
    seen = set()
    examined = 0
    for row in frame.loc[frame.formula.isin(needed)].drop_duplicates('smiles').itertuples(index=False):
        result = metric(row.smiles)
        examined += 1
        if result and result[0] in targets[needed[row.formula]]:
            seen.add(result[0])
    return seen, examined


def select(groups, quotas, fill_source=None):
    selections, ids, scaffolds = [], set(), Counter()
    sources = list(quotas)
    if fill_source:
        sources.remove(fill_source)
        sources.append(fill_source)
    shortage = 0
    for source in sources:
        limit = quotas[source] + (shortage if source == fill_source else 0)
        choices = [(key, rows) for (group, key), rows in groups.items() if group == source]
        choices.sort(key=lambda item: stable_id('s', item[0], 64))
        count = 0
        for key, rows in choices:
            if count == limit:
                break
            scaffold = rows[0][3]
            if key in ids or scaffolds[scaffold] >= 3:
                continue
            if any(record[3] != scaffold for record in rows):
                raise ValueError(f'{key}: normalized identity has conflicting scaffold')
            selections.append((source, key, rows))
            count += 1
            ids.add(key)
            scaffolds[scaffold] += 1
        if count != limit and fill_source and source != fill_source:
            shortage += limit-count
        elif count != limit:
            raise ValueError(f'Only selected {count}/{limit} eligible queries from {source}')
    if len({source for source, _, _ in selections}) < 2 or len(selections) != sum(quotas.values()):
        raise ValueError('Frozen panel does not retain the expected query and source counts')
    return selections


def write(args, selections, counts, group_counts, dev_seen):
    labels, rows, lineages, aliases, raw_keys = [], [], [], set(), set()
    for source, key, members in selections:
        mid = stable_id('m', source + '/' + key, 12)
        smiles, _, info, scaffold, _ = members[0]
        raw_keys.update(member[4] for member in members)
        labels.append(dict(molecule_id=mid, truth_key=key, truth_smiles=smiles, source=source,
                           scaffold=scaffold, natural_origin=info['natural_evidence']))
        ordered = sorted(members, key=lambda r: (r[1]['ionization_mode'], r[1]['collision_energy_ev'],
                                                  r[2]['accession']))
        picked, seen_hash, seen_regime = [], set(), set()
        for _, row, info, _, _ in ordered:
            raw_digest = spectrum_hash(row)
            regime = (row['adduct'], tuple(row['collision_energy_ev']), row['collision_energy_orig'])
            if raw_digest in seen_hash or regime in seen_regime:
                continue
            seen_hash.add(raw_digest)
            seen_regime.add(regime)
            picked.append((row, info, raw_digest))
        # Limit spectrum count using wider energy coverage, without looking at a candidate score.
        if len(picked) > 4:
            chosen = np.linspace(0, len(picked) - 1, num=4, dtype=int)
            picked = [picked[i] for i in chosen]
        for row, info, checksum in picked:
            sid = stable_id('s', info['archive'] + '/' + info['accession'], 16)
            rows.append(dict(row, molecule_id=mid, spectrum_id=sid))
            lineages.append(dict(info, molecule_id=mid, spectrum_id=sid, spectrum_sha256=checksum))
        if source.startswith('CASMI2017'):
            aliases.update(ALIASES)
        elif source.startswith('GNPS_'):
            aliases.update(('gnps', 'mona', info['source_collection']))
        elif source == 'MycoMSBase':
            aliases.update(('MycoMSB', 'MycoMSBase', 'Helmholtz'))
        else:
            aliases.update(('massbank', 'mona', info['source_collection']))
            if info['source_collection'].startswith('RIKEN'):
                aliases.update(('riken', 'ReSpect'))
        aliases.add(source)
    out = args.out
    blind, private = out/'blind', out/'private'
    for folder in (blind, private):
        folder.mkdir(parents=True, exist_ok=True)
    scans = pd.DataFrame(rows).sort_values(['molecule_id', 'spectrum_id'])
    truth = pd.DataFrame(labels).sort_values('molecule_id')
    assert scans.spectrum_id.is_unique and truth.molecule_id.is_unique and truth.truth_key.is_unique
    assert set(scans.molecule_id) == set(truth.molecule_id)
    schema = pq.read_schema(args.schema).remove_metadata()
    pq.write_table(pa.Table.from_pandas(scans[schema.names], schema=schema, preserve_index=False),
                   blind/'test.parquet')
    pd.DataFrame({'molecule_id':truth.molecule_id, 'smiles':'CCO'}).to_csv(blind/'sample_submission.csv',index=False)
    truth.to_parquet(private/'labels.parquet', index=False)
    pd.DataFrame(lineages).to_parquet(private/'spectra_provenance.parquet',index=False)
    holdout = dict(blocked_keys=sorted(set(truth.truth_key) | raw_keys),
                   blocked_scaffolds=sorted(set(truth.scaffold)), block_acyclic=False,
                   blocked_sources=sorted(aliases))
    (private/'holdout.json').write_text(json.dumps(holdout, indent=2)+'\n')
    exposure = pq.read_table(args.train, columns=['inchikey14', 'ingest_lib', 'normalized_smiles'],
                             filters=[('inchikey14', 'in', sorted(raw_keys | set(truth.truth_key)))])
    by_library = exposure.to_pandas().groupby('ingest_lib').inchikey14.nunique().to_dict()
    known_matches = len((raw_keys | set(truth.truth_key)) & args.train_keys)
    if args.require_unseen and (known_matches or by_library):
        raise ValueError('Frozen holdout unexpectedly shares an identity with Kaggle training data')
    manifest = dict(sampling=SEED, rdkit=rdBase.rdkitVersion, molecules=len(truth), spectra=len(scans),
                    molecules_by_source=truth.source.value_counts().to_dict(),
                    spectra_by_adduct=scans.adduct.value_counts().to_dict(),
                    source_availability=dict(group_counts), exclusions=dict(counts), prior_development_keys=dev_seen,
                    blocked_sources=holdout['blocked_sources'], blocked_scaffolds=len(holdout['blocked_scaffolds']),
                    train_structure_index_sha256=digest(args.structure_index) if args.structure_index else None,
                    matched_training_index_keys=known_matches,
                    exact_competition_normalization=args.exact_exposure,
                    exact_raw_identities_in_unmasked_competition_train=len(pc.unique(exposure['inchikey14'])),
                    unmasked_train_raw_identities_by_library=by_library,
                    source_urls=dict(URLS, **({'MassBank 2026.11':MASSBANK_URL} if args.massbank else {})),
                    source_hashes=dict(DIGESTS, **({'MassBank 2026.11 md5':MASSBANK_MD5,
                        'Natural product catalog sha256':digest(args.natural_catalog)} if args.massbank else {})),
                    scope='Reference and newly fitted components are held out by source, grouped generic scaffold and metric identity. Public pretrained checkpoint exposure is a separate limitation.',
                    output_sha256={p.relative_to(out).as_posix(): digest(p) for p in sorted(out.glob('*/*'))})
    if args.theobroma:
        manifest['source_hashes']['THEOBROMA v1.36 sha256'] = digest(args.theobroma)
        manifest['source_urls']['THEOBROMA v1.36'] = 'https://doi.org/10.5281/zenodo.22816330'
    for path in args.gnps:
        collection = next(iter(json.loads(path.read_text())))['library_membership']
        if 'GNPS_' + collection in manifest['molecules_by_source']:
            manifest['source_hashes'][collection + ' sha256'] = digest(path)
            manifest['source_urls'][collection] = f'https://external.gnps2.org/gnpslibrary/{collection}.json'
            manifest['source_urls'][collection + ' original study'] = GNPS_LITERATURE[collection]
    if args.mycomsb:
        manifest['source_urls']['MycoMSBase'] = MYCO_URL
        manifest['mycomsb_precursor_policy'] = ('Preserved the published focused-ion m/z exactly. The author resource '
            'does not state that this field is always an independently measured precursor; MS/MS fragment peaks '
            'are experimental Bruker MetaboScape library data.')
    if args.mycomsb_audit:
        manifest['source_hashes']['MycoMSBase independent accession audit sha256'] = digest(args.mycomsb_audit)
    (out/'panel_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--casmi-cache', type=Path, required=True)
    cli.add_argument('--massbank', type=Path)
    cli.add_argument('--natural-catalog', type=Path)
    cli.add_argument('--theobroma', type=Path)
    cli.add_argument('--gnps', type=Path, nargs='*', default=[])
    cli.add_argument('--mycomsb', type=Path)
    cli.add_argument('--mycomsb-audit', type=Path)
    cli.add_argument('--dev-labels', type=Path, nargs='+', required=True)
    cli.add_argument('--quota', nargs='+', required=True)
    cli.add_argument('--fill-source')
    cli.add_argument('--out', type=Path)
    cli.add_argument('--train', type=Path, default=Path('data/raw/train.parquet'))
    cli.add_argument('--schema', type=Path, default=Path('data/raw/test.parquet'))
    cli.add_argument('--structure-index', type=Path)
    cli.add_argument('--require-unseen', action='store_true')
    args = cli.parse_args()
    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError('Use competition scoring RDKit 2026.03.3')
    if args.require_unseen and not args.structure_index:
        raise ValueError('Strict unseen selection requires the training structure index')
    RDLogger.DisableLog('rdApp.warning')
    removed, dev_seen = dev_keys(args.dev_labels, args.train)
    args.train_keys = known_training_identities(args.structure_index) if args.structure_index else set()
    known_raw = removed | args.train_keys if args.require_unseen else removed
    groups, counts = collect(casmi_rows(args.casmi_cache), removed)
    if args.massbank:
        if not args.natural_catalog:
            raise ValueError('MassBank eligibility requires LOTUS and NPAtlas natural origin evidence')
        extra, exclusions = collect(massbank_rows(args.massbank,
                    natural_catalog(args.natural_catalog, args.theobroma), known_raw), removed)
        groups.update(extra)
        counts.update(exclusions)
    for path in args.gnps:
        extra, exclusions = collect(gnps_rows(path, known_raw), removed)
        groups.update(extra)
        counts.update(exclusions)
    if args.mycomsb:
        audited_ids = None
        if args.mycomsb_audit:
            audit = json.loads(args.mycomsb_audit.read_text())['rows']
            audited_ids = {accession for accession, record in audit.items()
                           if record['archive'] == 'Converted from Bruker MetaboScape .library export'}
        extra, exclusions = collect(mycomsb_rows(args.mycomsb, known_raw, audited_ids), removed)
        groups.update(extra)
        counts.update(exclusions)
    group_counts = Counter(source for source, _ in groups)
    groups_unseen = {(source, key):rows for (source, key), rows in groups.items()
                     if key not in args.train_keys and not any(row[4] in args.train_keys for row in rows)}
    unseen_counts = Counter(source for source, _ in groups_unseen)
    print(json.dumps({'available_groups':dict(group_counts), 'available_unseen_groups':dict(unseen_counts),
                      'exclusions':dict(counts)}, ensure_ascii=False), flush=True)
    if args.require_unseen:
        counts['known Kaggle training identity among eligible source groups'] += len(groups) - len(groups_unseen)
        groups, group_counts = groups_unseen, unseen_counts
    args.exact_exposure = None
    if args.out:
        quotas = {source:int(n) for source, n in (value.split('=',1) for value in args.quota)}
        if not 50 <= sum(quotas.values()) <= 100 or len(quotas) < 2 or min(quotas.values()) <= 0:
            raise ValueError('Freeze between 50 and 100 queries from multiple acquisition groups')
        if args.fill_source and args.fill_source not in quotas:
            raise ValueError('Backfill source must be one of the declared acquisition groups')
        selected = select(groups, quotas, args.fill_source)
        if args.require_unseen:
            detected, examined = exact_training_collisions(args.structure_index,
                {(source,key):rows for source,key,rows in selected})
            args.exact_exposure = dict(same_heavy_composition_training_rows_renormalized=examined,
                                       collision_identities_excluded=len(detected))
            if detected:
                groups = {x:y for x,y in groups.items() if x[1] not in detected}
                selected = select(groups, quotas, args.fill_source)
                detected, second = exact_training_collisions(args.structure_index,
                    {(source,key):rows for source,key,rows in selected})
                if detected:
                    raise ValueError('Replacement selection has additional normalized training matches')
                args.exact_exposure['same_heavy_composition_training_rows_renormalized'] += second
        write(args, selected, counts, group_counts, dev_seen)


if __name__ == '__main__':
    main()

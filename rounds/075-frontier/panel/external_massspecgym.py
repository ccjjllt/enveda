"""Build external experimental spectrum queries on Kaggle CPU."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


INPUT = Path('/kaggle/input')
OUTPUT = Path('/kaggle/working')
COMPETITION = 'enveda-casmi26-molecule-id-mass-spectra'
MSG_SHA256 = '0c9cc50450def3f0d4fe2dc09dea1105fc15e635db8c6656bc3e3be37a3bcd95'
ELEMENTS = re.compile(r'([A-Z][a-z]?)([0-9]*)')
ADDUCT = '[M+H]+'
EXTRA_ADDUCTS = {'[M-H]-': 'negative', '[M+Na]+': 'positive', '[M+NH4]+': 'positive'}
MASSBANK_ORIGINAL = 'enveda-075b-external-massbank-202611'
MASSBANK_ADDUCT_PANEL = 'enveda-075b-external-massbank-adducts'
PANEL_SIZE = 120
PROTON_MASS = 1.007276466621


def install_rdkit():
    try:
        installed = version('rdkit')
    except PackageNotFoundError:
        installed = None
    if installed not in ('2026.3.3', '2026.03.3'):
        abi = f'cp{sys.version_info.major}{sys.version_info.minor}'
        wheel = next(path for path in INPUT.rglob('rdkit-*.whl') if f'-{abi}-' in path.name)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-index',
                        '--no-deps', str(wheel)], check=True)


def init_chemistry(targets=None):
    global Chem, formula, tautomer, murcko, test_targets
    from rdkit import Chem, RDLogger, rdBase
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.Scaffolds import MurckoScaffold

    if rdBase.rdkitVersion != '2026.03.3':
        raise RuntimeError(f'Expected official RDKit 2026.03.3, got {rdBase.rdkitVersion}')
    RDLogger.DisableLog('rdApp.*')
    formula = rdMolDescriptors.CalcMolFormula
    tautomer = rdMolStandardize.TautomerEnumerator()
    murcko = MurckoScaffold
    test_targets = targets


def normalized(mol):
    mol = Chem.Mol(mol)
    Chem.RemoveStereochemistry(mol)
    return tautomer.Canonicalize(mol)


def heavy_composition(mol):
    return tuple((symbol, int(count or 1)) for symbol, count in ELEMENTS.findall(formula(mol)) if symbol != 'H')


def scaffolds(mol):
    scaffold = murcko.GetScaffoldForMol(mol)
    if not scaffold.GetNumAtoms():
        return '', ''
    exact = Chem.MolToSmiles(scaffold, isomericSmiles=False)
    # The generic ring/linker topology is invariant to bond-order changes and
    # terminal substituents, so its absence is a conservative scaffold tier.
    generic = murcko.GetScaffoldForMol(murcko.MakeScaffoldGeneric(scaffold))
    return exact, Chem.MolToSmiles(generic, isomericSmiles=False)


def describe_external(item):
    key, smiles = item
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or Chem.GetFormalCharge(mol) or '.' in smiles:
            return None
        raw = Chem.MolToInchiKey(mol)[:14]
        canonical = normalized(mol)
        identity = Chem.MolToInchiKey(canonical)[:14]
        scaffold, generic = scaffolds(canonical)
        if raw and identity:
            return (key, smiles), dict(raw_key=key or raw, raw_rdkit_key=raw, candidate_id=identity,
                                         scaffold=scaffold, generic=generic, composition=heavy_composition(mol))
    except (RuntimeError, ValueError):
        return None


def probe_reference(record):
    raw_key, smiles, track_scaffolds = record
    test_keys, compositions, exact_scaffolds, generic_scaffolds = test_targets
    matched_keys, exact_seen, generic_seen = set(), set(), set()
    if raw_key in test_keys:
        matched_keys.add(raw_key)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return matched_keys, exact_seen, generic_seen, 1
        variants = Chem.GetMolFrags(mol, asMols=True) if '.' in smiles else (mol,)
        for variant in variants:
            if heavy_composition(variant) in compositions:
                raw = Chem.MolToInchiKey(variant)[:14]
                if raw in test_keys:
                    matched_keys.add(raw)
                canonical = Chem.MolToInchiKey(normalized(variant))[:14]
                if canonical in test_keys:
                    matched_keys.add(canonical)
            if track_scaffolds and variant.GetRingInfo().NumRings():
                exact, generic = scaffolds(variant)
                if exact in exact_scaffolds:
                    exact_seen.add(exact)
                if generic in generic_scaffolds:
                    generic_seen.add(generic)
    except (RuntimeError, ValueError):
        return matched_keys, exact_seen, generic_seen, 1
    return matched_keys, exact_seen, generic_seen, 0


def unique_enveda(train):
    records = set()
    for batch in train.iter_batches(columns=['inchikey14', 'normalized_smiles'], batch_size=65536):
        records.update((str(raw) if raw else '', smiles)
                       for raw, smiles in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True)
                       if smiles)
    return records


def scan(pool, records, track_scaffolds):
    matched, exact_seen, generic_seen, failed = set(), set(), set(), 0
    topology_records = defaultdict(list)
    mapped = pool.map(probe_reference, ((raw, smiles, track_scaffolds) for raw, smiles in records), chunksize=256)
    for record, (keys, exact, generic, invalid) in zip(records, mapped, strict=True):
        matched.update(keys)
        exact_seen.update(exact)
        generic_seen.update(generic)
        for topology in generic:
            topology_records[topology].append(record)
        failed += invalid
    return matched, exact_seen, generic_seen, failed, topology_records


def reference_scaffold_matches(record):
    raw_key, smiles = record
    selected_scaffolds, selected_topologies = test_targets
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        variants = Chem.GetMolFrags(mol, asMols=True) if '.' in smiles else (mol,)
        rows = []
        for variant in variants:
            if not variant.GetRingInfo().NumRings() or scaffolds(variant)[1] not in selected_topologies:
                continue
            canonical = normalized(variant)
            if scaffolds(canonical)[0] in selected_scaffolds:
                identity = Chem.MolToInchiKey(canonical)[:14]
                if raw_key and identity:
                    rows.append(dict(raw_key=raw_key, candidate_id=identity))
        return rows
    except (RuntimeError, ValueError):
        return []


def parse_spectrum(row):
    try:
        mz = np.asarray(row.mzs.split(',') if isinstance(row.mzs, str) else row.mzs, dtype=float)
        intensity = np.asarray(row.intensities.split(',') if isinstance(row.intensities, str) else row.intensities,
                               dtype=float)
        precursor = float(row.precursor_mz)
    except (TypeError, ValueError):
        return None
    if mz.shape != intensity.shape or not np.isfinite(precursor) or precursor <= 0:
        return None
    finite = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
    mz, intensity = mz[finite], intensity[finite]
    if len(mz) < 3:
        return None
    order = np.argsort(mz, kind='stable')
    return mz[order].tolist(), (intensity[order] / intensity.max()).tolist(), precursor


def choose_spectra(rows, source_column, rng):
    rows = list(rows.itertuples(index=False))
    eligible = [(row, parse_spectrum(row)) for row in rows]
    eligible = [(row, spectrum) for row, spectrum in eligible if spectrum is not None]
    if not eligible:
        return [], 'unusable'
    random_order = rng.permutation(len(eligible))
    eligible = [eligible[index] for index in random_order]
    groups = defaultdict(dict)
    if source_column:
        for row, spectrum in eligible:
            source = getattr(row, source_column)
            if pd.notna(source) and str(source).strip() and pd.notna(row.instrument_type) and pd.notna(row.collision_energy):
                key = (str(source).strip(), str(getattr(row, 'instrument_name', row.instrument_type)), row.adduct)
                groups[key].setdefault(round(float(row.collision_energy), 3), (row, spectrum))
    if groups:
        strongest = max(groups.values(), key=len)
        if len(strongest) >= 2:
            energies = sorted(strongest)
            indices = np.rint(np.linspace(0, len(energies) - 1, min(len(energies), 4))).astype(int)
            return [strongest[energies[i]] for i in indices], 'same_source_multi_ce'
    return [min(eligible, key=lambda item: (pd.isna(item[0].collision_energy), pd.isna(item[0].instrument_type)))], 'single_spectrum'


def official_msg_path():
    source = next(path for path in INPUT.rglob('MassSpecGym.tsv') if 'casmi26-massspecgym' in path.parts)
    with source.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != MSG_SHA256:
            raise ValueError('The Kaggle mirror differs from the official MassSpecGym v1 TSV')
    return source


def msg_reference_records(msg):
    seen = msg[msg['fold'].isin(('train', 'val'))].dropna(subset=['inchikey', 'smiles'])
    return set(zip(seen['inchikey'].astype(str).str.slice(0, 14), seen['smiles'], strict=True))


def load_massspecgym():
    source = official_msg_path()
    fields = pd.read_csv(source, sep='\t', nrows=0).columns
    source_column = next((name for name in ('source_library', 'ingest_lib', 'library_source') if name in fields), None)
    columns = ['identifier', 'mzs', 'intensities', 'smiles', 'inchikey', 'precursor_mz', 'adduct',
               'instrument_type', 'collision_energy', 'fold', 'parent_mass']
    if source_column:
        columns.append(source_column)
    msg = pd.read_csv(source, sep='\t', usecols=columns, dtype={'identifier': str, 'inchikey': str})
    msg['raw_key'] = msg['inchikey'].str.slice(0, 14)
    msg['collision_energy'] = pd.to_numeric(msg['collision_energy'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    test = msg[msg['fold'].eq('test')].copy()
    return test, source_column, msg_reference_records(msg), dict(source='MassSpecGym official v1 TSV',
                                                               tsv_sha256=MSG_SHA256, official_fold='test')


def read_massbank(source_name):
    source_slug, filename = {
        'massbank': ('enveda-075b-massbank-202611-source', 'massbank_added_positive.parquet'),
        'massbank_adducts': ('enveda-075b-massbank-202611-adduct-source', 'massbank_added_adducts.parquet'),
        'massbank_historical': ('enveda-075b-massbank-202603-positive-source', 'massbank_added_positive.parquet'),
    }[source_name]
    receipt = next(path for path in INPUT.rglob('massbank_source.json') if source_slug in path.parts)
    provenance = json.loads(receipt.read_text())
    source = receipt.with_name(filename)
    if provenance['retained']:
        spectra = pd.read_parquet(source)
        spectra['raw_key'] = spectra['inchikey'].astype('string').str.slice(0, 14).fillna('')
    else:
        spectra = pd.DataFrame(columns=['identifier', 'adduct', 'polarity', 'raw_key', 'smiles', 'candidate_id',
                                        'collision_energy', 'source_library'])
    msg = pd.read_csv(official_msg_path(), sep='\t', usecols=['inchikey', 'smiles', 'fold'])
    return spectra, 'source_library', msg_reference_records(msg), dict(
        source=f'{provenance["release"]} accession additions', official_fold=None, massbank_release=provenance)


def earlier_massbank_panel(slug, expected_size):
    path = next(path for path in INPUT.rglob('labels.parquet') if slug in path.parts)
    labels = pd.read_parquet(path, columns=['candidate_id', 'scaffold']).drop_duplicates()
    if len(labels) != expected_size or not labels['candidate_id'].is_unique:
        raise ValueError(f'Expected exactly {expected_size} structures in {slug}')
    holdout = pd.read_parquet(path.with_name('heldout_structures.parquet'), columns=['candidate_id', 'raw_key'])
    aliases = holdout[holdout['candidate_id'].isin(labels['candidate_id'])]['raw_key']
    queries = pd.read_parquet(path.with_name('queries.parquet'), columns=['source_library'])
    return set(labels['candidate_id']) | set(aliases), set(labels['scaffold']) - {''}, set(queries['source_library'].dropna())


def counts_by_adduct(frame, adducts):
    counts = {}
    for adduct in adducts:
        group = frame[frame['adduct'].eq(adduct)]
        counts[adduct] = dict(spectra=len(group),
                              structures=int(group['candidate_id'].nunique()) if 'candidate_id' in group else None)
    return counts


def prepare(source_name='msg'):
    started = time.monotonic()
    install_rdkit()
    init_chemistry()
    extra_adducts = source_name == 'massbank_adducts'
    historical = source_name == 'massbank_historical'
    track_prior_panels = historical or extra_adducts
    test, source_column, msg_records, provenance = (load_massspecgym() if source_name == 'msg'
                                                    else read_massbank(source_name))
    adducts = EXTRA_ADDUCTS if extra_adducts else {ADDUCT: 'positive'}
    test = test[test['adduct'].isin(adducts)].dropna(subset=['raw_key', 'smiles']).copy()
    if track_prior_panels:
        if not test['polarity'].eq(test['adduct'].map(adducts)).all():
            raise ValueError('Unexpected MassBank accession polarity')
        add_counts = dict(licensed_experimental=counts_by_adduct(test, adducts))
        earlier_panels = {MASSBANK_ORIGINAL: earlier_massbank_panel(MASSBANK_ORIGINAL, 110)}
        if historical:
            earlier_panels[MASSBANK_ADDUCT_PANEL] = earlier_massbank_panel(MASSBANK_ADDUCT_PANEL, 48)
        prior_scaffolds = set().union(*(panel[1] for panel in earlier_panels.values()))
        prior_sources = set().union(*(panel[2] for panel in earlier_panels.values()))
    test['collision_energy'] = pd.to_numeric(test['collision_energy'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    test_records = sorted(set(zip(test['raw_key'], test['smiles'], strict=True)))

    workers = min(4, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers, initializer=init_chemistry) as pool:
        descriptions = dict(row for row in pool.map(describe_external, test_records, chunksize=32) if row is not None)
    keys = {record[field] for record in descriptions.values() for field in ('raw_key', 'raw_rdkit_key', 'candidate_id')}
    targets = (keys, {record['composition'] for record in descriptions.values()},
               {record['scaffold'] for record in descriptions.values()} - {''},
               {record['generic'] for record in descriptions.values()} - {''})
    train_source = next(path.parent for path in INPUT.rglob('train.parquet') if path.parent.name.lower() == COMPETITION)
    enveda = unique_enveda(pq.ParquetFile(train_source / 'train.parquet'))
    print(f'{provenance["source"]} {list(adducts)}: {len(test)} spectra and {len(descriptions)} standardized forms. '
          f'Comparing to {len(enveda)} Enveda structure records and {len(msg_records)} MSG train/val records.', flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=init_chemistry, initargs=(targets,)) as pool:
        enveda_matches, exact_seen, generic_seen, enveda_invalid, topology_records = scan(pool, enveda, True)
        msg_matches, _, _, msg_invalid, _ = scan(pool, msg_records, False)
    matches = enveda_matches | msg_matches if source_name == 'msg' or historical else enveda_matches
    enveda_excluded = set()
    msg_excluded = set()
    prior_excluded = defaultdict(set)
    by_record = {}
    by_identity = defaultdict(list)
    excluded_identities = set()
    for key, record in descriptions.items():
        aliases = {record[field] for field in ('raw_key', 'raw_rdkit_key', 'candidate_id')}
        identity = record['candidate_id']
        if aliases & enveda_matches:
            enveda_excluded.add(identity)
        if aliases & msg_matches:
            msg_excluded.add(identity)
        if track_prior_panels:
            for panel_slug, (prior_keys, _, _) in earlier_panels.items():
                if aliases & prior_keys:
                    prior_excluded[panel_slug].add(identity)
        if aliases & matches:
            excluded_identities.add(identity)
        by_record[key] = identity
        by_identity[identity].append(record)
    test['candidate_id'] = [by_record.get(key) for key in zip(test['raw_key'], test['smiles'], strict=True)]
    test = test.dropna(subset=['candidate_id'])
    if track_prior_panels:
        add_counts['standardized'] = counts_by_adduct(test, adducts)
        add_counts['enveda_overlap'] = counts_by_adduct(test[test['candidate_id'].isin(enveda_excluded)], adducts)
        add_counts['original_panel_overlap'] = counts_by_adduct(
            test[test['candidate_id'].isin(prior_excluded[MASSBANK_ORIGINAL])], adducts)
        if historical:
            add_counts['adduct_panel_overlap'] = counts_by_adduct(
                test[test['candidate_id'].isin(prior_excluded[MASSBANK_ADDUCT_PANEL])], adducts)
            add_counts['msg_train_val_overlap'] = counts_by_adduct(test[test['candidate_id'].isin(msg_excluded)], adducts)
        excluded_identities.update(set().union(*prior_excluded.values()))
    test = test[~test['candidate_id'].isin(excluded_identities)]
    if track_prior_panels:
        usable = [parse_spectrum(row) is not None for row in test.itertuples(index=False)]
        test = test.loc[usable]
        new_scaffold_ids = {identity for identity, records in by_identity.items()
                            if records[0]['scaffold'] and records[0]['scaffold'] not in prior_scaffolds}
        add_counts['eligible'] = counts_by_adduct(test, adducts)
        add_counts['eligible_new_murcko'] = counts_by_adduct(test[test['candidate_id'].isin(new_scaffold_ids)], adducts)
        extra_report = dict(adduct_polarity=adducts, by_adduct=add_counts,
                            independent_structures=int(test.candidate_id.nunique()),
                            eligible_structures_with_new_murcko=int(test[test['candidate_id'].isin(new_scaffold_ids)]
                                                                   .candidate_id.nunique()),
                            original_panel_overlap_structures=len(prior_excluded[MASSBANK_ORIGINAL]),
                            original_panel_structures=110, minimum_independent_structures=41 if historical else 30,
                            official_rdkit='2026.03.3')
        if historical:
            extra_report.update(adduct_panel_overlap_structures=len(prior_excluded[MASSBANK_ADDUCT_PANEL]),
                                msg_train_val_overlap_structures=len(msg_excluded),
                                eligible_sources=test.groupby(source_column).agg(spectra=('identifier', 'size'),
                                                                                structures=('candidate_id', 'nunique'))
                                .to_dict(orient='index'))
        if extra_report['independent_structures'] < extra_report['minimum_independent_structures']:
            report = dict(**provenance, **extra_report, panel_exported=False,
                          elapsed_s=round(time.monotonic() - started, 1))
            (OUTPUT / 'panel.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report), flush=True)
            return
    print(f'After exact structure exclusion: {test.candidate_id.nunique()} independent structures, '
          f'{len(enveda_excluded)} Enveda overlaps, {len(msg_excluded)} MSG train/val overlaps in source.', flush=True)

    rng = np.random.default_rng(7502)
    ranked = []
    for identity, rows in test.groupby('candidate_id', sort=True):
        record = by_identity[identity][0]
        if historical:
            unseen_sources = ~rows[source_column].isin(prior_sources)
            if unseen_sources.any():
                rows = rows[unseen_sources]
        if extra_adducts:
            chosen = [choose_spectra(arm, source_column, rng) for _, arm in rows.groupby('adduct', sort=True)]
            spectra = [spectrum for arm, _ in chosen for spectrum in arm]
            mode = ('same_source_multi_ce' if any(m == 'same_source_multi_ce' for _, m in chosen)
                    else 'multi_adduct' if len(chosen) > 1 else 'single_spectrum')
        else:
            spectra, mode = choose_spectra(rows, source_column, rng)
        if not spectra:
            continue
        if not record['scaffold']:
            tier = 'acyclic'
        elif record['generic'] and record['generic'] not in generic_seen:
            tier = 'generic_absent_in_enveda'
        elif record['scaffold'] not in exact_seen:
            tier = 'exact_unmatched_in_raw_enveda'
        else:
            tier = 'raw_scaffold_seen_in_enveda'
        priority = ('generic_absent_in_enveda', 'exact_unmatched_in_raw_enveda',
                    'raw_scaffold_seen_in_enveda', 'acyclic').index(tier)
        ranked.append((track_prior_panels and identity not in new_scaffold_ids,
                       historical and getattr(spectra[0][0], source_column) in prior_sources,
                       priority, mode != 'same_source_multi_ce', rng.random(), identity, tier, mode, spectra))
    ranked.sort()
    selected = []
    used_scaffolds = set()
    panel_size = 110 if historical else PANEL_SIZE
    for unique_scaffold in ((True, False) if track_prior_panels else (True,)):
        chosen_ids = {entry[0] for entry in selected}
        for _, _, _, _, _, identity, tier, mode, spectra in ranked:
            if len(selected) == panel_size:
                break
            scaffold = by_identity[identity][0]['scaffold']
            if identity in chosen_ids or (unique_scaffold and scaffold in used_scaffolds):
                continue
            # Fill in other valid identities when unique scaffold choices run out.
            used_scaffolds.add(scaffold)
            selected.append((identity, tier, mode, spectra))
    if not selected or (source_name == 'msg' and len(selected) < 80):
        raise ValueError(f'Only {len(selected)} nonoverlapping MSG test structures with distinct scaffolds')

    query_rows, pipeline_rows, label_rows, holdout_rows = [], [], [], []
    near_exact_precursors = 0
    prefix = 'H' if historical else 'A' if extra_adducts else 'U'
    for index, (identity, tier, mode, spectra) in enumerate(selected, 1):
        molecule_id = f'{prefix}{index:03d}'
        first = spectra[0][0]
        label_rows.append(dict(molecule_id=molecule_id, candidate_id=identity, panel='U',
                               scaffold=by_identity[identity][0]['scaffold'], scaffold_tier=tier, spectrum_group=mode,
                               truth_neutral_mass=float(first.parent_mass) if pd.notna(first.parent_mass) else None,
                               truth_smiles=first.smiles, msg_train_val_match=identity in msg_excluded,
                               truth_license=getattr(first, 'license', 'MIT')))
        for record in by_identity[identity]:
            for key in {record['raw_key'], record['raw_rdkit_key'], identity}:
                holdout_rows.append(dict(raw_key=key, candidate_id=identity))
        for row, (mz, intensity, precursor) in spectra:
            instrument = str(row.instrument_type) if pd.notna(row.instrument_type) else 'unknown'
            ce = float(row.collision_energy) if pd.notna(row.collision_energy) else None
            original_ce = getattr(row, 'collision_energy_orig', '') or 'unknown'
            ce_units = getattr(row, 'collision_energy_orig_units', 'unknown')
            if row.adduct == ADDUCT and pd.notna(row.parent_mass) and abs(precursor - float(row.parent_mass) - PROTON_MASS) <= precursor * 1e-7:
                near_exact_precursors += 1
            query_rows.append(dict(query_id=row.identifier, molecule_id=molecule_id, mz=mz, intensity=intensity,
                                   precursor_mz=precursor, adduct=row.adduct, collision_energy=ce,
                                   collision_energy_orig=original_ce, collision_energy_orig_units=ce_units,
                                   instrument_type=instrument, polarity=adducts[row.adduct],
                                   source_library=getattr(row, source_column) if source_column else None))
            if historical:
                query_rows[-1].update(instrument_name=row.instrument_name, license=row.license)
            pipeline_rows.append(dict(molecule_id=molecule_id, spectrum_id=row.identifier, ms2_mzs=mz,
                                      ms2_normalized_intensities=intensity, base_peak_intensity=1.0,
                                      adduct=row.adduct, ionization_mode=adducts[row.adduct], instrument_type=instrument,
                                      precursor_mz=precursor, collision_energy_orig=original_ce,
                                      collision_energy_ev=[] if ce is None else [ce], collision_energy_orig_units=ce_units))

    selected_topologies = {by_identity[identity][0]['generic'] for identity, tier, _, _ in selected
                           if tier in ('exact_unmatched_in_raw_enveda', 'raw_scaffold_seen_in_enveda')}
    scaffold_rows = []
    if selected_topologies:
        references = {record for topology in selected_topologies for record in topology_records[topology]}
        target_scaffolds = {by_identity[identity][0]['scaffold'] for identity, _, _, _ in selected
                            if by_identity[identity][0]['generic'] in selected_topologies}
        print(f'Checking {len(references)} Enveda records with selected shared scaffold topologies.', flush=True)
        with ProcessPoolExecutor(max_workers=workers, initializer=init_chemistry,
                                 initargs=((target_scaffolds, selected_topologies),)) as pool:
            scaffold_rows = [row for rows in pool.map(reference_scaffold_matches, references, chunksize=64) for row in rows]
        holdout_rows.extend(scaffold_rows)
    schema = pq.ParquetFile(train_source / 'test.parquet').schema_arrow.remove_metadata()
    pq.write_table(pa.Table.from_pylist(pipeline_rows, schema=schema), OUTPUT / 'pipeline_queries.parquet')
    pd.DataFrame(query_rows).to_parquet(OUTPUT / 'queries.parquet', index=False)
    pd.DataFrame(label_rows).to_parquet(OUTPUT / 'labels.parquet', index=False)
    pd.DataFrame(holdout_rows).drop_duplicates().to_parquet(OUTPUT / 'heldout_structures.parquet', index=False)
    report = dict(panel='U', external_test=True, **provenance,
                  adduct=list(adducts) if extra_adducts else ADDUCT,
                  molecules=len(label_rows), spectra=len(query_rows),
                  eligible_structures_after_overlap_exclusion=int(test.candidate_id.nunique()),
                  distinct_scaffolds_after_overlap_exclusion=len({by_identity[row[5]][0]['scaffold'] for row in ranked}),
                  shared_scaffold_reference_raw_keys=len({row['raw_key'] for row in scaffold_rows}),
                  formula_or_candidates_used_for_selection=False, raw_sourced_library_column=source_column,
                  scaffold_tiers=dict(Counter(row['scaffold_tier'] for row in label_rows)),
                  spectrum_groups=dict(Counter(row['spectrum_group'] for row in label_rows)),
                  train_overlapping_test_structures_excluded=len(enveda_excluded),
                  msg_train_val_structures_in_source=len(msg_excluded),
                  selected_msg_train_val_matches=sum(row['msg_train_val_match'] for row in label_rows),
                  enveda_reference_smiles_scan_failures=enveda_invalid, msg_reference_smiles_scan_failures=msg_invalid,
                  author_ce_values=sum(row['collision_energy'] is not None for row in query_rows),
                  precursors_within_point_one_ppm_of_exact_protonated_mass=near_exact_precursors,
                  precursor_field='MassBank PRECURSOR_M/Z; records may state calculated exact m/z'
                  if source_name != 'msg' else 'MassSpecGym precursor_mz',
                  elapsed_s=round(time.monotonic() - started, 1))
    if track_prior_panels:
        selected_rows = pd.DataFrame(query_rows).merge(pd.DataFrame(label_rows)[['molecule_id', 'candidate_id']],
                                                       on='molecule_id', validate='many_to_one')
        report.update(extra_report, panel_exported=True,
                      selected_structures_with_new_murcko=sum(identity in new_scaffold_ids for identity, _, _, _ in selected))
        add_counts['selected'] = counts_by_adduct(selected_rows, adducts)
        if historical:
            report['selected_sources'] = selected_rows.groupby('source_library').agg(
                spectra=('query_id', 'size'), structures=('candidate_id', 'nunique')).to_dict(orient='index')
    (OUTPUT / 'panel.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['msg', 'massbank', 'massbank_adducts', 'massbank_historical'], default='msg')
    prepare(parser.parse_args().source)

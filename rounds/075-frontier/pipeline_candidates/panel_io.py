"""Kaggle-side input preparation and candidate export for the 075b panel."""

from collections import defaultdict
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def prepare_panel():
    inputs = Path('/kaggle/input')
    root = Path('/kaggle/working')
    panel_inputs = [path.parent for path in inputs.rglob('pipeline_queries.parquet')
                    if (path.parent / 'heldout_structures.parquet').is_file()]
    if len(panel_inputs) != 1:
        raise ValueError(f'Expected exactly one 075b query artifact, found {panel_inputs}')
    panel = panel_inputs[0]
    source = next(path for path in inputs.rglob('train.parquet')
                  if path.parent.name.lower() == 'enveda-casmi26-molecule-id-mass-spectra')
    heldout = pd.read_parquet(panel / 'heldout_structures.parquet')
    labels = pd.read_parquet(panel / 'labels.parquet')
    ids = labels['molecule_id'].astype(str)
    query_ids = pd.read_parquet(panel / 'pipeline_queries.parquet', columns=['molecule_id'])
    if ids.empty or not ids.is_unique or set(ids) != set(query_ids['molecule_id'].astype(str)):
        raise ValueError('075b labels and queries disagree on molecule IDs')
    if set(labels['panel']) != {'U'} or not set(labels['candidate_id']).issubset(heldout['candidate_id']):
        raise ValueError('075b U identities missing from the reference exclusion list')
    comp = root / 'panel_competition'
    comp.mkdir(exist_ok=True)
    os.symlink(panel / 'pipeline_queries.parquet', comp / 'test.parquet')
    pd.DataFrame({'molecule_id': ids, 'smiles': 'CCO'}).to_csv(comp / 'sample_submission.csv', index=False)

    train = pq.ParquetFile(source)
    identities = set(heldout['candidate_id'].dropna())
    scaffolds = set(labels['scaffold'].dropna()) - {''}
    known_keys = set(heldout['raw_key'].dropna())
    panel_report_path = panel / 'panel.json'
    panel_report = json.loads(panel_report_path.read_text()) if panel_report_path.is_file() else {}
    complete_external_mask = (
        panel_report.get('external_test') is True
        and 'shared_scaffold_reference_raw_keys' in panel_report
        and panel_report.get('enveda_reference_smiles_scan_failures') == 0
    )
    extra_keys = set() if complete_external_mask else extra_holdout_keys(train, known_keys, identities, scaffolds)
    raw_keys = pa.array(sorted(known_keys | extra_keys), type=pa.string())
    key_column = train.schema_arrow.get_field_index('inchikey14')
    smiles_column = train.schema_arrow.get_field_index('normalized_smiles')
    retained_rows = 0
    with pq.ParquetWriter(comp / 'train.parquet', train.schema_arrow) as writer:
        for batch in train.iter_batches(batch_size=32768):
            raw_identity = pc.cast(batch.column(key_column), pa.string())
            identifiable = pc.and_(pc.is_valid(raw_identity), pc.is_valid(batch.column(smiles_column)))
            keep = pc.and_(identifiable, pc.invert(pc.is_in(raw_identity, value_set=raw_keys)))
            filtered = batch.filter(keep)
            writer.write_batch(filtered)
            retained_rows += len(filtered)
    if retained_rows == 0:
        raise ValueError('No training reference spectra remain after applying the holdout')

    os.environ['INSTRUMENT_COMP'] = str(comp)
    os.environ['PANEL_ORIGINAL_TRAIN'] = str(source)
    os.environ['PANEL_ENGINE1_SCORES'] = str(root / 'engine1_candidates.json')
    audit = {
        'query_source': str(panel),
        'query_molecules': len(ids),
        'query_spectra': len(query_ids),
        'excluded_reference_raw_identities': len(raw_keys),
        'excluded_additional_reference_aliases': len(extra_keys),
        'used_precomputed_reference_mask': complete_external_mask,
        'retained_train_spectra': retained_rows,
        'removed_train_spectra': train.metadata.num_rows - retained_rows,
        'engine1_linked_external_extension': 'disabled for holdout; use filtered train spectra',
        'engine2_training_candidate_pool': 'unchanged structures from original train scalar columns',
        'base_supervised_training_exposure': 'existing weights, overlap unknown or possible',
    }
    return ids.tolist(), identities, scaffolds, audit


def extra_holdout_keys(train, known_keys, identities, scaffolds):
    """The CPU panel indexed one neutral single-fragment representation per key."""
    charge = re.compile(r'\[[^\]]*[+-]')
    representative = {}
    variants = defaultdict(set)
    for batch in train.iter_batches(columns=['inchikey14', 'normalized_smiles'], batch_size=65536):
        for key, smiles in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            if not key or not smiles or key in known_keys:
                continue
            original = representative.setdefault(key, smiles)
            if smiles != original:
                variants[key].update((original, smiles))
            if '.' in smiles or charge.search(smiles):
                variants[key].add(smiles)
    return {key for key, forms in variants.items()
            if any(excluded_reference(smiles, identities, scaffolds) for smiles in forms)}


@lru_cache(maxsize=1)
def chemistry():
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    RDLogger.DisableLog('rdApp.*')
    return Chem, rdMolStandardize.TautomerEnumerator(), rdMolDescriptors.CalcMolFormula, MurckoScaffoldSmiles


@lru_cache(maxsize=200000)
def describe(smiles, scaffold=False):
    Chem, tautomer, mol_formula, murcko = chemistry()
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        Chem.RemoveStereochemistry(molecule)
        molecule = tautomer.Canonicalize(molecule)
        identity = Chem.MolToInchiKey(molecule)[:14]
        if identity:
            return identity, mol_formula(molecule), murcko(mol=molecule) if scaffold else None
    except (RuntimeError, TypeError, ValueError):
        return None


def excluded_reference(smiles, identities, scaffolds):
    structure = describe(smiles, scaffold=True)
    if structure is None or structure[0] in identities or structure[2] in scaffolds:
        return True
    return '.' in smiles and any(excluded_reference(fragment, identities, scaffolds)
                                 for fragment in smiles.split('.'))


def retrieval_metrics(final_rank, union_rank):
    recall = float(final_rank.notna().mean())
    return dict(
        molecules=len(final_rank),
        rank1=int(final_rank.eq(1).sum()),
        recall_at_25=recall,
        mrr_at_25=float((1 / final_rank).fillna(0).mean()),
        recall_at_union=float(union_rank.notna().mean()),
        truth_in_union_only=int((final_rank.isna() & union_rank.notna()).sum()),
    )


def export_candidates(root, panel_ids, audit):
    root = Path(root)
    submission = pd.read_csv(root / 'submission.csv', dtype=str, keep_default_na=False)
    if not submission['molecule_id'].is_unique or set(submission['molecule_id']) != set(panel_ids):
        raise ValueError('Final pipeline output does not match the fixed panel')
    readings = json.loads((root / 'e004_candidate_inputs.json').read_text())
    engine1 = json.loads((root / 'engine1_candidates.json').read_text())
    receipt = json.loads((root / 'execution_receipt.json').read_text())
    if not receipt['is_full_run'] or receipt['processed_molecules'] != len(panel_ids):
        raise ValueError('The baseline did not process the full fixed panel')

    def candidate(mid, rank, smiles):
        info = describe(smiles)
        if info is None:
            raise ValueError(f'Invalid candidate in {mid} at rank {rank}: {smiles!r}')
        return dict(molecule_id=mid, candidate_id=info[0], smiles=smiles, formula=info[1], rank=rank)

    final_rows = []
    for mid, text in submission.itertuples(index=False):
        smiles = text.split(';')
        if not 1 <= len(smiles) <= 25:
            raise ValueError(f'{mid} has {len(smiles)} final candidates')
        final_rows.extend(candidate(mid, rank, smi) for rank, smi in enumerate(smiles, 1))
    final = pd.DataFrame(final_rows)
    final.to_parquet(root / 'candidates.parquet', index=False)

    source_rows = []
    apex_path = root / 'apex_external_candidates.json'
    apex = json.loads(apex_path.read_text()) if apex_path.is_file() else {}
    for mid in panel_ids:
        engine2 = readings['engine2'].get(mid, {})
        pubchem = readings['pubchem'].get(mid, {})
        pubchem_evidence = pubchem.get('promotion_evidence', {})
        pubchem_raw_fz = [pubchem_evidence.get(key, {}).get('raw_fz') for key in pubchem.get('pc_keys', [])]
        arms = (
            ('engine1', engine1[mid]['smiles'], {'scs': engine1[mid]['scs']}),
            ('engine2', engine2.get('smiles', []),
             {'blend': engine2.get('blend', []), 'frag_ad': engine2.get('frag_ad', [])}),
            ('pubchem', pubchem.get('pc', []), {'pc_raw_fz': pubchem_raw_fz}),
            ('prefusion', readings['prefusion'].get(mid, []), {}),
            ('fusion_overflow', readings['fusion_overflow'].get(mid, []), {}),
            ('apex_external', [apex[mid]['smiles']] if mid in apex else [],
             {'apex_cosine': [apex[mid]['cosine']] if mid in apex else []}),
        )
        for source, smiles, scores in arms:
            for index, smi in enumerate(smiles):
                if describe(smi) is None:
                    continue
                row = candidate(mid, index + 1, smi)
                row['source'] = source
                row.update({name: values[index] if index < len(values) else None
                            for name, values in scores.items()})
                source_rows.append(row)
    sources = pd.DataFrame(source_rows)
    sources.to_parquet(root / 'source_candidates.parquet', index=False)

    final_identity = final.drop_duplicates(['molecule_id', 'candidate_id'])
    union = pd.concat([final, sources], ignore_index=True).drop_duplicates(['molecule_id', 'candidate_id'])
    labels = pd.read_parquet(Path(audit['query_source']) / 'labels.parquet')
    final_matches = labels.merge(final_identity, on=['molecule_id', 'candidate_id'], how='left')
    union_matches = labels.merge(union, on=['molecule_id', 'candidate_id'], how='left')
    final_rank = final_matches.set_index('molecule_id')['rank']
    union_rank = union_matches.set_index('molecule_id')['rank']
    metrics = retrieval_metrics(final_rank, union_rank)
    pool_keys = set(pd.read_parquet(engine1['pool_meta_path'], columns=['key'])['key'])
    pool_membership = labels['candidate_id'].isin(pool_keys)
    audit['by_stored_main_pool_key'] = {}
    for name, selected in (('matched', pool_membership), ('unmatched', ~pool_membership)):
        mids = labels.loc[selected, 'molecule_id']
        if not mids.empty:
            audit['by_stored_main_pool_key'][name] = retrieval_metrics(final_rank.loc[mids], union_rank.loc[mids])
    if 'scaffold_tier' in labels:
        audit['by_scaffold_tier'] = {
            tier: retrieval_metrics(final_rank.loc[group['molecule_id']], union_rank.loc[group['molecule_id']])
            for tier, group in labels.groupby('scaffold_tier')
        }
    query_adducts = pd.read_parquet(Path(audit['query_source']) / 'pipeline_queries.parquet',
                                   columns=['molecule_id', 'adduct']).drop_duplicates()
    if query_adducts['adduct'].nunique() > 1 or not query_adducts['adduct'].eq('[M+H]+').all():
        audit['by_query_adduct'] = {
            adduct: retrieval_metrics(final_rank.loc[group['molecule_id']], union_rank.loc[group['molecule_id']])
            for adduct, group in query_adducts.groupby('adduct')
        }
    observed = sources.merge(labels[['molecule_id', 'candidate_id']], on=['molecule_id', 'candidate_id'])
    audit['truth_rank_by_source'] = {}
    for name in sorted(sources['source'].unique()):
        ranks = observed.loc[observed['source'].eq(name)].groupby('molecule_id')['rank'].min()
        ranks = ranks.reindex(final_rank.index)
        top25 = ranks.where(ranks.le(25))
        audit['truth_rank_by_source'][name] = {
            'found': int(ranks.notna().sum()),
            'rank1': int(ranks.eq(1).sum()),
            'recall_at_25': float(top25.notna().mean()),
            'mrr_at_25': float((1 / top25).fillna(0).mean()),
            'truth_outside_final_25': int((ranks.notna() & final_rank.isna()).sum()),
        }
    audit.update(
        final_candidates=len(final), final_unique_identities=len(final_identity),
        union_unique_identities=len(union), source_candidates=sources['source'].value_counts().to_dict(),
        **metrics,
        top25_reranking_oracle_headroom=metrics['recall_at_25'] - metrics['mrr_at_25'],
        candidate_union_oracle_headroom=metrics['recall_at_union'] - metrics['mrr_at_25'],
        frozen_source_sha256=receipt['source_sha256'],
        physical_evidence=(root / 'physical_candidate_scores.jsonl').is_file(),
        public_neural_evidence=(root / 'neural_public.parquet').is_file(),
        second_neural_evidence=(root / 'neural_ours.parquet').is_file(),
    )
    (root / 'candidate_export.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit), flush=True)
    shutil.copyfile(root / 'submission.csv', root / 'panel_candidate_lists.csv')

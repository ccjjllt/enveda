"""Rank catalog-observed edit products among same-formula generated products."""

import argparse
from hashlib import file_digest, sha256
import json
from pathlib import Path


ENDPOINTS = 'transform_prior_endpoints.parquet'
EDGES = 'transform_prior_edges.parquet'
RECEIPT = 'transform_prior_supply_receipt.json'
FOLDS = 5
SEED = 'enveda-transform-prior-075-folds-v1'
CAPS = (2, 5, 25)
ARMS = ('hash_order', 'parent_count', 'train_rule_prior', 'learned')
POOL_SHA = '82b8c40315611289b9a4004b3af18d6d3402b3cd5baa4ee5723da07b7e2f6bb0'
DERIVE_SHA = '7c1e8c3afb4c17a3ae58fbc6c36fd4cfc552da97b9129c9b46cca3cdc6ee4892'


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def stable_hash(domain, text):
    return sha256(f'{SEED}|{domain}|{text}'.encode()).hexdigest()


def load_inputs(directory, expected_sha):
    import pandas as pd

    receipt = directory / RECEIPT
    if digest(receipt) != expected_sha:
        raise ValueError('Supply receipt changed after sealing')
    manifest = json.loads(receipt.read_bytes())
    if (manifest['status'] != 'complete' or manifest['pool_sha256'] != POOL_SHA
            or manifest['chem_source_sha256'] != DERIVE_SHA):
        raise ValueError('Supply receipt does not identify the pinned public inputs')
    for name in (ENDPOINTS, EDGES):
        if digest(directory / name) != manifest['output_sha256'][name]:
            raise ValueError(f'Supply parquet changed: {name}')
    endpoints = pd.read_parquet(directory / ENDPOINTS, columns=[
        'endpoint_id', 'family', 'formula', 'smiles', 'charge', 'pool_key_present', 'pool_match', 'sampled_parent'])
    edges = pd.read_parquet(directory / EDGES, columns=['parent_id', 'product_id', 'rule', 'multiplicity'])
    if (len(endpoints) != manifest['counts']['endpoints'] or len(edges) != manifest['counts']['edges']
            or endpoints.empty or edges.empty or endpoints.isna().any().any() or edges.isna().any().any()
            or not endpoints.endpoint_id.is_unique or not endpoints.charge.eq(0).all()
            or endpoints[['endpoint_id', 'family', 'formula', 'smiles']].eq('').any().any()
            or (endpoints.pool_match & ~endpoints.pool_key_present).any()
            or edges.parent_id.eq(edges.product_id).any() or edges.multiplicity.le(0).any()
            or edges.duplicated(['parent_id', 'product_id', 'rule']).any()):
        raise ValueError('Endpoint or path schema violates the neutral distinct-product study')
    if (not edges.parent_id.isin(endpoints.loc[endpoints.sampled_parent, 'endpoint_id']).all()
            or not edges.product_id.isin(endpoints.endpoint_id).all()):
        raise ValueError('Path refers to a missing endpoint or unsampled parent')
    return endpoints, edges


def grouped_paths(endpoints, edges):
    endpoints = endpoints.set_index('endpoint_id', drop=False)
    families = set(endpoints.family)
    family_to_fold = {name: int(stable_hash('family', name)[:16], 16) % FOLDS for name in families}
    endpoints['fold'] = endpoints.family.map(family_to_fold)

    parent_fold = edges.parent_id.map(endpoints.fold)
    product_fold = edges.product_id.map(endpoints.fold)
    same_fold = parent_fold.eq(product_fold)
    unambiguous = ~edges.product_id.map(endpoints.pool_key_present & ~endpoints.pool_match)
    retained = edges.loc[same_fold & unambiguous].assign(fold=product_fold[same_fold & unambiguous])
    if retained.empty:
        raise ValueError('No same-fold unambiguous paths')

    products = endpoints.loc[retained.product_id.unique(), ['formula', 'smiles', 'pool_match', 'fold']].copy()
    containment = products.groupby(['fold', 'formula']).pool_match.agg(['min', 'max'])
    groups = containment.index[containment['min'].ne(containment['max'])]
    eligible = products.set_index(['fold', 'formula'], append=True)
    products = products.loc[eligible.index.droplevel(0).isin(groups)].sort_index()
    matched = retained.loc[retained.product_id.isin(products.index)].copy()
    if products.empty:
        raise ValueError('No exact formula group contains both observed and unobserved products')
    support = dict(input_edges=len(edges), input_endpoints=len(endpoints), input_families=len(families),
                   same_fold_edges=int(same_fold.sum()), cross_fold_edges_removed=int((~same_fold).sum()),
                   ambiguous_edges_removed_within_fold=int((same_fold & ~unambiguous).sum()),
                   eligible_edges=len(matched), eligible_products=len(products),
                   eligible_formula_groups=len(groups), observed_products=int(products.pool_match.sum()))
    return endpoints, matched, products, support


def molecule_descriptors(smiles):
    from rdkit import Chem
    from rdkit.Chem import Crippen, rdMolDescriptors as rd

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Supply SMILES failed to parse')
    names = ('fraction_sp3', 'tpsa', 'logp', 'rotors', 'rings', 'aromatic_rings', 'hbd', 'hba', 'stereocentres')
    values = (rd.CalcFractionCSP3(mol), rd.CalcTPSA(mol), Crippen.MolLogP(mol), rd.CalcNumRotatableBonds(mol),
              rd.CalcNumRings(mol), rd.CalcNumAromaticRings(mol), rd.CalcNumHBD(mol), rd.CalcNumHBA(mol),
              rd.CalcNumAtomStereoCenters(mol))
    return dict(zip(names, values, strict=True))


def features(paths, products):
    import numpy as np
    import pandas as pd

    per_rule = paths.loc[:, ['product_id', 'rule']].drop_duplicates()
    rules = pd.crosstab(per_rule.product_id, per_rule.rule).gt(0).astype('float32')
    groups = paths.groupby('product_id')
    descriptors = pd.DataFrame.from_records([molecule_descriptors(text) for text in products.smiles],
                                           index=products.index)
    counts = groups.agg(distinct_parents=('parent_id', 'nunique'), distinct_rules=('rule', 'nunique'),
                        recorded_sites=('multiplicity', 'sum'), distinct_paths=('rule', 'size')).loc[products.index]
    indicators = rules.loc[products.index].add_prefix('rule:')
    design = pd.concat([descriptors, np.log1p(counts), indicators], axis=1).astype('float32')
    if not np.isfinite(design.to_numpy()).all():
        raise ValueError('Unexpected nonfinite descriptors or route counts')
    return design, counts.distinct_parents, per_rule


def equal_formula_class_weight(products):
    counts = products.groupby(['fold', 'formula', 'pool_match']).pool_match.transform('size')
    inverse = 1 / counts
    return inverse / inverse.mean()


def fold_scores(fold, train, evaluate, x, parent_count, rules):
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits

    # Formula enters the supervision weights and the evaluation strata, never the feature matrix.
    weights = equal_formula_class_weight(train)
    routes = rules.loc[rules.product_id.isin(train.index)].copy()
    routes['weight'] = routes.product_id.map(weights)
    routes['positive_weight'] = routes.weight * routes.product_id.map(train.pool_match)
    totals = routes.groupby('rule')[['weight', 'positive_weight']].sum()
    rate = (totals.positive_weight + 8) / (totals.weight + 16)
    unseen = rules.loc[rules.product_id.isin(evaluate.index)].assign(rule_score=lambda rows: rows.rule.map(rate).fillna(.5))
    score = pd.DataFrame(index=evaluate.index)
    score['hash_order'] = 0.0
    score['parent_count'] = parent_count.loc[evaluate.index]
    score['train_rule_prior'] = unseen.groupby('product_id').rule_score.mean().loc[evaluate.index]
    model = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=12, max_depth=4, learning_rate=.06,
                                           min_samples_leaf=30, l2_regularization=1., early_stopping=False,
                                           random_state=300 + fold)
    with threadpool_limits(limits=4):
        model.fit(x.loc[train.index], train.pool_match.astype(int), sample_weight=weights)
        score['learned'] = model.predict_proba(x.loc[evaluate.index])[:, 1]
    return score


def fam_set(endpoints, paths):
    ids = set(paths.parent_id) | set(paths.product_id)
    return set(endpoints.loc[list(ids), 'family'])


def rank(products, scores):
    base = products.loc[:, ['fold', 'formula', 'pool_match']].join(scores)
    base['tie_hash'] = [stable_hash('tie', key) for key in base.index]
    strata = ['fold', 'formula']
    base['candidate_count'] = base.groupby(strata).pool_match.transform('size')
    for arm in ARMS:
        order = base.sort_values([*strata, arm, 'tie_hash'], ascending=[True, True, False, True])
        pos = order.groupby(strata, sort=False).cumcount() + 1
        base[f'{arm}_rank'] = pos
    return base.drop(columns='tie_hash').reset_index()


def metrics(ranks):
    positives = ranks.loc[ranks.pool_match].copy()
    groups = ['fold', 'formula']
    outputs = {}
    for arm in ARMS:
        pos = positives[f'{arm}_rank']
        rr = (1 / pos).where(pos.le(max(CAPS)), 0)
        candidates = {'mrr25': rr, 'top1': pos.eq(1), **{f'recall_at_{k}': pos.le(k) for k in CAPS}}
        outputs[arm] = dict(positive_level={k: float(value.mean()) for k, value in candidates.items()},
                            formula_level={k: float(positives.assign(metric=value).groupby(groups).metric.mean().mean())
                                           for k, value in candidates.items()})
    return outputs


def coverage(ranks):
    sizes = ranks.drop_duplicates(['fold', 'formula']).candidate_count
    return dict(min_candidates=int(sizes.min()), median_candidates=float(sizes.median()),
                max_candidates=int(sizes.max()),
                formula_groups_above_cap={str(k): int(sizes.gt(k).sum()) for k in CAPS})


def run(supply, expected_sha256, out):
    import pandas as pd
    from rdkit import RDLogger, rdBase

    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError('Expected competition RDKit 2026.03.3')
    RDLogger.DisableLog('rdApp.*')
    endpoints, edges = load_inputs(supply, expected_sha256)
    endpoints, paths, products, support = grouped_paths(endpoints, edges)
    splits = {}
    for fold in range(FOLDS):
        p_test = products.fold.eq(fold)
        e_test = paths.fold.eq(fold)
        train_families = fam_set(endpoints, paths.loc[~e_test])
        eval_families = fam_set(endpoints, paths.loc[e_test])
        if train_families & eval_families:
            raise ValueError('Train/eval sharing a parent or product family')
        splits[fold] = dict(train_families=len(train_families), eval_families=len(eval_families), overlap=0,
                            train_products=int((~p_test).sum()), eval_products=int(p_test.sum()),
                            eval_observed_products=int(products.loc[p_test, 'pool_match'].sum()),
                            eval_formula_groups=products.loc[p_test, 'formula'].nunique())
    # Small folds are preserved in diagnostics. A model fit needs sufficient train rows and at least two usable folds.
    usable = [fold for fold, split in splits.items() if split['eval_formula_groups'] and split['train_products'] >= 60]
    out.mkdir(parents=True, exist_ok=True)
    report = dict(status='insufficient_support', input_supply_receipt_sha256=expected_sha256, support=support,
                  folds=splits, fold_seed=SEED, usable_folds=usable, arms=ARMS,
                  candidate_caps=CAPS, positives='Public catalog-observed product generated by a different parent',
                  unknowns='No matching catalog key. Unknown products can be chemically real.',
                  split='Generic Murcko folds shared by both endpoints. Drop edges crossing folds before features.',
                  comparison='Identical held-out exact neutral formula rosters, tie-breaking by the same stable hash.')
    if len(usable) >= 2:
        design, counts, rule_table = features(paths, products)
        assessments = []
        for fold in usable:
            choose = products.fold.eq(fold)
            train, evaluate = products.loc[~choose], products.loc[choose]
            score = fold_scores(fold, train, evaluate, design, counts, rule_table)
            assessed = rank(evaluate, score)
            splits[fold]['metrics'] = metrics(assessed)
            splits[fold]['roster_sizes'] = coverage(assessed)
            assessments.append(assessed)
        all_ranks = pd.concat(assessments, ignore_index=True)
        rank_file = out / 'transform_prior_heldout_ranks.parquet'
        all_ranks.to_parquet(rank_file, index=False)
        report.update(status='complete', metrics=metrics(all_ranks), roster_sizes=coverage(all_ranks),
                      feature_columns=list(design),
                      assessed_products=len(all_ranks), assessed_observed=int(all_ranks.pool_match.sum()),
                      output_sha256={rank_file.name: digest(rank_file)})
    (out / 'transform_prior_fit_receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], support=support, metrics=report.get('metrics')), allow_nan=False),
          flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supply', required=True, type=Path)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--out', required=True, type=Path)
    run(**vars(parser.parse_args()))

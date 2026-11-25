"""Compare two frozen forward models on their common measured, same-formula spectrum pairs."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from fit_dual_energy_pairs import evaluate, fit


SOURCES = {
    'ice': ('enveda-075-ice-forward-hard-pair', 'f09dd79d8c669aa808e4c2f9473574f0d56daa85a3ab27b7cde049c5cfe33b37'),
    'gl': ('enveda-075-gl-forward-hard-pair-cpu', '3f40b93bc00b3ebfab3d030f90ab0031096dbfdba85524205fbbbb48b222bb00'),
}
FEATURES = ('observed_cosine', 'observed_entropy', 'fixed_40ev', 'reversed_energy',
            'exclusive_cosine', 'fragment_response')
LEARNED_COLUMNS = (0, 1, 4, 5)
REFERENCE = 'equal_observed_cosine'
QUERY_LOSS = False


def verified(path, expected):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f'Unexpected SHA-256 for {path.name}')
    return raw


def indexed(raw):
    output = {}
    for line in raw.splitlines():
        row = json.loads(line)
        fold, structures, energies = row['fold'], tuple(row['structures']), tuple(row['measured_energies_ev'])
        key = (fold, structures, energies)
        if (type(fold) is not int or fold not in range(5) or len(structures) != 2 or structures[0] == structures[1]
                or len(set(energies)) != len(energies) or tuple(sorted(energies)) != energies
                or 40 not in energies or set(energies)-{20, 40, 60} or key in output
                or not 0.6 <= row['tanimoto'] <= 1):
            raise ValueError('Invalid, duplicated or non-hard training pair')
        output[key] = row
    return output


def load(inputs):
    model_rows, receipts = {}, {}
    for model, (slug, digest) in SOURCES.items():
        summary_path, = (path for path in inputs.rglob(f'{model}_forward_swap_summary.json') if slug in path.parts)
        receipt = json.loads(verified(summary_path, digest))
        raw = verified(summary_path.with_name(f'{model}_forward_swap_features.jsonl'), receipt['evidence_sha256'])
        data = indexed(raw)
        if (tuple(receipt['features']) != FEATURES or receipt['selected_pairs'] != 240
                or len(data) != receipt['scored_pairs'] or receipt['shared_instrument'] != 'QTOF'
                or receipt['measured_energy_ev'] != [20, 40, 60]):
            raise ValueError(f'{model} coverage or feature schema differs from the experiment')
        for row in data.values():
            scores = np.asarray(row['oriented_scores'], dtype=float)
            if scores.shape != (2, len(FEATURES)) or not np.isfinite(scores).all():
                raise ValueError(f'Invalid {model} oriented scores')
            row['scores'] = scores
        model_rows[model], receipts[model] = data, receipt
        if model == 'gl':
            selection_path = summary_path.with_name('gl_forward_swap_selection.jsonl')
            selection = indexed(verified(selection_path, receipt['selection_sha256']))
    if len(selection) != 240 or Counter(k[0] for k in selection) != {fold: 48 for fold in range(5)}:
        raise ValueError('The frozen selection must contain five folds of 48 pairs')
    folds = {}
    for (fold, keys, _), row in selection.items():
        if any(folds.setdefault(identity, fold) != fold for identity in keys):
            raise ValueError('A training identity occurs in multiple held-out folds')
    if {k for k, r in selection.items() if r['status'] == 'scored'} != set(model_rows['gl']):
        raise ValueError('GLACIER selection and stored features disagree')
    for model, rows in model_rows.items():
        if set(rows)-selection.keys() or any(r['tanimoto'] != selection[k]['tanimoto'] for k, r in rows.items()):
            raise ValueError(f'{model} scored a different pair, orientation, energy or similarity')
        for column, feature in enumerate(FEATURES):
            replay = evaluate([r['scores'][:, column] for r in rows.values()])
            archived = receipts[model]['paired_evaluation']['models'][feature]
            if any(not np.isclose(value, archived[field], atol=1e-12, rtol=0) for field, value in replay.items()):
                raise ValueError(f'{model} {feature} replay differs from its published result')
    return selection, model_rows, receipts


def fit_query_margins(features):
    """Fit each measured spectrum separately with the same paired candidates."""
    scale = np.maximum(np.sqrt((features ** 2).mean(axis=(0, 1))), 0.02)
    values = np.concatenate((features[:, 0], features[:, 1])) / scale
    labels = np.repeat((1, 0), len(features))
    model = LogisticRegression(C=0.6, fit_intercept=False, max_iter=750)
    model.fit(values, labels)
    return model.coef_[0] / scale


def score(keys, model_rows):
    folds = np.array([k[0] for k in keys])
    arrays = {model: np.stack([rows[k]['scores'] for k in keys]) for model, rows in model_rows.items()}
    arrays['equal'] = (arrays['ice'] + arrays['gl']) / 2
    arms = {f'{model}_{feature}': a[:, :, i]
            for model, a in arrays.items() for i, feature in enumerate(FEATURES)}
    for model, a in arrays.items():
        arms[f'{model}_cosine_plus_response'] = a[:, :, 0] + 0.15 * a[:, :, 5]
    learned_inputs = {model: arrays[model][:, :, LEARNED_COLUMNS] for model in ('ice', 'gl')}
    learned_inputs['joint'] = np.concatenate([learned_inputs['ice'], learned_inputs['gl']], axis=2)
    weights = {}
    objectives = [('fold_learned', fit)]
    if QUERY_LOSS:
        objectives.append(('query_fold_learned', fit_query_margins))
    for model, features in learned_inputs.items():
        for suffix, fit_model in objectives:
            margins = np.empty(features.shape[:2])
            arm = f'{model}_{suffix}'
            weights[arm] = {}
            for fold in range(5):
                held = folds == fold
                if not held.any() or held.all():
                    raise ValueError('All five folds must participate in cross-fitted comparison')
                w = fit_model(features[~held])
                margins[held] = features[held] @ w
                weights[arm][str(fold)] = dict(training_pairs=int((~held).sum()), held_pairs=int(held.sum()),
                                               weights=w.tolist())
            arms[arm] = margins
    return arms, weights


def describe(arms, mask):
    ref = arms[REFERENCE][mask]
    ref_metrics = evaluate(ref)
    def assignment(values):
        margin = values[:, 0] - values[:, 1]
        return (margin > 1e-10).astype(int) - (margin < -1e-10).astype(int)

    def metrics(values):
        selected = values[mask]
        data = evaluate(selected)
        if len(ref):
            delta = assignment(selected) - assignment(ref)
            data.update(improved_assignment_vs_equal_observed=int((delta > 0).sum()),
                        worsened_assignment_vs_equal_observed=int((delta < 0).sum()),
                        delta_mrr_at_two_vs_equal_observed=data['mean_mrr_at_two'] - ref_metrics['mean_mrr_at_two'])
        return data
    return {name: metrics(values) for name, values in arms.items()}


def run(inputs, out):
    selection, model_rows, receipts = load(inputs)
    keys = [k for k in selection if all(k in rows for rows in model_rows.values())]
    arms, weights = score(keys, model_rows)
    positions = {key: i for i, key in enumerate(keys)}
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    per_pair = out/'forward_swap_common_pairs.jsonl'
    with per_pair.open('w') as f:
        for key, item in selection.items():
            fold, structures, energies = key
            present = [model for model, rows in model_rows.items() if key in rows]
            coverage = '+'.join(present) if present else 'neither'
            counts[coverage] += 1
            margins = ({name: a[positions[key]].tolist() for name, a in arms.items()} if key in positions else None)
            f.write(json.dumps(dict(fold=fold, structures=structures, measured_energies_ev=energies,
                                    tanimoto=item['tanimoto'], coverage=coverage, oriented_margins=margins),
                               separators=(',', ':'), allow_nan=False)+'\n')
    fold_ids = np.asarray([k[0] for k in keys])
    subset = np.asarray([selection[k]['tanimoto'] >= 0.8 for k in keys])
    result = dict(scope='Enveda-180 training pairs, timsTOF [M+H]+, same formula and measured energies. '
                        'Five scaffold folds within this single source. MRR@2 is a two-candidate statistic.',
                  primary=REFERENCE, selected_pairs=len(selection), common_pairs=len(keys),
                  available_pairs={m: len(rows) for m, rows in model_rows.items()}, coverage_counts=dict(counts),
                  receipts={m: dict(kernel=f'jiyujia/{slug}/1', summary_sha256=digest,
                                    features_sha256=receipts[m]['evidence_sha256'])
                            for m, (slug, digest) in SOURCES.items()},
                  selection_sha256=receipts['gl']['selection_sha256'], pairs_sha256=sha256(per_pair.read_bytes()).hexdigest(),
                  model_contrasts=describe(arms, np.ones(len(keys), dtype=bool)),
                  similarity_at_least_080=describe(arms, subset),
                  by_fold={str(f): describe(arms, fold_ids == f) for f in range(5)},
                  learned_features=[FEATURES[i] for i in LEARNED_COLUMNS],
                  joint_features_model_order=['ice','gl'], other_fold_fit=weights)
    (out/'forward_swap_common_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('Common pairs', len(keys), 'of', len(selection), 'coverage', dict(counts),
          'mean observed-energy cosine blend', result['model_contrasts'][REFERENCE], flush=True)


if __name__ == '__main__':
    run(Path('/kaggle/input'), Path('/kaggle/working'))

"""Compare native PubChem heads and emit only the frozen lead-formula tail."""

from functools import lru_cache

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdMolDescriptors import CalcMolFormula


RDLogger.DisableLog('rdApp.*')
tautomers = rdMolStandardize.TautomerEnumerator()


@lru_cache(maxsize=20000)
def molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Native PubChem output contains invalid SMILES {smiles!r}')
    return mol


@lru_cache(maxsize=20000)
def metric_key(smiles):
    mol = Chem.Mol(molecule(smiles))
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToInchiKey(tautomers.Canonicalize(mol))[:14]


query_ids = pd.read_parquet(TEST, columns=['molecule_id'])['molecule_id'].unique()
if len(query_ids) != 101 or set(query_ids) != set(original_pc) or set(query_ids) != set(depth50):
    raise ValueError('Panel, frozen PubChem and expanded PubChem must have the same 101 molecule IDs')

tails = {}
matched = []
unique_smiles = set()
for group, mid in enumerate(sorted(query_ids), 1):
    frozen = original_pc[mid]
    expanded = depth50[mid]
    for record, limit in ((frozen, 25), (expanded, 50)):
        if not len(record['pc']) == len(record['pc_keys']) == len(record['pc_fz']) <= limit:
            raise ValueError(f'Native PubChem array lengths disagree for {mid}')
    old_head = frozen['pc']
    new_head = expanded['pc'][:25]
    same_smiles = old_head == new_head
    formula = CalcMolFormula(molecule(old_head[0])) if old_head else None
    eligible_ranks = [i for i in range(25, len(expanded['pc']))
                      if formula is not None and CalcMolFormula(molecule(expanded['pc'][i])) == formula]
    old_keys = [metric_key(smiles) for smiles in old_head] if eligible_ranks or not same_smiles else None
    if same_smiles:
        same_metric_order = same_metric_set = True
        same_first = bool(old_head)
        shared = len(set(old_keys)) if old_keys is not None else None
    else:
        new_keys = [metric_key(smiles) for smiles in new_head]
        same_metric_order = old_keys == new_keys
        same_metric_set = set(old_keys) == set(new_keys)
        same_first = bool(old_keys and old_keys[:1] == new_keys[:1])
        shared = len(set(old_keys).intersection(new_keys))
    score_delta = [abs(a - b) for a, b, ka, kb in zip(
        frozen['pc_fz'], expanded['pc_fz'][:25], frozen['pc_keys'], expanded['pc_keys'][:25]) if ka == kb]
    audit = dict(molecule_id=mid, frozen_count=len(old_head), expanded_count=len(expanded['pc']),
                 head_smiles_exact=same_smiles, head_raw_keys_exact=frozen['pc_keys'] == expanded['pc_keys'][:25],
                 head_metric_keys_exact=same_metric_order, head_metric_set_exact=same_metric_set,
                 head_shared_metric_keys=shared, same_first_metric=same_first,
                 max_aligned_head_fz_delta=max(score_delta, default=None))
    matched.append(audit)

    tail = []
    seen = set(old_keys or ())
    for index in eligible_ranks:
        smiles = expanded['pc'][index]
        normalized = metric_key(smiles)
        if normalized in seen:
            continue
        raw_key = expanded['pc_keys'][index]
        tail.append(dict(rank=index + 1, smiles=smiles, pc_key=raw_key, metric_key=normalized,
                         pc_fz=expanded['pc_fz'][index], promotion_evidence=expanded['promotion_evidence'][raw_key]))
        seen.add(normalized)
        unique_smiles.add(smiles)
    tails[mid] = dict(frozen_first_formula=formula, tail=tail)
    if group % 25 == 0 or group == len(query_ids):
        print(f'PubChem head and tail audit {group}/{len(query_ids)}', flush=True)

anchored = [row for row in matched if row['frozen_count']]
coverage = {
    'molecules': len(matched),
    'frozen_head_nonempty': len(anchored),
    'expanded_head_nonempty': sum(row['expanded_count'] > 0 for row in matched),
    'expanded_over_25': sum(row['expanded_count'] > 25 for row in matched),
    **{field: sum(row[field] for row in anchored) for field in (
        'head_smiles_exact', 'head_raw_keys_exact', 'head_metric_keys_exact',
        'head_metric_set_exact', 'same_first_metric')},
    'max_aligned_head_fz_delta': max((row['max_aligned_head_fz_delta'] for row in anchored
                                     if row['max_aligned_head_fz_delta'] is not None), default=None),
    'molecules_with_eligible_tail': sum(bool(row['tail']) for row in tails.values()),
    'eligible_candidate_query_pairs': sum(len(row['tail']) for row in tails.values()),
    'eligible_unique_smiles': len(unique_smiles),
}
summary = dict(panel=panel_source, frozen_anchor=anchor_source, frozen_anchor_sha256=anchor_sha256,
               coverage=coverage, head_audit=matched)
work = Path('/kaggle/working')
(work / 'pc50_leading_formula_tail.json').write_text(json.dumps(tails, indent=2, allow_nan=False) + '\n')
(work / 'pc50_leading_formula_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
print(json.dumps(coverage), flush=True)

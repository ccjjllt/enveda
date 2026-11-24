"""Audit candidate overlap and one-edit reachability on the public NP250 v10 panel."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdChemReactions


PANEL_URL = 'https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-natural-product-panel-benchmark/versions/10'
STRUCTURES = 'casmi26-identity-disjoint-validation-folds/structures.parquet'
TRANSFORMS = {
    'O-methyl': '[O:1][CH3]',
    'O-acetyl': '[O:1][C](=[O])[CH3]',
    'O-sulfate': '[O:1][S](=[O])(=[O])[OH]',
    'O-hexosyl': '[O:1][CH]1[O][CH]([CH2][OH])[CH]([OH])[CH]([OH])[CH]1[OH]',
}


def pubchem_rank(group):
    """Rank PubChem candidates in the counterfactual that truth was removed from the pool."""
    fp = group.fpnet_score.to_numpy()
    finite = np.isfinite(fp)
    n = int(finite.sum())
    if not n:
        return None
    partial = np.where(finite, group.partial_score, -np.inf)
    pop = group.log_substances.to_numpy(dtype=np.float32) + group.log_pubmed.to_numpy(
        dtype=np.float32
    )
    shortlist = (
        np.flatnonzero(finite) if n <= 5000 else np.argpartition(-partial, 4999)[:5000]
    )
    n_doc = min(200, n)
    documented = np.argpartition(-np.where(finite, pop, -np.inf), n_doc - 1)[:n_doc]
    shortlist = np.union1d(shortlist[finite[shortlist]], documented)
    full = fp[shortlist].astype(np.float64)
    score = (full - full.mean()) / (full.std() + 1e-9) + 0.25 * pop[shortlist]
    ranked = group.iloc[shortlist[np.argsort(-score, kind='stable')]]
    ranked = ranked.loc[~ranked.in_pool | ranked.is_truth]
    keys = ranked.scorer_key.where(ranked.scorer_key.ne(''), ranked.connectivity)
    hits = np.flatnonzero(ranked.loc[~keys.duplicated()].is_truth)
    return int(hits[0]) + 1 if len(hits) else None


def metrics(ranks, n):
    clipped = [r for r in ranks if r is not None and r <= 25]
    return dict(
        first=sum(r == 1 for r in clipped),
        top5=sum(r <= 5 for r in clipped),
        top25=len(clipped),
        mrr25=sum(1 / r for r in clipped) / n,
    )


def product_keys(reaction, mol):
    for (product,) in reaction.RunReactants((mol,)):
        try:
            Chem.SanitizeMol(product)
            key = Chem.MolToInchiKey(product).split('-')[0]
        except (ValueError, RuntimeError):
            continue
        if key:
            yield key


def reachable_anchors(structures, targets):
    matches, blocked_groups = {}, set()
    for row in structures.itertuples(index=False):
        for key in (row.inchikey14, row.scorer_key):
            if key in targets:
                matches.setdefault(key, row)
                blocked_groups.add(row.identity_group)
    if set(matches) != targets:
        raise ValueError('Some panel structures have no public structure-table match.')
    anchors = defaultdict(list)
    for row in structures.itertuples(index=False):
        if row.identity_group not in blocked_groups and row.libs != 'enveda-180':
            anchors[row.inchikey14].append(row)
    reactions = {}
    hydroxyl = '[O;H1;+0:1]'
    for name, product in TRANSFORMS.items():
        reactions[name] = (
            rdChemReactions.ReactionFromSmarts(hydroxyl + '>>' + product),
            rdChemReactions.ReactionFromSmarts(product + '>>' + hydroxyl),
        )
    hits = defaultdict(set)
    for key, row in matches.items():
        mol = Chem.MolFromSmiles(row.smiles)
        target_key = Chem.MolToInchiKey(mol).split('-')[0]
        for name, (add, remove) in reactions.items():
            for reverse, forward in ((add, remove), (remove, add)):
                for candidate in set(product_keys(reverse, mol)):
                    for anchor in anchors.get(candidate, []):
                        parent = Chem.MolFromSmiles(anchor.smiles)
                        if parent and target_key in set(product_keys(forward, parent)):
                            hits[key].add((name, anchor.inchikey14))
                            break
    return hits, len(blocked_groups), len(anchors)


def audit(root):
    panel = pd.read_parquet(root / 'panel.parquet')
    difficulty = pd.read_parquet(root / 'panel_difficulty.parquet')
    main = pd.read_parquet(root / 'main_candidates.parquet')
    pc = pd.read_parquet(root / 'pubchem_candidates.parquet')
    structures = pd.read_parquet(
        root / STRUCTURES,
        columns=['inchikey14', 'scorer_key', 'smiles', 'identity_group', 'libs'],
    )
    targets = set(panel.molecule)
    if len(panel) != 250 or len(targets) != 250 or set(difficulty.molecule) != targets:
        raise ValueError('Expected complete NP250 sample.')
    scenarios = {}
    gen_ranks = {}
    for scenario, frame in main.groupby('scenario', observed=True):
        truth = frame.loc[frame.is_truth]
        ranks = truth.groupby('molecule', observed=True)['rank'].min().to_dict()
        scenarios[scenario] = metrics(ranks.values(), len(targets))
        if scenario == 'structure_removed_from_pool':
            if not (truth.pool_row.lt(0) & truth.pool_source.eq(0)).all():
                raise ValueError(
                    'The removed-pool truth should only occur as a generated structure.'
                )
            gen_ranks = ranks
    pc_ranks = {
        k: r
        for k, group in pc.groupby('molecule', observed=True)
        if (r := pubchem_rank(group)) is not None
    }
    overlap = {}
    for cutoff in (1, 5, 25):
        a = {k for k, r in gen_ranks.items() if r <= cutoff}
        b = {k for k, r in pc_ranks.items() if r <= cutoff}
        overlap[str(cutoff)] = dict(
            edit=len(a),
            pubchem=len(b),
            both=len(a & b),
            union=len(a | b),
            only_edit=len(a - b),
            missing_both=len(targets - (a | b)),
        )
    old_misses = targets - {k for k, r in gen_ranks.items() if r <= 25}
    union_misses = old_misses - {k for k, r in pc_ranks.items() if r <= 25}
    hits, blocked_count, anchor_count = reachable_anchors(structures, targets)
    nref = difficulty.n_reference_spectra_elsewhere
    files = [
        'panel.parquet',
        'panel_difficulty.parquet',
        'main_candidates.parquet',
        'pubchem_candidates.parquet',
        STRUCTURES,
    ]
    return {
        'date_beijing': '2026-11-24',
        'dataset': PANEL_URL,
        'inputs_sha256': {
            f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files
        },
        'rdkit': rdBase.rdkitVersion,
        'panel': {
            'molecules': len(targets),
            'spectra': int(panel.n_spectra.sum()),
            'multiple_spectra': int((panel.n_spectra > 1).sum()),
            'multiple_adducts': sum(';' in str(s) for s in panel.adducts),
            'outside_all_other_spectral_libraries': int((nref == 0).sum()),
            'other_reference_spectra_min_median_max': [
                int(nref.min()),
                float(nref.median()),
                int(nref.max()),
            ],
            'in_coconut_by_panel_snapshot': int(panel.in_coconut.sum()),
            'pubchem_window_contains_truth': int(difficulty.truth_in_tier_window.sum()),
            'pubchem_candidate_rows': len(pc),
            'main_candidate_rows': len(main),
        },
        'old_main_stored_ranks': scenarios,
        'pubchem_masked_pool': metrics(pc_ranks.values(), len(targets)),
        'union_by_branch_cutoff': overlap,
        'union_misses_at25': sorted(union_misses),
        'edit_supply': {
            'anchor_raw_keys': anchor_count,
            'excluded_panel_identity_groups': blocked_count,
            'one_step_roundtrip': len(hits),
            'already_in_old_generation_top25': len(set(hits) - old_misses),
            'missed_by_old_generation_top25': len(set(hits) & old_misses),
            'missed_by_old_two_source_union_top25': len(set(hits) & union_misses),
            'reachable_union_misses': {
                k: sorted(hits[k]) for k in sorted(set(hits) & union_misses)
            },
            'by_transform_overlapping': {
                name: sum(any(step == name for step, _ in v) for v in hits.values())
                for name in TRANSFORMS
            },
        },
        'methods': [
            '三种情景读取旧单引擎主库表中去标签前已存的 rank，在 250 个不同分子上取正确身份最好的名次。去真值结构情景里所有命中行均是生成候选。',
            '模拟真结构已从主库删除时的 PubChem 分支，按缓存 FP 初筛 5000 行并入文档优先 200 行，在这些行内用 FP 的 z 分加 0.25 倍对数文档和。去除原主库候选时保留模拟移出主库的真值行，按 scorer_key 或原连接键去重。',
            '逐来源前25的并集最多有50个候选。五道并集漏题对应完全新增身份的机会。固定最终25的融合损失、库中已有正确结构的换首机会须另计。',
            '编辑供给只保留除 enveda-180 外至少还有一个实谱来源的母体，剔除面板 250 个 identity_group 和同组别名。对每个已知真结构反向枚举四类 O 修饰的增减，然后从匹配参考母体用逆算子重新生成，并核对原始二维 InChIKey14，忽略立体。各修饰覆盖数可以重叠。',
        ],
        'limits': [
            '均来自有其他库实测参考且广为收录的 250 个赛内天然产物，屏蔽参考/结构是反事实实验。缓存模型的训练曝光和面板历史调参使此处的绝对分不能估计隐藏分。',
            '旧公开分支没有完整 0.505 融合，PubChem 重算与作者原引擎的细节可能不同。分支并集和参考枚举可达性均不是实际盲检索收益。',
            '供给审计记录实际 RDKit 版本。母体与正向生成物采用原始连接键，未重建互变异构和所有表示的编辑图。只枚举所列四类单步 O 修饰，未使用谱检索选择母体或位点。',
        ],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument(
        '--train',
        type=Path,
        help='Request additional spectrum-guided replay on the five branch-union misses.',
    )
    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.*')
    result = audit(args.panel)
    if args.train:
        from np250_analog_probe import probe

        result['spectrum_guided_replay'] = probe(
            args.panel, args.train, result['union_misses_at25'], TRANSFORMS, STRUCTURES
        )
    report = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end='')

"""Score measured reference spectra for NP250's previously identified two-branch misses."""

from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from rdkit.Chem.MolStandardize import rdMolStandardize

# Net masses are for adding a substituent to a hydroxyl while losing its hydrogen.
MASS_SHIFTS = {
    'O-methyl': 14.015650064,
    'O-acetyl': 42.010564684,
    'O-sulfate': 79.95681486,
    'O-hexosyl': 162.05282342,
}


def measured_scans(path, wanted):
    scans = defaultdict(list)
    columns = [
        'inchikey14',
        'ingest_lib',
        'adduct',
        'precursor_mz',
        'ms2_mzs',
        'ms2_normalized_intensities',
    ]
    wanted = pa.array(sorted(wanted))
    for batch in pq.ParquetFile(path).iter_batches(batch_size=16384, columns=columns):
        keep = pc.is_in(batch[0], value_set=wanted)
        if not pc.any(keep).as_py():
            continue
        for key, source, adduct, mz, mzs, weights in zip(
            *batch.filter(keep).to_pydict().values()
        ):
            if source == 'enveda-180':
                continue
            mzs, weights = np.asarray(mzs, float), np.asarray(weights, float)
            valid = (
                (mzs > 0)
                & (weights > 0)
                & (mzs < mz - 0.5)
                & np.isfinite(mzs)
                & np.isfinite(weights)
            )
            mzs, weights = mzs[valid], weights[valid]
            if len(mzs) < 3:
                continue
            top = np.argsort(weights)[-50:]
            root_weights = np.sqrt(weights[top])
            intensity = root_weights / np.linalg.norm(root_weights)
            scans[key].append((source, adduct, mz, mzs[top], intensity))
    return scans


def modified_cosine(query, reference):
    _, query_adduct, query_precursor, query_peaks, query_intensity = query
    _, ref_adduct, ref_precursor, ref_peaks, ref_intensity = reference
    if query_adduct != ref_adduct:
        return 0.0
    offset = query_peaks[:, None] - ref_peaks[None, :]
    direct = np.abs(offset) <= 0.02
    shifted = np.abs(offset - (query_precursor - ref_precursor)) <= 0.02
    ii, jj = np.nonzero(direct | shifted)
    if len(ii) < 3:
        return 0.0
    weight = query_intensity[ii] * ref_intensity[jj]
    used_query, used_reference = set(), set()
    total = 0.0
    for i in np.argsort(-weight):
        q, r = ii[i], jj[i]
        if q not in used_query and r not in used_reference:
            total += weight[i]
            used_query.add(q)
            used_reference.add(r)
    return float(total) if len(used_query) >= 3 else 0.0


def metric_keys(mol, transform, enumerator):
    for (product,) in transform.RunReactants((mol,)):
        try:
            Chem.SanitizeMol(product)
            Chem.RemoveStereochemistry(product)
            yield Chem.MolToInchiKey(enumerator.Canonicalize(product)).split('-')[0]
        except (ValueError, RuntimeError):
            continue


def generated_list(ranked_parents):
    support = defaultdict(list)
    for rank, (negative_score, _, keys) in enumerate(ranked_parents[:6], 1):
        for key in keys:
            support[key].append((-negative_score, rank))
    # Apply the same support and tie rules to all molecules. Keep a single vote per parent.
    return sorted(
        support,
        key=lambda k: (
            -sum(s for s, _ in support[k]),
            min(n for _, n in support[k]),
            k,
        ),
    )[:25]


def probe(panel_dir, train_path, union_misses, transforms, structures_filename):
    panel = pd.read_parquet(panel_dir / 'panel.parquet').set_index('molecule')
    structures = pd.read_parquet(panel_dir / structures_filename)
    measured_panel = structures.inchikey14.isin(
        panel.index
    ) | structures.scorer_key.isin(panel.index)
    excluded = structures.loc[measured_panel, 'identity_group'].unique()
    parents = structures[
        ~structures.identity_group.isin(excluded) & structures.libs.ne('enveda-180')
    ]
    queries, candidates, parent_smiles, wanted = {}, {}, {}, set()
    for key in sorted(union_misses):
        queries[key] = set(
            structures.loc[
                (structures.inchikey14 == key) | (structures.scorer_key == key),
                'inchikey14',
            ]
        )
        wanted.update(queries[key])
        candidates[key] = defaultdict(set)
        target_mass = panel.loc[key, 'measured_neutral_mass']
        for name, delta in MASS_SHIFTS.items():
            for sign in (-1, 1):
                eligible = (
                    np.abs(parents.mono_mass + sign * delta - target_mass)
                    <= target_mass * 1e-5
                )
                for row in parents.loc[eligible, ['inchikey14', 'smiles']].itertuples(
                    index=False
                ):
                    candidates[key][row.inchikey14].add((name, sign))
                    parent_smiles[row.inchikey14] = row.smiles
                    wanted.add(row.inchikey14)
    scans = measured_scans(train_path, wanted)
    reactions = {}
    hydroxyl = '[O;H1;+0:1]'
    for name, substituent in transforms.items():
        reactions[name, 1] = rdChemReactions.ReactionFromSmarts(
            hydroxyl + '>>' + substituent
        )
        reactions[name, -1] = rdChemReactions.ReactionFromSmarts(
            substituent + '>>' + hydroxyl
        )
    enumerator = rdMolStandardize.TautomerEnumerator()
    outcomes = {}
    for key in sorted(union_misses):
        source_scans = [
            s
            for raw in queries[key]
            for s in scans[raw]
            if s[0] == 'enveda-np-examples'
        ]
        ranked = []
        for parent, allowed in candidates[key].items():
            references = [s for s in scans[parent] if s[0] != 'enveda-np-examples']
            score = max(
                (modified_cosine(q, r) for q in source_scans for r in references),
                default=0.0,
            )
            if score <= 0:
                continue
            mol = Chem.MolFromSmiles(parent_smiles[parent])
            products = set()
            if mol:
                for transform in allowed:
                    products.update(metric_keys(mol, reactions[transform], enumerator))
            ranked.append((-score, parent, products))
        ranked.sort(key=lambda r: (r[0], r[1]))
        eligible = [r for r in ranked if r[2]]
        # Only now use the known answer to score both frozen lists.
        known = structures[
            (structures.inchikey14 == key) | (structures.scorer_key == key)
        ].iloc[0]
        truth_mol = Chem.MolFromSmiles(known.smiles)
        Chem.RemoveStereochemistry(truth_mol)
        truth = Chem.MolToInchiKey(enumerator.Canonicalize(truth_mol)).split('-')[0]
        arms = {}
        for arm, choices in (
            ('rank_then_check_edit', ranked),
            ('check_edit_then_rank', eligible),
        ):
            ordered = generated_list(choices)
            arms[arm] = dict(
                generated=len(ordered),
                truth_rank=ordered.index(truth) + 1 if truth in ordered else None,
            )
        outcomes[key] = dict(
            query_spectra=len(source_scans),
            parents_in_mass_windows=len(candidates[key]),
            parents_with_spectral_score=len(ranked),
            scored_editable_parents=len(eligible),
            arms=arms,
        )
    return {
        'source': 'data/raw/train.parquet, enveda-np-examples 原始谱和比赛其他有结构标注的来源。',
        'design': '从上一步已知的 5 个并集漏题出发。方法和两种母体排序顺序在这些题上检查，故属探索性回放。生成阶段读取观测中性质量、峰、加合物及候选母体结构；以答案查找本题已隔离的输入谱，通过整个面板身份及别名屏蔽同一结构的所有参考。',
        'retrieval': '对四种可逆 O 修饰按实测质量 ±10 ppm 取母体，在同加合物的实测谱之间取最大改良余弦。去掉母峰邻域，取最强 50 峰、平方根强度；直接或平移相差 ≤0.02 Da，按乘积贪心一对一匹配，至少三个峰。',
        'ranking': '最多六个母体，每一母体对同一生成身份投一次票，按母体谱相似分之和、最佳母体名次、结构键依次排序，最多25个。两臂都采用相同上限。生成结构按该版 RDKit 的去立体、规范互变异构 InChIKey14 评价。',
        'scope': '新增生成支路内部排名，尚未和完整 0.505 的全部来源合并或测量对其他分子的误升。',
        'result': outcomes,
    }

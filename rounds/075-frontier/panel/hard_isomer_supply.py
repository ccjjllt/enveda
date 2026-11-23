"""估算 40 eV 同式实测结构的困难负例供给。仅扫描标量及 CE 元数据。"""
import argparse
from collections import defaultdict
from functools import lru_cache
from math import sqrt
import json

import numpy as np
import pyarrow.dataset as ds
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

from isomer_energy_pairs import stable_key


def load_structures(path):
    source = ds.dataset(path, format='parquet')
    predicate = ((ds.field('ingest_lib') == 'enveda-180') & (ds.field('instrument_type') == 'timsTOF')
                 & (ds.field('adduct') == '[M+H]+') & (ds.field('num_peaks') >= 4)
                 & (ds.field('collision_energy_orig_units') == 'eV') & (ds.field('collision_energy_orig') == '40'))
    columns = ['inchikey14', 'molecular_formula', 'normalized_smiles', 'collision_energy_ev']
    structures = {}
    rows = 0
    for batch in source.to_batches(columns=columns, filter=predicate, batch_size=16384):
        for key, formula, smiles, ce in zip(*(c.to_pylist() for c in batch.columns)):
            if ce != [40.] or not formula or not key or len(key) != 14:
                continue
            rows += 1
            value = (formula, smiles or '')
            if key not in structures or value < structures[key]:
                structures[key] = value
    return structures, rows


def wilson(success, total):
    z = 1.959963984540054
    fraction = success / total
    denominator = 1 + z**2 / total
    center = (fraction + z**2 / (2 * total)) / denominator
    radius = z / denominator * sqrt(fraction * (1 - fraction) / total + z**2 / (4 * total**2))
    return [center - radius, center + radius]


def main(path):
    structures, spectrum_rows = load_structures(path)
    query_keys = sorted(structures, key=stable_key)[:128]
    groups = defaultdict(list)
    for key, (formula, _) in structures.items():
        groups[formula].append(key)
    tautomer = rdMolStandardize.TautomerEnumerator()
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    invalid = set()

    @lru_cache(maxsize=None)
    def canonical(key):
        _, smiles = structures[key]
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            invalid.add(key)
            return None, None
        Chem.RemoveStereochemistry(mol)
        mol = tautomer.Canonicalize(mol)
        identity = Chem.MolToInchiKey(mol).split('-')[0]
        if len(identity) != 14:
            invalid.add(key)
            return None, None
        return identity, generator.GetFingerprint(mol)

    hits, total, singletons = 0, 0, 0
    degrees = []
    best_scores = []
    for key in query_keys:
        formula, _ = structures[key]
        qid, qfp = canonical(key)
        same_formula = groups[formula]
        total += len(same_formula) - 1
        singletons += len(same_formula) == 1
        scores = {}
        for competitor in same_formula:
            if competitor == key or not qid:
                continue
            other_id, other_fp = canonical(competitor)
            if other_id and other_id != qid:
                scores[other_id] = DataStructs.TanimotoSimilarity(qfp, other_fp)
        hard_count = sum(score >= 0.6 for score in scores.values())
        degrees.append(hard_count)
        best_scores.append(max(scores.values(), default=0.))
        hits += hard_count > 0
    n = len(query_keys)
    return {'selection': '全量 enveda-180 timsTOF [M+H]+，原始 40 eV 且 collision_energy_ev=[40]，存储峰数至少四个。对每个原始 inchikey14 取字典序首个（molecular_formula, normalized_smiles），按 SHA256(isomer-energy-v1/ + inchikey14) 取前 128，含单例。',
            'identity': '按存储分子式查所有同式竞争者。去立体、规范互变异构并按 InChIKey14 去同一身份及别名。',
            'fingerprint': '规范结构 Morgan radius=2, fpSize=2048, useChirality=False，Tanimoto≥0.60。',
            'rdkit': rdBase.rdkitVersion, 'scaffold_filter': '供给审计未按 Murcko 骨架分折，训练/验证分折后须重计。',
            'eligible_spectra': spectrum_rows, 'eligible_raw_structures': len(structures), 'queries': n,
            'different_stored_key_comparisons': total, 'raw_formula_singleton_queries': singletons,
            'hard_neighbor_queries': hits, 'fraction': hits / n, 'wilson95': wilson(hits,n),
            'queries_with_at_least_three_hard_neighbors': sum(d >= 3 for d in degrees),
            'queries_with_at_least_seven_hard_neighbors': sum(d >= 7 for d in degrees),
            'hard_degree_quartiles': np.quantile(degrees, [0.25,0.5,0.75]).tolist(),
            'best_similarity_quartiles_including_zero_for_absent': np.quantile(best_scores, [0.25,0.5,0.75]).tolist(),
            'distinct_keys_processed_by_rdkit': canonical.cache_info().currsize, 'invalid_keys': len(invalid),
            'confidence_scope': 'Wilson 分数区间近似描述固定哈希等概率结构抽样的不确定性。未计来源/分子式内相关、规范化版本差异或骨架留出后的供给损失。'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parquet')
    args = parser.parse_args()
    print(json.dumps(main(args.parquet), indent=2, ensure_ascii=False))

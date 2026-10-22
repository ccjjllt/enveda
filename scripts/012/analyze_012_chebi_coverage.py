#!/usr/bin/env python3
"""手动研究ChEBI固定来源的候选覆盖；不训练、不提交、不按query真值选来源。

先构建完整固定筛选的来源集合，再读既有开发查询，分开报告新增覆盖和干扰。
不计算神经排序或把候选命中当作MRR，亦非自动化测试文件。
"""
from __future__ import annotations
import gzip
import ast
import hashlib
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import Descriptors, rdMolDescriptors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from chemistry import canonical_group_key

OUT = ROOT / 'artifacts/runs/012-chebi'
SOURCE = OUT / 'chebi_lite_3_stars.sdf.gz'


def production_mass_contract():
    notebook = ROOT / 'notebooks/submissions/submission-v10/enveda-submission-v10.ipynb'
    cells = json.loads(notebook.read_text())['cells']
    source = ''.join(cells[5]['source'])
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, (ast.Assign, ast.FunctionDef))]
    env = {'np': np}
    exec(compile(tree, str(notebook) + ':mass-cell', 'exec'), env)
    return env['ADDUCTS'], env['neutral_mass'], hashlib.sha256(source.encode()).hexdigest()


def normalize(row):
    RDLogger.DisableLog('rdApp.warning')
    RDLogger.DisableLog('rdApp.error')
    try:
        row['canonical_key'] = canonical_group_key(row['smiles'])
        if not row['canonical_key']:
            return None
        return row
    except (ValueError, RuntimeError):
        return None


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def build_source():
    started = time.perf_counter()
    cache = OUT / 'source-index.json'
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached['source_sha256'] != hashlib.sha256(SOURCE.read_bytes()).hexdigest():
            raise ValueError('Source differs from saved canonical index')
        return cached
    counts = Counter()
    rows = []
    RDLogger.DisableLog('rdApp.warning')
    RDLogger.DisableLog('rdApp.error')
    with gzip.open(SOURCE, 'rb') as fh:
        for mol in Chem.ForwardSDMolSupplier(fh):
            counts['source_records'] += 1
            if mol is None:
                counts['parse_failed'] += 1
                continue
            if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
                counts['wildcard_excluded'] += 1
                continue
            if len(Chem.GetMolFrags(mol)) != 1:
                counts['disconnected_excluded'] += 1
                continue
            if Chem.GetFormalCharge(mol) != 0:
                counts['charged_excluded'] += 1
                continue
            if any(a.GetIsotope() for a in mol.GetAtoms()):
                counts['isotopic_excluded'] += 1
                continue
            mass = Descriptors.ExactMolWt(mol)
            if not math.isfinite(mass) or not 100 <= mass <= 1500:
                counts['mass_excluded'] += 1
                continue
            Chem.RemoveStereochemistry(mol)
            rows.append({'source_id': mol.GetProp('ChEBI ID'), 'smiles': Chem.MolToSmiles(mol),
                         'mass': mass, 'formula': rdMolDescriptors.CalcMolFormula(mol)})
    counts['eligible_records'] = len(rows)
    print('SOURCE_PARSED', dict(counts), flush=True)
    unique = {}
    with ProcessPoolExecutor(max_workers=4) as executor:
        for i, row in enumerate(executor.map(normalize, rows, chunksize=16)):
            if row is None:
                counts['identity_failed'] += 1
                continue
            key = row['canonical_key']
            if key not in unique:
                unique[key] = row
            elif row['source_id'] < unique[key]['source_id']:
                unique[key] = row
            if (i + 1) % 5000 == 0:
                print('NORMALIZED', i + 1, 'unique', len(unique), flush=True)
    result = {'counts': dict(counts), 'source': 'ChEBI release 256, 2026-10-06, 3-star LITE SDF',
              'source_url': 'https://ftp.ebi.ac.uk/pub/databases/chebi/SDF/chebi_lite_3_stars.sdf.gz',
              'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'source_bytes': SOURCE.stat().st_size, 'license': 'CC BY 4.0',
              'filters': 'connected, no wildcard, net charge zero, no explicit isotope, 100..1500 Da',
              'rdkit': rdBase.rdkitVersion, 'build_seconds': time.perf_counter() - started,
              'rows': sorted(unique.values(), key=lambda r: (r['mass'], r['canonical_key']))}
    dump('source-index.json', result)
    return result


def audit_queries(index, base, expanded):
    # 来源构建完成以后才加载query/labels，质量窗选择只用观测字段。
    masses = np.asarray([r['mass'] for r in index], float)
    keys = np.asarray([r['canonical_key'] for r in index])
    source = set(keys)
    adducts, neutral_mass, _ = production_mass_contract()
    def query_mass(mz, adduct):
        if adduct not in adducts:
            return None, False
        return float(neutral_mass(np.asarray([mz], float), [adduct])[0]), True
    out = {}
    for ds, folder in [('np', 'f1-bundle'), ('g1', 'g1-validation')]:
        molecules = json.loads((ROOT / f'artifacts/{folder}/inputs.json').read_text())['molecules']
        rows = []
        for m in molecules:
            spec = max(m['spectra'], key=lambda s: len(s['mz']))
            qm, known = query_mass(spec['prec'], spec['adduct'])
            found = set(keys[np.abs(masses - qm) <= qm * 10e-6]) if known else set()
            rows.append({'key': m['key'], 'known_mass': known, 'source_in_window': len(found),
                         'new_vs_base_in_window': len(found - base), 'new_vs_v10_in_window': len(found - expanded),
                         'truth_in_source': m['truth_canon'] in source,
                         'truth_in_source_window': m['truth_canon'] in found,
                         'truth_new_vs_v10_in_window': m['truth_canon'] in found - expanded})
        out[ds] = {'n': len(rows), 'new_vs_v10_candidate_pairs': sum(r['new_vs_v10_in_window'] for r in rows),
                   'queries_with_new_candidates': sum(r['new_vs_v10_in_window'] > 0 for r in rows),
                   'source_pool_truth_coverage': sum(r['truth_in_source'] for r in rows),
                   'source_window_truth_coverage': sum(r['truth_in_source_window'] for r in rows),
                   'new_truths_vs_v10': sum(r['truth_new_vs_v10_in_window'] for r in rows), 'rows': rows}
    observations = json.loads((ROOT / 'artifacts/runs/007-r3/observations_50.json').read_text())['molecules']
    rows = []
    for m in observations:
        s = max(m['spectra'], key=lambda v: v['n_peaks'])
        qm, known = query_mass(s['precursor_mz'], s['adduct'])
        found = set(keys[np.abs(masses - qm) <= qm * 10e-6]) if known else set()
        truth = m['canonical_key']
        rows.append({'key': truth, 'class_AB': m['class_AB'], 'known_mass': known,
                     'truth_in_base': truth in base, 'truth_in_v10': truth in expanded,
                     'truth_in_source': truth in source, 'truth_in_source_window': truth in found,
                     'source_in_window': len(found), 'new_vs_base_in_window': len(found - base),
                     'new_vs_v10_in_window': len(found - expanded)})
    out['r3'] = {'n': len(rows), 'known_mass': sum(r['known_mass'] for r in rows),
                 'new_truths_vs_base': sum(not r['truth_in_base'] and r['truth_in_source_window'] for r in rows),
                 'new_truths_vs_v10': sum(not r['truth_in_v10'] and r['truth_in_source_window'] for r in rows),
                 'source_pool_truth_coverage': sum(r['truth_in_source'] for r in rows),
                 'source_window_truth_coverage': sum(r['truth_in_source_window'] for r in rows),
                 'new_vs_v10_candidate_pairs': sum(r['new_vs_v10_in_window'] for r in rows), 'rows': rows}
    return out


def main():
    started = time.perf_counter()
    source = build_source()
    base = set()
    for name in ['train', 'coco']:
        values = pq.read_table(ROOT / f'artifacts/runs/005/structure_identity_{name}.parquet',
                               columns=['canonical_group_key']).column(0).to_pylist()
        base.update(k for k in values if k)
    mb = [json.loads(line) for line in (ROOT / 'artifacts/runs/008-expansion/candidate_index_records.jsonl').read_text().splitlines()]
    expanded = base | {r['canonical_key'] for r in mb if r['canonical_key']}
    assert len(expanded - base) == 538
    skeys = {r['canonical_key'] for r in source['rows']}
    result = {'source': {k:v for k,v in source.items() if k != 'rows'},
              'unique_canonical': len(skeys), 'base_canonical': len(base), 'v10_canonical': len(expanded),
              'new_vs_base': len(skeys-base), 'new_vs_v10': len(skeys-expanded),
              'queries': audit_queries(source['rows'], base, expanded),
              'seconds': time.perf_counter()-started, 'python': platform.python_version(),
              'production_mass_cell_sha256': production_mass_contract()[2],
              'device': 'local CPU, 4 canonicalization workers', 'seed': None,
              'boundary': 'Source coverage only. No fingerprint/ICE ranking, MRR, unseen-structure or hidden-score claim.'}
    dump('coverage.json', result)
    print(json.dumps({**{k:v for k,v in result.items() if k!='queries'},
                      'queries': {k:{a:b for a,b in v.items() if a!='rows'} for k,v in result['queries'].items()}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

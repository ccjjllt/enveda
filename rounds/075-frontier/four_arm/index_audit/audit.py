"""Audit five frozen misses in the v7 candidate pools and the complete PubChem index."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import ctypes
from hashlib import sha256
import json
from multiprocessing import get_context
from pathlib import Path
import pickle
import subprocess
from tempfile import TemporaryDirectory
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


V4G = 'casmi26-attributed-v4g-reproduction-assets'
ROOT = Path('/kaggle/input')
OUT = Path('/kaggle/working')
MISSES = ['challenge-'+x for x in ('011','033','034','035','178')]
ELECTRON_MASS = 0.000548579909
EXPECTED_STORES = {'tier':105875489, 'delta':36303, 'growth':11177}
WAVE = (10, 100)


def heavy_counts(mol):
    return tuple(sorted(Counter(a.GetAtomicNum() for a in mol.GetAtoms() if a.GetAtomicNum()>1).items()))


def canonical(mol):
    Chem.RemoveStereochemistry(mol)
    return TAUTOMER.Canonicalize(mol)


def identity(mol):
    return Chem.MolToInchiKey(mol)[:14], heavy_counts(mol)


def input_rows():
    isomers = {}
    for row in FROZEN_INPUT:
        name = row['challenge']
        mol = Chem.MolFromSmiles(row['truth_smiles'])
        if not mol:
            raise ValueError(f'Invalid truth structure {name}')
        canon = canonical(Chem.Mol(mol))
        raw = Chem.MolToInchiKey(mol)[:14]
        key, composition = identity(canon)
        if key != row['truth_key']:
            raise ValueError(f'Unexpected normalized identity {name}')
        charge = int(Chem.GetFormalCharge(mol))
        wt = float(rdMolDescriptors.CalcExactMolWt(mol))
        windows = {'original_C34':row['original_C34_target']}
        if name == 'challenge-011':
            if charge != 1:
                raise ValueError('011 requires the inherently charged monocation.')
            windows.update(intrinsic_ion_raw_mz=row['precursor_mz'],
                           intrinsic_ion_native_C34=row['precursor_mz']+ELECTRON_MASS)
        isomers[name] = dict(**row, raw_key=raw, normalized_key=key, composition=composition,
                             normalized_smiles=Chem.MolToSmiles(canon), formal_charge=charge,
                             exact_ion_or_neutral_mass=wt, exact_mass_sum_of_atoms=wt+charge*ELECTRON_MASS,
                             formula=rdMolDescriptors.CalcMolFormula(canon), windows=windows)
    if sorted(isomers) != MISSES:
        raise ValueError(f'Expected the archived five misses, got {sorted(isomers)}.')
    return isomers


def in_window(mass, center, ppm):
    return center*(1-ppm*1e-6) <= mass <= center*(1+ppm*1e-6)


def match_info(text, mol, row, mass, known, store_key=None):
    prop = Chem.GetFormalCharge(mol)
    raw = Chem.MolToInchiKey(mol)[:14]
    score_key, composition = identity(canonical(Chem.Mol(mol)))
    expected = known['normalized_key'], known['composition']
    if (score_key, composition)!=expected:
        return None
    return dict(row_id=int(row), smiles=text, indexed_mass=float(mass),
                stored_identity=store_key, raw_key=raw, normalized_key=score_key,
                formal_charge=prop, formula=rdMolDescriptors.CalcMolFormula(mol),
                windows={label:{'ppm_offset':(float(mass)-target)*1e6/target,
                                **{f'within_{w}ppm':in_window(float(mass),target,w) for w in WAVE}}
                         for label,target in known['windows'].items()})


def by_composition(queries):
    output = {}
    for row in queries.values():
        output.setdefault(row['composition'], []).append(row)
    return output


def empty_stats(queries):
    return {name:{'profile_rows':0, 'same_formula_all_mass':0, 'hits':[],
                  'within':{label:{str(w):dict(rows=0, same_formula=0, identity=0) for w in WAVE}
                            for label in q['windows']}} for name,q in queries.items()}


def bounds(masses, target, ppm):
    tolerance = target*ppm*1e-6
    return int(np.searchsorted(masses, target-tolerance, 'left')), int(np.searchsorted(masses, target+tolerance, 'right'))


def audit_rows(iterrows, queries):
    by_profile = by_composition(queries)
    result = empty_stats(queries)
    invalid = 0
    for row_id, text, mass, stored_key in iterrows:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            invalid += 1
            continue
        known = by_profile.get(heavy_counts(mol))
        if not known:
            continue
        formula = rdMolDescriptors.CalcMolFormula(mol)
        for query in known:
            stats = result[query['challenge']]
            stats['profile_rows'] += 1
            same_formula = formula == query['formula']
            stats['same_formula_all_mass'] += same_formula
            found = match_info(text,mol,row_id,mass,query,stored_key)
            if found:
                stats['hits'].append(found)
            for label,target in query['windows'].items():
                for w in WAVE:
                    if in_window(float(mass),target,w):
                        window = stats['within'][label][str(w)]
                        window['same_formula'] += same_formula
                        window['identity'] += bool(found)
    return result, invalid


def scan_idx_partition(task):
    mass_path, off_path, blob_path, start, stop, lib, queries = task
    mass, off, blob = [np.load(p,mmap_mode='r') for p in (mass_path,off_path,blob_path)]
    if mass.ndim != 1 or blob.dtype.itemsize != 1 or len(off)!=len(mass)+1 or off[0]!=0 or off[-1]!=len(blob):
        raise ValueError(f'Misaligned mass/SMILES rows {mass_path}')
    selector = ctypes.CDLL(str(lib)).select_profiles
    selector.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                         ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    selector.restype = ctypes.c_size_t
    by_profile = by_composition(queries)
    patterns = np.array([[dict(composition).get(i,0) for i in (6,7,8)] for composition in by_profile],dtype=np.intc)
    if not all(set(dict(c))<={6,7,8} for c in by_profile):
        raise ValueError('This lexical profile filter only discards known non-C/N/O atoms.')
    selected = 0
    chunk = 262144

    def candidate_rows():
        nonlocal selected
        for low in range(start,stop,chunk):
            high = min(low+chunk,stop)
            local = np.asarray(off[low:high+1],dtype=np.int64)
            output = np.empty(high-low,dtype=np.int64)
            n = selector(blob.ctypes.data,local.ctypes.data,high-low,patterns.ctypes.data,len(patterns),output.ctypes.data)
            selected += n
            for j in output[:n]:
                text = bytes(blob[local[j]:local[j+1]]).decode('ascii')
                yield low+j,text,mass[low+j],None
    stats, invalid = audit_rows(candidate_rows(),queries)
    print(f'{mass_path.name} rows [{start},{stop}): probed {selected} candidates with RDKit, invalid={invalid}',flush=True)
    return stats,selected,invalid


def add_stats(base, extension):
    for name, info in extension.items():
        item = base[name]
        item['profile_rows'] += info['profile_rows']
        item['same_formula_all_mass'] += info['same_formula_all_mass']
        item['hits'].extend(info['hits'])
        for label, windows in info['within'].items():
            for w, values in windows.items():
                for field,count in values.items():
                    item['within'][label][w][field] += count


def populate_counts(result,masses,queries,already_sorted):
    masses = np.asarray(masses)
    if not already_sorted:
        masses = np.sort(masses[np.isfinite(masses)])
    for name, q in queries.items():
        for label, target in q['windows'].items():
            for w in WAVE:
                low,high = bounds(masses,target,w)
                result[name]['within'][label][str(w)]['rows'] = high-low


def scan_pubchem(queries, work):
    lib = work/'fastscan.so'
    cxx = work/'fastscan.cpp'
    cxx.write_text(FASTSCAN_SOURCE)
    subprocess.run(['c++','-std=c++17','-O3','-shared','-fPIC','-o',str(lib),str(cxx)],check=True)
    report = {}
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context('fork')) as pool:
        for store,(masses,offsets,blob) in zip(EXPECTED_STORES,index_files(),strict=True):
            n = len(masses)
            if n != EXPECTED_STORES[store]:
                raise ValueError(f'{store} count changed: expected {EXPECTED_STORES[store]}, received {n}')
            if not np.isfinite(masses).all() or not np.all(masses[1:]>=masses[:-1]):
                raise ValueError(f'Mass array {store} is not finite and sorted')
            cuts = np.linspace(0,n,5 if store=='tier' else 2,dtype=np.int64).tolist()
            job = [(masses.filename,offsets.filename,blob.filename,start,stop,lib,queries)
                   for start,stop in zip(cuts[:-1],cuts[1:],strict=True)]
            result = empty_stats(queries)
            selected = invalid = 0
            for stats, n_selected,n_invalid in pool.map(scan_idx_partition,job):
                add_stats(result,stats)
                selected += n_selected
                invalid += n_invalid
            populate_counts(result,masses,queries,True)
            report[store] = dict(rows=n, smiles_bytes=blob.nbytes, rows_rdkit_checked=selected,
                                 invalid_smiles_among_checked=invalid, by_query=result)
    return report


def scan_table(table,queries):
    def rows():
        for t in table.itertuples(index=False):
            if isinstance(t.smiles,str) and t.smiles and np.isfinite(t.mass):
                yield t.row_id,t.smiles,t.mass,t.key
    result,invalid = audit_rows(rows(),queries)
    populate_counts(result,table.mass.to_numpy(),queries,False)
    return dict(rows=len(table),invalid_smiles=invalid, by_query=result)


def published_structures(queries):
    main_path = find('casmi26-v2-pool','pool_meta.parquet')
    checksum = sha256(main_path.read_bytes()).hexdigest()
    if checksum != '82b8c40315611289b9a4004b3af18d6d3402b3cd5baa4ee5723da07b7e2f6bb0':
        raise ValueError(f'Main-pool snapshot hash differs: {checksum}')
    report = {'engine1':dict(sha256=checksum,**scan_table(pq.read_table(main_path,columns=['key','smiles','mass'])
                                                       .to_pandas().rename_axis('row_id').reset_index(),queries))}
    e2_sources = [('coconut-casmi26-candidates','coco'),('casmi-chebi-lmsd-structures','bio'),
                  ('casmi26-pubchemlite-cc0-candidate-fingerprints','pubchemlite')]
    seen = set()
    next_row = 0
    for label,prefix in e2_sources:
        pkl = find(label,f'{prefix}_meta.pkl')
        raw = pickle.loads(pkl.read_bytes())
        masses = np.load(pkl.with_name(f'{prefix}_mass.npy'),mmap_mode='r')
        data = pd.DataFrame({'key':raw['keys'],'smiles':raw['smiles'],'mass':masses})
        data = data[~data.key.isin(seen)].copy()
        seen.update(data.key)
        data['row_id'] = np.arange(next_row,next_row+len(data))
        next_row += len(data)
        finite_data = data[np.isfinite(data.mass)]
        report[f'engine2_{prefix}'] = dict(raw_rows=len(masses), pre_fp_snapshot=True, **scan_table(finite_data,queries))
    train_path = find('enveda-CASMI26-molecule-id-mass-spectra','train.parquet')
    raw = pq.read_table(train_path, columns=['inchikey14','normalized_smiles']).to_pandas().dropna()
    raw = raw.drop_duplicates('inchikey14').rename(columns={'inchikey14':'key','normalized_smiles':'smiles'})
    raw = raw[~raw.key.isin(seen)].copy()
    train_result = empty_stats(queries)
    # Reconstruct the membership rule exactly. Only calculate the roughly 6K bits
    # for identities that may match after the official metric normalization.
    masses = []
    trial = []
    profiles = by_composition(queries)
    for i,row in enumerate(raw.itertuples(index=False),next_row):
        mol = Chem.MolFromSmiles(row.smiles)
        if mol is None:
            continue
        weight = rdMolDescriptors.CalcExactMolWt(mol)
        if not np.isfinite(weight):
            continue
        masses.append(weight)
        if heavy_counts(mol) in profiles:
            trial.append((i,row.smiles,weight,row.key))
    ts,invalid = audit_rows(trial,queries)
    for q,stats in ts.items():
        for hit in stats['hits']:
            # Engine 2 rejects only when parsing, these fingerprint constructors, or mass fail.
            fp_grid = [('Morgan-r2',rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=4096)),
                       ('Morgan-r3',rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=4096)),
                       ('RDKit-path',rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048,maxPath=6))]
            m = Chem.MolFromSmiles(hit['smiles'])
            try:
                for _,generator in fp_grid:
                    generator.GetFingerprintAsNumPy(m)
                MACCSkeys.GenMACCSKeys(m)
            except Exception as err:
                hit['engine2_fp_error'] = repr(err)
        if any('engine2_fp_error' in h for h in stats['hits']):
            raise RuntimeError(f'Exact Engine 2 match rejected during structure fingerprinting: {q}')
    add_stats(train_result,ts)
    populate_counts(train_result,np.array(masses),queries,False)
    report['engine2_train']=dict(raw_rows=len(raw), finite_parsed_rows=len(masses),
                                 rows=len(masses), invalid_chemistry_during_candidate_scan=invalid,
                                 by_query=train_result)
    return report


def main():
    global Chem,TAUTOMER,rdMolDescriptors,rdFingerprintGenerator,MACCSkeys
    started = time.monotonic()
    # Use the same official pinned RDKit installation as the earlier audit.
    chemistry()
    from rdkit import Chem,rdBase
    from rdkit.Chem import rdMolDescriptors,rdFingerprintGenerator,MACCSkeys
    from rdkit.Chem.MolStandardize import rdMolStandardize
    TAUTOMER = rdMolStandardize.TautomerEnumerator()
    queries = input_rows()
    OUT.mkdir(parents=True,exist_ok=True)
    report = dict(public_pipeline='jiyujia/enveda-075-np-four-arm-0505/2',
                  list_source='C34 pre-ranker pc_lists.json from four-arm v1, reused in v2',
                  rdkit=rdBase.rdkitVersion, method='All index offsets scanned. Conservative C/N/O count prefilter; parsed records require a matching RDKit-canonical InChIKey14 and heavy-atom composition. Charge-parent/salt-only matches are excluded under the official metric. Theoretical masses and retrospective ionic windows only diagnose coverage, never alter baseline search.',
                  frozen_input_hashes=FROZEN_HASHES, per_query=queries)
    if INCLUDE_MAIN:
        report['main_pools'] = published_structures(queries)
        (OUT/'five_miss_main_coverage.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with TemporaryDirectory(prefix='polychem-audit-') as workspace:
        report['pubchem']=scan_pubchem(queries,Path(workspace))
    report['elapsed_seconds']=round(time.monotonic()-started,1)
    report['scanned_pubchem_rows']=sum(item['rows'] for item in report['pubchem'].values())
    (OUT/'five_miss_index_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for name in queries:
        pools={k:v['by_query'][name]['hits'] for k,v in report.get('main_pools',{}).items()}
        pub={k:v['by_query'][name]['hits'] for k,v in report['pubchem'].items()}
        print(name,'main',{k:len(v) for k,v in pools.items()},
              'PubChem',{k:len(v) for k,v in pub.items()},'complete',flush=True)


if __name__=='__main__':
    main()

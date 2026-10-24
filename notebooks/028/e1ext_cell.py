'''028: Engine-1 pool extension. PubChemLite CC0 (franciscoangulo) + ChEBI/LIPID MAPS (sergiomalv, CC BY 4.0) structures
not already in the v2 pool, restricted to neutral masses within 15 ppm of the active test precursors, get Engine-1 FPNet
bits (chem.raw_fingerprint + fp_bits), fragment masses (frag.fragments_for_smiles, time-budgeted; empty after budget) and
join the pool (src=4), so Engine-1 scores and ranks them and the simulators see them via the base lists.
Any failure keeps the original pool symlinks.'''
import os, glob, time as _xt, pickle as _xp
import numpy as np, pandas as pd
E1EXT = dict(status='start'); _x0 = _xt.time(); E1EXT_ORDER = None; E1EXT_NNEW = 0
try:
    from casmi import chem as _xc, frag as _xf
    from rdkit import Chem as _xC, RDLogger as _xRL
    from rdkit.Chem.Descriptors import ExactMolWt as _xW
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula as _xF
    _xRL.DisableLog('rdApp.*')
    _AD = {'[M+H]+': 1.007276, '[M+Na]+': 22.989218, '[M+NH4]+': 18.033823, '[M+K]+': 38.963158, '[M-H2O+H]+': 1.007276 - 18.010565,
           '[M-H]-': -1.007276, '[M+Cl]-': 34.969402, '[M+FA-H]-': 44.998201, '[M+HCOO]-': 44.998201, '[M-H2O-H]-': -1.007276 - 18.010565}
    _te = pd.read_parquet(os.path.join(COMP, 'test.parquet'), columns=['adduct', 'precursor_mz'])
    _nm = []
    for _a, _p in zip(_te.adduct, _te.precursor_mz):
        if _a in _AD: _nm.append(float(_p) - _AD[_a])
        else: _nm += [float(_p) - 1.007276, float(_p) + 1.007276]
    _nm = np.unique(np.round(np.array(_nm), 5))
    def _inwin(m, ppm=15.0):
        i = np.searchsorted(_nm, m); ok = np.zeros(len(m), bool)
        for j in (i - 1, i):
            jj = np.clip(j, 0, len(_nm) - 1); ok |= np.abs(_nm[jj] - m) <= m * ppm * 1e-6
        return ok
    _meta = pd.read_parquet(os.path.join(POOL_DIR, 'pool_meta.parquet'))
    _have = set(_meta.key.values)
    _new = {}
    _pcd = os.path.dirname(find('pubchemlite_meta.pkl')); _pm = _xp.load(open(os.path.join(_pcd, 'pubchemlite_meta.pkl'), 'rb'))
    _pms = np.load(os.path.join(_pcd, 'pubchemlite_mass.npy')).astype(np.float64)
    for k, s in zip(np.asarray(_pm['keys'])[_inwin(_pms)], np.asarray(_pm['smiles'])[_inwin(_pms)]):
        if k not in _have: _new.setdefault(k, s)
    _bd = os.path.dirname(find('bio_meta.pkl')); _bm = _xp.load(open(os.path.join(_bd, 'bio_meta.pkl'), 'rb'))
    _bms = np.load(os.path.join(_bd, 'bio_mass.npy')).astype(np.float64)
    _bw = _inwin(_bms)
    for k, s, w in zip(_bm['keys'], _bm['smiles'], _bw):
        if w and k not in _have: _new.setdefault(k, s)
    E1EXT['cand'] = len(_new)
    _bits = np.load(os.path.join(POOL_DIR, 'fp_bits.npy'))
    rows, fps, frs = [], [], []
    _fb = float(os.environ.get('E1EXT_FRAG_BUDGET', '1500')); _tf = _xt.time(); nfr = 0
    for k, s in _new.items():
        mol = _xC.MolFromSmiles(s)
        if mol is None: continue
        r = _xc.raw_fingerprint(s)
        if r is None: continue
        fr = np.zeros(0, np.float32)
        if _xt.time() - _tf < _fb:
            try: fr = np.asarray(_xf.fragments_for_smiles(s), np.float32); nfr += 1
            except Exception: pass
        rows.append((k, s, _xF(mol), _xW(mol), mol.GetNumHeavyAtoms()))
        fps.append(np.packbits(r[_bits])); frs.append(fr)
    E1EXT.update(n_new=len(rows), n_frag=nfr)
    if rows:
        _fp0 = np.load(os.path.join(POOL_DIR, 'pool_fp.npy')); _fo0 = np.load(os.path.join(POOL_DIR, 'pool_frag_off.npy'))
        _fm0 = np.load(os.path.join(POOL_DIR, 'pool_frag_mass.npy'))
        _nd = pd.DataFrame(rows, columns=['key', 'smiles', 'formula', 'mass', 'n_heavy'])
        _nd['src'] = 4; _nd['train_sid'] = -1; _nd['np_likeness'] = np.nan
        _all = pd.concat([_meta.drop(columns=['pid']), _nd], ignore_index=True)
        _all['n_heavy'] = _all.n_heavy.astype(np.int16); _all['src'] = _all.src.astype(np.int8); _all['train_sid'] = _all.train_sid.astype(np.int64)
        E1EXT_ORDER = np.argsort(_all.mass.values, kind='stable'); E1EXT_NNEW = len(rows)
        _all = _all.iloc[E1EXT_ORDER].reset_index(drop=True); _all['pid'] = np.arange(len(_all))
        _fpa = np.vstack([_fp0, np.stack(fps)])[E1EXT_ORDER]
        _lens = np.concatenate([np.diff(_fo0), np.array([len(f) for f in frs])])
        _segs = [None] * len(_lens)
        n0 = len(_meta)
        _fo = np.zeros(len(_lens) + 1, np.int64); _fo[1:] = np.cumsum(_lens[E1EXT_ORDER])
        _fm = np.empty(int(_fo[-1]), np.float32)
        for newi, oi in enumerate(E1EXT_ORDER):
            seg = _fm0[_fo0[oi]:_fo0[oi + 1]] if oi < n0 else frs[oi - n0]
            _fm[_fo[newi]:_fo[newi + 1]] = seg
        for f in ['pool_meta.parquet', 'pool_fp.npy', 'pool_frag_off.npy', 'pool_frag_mass.npy']:
            p = os.path.join('work', f)
            if os.path.islink(p) or os.path.exists(p): os.remove(p)
        _all.to_parquet('work/pool_meta.parquet', index=False); np.save('work/pool_fp.npy', _fpa)
        np.save('work/pool_frag_off.npy', _fo); np.save('work/pool_frag_mass.npy', _fm)
        del _fp0, _fm0, _fpa, _fm
    E1EXT['status'] = 'ok'
except Exception as _xe:
    import traceback; traceback.print_exc(); E1EXT['status'] = 'failed: ' + repr(_xe); E1EXT_ORDER = None
    for f in ['pool_meta.parquet', 'pool_fp.npy', 'pool_frag_off.npy', 'pool_frag_mass.npy']:
        p = os.path.join('work', f)
        if not os.path.islink(p):
            if os.path.exists(p): os.remove(p)
            os.symlink(os.path.join(POOL_DIR, f), p)
E1EXT['s'] = round(_xt.time() - _x0, 1)
print('028 E1EXT', E1EXT, flush=True)

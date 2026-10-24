'''033: Engine-1 extra candidates as *generated* entries (pid=-1), not pool rows.
PubChemLite CC0 (franciscoangulo) + ChEBI/LIPID MAPS (sergiomalv, CC BY 4.0) structures not in the v2 pool, near the
test precursors, are appended to Engine.candidates() through the existing generated-candidate path, so every v4b
feature family computes their features from SMILES (no pool-aligned resources touched). Any failure leaves E unchanged.'''
import os, glob, time as _xt, pickle as _xp
import numpy as np, pandas as pd
E1GEN = dict(status='start'); _x0 = _xt.time()
try:
    from casmi import chem as _xc, frag as _xf
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
    _have = set(P.key)
    _new = {}
    _pcd = os.path.dirname(find('pubchemlite_meta.pkl')); _pm = _xp.load(open(os.path.join(_pcd, 'pubchemlite_meta.pkl'), 'rb'))
    _pms = np.load(os.path.join(_pcd, 'pubchemlite_mass.npy')).astype(np.float64); _w = _inwin(_pms)
    for k, s in zip(np.asarray(_pm['keys'])[_w], np.asarray(_pm['smiles'])[_w]):
        if k not in _have: _new.setdefault(k, s)
    _bd = os.path.dirname(find('bio_meta.pkl')); _bm = _xp.load(open(os.path.join(_bd, 'bio_meta.pkl'), 'rb'))
    _bms = np.load(os.path.join(_bd, 'bio_mass.npy')).astype(np.float64); _bw = _inwin(_bms)
    for k, s, w in zip(_bm['keys'], _bm['smiles'], _bw):
        if w and k not in _have: _new.setdefault(k, s)
    _G = []   # standardized entries without fp/frags (lazy)
    for k, s in _new.items():
        std = _xc.standardize_smiles(s)
        if std is None: continue
        kk = _xc.score_key(std)
        if kk is None or kk in _have: continue
        p = _xc.mol_props(std)
        if p is None: continue
        _G.append((float(p[1]), kk, std, p[0], int(p[2])))
    _G.sort(key=lambda x: x[0]); _GM = np.array([g[0] for g in _G]); _GC = {}
    E1GEN.update(cand=len(_new), n=len(_G))
    def _ext(target, keys):
        tol = target * E.cfg.ppm_win * 1e-6
        a, b = np.searchsorted(_GM, target - tol), np.searchsorted(_GM, target + tol, 'right')
        out = []
        for m, k, s, f, nh in _G[a:b]:
            if k in keys: continue
            if s not in _GC:
                fp = _xc.raw_fingerprint(s)
                _GC[s] = None if fp is None else (fp[E.pool.bits].astype(np.float32), _xf.fragments_for_smiles(s))
            c = _GC[s]
            if c is None: continue
            out.append(dict(smiles=s, key=k, formula=f, mass=m, n_heavy=nh, fp=c[0], frags=c[1], sim=0.0, steps=0, tan=0.0, parent=-1))
        return out
    _orig_cands = E.candidates
    E1GEN['added'] = 0
    def _cands(target, drop_pid=-1, generated=None, _o=_orig_cands):
        g = list(generated or [])
        try:
            keys = {x['key'] for x in g}
            x = _ext(target, keys); g += x; E1GEN['added'] += len(x)
        except Exception as e:
            E1GEN['err'] = repr(e)
        return _o(target, drop_pid, g)
    E.candidates = _cands
    E1GEN['status'] = 'ok'
except Exception as _xe:
    import traceback; traceback.print_exc(); E1GEN['status'] = 'failed: ' + repr(_xe)
E1GEN['s'] = round(_xt.time() - _x0, 1)
print('033 E1GEN', E1GEN, flush=True)

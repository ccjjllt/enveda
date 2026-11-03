"""048 step2: hand-crafted spectrum-candidate features (own implementation, RDKit only).
- in-silico bond-cut fragments (1 cut, 2 acyclic cuts, ring-opening 2 cuts in same ring) with ±H shifts
- peak explanation coverage (count / intensity-weighted / top-10), neutral-loss explanation
- candidate descriptors + within-query relative features"""
import itertools, numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
RDLogger.DisableLog('rdApp.*')
PROTON = 1.007276; HM = 1.007825
ADD = {'[M+H]+': PROTON, '[M-H]-': -PROTON, '[M+Na]+': 22.989218, '[M+K]+': 38.963158, '[M+NH4]+': 18.033823,
       '[M+H-H2O]+': PROTON - 18.010565, '[M+Cl]-': 34.969402, '[M+FA-H]-': 44.998201}

def frag_masses(m):
    """neutral fragment masses (incl. implicit H) from 1-2 bond cuts."""
    m = Chem.AddHs(m)
    heavy = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1]
    w = np.zeros(m.GetNumAtoms()); adj = [[] for _ in range(m.GetNumAtoms())]
    for a in m.GetAtoms(): w[a.GetIdx()] = a.GetMass() if False else _mono(a)
    hb = []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); adj[i].append((j, b.GetIdx())); adj[j].append((i, b.GetIdx()))
        if m.GetAtomWithIdx(i).GetAtomicNum() > 1 and m.GetAtomWithIdx(j).GetAtomicNum() > 1: hb.append(b)
    tot = w.sum(); out = set()
    def comp(cut):
        seen = {0}; st = [0]; s = w[0]
        while st:
            u = st.pop()
            for v, bi in adj[u]:
                if bi not in cut and v not in seen: seen.add(v); st.append(v); s += w[v]
        return s
    acyc = [b.GetIdx() for b in hb if not b.IsInRing()]
    ri = m.GetRingInfo()
    for bi in acyc:
        s = comp({bi}); out.add(round(s, 4)); out.add(round(tot - s, 4))
    if len(acyc) <= 40:
        for a, b in itertools.combinations(acyc, 2):
            s1 = _comps(adj, w, {a, b}); out.update(s1)
    for ring in ri.BondRings():
        for a, b in itertools.combinations(ring, 2):
            out.update(_comps(adj, w, {a, b}))
    return np.array(sorted(out)), tot

def _comps(adj, w, cut):
    n = len(w); lab = [-1] * n; res = []
    for st0 in range(n):
        if lab[st0] >= 0: continue
        lab[st0] = st0; st = [st0]; s = w[st0]
        while st:
            u = st.pop()
            for v, bi in adj[u]:
                if bi not in cut and lab[v] < 0: lab[v] = st0; st.append(v); s += w[v]
        if s > 12: res.append(round(s, 4))
    return res

_MONO = {1: 1.007825, 6: 12.0, 7: 14.003074, 8: 15.994915, 9: 18.998403, 15: 30.973762, 16: 31.972071, 17: 34.968853,
         35: 78.918338, 53: 126.904473, 11: 22.989770, 14: 27.976927, 5: 11.009305, 34: 79.916521}
def _mono(a): return _MONO.get(a.GetAtomicNum(), a.GetMass())

def match(peaks, cands, tol_da=0.01, ppm=15):
    if not len(cands): return np.zeros(len(peaks), bool)
    i = np.searchsorted(cands, peaks); i0 = np.clip(i - 1, 0, len(cands) - 1); i1 = np.clip(i, 0, len(cands) - 1)
    d = np.minimum(abs(cands[i0] - peaks), abs(cands[i1] - peaks))
    return d <= np.maximum(tol_da, peaks * ppm * 1e-6)

_FMC = {}
def features(smiles, mz, it, prec, adduct, mode, desc=True):
    m = Chem.MolFromSmiles(smiles)
    f = {}
    if m is None: return f
    if smiles not in _FMC: _FMC[smiles] = frag_masses(m)
    fm, tot = _FMC[smiles]
    pos = str(mode or '').lower().startswith('p') or str(adduct).endswith('+')
    ion = PROTON if pos else -PROTON
    ions = np.unique(np.concatenate([fm + ion + k * HM for k in (-2, -1, 0, 1, 2)]))
    mz = np.asarray(mz, float); it = np.asarray(it, float)
    ok = (mz < prec - 0.5) & (it > 0); mz, it = mz[ok], it[ok]
    if len(mz) == 0: return f
    it = it / it.max()
    hit = match(mz, ions); hit0 = match(mz, np.unique(fm + ion))
    top = np.argsort(-it)[:10]
    f['fr_cov'] = hit.mean(); f['fr_icov'] = (it * hit).sum() / it.sum(); f['fr_top10'] = hit[top].mean()
    f['fr_icov0'] = (it * hit0).sum() / it.sum(); f['fr_cov0'] = hit0.mean()
    f['fr_n'] = np.log1p(len(fm)); f['fr_sqi'] = (np.sqrt(it) * hit).sum() / np.sqrt(it).sum()
    # neutral losses prec-peak vs neutral fragment masses (complements)
    nl = prec - mz
    nlh = match(nl, np.unique(np.concatenate([fm + k * HM for k in (-1, 0, 1)])), 0.01, 20)
    f['nl_cov'] = nlh.mean(); f['nl_icov'] = (it * nlh).sum() / it.sum()
    f['fr_any'] = (it * (hit | nlh)).sum() / it.sum()
    if not desc: return f
    # descriptors
    f['d_rings'] = rdMolDescriptors.CalcNumRings(m); f['d_arom'] = rdMolDescriptors.CalcNumAromaticRings(m)
    f['d_fsp3'] = rdMolDescriptors.CalcFractionCSP3(m); f['d_logp'] = Crippen.MolLogP(m)
    f['d_tpsa'] = rdMolDescriptors.CalcTPSA(m); f['d_hbd'] = rdMolDescriptors.CalcNumHBD(m)
    f['d_rot'] = rdMolDescriptors.CalcNumRotatableBonds(m); f['d_stereo'] = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    f['d_hetero_ring'] = rdMolDescriptors.CalcNumHeterocycles(m); f['d_bertz'] = np.log1p(Descriptors.BertzCT(m))
    f['d_charge'] = sum(abs(a.GetFormalCharge()) for a in m.GetAtoms())
    f['d_spiro'] = rdMolDescriptors.CalcNumSpiroAtoms(m); f['d_bridge'] = rdMolDescriptors.CalcNumBridgeheadAtoms(m)
    return f

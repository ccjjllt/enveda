'''044 OWN HARD-NEGATIVE RERANK: model3.pt (044, warm start from 042 model2 + candidate-pool hard-negative listwise loss), fp_w from train_result3.json.
040 OWN NEURAL RERANK (own_spec model trained by us, src/own_spec; replaces sanpier weights).
Based on 036 cell (public idea: sanpier "PubChem + Neural025", LB 0.465 -> 0.466): after EVERY baseline rule,
score the final <=25 candidates with sanpier/casmi-spectrum-model-v1 (spectrum transformer + fingerprint head,
trained on train.parquet) and fuse ranks: 1/(3+baseline_rank) + W/(3+neural_rank). Candidate SET never changes.
Any failure -> submission.csv untouched.'''
NEURAL_W, NEURAL_OFF = 0.25, 3.0
OWN_FP_W = 0.5; OWN_FP_W2 = 1.0  # 042 val 0.7636@1.0; 039 train_result.json: best val isomer MRR 0.7606 @0.5
NEURAL_HEALTH = 'NEURAL NOT RUN'
# 048 own fragment-explanation correction (src/fe048, own implementation): per spectrum
#   s = a*emb/10 + w . z(frag feats); fitted on 044 holdout isomer sets (val2 holdout +0.017 MRR).
import types as _t48, time as _time48
_fe = _t48.ModuleType('_fe048'); exec(compile('"""048 step2: hand-crafted spectrum-candidate features (own implementation, RDKit only).\n- in-silico bond-cut fragments (1 cut, 2 acyclic cuts, ring-opening 2 cuts in same ring) with ±H shifts\n- peak explanation coverage (count / intensity-weighted / top-10), neutral-loss explanation\n- candidate descriptors + within-query relative features"""\nimport itertools, numpy as np\nfrom rdkit import Chem, RDLogger\nfrom rdkit.Chem import Descriptors, rdMolDescriptors, Crippen\nRDLogger.DisableLog(\'rdApp.*\')\nPROTON = 1.007276; HM = 1.007825\nADD = {\'[M+H]+\': PROTON, \'[M-H]-\': -PROTON, \'[M+Na]+\': 22.989218, \'[M+K]+\': 38.963158, \'[M+NH4]+\': 18.033823,\n       \'[M+H-H2O]+\': PROTON - 18.010565, \'[M+Cl]-\': 34.969402, \'[M+FA-H]-\': 44.998201}\n\ndef frag_masses(m):\n    """neutral fragment masses (incl. implicit H) from 1-2 bond cuts."""\n    m = Chem.AddHs(m)\n    heavy = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1]\n    w = np.zeros(m.GetNumAtoms()); adj = [[] for _ in range(m.GetNumAtoms())]\n    for a in m.GetAtoms(): w[a.GetIdx()] = a.GetMass() if False else _mono(a)\n    hb = []\n    for b in m.GetBonds():\n        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); adj[i].append((j, b.GetIdx())); adj[j].append((i, b.GetIdx()))\n        if m.GetAtomWithIdx(i).GetAtomicNum() > 1 and m.GetAtomWithIdx(j).GetAtomicNum() > 1: hb.append(b)\n    tot = w.sum(); out = set()\n    def comp(cut):\n        seen = {0}; st = [0]; s = w[0]\n        while st:\n            u = st.pop()\n            for v, bi in adj[u]:\n                if bi not in cut and v not in seen: seen.add(v); st.append(v); s += w[v]\n        return s\n    acyc = [b.GetIdx() for b in hb if not b.IsInRing()]\n    ri = m.GetRingInfo()\n    for bi in acyc:\n        s = comp({bi}); out.add(round(s, 4)); out.add(round(tot - s, 4))\n    if len(acyc) <= 40:\n        for a, b in itertools.combinations(acyc, 2):\n            s1 = _comps(adj, w, {a, b}); out.update(s1)\n    for ring in ri.BondRings():\n        for a, b in itertools.combinations(ring, 2):\n            out.update(_comps(adj, w, {a, b}))\n    return np.array(sorted(out)), tot\n\ndef _comps(adj, w, cut):\n    n = len(w); lab = [-1] * n; res = []\n    for st0 in range(n):\n        if lab[st0] >= 0: continue\n        lab[st0] = st0; st = [st0]; s = w[st0]\n        while st:\n            u = st.pop()\n            for v, bi in adj[u]:\n                if bi not in cut and lab[v] < 0: lab[v] = st0; st.append(v); s += w[v]\n        if s > 12: res.append(round(s, 4))\n    return res\n\n_MONO = {1: 1.007825, 6: 12.0, 7: 14.003074, 8: 15.994915, 9: 18.998403, 15: 30.973762, 16: 31.972071, 17: 34.968853,\n         35: 78.918338, 53: 126.904473, 11: 22.989770, 14: 27.976927, 5: 11.009305, 34: 79.916521}\ndef _mono(a): return _MONO.get(a.GetAtomicNum(), a.GetMass())\n\ndef match(peaks, cands, tol_da=0.01, ppm=15):\n    if not len(cands): return np.zeros(len(peaks), bool)\n    i = np.searchsorted(cands, peaks); i0 = np.clip(i - 1, 0, len(cands) - 1); i1 = np.clip(i, 0, len(cands) - 1)\n    d = np.minimum(abs(cands[i0] - peaks), abs(cands[i1] - peaks))\n    return d <= np.maximum(tol_da, peaks * ppm * 1e-6)\n\n_FMC = {}\ndef features(smiles, mz, it, prec, adduct, mode, desc=True):\n    m = Chem.MolFromSmiles(smiles)\n    f = {}\n    if m is None: return f\n    if smiles not in _FMC: _FMC[smiles] = frag_masses(m)\n    fm, tot = _FMC[smiles]\n    pos = str(mode or \'\').lower().startswith(\'p\') or str(adduct).endswith(\'+\')\n    ion = PROTON if pos else -PROTON\n    ions = np.unique(np.concatenate([fm + ion + k * HM for k in (-2, -1, 0, 1, 2)]))\n    mz = np.asarray(mz, float); it = np.asarray(it, float)\n    ok = (mz < prec - 0.5) & (it > 0); mz, it = mz[ok], it[ok]\n    if len(mz) == 0: return f\n    it = it / it.max()\n    hit = match(mz, ions); hit0 = match(mz, np.unique(fm + ion))\n    top = np.argsort(-it)[:10]\n    f[\'fr_cov\'] = hit.mean(); f[\'fr_icov\'] = (it * hit).sum() / it.sum(); f[\'fr_top10\'] = hit[top].mean()\n    f[\'fr_icov0\'] = (it * hit0).sum() / it.sum(); f[\'fr_cov0\'] = hit0.mean()\n    f[\'fr_n\'] = np.log1p(len(fm)); f[\'fr_sqi\'] = (np.sqrt(it) * hit).sum() / np.sqrt(it).sum()\n    # neutral losses prec-peak vs neutral fragment masses (complements)\n    nl = prec - mz\n    nlh = match(nl, np.unique(np.concatenate([fm + k * HM for k in (-1, 0, 1)])), 0.01, 20)\n    f[\'nl_cov\'] = nlh.mean(); f[\'nl_icov\'] = (it * nlh).sum() / it.sum()\n    f[\'fr_any\'] = (it * (hit | nlh)).sum() / it.sum()\n    if not desc: return f\n    # descriptors\n    f[\'d_rings\'] = rdMolDescriptors.CalcNumRings(m); f[\'d_arom\'] = rdMolDescriptors.CalcNumAromaticRings(m)\n    f[\'d_fsp3\'] = rdMolDescriptors.CalcFractionCSP3(m); f[\'d_logp\'] = Crippen.MolLogP(m)\n    f[\'d_tpsa\'] = rdMolDescriptors.CalcTPSA(m); f[\'d_hbd\'] = rdMolDescriptors.CalcNumHBD(m)\n    f[\'d_rot\'] = rdMolDescriptors.CalcNumRotatableBonds(m); f[\'d_stereo\'] = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))\n    f[\'d_hetero_ring\'] = rdMolDescriptors.CalcNumHeterocycles(m); f[\'d_bertz\'] = np.log1p(Descriptors.BertzCT(m))\n    f[\'d_charge\'] = sum(abs(a.GetFormalCharge()) for a in m.GetAtoms())\n    f[\'d_spiro\'] = rdMolDescriptors.CalcNumSpiroAtoms(m); f[\'d_bridge\'] = rdMolDescriptors.CalcNumBridgeheadAtoms(m)\n    return f\n', '_fe048', 'exec'), _fe.__dict__)
_P48 = {"feats": ["fr_cov", "fr_icov", "fr_top10", "fr_icov0", "fr_sqi", "nl_cov", "nl_icov", "fr_any", "fr_n"], "mu": [0.19493596255779266, 0.2835582494735718, 0.28193843364715576, 0.08691997826099396, 0.2501400411128998, 0.11655904352664948, 0.194571852684021, 0.3351682126522064, 4.385617733001709], "sd": [0.14663909375667572, 0.29765585064888, 0.23521985113620758, 0.16451911628246307, 0.19357137382030487, 0.10678865015506744, 0.26614972949028015, 0.31602799892425537, 0.2349994033575058], "w": [0.1614580750465393, 0.37123197317123413, 0.5612485408782959, -0.19691339135169983, 0.40985357761383057, 0.020308842882514, 0.3668596148490906, -0.3444433808326721, -0.22382110357284546], "a": 2.7672150135040283, "base": "emb", "base_div": 10.0}
_T48 = _time48.time(); _FR = dict(ok=0, err=0, timeout=0)
def _fr_adjust(_S, _raw, _c):
    """_S (spectra, cands) model score with fp_w=0 -> corrected score; fallback = _S unchanged."""
    if _time48.time() - _T48 > 5400 or not _raw:
        _FR['timeout'] += 1; return _S
    try:
        _mu, _sd, _w = _np.array(_P48['mu']), _np.array(_P48['sd']), _np.array(_P48['w'])
        _cor = _np.zeros((min(len(_raw), 16), len(_c)))
        for _i, (_mz, _it, _pr, _ad, _mo) in enumerate(_raw[:16]):
            for _j, _s in enumerate(_c):
                _f = _fe.features(_s, _mz, _it, _pr, _ad, _mo, desc=False)
                _x = _np.array([_f.get(k, 0.0) for k in _P48['feats']], float)
                _cor[_i, _j] = ((_x - _mu) / _sd) @ _w
        _FR['ok'] += 1
        return _P48['a'] * _S / _P48['base_div'] + _cor.mean(0)[None]
    except Exception:
        _FR['err'] += 1; return _S
try:
    import glob as _g, json as _j, types as _t, importlib.util as _iu, shutil as _sh
    import numpy as _np, pandas as _pd, torch as _torch
    from pathlib import Path as _P
    _cfgs = sorted(_g.glob('/kaggle/input/**/own_spec/model.pt', recursive=True), key=len) or sorted([p for p in _g.glob('/kaggle/input/**/model.pt', recursive=True) if os.path.exists(os.path.join(os.path.dirname(p), 'feats.py')) and 'sanpier' not in p and 'spectrum-model' not in p], key=len)
    assert _cfgs, 'own_spec bundle not mounted'
    _B = _P(_cfgs[0]).parent
    _pkg = '_casmi_own'; _m = _t.ModuleType(_pkg); _m.__path__ = [str(_B)]; sys.modules[_pkg] = _m
    for _n in ['feats', 'model']:
        _sp = _iu.spec_from_file_location(_pkg + '.' + _n, _B / (_n + '.py'))
        _mod = _iu.module_from_spec(_sp); sys.modules[_sp.name] = _mod; _sp.loader.exec_module(_mod)
    _feat = sys.modules[_pkg + '.feats']
    _dev = 'cuda' if _torch.cuda.is_available() else 'cpu'
    _ck = _torch.load(_B / 'model.pt', map_location=_dev, weights_only=True)
    _net = sys.modules[_pkg + '.model'].OwnSpec(**_ck['cfg']).to(_dev)
    _net.load_state_dict(_ck['state_dict']); _net.eval()
    assert (_B / 'model3.pt').exists(), 'model3.pt missing'
    _r3 = _j.load(open(_B / 'train_result3.json'))['val']
    _fw3 = max(_r3['isomer'], key=lambda k: (_r3['isomer'][k] or 0) + (_r3['pool'][k] or 0))
    OWN_FP_W3 = float(_fw3)
    _ck3 = _torch.load(_B / 'model3.pt', map_location=_dev, weights_only=True)
    _n3 = sys.modules[_pkg + '.model'].OwnSpec(**_ck3['cfg']).to(_dev); _n3.load_state_dict(_ck3['state_dict']); _n3.eval()
    _nets = [(_n3, OWN_FP_W3)]
    print('044 HN model3 fp_w', OWN_FP_W3, flush=True)
    _sub_p = '/kaggle/working/submission.csv' if os.path.exists('/kaggle/working/submission.csv') else 'submission.csv'
    _sub = _pd.read_csv(_sub_p, dtype=str, keep_default_na=False)
    _sh.copy2(_sub_p, _sub_p.replace('submission.csv', 'submission_pre_ownneural.csv'))
    _te = te.copy(); _te['molecule_id'] = _te.molecule_id.astype(str)
    _Q = {m: g for m, g in _te.groupby('molecule_id', sort=False)}
    _cache, _out, _st = {}, [], dict(scored=0, skipped=0, changed=0, changed_top1=0)
    with _torch.inference_mode():
        for _mid, _txt in _sub.itertuples(index=False):
            _c = _txt.split(';')
            try:
                assert _mid in _Q and len(_c) > 1
                _rec, _raw = [], []
                for _r in _Q[_mid].itertuples(index=False):
                    try:
                        _pk, _nn = _feat.spectrum(_r.ms2_mzs, _r.ms2_normalized_intensities, _r.precursor_mz)
                        if _nn < 3:
                            continue
                        _me, _ad, _in = _feat.meta(_r.precursor_mz, _r.adduct, _r.collision_energy_ev, getattr(_r, 'instrument_type', None))
                        _rec.append((_pk, _nn, _me, _ad, _in)); _raw.append((_r.ms2_mzs, _r.ms2_normalized_intensities, float(_r.precursor_mz), _r.adduct, _r.ionization_mode))
                    except (ValueError, TypeError):
                        continue
                assert _rec, 'no usable spectra'
                _fp = []
                for _s in _c:
                    if _s not in _cache:
                        _cache[_s] = _feat.mol_fp(_s)[1]; assert _cache[_s] is not None
                    _fp.append(_cache[_s])
                _F = _torch.tensor(_np.stack(_fp), device=_dev)
                _s_mol = 0
                for _net, _fw in _nets:
                    _M = _net.molecule(_F); _sc = []
                    for _i in range(0, len(_rec), 64):
                        _b = _rec[_i:_i + 64]
                        _Z, _PP = _net.spec(_torch.tensor(_np.stack([v[0] for v in _b]), device=_dev),
                                            _torch.tensor([v[1] for v in _b], device=_dev, dtype=_torch.long),
                                            _torch.tensor(_np.stack([v[2] for v in _b]), device=_dev),
                                            _torch.tensor([v[3] for v in _b], device=_dev, dtype=_torch.long),
                                            _torch.tensor([v[4] for v in _b], device=_dev, dtype=_torch.long))
                        _sc.append(_net.score(_Z, _PP, _M, _F, _fw).float().cpu().numpy())
                    _sm = _fr_adjust(_np.concatenate(_sc), _raw, _c).mean(0)
                    _s_mol = _s_mol + (_sm - _sm.mean()) / (_sm.std() + 1e-6)
                assert _np.isfinite(_s_mol).all()
                _o = _np.argsort(-_s_mol, kind='stable'); _nr = _np.empty(len(_c), int); _nr[_o] = _np.arange(1, len(_c) + 1)
                _br = _np.arange(1, len(_c) + 1)
                _new = [_c[i] for i in _np.argsort(-(1 / (NEURAL_OFF + _br) + NEURAL_W / (NEURAL_OFF + _nr)), kind='stable')]
                assert sorted(_new) == sorted(_c)
                _st['scored'] += 1; _st['changed'] += _new != _c; _st['changed_top1'] += _new[0] != _c[0]
                _out.append((_mid, ';'.join(_new)))
            except Exception:
                _st['skipped'] += 1; _out.append((_mid, _txt))
    _res = _pd.DataFrame(_out, columns=['molecule_id', 'smiles'])
    assert _res.molecule_id.tolist() == _sub.molecule_id.tolist()
    _res.to_csv(_sub_p, index=False)
    NEURAL_HEALTH = f'NEURAL OK {_st}'
except Exception as _e:
    NEURAL_HEALTH = f'NEURAL FAILED (submission untouched): {_e!r}'
print('048 FRAGCORR', _FR, flush=True)
print('044 HN', len(_nets) if '_nets' in dir() else 0, 'OWN', NEURAL_HEALTH, flush=True)

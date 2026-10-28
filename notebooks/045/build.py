"""Build 045a (leader-safe fusion) and 045b (045a + fragment rank-1 rule) from the 044 notebook.
Credits: leader-safe ranks 2-25 -- alex chilton / Roman Rozen (public 0.472 notebooks); merge recipe -- busyaprime
(0.477 "two 0.472 fixes merged the right way"); MetFrag-lite fragment check -- Roman Tamrazov (Spectral Parallax).
Ported code: Apache-2.0 public notebooks. Neural model = our own model3 (no new datasets)."""
import json, copy, sys
W_MID = float(sys.argv[1]) if len(sys.argv) > 1 else 0.75
nb = json.load(open('../044/enveda-044-own-hn.ipynb'))
ix = [i for i, c in enumerate(nb['cells']) if 'NEURAL_W, NEURAL_OFF' in ''.join(c['source'])]; assert len(ix) == 1; ix = ix[0]
src = ''.join(nb['cells'][ix]['source'])
OLD = "_new = [_c[i] for i in _np.argsort(-(1 / (NEURAL_OFF + _br) + NEURAL_W / (NEURAL_OFF + _nr)), kind='stable')]"
assert OLD in src
LS = f'''NEURAL_W_MID = {W_MID}  # 045 leader-safe (alex chilton / Roman Rozen public 0.472): rank1 from W=0.25, ranks 2-25 at W_MID
def _ls_order(_br, _nr, _wm):
    _v1 = 1 / (NEURAL_OFF + _br) + NEURAL_W / (NEURAL_OFF + _nr); _ld = int(_np.argmax(_v1))
    _s = 1 / (NEURAL_OFF + _br) + _wm / (NEURAL_OFF + _nr); _s[_ld] = _np.inf
    return list(_np.argsort(-_s, kind='stable'))
'''
NEW_A = "_new = [_c[i] for i in _ls_order(_br, _nr, NEURAL_W_MID)]"
FRAG = open('frag_physical.py').read()
NEW_B = '''_new = [_c[i] for i in _ls_order(_br, _nr, NEURAL_W_MID)]
                try:
                    _fw_ = _frag_top1(_mid, [_c[i] for i in _ls_order(_br, _nr, 0.5)], dict(zip(_c, _s_mol)))
                except Exception as _fe:
                    _fw_ = None; _st['frag_err'] = _st.get('frag_err', 0) + 1
                if _fw_ is not None and _fw_ != _new[0]:
                    _old = _new[0]; _new = [_fw_, _old] + [x for x in _new if x not in (_fw_, _old)]; _st['frag_top1'] = _st.get('frag_top1', 0) + 1'''
FRAG_HELP = '''# 045b fragment rank-1 rule (Roman Tamrazov "Spectral Parallax" MetFrag-lite; merge per busyaprime 0.477):
# run on the W_MID=0.5 leader-safe order; keep only its leader change; old leader -> rank 2; rest = W_MID order.
import time as _time, json as _json2
from rdkit.Chem import rdMolDescriptors as _rmd
_FRAG_SRC = %r
_fm = _t.ModuleType('_frag_phys'); exec(compile(_FRAG_SRC, '_frag_phys', 'exec'), _fm.__dict__)
_fcache, _FT0 = {}, _time.time()
try:
    _E4 = _json2.load(open('e004_candidate_inputs.json'))
except Exception:
    _E4 = {}
def _ff(s):
    _mm = Chem.MolFromSmiles(s); return _rmd.CalcMolFormula(_mm) if _mm is not None else None
def _fsel(frame, max_n=12):
    out, bb = [], {}
    for r in frame.itertuples(index=False):
        mz = _np.asarray(r.ms2_mzs, float).ravel(); it = _np.asarray(r.ms2_normalized_intensities, float).ravel()
        n = min(len(mz), len(it)); mz, it = mz[:n], it[:n]; g = _np.isfinite(mz) & _np.isfinite(it) & (it > 0)
        if g.sum() < 5: continue
        o = _np.argsort(it[g])[-120:]; mz, it = mz[g][o], it[g][o]
        try: e = float(_np.mean(_np.asarray(r.collision_energy_ev, float)))
        except Exception: e = 30.
        if not _np.isfinite(e): e = 30.
        ad = str(r.adduct); md = 1 if str(r.ionization_mode).lower() == 'positive' else -1
        it = it / it.max()
        bb.setdefault((ad, round(e / 10)), []).append((int(g.sum()), dict(mz=mz, it=it, adduct=ad, prec=float(r.precursor_mz), mode=md)))
    for k in sorted(bb): out += [v for _, v in sorted(bb[k], key=lambda x: -x[0])[:2]]
    return out[:max_n]
def _frank(mid, key, s):
    v = (_E4.get(key, {}).get(mid) or {})
    v = v.get('smiles') if key == 'engine2' else (v.get('pc') if key == 'pubchem' else v)
    v = v or []
    return v.index(s) + 1 if s in v else 0
def _frag_top1(mid, ctl, neu):
    if _time.time() - _FT0 > 1500: return None
    lib = float(BASE.get(mid, ([], [], 0.0))[2]) if 'BASE' in globals() else 0.0
    if lib >= .87 or len(ctl) < 3: return None
    anc = _ff(ctl[0]); spec = _fsel(_Q[mid])
    if anc is None or len(spec) < 3: return None
    cand = ctl[:min(6, len(ctl))]
    fn = _np.asarray([_np.nan if a is None else a for a in _fm.score_molecule(spec, cand, cache=_fcache)], float)
    fo = _np.asarray([_np.nan if a is None else a for a in _fm.score_molecule(spec, cand, cfg=_fm.BASELINE_CFG, cache=_fcache)], float)
    if not (_np.isfinite(fn).all() and _np.isfinite(fo).all()): return None
    en = _np.array([[_np.nan if v is None else v for v in _fm.score_molecule(spec[i:i + 2], cand, cache=_fcache)] for i in range(0, len(spec), 2)], float)
    best = None
    for j in range(1, len(cand)):
        s = cand[j]
        if _ff(s) != anc or fn[j] < .27: continue
        mg = fn[j] - fn[0]
        if mg < .095 or fo[j] - fo[0] < .028: continue
        if (en[:, j] > en[:, 0] + .025).mean() < .70: continue
        nm = float(neu[s] - neu[ctl[0]])  # model3 score is z-scored per molecule
        if nm < .23: continue
        sup = sum(0 < _frank(mid, k, s) <= 5 for k in ('engine2', 'pubchem', 'base_tail'))
        if sup < 1: continue
        sc = mg + .05 * nm + .01 * sup
        if best is None or sc > best[0]: best = (sc, s)
    return None if best is None else best[1]
''' % FRAG
def make(tag, new, extra):
    s = src.replace(OLD, new).replace("NEURAL_W, NEURAL_OFF = 0.25, 3.0", "NEURAL_W, NEURAL_OFF = 0.25, 3.0\n" + LS)
    s = s.replace("print('044 HN', len(_nets)", f"print('{tag}', NEURAL_W_MID, '044 HN', len(_nets)")
    if extra: s = s.replace("    _sub_p = '/kaggle/working/submission.csv'", extra + "    _sub_p = '/kaggle/working/submission.csv'", 1)
    s = '# ' + __doc__.replace('\n', '\n# ') + '\n' + s
    n2 = copy.deepcopy(nb); n2['cells'][ix]['source'] = s.splitlines(True); n2['cells'][ix]['outputs'] = []
    compile(s, tag, 'exec'); return n2
FRAG_IND = '    try:\n' + '\n'.join('        ' + l if l else l for l in FRAG_HELP.split('\n')) + '\n    except Exception as _fie:\n        print("045 FRAG init failed", repr(_fie)); _frag_top1 = lambda *a: None\n'
json.dump(make('045 LS', NEW_A, ''), open('enveda-045a-leader-safe.ipynb', 'w'), indent=1)
json.dump(make('045 FRAG', NEW_B, FRAG_IND), open('enveda-045b-frag.ipynb', 'w'), indent=1)
print('built W_MID', W_MID)

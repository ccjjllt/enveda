"""Build 046B (track A): 046A + ICE/GL simulator_vote in the fragment rank-1 rule, aligned with the public 0.477 gates
(busyaprime "LB 0.477 two 0.472 fixes merged the right way", Spectral Parallax decision()):
  supports = #{engine2, pubchem, main} metric-identity rank in 1..5 (main = final baseline list BASE[mid][0]);
  simulator_vote = sum over (ICE, GL): +1 if cand >= leader, -1 if cand < leader - 0.35 (both finite);
  reject if supports<1 and sim<1; reject if sim<=-1; contender score = margin + .05 nm + .025 sim + .01 supports.
Credit: sanpier (neural model), alex chilton / Roman Rozen (leader-safe), Roman Tamrazov (Spectral Parallax), busyaprime (merge)."""
import json, copy
nb = json.load(open('enveda-046-sanpier-ls-frag.ipynb'))
ix = [i for i, c in enumerate(nb['cells']) if 'NEURAL_W, NEURAL_OFF' in ''.join(c['source'])]; assert len(ix) == 1; ix = ix[0]
src = ''.join(nb['cells'][ix]['source'])
I = ' ' * 16
OLD = (I + "sup = sum(0 < _frank(mid, k, s) <= 5 for k in ('engine2', 'pubchem', 'base_tail'))\n"
       + I + "if sup < 1: continue\n"
       + I + "sc = mg + .05 * nm + .01 * sup\n")
NEW = (I + "sup = _fsup(mid, s)\n"
       + I + "sim = _fsim(mid, s, ctl[0])\n"
       + I + "if sup < 1 and sim < 1: _FSIM['gate_rej'] += 1; continue\n"
       + I + "if sim <= -1: _FSIM['sim_veto'] += 1; continue\n"
       + I + "_FSIM['sim_rescue'] += int(sup < 1)\n"
       + I + "sc = mg + .05 * nm + .025 * sim + .01 * sup\n")
HELP = '''_FSIM = dict(sim_rescue=0, sim_veto=0, gate_rej=0, sup_err=0, sim_err=0)
_FRMAP = {}
def _fsup(mid, s):
    try:
        if mid not in _FRMAP:
            def _rm(v):
                o = {}
                for r, x in enumerate(v or [], 1):
                    k = metric_identity(x)
                    if k is not None: o.setdefault(k, r)
                return o
            _FRMAP[mid] = [_rm(BASE.get(mid, ([],))[0]), _rm((ENG.get(mid) or {}).get('smiles')), _rm((PC.get(mid) or {}).get('pc'))]
        k = metric_identity(s)
        return sum(0 < m.get(k, 0) <= 5 for m in _FRMAP[mid])
    except Exception:
        _FSIM['sup_err'] += 1
        return sum(0 < _frank(mid, k, s) <= 5 for k in ('engine2', 'pubchem', 'base_tail'))
def _ffin(v):
    try:
        v = float(v); return v if _np.isfinite(v) else _np.nan
    except (TypeError, ValueError):
        return _np.nan
def _fsim(mid, s, ld):
    try:
        ice = (ICE_SCORES.get(str(mid), {}) if isinstance(ICE_SCORES, dict) else {}) or {}
        gl = gl_of(mid) or {}
        v = 0
        for d in (ice, gl):
            a = _ffin(d.get(s, d.get(chem.score_key(s)))); b = _ffin(d.get(ld, d.get(chem.score_key(ld))))
            if _np.isfinite(a) and _np.isfinite(b):
                if a >= b: v += 1
                elif a < b - .35: v -= 1
        return v
    except Exception:
        _FSIM['sim_err'] += 1; return 0
'''
assert src.count(OLD) == 1
src = src.replace(OLD, NEW)
A = "        def _frag_top1(mid, ctl, neu, cos, fpv):\n"; assert src.count(A) == 1
src = src.replace(A, '\n'.join('        ' + l if l else l for l in HELP.split('\n')) + A)
P = "print('046 SANPIER LS FRAG', NEURAL_W, NEURAL_W_MID, NEURAL_W_FRAG, NEURAL_HEALTH, flush=True)"; assert src.count(P) == 1
src = src.replace(P, "print('046B SANPIER LS FRAG SIM', NEURAL_W, NEURAL_W_MID, NEURAL_W_FRAG, NEURAL_HEALTH, globals().get('_FSIM'), flush=True)")
src = '# ' + __doc__.replace('\n', '\n# ') + '\n' + src
compile(src, '046b', 'exec')
n2 = copy.deepcopy(nb); n2['cells'][ix]['source'] = src.splitlines(True); n2['cells'][ix]['outputs'] = []
json.dump(n2, open('enveda-046b-sanpier-sim.ipynb', 'w'), indent=1)
print('built 046B')

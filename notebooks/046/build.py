"""Build 046-sanpier (track A, medal track): our 036 pipeline + sanpier/casmi-spectrum-model-v1 neural rerank,
reproducing the public 0.477 recipe:
  rank 1 from W_TOP=0.25 fusion; ranks 2-25 at W_MID=0.75 (alex chilton / Roman Rozen, public 0.472);
  fragment rank-1 rule (Roman Tamrazov "Spectral Parallax" MetFrag-lite) run on the W_MID=0.5 (V2) order, only its
  leader move kept: [frag_leader, V2_leader] + rest of 0.75 order (merge recipe: busyaprime, public 0.477).
Neural model: Saner Turhaner (sanpier) casmi-spectrum-model-v1 -- license unknown; NOT prize-eligible track."""
import json, copy, sys
W_TOP, W_MID, W_FRAG = 0.25, float(sys.argv[1]) if len(sys.argv) > 1 else 0.75, 0.5
nb = json.load(open('../036/enveda-036-v15-neural.ipynb'))
ix = [i for i, c in enumerate(nb['cells']) if 'NEURAL_W, NEURAL_OFF' in ''.join(c['source'])]; assert len(ix) == 1; ix = ix[0]
src = ''.join(nb['cells'][ix]['source'])
rep = [
 ("NEURAL_W, NEURAL_OFF = 0.40, 3.0", f"NEURAL_W, NEURAL_OFF = {W_TOP}, 3.0\nNEURAL_W_MID, NEURAL_W_FRAG = {W_MID}, {W_FRAG}\n" + '''def _ls_order(_br, _nr, _wm):
    _v1 = 1 / (NEURAL_OFF + _br) + NEURAL_W / (NEURAL_OFF + _nr); _ld = int(_np.argmax(_v1))
    _s = 1 / (NEURAL_OFF + _br) + _wm / (NEURAL_OFF + _nr); _s[_ld] = _np.inf
    return list(_np.argsort(-_s, kind='stable'))
'''),
 ("_sc = []\n", "_sc, _scos, _sfp = [], [], []\n"),
 ("                    _sc.append(((_Z @ _M.T) / _cfg['temperature'] + _cfg['fp_score_weight'] * (_PP @ _F.T / _cfg['fp_bits'])).cpu().numpy())",
  "                    _cs_ = (_Z @ _M.T).cpu().numpy(); _fp_ = (_PP @ _F.T / _cfg['fp_bits']).cpu().numpy(); _scos.append(_cs_); _sfp.append(_fp_)\n"
  "                    _sc.append(_cs_ / _cfg['temperature'] + _cfg['fp_score_weight'] * _fp_)"),
 ("                _new = [_c[i] for i in _np.argsort(-(1 / (NEURAL_OFF + _br) + NEURAL_W / (NEURAL_OFF + _nr)), kind='stable')]",
  '''                _cos_m = dict(zip(_c, _np.concatenate(_scos).mean(0))); _fp_m = dict(zip(_c, _np.concatenate(_sfp).mean(0)))
                _new = [_c[i] for i in _ls_order(_br, _nr, NEURAL_W_MID)]
                _v2 = [_c[i] for i in _ls_order(_br, _nr, NEURAL_W_FRAG)]
                assert _v2[0] == _new[0]
                try:
                    _fw_ = _frag_top1(_mid, _v2, dict(zip(_c, _s_mol)), _cos_m, _fp_m)
                except Exception as _fe:
                    _fw_ = None; _st['frag_err'] = _st.get('frag_err', 0) + 1
                if _fw_ is not None and _fw_ != _v2[0]:
                    _old = _v2[0]; _new = [_fw_, _old] + [x for x in _new if x not in (_fw_, _old)]; _st['frag_top1'] = _st.get('frag_top1', 0) + 1'''),
 ("print('036', NEURAL_HEALTH", "print('046 SANPIER LS FRAG', NEURAL_W, NEURAL_W_MID, NEURAL_W_FRAG, NEURAL_HEALTH"),
]
for a, b in rep:
    assert src.count(a) == 1, a; src = src.replace(a, b)
H = open('../045/build.py').read()
FRAG_HELP = H[H.index("FRAG_HELP = '''") + 15:H.index("''' % FRAG")]
FRAG_HELP = FRAG_HELP.replace("def _frag_top1(mid, ctl, neu):", "def _frag_top1(mid, ctl, neu, cos, fpv):")
OLDN = "        nm = float(neu[s] - neu[ctl[0]])  # model3 score is z-scored per molecule\n        if nm < .23: continue\n"
assert OLDN in FRAG_HELP
FRAG_HELP = FRAG_HELP.replace(OLDN, "        nm = float(neu[s] - neu[ctl[0]]) / max(float(_np.std(list(neu.values()))), 1e-6)\n        if nm < .23: continue\n"
    "        if cos[s] < cos[ctl[0]] + .003 or fpv[s] < fpv[ctl[0]] - .012: continue\n")
FRAG_HELP = FRAG_HELP.replace("045b fragment", "046 fragment") % open('frag_physical.py').read()
FRAG_IND = '    try:\n' + '\n'.join('        ' + l if l else l for l in FRAG_HELP.split('\n')) + '\n    except Exception as _fie:\n        print("046 FRAG init failed", repr(_fie)); _frag_top1 = lambda *a: None\n'
A = "    _sub_p = '/kaggle/working/submission.csv'"; assert src.count(A) == 1
src = src.replace(A, FRAG_IND + A)
src = '# ' + __doc__.replace('\n', '\n# ') + '\n' + src
compile(src, '046', 'exec')
n2 = copy.deepcopy(nb); n2['cells'][ix]['source'] = src.splitlines(True); n2['cells'][ix]['outputs'] = []
json.dump(n2, open('enveda-046-sanpier-ls-frag.ipynb', 'w'), indent=1)
m = json.load(open('../036/kernel-metadata.json'))
m.update(id='ccjjllt/enveda-046-sanpier', title='enveda-046-sanpier', code_file='enveda-046-sanpier-ls-frag.ipynb', is_private=True)
json.dump(m, open('kernel-metadata.json', 'w'), indent=1)
print('built', W_TOP, W_MID, W_FRAG)

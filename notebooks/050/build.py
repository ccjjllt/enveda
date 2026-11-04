"""050: 046A sanpier pipeline (W_TOP0.25 / leader-safe W_MID0.75 / frag rank-1) + round-048 own fragment-explanation /
neutral-loss linear residual (src/fe048) applied on the A track as a rank term W_FR/(3+frag_rank) in all three fusion orders.
Own cell = 048 cell (044 model3 + fe048) but stores ONLY the frag-residual rank per molecule (FR_RANK); never writes submission.
Sanpier cell additionally recomputes the exact 046A decision (term=0) and logs top1_vs_046A. Any failure -> 046A behaviour.
W_FR = argv[1] (default 0.15)."""
import json, copy, sys, subprocess
W_FR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
subprocess.run([sys.executable, 'build.py', '0.75'], cwd='../047', check=True)
nb = json.load(open('../047/enveda-047-sanpier-wmid.ipynb'))
ix = len(nb['cells']) - 1; src = ''.join(nb['cells'][ix]['source']); assert 'NEURAL_W_MID' in src
subprocess.run([sys.executable, 'build.py'], cwd='../048', check=True)
o = open('../048/cell_048.py').read()
orep = [
 ("        _FR['ok'] += 1\n", "        _FR['ok'] += 1; _FC48[0] = _cor.mean(0)\n"),
 ("_cache, _out, _st = {}, [], dict(", "FR_RANK = {}\n    _cache, _out, _st = {}, [], dict("),
 ("            _c = _txt.split(';')\n", "            _c = _txt.split(';'); _FC48[0] = None\n"),
 ("                _new = [_c[i] for i in _np.argsort(-(1 / (NEURAL_OFF + _br) + NEURAL_W / (NEURAL_OFF + _nr)), kind='stable')]\n",
  "                if _FC48[0] is not None and len(_FC48[0]) == len(_c):\n"
  "                    _fo = _np.argsort(-_FC48[0], kind='stable'); _frk = _np.empty(len(_c), int); _frk[_fo] = _np.arange(1, len(_c) + 1)\n"
  "                    FR_RANK[_mid] = dict(zip(_c, _frk.tolist()))\n"
  "                _new = list(_c)\n"),
 ("    _res.to_csv(_sub_p, index=False)\n", ""),
 ("    _sh.copy2(_sub_p, _sub_p.replace('submission.csv', 'submission_pre_ownneural.csv'))\n", ""),
 ("    NEURAL_HEALTH = f'NEURAL OK {_st}'", "    NEURAL_HEALTH = f'FRRANK READY {_st}'"),
 ("print('044 HN',", "print('050 FR', len(FR_RANK),"),
]
for a, b in orep:
    assert o.count(a) == 1, a; o = o.replace(a, b)
o = "FR_RANK = {}; _FC48 = [None]\n" + o
compile(o, 'own', 'exec')
rep = [
 ("def _ls_order(_br, _nr, _wm):\n    _v1 = 1 / (NEURAL_OFF + _br) + NEURAL_W / (NEURAL_OFF + _nr); _ld = int(_np.argmax(_v1))\n    _s = 1 / (NEURAL_OFF + _br) + _wm / (NEURAL_OFF + _nr); _s[_ld] = _np.inf",
  f"W_FR = {W_FR}; _OWT = [0.0]\n"
  "def _ls_order(_br, _nr, _wm):\n    _v1 = 1 / (NEURAL_OFF + _br) + NEURAL_W / (NEURAL_OFF + _nr) + _OWT[0]; _ld = int(_np.argmax(_v1))\n    _s = 1 / (NEURAL_OFF + _br) + _wm / (NEURAL_OFF + _nr) + _OWT[0]; _s[_ld] = _np.inf"),
 ("                _new = [_c[i] for i in _ls_order(_br, _nr, NEURAL_W_MID)]\n",
  "                _OWT[0] = 0.0\n"
  "                _bn = [_c[i] for i in _ls_order(_br, _nr, NEURAL_W_MID)]; _bv = [_c[i] for i in _ls_order(_br, _nr, NEURAL_W_FRAG)]\n"
  "                try:\n                    _bf = _frag_top1(_mid, _bv, dict(zip(_c, _s_mol)), _cos_m, _fp_m)\n"
  "                except Exception:\n                    _bf = None\n"
  "                _btop = _bf if (_bf is not None and _bf != _bv[0]) else _bn[0]\n"
  "                _frm = FR_RANK.get(_mid) if 'FR_RANK' in globals() else None\n"
  "                if _frm and all(x in _frm for x in _c):\n"
  "                    _OWT[0] = W_FR / (NEURAL_OFF + _np.array([_frm[x] for x in _c], float)); _st['fr_used'] = _st.get('fr_used', 0) + 1\n"
  "                _new = [_c[i] for i in _ls_order(_br, _nr, NEURAL_W_MID)]\n"),
 ("                _st['scored'] += 1;", "                _st['top1_vs_046A'] = _st.get('top1_vs_046A', 0) + (_new[0] != _btop); _OWT[0] = 0.0\n                _st['scored'] += 1;"),
 ("print('047b SANPIER WMID SCAN'", "print('050 FRA', W_FR, len(FR_RANK) if 'FR_RANK' in globals() else -1"),
]
for a, b in rep:
    assert src.count(a) == 1, a; src = src.replace(a, b)
src = src.replace('# 047: 046A with W_MID', '# 050 (046A + fe048 frag residual rank, see notebooks/050/build.py) on 047: 046A with W_MID', 1)
compile(src, '050', 'exec')
n2 = copy.deepcopy(nb); n2['cells'][ix]['source'] = src.splitlines(True); n2['cells'][ix]['outputs'] = []
oc = copy.deepcopy(nb['cells'][ix]); oc['source'] = o.splitlines(True); oc['outputs'] = []; oc.pop('id', None)
n2['cells'].insert(ix, oc)
json.dump(n2, open('enveda-050-sanpier-fragres.ipynb', 'w'), indent=1)
m = json.load(open('../047/kernel-metadata.json'))
m['dataset_sources'] = m['dataset_sources'] + ['ccjjllt/enveda-own-spec']; m['code_file'] = 'enveda-050-sanpier-fragres.ipynb'
json.dump(m, open('kernel-metadata.json', 'w'), indent=1)
subprocess.run(['git', 'checkout', '--', 'enveda-047-sanpier-wmid.ipynb', 'kernel-metadata.json'], cwd='../047', check=True)
subprocess.run(['git', 'checkout', '--', '.'], cwd='../048', check=True)
print('built 050 W_FR', W_FR)

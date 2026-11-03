"""049: A x B ensemble = 046A sanpier pipeline (W_TOP0.25 / leader-safe W_MID0.75 / frag rank-1) + own model3 (044/048 B track)
rank term W_OWN/(3+own_rank) added to all three fusion orders (V1 leader, W_MID order, V2 frag-control order).
Own cell runs BEFORE the sanpier cell on the same submission lists, only stores ranks (never writes submission).
Any own-model failure -> OWN_RANK empty -> exact 046A behaviour.  W_OWN = argv[1] (default 0.20)."""
import json, copy, sys, subprocess
W_OWN = float(sys.argv[1]) if len(sys.argv) > 1 else 0.20
subprocess.run([sys.executable, 'build.py', '0.75'], cwd='../047', check=True)  # regenerate 046A-equivalent notebook
nb = json.load(open('../047/enveda-047-sanpier-wmid.ipynb'))
ix = len(nb['cells']) - 1; src = ''.join(nb['cells'][ix]['source']); assert 'NEURAL_W_MID' in src
# --- own cell (044 model3 plain) -> OWN_RANK only
o = ''.join(json.load(open('../044/enveda-044-own-hn.ipynb'))['cells'][-1]['source'])
orep = [
 ("_cache, _out, _st = {}, [], dict(", "OWN_RANK = {}\n    _cache, _out, _st = {}, [], dict("),
 ("                _new = [_c[i] for i in _np.argsort(-(1 / (NEURAL_OFF + _br) + NEURAL_W / (NEURAL_OFF + _nr)), kind='stable')]\n",
  "                OWN_RANK[_mid] = dict(zip(_c, _nr.tolist())); _new = list(_c)\n"),
 ("    _res.to_csv(_sub_p, index=False)\n", ""),
 ("    _sh.copy2(_sub_p, _sub_p.replace('submission.csv', 'submission_pre_ownneural.csv'))\n", ""),
 ("    NEURAL_HEALTH = f'NEURAL OK {_st}'", "    NEURAL_HEALTH = f'OWNRANK READY {_st}'"),
 ("print('044 HN',", "print('049 OWN',"),
]
for a, b in orep:
    assert o.count(a) == 1, a; o = o.replace(a, b)
o = "OWN_RANK = {}\n" + o
compile(o, 'own', 'exec')
# --- sanpier cell: add own rank term
rep = [
 ("def _ls_order(_br, _nr, _wm):\n    _v1 = 1 / (NEURAL_OFF + _br) + NEURAL_W / (NEURAL_OFF + _nr); _ld = int(_np.argmax(_v1))\n    _s = 1 / (NEURAL_OFF + _br) + _wm / (NEURAL_OFF + _nr); _s[_ld] = _np.inf",
  f"W_OWN = {W_OWN}; _OWT = [0.0]\n"
  "def _ls_order(_br, _nr, _wm):\n    _v1 = 1 / (NEURAL_OFF + _br) + NEURAL_W / (NEURAL_OFF + _nr) + _OWT[0]; _ld = int(_np.argmax(_v1))\n    _s = 1 / (NEURAL_OFF + _br) + _wm / (NEURAL_OFF + _nr) + _OWT[0]; _s[_ld] = _np.inf"),
 ("                _br = _np.arange(1, len(_c) + 1)\n",
  "                _br = _np.arange(1, len(_c) + 1)\n"
  "                _orm = OWN_RANK.get(_mid) if 'OWN_RANK' in globals() else None\n"
  "                if _orm and all(x in _orm for x in _c):\n"
  "                    _OWT[0] = W_OWN / (NEURAL_OFF + _np.array([_orm[x] for x in _c], float)); _st['own_used'] = _st.get('own_used', 0) + 1\n"
  "                else:\n                    _OWT[0] = 0.0\n"),
 ("print('047b SANPIER WMID SCAN'", "print('049 ENS', W_OWN, len(OWN_RANK) if 'OWN_RANK' in globals() else -1"),
]
for a, b in rep:
    assert src.count(a) == 1, a; src = src.replace(a, b)
src = src.replace('# 047: 046A with W_MID', '# 049 (A x B ensemble, see notebooks/049/build.py) on 047: 046A with W_MID', 1)
compile(src, '049', 'exec')
n2 = copy.deepcopy(nb); n2['cells'][ix]['source'] = src.splitlines(True); n2['cells'][ix]['outputs'] = []
oc = copy.deepcopy(nb['cells'][ix]); oc['source'] = o.splitlines(True); oc['outputs'] = []; oc.pop('id', None)
n2['cells'].insert(ix, oc)
json.dump(n2, open('enveda-049-ens-ab.ipynb', 'w'), indent=1)
m = json.load(open('../047/kernel-metadata.json'))
m['dataset_sources'] = m['dataset_sources'] + ['ccjjllt/enveda-own-spec']; m['code_file'] = 'enveda-049-ens-ab.ipynb'
json.dump(m, open('kernel-metadata.json', 'w'), indent=1)
subprocess.run(['git', 'checkout', '--', 'enveda-047-sanpier-wmid.ipynb', 'kernel-metadata.json'], cwd='../047', check=True)
print('built 049 W_OWN', W_OWN)

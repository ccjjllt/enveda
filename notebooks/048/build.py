"""Build 048 (B track, license-clean): 044 model3 W0.25 + own fragment-explanation correction (src/fe048) on the neural score."""
import json
nb = json.load(open('../044/enveda-044-own-hn.ipynb'))
ix = [i for i, c in enumerate(nb['cells']) if 'NEURAL_W, NEURAL_OFF' in ''.join(c['source'])]; assert len(ix) == 1; ix = ix[0]
src = ''.join(nb['cells'][ix]['source'])
FF = open('../../src/fe048/frag_feats.py').read(); P = json.load(open('../../src/fe048/frag_lin.json'))
HELP = f'''# 048 own fragment-explanation correction (src/fe048, own implementation): per spectrum
#   s = a*emb/10 + w . z(frag feats); fitted on 044 holdout isomer sets (val2 holdout +0.017 MRR).
import types as _t48, time as _time48
_fe = _t48.ModuleType('_fe048'); exec(compile({FF!r}, '_fe048', 'exec'), _fe.__dict__)
_P48 = {json.dumps(P)}
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
'''
reps = [
 ("NEURAL_HEALTH = 'NEURAL NOT RUN'\n", "NEURAL_HEALTH = 'NEURAL NOT RUN'\n" + HELP),
 ("                _rec = []\n", "                _rec, _raw = [], []\n"),
 ("                        _rec.append((_pk, _nn, _me, _ad, _in))\n",
  "                        _rec.append((_pk, _nn, _me, _ad, _in)); _raw.append((_r.ms2_mzs, _r.ms2_normalized_intensities, float(_r.precursor_mz), _r.adduct, _r.ionization_mode))\n"),
 ("                    _sm = _np.concatenate(_sc).mean(0)\n", "                    _sm = _fr_adjust(_np.concatenate(_sc), _raw, _c).mean(0)\n"),
 ("print('044 HN', len(_nets)", "print('048 FRAGCORR', _FR, flush=True)\nprint('044 HN', len(_nets)"),
]
for a, b in reps:
    assert src.count(a) == 1, a; src = src.replace(a, b)
assert '_fr_adjust' in src and 'OWN_FP_W3 = float(_fw3)' in src
nb['cells'][ix]['source'] = src.splitlines(True)
json.dump(nb, open('enveda-048-fragcorr.ipynb', 'w'), indent=1)
m = json.load(open('../044/kernel-metadata-sub.json')); m['code_file'] = 'enveda-048-fragcorr.ipynb'
json.dump(m, open('kernel-metadata.json', 'w'), indent=1)
open('cell_048.py', 'w').write(src); print('ok', len(src))

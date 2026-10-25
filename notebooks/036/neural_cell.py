'''036 NEURAL RERANK (public idea: sanpier "PubChem + Neural025", LB 0.465 -> 0.466): after EVERY baseline rule,
score the final <=25 candidates with sanpier/casmi-spectrum-model-v1 (spectrum transformer + fingerprint head,
trained on train.parquet) and fuse ranks: 1/(3+baseline_rank) + W/(3+neural_rank). Candidate SET never changes.
Any failure -> submission.csv untouched.'''
NEURAL_W, NEURAL_OFF = 0.25, 3.0
NEURAL_HEALTH = 'NEURAL NOT RUN'
try:
    import glob as _g, json as _j, types as _t, importlib.util as _iu, shutil as _sh
    import numpy as _np, pandas as _pd, torch as _torch
    from pathlib import Path as _P
    _cfgs = sorted(_g.glob('/kaggle/input/**/casmi_spectrum_model_v1/config.json', recursive=True), key=len)
    assert _cfgs, 'casmi_spectrum_model_v1 bundle not mounted'
    _B = _P(_cfgs[0]).parent
    _pkg = '_casmi_own'; _m = _t.ModuleType(_pkg); _m.__path__ = [str(_B)]; sys.modules[_pkg] = _m
    for _n in ['features', 'model']:
        _sp = _iu.spec_from_file_location(_pkg + '.' + _n, _B / (_n + '.py'))
        _mod = _iu.module_from_spec(_sp); sys.modules[_sp.name] = _mod; _sp.loader.exec_module(_mod)
    _feat = sys.modules[_pkg + '.features']
    _cfg = _j.loads((_B / 'config.json').read_text())
    _dev = 'cuda' if _torch.cuda.is_available() else 'cpu'
    _net = sys.modules[_pkg + '.model'].SpectrumCandidateModel(_cfg).to(_dev)
    _ck = _torch.load(_B / 'final_model.pt', map_location=_dev, weights_only=True)
    _net.load_state_dict(_ck['state_dict']); _net.eval()
    _sub_p = '/kaggle/working/submission.csv' if os.path.exists('/kaggle/working/submission.csv') else 'submission.csv'
    _sub = _pd.read_csv(_sub_p, dtype=str, keep_default_na=False)
    _sh.copy2(_sub_p, _sub_p.replace('submission.csv', 'submission_pre_neural.csv'))
    _te = te.copy(); _te['molecule_id'] = _te.molecule_id.astype(str)
    _Q = {m: g for m, g in _te.groupby('molecule_id', sort=False)}
    _cache, _out, _st = {}, [], dict(scored=0, skipped=0, changed=0, changed_top1=0)
    with _torch.inference_mode():
        for _mid, _txt in _sub.itertuples(index=False):
            _c = _txt.split(';')
            try:
                assert _mid in _Q and len(_c) > 1
                _rec = []
                for _r in _Q[_mid].itertuples(index=False):
                    try:
                        _pk, _nn = _feat.peaks(_r.ms2_mzs, _r.ms2_normalized_intensities, _r.precursor_mz, _cfg['max_peaks'])
                        if _nn < _cfg['min_peaks']:
                            continue
                        _me, _ad = _feat.metadata(_r.precursor_mz, _r.adduct, _r.ionization_mode, _r.collision_energy_ev)
                        _rec.append((_pk, _nn, _me, _ad))
                    except (ValueError, TypeError):
                        continue
                assert _rec, 'no usable spectra'
                _fp = []
                for _s in _c:
                    if _s not in _cache:
                        _cache[_s] = _feat.structure(_s, _cfg['fp_bits'])[-1]
                    _fp.append(_cache[_s])
                _F = _torch.tensor(_np.stack(_fp), device=_dev, dtype=_torch.float32); _M = _net.encode_molecules(_F)
                _sc = []
                for _i in range(0, len(_rec), 64):
                    _b = _rec[_i:_i + 64]
                    _Z, _PP = _net.encode_spectra(_torch.tensor(_np.stack([v[0] for v in _b]), device=_dev),
                                                  _torch.tensor([v[1] for v in _b], device=_dev, dtype=_torch.long),
                                                  _torch.tensor(_np.stack([v[2] for v in _b]), device=_dev),
                                                  _torch.tensor([v[3] for v in _b], device=_dev, dtype=_torch.long))
                    _sc.append(((_Z @ _M.T) / _cfg['temperature'] + _cfg['fp_score_weight'] * (_PP @ _F.T / _cfg['fp_bits'])).cpu().numpy())
                _s_mol = _np.concatenate(_sc).mean(0)
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
print('036', NEURAL_HEALTH, flush=True)

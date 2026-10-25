# 038: main-engine top-1 protection (idea: lavinwins "CASMI 26 | LB top 1", reimplemented).
# Runs LAST. If Engine-1 (BASE) top-1 is inside the final 25, move it to rank 1; order otherwise kept.
PROTECT_POLICY = 'protect_main_top1'
try:
    import pandas as _pd, shutil as _sh
    from collections import Counter as _C
    _p = '/kaggle/working/submission.csv'
    _sh.copy(_p, '/kaggle/working/submission_pre_protect.csv')
    _f = _pd.read_csv(_p, dtype=str, keep_default_na=False)
    _done = set(te.molecule_id.astype(str)); _rows = []; _st = _C()
    for _mid, _ans in _f.itertuples(index=False):
        _b = _ans.split(';')
        _ml = (BASE.get(_mid) or ([], [], 0.))[0] if _mid in _done else []
        if not _ml: _rows.append(_ans); _st['no_main'] += 1; continue
        if PROTECT_POLICY == 'protect_agreed_top1':
            _el = (ENG.get(_mid) or {}).get('smiles') or []
            if not _el or metric_identity(_el[0]) != metric_identity(_ml[0]):
                _rows.append(_ans); _st['no_agree'] += 1; continue
        _k = metric_identity(_ml[0]); _ids = [metric_identity(s) for s in _b]
        _pos = _ids.index(_k) if (_k is not None and _k in _ids) else None
        if _pos is None: _st['not_in25'] += 1; _rows.append(_ans); continue
        if _pos == 0: _st['already'] += 1; _rows.append(_ans); continue
        _rows.append(';'.join([_b[_pos]] + _b[:_pos] + _b[_pos+1:])); _st['promoted'] += 1
    _f['smiles'] = _rows; _f.to_csv(_p, index=False)
    print('038 PROTECT OK', PROTECT_POLICY, dict(_st))
except Exception as _e:
    import traceback; traceback.print_exc(); print('038 PROTECT FAILED (submission untouched)', _e)

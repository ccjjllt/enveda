# ---- v12: strict cross-library direct-reference promotion (021 local B +0.0140, 7/0) ----
# Promote candidate j (within top 25) to rank 1 iff: q_j >= 0.8, q_j - q_top1 >= 0.2, q_top1 < 0.8,
# where q = best greedy cosine (same polarity, >= 3 matched peaks, 0.02 Da) between the molecule's query spectra and
# train reference spectra of the candidate's InChIKey14, excluding 'enveda-np-examples' (query-like source).
# Any failure keeps the v11 submission unchanged.
import json as _vj, numpy as _vn, pandas as _vp, time as _vt
_V12 = dict(status='start', promoted=0, considered=0)
_vt0 = _vt.time()
try:
    import pyarrow.dataset as _vds
    from rdkit import Chem as _vC, RDLogger as _vRL
    _vRL.DisableLog('rdApp.*')
    def _v12_clean(mz, it, prec):
        mz = _vn.asarray(mz, _vn.float64); it = _vn.asarray(it, _vn.float64)
        if not len(mz): return mz, it
        k = (mz < float(prec) - 0.5) & (it >= 0.01 * it.max())
        mz, it = mz[k], _vn.sqrt(it[k]); o = _vn.argsort(-it)[:60]
        return mz[o], it[o] / (_vn.linalg.norm(it[o]) + 1e-12)
    def _v12_cos(a, b, tol=0.02):
        (m1, i1), (m2, i2) = a, b
        if not len(m1) or not len(m2): return 0.0, 0
        pr = _vn.argwhere(_vn.abs(m1[:, None] - m2[None, :]) <= tol)
        if not len(pr): return 0.0, 0
        sc = i1[pr[:, 0]] * i2[pr[:, 1]]; u1, u2, tot, n = set(), set(), 0.0, 0
        for j in _vn.argsort(-sc):
            x, y = pr[j]
            if x in u1 or y in u2: continue
            u1.add(x); u2.add(y); tot += sc[j]; n += 1
        return float(tot), n
    def _v12_ik(s):
        try:
            m = _vC.MolFromSmiles(s); return _vC.MolToInchiKey(m)[:14] if m is not None else None
        except Exception: return None
    _vsub = _vp.read_csv('submission.csv')
    _vlists = {str(m): [x for x in str(s).split(';') if x][:25] for m, s in zip(_vsub.molecule_id, _vsub.smiles)}
    _vik = {s: _v12_ik(s) for l in _vlists.values() for s in l}
    _vwant = sorted({k for k in _vik.values() if k})
    _vtr = _vds.dataset(os.path.join(COMP, 'train.parquet')).to_table(
        filter=_vds.field('inchikey14').isin(_vwant),
        columns=['inchikey14', 'ionization_mode', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities', 'ingest_lib']).to_pandas()
    _vtr = _vtr[_vtr.ingest_lib != 'enveda-np-examples']
    _vrefs = {}
    for r in _vtr.itertuples(index=False):
        _vrefs.setdefault(r.inchikey14, []).append((r.ionization_mode, _v12_clean(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz)))
    del _vtr
    _vte = _vp.read_parquet(os.path.join(COMP, 'test.parquet'))
    _vq = {str(m): [(r.ionization_mode, _v12_clean(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz)) for r in g.itertuples(index=False)]
           for m, g in _vte.groupby('molecule_id')}
    _vrows, _vlog = [], []
    for m, l in _vlists.items():
        q = []
        for s in l:
            best = 0.0
            for pol, rs in _vrefs.get(_vik.get(s), [])[:40]:
                for qpol, qsp in _vq.get(m, []):
                    if qpol != pol: continue
                    c, n = _v12_cos(qsp, rs)
                    if n >= 3 and c > best: best = c
            q.append(best)
        l2 = list(l)
        if q:
            _V12['considered'] += 1
            j = int(_vn.argmax(q))
            if j >= 1 and q[j] >= 0.8 and q[0] < 0.8 and q[j] - q[0] >= 0.2:
                l2 = [l[j]] + [x for k, x in enumerate(l) if k != j]
                _V12['promoted'] += 1; _vlog.append(dict(molecule_id=m, from_rank=j + 1, q=round(q[j], 4), q_top1=round(q[0], 4)))
        _vrows.append((m, ';'.join(l2[:25]) if l2 else 'CCO'))
    _vout = _vp.DataFrame(_vrows, columns=['molecule_id', 'smiles'])
    assert list(_vout.molecule_id.astype(str)) == list(_vsub.molecule_id.astype(str)) and _vout.smiles.notna().all()
    _vsub.to_csv('submission_v11.csv', index=False)
    _vout.to_csv('submission.csv', index=False)
    _vp.DataFrame(_vlog).to_csv('v12_promotions.csv', index=False)
    _V12['status'] = 'ok'
except Exception as _ve:
    _V12['status'] = f'FAILED (v11 submission kept): {_ve!r}'
_V12['seconds'] = round(_vt.time() - _vt0, 1)
open('v12_direct_ref.json', 'w').write(_vj.dumps(_V12))
print('V12 DIRECT-REF', _vj.dumps(_V12), flush=True)

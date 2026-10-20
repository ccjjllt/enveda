"""023: Engine1 (v4n casmi) class-3 generation study on NP250 masks.

python src/study_e1_gen.py --mask C --variant base [--variant big ...]
Executes E006 notebook cells 0..15 (Engine2/PubChem reused from 021 runs), then for each variant re-runs V.run per
molecule with modified EngineCfg generation params, ranks by the v4n ranker (no pool-popularity / fusion tail),
and checkpoints per-molecule JSONL: n_cand, n_gen, truth in pool window?, truth generated?, truth rank (ranker),
top25 gen count. Truth identity = official metric key.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_e006_local as E21

VARIANTS = {
    'base': {},
    'big': dict(gen_n_analog=12, gen_max_per_parent=120, gen_max_total=400),
    'loose': dict(gen_n_analog=12, gen_min_sim=0.2, gen_max_per_parent=120, gen_max_total=400),
    'deep': dict(gen_n_analog=6, gen_max_per_parent=200, gen_max_total=400),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mask', default='C'); ap.add_argument('--variant', action='append', default=[])
    ap.add_argument('--out', default='/workspace/e021'); ap.add_argument('--study', default='/workspace/e023')
    ap.add_argument('--notebook', default='/workspace/k020_v11_backup.ipynb')
    a = ap.parse_args()
    out, st = Path(a.out), Path(a.study); st.mkdir(parents=True, exist_ok=True)
    work = st / f'work_{a.mask}'; work.mkdir(exist_ok=True)
    os.environ['E021_MASK'] = a.mask; os.environ['E021_DROP'] = str(out / 'drop.json')
    if a.mask == 'C': os.environ['E021_DROP1'] = str(out / 'drop1_e1pool.npy')
    src = out / 'runs' / f'{a.mask}_base'
    os.environ['E021_REUSE_ENG'] = str(src / 'eng_lists.json'); os.environ['E021_REUSE_PC'] = str(src / 'pc_lists.json')
    os.environ['PYTHONPATH'] = str(HERE) + os.pathsep + os.environ.get('PYTHONPATH', '')
    cells = E21.build_cells(a.notebook, out / f'in_{a.mask}', work, {})
    os.chdir(work)
    ns = {'__name__': '__main__', 'display': print, '_e21': E21, 'E021_ENG_EXTRA': {}, 'E021_PROMO_REL': 150.0, 'E021_WINDOW': 40}
    for i, c in cells[:16]:
        print(f'===== cell {i}', flush=True); exec(compile(c, f'<cell {i}>', 'exec'), ns)
    M = E21._metric()
    sol = dict(pd.read_csv(out / 'solution.csv').values)
    acc = {m: {M.canonicalize_smiles_to_ik14(s) for s in v.split(',')} - {None} for m, v in sol.items()}
    V, Eng, chem, rank_score, FEATURES, P = ns['V'], ns['E'], ns['chem'], ns['rank_score'], ns['FEATURES'], ns['P']
    te = pd.read_parquet(os.path.join(ns['COMP'], 'test.parquet'))
    te['nm'] = chem.neutral_mass(te.precursor_mz.values.astype(np.float64), te.adduct.values)
    base_cfg = dict(Eng.cfg.__dict__)
    for vn in a.variant or ['base']:
        f = st / f'{a.mask}_{vn}.jsonl'
        done = {json.loads(l)['mid'] for l in open(f)} if f.exists() else set()
        for k, v in base_cfg.items(): setattr(Eng.cfg, k, v)
        for k, v in VARIANTS[vn].items(): setattr(Eng.cfg, k, v)
        t0 = time.time()
        with open(f, 'a') as fh:
            for gi, (mid, sub) in enumerate(te.groupby('molecule_id', sort=False)):
                if mid in done: continue
                nms = sub.nm.values[np.isfinite(sub.nm.values)]
                rec = dict(mid=mid, n_cand=0, n_gen=0, rank=None, gen_truth=False, truth_any=False, top25_gen=0)
                if len(nms):
                    target = float(np.median(nms)); spectra = []
                    for r in sub.itertuples():
                        ce = r.collision_energy_ev
                        ca = np.atleast_1d(np.asarray(ce, dtype=float)) if ce is not None else np.zeros(0)
                        spectra.append(dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                                            mode=1 if r.ionization_mode == 'positive' else -1, adduct=r.adduct, prec=float(r.precursor_mz),
                                            ce=float(ca.mean()) if len(ca) else np.nan, ce_n=int(len(ca)) if len(ca) else 1))
                    try:
                        C, X, F, names, info = V.run(spectra, target)
                        rec['lib_max'] = float(info.get('lib_max', 0.0))
                        if len(C.pid):
                            sc = rank_score(np.concatenate([X, F], 1), list(FEATURES) + list(names))
                            order = np.argsort(-sc, kind='stable')
                            isg = np.asarray(C.is_gen) > 0
                            rec['n_cand'] = int(len(C.pid)); rec['n_gen'] = int(isg.sum())
                            seen, r_ = set(), 0
                            for i in order:
                                k = C.key[i]
                                if k in seen: continue
                                seen.add(k); r_ += 1
                                if r_ <= 25 and isg[i]: rec['top25_gen'] += 1
                                if M.canonicalize_smiles_to_ik14(C.smiles[i]) in acc[mid] and rec['rank'] is None:
                                    rec['rank'] = r_; rec['gen_truth'] = bool(isg[i])
                            rec['truth_any'] = rec['rank'] is not None
                    except Exception as e:
                        rec['err'] = repr(e)[:200]
                fh.write(json.dumps(rec) + '\n'); fh.flush()
                if gi % 25 == 0: print(vn, gi, f'{time.time()-t0:.0f}s', flush=True)
        print('VARIANT DONE', vn, flush=True)

if __name__ == '__main__':
    main()

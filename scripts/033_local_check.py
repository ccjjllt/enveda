import os, sys, json, time
sys.path.insert(0, '/workspace/enveda/src')
os.environ['KAGGLE_IS_COMPETITION_RERUN'] = '1'; os.environ['E021_MASK'] = 'A'
import eval_e006_local as H
from pathlib import Path
work = Path('/workspace/e033/run'); work.mkdir(parents=True, exist_ok=True)
cells = H.build_cells('/workspace/enveda/notebooks/033/enveda-033-e1gen.ipynb', Path('/workspace/kin'), work, {})
os.chdir(work)
ns = {'__name__': '__main__', 'display': print, '_e21': H}
N = int(os.environ.get('N033', '8')); PATCH = os.environ.get('P033', '1') == '1'
for i, c in cells:
    if "'''Our engine (two-ranker" in c[:40] or c.startswith('V17_PATCH') or (c.startswith("'''033") and not PATCH): continue
    if c.startswith("'''all molecules -> base lists"):
        c = c.replace("mols = list(te.groupby('molecule_id', sort=False))",
                      "mols = list(te.groupby('molecule_id', sort=False))[:%d]\nTOPN = globals().get('TOPN', 60)" % N)
    t = time.time(); print(f'--- cell {i}', flush=True)
    exec(compile(c, f'cell{i}', 'exec'), ns)
    print(f'--- cell {i} done {time.time()-t:.0f}s', flush=True)
    if c.startswith("'''all molecules -> base lists"): break
B = ns['BASE']; out = {m: [v[0][:5], (list(v[4][:5]) if len(v) > 4 else None)] for m, v in B.items()}
print('n_err', ns['n_err'], 'E1GEN', ns.get('E1GEN'))
json.dump(out, open(f'/workspace/e033/base_{"p" if PATCH else "o"}.json', 'w'), default=str)

"""021: E006 本地 CPU 验证器（NP250 timsTOF 面板，三种遮蔽）。

用法（box 上，venv=/workspace/.venv-e006，RDKit 2026.03.3）：
  python src/eval_e006_local.py panel  --kin /workspace/kin --out /workspace/e021
  python src/eval_e006_local.py run    --out /workspace/e021 --mask B --name base [--set LIB_TAU=0.85 ...]
  python src/eval_e006_local.py score  --out /workspace/e021 --name B_base
  python src/eval_e006_local.py compare --out /workspace/e021 B_base B_x

面板：train 中 ingest_lib == 'enveda-np-examples' 的 1184 张谱（250 个结构，按 inchikey14 成组）。
遮蔽：
  A 全可见：train 原样（查询谱本身在谱库中，近似上界）。
  B 本谱隐藏：只删除 1184 张 enveda-np-examples 查询谱本身；其他来源库的同结构参考谱保留（跨源参考场景）。
  R 参考全删：删除与真值同一官方指标身份（去立体 + 互变异构规范 InChIKey14）的所有谱（任意来源库），
    结构仍在候选池中（库内结构、无参考谱）。
  C 结构移除：R + 从 Engine1 池窗口、Engine2 池、PubChem 通道的“池内”排除集中移除真值身份；
    PubChem 列表中真值也被排除（模拟既不在 PubChem 也不在 COCONUT 的 class 3），生成路径仍允许命中。
执行：按顺序 exec E006 notebook 的全部代码单元，路径 /kaggle/input → 遮蔽输入根，/kaggle/working → 运行目录；
跳过 GPU 预检；ICEBERG/GLACIER 默认跳过（CPU 不可行），因此结果是 “E006 去模拟器” 的相对比较。
评分：src/casmi_metric.py（官方 notebook 原样代码），逐分子倒数排名。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SLUG = 'enveda-CASMI26-molecule-id-mass-spectra'
HERE = Path(__file__).resolve().parent
TEST_COLS = ['molecule_id', 'spectrum_id', 'ms2_mzs', 'ms2_normalized_intensities', 'base_peak_intensity', 'adduct',
             'ionization_mode', 'instrument_type', 'precursor_mz', 'collision_energy_orig', 'collision_energy_ev',
             'collision_energy_orig_units']


# ----------------------------------------------------------------------------------------- metric helpers
def _metric():
    sys.path.insert(0, str(HERE))
    import casmi_metric as M
    M.install_dependencies()
    return M


def _keys_of(smiles_list):
    M = _metric()
    return [M.canonicalize_smiles_to_ik14(s) for s in smiles_list]


def _comp_worker(s):
    M = _metric()
    return M.heavy_atom_composition(s)


def _key_worker(s):
    M = _metric()
    return M.canonicalize_smiles_to_ik14(s)


def _pmap(fn, xs, n=6):
    from multiprocessing import Pool
    with Pool(n) as p:
        return p.map(fn, xs, chunksize=256)


# ----------------------------------------------------------------------------------------- mask C drop (also imported by subprocesses)
_DROP = None


def _drop_spec():
    global _DROP
    if _DROP is None:
        d = json.load(open(os.environ['E021_DROP']))
        _DROP = dict(keys=set(d['keys']), ik14=set(d['ik14']), mass=np.sort(np.asarray(d['mass'], np.float64)))
    return _DROP


def drop_mask(mass, smiles, keys=None, ppm=20.0):
    """Boolean mask of pool entries whose official metric identity equals a panel truth.
    Candidates are first restricted to +-ppm of a truth neutral mass (or +-1.00728 for charged forms)."""
    D = _drop_spec()
    mass = np.asarray(mass, np.float64)
    out = np.zeros(len(mass), bool)
    if keys is not None:
        out |= pd.Series(np.asarray(keys, dtype=object)).isin(D['ik14']).values
    M = _metric()
    idx = set()
    order = np.argsort(mass, kind='stable') if not np.all(np.diff(mass[np.isfinite(mass)]) >= 0) else None
    ms = mass if order is None else mass[order]
    for t in D['mass']:
        for c in (t, t - 1.007276, t + 1.007276):
            tol = c * ppm * 1e-6
            a, b = np.searchsorted(ms, c - tol, 'left'), np.searchsorted(ms, c + tol, 'right')
            idx.update(range(a, b) if order is None else order[a:b].tolist())
    for i in idx:
        if not out[i]:
            k = M.canonicalize_smiles_to_ik14(str(smiles[i]))
            out[i] = k in D['keys']
    return out


# ----------------------------------------------------------------------------------------- panel + masks
def cmd_panel(a):
    kin, out = Path(a.kin), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    comp = kin / 'competitions' / SLUG
    import pyarrow.parquet as pq
    import pyarrow.dataset as pds
    tr = pd.read_parquet(comp / 'train.parquet', columns=['ingest_lib', 'inchikey14', 'normalized_smiles'])
    npx = pds.dataset(comp / 'train.parquet').to_table(filter=pds.field('ingest_lib') == 'enveda-np-examples').to_pandas()
    npx['molecule_id'] = 'np_' + npx.inchikey14.astype(str)
    npx = npx.sort_values(['molecule_id'], kind='stable').reset_index(drop=True)
    npx['spectrum_id'] = [f'nps_{i:05d}' for i in range(len(npx))]
    te = npx[TEST_COLS].reset_index(drop=True)
    sol = npx.groupby('molecule_id').normalized_smiles.agg(lambda s: ','.join(dict.fromkeys(s))).reset_index()
    sol.columns = ['molecule_id', 'smiles']
    M = _metric()
    truth = sorted(set(npx.normalized_smiles))
    tkeys = {s: M.canonicalize_smiles_to_ik14(s) for s in truth}
    tcomp = {s: M.heavy_atom_composition(s) for s in truth}
    from rdkit import Chem
    from rdkit.Chem.Descriptors import ExactMolWt
    tmass = [ExactMolWt(Chem.MolFromSmiles(s)) for s in truth]
    keyset, compset = set(tkeys.values()) - {None}, set(tcomp.values()) - {None}
    # mask B: every train spectrum (any library) whose structure has a truth metric identity
    us = pd.Series(tr.normalized_smiles.unique())
    comps = _pmap(_comp_worker, us.tolist())
    cand = us[[c in compset for c in comps]].tolist()
    ks = _pmap(_key_worker, cand)
    hit = {s for s, k in zip(cand, ks) if k in keyset}
    ik14 = set(npx.inchikey14)
    rm = tr.normalized_smiles.isin(hit) | tr.inchikey14.isin(ik14)
    rm_by_lib = tr[rm].ingest_lib.value_counts().to_dict()
    keep = ~rm.values
    n_train_B = int(keep.sum())
    del tr
    for m in 'ABRC':
        d = out / f'in_{m}' / 'competitions' / SLUG
        d.mkdir(parents=True, exist_ok=True)
        te.to_parquet(d / 'test.parquet', index=False)
        pd.DataFrame({'molecule_id': sorted(te.molecule_id.unique()), 'smiles': 'CCO'}).to_csv(d / 'sample_submission.csv', index=False)
        ds = out / f'in_{m}' / 'datasets'
        if not ds.exists():
            ds.symlink_to(kin / 'datasets')
    (out / 'in_A' / 'competitions' / SLUG / 'train.parquet').unlink(missing_ok=True)
    (out / 'in_A' / 'competitions' / SLUG / 'train.parquet').symlink_to(comp / 'train.parquet')
    import pyarrow as pa
    own = (lib_all := pd.read_parquet(comp / 'train.parquet', columns=['ingest_lib']).ingest_lib.values) != 'enveda-np-examples'
    del lib_all
    for m, kmask in (('B', own), ('R', keep)):
        pf = pq.ParquetFile(comp / 'train.parquet')
        w, pos = None, 0
        for b in pf.iter_batches(batch_size=100000):
            mm = kmask[pos:pos + b.num_rows]; pos += b.num_rows
            t = pa.Table.from_batches([b]).filter(pa.array(mm))
            if w is None:
                w = pq.ParquetWriter(out / f'in_{m}' / 'competitions' / SLUG / 'train.parquet', t.schema)
            w.write_table(t)
        w.close()
        assert pos == len(kmask)
    (out / 'in_C' / 'competitions' / SLUG / 'train.parquet').unlink(missing_ok=True)
    (out / 'in_C' / 'competitions' / SLUG / 'train.parquet').symlink_to(out / 'in_R' / 'competitions' / SLUG / 'train.parquet')
    sol.to_csv(out / 'solution.csv', index=False)
    json.dump(dict(keys=sorted(keyset), ik14=sorted(ik14), mass=tmass), open(out / 'drop.json', 'w'))
    info = dict(n_spectra=len(te), n_molecules=int(te.molecule_id.nunique()), n_train_removed=int(rm.sum()),
                removed_by_lib=rm_by_lib, n_train_B=n_train_B, n_truth_keys=len(keyset))
    json.dump(info, open(out / 'panel_info.json', 'w'), indent=1)
    print(json.dumps(info, indent=1))


# ----------------------------------------------------------------------------------------- notebook execution
def _cells(nb):
    nb = json.load(open(nb))
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']


def _rep(src, old, new, count=1, where=''):
    n = src.count(old)
    if n < 1:
        raise RuntimeError(f'patch anchor not found ({where}): {old[:80]!r}')
    return src.replace(old, new) if count == 0 else src.replace(old, new, count)


PC_POOLKEYS_OLD = "    _W['pool_keys'] = set(pd.read_parquet(pool_meta_path, columns=['key']).key.values.tolist())\n"
PC_POOLKEYS_NEW = PC_POOLKEYS_OLD + (
    "    if os.environ.get('E021_DROP') and os.environ.get('E021_MASK') == 'C':\n"
    "        import json as _j\n"
    "        _W['pool_keys'] |= set(_j.load(open(os.environ['E021_DROP']))['keys'])\n")
RUN_BEST_OLD = "            F = np.unpackbits(np.asarray(pfp[a:b]), axis=1)[:, :nbits].astype(np.float64)\n"
RUN_BEST_NEW = RUN_BEST_OLD + (
    "            if os.environ.get('E021_MASK') == 'C':\n"
    "                _dm = np.load(os.environ['E021_DROP1'])[a:b]\n"
    "                F = F[~_dm] if (~_dm).any() else F[:0]\n")
RUN_BEST_OLD2 = "            best = float((F @ z.astype(np.float64)).max())\n"
RUN_BEST_NEW2 = "            best = float((F @ z.astype(np.float64)).max()) if len(F) else float('nan')\n"
ENG_POOL_OLD = '    log(f"pool {len(o):,} structures")\n    return P\n'
ENG_POOL_NEW = ('    if os.environ.get("E021_MASK") == "C":\n'
                '        import eval_e006_local as _e21\n'
                '        _keep = ~_e21.drop_mask(P["mass"], P["smiles"], P["keys"])\n'
                '        log(f"E021 mask C: dropped {int((~_keep).sum())} pool structures")\n'
                '        for _k in ("fp", "mass", "keys", "smiles"):\n'
                '            P[_k] = P[_k][_keep]\n'
                '        P["k2i"] = pd.Series(np.arange(len(P["keys"])), index=P["keys"]); P["k2i"] = P["k2i"][~P["k2i"].index.duplicated()]\n'
                ) + ENG_POOL_OLD


def _patch_strings(ns):
    """Called inside the notebook namespace before sub-process sources are written."""
    if 'ENG_FILES' in ns and not ns.get('_E021_ENG'):
        ef = dict(ns['ENG_FILES'])
        ef['casmi_engine.py'] = _rep(ef['casmi_engine.py'], ENG_POOL_OLD, ENG_POOL_NEW, where='eng pool')
        # local layout: find('fp_bits.npy') resolves to the 10226-bit v2-pool file here; Engine2 needs the 6930-bit
        # COCONUT selection (Kaggle log shows COCONUT/train fingerprinting succeeded, i.e. 6930 bits were used there)
        ef['casmi_engine.py'] = _rep(ef['casmi_engine.py'], '_g["bits"] = np.load(find("fp_bits.npy"))',
                                     '_g["bits"] = np.load(os.path.join(os.path.dirname(find("coco_fp.npy")), "fp_bits.npy"))', where='eng bits')
        for k, v in list(ns.get('E021_ENG_EXTRA', {}).items()):
            ef[k] = v(ef[k])
        ns['ENG_FILES'] = ef; ns['_E021_ENG'] = True
    if 'CORE' in ns and not ns.get('_E021_PC'):
        ns['CORE'] = _rep(ns['CORE'], PC_POOLKEYS_OLD, PC_POOLKEYS_NEW, where='pc core')
        r = _rep(ns['RUNNER'], RUN_BEST_OLD, RUN_BEST_NEW, where='pc runner')
        ns['RUNNER'] = _rep(r, RUN_BEST_OLD2, RUN_BEST_NEW2, where='pc runner2')
        ns['_E021_PC'] = True


def _patch_pool(ns):
    """Engine 1 (v4n) pool window filter for mask C."""
    if os.environ.get('E021_MASK') not in ('C', 'H'):
        return
    P = ns['P']
    drop = np.load(os.environ['E021_DROP1'])
    assert len(drop) == len(P.key)
    orig = P.window
    P.window = lambda target, ppm, _o=orig, _d=drop: (lambda r: r[~_d[r]])(_o(target, ppm))
    print('E021 mask C: Engine1 pool window drops', int(drop.sum()), 'structures', flush=True)


def build_cells(nb, inroot, work, sets, skip_sim=True):
    cells = _cells(nb)
    out = []
    for i, c in enumerate(cells):
        if os.environ.get('E021_SKIP_LATE') and i >= 27:
            out.append((i, 'print("E021 skip late cell", %d)' % i)); continue
        c = c.replace('/kaggle/input', str(inroot)).replace('/kaggle/working', str(work))
        if 'E004_PREFLIGHT = _e004_preflight(globals())' in c:
            c = _rep(c, "gpu = subprocess.check_output(\n        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()\n    assert 'T4' in gpu, gpu",
                     "gpu = 'CPU-local (E021)'", where='preflight')
            c = _rep(c, "assert comp in {", "assert True or comp in {", where='preflight comp')
        if "r.check_returncode()" in c and 'rdkit wheel:' in c:
            c = _rep(c, "r.check_returncode()", "pass  # E021: rdkit preinstalled", where='pip')
            c = _rep(c, "print('Simulator RDKit wheel:', simulator_rdkit_wheel(_simulator_pkg))",
                     "print('E021: simulator wheel check skipped')", where='simwheel')
        if "for _n, _src in ENG_FILES.items():" in c:
            c = _rep(c, "for _n, _src in ENG_FILES.items():", "_e21._patch_strings(globals())\nfor _n, _src in ENG_FILES.items():")
            if os.environ.get('E021_REUSE_ENG'):
                c = "ENG = json.load(open(os.environ['E021_REUSE_ENG'])); print('E021 reuse eng', len(ENG), flush=True)\n"
        if "open(" in c and "pc_runner.py', 'w').write(RUNNER)" in c and os.environ.get('E021_REUSE_PC'):
            c = ("open('/tmp/_e021_pc_skip.py','w').write('')\n"  # keep cell index
                 "PC = json.load(open(os.environ['E021_REUSE_PC'])); print('E021 reuse pc', len(PC), flush=True)\n")
        if "open(" in c and "probe_core2.py', 'w').write(CORE + V17_PATCH)" in c:
            c = re.sub(r"(open\('[^']*probe_core2\.py', 'w'\)\.write\(CORE \+ V17_PATCH\))", r"_e21._patch_strings(globals())\n\1", c, count=1)
        _a24 = "_FULL = glob.glob('%s/**/casmi26-fpnet-full1/**/fpnet_full1.pt', recursive=True)" % inroot
        if os.environ.get('E024_FP0') and _a24 in c:
            c = _rep(c, _a24, "_FULL = []  # E024: NP-held-out fpnet_0")
        if "L = Library('work'); P = Pool('work')" in c:
            c = _rep(c, "L = Library('work'); P = Pool('work')", "L = Library('work'); P = Pool('work'); _e21._patch_pool(globals())")
        if "BASE_TAIL_CANDIDATES = {str(k): list(v[0]) for k, v in BASE.items()}" in c:
            c += ("\nimport json as _j24\n_j24.dump({m: [v[0][0] if v[0] else None, bool(len(v) > 4 and v[4] and str(v[4][0]).endswith('|gen')), "
                  "float(v[2])] for m, v in BASE.items()}, open('e024_e1top1.json', 'w'))\n")
        if "(rel > 150.0)" in c:
            c = _rep(c, "(rel > 150.0)", "(rel > E021_PROMO_REL)")
        if "'experiment': 'E006', 'window': 40," in c:
            c = _rep(c, "'experiment': 'E006', 'window': 40,", "'experiment': 'E006', 'window': E021_WINDOW,")
        if skip_sim and "ICE_PKG = os.path.dirname(find('casmi26-iceberg/**/ice_runner.py'))" in c:
            c = _rep(c, "    ICE_PKG = os.path.dirname(find('casmi26-iceberg/**/ice_runner.py'))",
                     "    ICE_PKG = os.path.dirname(find('casmi26-iceberg/**/ice_runner.py'))\n    sys.path.append(ICE_PKG)\n    import fuse as ice_fuse\n    del V, E, bank; gc.collect()\n    raise RuntimeError('E021: ICEBERG skipped on CPU')")
            c = _rep(c, "    GL_PKG = os.path.dirname(find('casmi26-glacier/**/gl_runner.py'))",
                     "    GL_PKG = os.path.dirname(find('casmi26-glacier/**/gl_runner.py'))\n    sys.path.append(GL_PKG)\n    import gl_fuse\n    raise RuntimeError('E021: GLACIER skipped on CPU')")
        if "assert _stage_meta['ice'].get('status') == 'ok'" in c and skip_sim:
            c = _rep(c, "assert _stage_meta['ice'].get('status') == 'ok', _stage_meta['ice']", "pass  # E021 sim skipped")
            c = _rep(c, "    assert _stage_meta['glacier'].get('status') == 'ok', _stage_meta['glacier']", "    pass  # E021 sim skipped")
        if sets:
            lines = []
            for ln in c.split('\n'):
                lines.append(ln)
                m = re.match(r'^([A-Za-z_][\w, ]*?)\s*=(?!=)', ln)
                if m:
                    lhs = {x.strip() for x in m.group(1).split(',')}
                    for k, v in sets.items():
                        if k in lhs:
                            lines.append(f'{k} = {v}  # E021 override')
            c = '\n'.join(lines)
        out.append((i, c))
    return out


def cmd_run(a):
    out = Path(a.out)
    name = f'{a.mask}_{a.name}'
    work = out / 'runs' / name
    work.mkdir(parents=True, exist_ok=True)
    inroot = out / f'in_{a.mask}'
    if a.inroot:
        inroot = Path(a.inroot)
    os.environ['E021_MASK'] = a.mask
    os.environ['E021_DROP'] = str(out / (f'drop_{a.mask}.json' if a.mask in 'HPQ' else 'drop.json'))
    os.environ['PYTHONPATH'] = str(HERE) + os.pathsep + os.environ.get('PYTHONPATH', '')
    sys.path.insert(0, str(HERE))
    if a.mask == 'H':
        os.environ['E021_DROP1'] = str(out / 'drop1_H.npy')
    if a.mask == 'C':
        d1 = out / 'drop1_e1pool.npy'
        if not d1.exists():
            meta = pd.read_parquet(next(Path(inroot / 'datasets').glob('*/casmi26-v2-pool/pool_meta.parquet')))
            np.save(d1, drop_mask(meta.mass.values, meta.smiles.values, meta.key.values))
        os.environ['E021_DROP1'] = str(d1)
    if a.reuse_eng: os.environ['E021_REUSE_ENG']=str(Path(a.reuse_eng).resolve())
    if a.reuse_pc: os.environ['E021_REUSE_PC']=str(Path(a.reuse_pc).resolve())
    sets = dict(s.split('=', 1) for s in a.set)
    cells = build_cells(a.notebook, inroot, work, sets)
    sweep = json.load(open(a.sweep)) if a.sweep else {}
    tail0 = next(j for j, (i, c) in enumerate(cells) if 'def merge(base, base_keys' in c)
    os.chdir(work)
    import eval_e006_local as _self
    ns = {'__name__': '__main__', 'display': print, '_e21': _self, 'E021_ENG_EXTRA': {}, 'E021_PROMO_REL': 150.0, 'E021_WINDOW': 40}
    if a.hook:
        exec(open(a.hook).read(), ns)
    t0 = time.time()
    log = []
    for i, c in cells[:tail0]:
        t = time.time()
        print(f'\n===== E021 cell {i} =====', flush=True)
        exec(compile(c, f'<e006 code cell {i}>', 'exec'), ns)
        log.append((i, round(time.time() - t, 1)))
    import copy
    snap = {k: copy.deepcopy(ns[k]) for k in ('BASE', 'PC', 'ENG') if k in ns}
    configs = {'base': {}}
    configs.update(sweep)
    for cname, cset in configs.items():
        wd = work if cname == 'base' else work / 'sweep' / cname
        wd.mkdir(parents=True, exist_ok=True)
        os.chdir(wd)
        ns2 = dict(ns)
        ns2.update({k: copy.deepcopy(v) for k, v in snap.items()})
        allset = dict(sets, **cset)
        for k, v in allset.items():
            exec(f'{k} = {v}', ns2)
        tcells = build_cells(a.notebook, inroot, work, allset)[tail0:] if cset else cells[tail0:]
        t = time.time()
        for i, c in tcells:
            print(f'\n===== E021 [{cname}] cell {i} =====', flush=True)
            exec(compile(c, f'<e006 code cell {i}>', 'exec'), ns2)
        log.append((f'tail:{cname}', round(time.time() - t, 1)))
        json.dump(dict(mask=a.mask, sets=allset, hook=a.hook, cell_seconds=log, total_s=round(time.time() - t0, 1)),
                  open(wd / 'e021_run.json', 'w'), indent=1)
        sc = score_dir(out, wd)
        print('E021_SCORE', name, cname, json.dumps(sc['summary']), flush=True)


def score_dir(out, work, sub_name='submission.csv'):
    M = _metric()
    _m = next((x.split('_')[0] for x in Path(work).parts[::-1] if len(x) > 2 and x[1] == '_' and x[0].isupper()), '')
    _sf = out / f'solution_{_m}.csv'
    sol = pd.read_csv(_sf if _sf.exists() else out / 'solution.csv')
    res = {}
    stages = [('final', sub_name), ('e006_pre_rank2', 'submission_e006.csv'), ('e004', 'submission_e004_control.csv'),
              ('reference', 'submission_reference.csv'), ('baseline', 'baseline_submission.csv')]
    for tag, f in stages:
        p = Path(work) / f
        if not p.exists():
            continue
        sub = pd.read_csv(p)
        g = dict(zip(sub.molecule_id, sub.smiles))
        rr = {}
        for mid, ans in zip(sol.molecule_id, sol.smiles):
            acc = {M.canonicalize_smiles_to_ik14(s) for s in ans.split(',')} - {None}
            comp = {M.heavy_atom_composition(s) for s in ans.split(',')} - {None}
            v = 0.0
            for r, s in enumerate([x.strip() for x in str(g.get(mid, '')).split(';') if x.strip()][:25], 1):
                if M.heavy_atom_composition(s) in comp and M.canonicalize_smiles_to_ik14(s) in acc:
                    v = 1.0 / r
                    break
            rr[mid] = v
        res[tag] = rr
    official = M.score(sol.assign(Usage='Public'), pd.read_csv(Path(work) / sub_name), 'molecule_id')
    summ = {t: dict(mrr=round(float(np.mean(list(r.values()))), 4), top1=sum(v == 1.0 for v in r.values()),
                    top25=sum(v > 0 for v in r.values())) for t, r in res.items()}
    summ['official_final'] = round(official, 4)
    assert abs(official - summ['final']['mrr']) < 1e-4
    json.dump(dict(summary=summ, rr=res), open(Path(work) / 'e021_score.json', 'w'), indent=1)
    return dict(summary=summ, rr=res)


def cmd_score(a):
    sc = score_dir(Path(a.out), Path(a.out) / 'runs' / a.name)
    print(json.dumps(sc['summary'], indent=1))


def cmd_compare(a):
    out = Path(a.out)
    b = json.load(open(out / 'runs' / a.base / 'e021_score.json'))['rr']['final']
    x = json.load(open(out / 'runs' / a.other / 'e021_score.json'))['rr']['final']
    d = np.array([x[m] - b[m] for m in b])
    rng = np.random.default_rng(0)
    boots = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    print(json.dumps(dict(base=a.base, other=a.other, base_mrr=round(np.mean(list(b.values())), 4),
                          other_mrr=round(np.mean(list(x.values())), 4), delta=round(d.mean(), 4),
                          wins=int((d > 0).sum()), losses=int((d < 0).sum()),
                          ci95=[round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)])))



# ----------------------------------------------------------------------------------------- (b) direct-reference quality post-hoc
def _clean(mz, it, prec):
    mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
    k = (mz < prec - 0.5) & (it >= 0.01 * (it.max() if len(it) else 1))
    mz, it = mz[k], np.sqrt(it[k])
    o = np.argsort(-it)[:60]
    return mz[o], it[o] / (np.linalg.norm(it[o]) + 1e-12)


def _cos(a, b, tol=0.02):
    (m1, i1), (m2, i2) = a, b
    if not len(m1) or not len(m2):
        return 0.0, 0
    d = np.abs(m1[:, None] - m2[None, :]) <= tol
    pairs = np.argwhere(d)
    if not len(pairs):
        return 0.0, 0
    sc = i1[pairs[:, 0]] * i2[pairs[:, 1]]
    u1, u2, tot, n = set(), set(), 0.0, 0
    for j in np.argsort(-sc):
        x, y = pairs[j]
        if x in u1 or y in u2:
            continue
        u1.add(x); u2.add(y); tot += sc[j]; n += 1
    return float(tot), n


def direct_quality(out, mask, work, min_match=3, max_refs=40):
    """q[mid][rank] = best greedy cosine between the molecule's query spectra and train reference spectra of the
    candidate's standard InChIKey14 (same polarity, >= min_match matched peaks) under the mask's train.parquet."""
    import pyarrow.dataset as pds
    from rdkit import Chem
    M = _metric()
    sub = pd.read_csv(Path(work) / 'submission.csv')
    lists = {m: [x for x in str(s).split(';') if x][:25] for m, s in zip(sub.molecule_id, sub.smiles)}
    def ik(s):
        try:
            return Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14]
        except Exception:
            return None
    cik = {s: ik(s) for l in lists.values() for s in l}
    want = sorted({k for k in cik.values() if k})
    tr = pds.dataset(Path(out) / f'in_{mask}' / 'competitions' / SLUG / 'train.parquet').to_table(
        filter=pds.field('inchikey14').isin(want),
        columns=['inchikey14', 'ionization_mode', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities', 'ingest_lib']).to_pandas()
    refs = {}
    for r in tr.itertuples(index=False):
        refs.setdefault(r.inchikey14, []).append((r.ionization_mode, _clean(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz)))
    te = pd.read_parquet(Path(out) / f'in_{mask}' / 'competitions' / SLUG / 'test.parquet')
    qs = {m: [(r.ionization_mode, _clean(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz)) for r in g.itertuples(index=False)]
          for m, g in te.groupby('molecule_id')}
    Q = {}
    for m, l in lists.items():
        row = []
        for s in l:
            best = 0.0
            for pol, rs in refs.get(cik.get(s), [])[:max_refs]:
                for qpol, qsp in qs[m]:
                    if qpol != pol:
                        continue
                    c, n = _cos(qsp, rs)
                    if n >= min_match and c > best:
                        best = c
            row.append(best)
        Q[m] = row
    return lists, Q


def cmd_posthoc(a):
    out, work = Path(a.out), Path(a.out) / 'runs' / a.name
    cache = work / 'e021_direct_quality.json'
    if cache.exists():
        lists, Q = json.load(open(cache))
    else:
        lists, Q = direct_quality(out, a.name.split('_')[0], work)
        json.dump([lists, Q], open(cache, 'w'))
    for qmin in a.qmin:
        for marg in a.margin:
            for pos in a.pos:
                rows = []
                for m, l in lists.items():
                    q = Q[m]
                    l2 = list(l)
                    if q:
                        j = int(np.argmax(q))
                        if j >= pos and q[j] >= qmin and q[j] - max(q[:pos]) >= marg:
                            l2 = l[:pos - 1] + [l[j]] + [x for k, x in enumerate(l) if k != j and k >= pos - 1]
                    rows.append((m, ';'.join(l2[:25]) if l2 else 'CCO'))
                tag = f'dq_q{qmin}_m{marg}_p{pos}'
                wd = work / 'posthoc' / tag
                wd.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows, columns=['molecule_id', 'smiles']).to_csv(wd / 'submission.csv', index=False)
                sc = score_dir(out, wd)
                print('E021_POSTHOC', a.name, tag, json.dumps(sc['summary']['final']), flush=True)

def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('panel'); p.add_argument('--kin', required=True); p.add_argument('--out', required=True)
    p = sp.add_parser('run'); p.add_argument('--out', required=True); p.add_argument('--mask', choices='ABRCHPQ', required=True)
    p.add_argument('--name', default='base'); p.add_argument('--set', action='append', default=[])
    p.add_argument('--notebook', default='/workspace/k020/enveda-020-e006-repro.ipynb')
    p.add_argument('--hook', default=None, help='python file exec-ed into the notebook namespace first (variant code)')
    p.add_argument('--inroot', default=None)
    p.add_argument('--sweep', default=None, help='json {name: {CONST: value}}; tail cells re-run per config')
    p.add_argument('--reuse-eng', default=None, help='path to eng_lists.json to skip Engine2')
    p.add_argument('--reuse-pc', default=None, help='path to pc_lists.json to skip PubChem')
    p = sp.add_parser('score'); p.add_argument('--out', required=True); p.add_argument('--name', required=True)
    p = sp.add_parser('compare'); p.add_argument('--out', required=True); p.add_argument('base'); p.add_argument('other')
    p = sp.add_parser('posthoc'); p.add_argument('--out', required=True); p.add_argument('--name', required=True)
    p.add_argument('--qmin', type=float, nargs='+', default=[0.7]); p.add_argument('--margin', type=float, nargs='+', default=[0.1])
    p.add_argument('--pos', type=int, nargs='+', default=[1])
    a = ap.parse_args()
    dict(posthoc=cmd_posthoc, panel=cmd_panel, run=cmd_run, score=cmd_score, compare=cmd_compare)[a.cmd](a)


if __name__ == '__main__':
    main()

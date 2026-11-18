"""Is the N-1c network rerank leader-safe? (072b follow-up, same defect class as RR8)

`_n1c` in 068a reorders the whole 25-candidate list with
    argsort(-(z(-log rank) + LAM * z(network agreement)))
including position 0, and prints `top1 changed`. RR8 had the same hole and closing it moved
MRR from -0.0013 to +0.0042. This probe asks the same question of N-1c on the 120-molecule
labelled panel, with LAM fixed at the shipped 0.35.

Only the ranking rule differs between the two arms; the network, fingerprints and weights
are computed once and shared.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
from rdkit import Chem, DataStructs, RDLogger                      # noqa: E402
from rdkit.Chem import rdFingerprintGenerator                       # noqa: E402

RDLogger.DisableLog('rdApp.*')
OUT = Path('artifacts/rr8x')
LAM, TAU = 0.35, 0.6
N_BOOT = 10000
RNG = np.random.default_rng(20261117)

PANEL = Path('artifacts/ds/catalogue-experiment-saner/catalogue_replay_checkpoint/panel')
SCORES = Path('artifacts/ds/catalogue-experiment-saner/catalogue_experiment/labelled_candidate_scores.parquet')


def z(v):
    return (v - v.mean()) / (v.std() + 1e-9)


def mcos(a, b):
    (ma, ia_, pa), (mb, ib, pb) = a, b
    sh = pa - pb
    cand = []
    for i in range(len(ma)):
        for off in (0.0, sh):
            lo = np.searchsorted(mb, ma[i] - off - 0.0101)
            hi = np.searchsorted(mb, ma[i] - off + 0.0101, 'right')
            for j in range(lo, hi):
                cand.append((ia_[i] * ib[j], i, j))
    cand.sort(reverse=True)
    ui, uj, s, n = set(), set(), 0.0, 0
    for v, i, j in cand:
        if i in ui or j in uj:
            continue
        ui.add(i)
        uj.add(j)
        s += v
        n += 1
    return s, n


def boot(d):
    d = np.asarray(d, float)
    idx = RNG.integers(0, len(d), size=(N_BOOT, len(d)))
    m = d[idx].mean(1)
    return dict(n=int(len(d)), mean=float(d.mean()), lo=float(np.quantile(m, .025)),
                hi=float(np.quantile(m, .975)), p_gt0=float((m > 0).mean()))


def main() -> int:
    te = pd.read_parquet(PANEL / 'test.parquet')
    sc = pd.read_parquet(SCORES)
    te['_m'] = te.molecule_id.astype(str)
    lists, ytrue = {}, {}
    for mid, g in sc.groupby('molecule_id', sort=False):
        g = g.sort_values('baseline_rank', kind='stable')
        lists[str(mid)] = g.smiles.tolist()
        y = g.label.values
        ytrue[str(mid)] = int(np.flatnonzero(y == 1)[0]) if y.sum() else None

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fpc = {}

    def fp(s):
        if s not in fpc:
            m = Chem.MolFromSmiles(s)
            fpc[s] = gen.GetFingerprint(m) if m is not None else None
        return fpc[s]

    SP = {}
    for (m, ad), g in te.groupby(['_m', 'adduct']):
        if str(ad) not in ('[M+H]+', '[M-H]-') or m not in lists:
            continue
        pr = float(np.median(pd.to_numeric(g.precursor_mz, errors='coerce')))
        mz = np.concatenate([np.asarray(x, float) for x in g.ms2_mzs])
        it = np.concatenate([np.asarray(x, float) / max(float(np.max(x)) if len(x) else 1.0, 1e-9)
                             for x in g.ms2_normalized_intensities])
        k = mz < pr - 1.5
        mz, it = mz[k], it[k]
        if not len(mz):
            continue
        kb = np.round(mz, 2)
        u, inv = np.unique(kb, return_inverse=True)
        w = np.zeros(len(u))
        np.maximum.at(w, inv, it)
        o = np.argsort(-w)[:120]
        u, w = u[o], np.sqrt(w[o])
        oo = np.argsort(u)
        SP.setdefault(m, {})[str(ad)] = (u[oo], (w / max(np.linalg.norm(w), 1e-12))[oo], pr)

    K = sorted(SP)
    E = {}
    for x in range(len(K)):
        for y in range(x + 1, len(K)):
            best = None
            for ad in set(SP[K[x]]) & set(SP[K[y]]):
                s, n = mcos(SP[K[x]][ad], SP[K[y]][ad])
                if n >= 6 and s >= TAU and (best is None or s > best):
                    best = s
            if best is not None:
                E.setdefault(K[x], []).append((K[y], best))
                E.setdefault(K[y], []).append((K[x], best))
    print(f'molecules with spectra {len(K)}, with neighbours {len(E)}, edges {sum(len(v) for v in E.values())//2}')

    results = {}
    for protect in (False, True):
        d, t1 = [], 0
        detail = {}
        for mid, lst0 in lists.items():
            lst = lst0[:25]
            nb_ = E.get(mid, [])
            if not (nb_ and len(lst) > 1):
                detail[mid] = 0.0
                continue
            base = -np.log(np.arange(1, len(lst) + 1))
            N = np.zeros(len(lst))
            for b, w in nb_:
                lb = lists.get(b, [])[:5]
                p = 1 / np.arange(1, len(lb) + 1)
                p = p / p.sum() if len(lb) else p
                jf = [(fp(j), pp) for j, pp in zip(lb, p)]
                for i, c in enumerate(lst):
                    fi = fp(c)
                    if fi is None:
                        continue
                    N[i] += w ** 4 * sum(pp * DataStructs.TanimotoSimilarity(fi, fj)
                                         for fj, pp in jf if fj is not None)
            score = z(base) + LAM * z(N)
            if protect:
                # position 0 keeps its place; the rest reorder among their own slots
                rest = np.argsort(-score[1:], kind='stable') + 1
                order = np.concatenate([[0], rest])
            else:
                order = np.argsort(-score, kind='stable')
            nl = [lst[i] for i in order]
            t1 += int(nl[0] != lst[0])
            t = ytrue.get(mid)
            rr = 0.0 if t is None else 1.0 / (nl.index(lst[t]) + 1 if lst[t] in nl else 1e9)
            rr = 0.0 if t is None else 1.0 / (nl.index(lst[t]) + 1)
            b0 = 0.0 if t is None else 1.0 / (lst.index(lst[t]) + 1)
            d.append(rr - b0)
            detail[mid] = rr - b0
        results['protect' if protect else 'plain'] = dict(
            delta=float(np.mean(d)), top1_changed=t1, boot=boot(d))
        print(f"[n1c] protect_top1={protect!s:5s} delta={np.mean(d):+.4f} top1_changed={t1} "
              f"boot[{results['protect' if protect else 'plain']['boot']['lo']:+.4f},"
              f"{results['protect' if protect else 'plain']['boot']['hi']:+.4f}] "
              f"p>0={boot(d)['p_gt0']:.3f}", flush=True)
    (OUT / 'n1c_leader_safe.json').write_text(json.dumps(results, indent=1))
    print('wrote', OUT / 'n1c_leader_safe.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

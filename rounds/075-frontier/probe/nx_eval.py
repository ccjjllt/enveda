#!/usr/bin/env python
"""075c 评估：NX 同式组重排在 072d 仪器上的 MRR 变化。

输入：
  artifacts/kout/075c/nx_scores.csv   075c kernel 输出（molecule_id, idx, smiles, score, model）
  artifacts/kout/072d_v3/submission.csv  仪器基线候选（150 x 25，molecule_id = 真值 inchikey14）

规则（与作者 nx_rerank.py 一致，只改同式组内顺序）：
  order = argsort( -( z(-rank) + w * z(NX) ) )   组内 z 标准化
  w 网格：0（=基线）、0.5、1.0、1.5、2.0、3.0

指标：MRR@25、top1、真值名次分布；另报「基线已错」子集（名次>1）上的 MRR。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

SCORES = Path("artifacts/kout/075c/nx_scores.csv")
SUB = Path("artifacts/kout/072d_v3/submission.csv")
OUT = Path("artifacts/kout/075c/nx_eval.json")

WEIGHTS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


def formula(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    from rdkit.Chem import rdMolDescriptors

    return rdMolDescriptors.CalcMolFormula(m)


def zrank(v):
    a = np.asarray(v, dtype=float)
    sd = a.std()
    return (a - a.mean()) / (sd if sd > 1e-9 else 1.0)


def rerank(cands, scores, w):
    """同式组内按 z(-rank) + w*z(NX) 重排；组外位置不动。"""
    order = np.arange(len(cands))
    by = {}
    for i, s in enumerate(cands):
        f = formula(s)
        if f is not None:
            by.setdefault(f, []).append(i)
    for idx in by.values():
        if len(idx) < 2:
            continue
        idx = np.asarray(idx)
        za = zrank(-idx.astype(float))
        zb = zrank(scores[idx])
        order[idx] = idx[np.argsort(-(za + w * zb), kind="stable")]
    return [cands[i] for i in order]


def mrr(ranks):
    return float(np.mean([1.0 / r for r in ranks]))


def main():
    if not SCORES.exists():
        sys.exit(f"missing {SCORES}")
    base = pd.read_csv(SUB)
    base["molecule_id"] = base.molecule_id.astype(str)
    nx = pd.read_csv(SCORES)
    nx["molecule_id"] = nx.molecule_id.astype(str)

    out = {}
    for model in sorted(nx.model.unique()):
        m = nx[nx.model == model]
        if m.molecule_id.nunique() < len(base):
            print(f"[warn] {model}: only {m.molecule_id.nunique()}/{len(base)} molecules scored")
        res = {}
        for w in WEIGHTS:
            ranks = []
            for r in base.itertuples(index=False):
                cands = str(r.smiles).split(";")
                mid = r.molecule_id
                sub = m[m.molecule_id == mid]
                if sub.empty or w == 0.0:
                    new = cands
                else:
                    sc = np.zeros(len(cands))
                    for row in sub.itertuples(index=False):
                        if 0 <= row.idx < len(cands) and cands[row.idx] == row.smiles:
                            sc[row.idx] = row.score
                    new = rerank(cands, sc, w)
                truth = [i for i, s in enumerate(new)]
                # 真值 = molecule_id（inchikey14）；候选里比较 InChIKey14
                rank = None
                for i, s in enumerate(new):
                    mol = Chem.MolFromSmiles(s)
                    if mol is not None and Chem.MolToInchiKey(mol)[:14] == mid:
                        rank = i + 1
                        break
                ranks.append(rank if rank else 26)
            ranks = np.array(ranks, dtype=float)
            hard = ranks[ranks > 1]
            res[f"w={w}"] = dict(
                mrr=round(mrr(ranks), 4),
                top1=round(float((ranks == 1).mean()), 4),
                mrr_hard=round(mrr(hard), 4) if len(hard) else None,
                n_hard=int(len(hard)),
                hist={int(k): int(v) for k, v in zip(*np.unique(ranks, return_counts=True))},
            )
        out[model] = res
        print(f"== {model}")
        for k, v in res.items():
            print(f"  {k:6s} MRR {v['mrr']:.4f}  top1 {v['top1']:.3f}  hard MRR {v['mrr_hard']}  n_hard {v['n_hard']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

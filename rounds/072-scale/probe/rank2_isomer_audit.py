#!/usr/bin/env python
"""072d 补充审计：残留排序错误是不是「分子式错误」？

输入（均已在本地）：
  artifacts/kout/072d_v3/submission.csv        150 x 25 候选（分号分隔 SMILES）
  data/raw/train.parquet 第 1 个 row group    inchikey14 / normalized_smiles / molecular_formula

做法：
  1. 用 RDKit 把每个候选 SMILES 转 InChIKey 前 14 位，与查询 molecule_id 比对得到真值名次。
  2. 对每个「真值不在榜首」的查询，比较榜首候选与真值的分子式。
  3. 对名次 2 的子集，再算榜首候选与真值的 Morgan 指纹 Tanimoto（衡量结构距离）。

输出：
  artifacts/kout/072d_v3/per_query_ranks.json   每查询的名次与候选列表
  artifacts/kout/072d_v3/rank2_formula.json     名次 2 子集的分子式/Tanimoto 明细

结论（2026-11-18 运行）：
  48 个错误中 47 个的榜首与真值同分子式（同分异构体），仅 1 个分子式不同；
  42 个名次 2 全部为同分异构体，Tanimoto 均值 0.470 / 中位 0.471。
"""
import json
import statistics

import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

SUB = "artifacts/kout/072d_v3/submission.csv"
OUT_RANKS = "artifacts/kout/072d_v3/per_query_ranks.json"
OUT_R2 = "artifacts/kout/072d_v3/rank2_formula.json"


def ik14(smiles):
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(m)[:14] if m else None


def molfp(smiles):
    m = Chem.MolFromSmiles(smiles) if smiles else None
    return rdMolDescriptors.CalcMolFormula(m) if m else None


def morgan(smiles):
    m = Chem.MolFromSmiles(smiles) if smiles else None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None


def main():
    sub = pd.read_csv(SUB)
    rows = []
    for _, r in sub.iterrows():
        cands = [x for x in str(r.smiles).split(";") if x]
        ids = [ik14(x) for x in cands]
        q = str(r.molecule_id)
        rows.append(dict(q=q, rank=(ids.index(q) + 1 if q in ids else None), cands=cands))
    json.dump(rows, open(OUT_RANKS, "w"))

    truth = (
        pq.ParquetFile("data/raw/train.parquet")
        .read_row_group(0, columns=["inchikey14", "normalized_smiles", "molecular_formula"])
        .to_pandas()
        .drop_duplicates("inchikey14")
    )
    tf = dict(zip(truth.inchikey14, truth.molecular_formula))
    ts = dict(zip(truth.inchikey14, truth.normalized_smiles))

    miss = [x for x in rows if x["rank"] != 1]
    iso = 0
    diff = []
    for x in miss:
        f1, ft = molfp(x["cands"][0]), tf.get(x["q"])
        if f1 == ft:
            iso += 1
        else:
            diff.append((x["q"], x["rank"], f1, ft))
    print(f"rank>1: {len(miss)} | same formula (isomer): {iso} | different: {len(diff)}")
    for d in diff:
        print("  ", d)

    out, tan = [], []
    for x in [y for y in rows if y["rank"] == 2]:
        a, b = morgan(x["cands"][0]), morgan(ts.get(x["q"]))
        sim = DataStructs.TanimotoSimilarity(a, b) if (a is not None and b is not None) else None
        if sim is not None:
            tan.append(sim)
        out.append(dict(q=x["q"], f1=molfp(x["cands"][0]), ft=tf.get(x["q"]), tan=sim))
    if tan:
        print("rank2 tanimoto mean %.3f median %.3f" % (statistics.mean(tan), statistics.median(tan)))
    json.dump(out, open(OUT_R2, "w"), indent=1)


if __name__ == "__main__":
    main()

"""004 分子级评估：按最终候选位置计 RR，非空率与真值召回分开，逐分子表.

相对 001 版的修正（见 docs/plan/004-tpu-finetuning.md“评分和化学结构语义”）：
- RR 按最终输出顺序计算，不在排名前去重。例：`CC;CC;CCO` 对真值 `CCO`
  的 RR 为 1/3（重复错误候选占位），不得压缩为 1/2。
- 空分号项删除（不占排名）；非空无效/重复项保留排名占位。
- `nonempty_rate`（有>=1 非空候选的分子占比）与 `recall@k`（真值在候选内，
  无论排名）分开统计；旧 `coverage` 更名为非空预测率，不得混用。
- 匹配键默认走 src.chemistry.official_match_key（RDKit v13 语义）；
  无 RDKit 时回退精确字符串，结果标注 caliber="placeholder_exact_string"。
- 去空后零候选或超 25 个候选的分子记 rejected（官方拒绝口径），RR=0。

旧占位函数（placeholder_match/dedup_keep_order/旧 evaluate_molecules）保留作
历史对照，标注 deprecated，不得用于新结论。

手动研究用模块，不是测试文件。不读凭据，不联网。
"""

from __future__ import annotations

from typing import Dict, List, Optional

K = 25


def split_candidates(cell: str) -> List[str]:
    """按 ';' 切分、strip、删除空项（官方规则：空项不占排名）."""
    if cell is None:
        return []
    return [p.strip() for p in str(cell).split(";") if p.strip() != ""]
# NOTE: 001 旧版同名函数语义相同，保留行为。


def _match_keys(preds: List[str], truth: str) -> List[str]:
    """返回候选与真值的匹配键列表；(caliber, pred_keys, truth_key)."""
    try:
        from src import chemistry as C  # noqa
        tk = C.official_match_key(truth)
        pk = []
        for p in preds:
            try:
                pk.append(C.official_match_key(p))
            except ValueError:
                pk.append(None)  # 非空无效：占位，永不命中
        return ("official_v13", pk, tk)
    except Exception:
        # 无 RDKit（本地 .venv）：精确字符串占位，调用方必须看 caliber
        return ("placeholder_exact_string", list(preds), truth)


def reciprocal_rank_by_position(
    ranked: List[str], truth: str, k: int = K
) -> Dict:
    """单分子 RR（按最终位置，不去重）.

    返回 {rank（1-based，无命中为 None）, rr, caliber, n_candidates,
    rejected（零候选或超 k）}。
    """
    cleaned = [s for s in (str(s).strip() for s in ranked) if s != ""]
    if len(cleaned) == 0 or len(cleaned) > k:
        return {"rank": None, "rr": 0.0, "caliber": "rejected",
                "n_candidates": len(cleaned), "rejected": True}
    caliber, pkeys, tkey = _match_keys(cleaned, truth)
    for i, pk in enumerate(pkeys):
        if pk is not None and pk == tkey:
            return {"rank": i + 1, "rr": 1.0 / (i + 1), "caliber": caliber,
                    "n_candidates": len(cleaned), "rejected": False}
    return {"rank": None, "rr": 0.0, "caliber": caliber,
            "n_candidates": len(cleaned), "rejected": False}


def evaluate_molecules(
    truth: Dict[str, str],
    preds: Dict[str, List[str]],
    truth_in_pool: Optional[Dict[str, bool]] = None,
    k: int = K,
) -> Dict:
    """分子级评估（004 口径）.

    truth: molecule_id -> 真值 SMILES。preds: molecule_id -> 最终有序候选
    （调用方已完成候选生成阶段去重；本函数不去重，直接按位置评分）。
    truth_in_pool: molecule_id -> 真值是否在候选池（用于 recall 分母；
    不传则 recall 缺失并标注）。

    返回聚合 {n, mrr@k, top1, top25, nonempty_rate, recall@k(或 None),
    rejected_ratio, caliber} + per_molecule 明细表。
    """
    rows: List[Dict] = []
    rr_sum = 0.0
    top1 = topk = nonempty = recalled = 0
    pool_n = 0
    rejected = 0
    calibers = set()
    for mol, t in truth.items():
        raw = preds.get(mol, [])
        r = reciprocal_rank_by_position(raw, t, k=k)
        calibers.add(r["caliber"])
        in_pool = None if truth_in_pool is None else bool(truth_in_pool.get(mol, False))
        rows.append({"molecule_id": mol, "rank": r["rank"], "rr": r["rr"],
                     "n_candidates": r["n_candidates"],
                     "rejected": r["rejected"],
                     "truth_in_pool": in_pool})
        rr_sum += r["rr"]
        if r["rr"] == 1.0:
            top1 += 1
        if r["rr"] > 0.0:
            topk += 1
        if not r["rejected"]:
            nonempty += 1
        else:
            rejected += 1
        if in_pool is True:
            pool_n += 1
            if r["rank"] is not None:
                recalled += 1
    n = len(truth)
    if n == 0:
        return {"n_molecules": 0, "mrr@%d" % k: 0.0, "top1": 0.0,
                "top25": 0.0, "nonempty_rate": 0.0, "recall@%d" % k: None,
                "rejected_ratio": 0.0, "caliber": "empty",
                "per_molecule": []}
    return {
        "n_molecules": float(n),
        "mrr@%d" % k: rr_sum / n,
        "top1": top1 / n,
        "top25": topk / n,
        "nonempty_rate": nonempty / n,
        "recall@%d" % k: (recalled / pool_n if pool_n else None),
        "rejected_ratio": rejected / n,
        "caliber": "|".join(sorted(calibers)),
        "per_molecule": rows,
    }


# ---- 001 旧占位（deprecated，仅历史对照） ----

def placeholder_normalize(smiles: str) -> str:  # deprecated
    return str(smiles).strip()


def placeholder_match(pred: str, truth: str) -> bool:  # deprecated
    return placeholder_normalize(pred) == placeholder_normalize(truth)


def reciprocal_rank(ranked: List[str], truth: str, k: int = K) -> float:  # deprecated
    for i, pred in enumerate(ranked[:k]):
        if placeholder_match(pred, truth):
            return 1.0 / (i + 1)
    return 0.0


def dedup_keep_order(ranked: List[str]) -> List[str]:  # deprecated
    seen, out = set(), []
    for smi in ranked:
        if smi not in seen:
            seen.add(smi)
            out.append(smi)
    return out

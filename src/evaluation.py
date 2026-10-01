"""001 分子级评估骨架(MRR@25 / Top-1 / Top-25 / 覆盖率 / 异常比例).

本文件是手动研究用评估代码,不是测试文件:
- 不依赖 pytest/unittest,不做自动化断言门控;
- 由研究人员在 notebook/命令行手动调用,把指标与观察记录进 progress 文档.

官方语义注释(以官方评分代码为准,本实现为无 RDKit 环境的占位):
1. 空分号项删除:提交行按 ';' 切分后,strip 为空的项在分配排名前删掉,
   不占排名.如 "CCO;;CC" -> ["CCO", "CC"].
2. 非空但无效的 SMILES 占排名且不命中:如 "not_a_smiles" 保留在原排名位置,
   浪费候选位,永远判 miss.本占位实现无 RDKit,无法判定化学有效性,仅把
   空白项按规则 1 删除,其余非空项一律视为“占位候选”(见 invalid_ratio 说明).
3. 去重:官方按 RDKit 2026.03.3 规范化(含互变异构体)后取 InChIKey 前 14 字符
   去重,保留最佳顺序,重复等价结构浪费候选位.本占位用精确字符串去重并报告
   duplicate_ratio;真实去重需 RDKit 到位后替换 placeholder_match/dedup.
4. 分子级聚合:MRR@25 = 每分子(首个正确候选排名 r -> 1/r,前 25 未命中 -> 0)
   后对分子平均.Top-1/Top-25 同理按分子平均.覆盖率与异常比例亦按分子统计.

匹配占位:placeholder_match 用精确字符串相等;官方用 InChIKey14 等价.
在 RDKit 未接入前,本模块分数只用于流程自检,不作为研究结论引用.

仅依赖标准库,Kaggle 离线可用.不读凭据,不联网.
"""

from __future__ import annotations

from typing import Dict, List

K = 25


def split_candidates(cell: str) -> List[str]:
    """切分一行提交候选:按 ';' 切分、strip、删除空项(官方语义规则 1).

    非空但化学无效的项保留(占排名,规则 2).本函数不做化学有效性判断.
    """
    if cell is None:
        return []
    parts = str(cell).split(";")
    return [p.strip() for p in parts if p.strip() != ""]


def placeholder_normalize(smiles: str) -> str:
    """占位规范化:仅 strip.

    TODO(RDKit 到位后):替换为 RDKit 2026.03.3 规范化 + 互变异构体处理,
    匹配键取 InChIKey 前 14 字符;无效 SMILES 单独计数,不静默丢弃.
    """
    return str(smiles).strip()


def placeholder_match(pred: str, truth: str) -> bool:
    """占位命中判定:精确字符串相等.官方语义为 InChIKey14 等价(见模块注释)."""
    return placeholder_normalize(pred) == placeholder_normalize(truth)


def reciprocal_rank(ranked: List[str], truth: str, k: int = K) -> float:
    """单分子倒数排名:前 k 中首个命中排名 r -> 1/r,否则 0.

    ranked 应为已按 split_candidates 清洗 + 去重后的有序列表(至多 k 个).
    无效 SMILES 已在 ranked 中占位(判 miss),空项不应传入.
    """
    for i, pred in enumerate(ranked[:k]):
        if placeholder_match(pred, truth):
            return 1.0 / (i + 1)
    return 0.0


def dedup_keep_order(ranked: List[str]) -> List[str]:
    """占位去重:精确字符串保序去重.官方为 InChIKey14 去重(见模块注释)."""
    seen = set()
    out: List[str] = []
    for smi in ranked:
        if smi not in seen:
            seen.add(smi)
            out.append(smi)
    return out


def evaluate_molecules(
    truth: Dict[str, str],
    preds: Dict[str, List[str]],
    k: int = K,
) -> Dict[str, float]:
    """分子级评估(占位实现,官方语义注释见模块头).

    参数:
        truth: molecule_id -> 真值 SMILES(研究留出集的真值;隐藏集不可用).
        preds: molecule_id -> 有序候选 SMILES 列表(原始顺序,本函数内清洗).
        k: 候选截断(默认 25).

    返回按分子平均的:
        n_molecules / mrr@k / top1 / top25 / coverage(有>=1 非空候选的分子占比)
        / no_candidate_ratio(无候选或全空项的分子占比)
        / duplicate_ratio(有精确字符串重复浪费槽位的分子占比)
        / invalid_or_empty_slot_ratio(占位口径:空项槽位占比,因无 RDKit,
          非空项暂无法判化学有效,此值只反映“空分号项”浪费,不代表真实无效
          SMILES 率;RDKit 到位后替换).
    """
    n = len(truth)
    if n == 0:
        return {
            "n_molecules": 0,
            "mrr@%d" % k: 0.0,
            "top1": 0.0,
            "top25": 0.0,
            "coverage": 0.0,
            "no_candidate_ratio": 0.0,
            "duplicate_ratio": 0.0,
            "invalid_or_empty_slot_ratio": 0.0,
        }
    rr_sum = 0.0
    top1_hit = 0
    topk_hit = 0
    with_cand = 0
    no_cand = 0
    with_dup = 0
    empty_slots = 0
    total_slots = 0
    for mol, t in truth.items():
        raw = preds.get(mol, [])
        # 空项统计:按原始 ';' 语义,这里 raw 为列表,空字符串元素视为空项.
        total_slots += max(len(raw), 1)
        empty_slots += sum(1 for s in raw if str(s).strip() == "")
        cleaned = [s for s in (str(s).strip() for s in raw) if s != ""]
        if len(cleaned) != len(set(cleaned)):
            with_dup += 1
        ranked = dedup_keep_order(cleaned)[:k]
        if ranked:
            with_cand += 1
        else:
            no_cand += 1
        rr = reciprocal_rank(ranked, t, k=k)
        rr_sum += rr
        if rr == 1.0:
            top1_hit += 1
        if rr > 0.0:
            topk_hit += 1
    return {
        "n_molecules": float(n),
        "mrr@%d" % k: rr_sum / n,
        "top1": top1_hit / n,
        "top25": topk_hit / n,
        "coverage": with_cand / n,
        "no_candidate_ratio": no_cand / n,
        "duplicate_ratio": with_dup / n,
        "invalid_or_empty_slot_ratio": empty_slots / total_slots
        if total_slots
        else 0.0,
    }

"""004-1c 候选排序：single 打分、候选级多谱聚合、简单学习排序器占位.

手动研究用模块，不是测试文件（无 pytest/unittest，不做断言门控）。
仅依赖标准库；不读凭据，不联网；不硬编码任何分子/谱 ID。

single（单谱 -> 候选打分）：
- 输入同一质量窗口 C 内候选，预测指纹 logits（FPNet 输出，6930 维）
  与候选二值指纹（6930 位，经 fp_bits.npy 选位后）做点积 f.z。
  这是 v32 "ranked by f.z" 的显式规则；sigmoid 单调，不改变排序，
  此处直接用 logits 点积，阈值/校准另行研究。
- 批量打分时候选指纹按 batch 解包搬运（见 src/candidates
  PackedFingerprintStore.get_batch），禁止全池一次性 unpack 上 GPU。

多谱聚合（候选级，明确规则）：
- 默认规则 candidate_mean：同一 Q 组（同一分子的多张谱）对同一候选的
  单谱分数取均值，再全局排序。要求 single/scorefusion/merged 共用同一
  Q/L/C；single 按不依赖真值的规则选一张 Q，融合只能读 Q；先做
  同极性/同 adduct 融合，再比较跨条件聚合（见 004 计划）。
- 另提供 true_round_robin：对“每谱一个有序候选列表”的列表级融合做
  真正轮转（按 rank 位置逐轮、谱间轮转、保序去重）。这修正了旧
  src/retrieval.aggregate_by_molecule 的实现问题：旧实现按 spectrum_id
  排序后把第一张谱的候选全部追加完再追加第二张（顺序拼接），不是真正
  的轮转；且截断语义只在桶内 break，调用方易误读为全局已截断。
  本模块两种聚合都执行严格的全局 <=25 截断，并返回截断计数
  （n_truncated、n_input_unique），调用方写入逐分子表。
- 超 25 截断发生在排序/去重之后；空项不占位（上游已按官方语义清洗），
  无效/重复候选的占位语义由 src/evaluation 与官方 scorer 负责，
  本模块只对传入的候选顺序排序，不做化学有效性判断。

学习排序器（占位接口）：
- SimpleLinearRanker 只是特征权重占位：特征列固定为
  [fp_dot, spectral_cosine, neg_abs_ppm, adduct_match, instrument_match]，
  供 R4 在同候选集下检验互补错误。fit 必须用 OOF 或独立 query 组样本，
  不得在验证标签上调负例、不得用随机假输入训练后宣称实验完成。
  未拟合时 predict 抛错，不静默回退等权。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

MAX_CANDIDATES = 25

# R4 占位特征列顺序（固定，权重与此对齐）。
RANKER_FEATURES: List[str] = [
    "fp_dot", "spectral_cosine", "neg_abs_ppm", "adduct_match",
    "instrument_match",
]


def dot_score(logits: Sequence[float], bits: Sequence[int]) -> float:
    """单谱单候选分数：sum(logit * bit)，即 f.z（v32 显式规则）。

    logits 与 bits 长度必须相等（6930）；否则抛错，不截断凑数。
    bits 按 truthiness 取 0/1。
    """
    if len(logits) != len(bits):
        raise ValueError("logits/bits length mismatch %d vs %d"
                         % (len(logits), len(bits)))
    s = 0.0
    for l, b in zip(logits, bits):
        if b:
            s += float(l)
    return float(s)


def score_single_spectrum(logits: Sequence[float],
                          candidates: Sequence[Tuple[str, Sequence[int]]]
                          ) -> List[Tuple[str, float]]:
    """单谱打分：返回 [(candidate_key, score)] 按分数降序。

    candidates 为 [(key, bits)]，bits 必须已是 6930 位选位后。
    同分按 key 排序以保证确定性；本函数不截断，截断由聚合侧执行。
    """
    scored = [(k, dot_score(logits, b)) for k, b in candidates]
    scored.sort(key=lambda p: (-p[1], p[0]))
    return scored


def aggregate_mean_scores(
    per_spectrum_scores: Dict[str, Dict[str, float]],
    max_candidates: int = MAX_CANDIDATES,
) -> Tuple[List[Tuple[str, float]], Dict[str, int]]:
    """候选级均值聚合（默认规则）：同候选跨谱分数取均值后排序。

    per_spectrum_scores: {spectrum_id: {candidate_key: score}}。
    缺席某谱的候选只按出现谱数平均（分母为出现次数），并在 stats 中记
    n_spectra 与 n_candidates_scored；调用方应保证同 Q 组、同 C 池。
    返回 ([(key, mean)] 降序截断, stats{n_spectra, n_input_unique,
    n_truncated, n_output})。同分按 key 确定性排序。
    """
    if not per_spectrum_scores:
        return [], {"n_spectra": 0, "n_input_unique": 0, "n_truncated": 0,
                    "n_output": 0}
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for spec_id in sorted(per_spectrum_scores.keys()):
        for key, s in per_spectrum_scores[spec_id].items():
            sums[key] = sums.get(key, 0.0) + float(s)
            counts[key] = counts.get(key, 0) + 1
    ranked = [(k, sums[k] / counts[k]) for k in sums]
    ranked.sort(key=lambda p: (-p[1], p[0]))
    n_in = len(ranked)
    out = ranked[: int(max_candidates)]
    stats = {"n_spectra": len(per_spectrum_scores), "n_input_unique": n_in,
             "n_truncated": max(0, n_in - len(out)), "n_output": len(out)}
    return out, stats


def round_robin_fuse(
    per_spectrum_rankings: Dict[str, List[str]],
    max_candidates: int = MAX_CANDIDATES,
) -> Tuple[List[str], Dict[str, int]]:
    """真正的轮转融合：按 rank 位置逐轮、谱间轮转、保序去重、全局截断。

    与旧 retrieval.aggregate_by_molecule 的区别：旧实现按 spectrum_id 排序后
    依次把每张谱的全部候选追加进桶（第一张谱排完才排第二张），不是轮转；
    本函数按 depth=0,1,2... 逐层轮转各谱，保证多谱首位候选优先交错。
    输入谱顺序按 spectrum_id 排序以确定性执行；去重保序；全局截断到
    max_candidates 并计数。返回 (ranked, stats{n_spectra, n_input_unique,
    n_truncated, n_output})。
    """
    spec_ids = sorted(per_spectrum_rankings.keys())
    lists = [list(per_spectrum_rankings[s]) for s in spec_ids]
    seen = set()
    unique_in = set()
    for lst in lists:
        unique_in.update(lst)
    out: List[str] = []
    depth = 0
    active = True
    while active:
        active = False
        for lst in lists:
            if depth < len(lst):
                active = True
                smi = lst[depth]
                if smi not in seen:
                    seen.add(smi)
                    out.append(smi)
                    if len(out) >= int(max_candidates):
                        stats = {"n_spectra": len(spec_ids),
                                 "n_input_unique": len(unique_in),
                                 "n_truncated": max(0, len(unique_in) - len(out)),
                                 "n_output": len(out)}
                        return out, stats
        depth += 1
    stats = {"n_spectra": len(spec_ids), "n_input_unique": len(unique_in),
             "n_truncated": max(0, len(out) - int(max_candidates)),
             "n_output": len(out[: int(max_candidates)])}
    return out[: int(max_candidates)], stats


@dataclass
class SimpleLinearRanker:
    """R4 简单线性重排器占位：score = w . features（特征列见 RANKER_FEATURES）。

    fit 必须用 OOF 或独立 query 组样本（含正负例与 group 标注），
    不得用随机假输入。未拟合 predict 抛错。权重保存由调用方连同
    config/split 一起落盘（见 src/train_fingerprint.save_train_state 口径）。
    """

    weights: List[float] = field(default_factory=list)
    fitted: bool = False
    n_fit_rows: int = 0

    def fit_mean_difference(self, positives: Sequence[Sequence[float]],
                            negatives: Sequence[Sequence[float]]) -> List[float]:
        """占位拟合：w = 正例均值 - 负例均值（仅用于互补性试探）。

        输入为已按 RANKER_FEATURES 对齐的特征行；空输入抛错。
        这不是最优学习排序，只是 R4 前的基线占位；正式 ranker 需另行实现。
        """
        if not positives or not negatives:
            raise ValueError("need non-empty positives and negatives")
        d = len(RANKER_FEATURES)
        for rows in (positives, negatives):
            for r in rows:
                if len(r) != d:
                    raise ValueError("feature width must be %d" % d)
        pos_m = [sum(r[j] for r in positives) / len(positives) for j in range(d)]
        neg_m = [sum(r[j] for r in negatives) / len(negatives) for j in range(d)]
        self.weights = [pos_m[j] - neg_m[j] for j in range(d)]
        self.fitted = True
        self.n_fit_rows = len(positives) + len(negatives)
        return list(self.weights)

    def predict(self, features: Sequence[float]) -> float:
        if not self.fitted or not self.weights:
            raise RuntimeError("ranker not fitted: fit on OOF/independent groups first")
        if len(features) != len(RANKER_FEATURES):
            raise ValueError("feature width must be %d" % len(RANKER_FEATURES))
        return float(sum(w * float(x) for w, x in zip(self.weights, features)))

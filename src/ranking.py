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

import math
import random as _random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

try:  # numpy 可用时用于 LogisticRanker；缺失不影响本模块导入。
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None  # type: ignore

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


# ---------------------------------------------------------------------------
# R4-local：numpy 手写 logistic 重排器 + 结构分组 CV（本地 CPU 手动研究用，
# 不是测试文件；不读凭据，不联网，不碰远端）。
#
# 背景：FPNet single（NP MRR 0.6217）在无参考谱时最强，cosine（0.7076）在有
# 独立参考谱时更强。R4 问：同候选集下简单重排器能否修正互补错误。
# 本地可重建的特征只有 cosine（Q/L 的 ms2 都在本地 JSON）、质量误差 ppm、
# adduct/instrument 一致性；逐候选 FPNet 分本地无 torch/RDKit，无法重算，
# 且已有 artifacts 只有真值名次、无全候选打分表，故本地 ranker 不用 fp_dot
#（见 run_r4_local_cv 文档与数据问题记录），结论只在相同本地候选集内比较。
# ---------------------------------------------------------------------------

# 本地 ranker 特征列（RANKER_FEATURES 去掉 fp_dot，相对顺序不变）。
LOCAL_RANKER_FEATURES: List[str] = [
    "spectral_cosine", "neg_abs_ppm", "adduct_match", "instrument_match",
]


def cosine_score_local(
    q_mz: Sequence[float], q_it: Sequence[float],
    l_mz: Sequence[float], l_it: Sequence[float],
    ppm: float = 20.0, topn: int = 128,
) -> float:
    """本地余弦特征：复刻 part2 格的 cos_score（topn/开方/L2/ppm 贪心匹配）。

    纯标准库实现，不依赖 numpy。空输入或零范数返回 0.0。
    """
    q_mz = [float(v) for v in q_mz]
    q_it = [float(v) for v in q_it]
    l_mz = [float(v) for v in l_mz]
    l_it = [float(v) for v in l_it]
    if not q_mz or not l_mz or len(q_mz) != len(q_it) or len(l_mz) != len(l_it):
        if not q_mz or not l_mz:
            return 0.0
        # 长度不一致时按最短截断，不抛错（研究特征，非门控）。
        nq = min(len(q_mz), len(q_it))
        nl = min(len(l_mz), len(l_it))
        q_mz, q_it = q_mz[:nq], q_it[:nq]
        l_mz, l_it = l_mz[:nl], l_it[:nl]
    qi = sorted(range(len(q_it)), key=lambda i: -q_it[i])[: int(topn)]
    li = sorted(range(len(l_it)), key=lambda i: -l_it[i])[: int(topn)]
    qmax = max(q_it[i] for i in qi)
    lmax = max(l_it[i] for i in li)
    if qmax <= 0.0 or lmax <= 0.0:
        return 0.0
    qv = sorted(((q_mz[i], math.sqrt(q_it[i] / qmax)) for i in qi),
                key=lambda p: p[0])
    lv = sorted(((l_mz[i], math.sqrt(l_it[i] / lmax)) for i in li),
                key=lambda p: p[0])
    qn = math.sqrt(sum(v * v for _, v in qv))
    ln = math.sqrt(sum(v * v for _, v in lv))
    if qn == 0.0 or ln == 0.0:
        return 0.0
    qv = [(m, v / qn) for m, v in qv]
    lv = [(m, v / ln) for m, v in lv]
    i = j = 0
    s = 0.0
    while i < len(qv) and j < len(lv):
        tol = max(qv[i][0], lv[j][0]) * float(ppm) * 1e-6
        d = qv[i][0] - lv[j][0]
        if abs(d) <= tol:
            s += qv[i][1] * lv[j][1]
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return float(s)


@dataclass
class LogisticRanker:
    """R4-local 点式 logistic 重排器（numpy 手写，无 sklearn 时用）。

    特征列见 LOCAL_RANKER_FEATURES（标准化后训练，与 r4_ranker.json 的
    {w,b,mu,sd} 口径兼容）。fit 必须按结构分组的训练折样本调用；
    未拟合时 score 抛错，不静默回退。
    训练协议与 part2 R4 对齐：全批量梯度、均值 BCE 梯度、lr=0.1、300 步、
    无 L2，供 train/test gap 做过拟合判断。
    """

    weights: List[float] = field(default_factory=list)
    bias: float = 0.0
    mu: List[float] = field(default_factory=list)
    sd: List[float] = field(default_factory=list)
    fitted: bool = False
    n_fit_rows: int = 0
    feature_names: List[str] = field(
        default_factory=lambda: list(LOCAL_RANKER_FEATURES))

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float],
            lr: float = 0.1, iters: int = 300) -> "LogisticRanker":
        if _np is None:
            raise RuntimeError("need numpy for LogisticRanker.fit")
        Xa = _np.asarray([[float(v) for v in r] for r in X], dtype=float)
        ya = _np.asarray([float(v) for v in y], dtype=float)
        if Xa.ndim != 2 or Xa.shape[0] == 0:
            raise ValueError("need non-empty 2D X")
        if Xa.shape[1] != len(LOCAL_RANKER_FEATURES):
            raise ValueError("feature width must be %d"
                             % len(LOCAL_RANKER_FEATURES))
        if Xa.shape[0] != ya.shape[0]:
            raise ValueError("X/y row mismatch")
        if _np.all(ya == ya[0]):
            raise ValueError("need both positive and negative rows")
        mu = Xa.mean(axis=0)
        sd = Xa.std(axis=0)
        Xs = _np.zeros_like(Xa)
        for j in range(Xa.shape[1]):
            if sd[j] < 1e-12:  # 零方差列（如 instrument_match 全 0）置零。
                Xs[:, j] = 0.0
            else:
                Xs[:, j] = (Xa[:, j] - mu[j]) / sd[j]
        d = Xa.shape[1]
        w = _np.zeros(d, dtype=float)
        b = 0.0
        for _ in range(int(iters)):
            z = Xs.dot(w) + b
            # 数值稳定的 sigmoid。
            p = _np.where(z >= 0.0, 1.0 / (1.0 + _np.exp(-z)),
                          _np.exp(z) / (1.0 + _np.exp(z)))
            e = p - ya
            gw = (Xs * e[:, None]).mean(axis=0)
            gb = float(e.mean())
            w = w - float(lr) * gw
            b = b - float(lr) * gb
        self.weights = [float(v) for v in w]
        self.bias = float(b)
        self.mu = [float(v) for v in mu]
        # sd 存原始标准差（含零方差列的真实值），调用方据此识别退化特征。
        self.sd = [float(v) for v in sd]
        self.fitted = True
        self.n_fit_rows = int(Xa.shape[0])
        return self

    def score(self, features: Sequence[float]) -> float:
        if not self.fitted or not self.weights:
            raise RuntimeError(
                "ranker not fitted: fit on structure-grouped folds first")
        if len(features) != len(LOCAL_RANKER_FEATURES):
            raise ValueError("feature width must be %d"
                             % len(LOCAL_RANKER_FEATURES))
        xs = []
        for x, m, s in zip(features, self.mu, self.sd):
            xs.append(0.0 if s < 1e-12 else (float(x) - m) / s)
        return float(sum(w * v for w, v in zip(self.weights, xs)) + self.bias)

    def to_dict(self) -> Dict:
        return {"w": list(self.weights), "b": float(self.bias),
                "mu": list(self.mu), "sd": list(self.sd),
                "features": list(LOCAL_RANKER_FEATURES),
                "n_fit_rows": int(self.n_fit_rows)}

    @classmethod
    def from_dict(cls, d: Dict) -> "LogisticRanker":
        r = cls()
        r.weights = [float(v) for v in d["w"]]
        r.bias = float(d["b"])
        r.mu = [float(v) for v in d["mu"]]
        r.sd = [float(v) for v in d["sd"]]
        r.fitted = True
        r.n_fit_rows = int(d.get("n_fit_rows", 0))
        return r


def structure_grouped_fold_of_key(
    sorted_unique_keys: Sequence[str], k: int = 5, seed: int = 2026,
) -> Dict[str, int]:
    """结构分组 K 折：seed 洗牌后轮转分配，保证同结构永不跨折。

    返回 {key: fold_id}。谱级切分禁用（同一结构的多个谱必须同折）。
    """
    keys = sorted(set(str(v) for v in sorted_unique_keys))
    rng = _random.Random(int(seed))
    order = list(keys)
    rng.shuffle(order)
    return {kk: (i % int(k)) for i, kk in enumerate(order)}


def mrr_at_25(ranks: Sequence) -> Dict[str, float]:
    """MRR@25/Top1/Top25：rank 为 None 或 >25 时 RR=0（top-25 外无分）。

    ranks 为每 query 的真值排名（1-based，None=窗外/无排名）。
    """
    n = len(list(ranks))
    if n == 0:
        return {"n": 0, "mrr@25": 0.0, "top1": 0.0, "top25": 0.0,
                "n_ranked_le25": 0}
    rr = 0.0
    top1 = 0
    top25 = 0
    for r in ranks:
        if r is not None and int(r) >= 1 and int(r) <= 25:
            rr += 1.0 / int(r)
            top25 += 1
            if int(r) == 1:
                top1 += 1
    return {"n": float(n), "mrr@25": rr / n, "top1": top1 / n,
            "top25": top25 / n, "n_ranked_le25": float(top25)}


def order_by_score(keys: Sequence[str], scores: Sequence[float]
                   ) -> List[str]:
    """按分数降序、同分按 key 确定性排序，返回有序 key 列表。"""
    idx = sorted(range(len(keys)), key=lambda i: (-float(scores[i]), keys[i]))
    return [keys[i] for i in idx]


def run_r4_local_cv(
    train_parquet: str = "data/raw/train.parquet",
    q_path: str = "artifacts/runs/004-research/np_queries_1184.json",
    l_path: str = "artifacts/runs/004-research/np_lib_spectra.json",
    r1_path: str = "artifacts/runs/004-research/r1_np1184.json",
    k: int = 5, seed: int = 2026, ppm: float = 20.0,
    lr: float = 0.1, iters: int = 300, verbose: bool = True,
) -> Dict:
    """R4 后半段本地 CV 驱动（手动研究入口，不是测试文件）。

    候选集：训练结构池（train.parquet 去重 inchikey14，分子式理论质量，
    float64 bisect 20ppm 窗），与 R1 同口径（本地窗 vs r1 win 逐谱核对）。
    特征（LOCAL_RANKER_FEATURES，可本地重建）：cosine（Q timsTOF vs L 非
    timsTOF 同结构谱取最大，复刻 part2 口径）、neg_abs_ppm、adduct_match、
    instrument_match（L 无 timsTOF，恒 0，退化特征如实报告）。
    fp_dot 缺失：本地无 torch/RDKit，artifacts 只有真值名次、无全候选打分，
    不伪造该特征；delta 只在相同本地候选集内（vs mass-only / cosine-only）
    计算，与已发表 FPNet single 0.6217 / cosine 0.7076 只作数值上下文对照。
    切分：按结构分组 K 折（同结构多谱永不跨折）；test 折预测即 OOF，
    不用验证标签调参（超参固定为 part2 R4 协议 lr=0.1/300 步/无 L2）。

    返回 report dict；verbose 时打印每折表。不写任何文件。
    """
    import bisect
    import json

    try:
        from src.candidates import (formula_mass, ppm_error,
                                    query_neutral_mass)
    except Exception:  # PYTHONPATH=src 直接跑时
        from candidates import (formula_mass, ppm_error,  # type: ignore
                                query_neutral_mass)
    import pyarrow.parquet as pq

    Q = json.load(open(q_path))
    L = json.load(open(l_path))
    R1 = {r["id"]: r for r in json.load(open(r1_path))}
    # L 按结构分组（只留标量 + ms2，供 cosine 用）。
    l_by_key: Dict[str, List[Dict]] = {}
    for row in L:
        l_by_key.setdefault(str(row["inchikey14"]), []).append(row)
    # 训练结构池质量索引（流式，只读两标量列）。
    pf = pq.ParquetFile(train_parquet)
    first_fm: Dict[str, str] = {}
    for batch in pf.iter_batches(columns=["inchikey14", "molecular_formula"],
                                 batch_size=65536):
        iks = batch.column(0).to_pylist()
        fms = batch.column(1).to_pylist()
        for kk, fm in zip(iks, fms):
            if kk is None:
                continue
            ks = str(kk)
            if ks not in first_fm:
                first_fm[ks] = "" if fm is None else str(fm)
    mass_of: Dict[str, float] = {}
    for ks, fm in first_fm.items():
        m, _note = formula_mass(fm)
        if m is not None:
            mass_of[ks] = float(m)
    pairs = sorted(((m, ks) for ks, m in mass_of.items()))
    mass_list = [m for m, _ in pairs]
    if verbose:
        print("pool: %d keys, mass_ok %d" % (len(first_fm), len(mass_of)))

    # 逐 query 建窗 + 特征。
    queries: List[Dict] = []
    win_mismatch = 0
    for qi, q in enumerate(Q):
        qk = str(q["inchikey14"])
        qm, known = query_neutral_mass(float(q["precursor_mz"]), q["adduct"])
        qm = float(qm)
        tol = abs(qm) * float(ppm) / 1e6
        a = bisect.bisect_left(mass_list, qm - tol)
        b = bisect.bisect_right(mass_list, qm + tol)
        ck = [pairs[i][1] for i in range(a, b)]
        feats: List[List[float]] = []
        for c in ck:
            err = ppm_error(mass_of[c], qm)
            rows = l_by_key.get(c)
            if rows:
                qmz = [float(v) for v in q["ms2_mzs"]]
                qit = [float(v) for v in q["ms2_normalized_intensities"]]
                best = -1.0
                bi = 0
                for ri, r in enumerate(rows):
                    s = cosine_score_local(
                        qmz, qit,
                        [float(v) for v in r["ms2_mzs"]],
                        [float(v) for v in r["ms2_normalized_intensities"]],
                        ppm=ppm)
                    if s > best:
                        best = s
                        bi = ri
                br = rows[bi]
                feats.append([float(best), -float(err),
                              1.0 if br.get("adduct") == q["adduct"] else 0.0,
                              1.0 if br.get("instrument_type")
                              == q["instrument_type"] else 0.0])
            else:
                feats.append([0.0, -float(err), 0.0, 0.0])
        truth_pos = None
        for pi, c in enumerate(ck):
            if c == qk:
                truth_pos = pi
                break
        r1 = R1.get(qi, {})
        if int(r1.get("win", -1)) != len(ck):
            win_mismatch += 1
        queries.append({"id": qi, "key": qk, "qm": qm,
                        "known_adduct": bool(known), "cands": ck,
                        "feats": feats, "truth_pos": truth_pos,
                        "r1_win": r1.get("win"), "r1_rank": r1.get("rank")})

    qkeys = sorted({q["key"] for q in queries})
    fold_of = structure_grouped_fold_of_key(qkeys, k=k, seed=seed)

    def ranks_for(order_fn) -> Dict[int, List]:
        out: Dict[int, List] = {}
        for q in queries:
            if q["truth_pos"] is None or not q["cands"]:
                out[q["id"]] = None
                continue
            ordered = order_fn(q)
            try:
                pos = ordered.index(q["key"])
                out[q["id"]] = pos + 1
            except ValueError:
                out[q["id"]] = None
        return out

    def ranks_from_scores(score_fn) -> Dict[int, List]:
        def _ord(q: Dict) -> List[str]:
            sc = [score_fn(f) for f in q["feats"]]
            return order_by_score(q["cands"], sc)
        return ranks_for(_ord)

    # OOF 容器：每折模型只排本折 test queries。
    oof_ranker: Dict[int, List] = {}
    oof_mass: Dict[int, List] = {}
    oof_cos: Dict[int, List] = {}
    per_fold: List[Dict] = []
    for f in range(int(k)):
        te = [q for q in queries if fold_of[q["key"]] == f]
        tr = [q for q in queries if fold_of[q["key"]] != f]
        Xtr: List[List[float]] = []
        ytr: List[float] = []
        for q in tr:
            for pi, feat in enumerate(q["feats"]):
                Xtr.append(feat)
                ytr.append(1.0 if pi == q["truth_pos"] else 0.0)
        model = LogisticRanker().fit(Xtr, ytr, lr=lr, iters=iters)
        te_ids = [q["id"] for q in te]
        rq = ranks_from_scores(model.score)
        rm = ranks_from_scores(lambda feat: feat[1])
        rc = ranks_from_scores(lambda feat: feat[0])
        for qid in te_ids:
            oof_ranker[qid] = rq[qid]
            oof_mass[qid] = rm[qid]
            oof_cos[qid] = rc[qid]
        m_ranker = mrr_at_25([rq[i] for i in te_ids])
        m_mass = mrr_at_25([rm[i] for i in te_ids])
        m_cos = mrr_at_25([rc[i] for i in te_ids])
        # 训练折复代（过拟合信号；同一模型排训练 queries）。
        tr_ids = [q["id"] for q in tr]
        m_train = mrr_at_25([rq[i] for i in tr_ids])
        per_fold.append({
            "fold": f, "n_struct": len({q["key"] for q in te}),
            "n_q": len(te), "n_fit_rows": model.n_fit_rows,
            "ranker": m_ranker, "mass_only": m_mass, "cosine_only": m_cos,
            "train_resub_ranker": m_train,
            "d_ranker_minus_mass": m_ranker["mrr@25"] - m_mass["mrr@25"],
            "d_ranker_minus_cos": m_ranker["mrr@25"] - m_cos["mrr@25"],
            "weights": model.to_dict(),
        })
        if verbose:
            print("fold %d: struct %d q %d rows %d | ranker MRR %.4f Top1 %.4f "
                  "| mass %.4f cos %.4f | d_mass %+.4f d_cos %+.4f | "
                  "train_resub %.4f | w %s" % (
                      f, len({q["key"] for q in te}), len(te),
                      model.n_fit_rows, m_ranker["mrr@25"], m_ranker["top1"],
                      m_mass["mrr@25"], m_cos["mrr@25"],
                      m_ranker["mrr@25"] - m_mass["mrr@25"],
                      m_ranker["mrr@25"] - m_cos["mrr@25"],
                      m_train["mrr@25"],
                      "[" + ",".join("%.3f" % v
                                     for v in model.weights) + "]"))

    all_ids = [q["id"] for q in queries]
    overall = {
        "ranker": mrr_at_25([oof_ranker[i] for i in all_ids]),
        "mass_only": mrr_at_25([oof_mass[i] for i in all_ids]),
        "cosine_only": mrr_at_25([oof_cos[i] for i in all_ids]),
    }
    report = {
        "k": int(k), "seed": int(seed), "ppm": float(ppm),
        "n_queries": len(queries), "n_structures": len(qkeys),
        "win_mismatch_vs_r1": win_mismatch,
        "n_truth_out_of_window": sum(
            1 for q in queries if q["truth_pos"] is None),
        "per_fold": per_fold,
        "overall_oof": overall,
        "d_oof_ranker_minus_mass": overall["ranker"]["mrr@25"]
        - overall["mass_only"]["mrr@25"],
        "d_oof_ranker_minus_cos": overall["ranker"]["mrr@25"]
        - overall["cosine_only"]["mrr@25"],
        "features": list(LOCAL_RANKER_FEATURES),
        "note": "fp_dot unavailable locally (no torch/RDKit; artifacts lack "
                "per-candidate scores); same-C deltas only vs local "
                "mass/cosine baselines.",
    }
    if verbose:
        print("OOF overall (n=%d): ranker MRR %.4f Top1 %.4f | mass MRR %.4f "
              "Top1 %.4f | cos MRR %.4f Top1 %.4f | d_mass %+.4f d_cos %+.4f "
              "| win_mismatch %d/%d out_of_window %d" % (
                  len(all_ids), overall["ranker"]["mrr@25"],
                  overall["ranker"]["top1"], overall["mass_only"]["mrr@25"],
                  overall["mass_only"]["top1"],
                  overall["cosine_only"]["mrr@25"],
                  overall["cosine_only"]["top1"],
                  report["d_oof_ranker_minus_mass"],
                  report["d_oof_ranker_minus_cos"],
                  win_mismatch, len(all_ids),
                  report["n_truth_out_of_window"]))
    return report

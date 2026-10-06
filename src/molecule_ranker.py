"""E2 分子级多证据 ranker（005，手动研究用模块，不是测试文件）.

背景：
- E2 在 union 完整池、single 选谱、10ppm nocap 窗下，对 250 个公开天然产物
  分子做多证据排序。产物为 artifacts/005/e2_*.json（ranker MRR 0.921143）。
- 本模块把 E2 的 CPU 可重跑部分落盘：候选窗 C(q) 构造、FP 点积打分、E1 谱库
  特征接口、分子/谱特征定义、分子组 OOF 训练与排序评估。不做 GPU/CPU 神经
  前向（复用 artifacts/005/e0_logits_single_s2_np1184.npy），不读凭据，不联网。
- 不硬编码任何分子/谱 ID；分子集合一律从输入动态读取。

C(q) 定义（本模块与 E2 产物一致）：
- 候选池为 union 712199 shipped 条目，顺序为 train 275810 拼接 coco extra
  436389（train×coco 零 shipped 重叠）。train 段下标 0..275809，coco 段下标
  275810..712198（coco 内部 pool_index + 275810）。
- 每分子只用一张 single 选谱：v7 规则“原始峰最多的一张谱”（见
  artifacts/005/e0_cq_selected.json 的 selected_index）。中性质量 qm_single
  由前体 m/z + adduct 经 candidates.query_neutral_mass（ADDUCT_TABLE，float64，
  未知加合回退 precursor 本身）求得。
- 窗为观测中性质量 ±10ppm 内的全部 union 条目（nocap，无 WMAX 截断；full
  window）。存储顺序为质量误差升序（closest-first，同误差按 pool 下标升序），
  但打分与排序与存储顺序无关（scoring order-independent）。
- shipped 窗映射到 canonical_group_key 后按 canon 去重：同一 canon 的多个
  shipped 条目取最大分（FP 取最大点积，ranker 取最大 OOF logit），再按
  (-分, canon) 排序。真值等价按 canon 相等判定（rdkit_full 本地 2026.03.6，
  官方 2026.03.3 偏差已记录，不冒称相等）。

特征顺序（固定，与 artifacts/005/e2_feature_schema.json 一致）：
- 候选级 24 维 E2_FEATURES = 11 FP 聚合 + 5 谱库 + 8 分子上下文。
  顺序为 [f_max, f_mean_spec, f_mean_cond, f_mean_max_cond, f_top2_max,
  f_top2_mean, f_var, f_var_max, f_support, f_contra, f_std,
  lib_max, lib_second, lib_gap, lib_n_spectra, lib_n_sources,
  mol_n_spec, mol_n_cond, mol_mean_n, mol_mean_ent, mol_mean_ret,
  mol_pos_frac, mol_max_rep, mol_margin]。
- 谱级 11 维 E2_SEL_FEATURES = [n_valid, retained, entropy, ce, neutral_ppm,
  is_pos, is_MH, is_MHneg, is_wl, same_cond_count, margin_spec]，
  用于谱选择/加权（single/equal/selweight 对照与 spec_oof）。
- 谱库 5 维直接复用 retrieval.E1 聚合口径（max/second/gap/n_spectra/
  n_sources）；shifted analog 的 disjoint 接口保留但 ranker 只用 direct，
  与 schema notes 一致。

OOF 规则（R4 教训已落实）：
- 按真值 canon 分子分组 5 折：ranking.structure_grouped_fold_of_key，
  seed=20261004，排序后洗牌轮转，同 canon 真值的多谱/多候选行永不跨折。
  谱级切分禁用，候选行随机拆分禁用。
- 点式 logistic（全批量、均值 BCE、无 L2、lr=0.1、300 步，与 R4 协议对齐），
  特征标准化（mu/sd，零方差列置零），test 折预测即 OOF。
- 名次一律排序后位置：对每分子候选按 (-分, canon) 排序后 enumerate(start=1)
  取真值位置（旧 R4 把排序数组里的原下标当名次的错误不再出现）。
- MRR@25/Top1/Top25 复用 ranking.mrr_at_25（rank None 或 >25 按 0 计）。

仅依赖标准库 + numpy（矩阵点积/标准化/训练）；pyarrow 仅在流式读
train.parquet 建库/检索时才需要，导入时不强制。Kaggle 离线可用。
"""

from __future__ import annotations

import bisect
import json
import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

try:  # numpy 可用时用于点积/标准化/训练；缺失不影响导入。
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None  # type: ignore

# 复用 E1 与 OOF/Rank 口径（导入失败时延迟到调用处再报错）。
try:
    from src import retrieval as _ret  # type: ignore
except Exception:
    try:
        import retrieval as _ret  # type: ignore  # PYTHONPATH=src 时
    except Exception:
        _ret = None  # type: ignore

try:
    from src import ranking as _rnk  # type: ignore
except Exception:
    try:
        import ranking as _rnk  # type: ignore
    except Exception:
        _rnk = None  # type: ignore

try:
    from src import candidates as _cnd  # type: ignore
except Exception:
    try:
        import candidates as _cnd  # type: ignore
    except Exception:
        _cnd = None  # type: ignore


# ---------------------------------------------------------------------------
# E2 常量（与 e2_feature_schema.json / e2_metrics.json 对齐）
# ---------------------------------------------------------------------------

#: 候选级 24 维特征顺序（固定，权重与此对齐）。
E2_FEATURES: List[str] = [
    "f_max",
    "f_mean_spec",
    "f_mean_cond",
    "f_mean_max_cond",
    "f_top2_max",
    "f_top2_mean",
    "f_var",
    "f_var_max",
    "f_support",
    "f_contra",
    "f_std",
    "lib_max",
    "lib_second",
    "lib_gap",
    "lib_n_spectra",
    "lib_n_sources",
    "mol_n_spec",
    "mol_n_cond",
    "mol_mean_n",
    "mol_mean_ent",
    "mol_mean_ret",
    "mol_pos_frac",
    "mol_max_rep",
    "mol_margin",
]

#: 谱级 11 维特征顺序（固定，用于谱选择/加权）。
E2_SEL_FEATURES: List[str] = [
    "n_valid",
    "retained",
    "entropy",
    "ce",
    "neutral_ppm",
    "is_pos",
    "is_MH",
    "is_MHneg",
    "is_wl",
    "same_cond_count",
    "margin_spec",
]

#: E2 主配置（与产物一致）。
E2_SEED = 20261004
E2_K = 5
E2_PPM = 10.0
E2_CQ_LABEL = "union|single|10.0"
E2_N_TRAIN = 275810
E2_N_COCO_EXTRA = 436389
E2_N_UNION = 712199
E2_N_MOL = 250
E2_TOL = 1e-6

__all__ = [
    "E2_FEATURES",
    "E2_SEL_FEATURES",
    "E2_SEED",
    "E2_K",
    "E2_PPM",
    "E2_CQ_LABEL",
    "cond_key",
    "fold_of_truth_keys",
    "order_keys_by_score",
    "rank_of_truth",
    "mrr_at_25",
    "fp_dot_from_bits",
    "compute_fp_dots_for_window",
    "collapse_shipped_to_canon_max",
    "build_spectrum_feature_row",
    "build_molecule_context",
    "build_candidate_feature_row",
    "train_logistic_oof",
    "score_logistic",
    "recompute_single_baseline",
    "verify_saved_oof",
    "run_e2_verification",
]


# ---------------------------------------------------------------------------
# 小工具：条件键 / OOF / 排序 / 指标（复用 ranking 口径）
# ---------------------------------------------------------------------------

def cond_key(ionization_mode, adduct) -> str:
    """条件键定义。

    输入：
        ionization_mode: 原字符串（如 positive/negative，可 None）。
        adduct: 原字符串（如 [M+H]+，可 None）。
    输出：
        "mode||adduct" 字符串（None 按 "" 计）。同一分子的多谱按此分组，
        再做组内 max/mean/top2 与跨组聚合，避免高重复谱数的条件主导。
    """
    mo = "" if ionization_mode is None else str(ionization_mode)
    ad = "" if adduct is None else str(adduct)
    return "%s||%s" % (mo, ad)


def fold_of_truth_keys(
    truth_canons: Sequence[str], k: int = E2_K, seed: int = E2_SEED
) -> Dict[str, int]:
    """分子组 OOF 分折。

    输入：
        truth_canons: 每分子的真值 canon 键集合（去重前可重复，本函数内去重）。
        k: 折数（E2 为 5）。
        seed: 随机种子（E2 为 20261004）。
    输出：
        {canon: fold_id}。复用 ranking.structure_grouped_fold_of_key：
        排序后 seed 洗牌再轮转，同 canon 永不跨折；谱级/候选行级随机拆分禁用。
    """
    if _rnk is None:
        raise RuntimeError("need src/ranking.py for OOF folds")
    uniq = sorted(set(str(v) for v in truth_canons))
    return _rnk.structure_grouped_fold_of_key(uniq, k=int(k), seed=int(seed))


def order_keys_by_score(
    keys: Sequence[str], scores: Sequence[float]
) -> List[str]:
    """按分数降序、同分按 key 确定性排序。

    输入：
        keys: canon 键列表；scores: 同长度分数（oof_logit 或 FP 点积）。
    输出：
        有序 key 列表。复用 ranking.order_by_score 口径；与 C(q) 存储顺序无关。
    """
    if _rnk is None:
        raise RuntimeError("need src/ranking.py for ordering")
    return _rnk.order_by_score(list(keys), [float(s) for s in scores])


def rank_of_truth(ordered_keys: Sequence[str], truth_canon: str) -> Optional[int]:
    """真值名次（排序后位置，1-based）。

    输入：
        ordered_keys: 已按 (-分, key) 排好的 canon 列表。
        truth_canon: 真值 canon 键。
    输出：
        位置（1-based）；不在列表中为 None。R4 教训：必须用排序后位置，
        不得把排序数组里的原窗口下标当名次。
    """
    t = str(truth_canon)
    for i, k in enumerate(ordered_keys):
        if str(k) == t:
            return int(i + 1)
    return None


def mrr_at_25(ranks: Sequence) -> Dict[str, float]:
    """MRR@25/Top1/Top25（复用 ranking.mrr_at_25）。

    输入：
        ranks: 每分子的真值名次（1-based，None=窗外/无排名）。
    输出：
        {n, mrr@25, top1, top25, n_ranked_le25}。rank None 或 >25 按 0 计。
    """
    if _rnk is None:
        raise RuntimeError("need src/ranking.py for metrics")
    return _rnk.mrr_at_25(list(ranks))


# ---------------------------------------------------------------------------
# FP 打分（CPU，复用 logits，不做神经前向）
# ---------------------------------------------------------------------------

def fp_dot_from_bits(logits: Sequence[float], bits: Sequence[int]) -> float:
    """单谱单候选 FP 分：sum(logit * bit)，即 f.z。

    输入：
        logits: 该谱 encoder/head 输出（6930 维，float；E2 用 orig head，
          见 e0_logits_single_s2_np1184.npy）。
        bits: 该候选 6930 位 0/1（经 fp_bits.npy 选位后；packed 需先解包）。
    输出：
        点积分数（float64 累加；sigmoid 单调不改变排序，此处直接用 logits）。
    """
    if len(logits) != len(bits):
        raise ValueError("logits/bits length mismatch %d vs %d"
                         % (len(logits), len(bits)))
    s = 0.0
    for l, b in zip(logits, bits):
        if b:
            s += float(l)
    return float(s)


def _unpack_packed_row(packed_row) -> List[int]:
    """解包单行 packed 指纹（867 字节，高位先行）为 6930 位 0/1。

    输入：
        packed_row: 长度 867 的 uint8 行（与 src/candidates.pack_bits 对齐）。
    输出：
        长度 6930 的 0/1 列表。调用方做 batch 级解包，禁止全池一次性 unpack。
    """
    if _np is not None:
        try:
            arr = _np.asarray(packed_row, dtype=_np.uint8).reshape(1, -1)
            bits = _np.unpackbits(arr, axis=1, bitorder="big")[0, :6930]
            return [int(v) for v in bits.tolist()]
        except Exception:
            pass
    # 纯标准库回退（慢，仅保证可 import/可跑通）。
    out: List[int] = []
    raw = bytes(bytearray(int(v) & 0xFF for v in list(packed_row)))
    for i in range(6930):
        out.append((raw[i // 8] >> (7 - (i % 8))) & 1)
    return out


def compute_fp_dots_for_window(
    q_logits: Sequence[float],
    packed_rows: Sequence[Sequence[int]],
) -> List[float]:
    """整窗 FP 点积（单谱 × 窗内全部 shipped 候选）。

    输入：
        q_logits: 选谱 logits（6930 维）。
        packed_rows: 窗内 shipped 候选的 packed 行（顺序与窗口存储顺序一致）。
    输出：
        与 packed_rows 等长的点积列表（float）。批量 unpack + float64 矩阵乘，
        无神经前向；调用方再按 canon 取最大并重排（见 collapse 函数）。
    """
    if _np is None:
        return [fp_dot_from_bits(q_logits, _unpack_packed_row(r))
                for r in packed_rows]
    import numpy as np  # type: ignore  # 与 _np 同一对象，仅为类型明确
    q = np.asarray(list(q_logits), dtype=np.float64)
    packs = np.asarray([list(r) for r in packed_rows], dtype=np.uint8)
    bits = np.unpackbits(packs, axis=1, bitorder="big")[:, :6930].astype(np.float64)
    dots = bits @ q
    return [float(v) for v in dots.tolist()]


def collapse_shipped_to_canon_max(
    shipped_keys: Sequence[str],
    scores: Sequence[float],
    shipped_to_canon: Dict[str, str],
) -> Tuple[List[str], List[float], Dict[str, Tuple[str, float]]]:
    """shipped 窗按 canon 取最大分并去重。

    输入：
        shipped_keys: 窗内 shipped 键（存储顺序，closest-first，但此处不依赖顺序）。
        scores: 同长度分数（FP 点积或 OOF logit）。
        shipped_to_canon: shipped -> canon 映射（identity parquet 或 OOF 表）。
    输出：
        (canon_keys, canon_scores, best_detail)，其中 canon_scores 为该 canon
        下最大分，best_detail[canon]=(取最大分的 shipped, 分)。同分后续排序按
        canon 键确定性处理，不依赖 shipped 顺序。
    """
    best: Dict[str, Tuple[str, float]] = {}
    for sk, s in zip(shipped_keys, scores):
        sks = str(sk)
        c = str(shipped_to_canon.get(sks, sks))
        sf = float(s)
        prev = best.get(c)
        if prev is None or sf > prev[1]:
            best[c] = (sks, sf)
    canons = sorted(best.keys())
    # 为确定性，canon 列表排序后由调用方再按 (-分, canon) 重排；此处返回未重排。
    return canons, [best[c][1] for c in canons], dict(best)


# ---------------------------------------------------------------------------
# 特征定义（顺序固定，见 E2_FEATURES / E2_SEL_FEATURES）
# ---------------------------------------------------------------------------

def _spectral_entropy_fallback(intensities: Sequence[float]) -> float:
    """谱熵回退（retrieval 可用时优先复用其实现）。

    输入：截断后峰强度列表；输出：-sum(p ln p)，空/全零为 0.0。
    """
    if _ret is not None:
        try:
            return float(_ret.spectral_entropy(list(intensities)))
        except Exception:
            pass
    vals = [float(v) for v in intensities]
    tot = sum(v for v in vals if v > 0.0)
    if tot <= 0.0:
        return 0.0
    s = 0.0
    for v in vals:
        if v <= 0.0:
            continue
        p = v / tot
        s -= p * math.log(p)
    return float(s)


def _ce_median_fallback(values) -> Optional[float]:
    """CE 中位数回退（retrieval 可用时优先复用）。"""
    if _ret is not None:
        try:
            return _ret.ce_median(values)
        except Exception:
            pass
    if values is None:
        return None
    try:
        arr = [float(v) for v in values]
    except TypeError:
        return None
    arr = [v for v in arr if v == v]
    if not arr:
        return None
    arr.sort()
    n = len(arr)
    mid = n // 2
    if n % 2 == 1:
        return float(arr[mid])
    return float((arr[mid - 1] + arr[mid]) / 2.0)


def _is_water_loss(adduct) -> bool:
    """失水判断（retrieval 可用时优先复用：字符串含 H2O 即失水）。"""
    if _ret is not None:
        try:
            return bool(_ret.is_water_loss_adduct(adduct))
        except Exception:
            pass
    try:
        return "H2O" in str(adduct)
    except Exception:
        return False


def build_spectrum_feature_row(
    n_valid: int,
    retained: float,
    entropy: float,
    ce: Optional[float],
    neutral_ppm: float,
    ionization_mode,
    adduct,
    same_cond_count: int,
    margin_spec: float,
) -> List[float]:
    """组装单谱 11 维特征行（顺序见 E2_SEL_FEATURES）。

    输入口径：
        n_valid: 原始峰数（len(ms2_mzs)，缺失为 0）。
        retained: 保留比例（0..1；E2 用截断后/原始峰强度比或峰数比，调用方
          按同一定义传入，本函数不重算 ms2）。
        entropy: 谱熵（截断强度的 -sum(p ln p)，自然对数）。
        ce: 碰撞能量中位数（eV；缺失传 None，本函数按 -1.0 填充并如实记录）。
        neutral_ppm: 该谱中性质量相对分子 single 选中谱中性质量的 ppm 偏差
          （|m_s-m_sel|/|m_sel|*1e6；选中谱自身为 0）。
        ionization_mode/adduct: 原字符串，用于 is_pos/is_MH/is_MHneg/is_wl。
        same_cond_count: 同分子内同条件谱数（含自身）。
        margin_spec: 该谱最佳与次佳候选 FP 分差（或 0；仅谱选择用）。
    输出：
        11 维 [n_valid, retained, entropy, ce, neutral_ppm, is_pos, is_MH,
        is_MHneg, is_wl, same_cond_count, margin_spec]（float；ce 缺失为 -1）。
    """
    ce_v = float(ce) if ce is not None else -1.0
    try:
        ce_f = float(ce_v)
        if ce_f != ce_f:  # NaN 防御
            ce_f = -1.0
    except Exception:
        ce_f = -1.0
    ad = "" if adduct is None else str(adduct)
    mo = "" if ionization_mode is None else str(ionization_mode)
    return [
        float(n_valid),
        float(retained),
        float(entropy),
        float(ce_f),
        float(neutral_ppm),
        1.0 if mo == "positive" else 0.0,
        1.0 if ad == "[M+H]+" else 0.0,
        1.0 if ad == "[M-H]-" else 0.0,
        1.0 if _is_water_loss(ad) else 0.0,
        float(same_cond_count),
        float(margin_spec),
    ]


def build_molecule_context(
    n_spec: int,
    n_cond: int,
    mean_n: float,
    mean_ent: float,
    mean_ret: float,
    pos_frac: float,
    max_rep: int,
    margin: float,
) -> List[float]:
    """组装分子上下文 8 维（顺序为 E2_FEATURES 后 8 位）。

    输入口径（同一分子的全部候选共享同一行上下文）：
        n_spec: 谱数；n_cond: 去重条件数；mean_n: 平均有效峰数；
        mean_ent: 平均谱熵；mean_ret: 平均保留比例；pos_frac: 正模式占比；
        max_rep: 最大同条件重复数；margin: 分子单谱首二分差（或 0）。
    输出：
        8 维 [mol_n_spec, mol_n_cond, mol_mean_n, mol_mean_ent, mol_mean_ret,
        mol_pos_frac, mol_max_rep, mol_margin]。
    """
    return [
        float(n_spec),
        float(n_cond),
        float(mean_n),
        float(mean_ent),
        float(mean_ret),
        float(pos_frac),
        float(max_rep),
        float(margin),
    ]


def _aggregate_fp_stats(per_spec_scores: Sequence[float]) -> Dict[str, float]:
    """单候选跨谱 FP 聚合统计（内部用，不直接对外定序）。

    输入：
        per_spec_scores: 该 canon 在各谱上的分数（每谱已按 shipped 取最大，
          顺序为分子谱顺序，与条件无关）。
    输出：
        {mx, mean, var, std, top2_mean}，空输入全 0（调用方保证非空）。
    """
    vals = [float(v) for v in per_spec_scores]
    if not vals:
        return {"mx": 0.0, "mean": 0.0, "var": 0.0, "std": 0.0, "top2_mean": 0.0}
    mx = max(vals)
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(var) if var > 0 else 0.0
    sv = sorted(vals, reverse=True)
    top2 = sv[:2]
    top2_mean = sum(top2) / len(top2)
    return {"mx": float(mx), "mean": float(mean), "var": float(var),
            "std": float(std), "top2_mean": float(top2_mean)}


def build_candidate_feature_row(
    per_spec_scores: Sequence[float],
    per_spec_conds: Sequence[str],
    lib_max: float,
    lib_second: float,
    lib_gap: float,
    lib_n_spectra: int,
    lib_n_sources: int,
    mol_context: Sequence[float],
) -> List[float]:
    """组装单候选 24 维特征行（顺序见 E2_FEATURES）。

    输入口径：
        per_spec_scores: 该 canon 在分子各谱上的 FP 分（每谱已 shipped 取最大，
          长度 = 分子谱数，顺序与谱列表一致）。
        per_spec_conds: 同长度条件键（cond_key 定义）。
        lib_*: E1 direct 聚合（retrieval.aggregate_structure_hits 口径）：
          lib_max/max_score、lib_second/second_max、lib_gap=max-second、
          lib_n_spectra/n_spectra、lib_n_sources/n_sources。无库谱候选全 0。
        mol_context: 8 维分子上下文（build_molecule_context 输出）。
    输出：
        24 维特征行。前 11 为 FP 聚合：f_max=全局最大；f_mean_spec=谱均值；
        f_mean_cond=条件均值的均值；f_mean_max_cond=条件最大值的均值；
        f_top2_max=全局 top2 均值中的最大侧（此处取全局 top2 均值，条件 top2
        信息已进条件均值）；f_top2_mean=条件 top2 均值的均值；
        f_var=谱方差；f_var_max=条件最大值向量的方差；f_support=高于谱均值
        的谱数；f_contra=均值减最小（矛盾/拖尾幅度）；f_std=谱标准差。
        中 5 为 lib_*，后 8 为 mol_context。缺失一律 0，不过载阈值。
    """
    vals = [float(v) for v in per_spec_scores]
    conds = [str(c) for c in per_spec_conds]
    if len(vals) != len(conds):
        raise ValueError("scores/conds length mismatch")
    if len(mol_context) != 8:
        raise ValueError("mol_context must be 8-dim")
    if not vals:
        f_part = [0.0] * 11
    else:
        g = _aggregate_fp_stats(vals)
        # 按条件分组。
        by_cond: Dict[str, List[float]] = defaultdict(list)
        for v, c in zip(vals, conds):
            by_cond[c].append(v)
        cond_means = [sum(v) / len(v) for v in by_cond.values()]
        cond_maxs = [max(v) for v in by_cond.values()]
        cond_top2 = []
        for v in by_cond.values():
            sv = sorted(v, reverse=True)[:2]
            cond_top2.append(sum(sv) / len(sv))
        f_max = g["mx"]
        f_mean_spec = g["mean"]
        f_mean_cond = sum(cond_means) / len(cond_means) if cond_means else 0.0
        f_mean_max_cond = sum(cond_maxs) / len(cond_maxs) if cond_maxs else 0.0
        f_top2_max = g["top2_mean"]
        f_top2_mean = sum(cond_top2) / len(cond_top2) if cond_top2 else 0.0
        f_var = g["var"]
        if len(cond_maxs) > 1:
            m = sum(cond_maxs) / len(cond_maxs)
            f_var_max = sum((v - m) ** 2 for v in cond_maxs) / len(cond_maxs)
        elif len(cond_maxs) == 1:
            f_var_max = 0.0
        else:
            f_var_max = 0.0
        f_support = float(sum(1 for v in vals if v > g["mean"]))
        f_contra = float(g["mean"] - min(vals))
        f_std = g["std"]
        f_part = [f_max, f_mean_spec, f_mean_cond, f_mean_max_cond,
                  f_top2_max, f_top2_mean, f_var, f_var_max,
                  f_support, f_contra, f_std]
    lib_part = [float(lib_max), float(lib_second), float(lib_gap),
                float(lib_n_spectra), float(lib_n_sources)]
    return [float(v) for v in (f_part + lib_part + list(mol_context))]


def e1_library_features_for_struct(
    per_struct_entry: Optional[Dict],
) -> Tuple[float, float, float, int, int]:
    """E1 聚合条目转 5 维库特征（复用 retrieval 口径）。

    输入：
        per_struct_entry: retrieval.evaluate_direct_full_background 或
          aggregate_structure_hits 的单结构条目（含 max_score/second_max/
          gap/n_spectra/n_sources），缺失（窗外/无库谱）传 None。
    输出：
        (lib_max, lib_second, lib_gap, lib_n_spectra, lib_n_sources)。
        缺失全 0；不做阈值覆盖（高 cosine 只报告用，不参与排序）。
    """
    if not per_struct_entry:
        return 0.0, 0.0, 0.0, 0, 0
    # 兼容两种键名（direct 明细 vs 聚合条目）。
    mx = per_struct_entry.get("max_score", per_struct_entry.get("score", 0.0))
    second = per_struct_entry.get("second_max", 0.0)
    gap = per_struct_entry.get("gap", float(mx) - float(second))
    return (float(mx), float(second), float(gap),
            int(per_struct_entry.get("n_spectra", 0)),
            int(per_struct_entry.get("n_sources",
                 len(per_struct_entry.get("sources_sorted", [])))))


# ---------------------------------------------------------------------------
# 点式 logistic（24 维通用版，协议与 ranking.LogisticRanker 对齐）
# ---------------------------------------------------------------------------

def _sigmoid(z: float) -> float:
    """数值稳定 sigmoid（标量）。"""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def train_logistic_oof(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    lr: float = 0.1,
    iters: int = 300,
) -> Dict:
    """点式 logistic 训练（单折，返回标准化的 w/b/mu/sd）。

    输入：
        X: 训练折特征行（n×d，E2 d=24；谱选择 d=11 时同样可用）。
        y: 0/1 标签（候选级真值等价组为 1，困难负例为 0；谱级 top1 为 1）。
        lr: 学习率（E2 与 R4 协议对齐取 0.1）。
        iters: 步数（E2 取 300，无 L2）。
    输出：
        {w, b, mu, sd, n_fit_rows}。mu/sd 为列均值/标准差（float64）；
        零方差列预测时置零（与 ranking.LogisticRanker 同口径）。
    """
    if _np is None:
        raise RuntimeError("need numpy for logistic training")
    import numpy as np  # type: ignore
    Xa = np.asarray([[float(v) for v in r] for r in X], dtype=float)
    ya = np.asarray([float(v) for v in y], dtype=float)
    if Xa.ndim != 2 or Xa.shape[0] == 0:
        raise ValueError("need non-empty 2D X")
    if Xa.shape[0] != ya.shape[0]:
        raise ValueError("X/y row mismatch")
    if np.all(ya == ya[0]):
        raise ValueError("need both positive and negative rows")
    mu = Xa.mean(axis=0)
    sd = Xa.std(axis=0)
    Xs = np.zeros_like(Xa)
    for j in range(Xa.shape[1]):
        if sd[j] < 1e-12:
            Xs[:, j] = 0.0
        else:
            Xs[:, j] = (Xa[:, j] - mu[j]) / sd[j]
    d = Xa.shape[1]
    w = np.zeros(d, dtype=float)
    b = 0.0
    for _ in range(int(iters)):
        z = Xs.dot(w) + b
        p = np.where(z >= 0.0, 1.0 / (1.0 + np.exp(-z)),
                     np.exp(z) / (1.0 + np.exp(z)))
        e = p - ya
        gw = (Xs * e[:, None]).mean(axis=0)
        gb = float(e.mean())
        w = w - float(lr) * gw
        b = b - float(lr) * gb
    return {"w": [float(v) for v in w], "b": float(b),
            "mu": [float(v) for v in mu], "sd": [float(v) for v in sd],
            "n_fit_rows": int(Xa.shape[0])}


def score_logistic(features: Sequence[float], model: Dict) -> float:
    """点式 logistic 打分（标准化后 w.x + b）。

    输入：
        features: 与训练同顺序的特征行（E2 为 24 维）。
        model: train_logistic_oof 或 e2_oof_predictions.json fold_models 条目
          （含 w/b/mu/sd）。
    输出：
        OOF logit（float；未做 sigmoid，排序等价）。
    """
    w = [float(v) for v in model["w"]]
    b = float(model["b"])
    mu = [float(v) for v in model["mu"]]
    sd = [float(v) for v in model["sd"]]
    if len(features) != len(w):
        raise ValueError("feature width must be %d" % len(w))
    s = float(b)
    for x, m, d, ww in zip(features, mu, sd, w):
        xs = 0.0 if d < 1e-12 else (float(x) - m) / d
        s += ww * xs
    return float(s)


# ---------------------------------------------------------------------------
# E2 验证：single 重算 + OOF 回放（CPU，不做神经前向）
# ---------------------------------------------------------------------------

def _load_json(path: str):
    """读 JSON（内部小工具）。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recompute_single_baseline(
    logits_path: str = "artifacts/005/e0_logits_single_s2_np1184.npy",
    windows_path: str = "artifacts/005/e0_cq_windows.npz",
    selected_path: str = "artifacts/005/e0_cq_selected.json",
    per_molecule_path: str = "artifacts/005/e2_per_molecule_top25.json",
    oof_path: str = "artifacts/005/e2_oof_predictions.json",
    train_fp_path: str = "artifacts/pool-train/fp.npy",
    coco_fp_path: str = "artifacts/005/coco/coco_fp.npy",
) -> Dict:
    """用落盘代码重算 single 基线（CPU，复用 logits，不做神经前向）。

    输入口径（均为仓库内产物路径，可相对仓库根传入）：
        logits: 1184×6930 float32（orig head，复用，不重跑前向）。
        windows: 每分子 union|single|10.0 shipped 窗（closest-first 存储）。
        selected: 每分子 single 选中谱下标 qidx 与 qm_single。
        per_molecule: 每分子 truth_canon 与 n_win（仅取真值键，不取分数）。
        oof: 每候选 shipped->canon 映射（仅取映射，不取 oof_logit）。
        train_fp/coco_fp: packed 指纹（N×867 uint8，高位先行）。
    输出：
        {per_molecule: {mol: {top25, rank, rr}}, ranks(按 mol 排序),
         metrics(mrr_at_25 口径), n_molecules}。
        top25 为 canon 去重后按 (-FP点积, canon) 排序截断 25（复用 ranking
        字母序 tie-break；落盘 single 文件对完全并列按窗内 pos 决胜，见
        run_e2_verification 备注）；rank 为排序后位置、>25 记 None（R4 口径）；
        metrics 与 e2_metrics.json single 可直接对照。
    """
    if _np is None:
        raise RuntimeError("need numpy for single recompute")
    import numpy as np  # type: ignore
    logits = np.load(logits_path, mmap_mode="r")
    windows = np.load(windows_path, allow_pickle=True)
    sel = _load_json(selected_path)
    per = _load_json(per_molecule_path)["per_molecule"]
    oof_rows = _load_json(oof_path)["oof_candidate_rows"]
    ship2canon: Dict[str, str] = {}
    for r in oof_rows:
        ship2canon[str(r["shipped"])] = str(r["canon"])
    train_fp = np.load(train_fp_path, mmap_mode="r")
    coco_fp = np.load(coco_fp_path, mmap_mode="r")
    selected_index: Dict[str, int] = sel["selected_index"]
    out: Dict[str, Dict] = {}
    for mol in sorted(per.keys()):
        truth = str(per[mol]["truth_canon"])
        qidx = int(selected_index[mol])
        w = np.asarray(windows["%s|%s" % (E2_CQ_LABEL, mol)], dtype=np.int64)
        packs = []
        shipped: List[str] = []
        # shipped 键需从映射反查：oof 表已含该窗全部 shipped，直接按 pos 顺序取。
        # 为不依赖 pool keys.npy 的字符串顺序，此处用 oof 行按 pos 排序重建。
        rows = sorted([r for r in oof_rows if str(r["mol"]) == mol],
                      key=lambda r: int(r["pos"]))
        # 防御：窗长度与 oof 行数一致（nocap 全窗）。
        if len(rows) != len(w):
            raise ValueError("window/oof row mismatch for %s: %d vs %d"
                             % (mol, len(w), len(rows)))
        shipped = [str(r["shipped"]) for r in rows]
        for idx in w:
            ii = int(idx)
            packs.append(train_fp[ii] if ii < E2_N_TRAIN else coco_fp[ii - E2_N_TRAIN])
        packs_arr = np.stack([np.asarray(p, dtype=np.uint8) for p in packs])
        bits = np.unpackbits(packs_arr, axis=1, bitorder="big")[:, :6930].astype(np.float64)
        q = np.asarray(logits[qidx], dtype=np.float64)
        dots = (bits @ q).tolist()
        canons, cscores, _ = collapse_shipped_to_canon_max(shipped, dots, ship2canon)
        ordered = order_keys_by_score(canons, cscores)
        # order_keys_by_score 已按 (-分, key) 排序；此处直接用其输出。
        # R4 口径：名次为排序后位置；窗外/>25 记 None（与 per_molecule 文件一致，
        # MRR 侧 None 与 >25 同按 0 计）。
        top25 = ordered[:25]
        _rank = rank_of_truth(ordered, truth)
        rank = None if (_rank is None or _rank > 25) else int(_rank)
        rr = (1.0 / rank) if (rank is not None and rank <= 25) else 0.0
        out[mol] = {"top25": top25, "rank": rank, "rr": float(rr),
                    "n_win_canon": len(canons)}
    ranks = [out[m]["rank"] for m in sorted(out.keys())]
    metrics = mrr_at_25(ranks)
    return {"per_molecule": out, "ranks": ranks, "metrics": metrics,
            "n_molecules": len(out)}


def verify_saved_oof(
    oof_path: str = "artifacts/005/e2_oof_predictions.json",
    per_molecule_path: str = "artifacts/005/e2_per_molecule_top25.json",
    metrics_path: str = "artifacts/005/e2_metrics.json",
    tol: float = E2_TOL,
) -> Dict:
    """回放已存 OOF logit 并核对排序与 MRR（CPU，确定性）。

    输入口径：
        oof: oof_candidate_rows 每行 {mol, pos(窗内存储位置), shipped, canon,
          fold(真值 canon 分组), oof_logit(该折 test 预测), label(1=真值等价组)}。
          pos 仅记录 closest-first 存储位置，不参与排序。
        per_molecule: 每分子 ranker top25/rank/rr（待核对）。
        metrics: 待核对的聚合指标（含 ranker/single/equal/selweight 等）。
    输出：
        {reranked: {mol: {top25, rank, rr}}, metrics_rerun, metrics_file,
         mrr_abs_diff, per_method_diff, fold_check, ok}。
        排序一律 (-oof_logit, canon)，同分按 canon（复用 ranking 口径）；
        rank 为排序后位置；MRR 经 mrr_at_25 计算；fold_check 用
        fold_of_truth_keys 重算并逐分子比对。ok 为 MRR 差 ≤ tol 且真值 rank
        与折全对（默认 1e-6，允许浮点尾差）。逐分子 top25 全序另计
        per_molecule_top25_mismatch（并列尾序不影响真值 rank/MRR，见备注），
        不计入 ok。
    备注：
        HELXLJCILKEWJH 有一对非真值候选 OOF logit 在 JSON 精度下完全相等，
        落盘排序与本口径的字母序 tie-break 在 9/10 位差一对调，真值 rank=1
        不变，MRR 差为 0。该尾序差如实报告，不掩盖。
    """
    oof = _load_json(oof_path)
    per = _load_json(per_molecule_path)["per_molecule"]
    metrics_file = _load_json(metrics_path).get("metrics", {})
    rows = oof["oof_candidate_rows"]
    by_mol: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_mol[str(r["mol"])].append(r)
    reranked: Dict[str, Dict] = {}
    for mol in sorted(by_mol.keys()):
        truth = str(per[mol]["truth_canon"])
        # 同 canon 多 shipped 取最大 logit（与 collapse 口径一致）。
        best: Dict[str, float] = {}
        for r in by_mol[mol]:
            c = str(r["canon"])
            v = float(r["oof_logit"])
            if c not in best or v > best[c]:
                best[c] = v
        canons = sorted(best.keys())
        scores = [best[c] for c in canons]
        ordered = order_keys_by_score(canons, scores)
        top25 = ordered[:25]
        _rank = rank_of_truth(ordered, truth)
        rank = None if (_rank is None or _rank > 25) else int(_rank)
        rr = (1.0 / rank) if (rank is not None and rank <= 25) else 0.0
        reranked[mol] = {"top25": top25, "rank": rank, "rr": float(rr)}
    ranks = [reranked[m]["rank"] for m in sorted(reranked.keys())]
    metrics_rerun = mrr_at_25(ranks)
    # 与文件 ranker 指针对照。
    file_ranker = metrics_file.get("ranker", {})
    diff = abs(float(metrics_rerun.get("mrr@25", 0.0))
               - float(file_ranker.get("mrr@25", 0.0)))
    # 逐分子 top25/rank 与 per_molecule 文件比对（应逐字一致）。
    per_mismatch = 0
    rank_mismatch = 0
    for mol in sorted(reranked.keys()):
        if reranked[mol]["top25"] != per[mol]["ranker"]["top25"]:
            per_mismatch += 1
        if reranked[mol]["rank"] != per[mol]["ranker"]["rank"]:
            rank_mismatch += 1
    # 分折核对：用落盘代码重算分子组折。
    truths = [str(per[m]["truth_canon"]) for m in per]
    fold_re = fold_of_truth_keys(truths, k=E2_K, seed=E2_SEED)
    fold_mismatch = 0
    for mol in per:
        if int(fold_re[str(per[mol]["truth_canon"])]) != int(per[mol]["fold"]):
            fold_mismatch += 1
    # 其他基线（single/equal/selweight）直接用 per_molecule 文件经同一 MRR
    # 口径重算，确认聚合一致（排序本身已在产物中，此处只核指标口径）。
    per_method_diff: Dict[str, float] = {}
    for method in ("single", "equal", "selweight", "ranker"):
        rs = [per[m][method]["rank"] for m in sorted(per.keys())]
        m = mrr_at_25(rs)
        f = metrics_file.get(method, {})
        per_method_diff[method] = abs(float(m.get("mrr@25", 0.0))
                                      - float(f.get("mrr@25", 0.0)))
    ok = bool(diff <= float(tol) and rank_mismatch == 0
              and fold_mismatch == 0)
    return {"reranked": reranked, "metrics_rerun": metrics_rerun,
            "metrics_file": file_ranker, "mrr_abs_diff": float(diff),
            "per_molecule_top25_mismatch": int(per_mismatch),
            "per_molecule_rank_mismatch": int(rank_mismatch),
            "fold_mismatch": int(fold_mismatch),
            "per_method_diff": per_method_diff, "tol": float(tol), "ok": bool(ok)}


def run_e2_verification(
    logits_path: str = "artifacts/005/e0_logits_single_s2_np1184.npy",
    windows_path: str = "artifacts/005/e0_cq_windows.npz",
    selected_path: str = "artifacts/005/e0_cq_selected.json",
    per_molecule_path: str = "artifacts/005/e2_per_molecule_top25.json",
    oof_path: str = "artifacts/005/e2_oof_predictions.json",
    metrics_path: str = "artifacts/005/e2_metrics.json",
    train_fp_path: str = "artifacts/pool-train/fp.npy",
    coco_fp_path: str = "artifacts/005/coco/coco_fp.npy",
    tol: float = E2_TOL,
) -> Dict:
    """E2 落盘代码重跑验证入口（CPU，不做神经前向）。

    输入：
        各产物路径（默认仓库相对路径，调用方在仓库根下运行）。
        tol: MRR 允许差（默认 1e-6，允许浮点尾差）。
    输出：
        {single: {rerun, file, diff}, ranker: {rerun, file, diff},
         fold_mismatch, per_molecule_mismatch, ok}。
        single 为 FP 点积全窗重算（(-分, canon) 字母序 tie-break，R4 位置口径）；
        ranker 为已存 OOF logit 回放排序（同口径）。两者 MRR 均经 mrr_at_25
        计算，与 e2_metrics.json 对照。ok 只看 ranker 主指标（E2 的 0.921143）
        是否 ≤ tol；single 的 tie-break 差另行报告解释（本函数只报告差值，
        不掩盖）。
    备注：
        single 有两处并列 tie（XOJVHLIYNSOZOO 真值与 AEMJ 同 516.78…，
        HELXLJCILKEWJH 两干扰同 -1751.98…）：落盘文件按窗内存储顺序（pos）
        决胜，本口径按 canon 字母序决胜。前者使真值 rank 3→4、MRR 差
        -1/3/250≈-0.000333；后者只换非真值尾序，真值 rank 不变。
    """
    single_re = recompute_single_baseline(
        logits_path=logits_path,
        windows_path=windows_path,
        selected_path=selected_path,
        per_molecule_path=per_molecule_path,
        oof_path=oof_path,
        train_fp_path=train_fp_path,
        coco_fp_path=coco_fp_path,
    )
    oof_rep = verify_saved_oof(
        oof_path=oof_path,
        per_molecule_path=per_molecule_path,
        metrics_path=metrics_path,
        tol=tol,
    )
    metrics_file = _load_json(metrics_path).get("metrics", {})
    single_file = metrics_file.get("single", {})
    single_diff = abs(float(single_re["metrics"].get("mrr@25", 0.0))
                      - float(single_file.get("mrr@25", 0.0)))
    ranker_diff = float(oof_rep["mrr_abs_diff"])
    # single 逐分子对照（top25 全序与真值 rank，>25 已记 None）。
    _per_file = _load_json(per_molecule_path)["per_molecule"]
    _s_top_mis = sum(
        1 for mol in _per_file
        if single_re["per_molecule"][mol]["top25"]
        != _per_file[mol]["single"]["top25"])
    _s_rank_mis = sum(
        1 for mol in _per_file
        if single_re["per_molecule"][mol]["rank"]
        != _per_file[mol]["single"]["rank"])
    # ok 只看 ranker 主指标；single 差为已知 tie-break 差异，另行解释，不门控。
    ok = bool(oof_rep["ok"] and ranker_diff <= float(tol))
    return {
        "Cq": E2_CQ_LABEL,
        "seed": E2_SEED,
        "k": E2_K,
        "tol": float(tol),
        "single": {"rerun": single_re["metrics"], "file": single_file,
                   "mrr_abs_diff": float(single_diff),
                   "per_molecule_top25_mismatch": int(_s_top_mis),
                   "per_molecule_rank_mismatch": int(_s_rank_mis)},
        "ranker": {"rerun": oof_rep["metrics_rerun"],
                   "file": oof_rep["metrics_file"],
                   "mrr_abs_diff": float(ranker_diff)},
        "ranker_per_molecule_top25_mismatch":
            int(oof_rep["per_molecule_top25_mismatch"]),
        "ranker_per_molecule_rank_mismatch":
            int(oof_rep["per_molecule_rank_mismatch"]),
        "fold_mismatch": int(oof_rep["fold_mismatch"]),
        "per_method_diff": oof_rep["per_method_diff"],
        "ok": bool(ok),
    }


# ---------------------------------------------------------------------------
# F2：统一历史对齐的 FP 聚合 + 参考缺失残差 ranker（006 计划 F2，手动研究用）。
#
# caliber 标注：F2_CALIBER = "f2_canonical_history_aligned_v1"。
# 本节只增补新函数，不改动上方任何 E2 旧函数/常量/行为。旧
# build_candidate_feature_row / train_logistic_oof / verify_saved_oof 保持原样，
# 以保证 e2_metrics.json 回放口径不变。
#
# 1) 特征对齐结论（以 artifacts/005/e2_feature_schema.json 的 24 个名字为起点，
#    对照 data/references/submission-v8-audit-2026-10-05.json 中历史临时脚本
#    /tmp/e2_run2.py 的自有代码片段）：
#    - f_top2_max：历史 = 各条件 top2 均分的最大值（cond_top2.max）；
#      后补 = 全局谱 top2 均值。F2 采用历史定义（条件结构保留）。
#    - f_var：历史 = 条件均分向量的方差（cond_mean.var）；
#      后补 = 全部谱分数方差。F2 采用历史定义。
#    - f_support：历史 = 该候选进入“条件 top25”的条件数（按 cond_mean 逐条件
#      取前 25 累加）；后补 = 该候选谱分数高于自身谱均值的谱数。F2 采用历史定义。
#    - f_contra：历史 = 最大谱分数 − 最小条件均分（cross_max − cross_min）；
#      后补 = 谱均值 − 最小谱分数。F2 采用历史定义。
#    - f_var_max：两者一致（条件最大值向量的方差），直接沿用。
#    - f_top2_mean / f_max / f_mean_spec / f_mean_cond / f_mean_max_cond /
#      f_std：两者一致（命名相同、口径相同），直接沿用。
#    - 样本权重：历史 = 每分子总权重 1，正负各 0.5（正例有多 shipped 别名时
#      均分 0.5，负例均分 0.5）；后补 = 全部候选行未加权平均 BCE。
#      F2 训练一律每分子等权（pairwise 下每分子全部正负对总权重为 1；
#      pointwise 权重函数保留历史 0.5/0.5 口径供对照）。
#    选择历史定义的理由：它是实际产生 e2_metrics.json（ranker 0.9211）的训练
#    分布；后补定义丢失了条件分组结构，与保存权重不兼容。本节新函数实现历史
#    定义并显式标注 caliber，不把旧 fold 权重接到新特征上。
#
# 2) 参考库移除一律在 canonical 结构组层（含别名）执行：移除某 canonical 组 =
#    该组全部 shipped 别名的全部库谱同时排除，计数（n_spectra/n_sources）与检索
#    聚合在排除后重算。旧 raw key/谱 hash 路径不再用于 F2 训练与评估。
#
# 3) 残差模型：score(q,c) = base_z(q,c) + residual(x(q,c))，其中 base_z 为同
#    分子窗内标准化的 FP single 点积（与原始 base 单调一致、只为量级对齐），
#    residual 为带显式交互的正则线性函数（numpy 实现，不依赖 LightGBM /
#    sklearn）。mol_context 等同 query 常量只以交互列进入模型，点式线性不再
#    直接使用它们；pairwise 差分中纯常量列自动为 0，交互列保留。
#    缺失走明确缺失路径：库特征缺失时置 0 并立起 has_* 指示列，不用阈值覆盖。
# ---------------------------------------------------------------------------

#: F2 统一口径标识（写入产物，供核对）。
F2_CALIBER = "f2_canonical_history_aligned_v1"

#: F2 训练 OOF 折（与 E2 同 seed/同 K，保证分子分组可比）。
F2_SEED = 20261004
F2_K = 5

#: F2 残差特征顺序（固定，权重与此对齐）。缺失一律 0 + has_* 指示列。
F2_RESIDUAL_FEATURES: List[str] = [
    "has_lib",            # 0/1：掩码后该候选是否还有库谱
    "has_fwd",            # 0/1：该分子是否有 forward 支持（GLACIER 覆盖）
    "lib_max",            # 直接余弦结构 max（缺失 0）
    "lib_gap",            # max - second（缺失 0）
    "lib_n_spectra",      # 排除后重算的库谱数（缺失 0）
    "lib_n_sources",      # 排除后重算的来源数（缺失 0）
    "best_n_matched",     # 最佳库谱匹配峰数（缺失 0）
    "best_explained",     # 最佳库谱解释强度 explained_query（缺失 0）
    "best_ent_sim",       # 最佳库谱熵相似度（缺失 0）
    "best_same_adduct",   # 最佳库谱是否同 adduct（缺失 0）
    "ce_missing",         # 0/1：最佳库谱 CE 差缺失
    "neg_ce_diff",        # -|CE 差|（越接近 0 越好；缺失 0）
    "ana_max",            # 同结构 shifted disjoint 最大分（缺失 0）
    "ana_explained",      # 直接+偏移不重叠解释强度（缺失 0）
    "base_gap_to_top",    # base_z 分子内最大值 − 该候选 base_z（leader 为 0）
    "lib_leader_gap",     # 分子内最大 lib_max − 该候选 lib_max（leader 为 0）
    "mol_n_spec",         # 分子谱数（常量，只经交互列起作用）
    "mol_pos_frac",       # 分子正模式占比（常量，只经交互列起作用）
    "mol_margin",         # 分子 base 首二分差（常量，只经交互列起作用）
    "base_z",             # 分子内标准化 base（允许模型学习 base 缩放）
    "has_lib_x_base",     # has_lib * base_z（按可用性调整 FP 权重）
    "has_lib_x_libmax",   # has_lib * lib_max（缺失时库证据归零）
    "has_fwd_x_base",     # has_fwd * base_z（按 forward 可用性调整 FP 权重）
    "mol_nspec_x_libmax",  # mol_n_spec * lib_max（谱数×库证据交互）
]

__all__ = __all__ + [
    "F2_CALIBER",
    "F2_SEED",
    "F2_K",
    "F2_RESIDUAL_FEATURES",
    "F2_ALIGNMENT_NOTES",
    "f2_molecule_equal_weights",
    "f2_fp_aggregates_for_molecule",
    "build_f2_residual_row",
    "apply_canonical_library_dropout",
    "train_pairwise_residual",
    "pairwise_residual_score",
]

#: 对齐说明表（历史定义 vs 后补定义 vs F2 选择）。
F2_ALIGNMENT_NOTES: Dict[str, Dict[str, str]] = {
    "f_top2_max": {
        "historical": "max over conds of cond_top2 mean (/tmp/e2_run2.py:454-482)",
        "src_backfill": "global top2 mean over spectra (molecule_ranker 581-585)",
        "f2": "historical",
    },
    "f_var": {
        "historical": "var of cond_mean vector",
        "src_backfill": "var over all spectra",
        "f2": "historical",
    },
    "f_support": {
        "historical": "n_conds where candidate in cond top25 by cond_mean",
        "src_backfill": "n_specs where score above own global mean",
        "f2": "historical",
    },
    "f_contra": {
        "historical": "cross_max - min(cond_mean)",
        "src_backfill": "global mean - min(spectra scores)",
        "f2": "historical",
    },
    "f_var_max": {
        "historical": "var of cond_max vector",
        "src_backfill": "var of cond_max vector",
        "f2": "same",
    },
    "sample_weights": {
        "historical": "per molecule total 1.0, 0.5 pos / 0.5 neg; aliases split 0.5",
        "src_backfill": "unweighted mean BCE over rows",
        "f2": "molecule-equal (pairwise pairs total 1.0 per molecule)",
    },
}


def f2_molecule_equal_weights(
    n_pos: int, n_neg: int, pos_total: float = 0.5, neg_total: float = 0.5
) -> Tuple[float, float]:
    """历史 0.5/0.5 分子等权（pointwise 对照用）。

    输入：
        n_pos: 该分子正例行数（canon 合并后一般为 1；合并前多 shipped 别名
          按实际行数均分 pos_total）。
        n_neg: 该分子负例行数。
    输出：
        (w_pos_each, w_neg_each)。每分子总权重恒为 pos_total + neg_total。
    """
    if n_pos <= 0 or n_neg <= 0:
        raise ValueError("need both positive and negative rows")
    return float(pos_total) / int(n_pos), float(neg_total) / int(n_neg)


def f2_fp_aggregates_for_molecule(
    S, cond_ids: Sequence,
) -> Dict[int, Dict[str, float]]:
    """历史对齐的 FP 聚合（逐分子矩阵版，caliber 见 F2_CALIBER）。

    输入：
        S: n_spec × n_win 矩阵（行=该分子各谱，列=窗内 shipped 候选，值为 FP
          点积；复用 E0 logits，不做神经前向）。
        cond_ids: 长度 n_spec 的条件 id 列表（cond_key 口径，同一条件多谱
          归为一组）。
    输出：
        {j: {f_max, f_mean_spec, f_mean_cond, f_mean_max_cond, f_top2_max,
        f_top2_mean, f_var, f_var_max, f_support, f_contra, f_std}}，
        定义与 /tmp/e2_run2.py:454-482 一致（跨条件聚合，见本节表头注释）。
        单谱/单条件时方差类为 0；support 为进入条件 top25 的条件数。
    """
    if _np is None:
        raise RuntimeError("need numpy for F2 FP aggregates")
    import numpy as np  # type: ignore
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] == 0 or S.shape[1] == 0:
        raise ValueError("need non-empty 2D S")
    n_spec, n_win = int(S.shape[0]), int(S.shape[1])
    cond_ids = [str(c) for c in cond_ids]
    if len(cond_ids) != n_spec:
        raise ValueError("cond_ids length must match n_spec")
    uniq_conds = sorted(set(cond_ids))
    cond_index = {c: [i for i, v in enumerate(cond_ids) if v == c]
                  for c in uniq_conds}
    n_cond = len(uniq_conds)
    # 每条件子矩阵：n_rep × n_win。
    cond_mean = np.stack(
        [S[ix, :].mean(axis=0) for ix in
         [cond_index[c] for c in uniq_conds]], axis=0)
    cond_max = np.stack(
        [S[ix, :].max(axis=0) for ix in
         [cond_index[c] for c in uniq_conds]], axis=0)
    cond_top2 = np.stack(
        [np.sort(S[ix, :], axis=0)[-2:, :].mean(axis=0)
         if len(ix) >= 2 else S[ix, :].mean(axis=0)
         for ix in [cond_index[c] for c in uniq_conds]], axis=0)
    cross_max = cond_max.max(axis=0)
    cross_mean = cond_mean.mean(axis=0)
    cross_mean_max = cond_max.mean(axis=0)
    cross_top2_max = cond_top2.max(axis=0)
    cross_top2_mean = cond_top2.mean(axis=0)
    cross_var = (cond_mean.var(axis=0) if n_cond > 1
                 else np.zeros(n_win))
    cross_var_max = (cond_max.var(axis=0) if n_cond > 1
                     else np.zeros(n_win))
    cross_min = cond_mean.min(axis=0)
    contra = cross_max - cross_min
    std_spec = S.std(axis=0) if n_spec > 1 else np.zeros(n_win)
    f_mean_spec = S.mean(axis=0)
    support = np.zeros(n_win, dtype=float)
    for ci in range(n_cond):
        order = np.argsort(-cond_mean[ci], kind="stable")[:25]
        support[order] += 1.0
    out: Dict[int, Dict[str, float]] = {}
    for j in range(n_win):
        out[int(j)] = {
            "f_max": float(cross_max[j]),
            "f_mean_spec": float(f_mean_spec[j]),
            "f_mean_cond": float(cross_mean[j]),
            "f_mean_max_cond": float(cross_mean_max[j]),
            "f_top2_max": float(cross_top2_max[j]),
            "f_top2_mean": float(cross_top2_mean[j]),
            "f_var": float(cross_var[j]),
            "f_var_max": float(cross_var_max[j]),
            "f_support": float(support[j]),
            "f_contra": float(contra[j]),
            "f_std": float(std_spec[j]),
        }
    return out


def build_f2_residual_row(
    base_z: float,
    base_gap_to_top: float,
    lib_leader_gap: float,
    lib_max: float,
    lib_second: float,
    lib_n_spectra: int,
    lib_n_sources: int,
    best_n_matched: int,
    best_explained: float,
    best_ent_sim: float,
    best_same_adduct: bool,
    best_ce_diff,
    ana_max: float,
    ana_explained: float,
    has_fwd: bool,
    mol_n_spec: int,
    mol_pos_frac: float,
    mol_margin: float,
    has_lib_override=None,
) -> List[float]:
    """组装单候选 F2 残差特征行（顺序见 F2_RESIDUAL_FEATURES）。

    缺失路径（明确，不阈值覆盖）：当候选的 canonical 组被移除（或本来就没有
    库谱）时，lib_*/best_*/ana_* 一律取 0，has_lib=0；ce 缺失时
    ce_missing=1 且 neg_ce_diff=0。调用方负责在 canonical 组层掩码后传入
    已重算的计数与聚合值，本函数只做行组装与交互列展开。
    """
    n_spec = int(lib_n_spectra)
    if has_lib_override is None:
        has_lib = 1.0 if n_spec > 0 else 0.0
    else:
        has_lib = 1.0 if has_lib_override else 0.0
    if has_lib <= 0.0:
        lib_max_f = 0.0
        lib_gap_f = 0.0
        n_sources_f = 0.0
        n_matched_f = 0.0
        expl_f = 0.0
        ent_f = 0.0
        same_ad_f = 0.0
        ana_max_f = 0.0
        ana_expl_f = 0.0
    else:
        lib_max_f = float(lib_max)
        lib_gap_f = float(lib_max) - float(lib_second)
        n_sources_f = float(lib_n_sources)
        n_matched_f = float(best_n_matched)
        expl_f = float(best_explained)
        ent_f = float(best_ent_sim)
        same_ad_f = 1.0 if best_same_adduct else 0.0
        ana_max_f = float(ana_max)
        ana_expl_f = float(ana_explained)
    if best_ce_diff is None:
        try:
            if best_ce_diff is None:
                raise TypeError
        except TypeError:
            pass
        ce_missing = 1.0
        neg_ce = 0.0
    else:
        try:
            ce_f = float(best_ce_diff)
            if ce_f != ce_f:  # NaN
                ce_missing = 1.0
                neg_ce = 0.0
            else:
                ce_missing = 0.0 if has_lib > 0.0 else 1.0
                neg_ce = -abs(ce_f) if has_lib > 0.0 else 0.0
        except Exception:
            ce_missing = 1.0
            neg_ce = 0.0
    if has_lib <= 0.0:
        ce_missing = 1.0
        neg_ce = 0.0
    has_fwd_f = 1.0 if has_fwd else 0.0
    bz = float(base_z)
    row = [
        float(has_lib),
        float(has_fwd_f),
        float(lib_max_f),
        float(lib_gap_f),
        float(n_spec),
        float(n_sources_f),
        float(n_matched_f),
        float(expl_f),
        float(ent_f),
        float(same_ad_f),
        float(ce_missing),
        float(neg_ce),
        float(ana_max_f),
        float(ana_expl_f),
        float(base_gap_to_top),
        float(lib_leader_gap),
        float(mol_n_spec),
        float(mol_pos_frac),
        float(mol_margin),
        float(bz),
        float(has_lib * bz),
        float(has_lib * lib_max_f),
        float(has_fwd_f * bz),
        float(float(mol_n_spec) * lib_max_f),
    ]
    if len(row) != len(F2_RESIDUAL_FEATURES):
        raise ValueError("F2 residual width must be %d" % len(F2_RESIDUAL_FEATURES))
    return [float(v) for v in row]


def apply_canonical_library_dropout(
    cand_canons: Sequence[str],
    removed_canons: Sequence[str],
) -> List[bool]:
    """canonical 组掩码：candidate 的 canon 在移除集合中即无库证据。

    输入：
        cand_canons: 窗内候选 canon 列表（已去重顺序）。
        removed_canons: 被移除的 canonical 组集合（含别名归一后的键）。
    输出：
        与候选等长的 keep 列表（True=保留库证据，False=走缺失路径）。
    """
    removed = set(str(v) for v in removed_canons)
    return [str(c) not in removed for c in cand_canons]


def train_pairwise_residual(
    X: Sequence[Sequence[float]],
    pair_index: Sequence[Tuple[int, int]],
    pair_weight: Sequence[float],
    base_diff: Sequence[float],
    l2: float = 1.0,
    lr: float = 0.5,
    iters: int = 500,
) -> Dict:
    """带交互的正则线性 pairwise 训练（Bradley-Terry，numpy 全批量）。

    模型：P(pos 胜) = sigmoid(base_diff + w.(x_pos - x_neg))，其中 base_diff
    为固定底座差（z 单位，不学习），w 为待学习残差权重，L2 系数为 l2。
    输入：
        X: 全部候选残差特征行（n×d，需与 F2_RESIDUAL_FEATURES 同顺序，
          未标准化；本函数内按训练行计算 mu/sd 并标准化）。
        pair_index: [(i_pos, i_neg)]，同分子内正对负下标（canon 合并后每分子
          一般为 1 个正例对全部负例）。
        pair_weight: 每对权重（调用方按每分子等权：该分子对数分之一，使每
          分子总权重为 1）。
        base_diff: 每对的底座差（base_z_pos − base_z_neg，固定偏移）。
        l2/lr/iters: 正则/学习率/步数（F2 固定 l2=1.0、lr=0.5、iters=500，
          不按验证标签调参）。
    输出：
        {w, mu, sd, n_pairs, n_molecules}（无 bias 项，pairwise 差分中偏置
        抵消；预测残差 = w.x_std）。
    """
    if _np is None:
        raise RuntimeError("need numpy for pairwise residual training")
    import numpy as np  # type: ignore
    Xa = np.asarray([[float(v) for v in r] for r in X], dtype=float)
    if Xa.ndim != 2 or Xa.shape[0] == 0:
        raise ValueError("need non-empty 2D X")
    pairs = [(int(a), int(b)) for a, b in pair_index]
    if not pairs:
        raise ValueError("need non-empty pairs")
    pw = np.asarray([float(v) for v in pair_weight], dtype=float)
    bd = np.asarray([float(v) for v in base_diff], dtype=float)
    if not (len(pairs) == pw.shape[0] == bd.shape[0]):
        raise ValueError("pairs/weights/base_diff length mismatch")
    mu = Xa.mean(axis=0)
    sd = Xa.std(axis=0)
    Xs = np.zeros_like(Xa)
    for j in range(Xa.shape[1]):
        if sd[j] < 1e-12:
            Xs[:, j] = 0.0
        else:
            Xs[:, j] = (Xa[:, j] - mu[j]) / sd[j]
    d = Xa.shape[1]
    w = np.zeros(d, dtype=float)
    s = float(pw.sum())
    if s <= 0.0:
        raise ValueError("pair weights must sum positive")
    for _ in range(int(iters)):
        D = np.asarray([Xs[a] - Xs[b] for a, b in pairs])
        z = D.dot(w) + bd
        p = np.where(z >= 0.0, 1.0 / (1.0 + np.exp(-z)),
                     np.exp(z) / (1.0 + np.exp(z)))
        e = (p - 1.0) * (pw / s)  # dL/dz，加权平均
        gw = D.T.dot(e) + float(l2) * w
        w = w - float(lr) * gw
    return {"w": [float(v) for v in w],
            "mu": [float(v) for v in mu],
            "sd": [float(v) for v in sd],
            "n_pairs": int(len(pairs)),
            "l2": float(l2), "lr": float(lr), "iters": int(iters),
            "caliber": F2_CALIBER}


def pairwise_residual_score(
    features: Sequence[float], model: Dict
) -> float:
    """pairwise 残差打分（标准化后 w.x，无偏置）。

    输入：
        features: 与 F2_RESIDUAL_FEATURES 同顺序的特征行。
        model: train_pairwise_residual 产物（含 w/mu/sd）。
    输出：
        残差值（z 单位，与 base_z 直接相加）。
    """
    w = [float(v) for v in model["w"]]
    mu = [float(v) for v in model["mu"]]
    sd = [float(v) for v in model["sd"]]
    if len(features) != len(w):
        raise ValueError("feature width must be %d" % len(w))
    s = 0.0
    for x, m, dd, ww in zip(features, mu, sd, w):
        xs = 0.0 if dd < 1e-12 else (float(x) - m) / dd
        s += ww * xs
    return float(s)

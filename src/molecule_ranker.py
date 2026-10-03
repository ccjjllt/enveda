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

"""001 B0 检索:adduct 质量过滤 + 谱匹配 + 按分子聚合.

手动研究用模块,不是测试文件.

B0 实验问题:质量/adduct 过滤 + 精确谱库/余弦匹配,对“已有参考谱”分子的
识别能力与候选覆盖上限(见 docs/plan/001-baseline.md).

关键评估边界(注释写清,实现不负责强制):
- 同谱必须剔除:query 谱自身及同次测量的重复拷贝不得作为自己的候选证据.
- 未见结构轨道要求整组留出:同一规范化结构的所有谱 + 跨来源重复归为一组,
  训练/建库时整组留出.分组与 split manifest 由调用方(后续 split 导出步骤)
  负责,本模块只接受“已过滤好的谱库”作为输入,不在这里硬编码任何 ID.
- 本文件不硬编码当前 400 个可见 test ID;分子集合一律从输入数据动态读取.
  可见 test 是占位集,不能把其上的命中率当泛化效果.

匹配键占位说明(重要):
- 当前候选 SMILES 匹配键为精确字符串相等(见 src/evaluation.py
  placeholder_match),RDKit 未接入,无 InChIKey14 规范化去重.
  本模块分数仅流程自检,不作为泛化结论.

参考(复用对象,非本项目结果):
- prvsiyan/analog-propagation v32(FPNet d=512/layers=6/heads=8/MAX_PEAKS=128,
  BCE+排序损失,merge_p=0.6)是 B1 复用对象,本 B0 不复制其权重与训练逻辑.
- fp-models-v2 v1(CC0,两文件各约 144MB)是 T1 首选,本轮不下载.

仅依赖标准库(无 pyarrow/numpy/RDKit 需求),Kaggle 离线可用.
精确质量一律 float64(Python float),不提前转 float32/BF16.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

# B0 占位候选:与 sample_submission.csv 的占位语义对齐(每行 25 个 "CCO").
# 这只是让 submission.csv 格式合法的占位符,不代表任何模型效果.
PLACEHOLDER_SMILES = "CCO"
MAX_CANDIDATES = 25

# B0 默认参数(调用方可覆盖,b0_run.py 会把实际取值写入 metrics).
DEFAULT_PPM_WINDOW = 20.0
DEFAULT_TOP_N = 128
DEFAULT_FRAG_PPM = 20.0
DEFAULT_WINDOW_MODE = "global_topn"  # 二选一:"global_topn" | "window_50da"
WINDOW_50DA_WIDTH = 50.0

# 精确质量常量(float64,单同位素):
# H/C/N/O/Na/K/Cl35 取 IUPAC 单同位素质量;electron/proton 取 CODATA.
# 加合 delta 定义:观测 m/z = (n * M + delta) / |charge|,
#   中性质量 M = (precursor_mz * |charge| - delta) / n.
_PROTON = 1.00727646688
_ELECTRON = 0.00054857990946
_H = 1.00782503223
_C = 12.0
_N = 14.0030740052
_O = 15.9949146221
_NA = 22.9897692809
_K = 38.9637074864
_CL35 = 34.968852682
_H2O = 2.0 * _H + _O  # 18.01056468656
_FA_CH2O2 = _C + 2.0 * _H + 2.0 * _O  # 46.00547930866
_ACETATE_C2H4O2 = 2.0 * _C + 4.0 * _H + 2.0 * _O  # 60.02112937312

# adduct -> (multimer_n, charge, delta).覆盖可见 7 类 + train top adduct.
# 可见 7 类:test.parquet 实测 [M+H]+/[M-H]-/[M+CH2O2-H]-/[M+Na]+/[M+NH4]+/[M+K]+/[M+Cl]-.
# train top 补充:[2M+H]+/[2M+Na]+/[2M-H]-/[M-H2O+H]+/[M]+/[M+C2H4O2-H]- 等.
# 未列出的约 110 种稀有加合走未知回退(见 adduct_to_neutral_mass).
ADDUCT_TABLE: Dict[str, Tuple[int, int, float]] = {
    "[M+H]+": (1, 1, _PROTON),
    "[M-H]-": (1, -1, -_PROTON),
    "[M+Na]+": (1, 1, _NA - _ELECTRON),
    "[M+NH4]+": (1, 1, _N + 4.0 * _H - _ELECTRON),
    "[M+K]+": (1, 1, _K - _ELECTRON),
    "[M+Cl]-": (1, -1, _CL35 + _ELECTRON),
    "[M+CH2O2-H]-": (1, -1, _FA_CH2O2 - _PROTON),
    "[2M+H]+": (2, 1, _PROTON),
    "[2M+Na]+": (2, 1, _NA - _ELECTRON),
    "[2M-H]-": (2, -1, -_PROTON),
    "[M-H2O+H]+": (1, 1, -_H2O + _PROTON),
    "[M]+": (1, 1, -_ELECTRON),
    "[M+C2H4O2-H]-": (1, -1, _ACETATE_C2H4O2 - _PROTON),
    "[M-2H2O+H]+": (1, 1, -2.0 * _H2O + _PROTON),
    "[M+2H]2+": (1, 2, 2.0 * _PROTON),
}


def adduct_is_known(adduct: str) -> bool:
    """adduct 是否在质量表内."""
    return adduct in ADDUCT_TABLE


def adduct_to_neutral_mass(
    precursor_mz: float, adduct: str, ppm_window: float = DEFAULT_PPM_WINDOW
) -> Tuple[float, float]:
    """由前体 m/z + adduct 计算中性质量(真实实现,float64).

    参数:
        precursor_mz: 实测前体 m/z(按 float64 传入,不做精度截断).
        adduct: 如 "[M+H]+" 等.未知加合走回退:直接返回 precursor_mz 本身,
          调用方应经 adduct_is_known() 统计 unknown_adduct 比例,本函数不抛错.
        ppm_window: 下游候选过滤的 ppm 窗口(参数化,默认 20,透传返回).

    返回:
        (neutral_mass, ppm_window).公式 M = (mz*|z| - delta)/n.

    调用方保证:library 输入已按 manifest 过滤(同谱/整组留出在外层完成).
    """
    info = ADDUCT_TABLE.get(adduct)
    if info is None:
        return float(precursor_mz), float(ppm_window)
    n, charge, delta = info
    neutral = (float(precursor_mz) * abs(int(charge)) - float(delta)) / float(n)
    return float(neutral), float(ppm_window)


def exact_match_placeholder(
    query_neutral_mass: float,
    library_masses: Sequence[float],
    ppm_window: float = DEFAULT_PPM_WINDOW,
) -> List[int]:
    """精确质量匹配:返回库中命中下标列表(真实实现).

    命中条件 |m_lib - m_q| / m_q * 1e6 <= ppm_window,全程 float64.
    调用方必须传入已剔除同谱/整组留出后的 library_masses,本函数不做去留-out.
    未知 adduct 回退的 query/library 质量仍按此式比较,偏差会计入
    无候选/覆盖率指标,不在此处特殊处理.
    """
    q = float(query_neutral_mass)
    tol = float(ppm_window)
    if q == 0.0:
        return []
    out: List[int] = []
    for i, m in enumerate(library_masses):
        mf = float(m)
        if abs(mf - q) / abs(q) * 1e6 <= tol:
            out.append(i)
    return out


def _truncate_global_topn(
    peaks: Sequence[Tuple[float, float]], top_n: int
) -> List[Tuple[float, float]]:
    """global topN:按强度降序取前 top_n 个."""
    if top_n is not None and top_n > 0 and len(peaks) > top_n:
        return sorted(peaks, key=lambda p: float(p[1]), reverse=True)[: int(top_n)]
    return list(peaks)


def _truncate_window_50da(
    peaks: Sequence[Tuple[float, float]],
    top_n: int,
    width: float = WINDOW_50DA_WIDTH,
) -> List[Tuple[float, float]]:
    """50Da 窗口截断:按 floor(mz/width) 分桶,每桶按强度取配额,总数约 top_n.

    配额分配:设桶数 nb,每桶 base = max(1, top_n // nb),按桶内强度排序取.
    若总数仍超 top_n,全局按强度再截一次.该策略保留跨质量段弱峰,与
    global_topn 形成对照;b0_run.py 记录实际使用的 window_mode.
    """
    if not peaks:
        return []
    if top_n is None or top_n <= 0:
        return list(peaks)
    buckets: Dict[int, List[Tuple[float, float]]] = {}
    for mz, it in peaks:
        b = int(math.floor(float(mz) / float(width)))
        buckets.setdefault(b, []).append((float(mz), float(it)))
    nb = max(1, len(buckets))
    per = max(1, int(top_n) // nb)
    kept: List[Tuple[float, float]] = []
    for b in sorted(buckets.keys()):
        arr = sorted(buckets[b], key=lambda p: p[1], reverse=True)[:per]
        kept.extend(arr)
    if len(kept) > int(top_n):
        kept = sorted(kept, key=lambda p: p[1], reverse=True)[: int(top_n)]
    return kept


def _l2_normalize(
    peaks: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """峰强度 L2 归一化(输入为已截断峰;零范数返回空)."""
    s = 0.0
    for _, it in peaks:
        f = float(it)
        s += f * f
    if s <= 0.0:
        return []
    norm = math.sqrt(s)
    return [(float(mz), float(it) / norm) for mz, it in peaks]


def cosine_score(
    query_peaks: Sequence[Tuple[float, float]],
    library_peaks: Sequence[Tuple[float, float]],
    fragment_ppm: float = DEFAULT_FRAG_PPM,
    top_n: int = DEFAULT_TOP_N,
    window_mode: str = DEFAULT_WINDOW_MODE,
) -> float:
    """单对谱余弦分(真实实现).

    步骤:截断(top_n,window_mode 二选一) -> L2 归一化 -> ppm 对齐双指针点积.
    对齐容差 |mz_lib - mz_q|/mz_q*1e6 <= fragment_ppm,按 m/z 升序贪心匹配.
    未匹配峰贡献 0;两边任一空/零范数得 0.分值域 [0,1].
    """
    if not query_peaks or not library_peaks:
        return 0.0
    if window_mode == "window_50da":
        q_tr = _truncate_window_50da(query_peaks, top_n)
        l_tr = _truncate_window_50da(library_peaks, top_n)
    else:
        q_tr = _truncate_global_topn(query_peaks, top_n)
        l_tr = _truncate_global_topn(library_peaks, top_n)
    qn = _l2_normalize(q_tr)
    ln = _l2_normalize(l_tr)
    if not qn or not ln:
        return 0.0
    qn = sorted(qn, key=lambda p: p[0])
    ln = sorted(ln, key=lambda p: p[0])
    tol = float(fragment_ppm)
    i = 0
    j = 0
    dot = 0.0
    while i < len(qn) and j < len(ln):
        mq, iq = qn[i]
        ml, il = ln[j]
        denom = abs(mq) if abs(mq) > 0 else 1.0
        err = abs(ml - mq) / denom * 1e6
        if err <= tol:
            dot += iq * il
            i += 1
            j += 1
        elif ml < mq:
            j += 1
        else:
            i += 1
    if dot < 0.0:
        return 0.0
    if dot > 1.0:
        return 1.0
    return float(dot)


def cosine_match_placeholder(
    query_peaks: Optional[Sequence[Tuple[float, float]]] = None,
    library_peaks: Optional[Sequence[Sequence[Tuple[float, float]]]] = None,
    topk: int = 25,
    fragment_ppm: float = DEFAULT_FRAG_PPM,
    top_n: int = DEFAULT_TOP_N,
    window_mode: str = DEFAULT_WINDOW_MODE,
) -> List[Tuple[int, float]]:
    """余弦谱匹配(真实实现,保持占位接口名与前三参数兼容).

    参数:
        query_peaks: [(mz, intensity)] (强度为 ms2_normalized_intensities 原值,
          本函数内做 top-N 截断 + L2 归一化).
        library_peaks: 每库谱的峰列表,与 exact 质量过滤后的候选顺序对应.
        topk: 返回前 topk 个 (库下标, 分数),按分数降序.
        fragment_ppm: 碎片对齐 ppm 容差(参数化,默认 20).
        top_n: 峰截断数(默认 128).
        window_mode: "global_topn" | "window_50da",二选一,b0_run.py 记录实测值.

    同谱/整组留出由调用方保证,本函数只对传入的 library_peaks 排序.
    空输入返回 [].
    """
    if not query_peaks or not library_peaks:
        return []
    scored: List[Tuple[int, float]] = []
    for idx, lib in enumerate(library_peaks):
        s = cosine_score(
            query_peaks,
            lib,
            fragment_ppm=fragment_ppm,
            top_n=top_n,
            window_mode=window_mode,
        )
        scored.append((idx, float(s)))
    scored.sort(key=lambda p: p[1], reverse=True)
    if topk is not None and topk > 0:
        scored = scored[: int(topk)]
    return scored


def aggregate_by_molecule(
    per_spectrum_rankings: Dict[str, List[str]],
    spectrum_to_molecule: Dict[str, str],
    max_candidates: int = MAX_CANDIDATES,
) -> Dict[str, List[str]]:
    """把单谱候选按 molecule_id 聚合成每分子一个有序列表(真实最小实现).

    策略:按 spectrum_id 排序后轮转(round-robin)取首见顺序,精确字符串去重,
    每分子至多 max_candidates 个.这只是多谱融合的顺序占位(对应 B2 要比较的
    单谱/合并谱/分数聚合中的最简一种),不含任何相似度加权.

    参数均为调用方动态传入,不硬编码 ID 集合.
    """
    per_mol: Dict[str, List[str]] = {}
    for spec_id in sorted(per_spectrum_rankings.keys()):
        mol = spectrum_to_molecule.get(spec_id)
        if mol is None:
            continue
        bucket = per_mol.setdefault(mol, [])
        for smi in per_spectrum_rankings[spec_id]:
            if smi not in bucket:
                bucket.append(smi)
            if len(bucket) >= max_candidates:
                break
    # 没有任何谱证据的分子不在输入中出现 —— “无候选”比例由 evaluation 侧统计,
    # 这里不凭空补齐,避免掩盖覆盖盲区.
    return per_mol


def b0_placeholder_candidates(
    n_spectra: int = 1, placeholder: str = PLACEHOLDER_SMILES
) -> List[str]:
    """B0 无库时的占位候选:返回 25 个 placeholder(默认 CCO).

    用于 baseline.ipynb 在“尚无谱库/权重”阶段走通 submission.csv 全流程.
    输出行的分号拼接格式与 sample_submission 一致.不代表基线分数.
    n_spectra 仅记录谱数信息,当前占位不改变输出(后续 B2 按谱数比较融合策略).
    """
    _ = n_spectra
    return [placeholder] * MAX_CANDIDATES

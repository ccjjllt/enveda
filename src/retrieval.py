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


# ---------------------------------------------------------------------------
# E1:完整背景谱库与 shifted analog(005 计划 E1,手动研究用,不是测试文件).
#
# 口径与边界(调用方必须如实记录,本模块不强制):
# - 结构身份当前为 shipped inchikey14 占位(caliber="shipped_placeholder"),
#   未做 RDKit 去立体/互变异构规范化.整组留出按 shipped key 执行,不得称为
#   官方规范化分组.官方键见 src/chemistry.official_match_key.
# - 候选结构集 C 来自理论质量窗(分子式单同位素质量 vs 观测中性质量,20ppm),
#   与 src/candidates.mass_recall_stats 同口径.库谱证据 L 通过“结构 key 在
#   候选集内”关联,不再对库谱观测中性质量二次设窗(等价于质量桶过滤,因候选
#   已按质量窗枚举).标量背景索引另按观测中性质量/极性/adduct 分桶统计规模.
# - 同谱剔除:seen 轨仅排除“同结构+精确峰 hash 相等”的库谱(query 自身及逐字
#   拷贝);unseen 轨排除该 query 真值结构的全部库谱(整组留出).近重复不合并,
#   不以 cosine 阈值预排除,调用方不得把本组数当作去重后唯一谱数.
# - 直接与偏移证据分开:先算直接匹配,再算偏移匹配时跳过直接已匹配的 query
#   峰(保持原 L2 归一化不变,仅把已用峰贡献置 0,保证分数可比且计数不重复).
#   后续 ranker 接收两路特征,不用未校准阈值硬覆盖 FPNet.
# - unseen 轨没有真值参考谱时,direct 真值分恒为 0;analog 只报相近结构证据/
#   衍生种子(top 结构+delta+分数+条件),不计作“找回真值”的命中.
# - “有候选/高 cosine/真值命中”三分开:有候选=质量窗非空;高 cosine=结构
#   max 分>=阈值(仅报告用,不参与排序);真值命中=真值在窗内且排序 rank<=25.
# - 负模式与失水离子单列:极性按 ionization_mode 原字符串( positive/negative);
#   失水指 adduct 字符串含 "H2O"(如 [M-H2O+H]+/[M-2H2O+H]+/[M-H2O-H]- 等).
# - 仅依赖标准库;流式函数在有 pyarrow 时才可用,无 pyarrow 时抛错,不影响导入.
# ---------------------------------------------------------------------------

# E1 默认参数(调用方可覆盖,实际取值由调用方写入报告).
E1_FRAG_PPM = 20.0
E1_TOP_N = 128
E1_WINDOW_MODE = "global_topn"  # 与 B0 默认一致,二选一 global_topn|window_50da
E1_MASS_PPM = 20.0
# 仅报告用阈值,不参与排序/覆盖.0.5 为“高”,调用方可另计 >=0.7 的很高.
E1_HIGH_COSINE = 0.5
E1_VERY_HIGH_COSINE = 0.7


def is_water_loss_adduct(adduct) -> bool:
    """失水离子判断:adduct 字符串含 H2O 即为失水(大小写敏感,原样匹配)."""
    try:
        return "H2O" in str(adduct)
    except Exception:
        return False


def spectral_entropy(intensities: Sequence[float]) -> float:
    """谱熵: -sum(p*ln p),p=it/sum(it).空/全零返回 0.0.

    输入为截断后峰强度(与 cosine 输入一致),自然对数.熵越大谱越分散.
    """
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


def entropy_similarity(s_q: float, s_l: float) -> float:
    """谱熵相似度:1-|Sq-Sl|/max(Sq,Sl,eps),域 [0,1].缺失(全 0)时返回 0.

    仅为可解释特征,未校准,进 ranker 由 E2 学习权重,不单独做阈值判断.
    """
    q = float(s_q)
    l = float(s_l)
    m = max(q, l)
    if m <= 1e-12:
        return 0.0
    sim = 1.0 - abs(q - l) / m
    if sim < 0.0:
        return 0.0
    if sim > 1.0:
        return 1.0
    return float(sim)


def ce_median(values) -> Optional[float]:
    """collision_energy_ev 中位数;None/空/全非数返回 None."""
    if values is None:
        return None
    try:
        arr = [float(v) for v in values]
    except TypeError:
        return None
    arr = [v for v in arr if v == v]  # 去 NaN
    if not arr:
        return None
    arr.sort()
    n = len(arr)
    mid = n // 2
    if n % 2 == 1:
        return float(arr[mid])
    return float((arr[mid - 1] + arr[mid]) / 2.0)


def _prepare_cosine_vectors(
    peaks: Sequence[Tuple[float, float]],
    top_n: int = E1_TOP_N,
    window_mode: str = E1_WINDOW_MODE,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], float, float]:
    """截断+L2 归一化+m/z 排序,供 direct/shifted 共用,保证同一 query 可比.

    返回 (norm_sorted, trunc_list, trunc_sum, entropy).norm_sorted 为
    [(mz, w)] 按 mz 升序,w 为 L2 归一化权重;trunc_list 为截断后未归一化峰;
    trunc_sum 为截断强度和(解释强度分母);entropy 为截断强度的谱熵.
    空/零范数时 norm_sorted 为 [],调用方按 0 分计.
    """
    if not peaks:
        return [], [], 0.0, 0.0
    if window_mode == "window_50da":
        trunc = _truncate_window_50da(peaks, top_n)
    else:
        trunc = _truncate_global_topn(peaks, top_n)
    trunc = [(float(m), float(i)) for m, i in trunc]
    tot = sum(i for _, i in trunc if i > 0.0)
    ent = spectral_entropy([i for _, i in trunc])
    norm = _l2_normalize(trunc)
    norm_sorted = sorted(norm, key=lambda p: p[0])
    return norm_sorted, trunc, float(tot), float(ent)


def cosine_detailed(
    query_peaks: Sequence[Tuple[float, float]],
    library_peaks: Sequence[Tuple[float, float]],
    fragment_ppm: float = E1_FRAG_PPM,
    top_n: int = E1_TOP_N,
    window_mode: str = E1_WINDOW_MODE,
) -> Dict:
    """单对谱直接匹配明细(与 cosine_score 同截断/归一化/贪心对齐).

    返回 {score, n_matched, explained_query, explained_library,
    matched_query_positions(在 query norm_sorted 中的下标),
    query_entropy, library_entropy, entropy_sim}.
    explained 为匹配强度和/截断强度和(截断和为 0 时为 0).分数域 [0,1].
    """
    qn, qtr, qtot, qent = _prepare_cosine_vectors(query_peaks, top_n, window_mode)
    ln, ltr, ltot, lent = _prepare_cosine_vectors(library_peaks, top_n, window_mode)
    base: Dict = {
        "score": 0.0, "n_matched": 0, "explained_query": 0.0,
        "explained_library": 0.0, "matched_query_positions": [],
        "query_entropy": float(qent), "library_entropy": float(lent),
        "entropy_sim": float(entropy_similarity(qent, lent)),
    }
    if not qn or not ln:
        return base
    # 解释强度需映射回截断强度:按 mz 排序后的对应强度.
    # qtr 未排序,需按 mz 排序后与 qn 对齐.
    qtr_sorted = sorted(qtr, key=lambda p: p[0])
    # qn 与 qtr_sorted 长度一致且顺序一致(同一截断集),可按下标取强度.
    tol = float(fragment_ppm)
    i = 0
    j = 0
    dot = 0.0
    matched_q: List[int] = []
    matched_q_sum = 0.0
    matched_l_sum = 0.0
    ltr_sorted = sorted(ltr, key=lambda p: p[0])
    while i < len(qn) and j < len(ln):
        mq, iq = qn[i]
        ml, il = ln[j]
        denom = abs(mq) if abs(mq) > 0 else 1.0
        err = abs(ml - mq) / denom * 1e6
        if err <= tol:
            dot += iq * il
            matched_q.append(i)
            try:
                matched_q_sum += float(qtr_sorted[i][1])
                matched_l_sum += float(ltr_sorted[j][1])
            except IndexError:
                pass
            i += 1
            j += 1
        elif ml < mq:
            j += 1
        else:
            i += 1
    if dot < 0.0:
        dot = 0.0
    if dot > 1.0:
        dot = 1.0
    base["score"] = float(dot)
    base["n_matched"] = int(len(matched_q))
    base["matched_query_positions"] = [int(v) for v in matched_q]
    base["explained_query"] = float(matched_q_sum / qtot) if qtot > 0 else 0.0
    base["explained_library"] = float(matched_l_sum / ltot) if ltot > 0 else 0.0
    return base


def shifted_cosine_detailed(
    query_peaks: Sequence[Tuple[float, float]],
    library_peaks: Sequence[Tuple[float, float]],
    delta: float,
    fragment_ppm: float = E1_FRAG_PPM,
    top_n: int = E1_TOP_N,
    window_mode: str = E1_WINDOW_MODE,
    exclude_query_positions: Optional[Sequence[int]] = None,
) -> Dict:
    """单对谱质量偏移 analog 明细(库峰整体平移 delta 后匹配).

    delta 定义:query_neutral - library_neutral(观测中性质量差,float64).
    library 平移后 mz'=mz+delta,再与 query 做 ppm 贪心匹配.
    exclude_query_positions 为直接已匹配的 query 下标(在 query norm_sorted
    中),匹配时跳过这些峰(贡献置 0,归一化保持直接口径不变,保证可比).
    返回 {score_shifted, n_matched_shifted, explained_query_shifted,
    explained_library_shifted, matched_query_positions_shifted, delta,
    query_entropy, library_entropy, entropy_sim(与直接相同,未平移)}.
    """
    qn, qtr, qtot, qent = _prepare_cosine_vectors(query_peaks, top_n, window_mode)
    ln_raw, ltr, ltot, lent = _prepare_cosine_vectors(library_peaks, top_n, window_mode)
    base: Dict = {
        "score_shifted": 0.0, "n_matched_shifted": 0,
        "explained_query_shifted": 0.0, "explained_library_shifted": 0.0,
        "matched_query_positions_shifted": [], "delta": float(delta),
        "query_entropy": float(qent), "library_entropy": float(lent),
        "entropy_sim": float(entropy_similarity(qent, lent)),
    }
    if not qn or not ln_raw:
        return base
    d = float(delta)
    # 库峰平移后重排(权重不变,仅 mz 平移).
    ln = sorted([(float(m) + d, float(w)) for m, w in ln_raw], key=lambda p: p[0])
    # 解释强度映射:库截断强度按原 mz 排序,平移后顺序不变(整体平移不改变顺序).
    ltr_sorted = sorted(ltr, key=lambda p: p[0])
    qtr_sorted = sorted(qtr, key=lambda p: p[0])
    excluded = set(int(v) for v in (exclude_query_positions or []))
    tol = float(fragment_ppm)
    i = 0
    j = 0
    dot = 0.0
    matched_q: List[int] = []
    matched_q_sum = 0.0
    matched_l_sum = 0.0
    # 双指针中跳过已用 query 峰:若 query 下标在 excluded,直接进 i,不尝试匹配.
    while i < len(qn) and j < len(ln):
        if i in excluded:
            i += 1
            continue
        mq, iq = qn[i]
        ml, il = ln[j]
        denom = abs(mq) if abs(mq) > 0 else 1.0
        err = abs(ml - mq) / denom * 1e6
        if err <= tol:
            dot += iq * il
            matched_q.append(i)
            try:
                matched_q_sum += float(qtr_sorted[i][1])
                # 库侧下标 j 对应平移后顺序,与原截断排序一致.
                matched_l_sum += float(ltr_sorted[j][1]) if j < len(ltr_sorted) else 0.0
            except IndexError:
                pass
            i += 1
            j += 1
        elif ml < mq:
            j += 1
        else:
            i += 1
    if dot < 0.0:
        dot = 0.0
    if dot > 1.0:
        dot = 1.0
    base["score_shifted"] = float(dot)
    base["n_matched_shifted"] = int(len(matched_q))
    base["matched_query_positions_shifted"] = [int(v) for v in matched_q]
    base["explained_query_shifted"] = float(matched_q_sum / qtot) if qtot > 0 else 0.0
    base["explained_library_shifted"] = float(matched_l_sum / ltot) if ltot > 0 else 0.0
    return base


def direct_plus_shifted_disjoint(
    query_peaks: Sequence[Tuple[float, float]],
    library_peaks: Sequence[Tuple[float, float]],
    delta: float,
    fragment_ppm: float = E1_FRAG_PPM,
    top_n: int = E1_TOP_N,
    window_mode: str = E1_WINDOW_MODE,
) -> Tuple[Dict, Dict, Dict]:
    """直接+偏移联合证据(同一 query 峰不重复计).

    先算直接明细,再以直接已匹配 query 峰为排除集算偏移.返回
    (direct, shifted, combined),其中 combined 含
    {n_unique_matched, explained_query_unique(直接+偏移解释强度之和,因峰
    不重叠可直接相加), delta}.库侧解释强度不相加(库峰平移前后为不同坐标).
    """
    direct = cosine_detailed(query_peaks, library_peaks, fragment_ppm, top_n, window_mode)
    shifted = shifted_cosine_detailed(
        query_peaks, library_peaks, delta, fragment_ppm, top_n, window_mode,
        exclude_query_positions=direct.get("matched_query_positions", []),
    )
    combined = {
        "n_unique_matched": int(direct.get("n_matched", 0))
        + int(shifted.get("n_matched_shifted", 0)),
        "explained_query_unique": float(direct.get("explained_query", 0.0))
        + float(shifted.get("explained_query_shifted", 0.0)),
        "delta": float(delta),
    }
    # 防御性截断(浮点累加不超 1 太多时保持原值,明显超限说明口径误用).
    if combined["explained_query_unique"] > 1.0:
        combined["explained_query_unique"] = 1.0
    return direct, shifted, combined


def aggregate_structure_hits(
    per_spectrum_hits: Dict[str, Dict],
) -> List[Dict]:
    """按规范化结构(当前 shipped 占位)聚合库谱命中,再取结构 top-k 排序.

    per_spectrum_hits: {spectrum_id: {structure_key, score, n_matched,
    explained_query, explained_library, entropy_sim, same_adduct(bool),
    ce_diff(可 None), ingest_lib, polarity, adduct}}.
    对每个结构计算 {structure_key, max_score, second_max,
    best_n_matched, best_explained_query, best_entropy_sim,
    best_same_adduct, best_ce_diff, n_spectra, n_sources, gap}.
    gap=max-second(单谱结构 second=0.0).返回按 (-max_score, key) 排序的列表.
    调用方按需截断 top-k,本函数不截断,避免掩盖长尾.
    """
    from collections import defaultdict as _dd

    grouped: Dict[str, List[Dict]] = _dd(list)
    for _sid, h in per_spectrum_hits.items():
        grouped[str(h.get("structure_key", ""))].append(h)
    out: List[Dict] = []
    for key in sorted(grouped.keys()):
        rows = grouped[key]
        scores = sorted([float(r.get("score", 0.0)) for r in rows], reverse=True)
        mx = float(scores[0]) if scores else 0.0
        second = float(scores[1]) if len(scores) > 1 else 0.0
        # 最佳行:最大分,同分取 n_matched 大者,再按 spectrum_id 确定性.
        best = None
        best_tuple = None
        for r in rows:
            t = (float(r.get("score", 0.0)), int(r.get("n_matched", 0)))
            if best_tuple is None or t > best_tuple:
                best_tuple = t
                best = r
        if best is None:
            best = {}
        sources = sorted({str(r.get("ingest_lib", "")) for r in rows if r.get("ingest_lib")})
        out.append({
            "structure_key": str(key),
            "max_score": float(mx),
            "second_max": float(second),
            "gap": float(mx - second),
            "best_n_matched": int(best.get("n_matched", 0) or 0),
            "best_explained_query": float(best.get("explained_query", 0.0) or 0.0),
            "best_entropy_sim": float(best.get("entropy_sim", 0.0) or 0.0),
            "best_same_adduct": bool(best.get("same_adduct", False)),
            "best_ce_diff": None if best.get("ce_diff") is None else float(best.get("ce_diff")),
            "n_spectra": int(len(rows)),
            "n_sources": int(len(sources)),
            "sources_sorted": sources,
        })
    out.sort(key=lambda d: (-d["max_score"], d["structure_key"]))
    return out


def mrr_at_25_simple(ranks: Sequence) -> Dict:
    """MRR@25/Top1/Top25(rank None 或 >25 按 0 计).纯标准库,供 E1 报告用."""
    seq = list(ranks)
    n = len(seq)
    if n == 0:
        return {"n": 0, "mrr@25": 0.0, "top1": 0.0, "top25": 0.0, "n_ranked_le25": 0}
    rr = 0.0
    top1 = 0
    top25 = 0
    for r in seq:
        if r is not None:
            try:
                ri = int(r)
            except Exception:
                continue
            if 1 <= ri <= 25:
                rr += 1.0 / ri
                top25 += 1
                if ri == 1:
                    top1 += 1
    return {"n": float(n), "mrr@25": float(rr / n) if n else 0.0,
            "top1": float(top1 / n) if n else 0.0,
            "top25": float(top25 / n) if n else 0.0,
            "n_ranked_le25": float(top25)}


def build_full_background_scalar_index(
    train_path: str = "data/raw/train.parquet",
    batch_size: int = 65536,
) -> Dict:
    """完整背景标量索引统计(流式,不读 ms2,不驻留全量).

    读取 inchikey14/ionization_mode/adduct/precursor_mz/ingest_lib/
    instrument_type,计算观测中性质量(复用 ADDUCT_TABLE,未知回退 precursor),
    按极性/adduct/来源分桶计数.返回 {n_spectra, n_structures, per_mode,
    per_adduct_top, per_lib, unknown_adduct_rows, neutral_mass_min/max,
    null_key_rows}.中性质量 min/max 为 float64,未知加合回退值同样参与(调用
    方看 unknown 比例,不得当精确质量用).本函数不读 ms2,不建谱峰索引.
    """
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:
        raise RuntimeError("need pyarrow for background index: %s" % e)
    from collections import Counter as _C
    pf = pq.ParquetFile(train_path)
    per_mode: _C = _C()
    per_adduct: _C = _C()
    per_lib: _C = _C()
    per_inst: _C = _C()
    keys = set()
    n = 0
    n_unknown_ad = 0
    n_null_key = 0
    n_empty_key = 0
    mass_min: Optional[float] = None
    mass_max: Optional[float] = None
    for batch in pf.iter_batches(
        columns=["inchikey14", "ionization_mode", "adduct", "precursor_mz",
                 "ingest_lib", "instrument_type"],
        batch_size=batch_size,
    ):
        iks = batch.column(0).to_pylist()
        modes = batch.column(1).to_pylist()
        ads = batch.column(2).to_pylist()
        precs = batch.column(3).to_pylist()
        libs = batch.column(4).to_pylist()
        insts = batch.column(5).to_pylist()
        for k, mo, ad, pr, li, ins in zip(iks, modes, ads, precs, libs, insts):
            n += 1
            if k is None:
                n_null_key += 1
                continue
            ks = str(k)
            if ks == "":
                n_empty_key += 1
                continue
            keys.add(ks)
            mstr = "" if mo is None else str(mo)
            astr = "" if ad is None else str(ad)
            per_mode[mstr] += 1
            per_adduct[astr] += 1
            per_lib[str(li)] += 1
            per_inst[str(ins)] += 1
            try:
                prf = float(pr) if pr is not None else None
            except Exception:
                prf = None
            if prf is None:
                continue
            nm, _ = adduct_to_neutral_mass(prf, astr, DEFAULT_PPM_WINDOW)
            # 未知加合回退计数(adduct_to_neutral_mass 不抛错,此处按表查).
            if astr not in ADDUCT_TABLE:
                n_unknown_ad += 1
            if mass_min is None or nm < mass_min:
                mass_min = float(nm)
            if mass_max is None or nm > mass_max:
                mass_max = float(nm)
    return {
        "n_spectra": int(n),
        "n_structures": int(len(keys)),
        "per_mode": dict(sorted(per_mode.items())),
        "per_adduct": dict(sorted(per_adduct.items(), key=lambda kv: -kv[1])),
        "per_lib": dict(sorted(per_lib.items(), key=lambda kv: -kv[1])),
        "per_instrument_raw": dict(sorted(per_inst.items(), key=lambda kv: -kv[1])),
        "unknown_adduct_rows": int(n_unknown_ad),
        "null_key_rows": int(n_null_key),
        "empty_key_rows": int(n_empty_key),
        "neutral_mass_min": None if mass_min is None else float(mass_min),
        "neutral_mass_max": None if mass_max is None else float(mass_max),
        "key_caliber": "shipped_placeholder",
    }


def evaluate_direct_full_background(
    queries: Sequence[Dict],
    candidate_sets: Dict[int, List[str]],
    key_to_query_ids: Dict[str, List[int]],
    query_hashes: Dict[int, Optional[str]],
    train_path: str = "data/raw/train.parquet",
    track: str = "seen",
    fragment_ppm: float = E1_FRAG_PPM,
    top_n: int = E1_TOP_N,
    window_mode: str = E1_WINDOW_MODE,
    batch_size: int = 4096,
) -> Dict[int, Dict]:
    """完整背景直接检索:单遍流式,对候选集内结构取结构 max cosine.

    queries: [{inchikey14, adduct, precursor_mz, ionization_mode,
      instrument_type, collision_energy_ev, ms2_mzs,
      ms2_normalized_intensities}, ...],下标即 query_id.
    candidate_sets: {qid: [candidate_key,...]}(理论质量窗,调用方已算好).
    key_to_query_ids: {structure_key: [qid,...]} 倒排(调用方由 candidate_sets
      建好,本函数只查表,不重建).
    query_hashes: {qid: spectrum_hash 或 None}(调用方用 splits.spectrum_hash
      算好;seen 轨同结构同 hash 跳过,unseen 轨整结构跳过,不在此处重算).
    track: "seen"(排除同结构精确重复)或 "unseen"(排除真值结构全部谱).
    返回 {qid: {ranked: [structure_key...按 max 分降序,同分按 key],
      scores: {key: max}, per_struct: {key: 聚合特征}, truth_rank,
      has_candidate, max_score, high_cosine}}.
    排序只用 max 分(纯 cosine-only),特征供 ranker 用,不做阈值覆盖.
    """
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:
        raise RuntimeError("need pyarrow for streaming evaluation: %s" % e)
    if track not in ("seen", "unseen"):
        raise ValueError("track must be seen|unseen")
    # 预处理 query:截断+归一化只做一次的轻量缓存为峰列表本身(详细函数内仍会
    # 重做截断,但此处先转 float 元组,避免每对重复转类型).
    q_peaks: Dict[int, List[Tuple[float, float]]] = {}
    q_meta: Dict[int, Dict] = {}
    for qid, q in enumerate(queries):
        try:
            mzs = [float(v) for v in q.get("ms2_mzs", [])]
            its = [float(v) for v in q.get("ms2_normalized_intensities", [])]
        except Exception:
            mzs, its = [], []
        n = min(len(mzs), len(its))
        q_peaks[qid] = [(mzs[i], its[i]) for i in range(n)]
        q_meta[qid] = {
            "key": str(q.get("inchikey14", "")),
            "adduct": q.get("adduct"),
            "mode": q.get("ionization_mode"),
            "ce": ce_median(q.get("collision_energy_ev")),
        }
    # 累加器:qid -> key -> {best, second, best_detail, n, sources set, ce/same}
    from collections import defaultdict as _dd
    acc_best: Dict[int, Dict[str, float]] = _dd(dict)
    acc_second: Dict[int, Dict[str, float]] = _dd(dict)
    acc_detail: Dict[int, Dict[str, Dict]] = _dd(dict)
    acc_n: Dict[int, Dict[str, int]] = _dd(dict)
    acc_sources: Dict[int, Dict[str, set]] = _dd(dict)
    # 真值 key 速查(unseen 用).
    q_truth: Dict[int, str] = {qid: str(m["key"]) for qid, m in q_meta.items()}
    pf = pq.ParquetFile(train_path)
    cols = ["inchikey14", "ionization_mode", "adduct", "precursor_mz",
            "ingest_lib", "instrument_type", "collision_energy_ev",
            "ms2_mzs", "ms2_normalized_intensities"]
    # 为 hash 排除,需与 splits.spectrum_hash 同口径;此处内联最小实现以保
    # 持本模块仅标准库(排序+float64 打包+sha256),与 splits 逐字对齐.
    import hashlib as _hl
    import struct as _st

    def _lib_hash(mzs, its) -> Optional[str]:
        if mzs is None or its is None:
            return None
        try:
            nn = len(mzs)
        except TypeError:
            return None
        if nn == 0 or len(its) != nn:
            return None
        try:
            pairs = sorted(((float(a), float(b)) for a, b in zip(mzs, its)),
                           key=lambda p: p[0])
        except (TypeError, ValueError):
            return None
        h = _hl.sha256()
        h.update(_st.pack("<q", nn))
        for a, b in pairs:
            h.update(_st.pack("<2d", a, b))
        return h.hexdigest()

    for batch in pf.iter_batches(columns=cols, batch_size=batch_size):
        iks = batch.column(0).to_pylist()
        modes = batch.column(1).to_pylist()
        ads = batch.column(2).to_pylist()
        precs = batch.column(3).to_pylist()
        libs = batch.column(4).to_pylist()
        insts = batch.column(5).to_pylist()
        ces = batch.column(6).to_pylist()
        mzs_col = batch.column(7).to_pylist()
        its_col = batch.column(8).to_pylist()
        for k, mo, ad, pr, li, ins, ce_raw, lmz, lit in zip(
                iks, modes, ads, precs, libs, insts, ces, mzs_col, its_col):
            if k is None:
                continue
            ks = str(k)
            if ks == "":
                continue
            qids = key_to_query_ids.get(ks)
            if not qids:
                continue
            # 库峰预处理一次(本行对应多 query 时复用).
            try:
                l_m = [float(v) for v in (lmz or [])]
                l_i = [float(v) for v in (lit or [])]
            except Exception:
                l_m, l_i = [], []
            nn = min(len(l_m), len(l_i))
            l_peaks = [(l_m[i], l_i[i]) for i in range(nn)]
            # 库 hash 仅在需要排除时计算:先判是否有 seen 轨同结构 query.
            need_hash = False
            for _qid in qids:
                if track == "seen" and q_truth.get(_qid) == ks:
                    need_hash = True
                    break
            lh = _lib_hash(l_m, l_i) if need_hash else None
            lce = ce_median(ce_raw)
            for qid in qids:
                tk = q_truth.get(qid)
                if track == "unseen" and ks == tk:
                    continue
                if track == "seen" and ks == tk:
                    qh = query_hashes.get(qid)
                    if qh is not None and lh is not None and qh == lh:
                        continue
                # 直接明细(库侧全谱,不按极性/adduct 预过滤;条件记为特征).
                det = cosine_detailed(q_peaks[qid], l_peaks, fragment_ppm,
                                      top_n, window_mode)
                sc = float(det.get("score", 0.0))
                # 同 adduct/CE 差/来源.
                qm = q_meta[qid]
                same_ad = (str(ad) == str(qm.get("adduct"))) if ad is not None else False
                qce = qm.get("ce")
                if qce is not None and lce is not None:
                    try:
                        ced = abs(float(qce) - float(lce))
                    except Exception:
                        ced = None
                else:
                    ced = None
                cur_best = acc_best[qid].get(ks)
                if cur_best is None or sc > cur_best:
                    # 旧 best 降为 second 候选.
                    if cur_best is not None:
                        prev_second = acc_second[qid].get(ks, 0.0)
                        acc_second[qid][ks] = float(max(prev_second, cur_best))
                    acc_best[qid][ks] = float(sc)
                    acc_detail[qid][ks] = {
                        "score": float(sc),
                        "n_matched": int(det.get("n_matched", 0)),
                        "explained_query": float(det.get("explained_query", 0.0)),
                        "explained_library": float(det.get("explained_library", 0.0)),
                        "entropy_sim": float(det.get("entropy_sim", 0.0)),
                        "query_entropy": float(det.get("query_entropy", 0.0)),
                        "library_entropy": float(det.get("library_entropy", 0.0)),
                        "same_adduct": bool(same_ad),
                        "ce_diff": None if ced is None else float(ced),
                        "ingest_lib": str(li),
                        "polarity": "" if mo is None else str(mo),
                        "adduct": "" if ad is None else str(ad),
                        "is_water_loss_lib": bool(is_water_loss_adduct(ad)),
                    }
                else:
                    prev_second = acc_second[qid].get(ks, 0.0)
                    if sc > prev_second:
                        acc_second[qid][ks] = float(sc)
                acc_n[qid][ks] = int(acc_n[qid].get(ks, 0) + 1)
                sset = acc_sources[qid].get(ks)
                if sset is None:
                    acc_sources[qid][ks] = {str(li)}
                else:
                    sset.add(str(li))
    # 收尾:按候选集排序,补 0 分候选(无库谱或全被排除的候选按 0 分、同分按 key).
    out: Dict[int, Dict] = {}
    for qid in range(len(queries)):
        cands = list(candidate_sets.get(qid, []))
        per_struct: Dict[str, Dict] = {}
        for key in cands:
            best = float(acc_best[qid].get(key, 0.0))
            second = float(acc_second[qid].get(key, 0.0))
            det = acc_detail[qid].get(key, {})
            srcs = acc_sources[qid].get(key, set())
            per_struct[key] = {
                "structure_key": str(key),
                "max_score": float(best),
                "second_max": float(second),
                "gap": float(best - second),
                "best_n_matched": int(det.get("n_matched", 0)),
                "best_explained_query": float(det.get("explained_query", 0.0)),
                "best_entropy_sim": float(det.get("entropy_sim", 0.0)),
                "best_same_adduct": bool(det.get("same_adduct", False)),
                "best_ce_diff": det.get("ce_diff"),
                "n_spectra": int(acc_n[qid].get(key, 0)),
                "n_sources": int(len(srcs)),
                "sources_sorted": sorted(srcs),
                "is_water_loss_query": bool(is_water_loss_adduct(q_meta[qid].get("adduct"))),
            }
        ranked = sorted(cands, key=lambda k: (-per_struct[k]["max_score"], k))
        tk = q_truth.get(qid, "")
        truth_rank = None
        try:
            truth_rank = ranked.index(tk) + 1
        except ValueError:
            truth_rank = None
        max_score = max((per_struct[k]["max_score"] for k in cands), default=0.0)
        out[qid] = {
            "ranked": ranked, "scores": {k: per_struct[k]["max_score"] for k in cands},
            "per_struct": per_struct, "truth_rank": truth_rank,
            "has_candidate": bool(len(cands) > 0),
            "max_score": float(max_score),
            "high_cosine": bool(max_score >= E1_HIGH_COSINE),
            "very_high_cosine": bool(max_score >= E1_VERY_HIGH_COSINE),
        }
    return out

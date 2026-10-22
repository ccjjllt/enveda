"""005-E3 GLACIER 同式重排与自适应预算：原始模型 adapter、条件映射、缓存与预算.

手动研究用模块，不是测试文件（无 pytest/unittest，不做断言门控）。
不读凭据，不联网，不硬编码任何分子/谱 ID。

来源与归属（已核实，见 LICENSE-NOTICE 口径）：
- 模型：ms-pred GLACIER JointModel，MassSpecGym 训练权重（MIT 推理子集）。
  上游：ms-pred README “GLACIER” 节 Dropbox 文件夹
  https://www.dropbox.com/scl/fo/ta99j0mp1w7qzek5zvy3w/AFCLQNO8W2EP7ZDMOj9nxWI?rlkey=563zxtyvvhwfu1uyz6n10n2ui&st=k8km33p3&e=1&dl=0
  （文件夹页，非单文件直链；直接下载需 JS/官方 API，本项目改走 Kaggle 数据集）。
  上游 GLACIER_checkpoint.zip -> best.ckpt sha256
  5a47cecca707d3abd5a49c7dbac99d100aa2a586d5d4f848f1e8e35140d7db11。
- 本地权重：artifacts/runs/005/glacier.pt（Kaggle 数据集
  ahmedberatozer/casmi26-glacier v1 内 ckpt/glacier.pt 的 stripped 重存，
  {hyper_parameters, state_dict}，优化器/trainer 状态已丢，fp32 张量一致；
  NIST 权重一律排除，见主办方答复 discussion/744470）。
- 条件/分数口径：复刻该数据集 gl_runner.py（fork 自 casmi26-iceberg
  ice_runner.py）的作业构造与计分定义：覆盖 adduct、instrument token、
  CE 解析/分桶、clean/prep/entropy_sim、post/merge、same-formula
  slot-preserving 融合（见 gl_fuse.rerank_multi 单 scorer 时等价 fuse.rerank）。
  本文件是研究用重写，不是逐行复用；数值等价性未验证，GPU 实测前不得
  称为已复现公开分数。

硬边界（调用方必须如实记录，本模块不强制）：
- 仅支持正模式 [M+H]+/[M+Na]+（公开 wrapper 口径；v29 进一步限 [M+H]+，
  本模块默认 COV 两者全开并分别计数，调用方可收紧为仅 [M+H]+ 做对照）。
  其余 adduct/负模式一律记 unsupported，不写成 0 分惩罚候选。
- timsTOF->QTOF 与 CE 5eV bucket 是 wrapper 的映射假设（timsTOF 无原生
  token；CE 取整到 5eV，NaN 走 20+40+60 merged 'sum' 预测）。保留对照，
  不得称为仪器真实条件。
- unsupported / inference_failed / no_useful_peaks 与有效低分分开计数；
  ranker 接收缺失标记（None），不把未预测负谱写成低相似度。
- 缓存键包含结构、adduct、CE、instrument、模型/预处理版本；同一规范
  SMILES 只预测一次。
- 输出相似度同时保留匹配峰数与解释强度（query 侧解释强度必留；库/预测
  侧解释强度在有预测谱时留）。

依赖：
- 条件映射/缓存键/预算/融合：仅标准库，可在任何 Python 直接 import。
- 查询谱 prep 与 entropy 相似度：需要 numpy（本地 brew python3.11 可用）。
- 真实 GLACIER 前向：需要 torch + ms-pred 推理子集 + DGL/torch_scatter/
  LinSATNet（公开 wrapper 用 pure-torch shim 替代）+ RDKit 2025.03。
  本模块延迟导入，缺失时 predict 记 inference_failed，不在导入期抛错。
  本地 M5 已有 torch 2.14.1 + RDKit 2026.03.6，但无 ms-pred/DGL shim 且
  RDKit 版本与 wrapper 要求（2025.03）不一致，故本地只跑条件映射与
  查询侧 prep，不跑模型前向；前向留待 GPU 一键入口。

Colab cp312/3.13 说明（已核实方向，未在远端执行）：
- 公开 iceberg/glacier wrapper 轮子为 cp312；Colab 当前默认 Python3.13
  无法直接 import cp312 轮子（import 即报 ELF/abi 不匹配或找不到符号）。
- 绕行方案（按优先序）：(a) 把 Colab runtime 切到 Python 3.12 后重装轮子；
  (b) 复用 gl_runner.py 的子进程隔离（notebook 保持 RDKit 2026.03，主进程
  不 import torch-DGL shim，子进程用 --site 独立 site 安装轮子）；
  (c) 在本地 brew python3.11（torch + RDKit 已验证可 import）只做条件映射
  与查询侧 prep，模型前向仍在 GPU 子进程做。本文件 adapter 即按 (b) 设计。
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as _np  # type: ignore
except Exception:  # 条件映射/缓存键不受影响；相似度函数在调用时抛错
    _np = None  # type: ignore

# ---- 模型/预处理版本（进缓存键） ----
MODEL_ID = "glacier-massspecgym-stripped"
MODEL_VERSION = "ckpt-sha9192e4c2eeafe833-state264-params15108154"
PREPROC_VERSION = "glrunner-fork-v1:FLOOR0.002-TOPK256-POWER1-ENTW-TOL0.01-PPM20-POSTmax-top100-merge4dpMAX"

# ---- 覆盖条件（公开 wrapper 口径；v29 对照可收紧为仅 [M+H]+） ----
COVERED_ADDUCTS: Tuple[str, ...] = ("[M+H]+", "[M+Na]+")
SUM_CE: Tuple[float, float, float] = (20.0, 40.0, 60.0)

# 相似度参数（与 gl_runner 一致；TOL=0.01Da，PPM=20）。
SIM_TOL_DA = 0.01
SIM_PPM = 20.0
PREP_FLOOR = 0.002
PREP_TOPK = 256
PREP_POWER = 1.0
PREP_ENTW = True
CLEAN_FLOOR = 0.001
CLEAN_TOPK = 512
CLEAN_MAX_OVER_PREC = 2.0
POST_TOPK = 100

# 预算默认（研究预算，非性能承诺）。
DEFAULT_BS = 32
ATOM_BUDGET_PER_BS = 40 * 40
FAILFAST_N = 64


# ---------------------------------------------------------------- 条件映射
def instr_family(s) -> int:
    """仪器族下标（与 gl_runner.instr_family 逐字对齐）。

    0 timsTOF，1 Orbitrap/FT（含 FT-ICR），2 TOF，3 IT，4 QQQ，5 other。
    timsTOF 优先检查，不被 'tof' 截胡。None/NaN -> 5。
    """
    try:
        if s is None:
            return 5
        if isinstance(s, float) and s != s:  # NaN
            return 5
    except Exception:
        return 5
    t = str(s).lower()
    if "timstof" in t:
        return 0
    if ("orbitrap" in t or "qft" in t or "ftms" in t or "hybrid ft" in t
            or "itft" in t or "exactive" in t or "ft-icr" in t or "fticr" in t):
        return 1
    if "tof" in t:
        return 2
    if "trap" in t or "qit" in t or "lit" in t:
        return 3
    if "qq" in t or "quattro" in t or "triple" in t:
        return 4
    return 5


def instr_token(instrument_raw) -> str:
    """instrument token：Orbitrap 族 -> 'Orbitrap'，其余（含 timsTOF）-> 'QTOF'。

    timsTOF->QTOF 是公开 wrapper 的映射假设，不是仪器真实条件；调用方记录。
    已解析的 'Orbitrap'/'QTOF' 直接透传。
    """
    if instrument_raw in ("Orbitrap", "QTOF"):
        return str(instrument_raw)
    return "Orbitrap" if instr_family(instrument_raw) == 1 else "QTOF"


def _nce_mean(s) -> float:
    vals = [float(x) for x in re.findall(r"[0-9]+\.?[0-9]*", str(s))]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def spectrum_ce(sp: Dict) -> float:
    """CE 数值（与 gl_runner.spectrum_ce / iceberg jobs.py 同口径）。

    NCE 单位 -> collision_energy_orig 数字均值；否则 collision_energy_ev 均值；
    两者都无时用已解析 'ce'；空/非有限 -> NaN。
    """
    try:
        if "ce_units" in sp or "ce_ev" in sp:
            if sp.get("ce_units") == "NCE":
                return float(_nce_mean(sp.get("ce_orig")))
            v = sp.get("ce_ev")
            if v is None:
                return float("nan")
            arr = list(v) if not isinstance(v, (str, bytes)) else [v]
            if len(arr) == 0:
                return float("nan")
            return float(sum(float(x) for x in arr) / len(arr))
        v = sp.get("ce")
        if v is None:
            return float("nan")
        f = float(v)
        return f if f == f else float("nan")
    except Exception:
        return float("nan")


def ce_bucket(ce: float):
    """CE 分桶：round(ce/5)*5，最小 5；NaN 透传（调用方走 sum 预测）。"""
    try:
        f = float(ce)
    except Exception:
        return float("nan")
    if f != f:  # NaN
        return float("nan")
    b = float(round(f / 5.0) * 5.0)
    if b < 5.0:
        b = 5.0
    return b


def is_supported_adduct(adduct) -> bool:
    """是否在 forward 覆盖内（默认 [M+H]+/[M+Na]+）。"""
    return str(adduct) in COVERED_ADDUCTS


def is_positive_mode(mode) -> bool:
    """正模式判断：1 / 'positive' 等价（与 gl_runner.is_covered 同口径）。

    mode 可为 1/-1 或字符串；字符串小写后 in ('positive','pos','+','1') 为正。
    """
    if mode == 1:
        return True
    if isinstance(mode, str) and mode.lower() in ("positive", "pos", "+", "1"):
        return True
    return False


def map_condition(sp: Dict, covered: Sequence[str] = COVERED_ADDUCTS) -> Dict:
    """单谱条件映射：返回 {supported, reason, adduct, instrument_token, ce, ce_bucket}。

    supported=False 时 reason 为 'negative_mode' | 'unsupported_adduct:<adduct>'
    （其余标 unsupported，不写成低分）。supported=True 但 ce 为 NaN 时
    ce_bucket 为 NaN，调用方走 SUM_CE merged 预测（'sum' 路径），仍算 supported。
    instrument 恒有 token（缺失 raw -> QTOF，映射假设）。
    """
    ad = sp.get("adduct")
    mode = sp.get("mode", sp.get("ionization_mode"))
    if not is_positive_mode(mode):
        return {"supported": False, "reason": "negative_mode",
                "adduct": ad, "instrument_token": instr_token(sp.get("instrument")),
                "ce": spectrum_ce(sp), "ce_bucket": float("nan")}
    if str(ad) not in list(covered):
        return {"supported": False, "reason": "unsupported_adduct:%s" % (ad,),
                "adduct": ad, "instrument_token": instr_token(sp.get("instrument")),
                "ce": spectrum_ce(sp), "ce_bucket": float("nan")}
    ce = spectrum_ce(sp)
    return {"supported": True, "reason": "",
            "adduct": ad, "instrument_token": instr_token(sp.get("instrument")),
            "ce": ce, "ce_bucket": ce_bucket(ce)}


def cache_key(smiles_canon: str, adduct: str, ce_bucket_val, instrument_token: str,
              model_version: str = MODEL_VERSION,
              preproc_version: str = PREPROC_VERSION) -> str:
    """预测缓存键：结构 + adduct + CE + instrument + 模型/预处理版本。

    ce NaN（sum 路径）规范化为 'sum'；其余 CE 保留 float 值。返回 hex sha256
   （短键场景调用方可截断，但落盘/去重必须用全键）。
    """
    try:
        f = float(ce_bucket_val)
        ce_part = "sum" if f != f else ("%.1f" % f)
    except Exception:
        ce_part = "sum"
    raw = "|".join([str(smiles_canon), str(adduct), ce_part, str(instrument_token),
                    str(model_version), str(preproc_version)])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- 查询侧 prep
def _require_numpy():
    if _np is None:
        raise RuntimeError("need numpy for query prep / entropy similarity")


def clean_peaks(mz, it, prec_mz: float, floor: float = CLEAN_FLOOR,
                topk: int = CLEAN_TOPK,
                max_over_prec: float = CLEAN_MAX_OVER_PREC):
    """库缓存清洗（gl_runner.clean_peaks 的 numpy 等价实现）。

    保留 mz <= prec+max_over_prec 且 it >= floor*max；超 topk 按强度取前 topk；
    按 m/z 排序；强度除以 max（max-normalised）。返回 float32 (mz, it)。
    空/全滤除返回空数组（调用方记 no_useful_peaks）。
    """
    _require_numpy()
    mz = _np.asarray(list(mz), dtype=_np.float64)
    it = _np.asarray(list(it), dtype=_np.float64)
    n = len(mz)
    if n == 0:
        return (_np.zeros(0, dtype=_np.float32), _np.zeros(0, dtype=_np.float32))
    prec = float(prec_mz)
    keep = (mz <= prec + float(max_over_prec)) & (it > 0) & (mz > 0)
    mz, it = mz[keep], it[keep]
    if len(mz) == 0:
        return (_np.zeros(0, dtype=_np.float32), _np.zeros(0, dtype=_np.float32))
    mx = float(it.max())
    if not (mx > 0):
        return (_np.zeros(0, dtype=_np.float32), _np.zeros(0, dtype=_np.float32))
    keep = it >= float(floor) * mx
    mz, it = mz[keep], it[keep]
    if len(mz) > int(topk):
        o = _np.argsort(it)[len(it) - int(topk):]
        mz, it = mz[o], it[o]
    o = _np.argsort(mz)
    mz, it = mz[o], it[o]
    return mz.astype(_np.float32), (it / float(it.max())).astype(_np.float32)


def _entropy_weights(it, power: float = PREP_POWER, ent_weight: bool = PREP_ENTW):
    _require_numpy()
    it = _np.asarray(it, dtype=_np.float64)
    p = _np.power(it, float(power))
    s = float(p.sum())
    if not (s > 0):
        return p
    p = p / s
    if ent_weight:
        S = 0.0
        for v in p:
            if v > 0:
                S -= float(v) * math.log(float(v))
        if S < 3.0:
            w = 0.25 + 0.25 * S
            p = _np.power(p, w)
            s2 = float(p.sum())
            if s2 > 0:
                p = p / s2
    return p


def prep_query(mz, it, floor: float = PREP_FLOOR, topk: int = PREP_TOPK,
               power: float = PREP_POWER, ent_weight: bool = PREP_ENTW):
    """查询向量化（gl_runner.prep_query 的 numpy 等价实现）。

    floor*max 过滤 -> 超 topk 按强度取前 topk（保 m/z 升序）-> 熵加权。
    返回 (mz f64 升序，weights f64)。空返回空数组。
    """
    _require_numpy()
    mz = _np.asarray(list(mz), dtype=_np.float64)
    it = _np.asarray(list(it), dtype=_np.float64)
    n = len(mz)
    if n == 0:
        return (_np.zeros(0, dtype=_np.float64), _np.zeros(0, dtype=_np.float64))
    mx = float(it.max()) if n else 0.0
    if not (mx > 0):
        return (_np.zeros(0, dtype=_np.float64), _np.zeros(0, dtype=_np.float64))
    keep = it >= float(floor) * mx
    mz, it = mz[keep], it[keep]
    if len(mz) > int(topk):
        o = _np.argsort(it)[len(it) - int(topk):]
        o = _np.sort(o)  # 保 m/z 升序前先按原序（已是 m/z 升序输入时等价）
        mz, it = mz[o], it[o]
    o = _np.argsort(mz)
    mz, it = mz[o], it[o]
    return mz.astype(_np.float64), _entropy_weights(it, power, ent_weight).astype(_np.float64)


def prep_spectrum_peaks(mz, it, prec_mz: float):
    """原始峰 -> 两次 clean（库缓存口径）-> prep_query（与 molecule_plan 一致）。

    返回 (qm f64, qp f64)；空表示该谱无可用峰（no_useful_peaks）。
    """
    m1, i1 = clean_peaks(mz, it, prec_mz, CLEAN_FLOOR, CLEAN_TOPK, CLEAN_MAX_OVER_PREC)
    if len(m1) == 0:
        return (_np.zeros(0, dtype=_np.float64) if _np is not None else [],
                _np.zeros(0, dtype=_np.float64) if _np is not None else [])
    m2, i2 = clean_peaks(m1.astype(float), i1.astype(float), prec_mz,
                          CLEAN_FLOOR, CLEAN_TOPK, CLEAN_MAX_OVER_PREC)
    if len(m2) == 0:
        return (m2.astype(_np.float64), i2.astype(_np.float64))
    return prep_query(m2.astype(float), i2.astype(float),
                      PREP_FLOOR, PREP_TOPK, PREP_POWER, PREP_ENTW)


def post_prediction(mz, it, k: int = POST_TOPK):
    """预测谱后处理（与 gl_runner.post 一致）。

    drop m/z<=0 或 intensity<=0 -> 相同 m/z（4 位小数）取 MAX -> top-k ->
    按 m/z 排序 -> max 归一。返回 float32 (mz, it)。空表示无可用预测峰。
    """
    _require_numpy()
    mz = _np.asarray(list(mz), dtype=_np.float64)
    it = _np.asarray(list(it), dtype=_np.float64)
    keep = (mz > 0) & (it > 0)
    mz, it = mz[keep], it[keep]
    if len(mz) == 0:
        return (_np.zeros(0, dtype=_np.float32), _np.zeros(0, dtype=_np.float32))
    key = _np.round(mz, 4)
    u, inv = _np.unique(key, return_inverse=True)
    agg = _np.zeros(len(u), dtype=_np.float64)
    _np.maximum.at(agg, inv, it.astype(_np.float64))
    if len(u) > int(k):
        o = _np.argsort(-agg, kind="stable")[:int(k)]
        u, agg = u[o], agg[o]
    o = _np.argsort(u)
    u, agg = u[o], agg[o]
    return u.astype(_np.float32), (agg / float(agg.max())).astype(_np.float32)


def merge_predictions(specs):
    """sum-CE 合并（与 gl_runner.merge 一致；CE NaN 路径用）。

    specs 为 [(mz, it)]；每张先自归一再按 4 位小数求和，最后 max 归一。
    输入为空或全空时返回空数组。
    """
    _require_numpy()
    parts_mz, parts_it = [], []
    for mz, it in specs:
        mz = _np.asarray(list(mz), dtype=_np.float64)
        it = _np.asarray(list(it), dtype=_np.float64)
        if len(mz) == 0:
            continue
        mx = float(it.max())
        if not (mx > 0):
            continue
        parts_mz.append(mz)
        parts_it.append(it / mx)
    if not parts_mz:
        return (_np.zeros(0, dtype=_np.float64), _np.zeros(0, dtype=_np.float64))
    mz = _np.concatenate(parts_mz)
    it = _np.concatenate(parts_it)
    key = _np.round(mz, 4)
    u, inv = _np.unique(key, return_inverse=True)
    w = _np.bincount(inv, weights=it)
    return u.astype(_np.float64), (w / float(w.max())).astype(_np.float64)


def entropy_similarity(qmz, qp, cmz, cp, tol: float = SIM_TOL_DA,
                       ppm_tol: float = SIM_PPM) -> Dict:
    """熵相似度 + 匹配峰数 + 解释强度（与 gl_runner.entropy_sim 同定义）。

    对齐容差 max(tol, qmz*ppm*1e-6)；贪心双指针；未匹配峰按 0.5 权重计入 SAB。
    返回 {similarity, n_matched, explained_query, query_entropy,
    pred_entropy}。explained_query 为匹配上的 query 熵权重和（权重和为 1，
    故即匹配权重占比）。空输入 similarity=0.0，n_matched=0。
    """
    _require_numpy()
    qmz = _np.asarray(list(qmz), dtype=_np.float64)
    qp = _np.asarray(list(qp), dtype=_np.float64)
    cmz = _np.asarray(list(cmz), dtype=_np.float64)
    cp = _np.asarray(list(cp), dtype=_np.float64)
    n, m = len(qmz), len(cmz)
    if n == 0 or m == 0:
        return {"similarity": 0.0, "n_matched": 0, "explained_query": 0.0,
                "query_entropy": 0.0, "pred_entropy": 0.0}
    SA = float(-sum(float(v) * math.log(float(v)) for v in qp if float(v) > 0))
    SB = float(-sum(float(v) * math.log(float(v)) for v in cp if float(v) > 0))
    SAB = 0.0
    i = j = 0
    n_matched = 0
    matched_w = 0.0
    LOG4 = math.log(4.0)
    while i < n and j < m:
        t = float(tol)
        pt = float(qmz[i]) * float(ppm_tol) * 1e-6
        if pt > t:
            t = pt
        d = float(qmz[i]) - float(cmz[j])
        if d < -t:
            v = float(qp[i]) * 0.5
            i += 1
        elif d > t:
            v = float(cp[j]) * 0.5
            j += 1
        else:
            v = (float(qp[i]) + float(cp[j])) * 0.5
            n_matched += 1
            matched_w += float(qp[i])
            i += 1
            j += 1
        if v > 0:
            SAB -= v * math.log(v)
    while i < n:
        v = float(qp[i]) * 0.5
        if v > 0:
            SAB -= v * math.log(v)
        i += 1
    while j < m:
        v = float(cp[j]) * 0.5
        if v > 0:
            SAB -= v * math.log(v)
        j += 1
    s = 1.0 - (2.0 * SAB - SA - SB) / LOG4
    if s < 1e-9:
        s = 0.0
    if s > 1.0:
        s = 1.0
    return {"similarity": float(s), "n_matched": int(n_matched),
            "explained_query": float(matched_w),
            "query_entropy": float(SA), "pred_entropy": float(SB)}


# ---------------------------------------------------------------- 计数器
STATUS_KEYS = ("unsupported", "inference_failed", "no_useful_peaks",
               "scored", "scored_low")


class StatusCounter:
    """E3 四类计数：unsupported / inference_failed / no_useful_peaks / 有效分。

    有效分再按阈值拆 scored / scored_low（仅报告用，不改变排序）；
    有效但低分与三类缺失严格分开，调用方不得把 None 当 0 分。
    """

    def __init__(self, low_threshold: float = 0.1):
        self.low_threshold = float(low_threshold)
        self.counts: Dict[str, int] = {k: 0 for k in STATUS_KEYS}
        self.reasons: Dict[str, int] = {}

    def add_unsupported(self, reason: str = "") -> None:
        self.counts["unsupported"] += 1
        if reason:
            self.reasons[str(reason)] = self.reasons.get(str(reason), 0) + 1

    def add_failure(self, reason: str = "") -> None:
        self.counts["inference_failed"] += 1
        if reason:
            self.reasons[str(reason)] = self.reasons.get(str(reason), 0) + 1

    def add_no_peaks(self, reason: str = "") -> None:
        self.counts["no_useful_peaks"] += 1
        if reason:
            self.reasons[str(reason)] = self.reasons.get(str(reason), 0) + 1

    def add_scored(self, similarity: float) -> None:
        if similarity is None or not math.isfinite(float(similarity)):
            self.add_failure("nonfinite_similarity")
            return
        if float(similarity) < self.low_threshold:
            self.counts["scored_low"] += 1
        else:
            self.counts["scored"] += 1

    def to_dict(self) -> Dict:
        return {"counts": dict(self.counts), "reasons": dict(sorted(self.reasons.items())),
                "low_threshold": float(self.low_threshold)}


# ---------------------------------------------------------------- 作业构造
def molecule_plan(spectra: Sequence[Dict]) -> Tuple[List[Dict], StatusCounter]:
    """覆盖谱 -> 计划列表 [{qm, qp, adduct, instr, ce_bucket, spectrum_idx}]。

    unsupported 谱记入 counter（原因细分）；无可用峰的覆盖谱记 no_useful_peaks。
    qm/qp 为 prep 后向量（numpy 数组）；调用方缓存复用，不得每候选重算。
    """
    counter = StatusCounter()
    plan: List[Dict] = []
    for idx, sp in enumerate(list(spectra)):
        mapped = map_condition(sp)
        if not mapped["supported"]:
            counter.add_unsupported(str(mapped.get("reason", "")))
            continue
        try:
            mz = sp.get("mz", sp.get("ms2_mzs", []))
            it = sp.get("it", sp.get("ms2_normalized_intensities", []))
            prec = float(sp.get("prec", sp.get("precursor_mz")))
        except Exception as e:
            counter.add_no_peaks("bad_spectrum_fields:%r" % (e,)[:80])
            continue
        try:
            qm, qp = prep_spectrum_peaks(mz, it, prec)
        except Exception as e:
            counter.add_no_peaks("prep_exception:%r" % (e,)[:80])
            continue
        if len(qm) == 0:
            counter.add_no_peaks("empty_after_prep")
            continue
        plan.append({"qm": qm, "qp": qp, "adduct": mapped["adduct"],
                     "instr": mapped["instrument_token"],
                     "ce_bucket": mapped["ce_bucket"], "spectrum_idx": int(idx)})
    return plan, counter


def molecule_jobs(plan: Sequence[Dict], candidates: Sequence[str]) -> set:
    """去重作业集合 {(smiles, adduct, instr, ce)}；CE NaN 展开为 SUM_CE。"""
    jobs = set()
    for row in plan:
        ad, ins, ceb = row["adduct"], row["instr"], row["ce_bucket"]
        try:
            f = float(ceb)
            finite = f == f
        except Exception:
            finite = False
        for c in candidates:
            if finite:
                jobs.add((str(c), str(ad), str(ins), float(ceb)))
            else:
                for ce in SUM_CE:
                    jobs.add((str(c), str(ad), str(ins), float(ce)))
    return jobs


def batch_size_for(n_atoms: int, bs_max: int = DEFAULT_BS,
                   budget: int = DEFAULT_BS * ATOM_BUDGET_PER_BS) -> int:
    """GLACIER 批大小（与 glacier_load.batch_size_for 一致：B*N^2 预算）。"""
    try:
        n = max(1, int(n_atoms))
    except Exception:
        n = 1
    return int(max(1, min(int(bs_max), int(budget) // max(1, n * n))))


# ---------------------------------------------------------------- 融合/预算
def _zscores(vals: Sequence[Optional[float]]) -> List[float]:
    """pandas 语义 z（与 gl_fuse._z 一致）：缺失->0；n<2/std=0->全 0。"""
    ok = []
    xs = []
    for v in vals:
        try:
            o = v is not None and math.isfinite(float(v))
        except Exception:
            o = False
        ok.append(bool(o))
        if o:
            xs.append(float(v))  # type: ignore[arg-type]
    n = len(xs)
    if n < 2:
        return [0.0] * len(vals)
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))
    if not (sd > 0):
        return [0.0] * len(vals)
    out = []
    for v, o in zip(vals, ok):
        out.append(((float(v) - mu) / sd) if o else 0.0)
    return out


def rerank_same_formula(order_keys: Sequence[str], base_scores: Sequence[float],
                        formulas: Sequence[str],
                        forward_scores: Sequence[Optional[float]],
                        lam: float = 0.5, top_n: int = 60,
                        min_covered: int = 2) -> List[int]:
    """同式组 slot-preserving 重排（单 forward scorer；gl_fuse 单 dict 等价）。

    输入为当前排序下的对齐列表（index 0 = 最优）；fused = z(base) + lam*z(fwd)。
    仅当组内 >=2 成员且 forward 覆盖 >=max(2,min_covered) 才重排组内顺序；
    重排后放回原槽位，其余不动。返回新 order（原下标排列）。
    forward 全缺失时返回原序（不编造）。
    """
    n = len(order_keys)
    order = list(range(n))
    if n == 0:
        return order
    m = min(int(top_n), n)
    groups: Dict[str, List[int]] = {}
    for i in range(m):
        groups.setdefault(str(formulas[i]), []).append(i)
    for _f, idx in groups.items():
        if len(idx) < 2:
            continue
        vals = [forward_scores[i] for i in idx]
        n_cov = sum(1 for v in vals
                    if v is not None and math.isfinite(float(v)))
        if n_cov < max(2, int(min_covered)):
            continue
        fused = _zscores([base_scores[i] for i in idx])
        zf = _zscores(vals)
        fused = [a + float(lam) * b for a, b in zip(fused, zf)]
        new = [idx[k] for k in sorted(range(len(idx)),
                                      key=lambda k: (-fused[k], k))]
        for slot, src in zip(idx, new):
            order[slot] = src
    return order


def fixed_topk(candidates: Sequence[str], k: int = 25) -> List[str]:
    """固定预算：取当前排序前 k（k<=0 返回空；超长返回全部）。"""
    kk = int(k)
    if kk <= 0:
        return []
    return list(candidates[:kk])


def dynamic_topk_by_margin(candidates: Sequence[str], scores: Sequence[float],
                           k_min: int = 10, k_max: int = 50,
                           margin_eps: float = 0.05) -> Dict:
    """动态预算（margin 规则）：首二差距大则取 k_min，否则放宽到 k_max。

    margin = s0 - s1（降序输入假设；长度<2 时 margin=inf -> k_min）。
    只定预算规模，不改顺序；返回 {shortlist, k, margin, rule}。
    同式歧义规则见 dynamic_topk_by_formula_ambiguity。
    """
    cands = list(candidates)
    sc = [float(v) for v in scores]
    if len(sc) < 2:
        margin = float("inf")
    else:
        margin = float(sc[0]) - float(sc[1])
    k = int(k_min) if (margin == float("inf") or margin >= float(margin_eps)) else int(k_max)
    k = max(0, min(k, len(cands)))
    return {"shortlist": cands[:k], "k": k, "margin": margin,
            "rule": "margin:k_min=%d,k_max=%d,eps=%g" % (k_min, k_max, margin_eps)}


def dynamic_topk_by_formula_ambiguity(candidates: Sequence[str], scores: Sequence[float],
                                      formulas: Sequence[str], k_min: int = 10,
                                      k_max: int = 50) -> Dict:
    """动态预算（同式歧义规则）：top-k_max 窗内同式组越大越放宽。

    若 top-k_min 内已含同式多成员（>=2 同 formula），放宽到 k_max 以纳入同组
    全部前 k_max 成员做 forward；否则取 k_min。返回 {shortlist, k, rule,
    max_group_in_window}。公式缺失（''/None）不计组。
    """
    cands = list(candidates)
    fms = [str(f) for f in formulas]
    from collections import Counter as _C
    win = fms[:int(k_max)]
    cnt = _C(w for w in win if w not in ("", "None"))
    mg = max(cnt.values()) if cnt else 0
    head = fms[:int(k_min)]
    cnt_h = _C(w for w in head if w not in ("", "None"))
    mg_h = max(cnt_h.values()) if cnt_h else 0
    k = int(k_max) if mg_h >= 2 else int(k_min)
    k = max(0, min(k, len(cands)))
    return {"shortlist": cands[:k], "k": k, "max_group_in_window": int(mg),
            "rule": "formula_ambiguity:k_min=%d,k_max=%d" % (k_min, k_max)}


# ---------------------------------------------------------------- 原始模型 adapter
class GlacierAdapter:
    """GLACIER 原始模型 adapter（延迟加载；缺依赖时只记失败不抛错）。

    用法：
      ad = GlacierAdapter(ckpt_path='artifacts/runs/005/glacier.pt', device='cuda')
      ok, info = ad.load()   # ok=False 时 info 含 reason（供计数）
      out = ad.predict_scores(plan, candidates, cache, deadline_s=...)
      # out: {smiles: {'similarity'|None, 'n_matched', 'explained_query',
      #        'status', 'reason'}}；status in scored/unsupported/
      #        inference_failed/no_useful_peaks（与 StatusCounter 对齐）。

    predict_scores 本身不做条件映射之外的过滤；plan 应由 molecule_plan 产生
    （已排除 unsupported/空峰谱）。候选 SMILES 规范化失败 -> 该候选记
    inference_failed（reason='smiles_rejected'），不污染其他候选。
    缓存：self.cache 以 (smiles, adduct, instr, ce) 存预测峰或 None；
    self.pcache 以 (canonical, adduct, instr, ce) 去重；键版本见 cache_key。
    """

    def __init__(self, ckpt_path: str = "artifacts/runs/005/glacier.pt",
                 device: str = "cuda", batch_size: int = DEFAULT_BS,
                 model_version: str = MODEL_VERSION,
                 preproc_version: str = PREPROC_VERSION):
        self.ckpt_path = str(ckpt_path)
        self.device = str(device)
        self.batch_size = int(batch_size)
        self.model_version = str(model_version)
        self.preproc_version = str(preproc_version)
        self._model = None
        self._load_error: str = ""
        self.cache: Dict[Tuple, Optional[Tuple]] = {}
        self.pcache: Dict[Tuple, Optional[Tuple]] = {}
        self.n_pred = 0
        self.n_fail = 0

    def load(self) -> Tuple[bool, Dict]:
        """加载 checkpoint + ms-pred JointModel（GPU 一键入口在此失败即知卡点）。

        返回 (ok, info)。ok=False 时 info['reason'] 为首个缺失/异常说明，
        调用方记 inference_failed，不继续 predict。
        """
        try:
            import torch  # type: ignore
        except Exception as e:
            self._load_error = "no_torch:%r" % (e,)[:160]
            return False, {"ok": False, "reason": self._load_error}
        try:
            import ms_pred.glacier.joint_model  # type: ignore  # noqa
            import ms_pred.glacier.dataset  # type: ignore  # noqa
            import ms_pred.common  # type: ignore  # noqa
        except Exception as e:
            self._load_error = "no_ms_pred_subset:%r" % (e,)[:200]
            return False, {"ok": False, "reason": self._load_error,
                           "hint": "mount Kaggle 数据集 casmi26-glacier 的 src/ms_pred + shim，并用 cp312/RDKit2025.03 子进程"}
        try:
            from rdkit import Chem  # type: ignore  # noqa
            import rdkit  # type: ignore  # noqa
            if not str(rdkit.__version__).startswith("2025.03"):
                return False, {"ok": False,
                               "reason": "rdkit_version_mismatch:%s,need_2025.03" % rdkit.__version__,
                               "hint": "用 gl_runner 子进程隔离 RDKit，主进程保持 2026.03"}
        except Exception as e:
            self._load_error = "no_rdkit:%r" % (e,)[:160]
            return False, {"ok": False, "reason": self._load_error}
        try:
            import torch
            c = torch.load(str(self.ckpt_path), map_location="cpu", weights_only=True)
            sd = c["state_dict"]
            hp = {k: v for k, v in dict(c["hyper_parameters"]).items()
                  if not str(k).startswith("_")}
            import ms_pred.glacier.joint_model as gl_jm  # type: ignore
            m = gl_jm.JointModel(**hp)
            m.load_state_dict(sd, strict=True)
            dev = self.device
            if dev.startswith("cuda"):
                try:
                    if not torch.cuda.is_available():
                        dev = "cpu"
                except Exception:
                    dev = "cpu"
            m = m.to(dev).eval()
            try:
                m.freeze()
            except Exception:
                pass
            self._model = m
            self._device_resolved = dev
            return True, {"ok": True, "device": dev,
                          "params": int(sum(p.numel() for p in m.parameters())),
                          "tensors": int(len(sd))}
        except Exception as e:
            self._load_error = "load_failed:%r" % (e,)[:300]
            return False, {"ok": False, "reason": self._load_error}

    def predict_scores(self, plan: Sequence[Dict], candidates: Sequence[str],
                       deadline_s: float = 0.0) -> Dict[str, Dict]:
        """对 shortlist 内候选打 forward 分（需先 load() 成功）。

        每个候选 x 每条覆盖谱：取缓存/预测峰 -> prep_query -> entropy_sim；
        分子分 = 各覆盖谱相似度的均值（NaN 跳过，与 panel 的 nanmean 一致）；
        同时保留 n_matched 总和与 explained_query 均值。无任何有效谱相似度
        的候选记 no_useful_peaks（有覆盖谱但全失败/空预测）或 inference_failed
       （模型未加载/特征化异常）。返回 {smiles: record}。
        deadline_s>0 时超时即停（剩下的记 inference_failed:deadline）。
        """
        out: Dict[str, Dict] = {}
        if self._model is None:
            for c in candidates:
                out[str(c)] = {"similarity": None, "n_matched": 0,
                               "explained_query": 0.0, "status": "inference_failed",
                               "reason": self._load_error or "model_not_loaded"}
            return out
        _require_numpy()
        t0 = time.time()
        deadline = (t0 + float(deadline_s)) if float(deadline_s or 0) > 0 else float("inf")
        for c in candidates:
            cs = str(c)
            if time.time() > deadline:
                out[cs] = {"similarity": None, "n_matched": 0, "explained_query": 0.0,
                           "status": "inference_failed", "reason": "deadline"}
                continue
            sims: List[float] = []
            n_match_total = 0
            expl_list: List[float] = []
            fail_reasons: List[str] = []
            for row in plan:
                ad, ins = row["adduct"], row["instr"]
                try:
                    f = float(row["ce_bucket"])
                    finite = f == f
                except Exception:
                    finite = False
                if finite:
                    pred = self._cached_predict(cs, ad, ins, float(row["ce_bucket"]))
                    preds = [pred] if pred is not None else []
                    if pred is None:
                        fail_reasons.append("pred_none")
                        continue
                else:
                    preds = []
                    ok = True
                    for ce in SUM_CE:
                        p = self._cached_predict(cs, ad, ins, float(ce))
                        if p is None:
                            ok = False
                            break
                        preds.append(p)
                    if not ok or len(preds) != len(SUM_CE):
                        fail_reasons.append("pred_none_sum")
                        continue
                    mz_m, it_m = merge_predictions(preds)
                    if len(mz_m) == 0:
                        fail_reasons.append("empty_merged_pred")
                        continue
                    preds = [(mz_m, it_m)]
                pm, pp = prep_query(preds[0][0], preds[0][1],
                                    PREP_FLOOR, PREP_TOPK, PREP_POWER, PREP_ENTW)
                if len(pm) == 0:
                    fail_reasons.append("empty_pred_after_prep")
                    continue
                det = entropy_similarity(row["qm"], row["qp"], pm, pp,
                                         SIM_TOL_DA, SIM_PPM)
                s = det["similarity"]
                if s != s:  # NaN 跳过（panel nanmean 口径）
                    continue
                sims.append(float(s))
                n_match_total += int(det["n_matched"])
                expl_list.append(float(det["explained_query"]))
            if sims:
                out[cs] = {"similarity": float(sum(sims) / len(sims)),
                           "n_matched": int(n_match_total),
                           "explained_query": float(sum(expl_list) / len(expl_list)) if expl_list else 0.0,
                           "status": "scored", "reason": "",
                           "n_spectra_scored": int(len(sims)),
                           "n_spectra_planned": int(len(plan))}
            else:
                reason = ";".join(fail_reasons[:3]) or "no_scored_spectra"
                out[cs] = {"similarity": None, "n_matched": 0, "explained_query": 0.0,
                           "status": "no_useful_peaks", "reason": reason,
                           "n_spectra_scored": 0, "n_spectra_planned": int(len(plan))}
        return out

    def _cached_predict(self, smiles: str, adduct: str, instr: str, ce: float):
        """单作业预测（缓存命中直接返回；失败返回 None 并计数）。"""
        key = (str(smiles), str(adduct), str(instr), float(ce))
        if key in self.cache:
            return self.cache[key]
        # 规范化（与 gl_runner.canon 同规则；失败只记该候选）。
        try:
            c = self._canon(smiles)
        except Exception as e:
            self.n_fail += 1
            self.cache[key] = None
            return None
        if c is None:
            self.n_fail += 1
            self.cache[key] = None
            return None
        pk = (c, str(adduct), str(instr), float(ce))
        if pk in self.pcache:
            self.cache[key] = self.pcache[pk]
            return self.cache[key]
        try:
            pred = self._predict_one(c, str(adduct), str(instr), float(ce))
        except Exception:
            self.n_fail += 1
            self.pcache[pk] = None
            self.cache[key] = None
            return None
        self.n_pred += 1
        if pred is None:
            self.n_fail += 1
        self.pcache[pk] = pred
        self.cache[key] = pred
        return pred

    def _canon(self, smi: str) -> Optional[str]:
        from rdkit import Chem  # type: ignore
        import ms_pred.common.chem_utils as chem_utils  # type: ignore
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() not in chem_utils.VALID_ATOM_NUM:
                return None
        mol = Chem.RemoveHs(mol)
        if mol.GetNumAtoms() > 100:
            return None
        return Chem.MolToSmiles(mol)

    def _predict_one(self, smiles_canon: str, adduct: str, instr: str, ce: float):
        """单作业真实前向（需 ms-pred + torch；GPU 一键路径）。"""
        import torch  # type: ignore
        import ms_pred.glacier.dataset as gl_ds  # type: ignore
        import ms_pred.common as common  # type: ignore
        m = self._model
        tp = gl_ds.TreeProcessor(pe_embed_k=m.pe_embed_k, root_encode="graphormer",
                                 embed_elem_group=m.embed_elem_group,
                                 multi_hop_max_dist=m.multi_hop_max_dist)
        tree = {"root_canonical_smiles": smiles_canon, "adduct": adduct,
                "collision_energy": float(ce)}
        feat = tp.featurize_tree(tree, include_inten_targs=False, include_frag_targs=False)
        item = {"name": "", "adduct": common.ion2onehot_pos[adduct], "precursor": 0.0,
                "instrument": common.instrument2onehot_pos.get(
                    instr, common.instrument2onehot_pos.get("Orbitrap", 0))}
        item.update(dict(feat))
        batch = gl_ds.IntenDataset.collate_fn([item])
        dev = getattr(self, "_device_resolved", "cpu")
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(dev, non_blocking=True)
            elif k == "graphormer_input" and v is not None:
                for gk in v:
                    v[gk] = v[gk].to(dev, non_blocking=True)
        with torch.no_grad():
            raw = m.predict_inten_frag_batch(batch)["spec"]
        arr = raw[0].float().cpu().numpy()
        pmz, pit = post_prediction(arr[:, 0], arr[:, 1], POST_TOPK)
        if len(pmz) == 0:
            return None
        return (_np.asarray(pmz, dtype=_np.float64), _np.asarray(pit, dtype=_np.float64))


def summarize_records(records: Dict[str, Dict],
                      low_threshold: float = 0.1) -> Dict:
    """按 E3 口径汇总 predict_scores 输出（四类分开；有效低分单列）。"""
    c = StatusCounter(low_threshold=low_threshold)
    for _smi, r in records.items():
        st = str(r.get("status", ""))
        if st == "scored":
            c.add_scored(float(r.get("similarity", float("nan"))))
        elif st == "unsupported":
            c.add_unsupported(str(r.get("reason", "")))
        elif st == "inference_failed":
            c.add_failure(str(r.get("reason", "")))
        elif st == "no_useful_peaks":
            c.add_no_peaks(str(r.get("reason", "")))
        else:
            c.add_failure("unknown_status:%s" % st)
    return c.to_dict()


def build_gpu_entry(pilot_path: str = "artifacts/runs/005/pilot.json",
                    ckpt_path: str = "artifacts/runs/005/glacier.pt",
                    out_path: str = "artifacts/runs/005/glacier_scores.json",
                    device: str = "cuda", budget_s: float = 0.0) -> Dict:
    """待 GPU 一键执行入口（Colab T4 / Kaggle GPU 上人工运行）。

    输入 pilot.json: {molecules: [{key, spectra: [...], shortlists:
    {top10: [...], top25: [...], top50: [...]}, base_scores: {...},
    formulas: {...}}]}（shortlists 来自 E0 同池 FPNet 排序，Sharp 不在本文件内编造）。
    输出 glacier_scores.json: {key: {candidate: record}} + meta（版本/设备/耗时/
    四类计数）。本函数在模型缺失时仍写出全 None 结果 + meta（exit 0 口径，
    与 gl_runner 一致），调用方读 meta 而非只看退出码。
    """
    import json as _json
    import os as _os
    t0 = time.time()
    meta: Dict = {"model_version": MODEL_VERSION, "preproc_version": PREPROC_VERSION,
                  "ckpt": str(ckpt_path), "device_requested": str(device),
                  "covered_adducts": list(COVERED_ADDUCTS)}
    try:
        with open(pilot_path) as f:
            pilot = _json.load(f)
    except Exception as e:
        meta.update({"status": "no_pilot", "reason": "%r" % (e,)[:200],
                     "seconds": time.time() - t0})
        return {"scores": {}, "meta": meta}
    mols = pilot.get("molecules", []) if isinstance(pilot, dict) else []
    adapter = GlacierAdapter(ckpt_path=ckpt_path, device=device)
    ok, info = adapter.load()
    meta["load"] = info
    scores: Dict[str, Dict] = {}
    counter = StatusCounter()
    per_mol: Dict[str, Dict] = {}
    for mol in mols:
        key = str(mol.get("key", ""))
        spectra = mol.get("spectra", [])
        short = (mol.get("shortlists", {}) or {}).get("top25", [])
        plan, pc = molecule_plan(spectra)
        for _k, _v in pc.counts.items():
            counter.counts[_k] = counter.counts.get(_k, 0) + _v
        if not ok:
            rec = {str(c): {"similarity": None, "n_matched": 0, "explained_query": 0.0,
                            "status": "inference_failed",
                            "reason": str(info.get("reason", "load_failed"))}
                   for c in short}
        elif not plan:
            rec = {}
        else:
            if float(budget_s or 0) > 0 and (time.time() - t0) > float(budget_s):
                rec = {str(c): {"similarity": None, "n_matched": 0, "explained_query": 0.0,
                                "status": "inference_failed", "reason": "budget"}
                       for c in short}
            else:
                rec = adapter.predict_scores(plan, [str(c) for c in short])
        scores[key] = rec
        per_mol[key] = summarize_records(rec)
        for _k, _v in per_mol[key]["counts"].items():
            counter.counts[_k] = counter.counts.get(_k, 0) + int(_v)
    meta.update({"status": "done" if ok else "model_unavailable",
                 "n_molecules": len(mols), "seconds": time.time() - t0,
                 "counts": counter.to_dict()})
    try:
        _d = _os.path.dirname(out_path)
        if _d:
            _os.makedirs(_d, exist_ok=True)
        with open(out_path, "w") as f:
            _json.dump({"scores": scores, "meta": meta, "per_molecule": per_mol}, f)
    except Exception as e:
        meta["write_error"] = "%r" % (e,)[:200]
    return {"scores": scores, "meta": meta}

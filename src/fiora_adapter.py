"""006-F3 FIORA-OS 负离子小试 adapter（手动研究模块，非测试文件）.

固定 checkpoint（见 data/references/research-frontier-2026-10-05.json）:
- commit e19ef82c9a6cb9dbac92bce23e914008f1aeb44e
- fiora/resources/models/fiora_OS_v1.0.0.pt, 53,845,076 B
- params 声明训练库 MSnLib v7；代码 MIT；MSnLib v7 CC BY 4.0 需保留来源
- 支持 [M+H]+/[M-H]-/[M]+/[M]-；instrument 仅 HCD；CE 归一化尺度 100
- 不凭 default 之名选权重；NIST 相关一律排除

硬边界（调用方如实记录）:
- timsTOF->HCD 是域迁移假设，不与 QTOF 映射混同；按实际 eV 输入，不把 NCE 当 eV
- 仅 4 个 adduct 直接使用；[M+Na]+/[M+NH4]+/[M+Cl]-/[M+CH2O2-H]- 等一律 unsupported，不映射
- ce_units==NCE 一律 unsupported（reason=nce_needs_conversion），不直接当 eV
- unsupported / inference_failed / no_useful_peaks 与有效低分分开；缺失记 None，不写 0 分
- 缓存键含结构+adduct+CE+instrument+模型版本

本文件只做条件映射/缓存键/融合/评分口径；真实 FIORA 前向在 Kaggle GPU 子进程/CLI 完成。
Notebook 入口自包含（kernels push 只上传 code_file），本文件是本地镜像，方便复核口径。
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Sequence, Tuple

# ---- 固定 provenance ----
FIORA_COMMIT = "e19ef82c9a6cb9dbac92bce23e914008f1aeb44e"
FIORA_WEIGHT_PATH = "fiora/resources/models/fiora_OS_v1.0.0.pt"
FIORA_WEIGHT_URL = (
    "https://raw.githubusercontent.com/BAMeScience/fiora/"
    "e19ef82c9a6cb9dbac92bce23e914008f1aeb44e/"
    "fiora/resources/models/fiora_OS_v1.0.0.pt"
)
FIORA_WEIGHT_BYTES = 53845076
FIORA_PARAMS_URL = (
    "https://raw.githubusercontent.com/BAMeScience/fiora/"
    "e19ef82c9a6cb9dbac92bce23e914008f1aeb44e/"
    "fiora/resources/models/fiora_OS_v1.0.0_params.json"
)
FIORA_TRAIN_LIB = "MSnLib v7"
FIORA_CODE_LICENSE = "MIT"
FIORA_DATA_LICENSE = "CC BY 4.0 (MSnLib v7, Zenodo 16984129)"
FIORA_MODEL_ID = "fiora-OS-v1.0.0-msnlib-v7"
FIORA_PREPROC_VERSION = "fiora-os-cli-v1:HCD-only-CEscale100-ENTW-TOL0.01-PPM20"

SUPPORTED_ADDUCTS: Tuple[str, ...] = ("[M+H]+", "[M-H]-", "[M]+", "[M]-")
FIORA_INSTRUMENT = "HCD"
CE_SCALE = 100.0

# 相似度口径与 GL runner 对齐（TOL=0.01Da, PPM=20），便于跨 forward 比较；
# 这是评分选择，不是 FIORA 原生指标，如实记录。
SIM_TOL_DA = 0.01
SIM_PPM = 20.0


def is_supported_adduct(adduct) -> bool:
    return str(adduct) in SUPPORTED_ADDUCTS


def mean_ce_ev(sp: Dict) -> float:
    """返回 ce_ev 均值（eV 语义）；空/非有限 -> NaN。NCE 由上游先拦截。"""
    try:
        v = sp.get("ce_ev")
        if v is None:
            return float("nan")
        arr = list(v) if not isinstance(v, (str, bytes)) else [v]
        if len(arr) == 0:
            return float("nan")
        return float(sum(float(x) for x in arr) / len(arr))
    except Exception:
        return float("nan")


def list_ce_ev(sp: Dict) -> List[float]:
    try:
        v = sp.get("ce_ev")
        if v is None:
            return []
        arr = list(v) if not isinstance(v, (str, bytes)) else [v]
        out = []
        for x in arr:
            f = float(x)
            if f == f:
                out.append(f)
        return out
    except Exception:
        return []


def map_fiora_condition(sp: Dict) -> Dict:
    """单谱映射 -> {supported, reason, adduct, instrument, ce_mean, ce_list}.

    supported=False 原因：unsupported_adduct:<adduct> | nce_needs_conversion | no_ce
    instrument 固定 HCD（timsTOF->HCD 为域迁移假设）；ce 保持 eV，不做 NCE 换算。
    """
    ad = sp.get("adduct")
    if not is_supported_adduct(ad):
        return {"supported": False, "reason": "unsupported_adduct:%s" % (ad,),
                "adduct": ad, "instrument": FIORA_INSTRUMENT,
                "ce_mean": float("nan"), "ce_list": []}
    if sp.get("ce_units") == "NCE":
        return {"supported": False, "reason": "nce_needs_conversion",
                "adduct": ad, "instrument": FIORA_INSTRUMENT,
                "ce_mean": float("nan"), "ce_list": []}
    ce_list = list_ce_ev(sp)
    ce_mean = mean_ce_ev(sp)
    if len(ce_list) == 0 or not (ce_mean == ce_mean):
        return {"supported": False, "reason": "no_ce",
                "adduct": ad, "instrument": FIORA_INSTRUMENT,
                "ce_mean": float("nan"), "ce_list": []}
    return {"supported": True, "reason": "",
            "adduct": str(ad), "instrument": FIORA_INSTRUMENT,
            "ce_mean": float(ce_mean), "ce_list": [float(x) for x in ce_list]}


def fiora_cache_key(smiles_canon: str, adduct: str, ce: float,
                    instrument: str = FIORA_INSTRUMENT,
                    model_version: str = FIORA_MODEL_ID) -> str:
    raw = "|".join([str(smiles_canon), str(adduct), "%.1f" % float(ce),
                    str(instrument), str(model_version), str(FIORA_PREPROC_VERSION)])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def zscores(vals: Sequence[Optional[float]]) -> List[float]:
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
    return [((float(v) - mu) / sd) if o else 0.0 for v, o in zip(vals, ok)]


def fuse_base_fiora(base_scores: Sequence[float],
                    fiora_scores: Sequence[Optional[float]],
                    lam: float = 0.5) -> List[float]:
    """残差融合：fused = z(base) + lam*z(fiora)；缺失 fiora 记 0，不覆盖原序。"""
    zb = zscores(list(base_scores))
    zf = zscores(list(fiora_scores))
    return [a + float(lam) * b for a, b in zip(zb, zf)]


def rr_at_25(order: Sequence[str], truth: str, k: int = 25) -> float:
    for i, key in enumerate(list(order)[:int(k)], 1):
        if key == truth:
            return 1.0 / i
    return 0.0

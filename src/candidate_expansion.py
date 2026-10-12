"""008-B 真实候选召回：固定外部结构索引与无标签查询（手动研究用，非测试文件）。

职责与禁区（见 docs/plan/008-submission-readiness-and-recall.md 008-B 六条）：
- 固定来源、快照、许可筛选与 canonical 规则，建立外部结构索引。可复用已下载
  MassBank 快照；候选资产须来自独立完整来源筛选，不能仅从待评价分子真值 SMILES 提取。
- 查询入口只接受谱峰、前体质量、adduct、仪器与 CE；分子式仅允许谱预测输入，
  真值公式仅作明确 oracle 诊断分支。本模块任何检索函数不得读取 class_AB、
  truth key、truth SMILES 来决定候选或分配预算（导入期不读 observations，
  函数签名不接受 truth 字段）。
- 对观测质量窗返回全部合格候选或预先固定无标签截断，同时保留真实干扰。
  未知 adduct 不得把 precursor 当准确中性质量；质量不确定单列。
- 四阶段记录：全库→质量窗→FP top100→最终 top25（覆盖/MRR/候选量/挤下损失）。
  现有 R3 top100 是 mass-ranked，不得冒充 FPNet shortlist。
- A/B 只作分层描述，不估隐藏比例；本地 B 仅称当前 712K 库外。
- CPU-only，不联网；最终索引可离线使用（.npz/.json，不做在线查询）。

canonical 口径：src.chemistry.canonical_group_key（去立体→互变异构规范化→InChIKey14，
RDKit 2026.03.6），与 007-R3 一致。
质量口径：候选质量为 RDKit ExactMolWt float64（与 pool mass.npy / v32 fp_and_mass
verbatim 一致）；查询中性质量 QM 经 ADDUCT_TABLE float64 计算，未知回退不作准确质量。
指纹口径：v32 concat[Morgan r2 4096 + r3 4096 + RDKitFP 2048 maxPath6 + MACCS 167]=10407，
按 fp_bits.npy 选位 6930（sha a61e37 开头），与 r3_pool_pilot.py:128-141 一致。

adduct 扩展说明：
- 主表复用 src.retrieval.ADDUCT_TABLE（15 类，公式 M=(mz*|z|-delta)/n）。
- 新增 [M-H2O-H]-（Bruker 记录显式标注，007-R3 50 集 1 张）：离子为 M 失水+去质子，
  mz = M - H2O - proton，delta = -(H2O)-proton = -19.01784115344，
  M = mz + 19.01784115344。仅因记录自带 PRECURSOR_TYPE 字符串而支持，不猜测
  无标注谱（4 张 NILU 无 PRECURSOR_TYPE 仍为未知，单列质量不确定）。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

# 许可筛选：仅接受可再分发（007-R3 决策：CC0/CC-BY/CC-BY-SA/MIT 或等效）。
# MassBank 记录实测只有 CC* 系；MIT 保留为政策位（本次快照无 MIT 记录，如实记录 0）。
ALLOWED_LICENSES = frozenset({"CC0", "CC BY", "CC BY-SA", "MIT"})

# 预先固定无标签截断（不随 truth 调整，与 v9/R3 WMAX=100 同口径）。
PPM_DEFAULT = 10.0
WMAX_FP_TOP100 = 100
FINAL_TOP25 = 25

# [M-H2O-H]- delta：-(2H+O) - proton。H2O=2*1.00782503223+15.9949146221，
# proton=1.00727646688。float64。
_H = 1.00782503223
_O = 15.9949146221
_PROTON = 1.00727646688
_H2O = 2.0 * _H + _O
_M_H2O_H_DELTA = -_H2O - _PROTON  # ≈ -19.01784115344


def adduct_table_extended() -> Dict[str, Tuple[int, int, float]]:
    """ADDUCT 查询表：主表 + [M-H2O-H]- 显式扩展（不改主表文件）。"""
    from retrieval import ADDUCT_TABLE as _base  # PYTHONPATH=src 时

    t = dict(_base)
    if "[M-H2O-H]-" not in t:
        t["[M-H2O-H]-"] = (1, -1, float(_M_H2O_H_DELTA))
    return t


def query_neutral_mass_strict(
    precursor_mz: float, adduct: Optional[str]
) -> Tuple[Optional[float], bool, bool]:
    """严格中性质量：返回 (QM_or_None, known_adduct, mass_uncertain)。

    - adduct 在扩展表内（含新增 [M-H2O-H]-）→ (QM, True, False)。
    - adduct 为 None/空/不在表内 → (None, False, True)，调用方不得用 precursor
      当准确中性质量做 10ppm 窗；必须单列质量不确定。
    与 src/candidates.query_neutral_mass 的回退 (precursor, False) 不同：
    此处回退不返回 precursor 数值，防止下游误作准确质量。
    """
    table = adduct_table_extended()
    if adduct is None:
        return None, False, True
    s = str(adduct).strip()
    if s == "":
        return None, False, True
    info = table.get(s)
    if info is None:
        return None, False, True
    n, charge, delta = info
    qm = (float(precursor_mz) * abs(int(charge)) - float(delta)) / float(n)
    return float(qm), True, False


def parse_massbank_txt(txt: str, zip_path: str = "") -> Dict:
    """解析 MassBank 2026.03 bulk 单条 txt（只读记录字段，不读任何 truth 文件）。

    返回 {accession, smiles, license, instrument, instrument_type, precursor_mz,
    precursor_type, formula_record, exact_mass_record, ms_type, ion_mode, n_peaks}。
    缺失记 None/空，不抛错（由调用方计数）。
    """
    def _one(pat: str) -> Optional[str]:
        m = re.search(pat, txt, re.M)
        return m.group(1).strip() if m else None

    acc = _one(r"^ACCESSION:\s*(.+)$")
    smi = _one(r"^CH\$SMILES:\s*(.+)$")
    lic = _one(r"^LICENSE:\s*(.+)$")
    instr = _one(r"^AC\$INSTRUMENT:\s*(.+)$")
    itype = _one(r"^AC\$INSTRUMENT_TYPE:\s*(.+)$")
    ms_type = _one(r"^AC\$MASS_SPECTROMETRY:\s*MS_TYPE\s*(.+)$")
    ion_mode = _one(r"^AC\$MASS_SPECTROMETRY:\s*ION_MODE\s*(.+)$")
    prec_raw = _one(r"^MS\$FOCUSED_ION:\s*PRECURSOR_M/Z\s*(.+)$")
    prec_type = _one(r"^MS\$FOCUSED_ION:\s*PRECURSOR_TYPE\s*(.+)$")
    formula_rec = _one(r"^CH\$FORMULA:\s*(.+)$")
    exact_rec = _one(r"^CH\$EXACT_MASS:\s*(.+)$")
    npeak_raw = _one(r"^PK\$NUM_PEAK:\s*(\d+)")
    try:
        prec = float(str(prec_raw).split()[0]) if prec_raw else None
    except Exception:
        prec = None
    try:
        npeaks = int(npeak_raw) if npeak_raw else 0
    except Exception:
        npeaks = 0
    if smi == "N/A":
        smi = None
    return {
        "accession": acc,
        "smiles": smi,
        "license": lic,
        "instrument": instr,
        "instrument_type": itype,
        "precursor_mz": prec,
        "precursor_type": prec_type,
        "formula_record": formula_rec,
        "exact_mass_record": exact_rec,
        "ms_type": ms_type,
        "ion_mode": ion_mode,
        "n_peaks": int(npeaks),
        "zip_path": str(zip_path),
    }


def fp_bits_for_smiles(smiles: str, bits_idx: Sequence[int]) -> Tuple[List[int], str, float]:
    """v32 指纹选位（Morgan r2/r3 4096 + RDKitFP 2048 maxPath6 + MACCS 167）。

    返回 (sel_bits_6930, formula_calc, mass_theo)。RDKit 缺失抛错，不静默回退。
    与 r3_pool_pilot.py:128-141 verbatim 一致。
    """
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator, MACCSkeys, rdMolDescriptors
    from rdkit.Chem.Descriptors import ExactMolWt

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError("unparseable SMILES: %r" % (smiles,))
    m2 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    m3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
    rk = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)
    try:
        import numpy as _np  # type: ignore
    except Exception as e:
        raise RuntimeError("need numpy for fp_bits_for_smiles: %s" % e)
    full = _np.concatenate([
        m2.GetFingerprintAsNumPy(mol).astype(_np.uint8),
        m3.GetFingerprintAsNumPy(mol).astype(_np.uint8),
        rk.GetFingerprintAsNumPy(mol).astype(_np.uint8),
        _np.array(MACCSkeys.GenMACCSKeys(mol), dtype=_np.uint8),
    ])
    if len(full) != 10407:
        raise ValueError("expect 10407 full bits, got %d" % len(full))
    sel = [int(full[int(i)]) for i in bits_idx]
    if len(sel) != 6930:
        raise ValueError("expect 6930 selected bits, got %d" % len(sel))
    formula_calc = rdMolDescriptors.CalcMolFormula(mol)
    mass_theo = float(ExactMolWt(mol))
    return sel, str(formula_calc), float(mass_theo)


def query_window_indices(
    query_mass: Optional[float],
    index_masses: Sequence[float],
    ppm: float = PPM_DEFAULT,
    mass_uncertain: bool = False,
) -> Dict:
    """无标签质量窗：返回全部合格候选（不限数、不打分）或固定截断标记。

    - mass_uncertain=True 或 query_mass is None → 返回空窗 + uncertain 标记，
      调用方必须单列，不得按 precursor 回退计覆盖。
    - 否则返回全部 |m-QM|<=|QM|*ppm*1e-6 的下标，按 (mass_error asc, index asc)
      确定性排序（与 R3 mass-ranked 同排序，但此处仅为窗内枚举，不作 FP shortlist）。
    - 是否截断由调用方按预先固定 WMAX 执行（本模块不按 truth 自适应截断）。
    """
    import math as _math

    if mass_uncertain or query_mass is None:
        return {
            "window_indices": [],
            "window_size": 0,
            "mass_uncertain": True,
            "query_mass": None,
        }
    qm = float(query_mass)
    if not _math.isfinite(qm) or qm == 0.0:
        return {"window_indices": [], "window_size": 0, "mass_uncertain": True, "query_mass": None}
    tol = abs(qm) * float(ppm) / 1e6
    lo = qm - tol
    hi = qm + tol
    scored = []
    for idx, m in enumerate(index_masses):
        try:
            mf = float(m)
        except Exception:
            continue
        if lo <= mf <= hi:
            err = abs(mf - qm) / abs(qm) * 1e6 if qm != 0 else float("inf")
            scored.append((err, int(idx)))
    scored.sort(key=lambda t: (t[0], t[1]))
    order = [i for _, i in scored]
    return {
        "window_indices": order,
        "window_size": int(len(order)),
        "mass_uncertain": False,
        "query_mass": float(qm),
    }


def query_candidates_unlabeled(
    peaks_mz: Sequence[float],
    peaks_it: Sequence[float],
    precursor_mz: float,
    adduct: Optional[str],
    instrument: Optional[str],
    ce: Optional[object],
    index_masses: Sequence[float],
    index_keys: Sequence[str],
    ppm: float = PPM_DEFAULT,
    predicted_formula: Optional[str] = None,
) -> Dict:
    """008-B 查询入口（无标签）：只接受谱峰/前体/adduct/仪器/CE + 谱预测分子式。

    - peaks_mz/peaks_it 仅透传计数与校验（本阶段质量窗不使用峰形；FP 阶段由
      evaluate 脚本按 v32 prep_peaks 另行前向，避免此处混入模型）。
    - predicted_formula 若提供，必须来自谱预测器；本函数仅做等值过滤（大小写
      去空格），不接受真值公式参数。若为 None 则不过滤（本次 008-B 开发即此口径）。
    - 真值公式/真值 key/class_AB 不在本签名中；任何 truth 诊断走
      oracle_diagnose_truth_in_list（明确 oracle 分支）。
    - 未知 adduct → mass_uncertain=True，window 为空（不得用 precursor 计覆盖）。
    返回 {QM, known_adduct, mass_uncertain, window_indices, window_size,
    n_peaks, predicted_formula_used}。
    """
    try:
        n_peaks = int(min(len(list(peaks_mz)), len(list(peaks_it))))
    except Exception:
        n_peaks = 0
    qm, known, uncertain = query_neutral_mass_strict(float(precursor_mz), adduct)
    win = query_window_indices(qm, index_masses, ppm=ppm, mass_uncertain=uncertain)
    pf_used = None
    if predicted_formula is not None:
        pf_used = str(predicted_formula).strip()
        # 公式过滤需要候选 formula 表，由调用方在索引侧执行；此处仅回传标记，
        # 不在此处读 truth（索引 formula 来自 bulk 记录/RDKit 计算，非评价真值）。
        if pf_used == "":
            pf_used = None
    _ = instrument
    _ = ce
    return {
        "QM": win.get("query_mass"),
        "known_adduct": bool(known),
        "mass_uncertain": bool(win.get("mass_uncertain", uncertain)),
        "window_indices": list(win.get("window_indices", [])),
        "window_size": int(win.get("window_size", 0)),
        "n_peaks": int(n_peaks),
        "predicted_formula_used": pf_used,
        "ppm": float(ppm),
    }


def oracle_diagnose_truth_in_list(
    truth_canonical_key: str, candidate_canonical_list: Sequence[str]
) -> Dict:
    """oracle 诊断分支（明确使用真值，仅评测用，禁用于检索/预算）。

    输入真值 canonical + 已排好候选 canonical 列表，返回 {in_list, rank_1based_or_None}。
    调用方必须在报告中标注 oracle，不得称无标签召回。
    """
    t = str(truth_canonical_key)
    lst = [str(k) for k in candidate_canonical_list]
    try:
        rank = lst.index(t) + 1
        return {"in_list": True, "rank_1based": int(rank)}
    except ValueError:
        return {"in_list": False, "rank_1based": None}


def rr_at_k(rank_1based: Optional[int], k: int = 25) -> float:
    """RR@K：rank None 或 >K 按 0 计（G1/007-R3 口径）。"""
    if rank_1based is None:
        return 0.0
    try:
        r = int(rank_1based)
    except Exception:
        return 0.0
    if 1 <= r <= int(k):
        return 1.0 / float(r)
    return 0.0

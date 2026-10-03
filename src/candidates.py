"""004-1b 候选库：结构候选库 C 与参考谱库 L 分离、float64 质量索引、ppm 召回、packed 指纹占位.

手动研究用模块，不是测试文件（不依赖 pytest/unittest，不做断言门控）。
仅依赖标准库 + pyarrow（建库时读 parquet）。不读凭据，不联网。

职责边界（见 docs/plan/004-t4-candidate-ranking.md）：
- StructureLibrary（C）：合法结构库。允许含真值（未见结构轨道中真值可作
  无谱结构存在）；单独计算其覆盖。本模块的候选质量来自 molecular_formula
  的单同位素理论质量（float64），与观测谱无关；公式无法解析/元素未知时
  记 mass_ok=False，不伪造质量。
- ReferenceSpectrumLibrary（L）：参考谱库。只存谱行标量（行号、结构 key、
  adduct、前体 m/z、来源等），不存 ms2 全量；训练/建库时整组留出由调用方
  按 manifest 行号/ key 过滤，本模块不硬编码任何 ID。
- 质量过滤一律 float64；5/10/20ppm 真值召回逐步观察，不按质量误差硬截
  同分子式异构体（窗口内候选数单独统计）。
- packed 指纹位存储只是占位：6930 位按 bit 打包为 bytes， full pool 不得
  一次性 unpack 到 GPU（见 004 计划 4.59GiB 估算）；本模块只提供打包、
  解包与内存估算，不做模型前向。

adduct 中性质量：优先复用 src/retrieval.ADDUCT_TABLE（同一公式
M=(mz*|z|-delta)/n）；独立运行时若 import 不到，则用本文件内镜像表
（已注明镜像，15 类常见加合）。未知加合返回 (precursor_mz, False)，
调用方必须统计 unknown_adduct 比例，不得静默当作精确质量。
"""

from __future__ import annotations

import bisect
import math
import re
import struct
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # 本地 numpy/RDKit 环境可能无 pyarrow，import 必须不报错
    pq = None  # type: ignore


def _require_pyarrow():
    if pq is None:
        raise RuntimeError("need pyarrow to read parquet (not installed in this env)")

# 单同位素质量（float64）。H/C/N/O/Na/K/Cl35 取 IUPAC/CODATA；
# 稀有元素取最轻同位素近似（B10/Si28/Se74/Sn112 等），仅用于候选质量索引，
# 非官方理论质量口径，manifest/报告中不得称为精确鉴定质量。
MONOISOTOPIC_MASS: Dict[str, float] = {
    "H": 1.00782503223,
    "C": 12.0,
    "N": 14.0030740052,
    "O": 15.9949146221,
    "P": 30.9737619985,
    "S": 31.9720711744,
    "F": 18.99840316273,
    "Cl": 34.968852682,
    "Br": 78.9183376,
    "I": 126.9044719,
    "As": 74.921595,
    "B": 10.0129370,
    "Si": 27.9769265,
    "Se": 73.9224764,
    "Sn": 111.9048239,
}

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")

# retrieval.ADDUCT_TABLE 的镜像（15 类）。主用 retrieval，缺失时回退到此表。
# 公式 M=(mz*|z|-delta)/n；delta 定义见 src/retrieval.py。
_MIRRORED_ADDUCT_DELTA: Dict[str, Tuple[int, int, float]] = {
    "[M+H]+": (1, 1, 1.00727646688),
    "[M-H]-": (1, -1, -1.00727646688),
    "[M+Na]+": (1, 1, 22.9897692809 - 0.00054857990946),
    "[M+NH4]+": (1, 1, 14.0030740052 + 4.0 * 1.00782503223 - 0.00054857990946),
    "[M+K]+": (1, 1, 38.9637074864 - 0.00054857990946),
    "[M+Cl]-": (1, -1, 34.968852682 + 0.00054857990946),
    "[M+CH2O2-H]-": (
        1, -1,
        (12.0 + 2.0 * 1.00782503223 + 2.0 * 15.9949146221) - 1.00727646688,
    ),
    "[2M+H]+": (2, 1, 1.00727646688),
    "[2M+Na]+": (2, 1, 22.9897692809 - 0.00054857990946),
    "[2M-H]-": (2, -1, -1.00727646688),
    "[M-H2O+H]+": (
        1, 1,
        -(2.0 * 1.00782503223 + 15.9949146221) + 1.00727646688,
    ),
    "[M]+": (1, 1, -0.00054857990946),
    "[M+C2H4O2-H]-": (
        1, -1,
        (2.0 * 12.0 + 4.0 * 1.00782503223 + 2.0 * 15.9949146221)
        - 1.00727646688,
    ),
    "[M-2H2O+H]+": (
        1, 1,
        -2.0 * (2.0 * 1.00782503223 + 15.9949146221) + 1.00727646688,
    ),
    "[M+2H]2+": (1, 2, 2.0 * 1.00727646688),
}


def _adduct_table() -> Dict[str, Tuple[int, int, float]]:
    try:
        import retrieval as _ret  # PYTHONPATH=src 时可用

        t = getattr(_ret, "ADDUCT_TABLE", None)
        if isinstance(t, dict) and "[M+H]+" in t:
            return t
    except Exception:
        pass
    try:
        from retrieval import ADDUCT_TABLE as _t2

        return _t2
    except Exception:
        pass
    return _MIRRORED_ADDUCT_DELTA


def query_neutral_mass(
    precursor_mz: float, adduct: Optional[str]
) -> Tuple[float, bool]:
    """由前体 m/z + adduct 求中性质量（float64）。

    返回 (neutral_mass, known_adduct)。未知加合（含 None/空）回退为
    (precursor_mz, False)，调用方必须统计 unknown 比例。
    """
    mz = float(precursor_mz)
    table = _adduct_table()
    info = table.get(str(adduct)) if adduct is not None else None
    if info is None:
        return mz, False
    n, charge, delta = info
    return (mz * abs(int(charge)) - float(delta)) / float(n), True


def parse_formula(formula: str) -> Tuple[Optional[Dict[str, int]], str]:
    """解析分子式，返回 (composition, note)。

    处理 train 中尾部电荷后缀：'C12H19N2O2+'（单电荷，约 11,748 行）及
    'C12H30N2+2' / 'C30H60N3O3+3'（多电荷，12 个结构）：剥离尾部
    r'[+-]\d*' 后解析，note 记 'stripped_charge_suffix'。
    无法完整消费字符串时返回 (None, 'unparseable:...')，调用方计数。
    空/None 返回 (None, 'missing')。
    """
    if formula is None:
        return None, "missing"
    s = str(formula).strip()
    if s == "":
        return None, "missing"
    note = "ok"
    m_charge = re.search(r"[+-]\d*$", s)
    # 只有尾部带电荷且头部仍像分子式（含 C/H）时才剥离，避免误伤正常式。
    if m_charge and re.search(r"C", s[: m_charge.start()]):
        s = s[: m_charge.start()]
        note = "stripped_charge_suffix"
        if s == "":
            return None, "missing"
    pos = 0
    comp: Dict[str, int] = {}
    for m in _FORMULA_TOKEN.finditer(s):
        if m.start() != pos:
            return None, "unparseable:%r" % formula
        el, num = m.group(1), m.group(2)
        comp[el] = comp.get(el, 0) + (int(num) if num != "" else 1)
        pos = m.end()
    if pos != len(s) or not comp:
        return None, "unparseable:%r" % formula
    return comp, note


def formula_mass(formula: str) -> Tuple[Optional[float], str]:
    """分子式的单同位素理论质量（float64），失败返回 (None, reason)。"""
    comp, note = parse_formula(formula)
    if comp is None:
        return None, note
    total = 0.0
    for el, cnt in comp.items():
        base = MONOISOTOPIC_MASS.get(el)
        if base is None:
            return None, "unknown_element:%s" % el
        total += float(base) * int(cnt)
    return float(total), note


def ppm_error(m_lib: float, m_q: float) -> float:
    """|m_lib-m_q|/m_q*1e6（float64）。m_q==0 返回 inf，调用方按 miss 计。"""
    q = float(m_q)
    if q == 0.0:
        return math.inf
    return abs(float(m_lib) - q) / abs(q) * 1e6


@dataclass
class StructureCandidate:
    key: str
    molecular_formula: str = ""
    smiles: str = ""
    neutral_mass: Optional[float] = None
    mass_ok: bool = False
    mass_note: str = ""
    n_source_rows: int = 0


@dataclass
class StructureLibrary:
    """结构候选库 C（与参考谱库 L 分离的对象）。"""

    candidates: List[StructureCandidate] = field(default_factory=list)
    key_to_index: Dict[str, int] = field(default_factory=dict)
    # 质量索引：仅 mass_ok 的 (mass, idx)，按 mass 升序；mass_list 为 float64。
    mass_sorted: List[Tuple[float, int]] = field(default_factory=list)
    mass_list: List[float] = field(default_factory=list)
    n_mass_unknown: int = 0
    formula_notes: Dict[str, int] = field(default_factory=dict)

    def build_index(self) -> None:
        pairs = [
            (float(c.neutral_mass), i)
            for i, c in enumerate(self.candidates)
            if c.mass_ok and c.neutral_mass is not None
        ]
        pairs.sort(key=lambda p: p[0])
        self.mass_sorted = pairs
        self.mass_list = [p[0] for p in pairs]
        self.key_to_index = {c.key: i for i, c in enumerate(self.candidates)}

    def __len__(self) -> int:
        return len(self.candidates)

    def window_indices(self, query_mass: float, ppm: float) -> List[int]:
        """质量窗口内候选下标（float64 bisect，不截断、不排序打分）。"""
        q = float(query_mass)
        tol = abs(q) * float(ppm) / 1e6
        lo = q - tol
        hi = q + tol
        a = bisect.bisect_left(self.mass_list, lo)
        b = bisect.bisect_right(self.mass_list, hi)
        return [idx for _, idx in self.mass_sorted[a:b]]

    def window_size(self, query_mass: float, ppm: float) -> int:
        q = float(query_mass)
        tol = abs(q) * float(ppm) / 1e6
        a = bisect.bisect_left(self.mass_list, q - tol)
        b = bisect.bisect_right(self.mass_list, q + tol)
        return max(0, b - a)


@dataclass
class ReferenceSpectrumEntry:
    row: int
    structure_key: str
    adduct: str = ""
    precursor_mz: Optional[float] = None
    ingest_lib: str = ""
    instrument_family: str = "other"


@dataclass
class ReferenceSpectrumLibrary:
    """参考谱库 L（与结构候选库 C 分离的对象，只存标量行信息）。"""

    spectra: List[ReferenceSpectrumEntry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.spectra)

    def n_structures(self) -> int:
        return len({s.structure_key for s in self.spectra})

    def exclude_keys(self, keys: Sequence[str]) -> "ReferenceSpectrumLibrary":
        drop = set(keys)
        return ReferenceSpectrumLibrary(
            [s for s in self.spectra if s.structure_key not in drop]
        )


def build_structure_library_from_train(
    train_path: str = "data/raw/train.parquet",
    batch_size: int = 16384,
) -> StructureLibrary:
    """从 train 去重 inchikey14 建结构候选库（流式，只读标量列）。

    每个 key 取首次出现的 (molecular_formula, normalized_smiles) 为代表，
    理论质量由 formula_mass() 计算；同一 key 多行只计 n_source_rows。
    空 key 不入库（行数在 notes 中单独记，不静默丢弃）。
    """
    _require_pyarrow()
    pf = pq.ParquetFile(train_path)
    first: Dict[str, Dict] = {}
    order: List[str] = []
    null_keys = 0
    empty_keys = 0
    for batch in pf.iter_batches(
        columns=["inchikey14", "molecular_formula", "normalized_smiles"],
        batch_size=batch_size,
    ):
        iks = batch.column(0).to_pylist()
        fms = batch.column(1).to_pylist()
        sms = batch.column(2).to_pylist()
        for k, fm, sm in zip(iks, fms, sms):
            if k is None:
                null_keys += 1
                continue
            ks = str(k)
            if ks == "":
                empty_keys += 1
                continue
            ent = first.get(ks)
            if ent is None:
                first[ks] = {
                    "formula": "" if fm is None else str(fm),
                    "smiles": "" if sm is None else str(sm),
                    "n": 1,
                }
                order.append(ks)
            else:
                ent["n"] += 1
    lib = StructureLibrary()
    notes: Counter = Counter()
    n_unknown = 0
    for ks in order:
        ent = first[ks]
        mass, note = formula_mass(ent["formula"])
        notes[note] += 1
        ok = mass is not None
        if not ok:
            n_unknown += 1
        lib.candidates.append(
            StructureCandidate(
                key=ks,
                molecular_formula=ent["formula"],
                smiles=ent["smiles"],
                neutral_mass=mass,
                mass_ok=ok,
                mass_note=note,
                n_source_rows=ent["n"],
            )
        )
    lib.n_mass_unknown = n_unknown
    lib.formula_notes = dict(notes)
    lib.build_index()
    # 建库口径附在对象外由调用方记录；这里把空键计数挂到 notes 供报告用。
    lib.formula_notes["_null_inchikey14_rows"] = null_keys
    lib.formula_notes["_empty_inchikey14_rows"] = empty_keys
    return lib


def load_np_queries(
    train_path: str,
    np_keys: Sequence[str],
    batch_size: int = 8192,
) -> Tuple[List[Dict], Dict]:
    """载入 NP query 谱的观测中性质量（只读标量列，不读 ms2）。

    返回 (queries, info)。每个 query：{row, true_key, adduct, precursor_mz,
    query_mass, known_adduct}。precursor 缺失时 query_mass=None，按 miss 计。
    """
    want = set(np_keys)
    queries: List[Dict] = []
    n_unknown_adduct = 0
    n_missing_prec = 0
    _require_pyarrow()
    pf = pq.ParquetFile(train_path)
    offset = 0
    for batch in pf.iter_batches(
        columns=["inchikey14", "adduct", "precursor_mz"], batch_size=batch_size
    ):
        iks = batch.column(0).to_pylist()
        ads = batch.column(1).to_pylist()
        prs = batch.column(2).to_pylist()
        for i, (k, ad, pr) in enumerate(zip(iks, ads, prs)):
            g = offset + i
            if k is None or str(k) not in want:
                continue
            if pr is None:
                n_missing_prec += 1
                queries.append(
                    {
                        "row": g,
                        "true_key": str(k),
                        "adduct": None if ad is None else str(ad),
                        "precursor_mz": None,
                        "query_mass": None,
                        "known_adduct": False,
                    }
                )
                continue
            qm, known = query_neutral_mass(float(pr), ad)
            if not known:
                n_unknown_adduct += 1
            queries.append(
                {
                    "row": g,
                    "true_key": str(k),
                    "adduct": None if ad is None else str(ad),
                    "precursor_mz": float(pr),
                    "query_mass": float(qm),
                    "known_adduct": bool(known),
                }
            )
        offset += batch.num_rows
    queries.sort(key=lambda q: q["row"])
    return queries, {
        "n_queries": len(queries),
        "n_unknown_adduct": n_unknown_adduct,
        "n_missing_precursor": n_missing_prec,
    }


def mass_recall_stats(
    queries: Sequence[Dict],
    library: StructureLibrary,
    windows: Sequence[float] = (5.0, 10.0, 20.0),
) -> Dict:
    """质量窗口真值召回统计（float64，不做候选截断打分）。

    对每个 query：若 query_mass 缺失 / adduct 未知回退 / 真值不在库 /
    真值质量未知 -> 按 miss 计，并在 miss_reasons 中分类计数；
    否则计算真值理论质量与观测质量的 ppm 误差，判断各窗口命中，
    并用 bisect 统计各窗口内候选数（同分子式异构体拥挤度）。
    返回 per-window 召回与窗口大小分布（median/p90/max）。
    """
    key_to_idx = library.key_to_index
    out: Dict = {
        "n_queries": len(queries),
        "windows": list(windows),
        "miss_reasons": Counter(),
    }
    hits: Dict[float, int] = {float(w): 0 for w in windows}
    denom_true_available = 0
    sizes: Dict[float, List[int]] = {float(w): [] for w in windows}
    errs: List[float] = []
    for q in queries:
        qm = q.get("query_mass")
        tk = q.get("true_key")
        if qm is None:
            out["miss_reasons"]["missing_precursor"] += 1
            for w in windows:
                sizes[float(w)].append(0)
            continue
        idx = key_to_idx.get(tk)
        if idx is None:
            out["miss_reasons"]["true_not_in_pool"] += 1
            for w in windows:
                sizes[float(w)].append(library.window_size(float(qm), float(w)))
            continue
        cand = library.candidates[idx]
        if not cand.mass_ok or cand.neutral_mass is None:
            out["miss_reasons"]["true_mass_unknown"] += 1
            for w in windows:
                sizes[float(w)].append(library.window_size(float(qm), float(w)))
            continue
        if not q.get("known_adduct", True):
            # 未知加合回退的质量仍参与比较，但单独计数，不混入已知口径。
            out["miss_reasons"]["unknown_adduct_query"] += 1
        denom_true_available += 1
        err = ppm_error(float(cand.neutral_mass), float(qm))
        errs.append(err)
        for w in windows:
            wf = float(w)
            sizes[wf].append(library.window_size(float(qm), wf))
            if err <= wf:
                hits[wf] += 1
    res: Dict = {
        "n_queries": len(queries),
        "n_true_available": denom_true_available,
        "n_true_in_pool": sum(
            1
            for q in queries
            if q.get("true_key") in key_to_idx
            and library.candidates[key_to_idx[q["true_key"]]].mass_ok
        ),
        "miss_reasons": dict(out["miss_reasons"]),
        "per_window": {},
    }
    errs_sorted = sorted(errs)
    for w in windows:
        wf = float(w)
        sz = sorted(sizes[wf])
        n = len(sz)
        if n:
            mid = n // 2
            med = sz[mid] if n % 2 == 1 else (sz[mid - 1] + sz[mid]) / 2.0
            p90 = sz[min(n - 1, int(n * 0.9))]
            mx = sz[-1]
        else:
            med, p90, mx = 0, 0, 0
        res["per_window"][str(wf)] = {
            "n_hit": hits[wf],
            "recall_over_queries": hits[wf] / len(queries) if queries else 0.0,
            "recall_over_available": hits[wf] / denom_true_available
            if denom_true_available
            else 0.0,
            "window_size_median": med,
            "window_size_p90": p90,
            "window_size_max": mx,
        }
    if errs_sorted:
        mid = len(errs_sorted) // 2
        med_err = (
            errs_sorted[mid]
            if len(errs_sorted) % 2 == 1
            else (errs_sorted[mid - 1] + errs_sorted[mid]) / 2.0
        )
        res["true_ppm_error_median"] = med_err
        res["true_ppm_error_p90"] = errs_sorted[
            min(len(errs_sorted) - 1, int(len(errs_sorted) * 0.9))
        ]
        res["true_ppm_error_max"] = errs_sorted[-1]
    else:
        res["true_ppm_error_median"] = None
        res["true_ppm_error_p90"] = None
        res["true_ppm_error_max"] = None
    return res


def ppm_scan_numpy(
    pool_masses: Sequence[float],
    pool_keys: Sequence[str],
    query_masses: Sequence[Optional[float]],
    query_keys: Sequence[str],
    windows: Sequence[float] = (5.0, 10.0, 20.0),
) -> Dict:
    """纯 numpy 的质量窗召回扫描（本地 CPU，不读 parquet）。

    输入一律 float64；pool_masses 与 pool_keys 等长，query_masses 与
    query_keys 等长，query_masses 允许 None（缺失前体，按 miss 计）。
    逻辑与 mass_recall_stats 一致，但窗口计数用 numpy searchsorted，
    真值误差用 |m_lib-m_q|/m_q*1e6。返回 per-window 召回与窗大小分布
    （mean/median/p90/max）及真值误差分布。不做任何候选打分截断，
    同分子式异构体拥挤度由窗口大小分布体现。
    """
    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise RuntimeError("need numpy for ppm_scan_numpy: %s" % e)
    pm = np.asarray(list(pool_masses), dtype=np.float64)
    pk = list(pool_keys)
    order = np.argsort(pm, kind="mergesort")
    sorted_masses = pm[order]
    key_to_mass: Dict[str, float] = {}
    for k, mm in zip(pk, pm):
        ks = str(k)
        if ks not in key_to_mass and np.isfinite(mm):
            key_to_mass[ks] = float(mm)
    qm_list: List[Optional[float]] = list(query_masses)
    qk_list: List[str] = [str(k) for k in query_keys]
    n = len(qm_list)
    hits: Dict[float, int] = {float(w): 0 for w in windows}
    sizes: Dict[float, List[int]] = {float(w): [] for w in windows}
    errs: List[float] = []
    miss_reasons: Dict[str, int] = {}
    n_true_in_pool = 0
    for qm, tk in zip(qm_list, qk_list):
        if qm is None or not math.isfinite(float(qm)):
            miss_reasons["missing_precursor"] = miss_reasons.get("missing_precursor", 0) + 1
            for w in windows:
                sizes[float(w)].append(0)
            continue
        qf = float(qm)
        tm = key_to_mass.get(tk)
        # 窗口大小（与真值是否在池无关，照计）
        for w in windows:
            wf = float(w)
            tol = abs(qf) * wf / 1e6
            import bisect as _bis

            a = _bis.bisect_left(sorted_masses.tolist(), qf - tol)
            b = _bis.bisect_right(sorted_masses.tolist(), qf + tol)
            sizes[wf].append(max(0, b - a))
        if tm is None or not math.isfinite(tm):
            miss_reasons["true_not_in_pool"] = miss_reasons.get("true_not_in_pool", 0) + 1
            continue
        n_true_in_pool += 1
        err = abs(float(tm) - qf) / abs(qf) * 1e6 if qf != 0.0 else math.inf
        errs.append(err)
        for w in windows:
            if err <= float(w):
                hits[float(w)] += 1
    res: Dict = {
        "n_queries": n,
        "n_true_in_pool": n_true_in_pool,
        "miss_reasons": dict(miss_reasons),
        "per_window": {},
    }
    for w in windows:
        wf = float(w)
        sz = sorted(sizes[wf])
        nn = len(sz)
        if nn:
            mid = nn // 2
            med = sz[mid] if nn % 2 == 1 else (sz[mid - 1] + sz[mid]) / 2.0
            mean = sum(sz) / nn
            p90 = sz[min(nn - 1, int(nn * 0.9))]
            mx = sz[-1]
        else:
            med, mean, p90, mx = 0, 0.0, 0, 0
        res["per_window"][str(float(wf))] = {
            "n_hit": hits[wf],
            "recall_over_queries": hits[wf] / n if n else 0.0,
            "recall_over_in_pool": hits[wf] / n_true_in_pool if n_true_in_pool else 0.0,
            "window_size_mean": float(mean),
            "window_size_median": med,
            "window_size_p90": p90,
            "window_size_max": mx,
        }
    if errs:
        es = sorted(errs)
        mid = len(es) // 2
        med_e = es[mid] if len(es) % 2 == 1 else (es[mid - 1] + es[mid]) / 2.0
        res["true_ppm_error_median"] = med_e
        res["true_ppm_error_p90"] = es[min(len(es) - 1, int(len(es) * 0.9))]
        res["true_ppm_error_max"] = es[-1]
        res["true_ppm_error_mean"] = sum(es) / len(es)
    else:
        res["true_ppm_error_median"] = None
        res["true_ppm_error_p90"] = None
        res["true_ppm_error_max"] = None
        res["true_ppm_error_mean"] = None
    return res


def wmax_truncation_stats(
    pool_masses: Sequence[float],
    pool_keys: Sequence[str],
    query_masses: Sequence[Optional[float]],
    query_keys: Sequence[str],
    windows: Sequence[float] = (5.0, 10.0, 20.0),
    wmax_list: Sequence[int] = (50, 100, 200, 500, 2000),
) -> Dict:
    """WMAX 截断影响：窗内按 |m_lib-m_q| 升序取前 WMAX，测真值保留率。

    提交 v7 口径：w = where(|allm-QM|<=QM*ppm*1e-6)；若 len(w)>WMAX 则按
    质量误差取前 WMAX（同分子式异构体同为 ~0 误差， trunc 顺序不定但
    只要总数<WMAX 就全保留）。本函数对每个 (ppm,WMAX) 计算：
    true_after_mass（窗内命中）与 true_after_truncation（截断后保留），
    以及被截断的 query 比例与平均截断后窗大小。纯 numpy + bisect，
    不做模型打分。
    """
    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise RuntimeError("need numpy for wmax_truncation_stats: %s" % e)
    pm = np.asarray(list(pool_masses), dtype=np.float64)
    pk = [str(k) for k in pool_keys]
    order = np.argsort(pm, kind="mergesort")
    sorted_masses = pm[order]
    sorted_keys = [pk[i] for i in order]
    key_to_mass: Dict[str, float] = {}
    for k, mm in zip(pk, pm):
        if k not in key_to_mass and np.isfinite(mm):
            key_to_mass[k] = float(mm)
    out: Dict = {"n_queries": len(list(query_masses)), "per_ppm": {}}
    for w in windows:
        wf = float(w)
        ppm_entry: Dict = {"wmax": {}}
        # 先算每 query 的窗内候选（质量，key，误差）
        per_q_windows: List[Dict] = []
        for qm, tk in zip(query_masses, query_keys):
            tks = str(tk)
            if qm is None or not math.isfinite(float(qm)):
                per_q_windows.append({"qm": None, "true_key": tks, "win": [], "err_true": None})
                continue
            qf = float(qm)
            tol = abs(qf) * wf / 1e6
            import bisect as _bis

            sm_list = sorted_masses.tolist()
            a = _bis.bisect_left(sm_list, qf - tol)
            b = _bis.bisect_right(sm_list, qf + tol)
            win_idx = list(range(a, b))
            win = [
                (sorted_keys[i], float(sorted_masses[i]), abs(float(sorted_masses[i]) - qf) / abs(qf) * 1e6 if qf != 0 else math.inf)
                for i in win_idx
            ]
            win.sort(key=lambda t: t[2])
            tm = key_to_mass.get(tks)
            if tm is None:
                err_true = None
            else:
                err_true = abs(float(tm) - qf) / abs(qf) * 1e6 if qf != 0 else math.inf
            per_q_windows.append({"qm": qf, "true_key": tks, "win": win, "err_true": err_true})
        n_hit_mass = sum(1 for q in per_q_windows if q["err_true"] is not None and q["err_true"] <= wf)
        ppm_entry["n_hit_mass"] = n_hit_mass
        ppm_entry["recall_mass"] = n_hit_mass / len(per_q_windows) if per_q_windows else 0.0
        for wmax in wmax_list:
            wm = int(wmax)
            n_keep = 0
            n_truncated_q = 0
            trunc_sizes: List[int] = []
            for q in per_q_windows:
                win = q["win"]
                trunc_sizes.append(min(len(win), wm))
                if len(win) > wm:
                    n_truncated_q += 1
                kept_keys = {k for k, _, _ in win[:wm]}
                if q["err_true"] is not None and q["err_true"] <= wf and q["true_key"] in kept_keys:
                    # 同分子式异构体同为小误差：只要真值在窗内且窗<=WMAX 必保留；
                    # 窗>WMAX 时按误差排序，真值误差小则保留，大则丢（此处如实计）。
                    n_keep += 1
            ppm_entry["wmax"][str(wm)] = {
                "n_hit_after_trunc": n_keep,
                "recall_after_trunc": n_keep / len(per_q_windows) if per_q_windows else 0.0,
                "n_truncated_queries": n_truncated_q,
                "frac_truncated": n_truncated_q / len(per_q_windows) if per_q_windows else 0.0,
                "mean_trunc_size": sum(trunc_sizes) / len(trunc_sizes) if trunc_sizes else 0.0,
            }
        out["per_ppm"][str(wf)] = ppm_entry
    return out


# ---- packed 指纹位存储占位 ----

def pack_bits(bits: Sequence[int]) -> bytes:
    """01 bit 序列打包为 bytes（高位先行），长度不足整字节补 0。

    6930 位 -> 867 字节。输入值按 truthiness 取 0/1，不做阈值学习。
    """
    out = bytearray((len(bits) + 7) // 8)
    for i, b in enumerate(bits):
        if b:
            out[i // 8] |= 1 << (7 - (i % 8))
    return bytes(out)


def unpack_bits(packed: bytes, nbits: int) -> List[int]:
    """pack_bits 的逆操作，返回 0/1 列表（长度 nbits）。"""
    res: List[int] = []
    for i in range(int(nbits)):
        res.append((packed[i // 8] >> (7 - (i % 8))) & 1)
    return res


def packed_memory_bytes(n_candidates: int, nbits: int = 6930) -> int:
    """packed 候选指纹池的字节数估算（不含 Python 对象开销）。"""
    per = (int(nbits) + 7) // 8
    return int(n_candidates) * per


@dataclass
class PackedFingerprintStore:
    """packed 指纹池占位：CPU 常驻 bytes，禁止整池 unpack 到 GPU。

    用法：按 batch 取 pack 行搬运；谱按分片读。 Ferdowsi 池 711,705×6930
    的 packed 量约 587MB（估算，见 packed_memory_bytes），unpack uint8
    则约 4.59GiB——后者是 004 计划的禁止项，本类不提供 to_dense_all()。
    """

    nbits: int = 6930
    rows: List[bytes] = field(default_factory=list)
    keys: List[str] = field(default_factory=list)

    def add(self, key: str, bits: Sequence[int]) -> None:
        if len(bits) != self.nbits:
            raise ValueError(
                "expect %d bits, got %d" % (self.nbits, len(bits))
            )
        self.rows.append(pack_bits(bits))
        self.keys.append(key)

    def __len__(self) -> int:
        return len(self.rows)

    def memory_bytes(self) -> int:
        return sum(len(r) for r in self.rows)

    def get_batch(self, indices: Sequence[int]) -> List[List[int]]:
        """按下标取一批解包指纹（仅 batch 级解包，禁止全池解包）。"""
        return [unpack_bits(self.rows[i], self.nbits) for i in indices]

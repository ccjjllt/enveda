"""004-1b 数据划分：仪器族归一化、重复谱组检测、NP250 留出 manifest.

手动研究用模块，不是测试文件（不依赖 pytest/unittest，不做断言门控）。
仅依赖 pyarrow + 标准库。不读凭据，不联网。

背景（已核实，见 docs/plan/004-t4-candidate-ranking.md“目标域与数据划分”）：
- train.parquet：2,539,608 行、18 列，无 molecule_id/spectrum_id。
- enveda-np-examples：1,184 谱 / 250 个 shipped inchikey14，全 timsTOF。
- 这 250 结构与原 seen/unseen 各 500 结构交集为 0（按 shipped key 比较）。
- 本模块把 NP250 固定为结构留出主观察集，训练侧跨全部来源移除其
  shipped inchikey14。canonical key 当前只能用 shipped key 占位，
  调用方必须标注 caliber="shipped_placeholder"，不得称为官方规范化分组。
  官方规范化需 RDKit 2026.03.3 的去立体 + 互变异构重算（见 src/chemistry.py
  canonical_group_key），待后续重算。

仪器族（5 类，与 v32 INSTR_LIST / instr_family 逐字对齐）：
    ["timsTOF", "Orbitrap", "QTOF", "IT", "other"]
归一化规则复刻 artifacts/004-research/reference-fpnet-v32.py 的 instr_family：
    None -> other；否则小写后依次检查 'timstof'、orbitrap 组
    （orbitrap/qft/ftms/hybrid ft/itft/exactive）、'tof'、trap/qq 组。
qTof/qTOF/qtof/Qtof/QTOF/LC-ESI-QTOF 等同义写法都因含 'tof' 归入 QTOF；
timsTOF 因优先检查 'timstof' 而不被 'tof' 截胡。大小写不敏感。
"nan"/"N/A"/"" 等缺失写法归入 other，并在 manifest 中单独计数，不静默丢弃。

重复谱组：对 ms2_mzs + ms2_normalized_intensities 做 sha256 分块哈希，
一次只驻留一个 batch（默认 2048 行），不全量加载 ms2。哈希前按 m/z
排序并以 float64 原样打包，长度前缀区分空谱；顶层 null 记为缺失，
不参与分组。精确重复（verbatim 拷贝）会落到同一组；近似重复不合并，
调用方不得把本组数当作去重后的唯一谱数。
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import struct
from collections import Counter
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pyarrow.parquet as pq

NP_LIB = "enveda-np-examples"

INSTR_FAMILIES: List[str] = ["timsTOF", "Orbitrap", "QTOF", "IT", "other"]
INSTR_INDEX: Dict[str, int] = {n: i for i, n in enumerate(INSTR_FAMILIES)}

# 缺失仪器写法的显式集合（小写 strip 后）。除此之外的未知字符串同样归 other，
# 这里只为 manifest 计数口径明确，不改变归一化结果。
MISSING_INSTR_TOKENS = {"", "nan", "n/a", "none", "null"}


def normalize_instrument_family(raw) -> str:
    """仪器族归一化，返回 5 类名称之一。

    与 v32 instr_family 逐字对齐（见 artifacts/004-research/reference-fpnet-v32.py）：
    timsTOF 优先，其次 Orbitrap 组，再次含 tof 即 QTOF，再次 trap/qq 即 IT，
    其余 other。None 与缺失写法一律 other。
    """
    if raw is None:
        return "other"
    t = str(raw).strip().lower()
    if t in MISSING_INSTR_TOKENS:
        return "other"
    if "timstof" in t:
        return "timsTOF"
    if (
        "orbitrap" in t
        or "qft" in t
        or "ftms" in t
        or "hybrid ft" in t
        or "itft" in t
        or "exactive" in t
    ):
        return "Orbitrap"
    if "tof" in t:
        return "QTOF"
    if "trap" in t or "qq" in t:
        return "IT"
    return "other"


def instrument_family_index(raw) -> int:
    """仪器族下标（0..4），顺序与 v32 INSTR_LIST 一致。"""
    return INSTR_INDEX[normalize_instrument_family(raw)]


def spectrum_hash(
    mzs: Optional[Sequence[float]],
    intensities: Optional[Sequence[float]],
) -> Optional[str]:
    """单谱精确重复哈希（sha256 hex），缺失返回 None。

    步骤：两列任一顶层 null/空 -> None；按 m/z 升序配对排序；
    struct "<q2d" 打包 (n, mz, it) 后哈希。float64 原样打包，不做
    舍入（近似谱不合并）。调用方按 batch 逐行调用，本函数不驻留全量。
    """
    if mzs is None or intensities is None:
        return None
    try:
        n = len(mzs)
    except TypeError:
        return None
    if n == 0 or len(intensities) != n:
        return None
    try:
        pairs = sorted(
            ((float(m), float(i)) for m, i in zip(mzs, intensities)),
            key=lambda p: p[0],
        )
    except (TypeError, ValueError):
        return None
    h = hashlib.sha256()
    h.update(struct.pack("<q", n))
    for m, it in pairs:
        h.update(struct.pack("<2d", m, it))
    return h.hexdigest()


def _utc_now() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def collect_np_keys(train_path: str, batch_size: int = 65536) -> List[str]:
    """收集 NP250 shipped key：ingest_lib == 'enveda-np-examples' 的去重 inchikey14。

    只读两列标量，流式。返回排序后的 key 列表。空 key 不收（单独计数除外，
    本函数只返回非空 key；空 key 计数由 build_np_manifest 负责）。
    """
    keys: Set[str] = set()
    pf = pq.ParquetFile(train_path)
    for batch in pf.iter_batches(
        columns=["ingest_lib", "inchikey14"], batch_size=batch_size
    ):
        libs = batch.column(0).to_pylist()
        iks = batch.column(1).to_pylist()
        for li, k in zip(libs, iks):
            if li == NP_LIB and k is not None and str(k) != "":
                keys.add(str(k))
    return sorted(keys)


def build_np_manifest(
    train_path: str = "data/raw/train.parquet",
    out_path: str = "artifacts/004-research/np_manifest.json",
    batch_size: int = 8192,
    ms2_batch_size: int = 2048,
) -> Dict:
    """构建 NP250 结构留出 manifest 并落盘。

    Q（主观察 query 谱）：NP250 结构在全部来源中的所有谱行（含 np-examples
    自身）。L（训练/参考谱）侧必须移除这些行号。C（结构候选库）允许保留
    真值作无谱结构，覆盖另计（见 src/candidates.py）。

    三遍流式：
      A. 标量两列收 NP250 key（250 个预期，实测写入 manifest，不硬断言）。
      B. 标量六列按全局行号收集每个 NP key 在跨全部来源的行号、来源、
         仪器族、模式、adduct、前体 m/z。
      C. ms2 两列分块哈希，仅对 holdout 行号计算，归纳重复谱组。

    返回 manifest dict，同时写入 out_path（json，行号升序）。
    大产物在 artifacts/（已忽略），不入 Git。
    """
    pf = pq.ParquetFile(train_path)
    num_rows = pf.metadata.num_rows
    num_row_groups = pf.metadata.num_row_groups
    schema_names = pf.schema_arrow.names

    # ---- A: NP keys ----
    np_keys = collect_np_keys(train_path, batch_size=65536)
    np_key_set = set(np_keys)

    # ---- B: holdout rows ----
    holdout_rows: List[int] = []
    per_key_rows: Dict[str, List[int]] = {k: [] for k in np_keys}
    per_key_sources: Dict[str, Counter] = {k: Counter() for k in np_keys}
    per_key_fams: Dict[str, Counter] = {k: Counter() for k in np_keys}
    per_key_modes: Dict[str, Counter] = {k: Counter() for k in np_keys}
    per_key_adducts: Dict[str, Counter] = {k: Counter() for k in np_keys}
    row_details: Dict[int, Dict] = {}
    fam_total = Counter()
    src_total = Counter()
    null_inchikey14 = 0
    empty_inchikey14 = 0
    offset = 0
    cols_b = [
        "inchikey14",
        "ingest_lib",
        "instrument_type",
        "ionization_mode",
        "adduct",
        "precursor_mz",
    ]
    for batch in pf.iter_batches(columns=cols_b, batch_size=batch_size):
        iks = batch.column(0).to_pylist()
        libs = batch.column(1).to_pylist()
        insts = batch.column(2).to_pylist()
        modes = batch.column(3).to_pylist()
        ads = batch.column(4).to_pylist()
        precs = batch.column(5).to_pylist()
        for i, (k, li, ins, mo, ad, pr) in enumerate(
            zip(iks, libs, insts, modes, ads, precs)
        ):
            g = offset + i
            if k is None:
                null_inchikey14 += 1
                continue
            ks = str(k)
            if ks == "":
                empty_inchikey14 += 1
                continue
            if ks in np_key_set:
                fam = normalize_instrument_family(ins)
                holdout_rows.append(g)
                per_key_rows[ks].append(g)
                per_key_sources[ks][str(li)] += 1
                per_key_fams[ks][fam] += 1
                per_key_modes[ks][str(mo)] += 1
                per_key_adducts[ks][str(ad)] += 1
                fam_total[fam] += 1
                src_total[str(li)] += 1
                row_details[g] = {
                    "row": g,
                    "shipped_key": ks,
                    "ingest_lib": str(li),
                    "instrument_raw": None if ins is None else str(ins),
                    "instrument_family": fam,
                    "ionization_mode": None if mo is None else str(mo),
                    "adduct": None if ad is None else str(ad),
                    "precursor_mz": None if pr is None else float(pr),
                }
        offset += batch.num_rows

    holdout_rows_sorted = sorted(holdout_rows)
    holdout_set = set(holdout_rows_sorted)

    # ---- C: duplicate groups (chunked ms2 hash, holdout only) ----
    hash_to_rows: Dict[str, List[int]] = {}
    n_ms2_missing = 0
    offset = 0
    for batch in pf.iter_batches(
        columns=["ms2_mzs", "ms2_normalized_intensities"],
        batch_size=ms2_batch_size,
    ):
        mzs_col = batch.column(0).to_pylist()
        it_col = batch.column(1).to_pylist()
        for i, (m, it) in enumerate(zip(mzs_col, it_col)):
            g = offset + i
            if g not in holdout_set:
                continue
            h = spectrum_hash(m, it)
            if h is None:
                n_ms2_missing += 1
                continue
            hash_to_rows.setdefault(h, []).append(g)
        offset += batch.num_rows

    dup_groups: List[Dict] = []
    for h in sorted(hash_to_rows.keys()):
        rows = sorted(hash_to_rows[h])
        if len(rows) > 1:
            dup_groups.append(
                {"group_id": "dup_%04d" % len(dup_groups), "sha256": h,
                 "size": len(rows), "rows": rows}
            )
    n_dup_rows = sum(g["size"] for g in dup_groups)
    # 行号 -> 重复组 id（单例谱不在此 dict，调用方不得误读为缺失）
    row_to_dup: Dict[int, str] = {}
    for g in dup_groups:
        for r in g["rows"]:
            row_to_dup[r] = g["group_id"]
    for g, d in row_details.items():
        d["duplicate_group"] = row_to_dup.get(g)

    # ---- per-structure entries ----
    structures: List[Dict] = []
    for k in np_keys:
        rows = sorted(per_key_rows[k])
        precs = [
            row_details[r]["precursor_mz"]
            for r in rows
            if row_details[r]["precursor_mz"] is not None
        ]
        precs_sorted = sorted(precs)
        if precs_sorted:
            mid = len(precs_sorted) // 2
            if len(precs_sorted) % 2 == 1:
                med = precs_sorted[mid]
            else:
                med = (precs_sorted[mid - 1] + precs_sorted[mid]) / 2.0
        else:
            med = None
        structures.append(
            {
                "shipped_key": k,
                # 占位：当前直接复制 shipped key，待官方规范化重算。
                "canonical_key": k,
                "caliber": "shipped_placeholder",
                "n_rows": len(rows),
                "rows_sorted": rows,
                "sources": dict(sorted(per_key_sources[k].items())),
                "instrument_families": dict(sorted(per_key_fams[k].items())),
                "ionization_modes": dict(sorted(per_key_modes[k].items())),
                "adducts": dict(sorted(per_key_adducts[k].items())),
                "precursor_mz_median": med,
            }
        )

    try:
        train_bytes = os.path.getsize(train_path)
    except OSError:
        train_bytes = -1

    manifest: Dict = {
        "task": "004-1b np250 holdout manifest (main observation set)",
        "created_utc": _utc_now(),
        "train": {
            "path": train_path,
            "bytes": train_bytes,
            "num_rows": num_rows,
            "num_row_groups": num_row_groups,
            "schema_names": schema_names,
            "has_molecule_id": "molecule_id" in schema_names,
            "has_spectrum_id": "spectrum_id" in schema_names,
        },
        "key_caliber": "shipped_placeholder",
        "key_note": (
            "canonical_key 当前占位复制 shipped inchikey14；"
            "待 RDKit 2026.03.3 去立体 + TautomerEnumerator 互变异构重算"
            "（src/chemistry.canonical_group_key），不得称为规范化分组。"
        ),
        "np_lib": NP_LIB,
        "n_np_keys": len(np_keys),
        "np_keys_sorted": np_keys,
        "n_np_lib_rows": int(src_total.get(NP_LIB, 0)),
        "n_holdout_rows": len(holdout_rows_sorted),
        "holdout_rows_sorted": holdout_rows_sorted,
        "train_keep_rows": num_rows - len(holdout_rows_sorted),
        "null_inchikey14_rows": null_inchikey14,
        "empty_inchikey14_rows": empty_inchikey14,
        "holdout_sources": dict(sorted(src_total.items())),
        "holdout_instrument_families": dict(sorted(fam_total.items())),
        "instrument_family_rule": (
            "v32 instr_family verbatim: None->other; lower() 后依次匹配 "
            "'timstof'->timsTOF, orbitrap/qft/ftms/hybrid ft/itft/exactive"
            "->Orbitrap, 'tof'->QTOF, 'trap'/'qq'->IT, 其余 other。"
            "qTof/qTOF/qtof/Qtof 等因含 'tof' 归入 QTOF。"
        ),
        "structures": structures,
        "duplicates": {
            "method": (
                "sha256 over sorted (mz,it) float64 pairs with <q length prefix; "
                "chunked iter_batches ms2 only, holdout rows hashed; "
                "top-level null/empty/len-mismatch counted as missing, not grouped."
            ),
            "n_hashable_holdout_rows": len(holdout_rows_sorted) - n_ms2_missing,
            "n_ms2_missing": n_ms2_missing,
            "n_groups": len(dup_groups),
            "n_rows_in_groups": n_dup_rows,
            "groups": dup_groups,
        },
        "usage": (
            "Q=holdout_rows_sorted（NP250 跨全来源所有谱）；"
            "L/训练侧移除这些行号；C 允许保留真值作无谱结构，覆盖另计。"
            "single/scorefusion/merged 共用同一 Q/L/C（见 004 计划）。"
        ),
    }

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def manifest_summary(m: Dict) -> Dict:
    """manifest 的小摘要（手动记录用，不读 ms2）。"""
    n_keys = m.get("n_np_keys", 0)
    n_hold = m.get("n_holdout_rows", 0)
    structs = m.get("structures", [])
    rows_per_struct = sorted(s.get("n_rows", 0) for s in structs)
    if rows_per_struct:
        mid = len(rows_per_struct) // 2
        med = (
            rows_per_struct[mid]
            if len(rows_per_struct) % 2 == 1
            else (rows_per_struct[mid - 1] + rows_per_struct[mid]) / 2.0
        )
    else:
        med = 0
    return {
        "n_np_keys": n_keys,
        "n_np_lib_rows": m.get("n_np_lib_rows"),
        "n_holdout_rows": n_hold,
        "train_keep_rows": m.get("train_keep_rows"),
        "rows_per_structure_min": min(rows_per_struct) if rows_per_struct else 0,
        "rows_per_structure_median": med,
        "rows_per_structure_max": max(rows_per_struct) if rows_per_struct else 0,
        "holdout_sources": m.get("holdout_sources"),
        "holdout_instrument_families": m.get("holdout_instrument_families"),
        "n_duplicate_groups": m.get("duplicates", {}).get("n_groups"),
        "n_rows_in_duplicate_groups": m.get("duplicates", {}).get(
            "n_rows_in_groups"
        ),
        "key_caliber": m.get("key_caliber"),
    }

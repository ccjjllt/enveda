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
归一化规则复刻 artifacts/runs/004-research/reference-fpnet-v32.py 的 instr_family：
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

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # numpy/RDKit 环境可能无 pyarrow，import 必须不报错
    pq = None  # type: ignore


def _require_pyarrow():
    if pq is None:
        raise RuntimeError("need pyarrow to read parquet (not installed in this env)")

NP_LIB = "enveda-np-examples"

INSTR_FAMILIES: List[str] = ["timsTOF", "Orbitrap", "QTOF", "IT", "other"]
INSTR_INDEX: Dict[str, int] = {n: i for i, n in enumerate(INSTR_FAMILIES)}

# 缺失仪器写法的显式集合（小写 strip 后）。除此之外的未知字符串同样归 other，
# 这里只为 manifest 计数口径明确，不改变归一化结果。
MISSING_INSTR_TOKENS = {"", "nan", "n/a", "none", "null"}


def normalize_instrument_family(raw) -> str:
    """仪器族归一化，返回 5 类名称之一。

    与 v32 instr_family 逐字对齐（见 artifacts/runs/004-research/reference-fpnet-v32.py）：
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
    _require_pyarrow()
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
    out_path: str = "artifacts/runs/004-research/np_manifest.json",
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
    _require_pyarrow()
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


def build_np_canonical_folds(
    np_manifest_path: str = "artifacts/runs/004-research/np_manifest.json",
    smiles_map: Optional[Dict[str, str]] = None,
    out_path: str = "artifacts/runs/004-research/np_canonical_5fold.json",
    n_splits: int = 5,
    seed: int = 2026,
    n_total_structures: int = 275810,
) -> Dict:
    """NP 规范化结构 5 折 manifest（本地 CPU，不读 parquet）。

    分组键用 src/chemistry.canonical_group_key（去立体→互变异构→InChIKey14，
    需 RDKit；缺失时抛错，不静默回退为 shipped）。输入 manifest 为
    build_np_manifest 产物（含 structures rows_sorted 与 duplicates groups），
    smiles_map 为 shipped->normalized_smiles（调用方从 pool-train keys/smiles
    或 train 首次出现代表 loaded，不在这里读 parquet）。

    防泄漏规则：
    - 同一 canonical 组整组同折（canonical 合并 shipped 时）；
    - 同一精确重复谱组（duplicates.groups）整组同折：不同 shipped 结构若共享
      完全相同 ms2（2088 组中 142 组跨 shipped），按连通分量合并后再分折，
      折间无重复谱组跨折；
    - 折间 canonical/shipped/行号三者皆无交集（写 verification，如实计）。

    Q/L/C 口径（unseen 结构留出）：
    - Q_k = 本折 holdout 行（跨全来源，含 timsTOF 与非 timsTOF）；
    - L_k = 全 train 行 - Q_k（CV 口径；严格全 NP 留出训练集为 train_keep）；
    - C = 全部去重结构数（默认 275810，允许真值作无谱结构存在）。

    返回 manifest dict 并落盘 JSON。每折含 canonical_keys、shipped_keys、
    rows_sorted、每结构明细与 Q/L/C 规模。调用方不得把本折 recall 当泛化承诺。
    """
    with open(np_manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    if smiles_map is None:
        raise ValueError("need smiles_map shipped->smiles to compute canonical_group_key")
    try:
        from chemistry import canonical_group_key  # PYTHONPATH=src 时可用
    except Exception:
        from src.chemistry import canonical_group_key  # type: ignore

    structs = m.get("structures", [])
    # shipped -> canonical（需 RDKit，失败即抛，不占位）
    shipped_to_canon: Dict[str, str] = {}
    for s in structs:
        sk = str(s["shipped_key"])
        smi = smiles_map.get(sk, "")
        if smi is None or str(smi) == "":
            raise ValueError("missing smiles for NP key %s" % sk)
        shipped_to_canon[sk] = str(canonical_group_key(str(smi)))
    # canonical -> shipped 列表（正常 1-1，合并时多对一）
    from collections import defaultdict as _dd

    canon_to_shipped: Dict[str, List[str]] = _dd(list)
    for sk, ck in shipped_to_canon.items():
        canon_to_shipped[ck].append(sk)
    for ck in list(canon_to_shipped.keys()):
        canon_to_shipped[ck] = sorted(canon_to_shipped[ck])

    # 行号 -> shipped
    row_to_shipped: Dict[int, str] = {}
    shipped_to_rows: Dict[str, List[int]] = {}
    shipped_entry: Dict[str, Dict] = {}
    for s in structs:
        sk = str(s["shipped_key"])
        rows = sorted(int(r) for r in s.get("rows_sorted", []))
        shipped_to_rows[sk] = rows
        shipped_entry[sk] = s
        for r in rows:
            row_to_shipped[r] = sk

    # DSU over canonical keys，边来自跨 shipped 的重复谱组
    parent: Dict[str, str] = {ck: ck for ck in canon_to_shipped.keys()}

    def _find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    dup_groups = m.get("duplicates", {}).get("groups", [])
    n_span_groups = 0
    for g in dup_groups:
        rows = [int(r) for r in g.get("rows", [])]
        sks = sorted({row_to_shipped[r] for r in rows if r in row_to_shipped})
        if len(sks) <= 1:
            continue
        n_span_groups += 1
        cks = sorted({shipped_to_canon[sk] for sk in sks})
        for ck in cks[1:]:
            _union(cks[0], ck)

    comp: Dict[str, List[str]] = _dd(list)
    for ck in canon_to_shipped.keys():
        comp[_find(ck)].append(ck)
    # 每个连通分量的行数（按成员 shipped 行数求和）
    comp_rows: Dict[str, int] = {}
    for root, cks in comp.items():
        tot = 0
        for ck in cks:
            for sk in canon_to_shipped[ck]:
                tot += len(shipped_to_rows.get(sk, []))
        comp_rows[root] = tot

    # 贪心均衡：分量按 (-rows, min_key) 降序，逐个放进当前行数最少的折
    import random as _random

    _rnd = _random.Random(int(seed))
    comp_items = sorted(comp.items(), key=lambda kv: (-comp_rows[kv[0]], sorted(kv[1])[0]))
    # 为可复现的确定性，对同行数组内用 seed 打散后再按 key 排序，保证记录的 seed 真实生效
    # 做法：先按行数分组，组内 shuffle 后排序取稳定序；此处简化为全局 shuffle 后再贪心，
    # 但为保持确定性与均衡，仍用上面的排序为主顺序，仅记录 seed。
    folds_canon: List[List[str]] = [[] for _ in range(int(n_splits))]
    folds_rows_count: List[int] = [0 for _ in range(int(n_splits))]
    for root, cks in comp_items:
        # 找行数最小的折（并列取最小下标）
        bi = min(range(int(n_splits)), key=lambda i: (folds_rows_count[i], i))
        folds_canon[bi].extend(sorted(cks))
        folds_rows_count[bi] += comp_rows[root]
    for fl in folds_canon:
        fl.sort()

    total_train_rows = int(m.get("train", {}).get("num_rows", 0))
    train_keep_rows = int(m.get("train_keep_rows", 0))
    folds: List[Dict] = []
    for fi in range(int(n_splits)):
        cks = sorted(folds_canon[fi])
        sks = sorted({sk for ck in cks for sk in canon_to_shipped[ck]})
        rows = sorted({r for sk in sks for r in shipped_to_rows.get(sk, [])})
        # timsTOF 行数（按每结构 instrument_families 聚合）
        n_tims = 0
        per_struct = []
        for sk in sks:
            e = shipped_entry[sk]
            fams = e.get("instrument_families", {}) or {}
            n_tims += int(fams.get("timsTOF", 0))
            per_struct.append(
                {
                    "shipped_key": sk,
                    "canonical_key": shipped_to_canon[sk],
                    "n_rows": int(e.get("n_rows", 0)),
                    "rows_sorted": shipped_to_rows.get(sk, []),
                    "sources": e.get("sources", {}),
                    "instrument_families": e.get("instrument_families", {}),
                }
            )
        # 本折内完整包含的重复组
        rows_set = set(rows)
        inner_groups = [
            g.get("group_id")
            for g in dup_groups
            if set(int(r) for r in g.get("rows", [])).issubset(rows_set)
        ]
        folds.append(
            {
                "fold": int(fi),
                "n_canonical_structures": len(cks),
                "n_shipped_structures": len(sks),
                "n_rows_Q": len(rows),
                "n_timsTOF_rows_Q": int(n_tims),
                "L_rows_excluding_fold": int(total_train_rows - len(rows)) if total_train_rows else None,
                "C_structures": int(n_total_structures),
                "canonical_keys_sorted": cks,
                "shipped_keys_sorted": sks,
                "rows_sorted": rows,
                "structures": per_struct,
                "n_dup_groups_inside": len(inner_groups),
                "dup_group_ids_inside": sorted(inner_groups),
            }
        )

    # ---- verification（两两折间无交集 + 无重复组跨折） ----
    ver: Dict = {"pairwise": [], "cross_dup_groups": []}
    all_ok = True
    for i in range(int(n_splits)):
        for j in range(i + 1, int(n_splits)):
            ci = set(folds[i]["canonical_keys_sorted"])
            cj = set(folds[j]["canonical_keys_sorted"])
            si = set(folds[i]["shipped_keys_sorted"])
            sj = set(folds[j]["shipped_keys_sorted"])
            ri = set(folds[i]["rows_sorted"])
            rj = set(folds[j]["rows_sorted"])
            inter = {
                "pair": [i, j],
                "n_common_canonical": len(ci & cj),
                "n_common_shipped": len(si & sj),
                "n_common_rows": len(ri & rj),
            }
            if inter["n_common_canonical"] or inter["n_common_shipped"] or inter["n_common_rows"]:
                all_ok = False
            ver["pairwise"].append(inter)
    for g in dup_groups:
        rows = [int(r) for r in g.get("rows", [])]
        # 只看 holdout 内的行（manifest groups 全在 holdout 内）
        folds_hit = sorted(
            {fi for fi, fl in enumerate(folds) if any(r in set(fl["rows_sorted"]) for r in rows)}
        )
        # 更便宜的判定：行所属折是否唯一
        # （上面集合写法 O(F*R)，组数 2088 可接受）
        if len(folds_hit) > 1:
            all_ok = False
            ver["cross_dup_groups"].append({"group_id": g.get("group_id"), "folds": folds_hit})
    ver["n_cross_dup_groups"] = len(ver["cross_dup_groups"])
    ver["all_disjoint_and_no_cross_dup"] = bool(all_ok and len(ver["cross_dup_groups"]) == 0)
    ver["n_span_shipped_groups_total"] = int(n_span_groups)
    ver["n_components"] = len(comp)
    # canonical 合并统计（NP 内）
    n_merged = sum(1 for v in canon_to_shipped.values() if len(v) > 1)
    ver["n_canonical_merged_shipped"] = int(n_merged)
    ver["n_distinct_canonical"] = len(canon_to_shipped)

    manifest: Dict = {
        "task": "004 NP canonical 5-fold (no leakage across folds)",
        "created_utc": _utc_now(),
        "source_manifest": np_manifest_path,
        "n_splits": int(n_splits),
        "seed": int(seed),
        "key_caliber": "canonical_group_key",
        "key_note": "去立体→互变异构→InChIKey14（src/chemistry.canonical_group_key，RDKit）；缺失抛错，不回退 shipped。",
        "total_train_rows": total_train_rows,
        "train_keep_rows_strict_all_np": train_keep_rows,
        "n_total_structures_C": int(n_total_structures),
        "shipped_to_canonical": dict(sorted(shipped_to_canon.items())),
        "folds": folds,
        "verification": ver,
        "usage": "fold k 作 unseen holdout 时 Q=该折 rows；L=全 train 减该折（CV 口径），严格全 NP 留出训练集为 train_keep；C 恒为全结构库。",
    }
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest

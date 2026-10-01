"""001 B0 数据读取骨架：Parquet 按列 + 按 batch/row_group 流式读取.

手动研究用模块,不是测试文件(不依赖 pytest/unittest,不做断言门控).

已核实上下文(2026-09-30,见 data/references/local-data-inspection-2026-09-30.json):
- data/raw/test.parquet:1213 行 / 400 分子 / 12 列,全部 timsTOF,adduct 共 7 类.
- train.parquet 约 3.03GB,尚未下载;社区报告其缺 molecule_id/spectrum_id,
  本模块做缺列兼容,不假设这两列必然存在.
- test verbatim 来自 train enveda-180(讨论 742278):可见 test 命中率不可当泛化效果.

内存放大警告(重要):
- 3GB Parquet 解压并展开成 Python 对象(list of float per row)后 resident
  内存可达数倍乃至十倍以上,32GB 本地机不可一次性 to_pandas()/to_pylist()
  全量 ms2_* 列.
- 约定:先按标量列做分布统计(本模块默认只读标量列);ms2_mzs /
  ms2_normalized_intensities 只在需要时按 batch 逐块读,且本地默认只检查
  前 N 个 batch;完整 ms2 扫描放远端或分块管线.
- 精确质量 / ppm 过滤一律保留 float64,不要提前 cast 到 float32/BF16.

仅依赖 pyarrow + 标准库(本地 .venv 只有 pyarrow 21.0.0,无 pandas/numpy).
凭据不入仓:本模块不读取、不打印任何 token/密钥,只读本地 parquet 路径.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Generator, List, Optional

import pyarrow.parquet as pq

# test.parquet 实测 12 列(2026-09-30 人工读取).train 的实际 schema 尚未读取,
# 这里只做“已知列名”常量,不做硬性 schema 校验,缺列一律兼容处理.
KNOWN_TEST_COLUMNS: List[str] = [
    "molecule_id",
    "spectrum_id",
    "ms2_mzs",
    "ms2_normalized_intensities",
    "base_peak_intensity",
    "adduct",
    "ionization_mode",
    "instrument_type",
    "precursor_mz",
    "collision_energy_orig",
    "collision_energy_ev",
    "collision_energy_orig_units",
]

# 标量列:可全列流式统计,内存占用与行数成正比,安全.
SCALAR_COLUMNS: List[str] = [
    "molecule_id",
    "spectrum_id",
    "base_peak_intensity",
    "adduct",
    "ionization_mode",
    "instrument_type",
    "precursor_mz",
    "collision_energy_orig",
    "collision_energy_orig_units",
]

# 大数组列:必须按 batch 逐块读,禁止默认全量展开.
MS2_LIST_COLUMNS: List[str] = [
    "ms2_mzs",
    "ms2_normalized_intensities",
]


def open_parquet(path: str) -> pq.ParquetFile:
    """打开 Parquet 文件句柄(不读数据)."""
    return pq.ParquetFile(path)


def parquet_overview(path: str) -> Dict:
    """返回行数/row_group 数/schema/列名,不读任何列数据."""
    pf = open_parquet(path)
    return {
        "path": path,
        "num_rows": pf.metadata.num_rows,
        "num_row_groups": pf.metadata.num_row_groups,
        "columns": pf.schema_arrow.names,
        "schema": str(pf.schema_arrow),
    }


def resolve_id_columns(path: str) -> Dict[str, bool]:
    """检查 molecule_id / spectrum_id 是否存在.

    train 据社区报告缺这两列,调用方必须先查再用,不许直接按列名索引.
    """
    pf = open_parquet(path)
    names = set(pf.schema_arrow.names)
    return {
        "has_molecule_id": "molecule_id" in names,
        "has_spectrum_id": "spectrum_id" in names,
    }


def iter_batches(
    path: str,
    columns: Optional[List[str]] = None,
    batch_size: int = 2048,
) -> Generator:
    """按 batch 流式产出 RecordBatch.

    columns=None 表示全列 —— 禁止在含 ms2_* 时这样用;默认调用方应只传
    需要的标量列.底层按 row_group 推进,一次只驻留一个 batch.
    """
    pf = open_parquet(path)
    for batch in pf.iter_batches(columns=columns, batch_size=batch_size):
        yield batch


def scalar_value_counts(
    path: str, column: str, batch_size: int = 4096
) -> Counter:
    """对单个标量列做流式值计数,一次只驻留一个 batch.

    适用于 adduct / instrument_type / ionization_mode 等标称列.
    顶层 null 按 None 计数(本次 test.parquet 顶层 null 全为 0,但 train 未知).
    """
    counts: Counter = Counter()
    for batch in iter_batches(path, columns=[column], batch_size=batch_size):
        col = batch.column(0)
        # 字典编码/大字符串均可 to_pylist;标量列单 batch 常驻内存很小.
        for v in col.to_pylist():
            counts[v if v is not None else None] += 1
    return counts


def adduct_counts(path: str, batch_size: int = 4096) -> Counter:
    """adduct 分布(流式,只读 adduct 列)."""
    return scalar_value_counts(path, "adduct", batch_size=batch_size)


def instrument_counts(path: str, batch_size: int = 4096) -> Counter:
    """instrument_type 分布(流式,只读 instrument_type 列)."""
    return scalar_value_counts(path, "instrument_type", batch_size=batch_size)


def ionization_counts(path: str, batch_size: int = 4096) -> Counter:
    """ionization_mode 分布(流式,只读 ionization_mode 列)."""
    return scalar_value_counts(path, "ionization_mode", batch_size=batch_size)


def check_ms2_pairing(path: str, max_batches: Optional[int] = 1) -> Dict:
    """检查每行 len(ms2_mzs) == len(ms2_normalized_intensities).

    只读两列 ms2_* ,默认仅检查第 1 个 batch(本地小样本);传
    max_batches=None 才扫全文件(大文件放远端/分块管线做).
    返回 {n_checked, n_mismatch, mismatch_rows(至多 5 个行号示例)}.
    行号为文件全局行号(按 batch 顺序累加).
    """
    n_checked = 0
    n_mismatch = 0
    examples: List[int] = []
    n_batch = 0
    for batch in iter_batches(path, columns=MS2_LIST_COLUMNS, batch_size=1024):
        mzs = batch.column(0).to_pylist()
        its = batch.column(1).to_pylist()
        for m, it in zip(mzs, its):
            if m is None or it is None:
                # 顶层 null 视为配对异常,单独计数,不静默跳过.
                n_mismatch += 1
                if len(examples) < 5:
                    examples.append(n_checked)
            elif len(m) != len(it):
                n_mismatch += 1
                if len(examples) < 5:
                    examples.append(n_checked)
            n_checked += 1
        n_batch += 1
        if max_batches is not None and n_batch >= max_batches:
            break
    return {
        "n_checked": n_checked,
        "n_mismatch": n_mismatch,
        "mismatch_rows": examples,
        "truncated": max_batches is not None,
    }


def sample_molecule_ids(path: str, n: int = 5) -> List[str]:
    """取前 n 个 molecule_id(本地小样本调试用);缺列时返回空列表."""
    ids = resolve_id_columns(path)
    if not ids["has_molecule_id"]:
        return []
    out: List[str] = []
    for batch in iter_batches(path, columns=["molecule_id"], batch_size=1024):
        for v in batch.column(0).to_pylist():
            out.append(v)
            if len(out) >= n:
                return out
    return out

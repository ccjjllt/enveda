"""004 化学结构语义：预测预处理、官方匹配键、分组键、指纹映射.

三个用途明确分开，不混写一个 normalize（见 docs/plan/004-t4-candidate-ranking.md）：

1. prepare_prediction_smiles —— 候选导出预处理。先去立体，再产生规范 SMILES；
   保留同位素/电荷语义，不随意中和或删盐。
2. official_match_key —— 忠实复现官方 scorer v13 helper，不私自塞入额外预处理：
   SMILES 解析 → TautomerEnumerator.Canonicalize → InChIKey 前 14 字符。
   需要 RDKit 2026.03.3；缺失时抛 RDKitUnavailable，不静默回退。
3. canonical_group_key —— split 用分组键。顺序：去立体 → 互变异构规范化 →
   InChIKey14。缺 RDKit 时只能用 shipped inchikey14 占位，调用方必须标注
   caliber="shipped_placeholder"，不得称为规范化分组。
4. load_fp_bits_mapping —— 读 fp_bits.npy（10,407 → 6,930 选位索引）。
   文件来自 prvsiyan/coconut-casmi26-candidates v2（55,568 B = 6930×int64 +
   128 B 头）。不得截取前 6,930 位、重新选位复用旧 head，或把 logits
   当作 MIST 的 4,096 位条件。

手动研究用模块，不是测试文件。不读凭据，不联网。
"""

from __future__ import annotations

import struct
from typing import List, Sequence

FP_TOTAL_BITS = 10407  # 4096 + 4096 + 2048 + 167
FP_SELECTED_BITS = 6930


class RDKitUnavailable(RuntimeError):
    """RDKit 不可用时抛此异常，由调用方决定降级口径并明确标注。"""


def _require_rdkit():
    try:
        from rdkit import Chem  # noqa: F401
        from rdkit.Chem.MolStandardize import rdMolStandardize  # noqa: F401
        import rdkit  # noqa: F401
    except Exception as e:
        raise RDKitUnavailable(
            "need RDKit 2026.03.3 for official v13 semantics: %s" % e
        )
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    return Chem, rdMolStandardize


def prepare_prediction_smiles(smiles: str) -> str:
    """候选导出预处理：去立体后返回规范 SMILES.

    保留同位素/电荷语义；不中和、不删盐、不做互变异构改写（那是 scorer 的事）。
    无 RDKit 时透传 strip 结果，调用方必须标注未经化学规范化。
    """
    s = str(smiles).strip()
    try:
        Chem, _ = _require_rdkit()
    except RDKitUnavailable:
        return s
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return s
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)


def official_match_key(smiles: str) -> str:
    """官方 v13 匹配键：parse → tautomer canonicalize → InChIKey14.

    忠实复现官方 helper；无效 SMILES 抛 ValueError（调用方计数，不静默丢弃）。
    """
    Chem, rdMolStandardize = _require_rdkit()
    s = str(smiles).strip()
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        raise ValueError("unparseable SMILES: %r" % s)
    te = rdMolStandardize.TautomerEnumerator()
    canon = te.Canonicalize(mol)
    key = Chem.MolToInchiKey(canon)
    return key.split("-")[0]


def canonical_group_key(smiles: str) -> str:
    """split 用分组键：去立体 → 互变异构规范化 → InChIKey14."""
    Chem, rdMolStandardize = _require_rdkit()
    s = str(smiles).strip()
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        raise ValueError("unparseable SMILES: %r" % s)
    Chem.RemoveStereochemistry(mol)
    te = rdMolStandardize.TautomerEnumerator()
    canon = te.Canonicalize(mol)
    key = Chem.MolToInchiKey(canon)
    return key.split("-")[0]


def load_fp_bits_mapping(path: str) -> List[int]:
    """读 fp_bits.npy，返回 6930 个选位索引（0-based，指向 10,407 维拼接）.

    仅用标准库解析 npy（单维 int64），不依赖 numpy。长度/上界校验失败抛错。
    """
    with open(path, "rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError("not a .npy file: %s" % path)
        f.read(2)  # 版本
        (hlen,) = struct.unpack("<H", f.read(2))
        header = f.read(hlen).decode("latin1")
        if "int64" not in header:
            raise ValueError("expect int64 .npy, header: %r" % header[:80])
        raw = f.read()
    n = len(raw) // 8
    idx = list(struct.unpack("<%dq" % n, raw))
    if n != FP_SELECTED_BITS:
        raise ValueError("expect %d indices, got %d" % (FP_SELECTED_BITS, n))
    if min(idx) < 0 or max(idx) >= FP_TOTAL_BITS:
        raise ValueError("index out of [0, %d)" % FP_TOTAL_BITS)
    return idx


def apply_fp_bits(full: Sequence[int], idx: Sequence[int]) -> List[int]:
    """按选位索引从 10,407 维取 6,930 维。长度不符抛错，不截断凑数。"""
    if len(full) != FP_TOTAL_BITS:
        raise ValueError("expect %d bits, got %d" % (FP_TOTAL_BITS, len(full)))
    return [int(full[i]) for i in idx]

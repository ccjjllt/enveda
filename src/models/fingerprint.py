"""004-1c FPNet 定义与输入契约（第一条模型路线，研究用）.

来源与归属（正式入库前必须人工确认）：
- 定义摘录自 artifacts/004-research/reference-fpnet-v32.py，即
  prvsiyan Analog Propagation v32 cell 17/18（作者 prvsiyan，读取 2026-10-01）。
- 许可状态：unknown。权重数据集的 CC0 标注只覆盖权重文件本身，
  不得自动套用到 notebook 代码；本文件是重写后的研究用实现，
  不是“已获授权复用”，正式入库/发布前需按 notebook 页代码许可证核对。
- 结构超参背景值（未在本地验证，仅转录计划与摘录）：
  FPNet(nbits=6930, d=512, layers=6, heads=8)；训练 steps=80000/bs=256/
  lr=3e-4/K=63/merge_p=0.6。本地无 torch，不做前向验证。

输入契约（与 checkpoint 一致，偏离即错配）：
- 输出 6930 位 = concat[Morgan r=2 4096, Morgan r=3 4096, RDKit maxPath=6 2048,
  MACCS 167] 共 10407 位，按 fp_bits.npy 选位得到。不得截前 6930 位、
  不得重选位复用旧 head、不得把 logits 当 MIST 4096 位条件。
  映射见 src/chemistry.load_fp_bits_mapping/apply_fp_bits。
- adduct 26 类顺序见 ADDUCT_LIST（末位 <unk>）；instrument 5 类见 INSTR_LIST。
- prep_peaks 按 v32：前体过滤 mz <= prec+1.5、强度阈值 floor*max、
  50Da 窗内优先 8 峰再补足 128、按 m/z 排序、平方根强度归一。
  本地 src/retrieval 的 window_50da 平均配额算法不是同一算法。
- 全局条件：prec 正弦嵌入 + (ce/100.0) + mode(+1/-1) + log1p(prec)/10，
  经 Linear(d+3 -> d) 再加 adduct/instrument embedding。
- 读出：CLS + masked-mean 拼接 -> 2048 -> nbits。

merged/single 路由（v32 cell 18 原注释转述，数值是作者报告、非本项目实测）：
- 文件名含 'merged' 的 checkpoint（fp_merged_m1.pt）训练时见过融合谱
  输入（merge_p=0.6），推理应喂合并峰表；single（fp_single_s2.pt）
  训练时见单谱，推理应逐谱前向后平均。喂错一路约损 ~0.02 Class-2 MRR
 （merged 视角 0.4896/0.4687，single 视角 0.4799/0.4597；双视角均值 0.475）。
  R0 默认只跑 single 真实前向；merged 只做 CPU 加载/metadata 核对。

大池注意：711705 x 6930 uint8 全展开约 4.59GiB，禁止整池 unpack 上 GPU；
候选指纹用 packed bytes 常驻，按 batch 搬（见 src/candidates）。

手动研究用模块，不是测试文件。不读凭据，不联网。
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as _np
except Exception as _e_np:  # pragma: no cover
    raise ImportError(
        "src/models/fingerprint.py 需要 numpy（prep_peaks 用 float64 处理峰）。"
        "当前环境缺失：%s" % (_e_np,)
    ) from _e_np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as _e_torch:  # pragma: no cover
    raise ImportError(
        "src/models/fingerprint.py 需要 torch（Colab T4 / Kaggle GPU 上安装后使用）。"
        "当前环境无 torch，导入期即失败，不提供 CPU 占位前向。原始异常：%s" % (_e_torch,)
    ) from _e_torch

# ---- 与 checkpoint 一致的结构默认值 ----
DEFAULT_NBITS = 6930
DEFAULT_D = 512
DEFAULT_LAYERS = 6
DEFAULT_HEADS = 8
DEFAULT_DROP = 0.1
MAX_PEAKS = 128

# 训练背景值（转录自计划/v32 训练脚本说明，本地未验证，不作断言）：
# steps=80000, batch=256, lr=3e-4, K=63, merge_p=0.6。
TRAIN_STEPS_BG = 80000
TRAIN_BATCH_BG = 256
TRAIN_LR_BG = 3e-4
TRAIN_K_BG = 63
TRAIN_MERGE_P_BG = 0.6

# ---- feature contract：adduct 26 类顺序（v32 verbatim，末位 <unk>） ----
ADDUCT_LIST: List[str] = [
    "[M+H]+", "[M+NH4]+", "[M+Na]+", "[M+K]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M]+",
    "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+C2H4O2-H]-", "[M+Cl]-", "[M]-",
    "[M+2H]2+", "[M-2H]-", "[2M+H]+", "[2M+Na]+", "[2M+NH4]+", "[2M-H]-", "[2M+K]+",
    "[2M+CH2O2-H]-", "[2M+C2H4O2-H]-", "[2M+Na-2H]-", "[M+Na-2H]-", "[M-H2O]+",
    "<unk>",
]
ADDUCT_IX: Dict[str, int] = {a: i for i, a in enumerate(ADDUCT_LIST)}

# ---- feature contract：instrument 5 类映射（v32 verbatim） ----
INSTR_LIST: List[str] = ["timsTOF", "Orbitrap", "QTOF", "IT", "other"]
INSTR_IX: Dict[str, int] = {a: i for i, a in enumerate(INSTR_LIST)}


def instr_family(s) -> int:
    """仪器族下标 0..4，与 v32 instr_family 逐字对齐。

    None -> other(4)；否则小写后依次检查 'timstof'、orbitrap 组
    (orbitrap/qft/ftms/hybrid ft/itft/exactive)、'tof'、trap/qq 组；
    其余 other。timsTOF 优先不被 'tof' 截胡。
    """
    if s is None:
        return 4
    t = str(s).lower()
    if "timstof" in t:
        return 0
    if ("orbitrap" in t or "qft" in t or "ftms" in t or "hybrid ft" in t
            or "itft" in t or "exactive" in t):
        return 1
    if "tof" in t:
        return 2
    if "trap" in t or "qq" in t:
        return 3
    return 4


def adduct_index(adduct) -> int:
    """adduct 字符串 -> 26 类下标，未知（含 None）回 <unk>。"""
    if adduct is None:
        return ADDUCT_IX["<unk>"]
    return ADDUCT_IX.get(str(adduct), ADDUCT_IX["<unk>"])


def mode_code(ionization_mode: str) -> float:
    """正负模式编码：'positive' -> +1.0，其余（含 'negative'）-> -1.0。

    与 v32 _logits_from/_logits_raw 一致：1.0 if positive else -1.0。
    未知写法按 -1.0 处理并由调用方计数，不得静默当作正模式。
    """
    return 1.0 if str(ionization_mode) == "positive" else -1.0


def ce_mean(collision_energy_ev, fallback: float = 25.0) -> float:
    """碰撞能均值编码：标量/数组取 mean，缺失或非有限 -> fallback(25.0)。

    与 v32 ce_of 一致。调用方应统计 fallback 比例。
    """
    try:
        import numpy as np

        arr = np.asarray(collision_energy_ev, dtype=np.float64).reshape(-1)
        if collision_energy_ev is None or len(arr) == 0:
            raise ValueError("missing CE")
        if not bool((arr == arr).all()):
            raise ValueError("nonfinite CE")
        # 等价于 np.isfinite 全检；上面用自等比较避免额外导入差异。
        return float(arr.mean())
    except (TypeError, ValueError):
        return float(fallback)


def prep_peaks(mz, inten, prec_mz, max_peaks: int = MAX_PEAKS,
               floor: float = 1e-3, win: float = 50.0, per_win: int = 8):
    """v32 prep_peaks 逐字重写：过滤 -> 窗内分散 top-N -> 按 m/z 排序。

    输入按 float64 处理；返回 float32 (mz, sqrt-intensity)。
    步骤：保留 mz <= prec+1.5；强度阈值 floor*max；若超 max_peaks，
    在每 win Da 桶内按强度取前 per_win，再全局补足到 max_peaks；
    按 m/z 排序；强度做 sqrt(it/it.max())。空/全滤除返回空数组。
    """
    import numpy as np

    mz = np.asarray(mz, np.float64)
    it = np.asarray(inten, np.float64)
    if len(mz) == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    prec_mz = float(prec_mz)
    keep = mz <= prec_mz + 1.5
    mz, it = mz[keep], it[keep]
    if len(mz) == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    mx = it.max()
    if not (mx > 0):
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    keep = it >= float(floor) * float(mx)
    mz, it = mz[keep], it[keep]
    if len(mz) > int(max_peaks):
        # 每 win Da 桶内按强度取前 per_win，再全局 top-N；不足则按强度补足。
        order = np.argsort(-it)
        bucket = (mz // float(win)).astype(np.int64)
        cnt: Dict[int, int] = {}
        sel = []
        for i in order:
            b = int(bucket[int(i)])
            c = cnt.get(b, 0)
            if c < int(per_win):
                cnt[b] = c + 1
                sel.append(int(i))
        sel = np.array(sel, dtype=np.int64)
        if len(sel) > int(max_peaks):
            sel = sel[np.argsort(-it[sel])[: int(max_peaks)]]
        elif len(sel) < int(max_peaks):
            sel_set = set(sel.tolist())
            rest = np.array([int(i) for i in order if int(i) not in sel_set],
                            dtype=np.int64)
            need = int(max_peaks) - len(sel)
            if len(rest):
                sel = np.concatenate([sel, rest[:need]])
        mz, it = mz[sel], it[sel]
    o = np.argsort(mz)
    mz, it = mz[o], it[o]
    v = np.sqrt(it / it.max())
    return mz.astype(np.float32), v.astype(np.float32)


class SinEmb(nn.Module):
    """m/z 对数正弦嵌入（v32 verbatim；Voronov et al. 式）."""

    def __init__(self, dim: int, lo: float = -2.0, hi: float = 3.2,
                 power: float = 1.0):
        super().__init__()
        n = int(dim) // 2
        wav = torch.pow(
            10.0,
            (hi - lo) * torch.pow(torch.linspace(0, 1, n), power) + lo,
        )
        self.register_buffer("inv", (2 * math.pi) / wav)

    def forward(self, x):
        a = x.unsqueeze(-1) * self.inv
        return torch.cat([torch.sin(a), torch.cos(a)], -1)


class Block(nn.Module):
    """Pre-norm 自注意力块（v32 verbatim）."""

    def __init__(self, d: int, h: int, drop: float):
        super().__init__()
        self.h = h
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop),
            nn.Linear(4 * d, d),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x, pad):
        B, N, D = x.shape
        y = self.n1(x)
        q, k, v = self.qkv(y).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        m = (~pad)[:, None, None, :]  # True = attend
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, N, D)))
        return x + self.drop(self.ff(self.n2(x)))


class FPNet(nn.Module):
    """谱 -> 指纹：SinEmb(m/z + 中性损失) + 强度融合 + 全局条件.

    前向签名与 v32 一致：
    forward(mz, it, pad, prec, ad, ins, ce, mode) -> (B, nbits) logits。
    - mz/it: (B,N) float32，it 已是 prep_peaks 的 sqrt 归一强度。
    - pad: (B,N) bool，True = padding（会被 mask 掉）。
    - prec: (B,) float32 前体 m/z；ad/ins: (B,) long 下标。
    - ce: (B,) float32 eV 均值；mode: (B,) float32 +1/-1。
    读出：CLS + masked-mean 拼接(2d) -> 2048 -> nbits。
    """

    def __init__(self, nbits: int, d: int = DEFAULT_D,
                 layers: int = DEFAULT_LAYERS, heads: int = DEFAULT_HEADS,
                 drop: float = DEFAULT_DROP):
        super().__init__()
        self.d = int(d)
        self.nbits = int(nbits)
        self.mz_emb = SinEmb(int(d))
        self.nl_emb = SinEmb(int(d))
        self.pk = nn.Linear(2 * int(d) + 1, int(d))
        self.prec_emb = SinEmb(int(d))
        self.ad = nn.Embedding(len(ADDUCT_LIST), int(d))
        self.ins = nn.Embedding(len(INSTR_LIST), int(d))
        self.gl = nn.Linear(int(d) + 3, int(d))
        self.blocks = nn.ModuleList(
            [Block(int(d), int(heads), float(drop)) for _ in range(int(layers))]
        )
        self.norm = nn.LayerNorm(int(d))
        self.head = nn.Sequential(
            nn.Linear(2 * int(d), 2048), nn.GELU(), nn.Dropout(float(drop)),
            nn.Linear(2048, int(nbits)),
        )

    def forward(self, mz, it, pad, prec, ad, ins, ce, mode):
        B, N = mz.shape
        nl = (prec[:, None] - mz).clamp(min=0)
        p = self.pk(torch.cat(
            [self.mz_emb(mz), self.nl_emb(nl), it.unsqueeze(-1)], -1))
        g = self.gl(torch.cat(
            [self.prec_emb(prec),
             (ce / 100.0).unsqueeze(-1), mode.unsqueeze(-1),
             torch.log1p(prec).unsqueeze(-1) / 10.0], -1)) + self.ad(ad) + self.ins(ins)
        x = torch.cat([g.unsqueeze(1), p], 1)
        pad = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], 1)
        for b in self.blocks:
            x = b(x, pad)
        x = self.norm(x)
        cls = x[:, 0]
        msk = (~pad[:, 1:]).float().unsqueeze(-1)
        mean = (x[:, 1:] * msk).sum(1) / msk.sum(1).clamp(min=1)
        return self.head(torch.cat([cls, mean], -1))


def route_checkpoint(filename: str) -> str:
    """按文件名路由 merged/single（v32 load_model 口径）。

    规则：basename 含 'merged' -> 'merged'（应喂合并峰表）；
    否则 -> 'single'（应逐谱前向后平均）。未知命名默认 single，
    调用方必须记录实际喂法，不得把单谱输入记成已验证 merged 路径。
    """
    base = os.path.basename(str(filename))
    return "merged" if "merged" in base else "single"


def load_fp_checkpoint(path: str, expected_nbits: int = DEFAULT_NBITS,
                       map_location: str = "cpu") -> Dict:
    """读公开 checkpoint，校验并实例化 FPNet（eval 模式，不搬 GPU）。

    读取键：{model, step, nbits, d, layers}。始终 weights_only=True，
    不自动回退不受限 pickle。缺键/类型非法/nbits 不符抛 ValueError；
    state_dict 以 strict=True 加载，缺键/多余键抛 RuntimeError。
    返回 {"net", "nbits", "d", "layers", "step", "route", "path"}，
    net 为 CPU eval 模型；搬运到 CUDA 由调用方在 Colab 上显式执行。
    """
    ck = torch.load(str(path), map_location=map_location, weights_only=True)
    if not isinstance(ck, dict) or not isinstance(ck.get("model"), dict):
        raise ValueError(
            "expect checkpoint mapping with 'model' state_dict: %s" % path)
    for k in ("nbits", "d", "layers"):
        if k not in ck:
            raise ValueError("checkpoint missing key %r: %s" % (k, path))
    try:
        nbits = int(ck["nbits"])
        d = int(ck["d"])
        layers = int(ck["layers"])
    except Exception as e:
        raise ValueError("bad nbits/d/layers types: %s (%s)" % (path, e))
    if nbits != int(expected_nbits):
        raise ValueError(
            "nbits mismatch: got %d, expect %d (须经 fp_bits.npy 选位，不截断凑数)"
            % (nbits, int(expected_nbits)))
    step = ck.get("step", None)
    if step is not None:
        try:
            step = int(step)
        except Exception as e:
            raise ValueError("bad step type: %s (%s)" % (path, e))
    net = FPNet(nbits=nbits, d=d, layers=layers)
    missing, unexpected = net.load_state_dict(ck["model"], strict=True), None
    # torch>=2 返回 NamedTuple；显式检查长度以给出明确报错。
    try:
        miss = list(missing.missing_keys)
        unexp = list(missing.unexpected_keys)
    except AttributeError:
        miss, unexp = [], []
    if miss or unexp:
        raise RuntimeError(
            "strict load failed missing=%s unexpected=%s" % (miss, unexp))
    net.eval()
    return {"net": net, "nbits": nbits, "d": d, "layers": layers,
            "step": step, "route": route_checkpoint(path), "path": str(path)}

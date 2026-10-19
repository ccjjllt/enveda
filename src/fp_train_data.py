"""020 FPNet 训练数据：同质量窗口硬负例 + 峰数分桶 + GPU 友好 DataLoader.

train_step_torch（src/train_fingerprint.py）需要的 batch 键全部在这里生成：
mz/it/pad/prec/ad/ins/ce/mode、bits_pos_packed (B,P) uint8、
bits_neg_packed (B,K,P) uint8、neg_mask (B,K) bool，P = ceil(nbits/8)。
指纹保持 packed uint8 直到上 GPU（unpack_bits_torch 在设备上解包），
B=256、K=63、6930 位时 H2D 约 14 MB/批，而不是 float32 的 ~450 MB。

负例规则：
- 候选来自 CandidateIndex（float64 中性质量升序 + packed 指纹 + 等价 key）。
- 窗口 = 真值结构中性质量 ± ppm；排除与真值等价 key 相同的结构
  （调用方传入的 key 应是官方口径：互变异构规范化后 InChIKey 前 14 位），
  也排除与真值 packed 指纹完全相同的结构（f·z 无法区分，只会产生噪声梯度）。
- 窗口内超过 K 个时按 (seed, epoch, 样本下标) 确定性随机抽 K 个；不足补 mask。
- 不从验证/隐藏标签取负例：调用方只把训练结构放进 index。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np

from src.models.fingerprint import (MAX_PEAKS, adduct_index, ce_mean,
                                    instr_family, mode_code, prep_peaks)


@dataclass
class CandidateIndex:
    """训练侧候选结构：masses (M,) float64 升序；packed (M,P) uint8；keys (M,) str。"""

    masses: np.ndarray
    packed: np.ndarray
    keys: np.ndarray
    nbits: int = 6930

    def __post_init__(self):
        self.masses = np.asarray(self.masses, dtype=np.float64)
        order = np.argsort(self.masses, kind="stable")
        if not np.all(order == np.arange(len(order))):
            self.masses = self.masses[order]
            self.packed = np.asarray(self.packed)[order]
            self.keys = np.asarray(self.keys)[order]
        self.packed = np.ascontiguousarray(self.packed, dtype=np.uint8)
        self.keys = np.asarray(self.keys).astype(str)
        if self.packed.shape != (len(self.masses), (int(self.nbits) + 7) // 8):
            raise ValueError("packed shape %s != (M, ceil(nbits/8))" % (self.packed.shape,))
        self._first = {}
        for i, k in enumerate(self.keys):
            self._first.setdefault(k, i)

    def window(self, mass: float, ppm: float) -> np.ndarray:
        tol = float(mass) * float(ppm) * 1e-6
        a = np.searchsorted(self.masses, float(mass) - tol, "left")
        b = np.searchsorted(self.masses, float(mass) + tol, "right")
        return np.arange(a, b)

    def lookup(self, key: str) -> Optional[int]:
        return self._first.get(str(key))


class HardNegativeDataset:
    """每条记录 = 一张训练谱 + 真值结构 key.

    record 键：mz, it（原始强度）, prec, adduct, instrument, ce, mode('positive'/...)，
    key（真值等价 key，须在 index 中）；可选 mass（真值中性质量，缺省用 index 中的质量）。
    """

    def __init__(self, records: Sequence[Dict], index: CandidateIndex, k_neg: int = 63,
                 ppm: float = 10.0, seed: int = 2026, max_peaks: int = MAX_PEAKS):
        self.records = list(records)
        self.index = index
        self.k_neg = int(k_neg)
        self.ppm = float(ppm)
        self.seed = int(seed)
        self.max_peaks = int(max_peaks)
        self.epoch = 0
        self.pos_ix = np.array([index.lookup(r["key"]) if index.lookup(r["key"]) is not None
                                else -1 for r in self.records], dtype=np.int64)
        if (self.pos_ix < 0).any():
            raise KeyError("%d records have keys missing from CandidateIndex"
                           % int((self.pos_ix < 0).sum()))
        # 峰数供分桶（prep 后最多 max_peaks）。
        self.n_peaks = np.array([min(len(r["mz"]), self.max_peaks) for r in self.records])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def negatives(self, i: int) -> np.ndarray:
        r = self.records[i]
        p = int(self.pos_ix[i])
        mass = float(r.get("mass", self.index.masses[p]))
        w = self.index.window(mass, self.ppm)
        if len(w) == 0:
            return w
        key = self.index.keys[p]
        w = w[self.index.keys[w] != key]
        if len(w):
            same_fp = (self.index.packed[w] == self.index.packed[p]).all(1)
            w = w[~same_fp]
        if len(w) > self.k_neg:
            rng = np.random.default_rng((self.seed, self.epoch, int(i)))
            w = np.sort(rng.choice(w, self.k_neg, replace=False))
        return w

    def __getitem__(self, i: int) -> Dict:
        r = self.records[i]
        mz, it = prep_peaks(np.asarray(r["mz"], float), np.asarray(r["it"], float),
                            float(r["prec"]), max_peaks=self.max_peaks)
        neg = self.negatives(i)
        return {"mz": mz, "it": it, "prec": float(r["prec"]),
                "ad": adduct_index(r.get("adduct")), "ins": instr_family(r.get("instrument")),
                "ce": ce_mean(r.get("ce")), "mode": mode_code(r.get("mode", "positive")),
                "pos": self.index.packed[int(self.pos_ix[i])],
                "neg": self.index.packed[neg] if len(neg) else
                np.zeros((0, self.index.packed.shape[1]), np.uint8)}


class HardNegativeCollate:
    """collate_fn：峰表按批内最大峰数填充，负例填到 K 并给 neg_mask。

    写成类而非闭包，spawn 方式启动的 DataLoader worker 也能 pickle。
    """

    def __init__(self, k_neg: int, nbytes: int):
        self.k_neg = int(k_neg)
        self.nbytes = int(nbytes)

    def __call__(self, items: List[Dict]) -> Dict:
        import torch
        B, K = len(items), self.k_neg
        N = max([len(x["mz"]) for x in items] + [1])
        mz = np.zeros((B, N), np.float32)
        it = np.zeros((B, N), np.float32)
        pad = np.ones((B, N), bool)
        negp = np.zeros((B, K, self.nbytes), np.uint8)
        nmask = np.zeros((B, K), bool)
        for j, x in enumerate(items):
            n = len(x["mz"])
            mz[j, :n], it[j, :n], pad[j, :n] = x["mz"], x["it"], False
            k = min(len(x["neg"]), K)
            negp[j, :k] = x["neg"][:k]
            nmask[j, :k] = True
        return {
            "mz": torch.from_numpy(mz), "it": torch.from_numpy(it), "pad": torch.from_numpy(pad),
            "prec": torch.tensor([x["prec"] for x in items], dtype=torch.float32),
            "ad": torch.tensor([x["ad"] for x in items], dtype=torch.long),
            "ins": torch.tensor([x["ins"] for x in items], dtype=torch.long),
            "ce": torch.tensor([x["ce"] for x in items], dtype=torch.float32),
            "mode": torch.tensor([x["mode"] for x in items], dtype=torch.float32),
            "bits_pos_packed": torch.from_numpy(np.stack([x["pos"] for x in items])),
            "bits_neg_packed": torch.from_numpy(negp),
            "neg_mask": torch.from_numpy(nmask),
        }


def make_collate(k_neg: int, nbytes: int) -> HardNegativeCollate:
    return HardNegativeCollate(k_neg, nbytes)


class PeakBucketBatchSampler:
    """峰数分桶：随机打乱后每 bucket_mult 个 batch 的样本按峰数排序再切批，批序再打乱.

    批内填充长度接近，注意力计算 O(N^2) 的浪费明显减少；随机性按 epoch 变化。
    """

    def __init__(self, n_peaks: Sequence[int], batch_size: int, bucket_mult: int = 50,
                 drop_last: bool = True, seed: int = 2026):
        self.n_peaks = np.asarray(n_peaks)
        self.batch_size = int(batch_size)
        self.bucket_mult = int(bucket_mult)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng((self.seed, self.epoch))
        idx = rng.permutation(len(self.n_peaks))
        chunk = self.batch_size * self.bucket_mult
        batches = []
        for s in range(0, len(idx), chunk):
            c = idx[s:s + chunk]
            c = c[np.argsort(self.n_peaks[c], kind="stable")]
            for b in range(0, len(c), self.batch_size):
                bb = c[b:b + self.batch_size]
                if len(bb) < self.batch_size and self.drop_last:
                    continue
                batches.append(bb.tolist())
        order = rng.permutation(len(batches))
        return iter([batches[o] for o in order])

    def __len__(self) -> int:
        n = len(self.n_peaks)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)


def make_loader(dataset: HardNegativeDataset, batch_size: int = 256,
                num_workers: Optional[int] = None, bucket: bool = True,
                pin_memory: Optional[bool] = None, prefetch_factor: int = 4,
                seed: int = 2026):
    """GPU 友好默认：num_workers=min(4, CPU 数)、pin_memory（有 CUDA 时）、
    persistent_workers、prefetch_factor=4、峰数分桶。Kaggle/Colab T4 机器
    通常 2–4 核，worker 再多只会争 CPU。"""
    import torch
    from torch.utils.data import DataLoader
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    collate = make_collate(dataset.k_neg, dataset.index.packed.shape[1])
    kw = dict(num_workers=int(num_workers), pin_memory=bool(pin_memory), collate_fn=collate)
    if num_workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=int(prefetch_factor))
    if bucket:
        sampler = PeakBucketBatchSampler(dataset.n_peaks, batch_size, seed=seed)
        return DataLoader(dataset, batch_sampler=sampler, **kw)
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=True, drop_last=True,
                      generator=g, **kw)

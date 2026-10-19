"""020：训练损失（BCE + 同窗口 CE 排序）与硬负例 collate 的 CPU 检查。"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.fp_train_data import CandidateIndex, HardNegativeDataset, make_loader
from src.models.fingerprint import FPNet
from src.train_fingerprint import (TrainConfig, bce_with_ranking_loss_numpy,
                                   fingerprint_ranking_loss_torch, train_step_torch,
                                   unpack_bits_torch)

NB = 6930


def _toy_batch(B=4, K=5, N=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    mz = torch.sort(torch.rand(B, N, generator=g) * 300 + 50, 1).values
    pad = torch.zeros(B, N, dtype=torch.bool)
    pad[:, -3:] = True
    pos = (torch.rand(B, NB, generator=g) < 0.05).float()
    neg = (torch.rand(B, K, NB, generator=g) < 0.05).float()
    mask = torch.ones(B, K, dtype=torch.bool)
    mask[0, 2:] = False
    return {"mz": mz, "it": torch.rand(B, N, generator=g), "pad": pad,
            "prec": torch.full((B,), 400.0), "ad": torch.zeros(B, dtype=torch.long),
            "ins": torch.zeros(B, dtype=torch.long), "ce": torch.full((B,), 30.0),
            "mode": torch.ones(B), "bits_pos": pos, "bits_neg": neg, "neg_mask": mask}


def test_loss_decreases_on_toy_batch():
    torch.manual_seed(0)
    net = FPNet(NB, d=32, layers=1, heads=2, drop=0.0)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    cfg = TrainConfig(mode="from_scratch", amp_fp16=True, ranking_temperature=10.0)
    b = _toy_batch()
    first = train_step_torch(net, opt, None, b, cfg, device="cpu")
    for _ in range(30):
        last = train_step_torch(net, opt, None, b, cfg, device="cpu")
    assert first["rank"] > 0 and first["n_neg"] == 4 * 5 - 3
    assert last["loss"] < first["loss"] * 0.7
    assert last["rank"] < first["rank"]


def test_all_masked_equals_bce_only_and_legacy_target():
    b = _toy_batch()
    logits = torch.randn(4, NB)
    full = fingerprint_ranking_loss_torch(logits, b["bits_pos"], b["bits_neg"],
                                          torch.zeros(4, 5, dtype=torch.bool))
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, b["bits_pos"])
    assert torch.allclose(full["total"], bce) and float(full["rank"]) == 0.0
    none = fingerprint_ranking_loss_torch(logits, b["bits_pos"], None, None)
    assert torch.allclose(none["total"], bce)
    # 旧 batch['target']：无负例键，只算 BCE
    torch.manual_seed(1)
    net = FPNet(NB, d=32, layers=1, heads=2, drop=0.0)
    legacy = {k: v for k, v in b.items() if k not in ("bits_pos", "bits_neg", "neg_mask")}
    legacy["target"] = b["bits_pos"]
    out = train_step_torch(net, torch.optim.SGD(net.parameters(), lr=0.0), None, legacy,
                           TrainConfig(mode="from_scratch"), device="cpu")
    assert out["rank"] == 0.0 and abs(out["loss"] - out["bce"]) < 1e-7


def test_masked_negative_is_ignored():
    b = _toy_batch()
    logits = torch.randn(4, NB)
    m = b["neg_mask"]
    ref = fingerprint_ranking_loss_torch(logits, b["bits_pos"], b["bits_neg"], m)
    neg2 = b["bits_neg"].clone()
    neg2[~m] = 1.0  # 被 mask 的负例改成极高分，结果不变
    alt = fingerprint_ranking_loss_torch(logits, b["bits_pos"], neg2, m)
    assert torch.allclose(ref["total"], alt["total"])


def test_torch_matches_numpy_reference():
    b = _toy_batch(B=1, K=5)
    logits = torch.randn(1, NB, dtype=torch.float64).float() * 0.1
    for tau in (1.0, 7.5):
        t = fingerprint_ranking_loss_torch(logits, b["bits_pos"], b["bits_neg"], b["neg_mask"],
                                           ranking_weight=0.5, ranking_temperature=tau)
        n = bce_with_ranking_loss_numpy(logits[0].numpy(), b["bits_neg"][0].numpy(),
                                        b["bits_pos"][0].numpy(), b["neg_mask"][0].numpy(),
                                        ranking_weight=0.5, ranking_temperature=tau)
        assert abs(float(t["bce"]) - n["bce"]) < 1e-5
        assert abs(float(t["rank"]) - n["rank"]) < 1e-4
        assert abs(float(t["total"]) - n["total"]) < 1e-4


def test_unpack_matches_numpy():
    bits = (np.random.default_rng(0).random((3, NB)) < 0.3).astype(np.uint8)
    packed = np.packbits(bits, axis=1)
    got = unpack_bits_torch(torch.from_numpy(packed), NB).numpy()
    assert (got == bits).all()


def test_hard_negative_loader_excludes_truth_equivalents():
    rng = np.random.default_rng(0)
    M = 40
    masses = np.sort(rng.uniform(300, 302, M))
    bits = (rng.random((M, NB)) < 0.05).astype(np.uint8)
    keys = np.array(["K%02d" % i for i in range(M)])
    keys[5] = keys[4]                 # 4/5 互为真值等价结构
    bits[6] = bits[4]                 # 6 与 4 指纹完全相同
    idx = CandidateIndex(masses, np.packbits(bits, axis=1), keys, NB)
    recs = [{"mz": rng.uniform(50, 290, 20 + 5 * j), "it": rng.random(20 + 5 * j),
             "prec": 301.0 + 1.007, "adduct": "[M+H]+", "instrument": "timsTOF",
             "ce": 30.0, "mode": "positive", "key": keys[4 + j]} for j in range(4)]
    ds = HardNegativeDataset(recs, idx, k_neg=8, ppm=10000.0)
    neg0 = ds.negatives(0)
    assert len(neg0) == 8
    assert not set(keys[neg0]) & {keys[4]} and 6 not in set(neg0.tolist())
    dl = make_loader(ds, batch_size=2, num_workers=2, bucket=True)
    b = next(iter(dl))
    assert b["bits_pos_packed"].shape == (2, (NB + 7) // 8)
    assert b["bits_neg_packed"].shape == (2, 8, (NB + 7) // 8) and b["neg_mask"].shape == (2, 8)
    assert b["mz"].shape[1] == b["pad"].shape[1] <= 128
    torch.manual_seed(0)
    net = FPNet(NB, d=32, layers=1, heads=2, drop=0.0)
    out = train_step_torch(net, torch.optim.AdamW(net.parameters(), 1e-3), None, b,
                           TrainConfig(mode="from_scratch"), device="cpu")
    assert out["n_neg"] == int(b["neg_mask"].sum()) and np.isfinite(out["loss"])

"""004-1c FPNet 目标域训练脚本（R3 研究用，未执行训练）.

手动研究用模块，不是测试文件（无 pytest/unittest，不做断言门控）。
torch 只在实际训练函数内延迟导入；本文件顶层导入不要求 torch，
以便本地做配置/预算/损失公式的人工核对。真实训练在 Colab T4 上手动发起。

三模式（冻结口径显式）：
- head_only：冻结 encoder（mz/nl/prec/ad/ins/blocks/norm），只训 head
  （2d->2048->nbits）。公开 checkpoint 作微调初始化。
- unfreeze_last2：head + 最后两层 Block 解冻，其余冻结。
- from_scratch：全部参数可训的对照；并排除 NP250 规范化结构
 （调用方按 manifest 行号/key 过滤，本文件不硬编码任何 key）。

预算入口（先小后大）：
- 200-500 步小预算估计：estimate_budget() 只算配置与步数，不启动训练；
  batch 从 16/32 选能稳定工作的值，峰数 128（MAX_PEAKS）。
  依据实测 step 时间再安排完整实验（见 004 计划）。
- 以结构均匀抽样并限制每结构谱数，比较 timsTOF 混合比例，避免
  enveda-180 大量谱单纯凭数量主导训练（采样由调用方实现，此处只记口径）。

精度与损失（020 修复：此前 train_step_torch 只算 BCE，排序项未接入）：
- CUDA 上前向用 torch.autocast('cuda', dtype=float16) + torch.amp.GradScaler；
  CPU 默认不开 autocast（cpu_bf16=True 时用 bfloat16，无 scaler）。
  损失一律在 float32 里计算；质量窗口过滤在 collate 外层 float64 完成。
- 损失 = BCE(logits, bits_pos) + ranking_weight * rank：
  s = logits · bits（与推理 Fmat @ Z 同口径的 f·z 分数），
  rank = CE([s_pos, s_neg_1..K] / ranking_temperature, target=0)，
  neg_mask=False 的填充负例置 -inf；没有有效负例的行不计入 rank，
  整批都没有负例时 total == BCE（向后兼容旧 batch['target']）。
  负例是同质量窗口候选，已排除真值等价结构（src/fp_train_data 负责）。
  旧的 hinge margin（ranking_margin）已删除：f·z 是几百位 logits 之和，
  固定 0.1 margin 没有尺度意义；改由 ranking_temperature 控制。
- 反向：scaler.scale(loss).backward() -> scaler.unscale_ -> 对可训参数
  clip_grad_norm_(grad_clip=1.0) -> scaler.step/update。
  推理 checkpoint 没有已确认的完整优化器状态，加载只作微调初始化，
  不称精确续训；保存必须含 optimizer/config/split（见 save_train_state）。

产物：artifacts/runs/004-research/<run-id>/ 下存 config/environment/metrics/
checkpoint/日志/逐分子表（列见 004 计划）。不对随机假输入训练后宣称完成。
"""

from __future__ import annotations

import copy
import datetime
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

TRAIN_MODES: Tuple[str, ...] = ("head_only", "unfreeze_last2", "from_scratch")
ESTIMATE_STEPS: Tuple[int, ...] = (200, 500)
BATCH_CHOICES: Tuple[int, ...] = (16, 32)
MAX_PEAKS = 128
DEFAULT_LR = 3e-4
DEFAULT_NBITS = 6930


def _utc_now() -> str:
    return (datetime.datetime.now(datetime.timezone.utc)
            .isoformat(timespec="microseconds").replace("+00:00", "Z"))


def _require_torch():
    try:
        import torch
        return torch
    except Exception as e:
        raise ImportError(
            "训练函数需要 torch（Colab T4 上安装后手动运行）。"
            "本地仅做配置/预算核对。原始异常：%s" % (e,)
        ) from e


@dataclass
class TrainConfig:
    """R3 训练配置（全部落盘，逐字可复现）."""

    mode: str = "head_only"
    steps: int = 500
    batch: int = 16
    max_peaks: int = MAX_PEAKS
    lr: float = DEFAULT_LR
    nbits: int = DEFAULT_NBITS
    seed: int = 2026
    amp_fp16: bool = True
    cpu_bf16: bool = False
    ranking_weight: float = 1.0
    # CE 温度：s/tau。f·z 尺度随模型而变，需按实测 loss 曲线调。
    ranking_temperature: float = 1.0
    grad_clip: float = 1.0
    # split 口径：训练排除的结构 key 集合名称与 manifest 路径（不内嵌 key）。
    split_manifest: str = "artifacts/runs/004-research/np_manifest.json"
    split_note: str = ("exclude NP250 shipped keys train-side; "
                       "canonical recheck pending RDKit")
    timstof_mix_note: str = ("stratified by structure, cap spectra per structure; "
                             "report timsTOF mix, avoid enveda-180 count dominance")

    def validated(self) -> "TrainConfig":
        if self.mode not in TRAIN_MODES:
            raise ValueError("mode must be one of %s" % (TRAIN_MODES,))
        if self.steps <= 0 or self.batch not in BATCH_CHOICES + (64, 128, 256):
            # 估计入口只推荐 16/32；完整训练允许更大 batch，但必须记录。
            if self.batch <= 0:
                raise ValueError("bad batch %r" % (self.batch,))
        if self.max_peaks != MAX_PEAKS:
            raise ValueError("max_peaks must be %d (feature contract)" % MAX_PEAKS)
        if self.nbits != DEFAULT_NBITS:
            raise ValueError("nbits must be %d (fp_bits选位)" % DEFAULT_NBITS)
        return self


def estimate_budget(steps_options: Sequence[int] = ESTIMATE_STEPS,
                    batch_options: Sequence[int] = BATCH_CHOICES,
                    max_peaks: int = MAX_PEAKS) -> Dict:
    """200-500 步小预算估计入口：只返回待测配置表，不启动训练。

    返回 {"configs": [...], "note": ...}，每个 config 为 TrainConfig 字典。
    调用方在 T4 上逐个手动跑，记录实测 step 时间/吞吐/峰值显存后再定完整轮数。
    """
    if max_peaks != MAX_PEAKS:
        raise ValueError("max_peaks must be %d" % MAX_PEAKS)
    configs = []
    for s in steps_options:
        for b in batch_options:
            c = TrainConfig(mode="head_only", steps=int(s), batch=int(b),
                            max_peaks=int(max_peaks))
            c.validated()
            configs.append(asdict(c))
    return {"configs": configs,
            "note": ("run each manually on T4, record seconds/step, throughput, "
                     "peak CUDA memory; then decide full schedule. "
                     "No training started by this call.")}


def trainable_names_for_mode(all_names: Sequence[str],
                             mode: str) -> List[str]:
    """按模式返回可训参数名（纯字符串规则，本地可核对，无需 torch）。

    规则：head_only -> 仅 'head.*'；unfreeze_last2 -> 'head.*' +
    最后两层 'blocks.{L-2, L-1}.*'（层数从 all_names 解析 max blocks idx）；
    from_scratch -> 全部。未知 mode 抛错。
    """
    names = list(all_names)
    if mode not in TRAIN_MODES:
        raise ValueError("mode must be one of %s" % (TRAIN_MODES,))
    if mode == "from_scratch":
        return names
    head = [n for n in names if n.startswith("head.")]
    if mode == "head_only":
        if not head:
            raise ValueError("no head.* params found")
        return head
    # unfreeze_last2：解析 blocks.i.* 的最大 i。
    idxs = []
    for n in names:
        if n.startswith("blocks."):
            try:
                idxs.append(int(n.split(".")[1]))
            except (IndexError, ValueError):
                continue
    if not idxs:
        raise ValueError("no blocks.* params found for unfreeze_last2")
    top = max(idxs)
    keep_layers = {top - 1, top}
    out = list(head)
    for n in names:
        if n.startswith("blocks."):
            try:
                if int(n.split(".")[1]) in keep_layers:
                    out.append(n)
            except (IndexError, ValueError):
                continue
    return sorted(set(out))


def apply_freeze(model, mode: str) -> List[str]:
    """对 torch 模型按模式设置 requires_grad（需 torch，T4 上调用）。

    先全冻（head_only/unfreeze_last2）或全开（from_scratch），返回可训名。
    """
    torch = _require_torch()
    _ = torch
    if mode not in TRAIN_MODES:
        raise ValueError("mode must be one of %s" % (TRAIN_MODES,))
    all_names = [n for n, _ in model.named_parameters()]
    keep = set(trainable_names_for_mode(all_names, mode))
    for n, p in model.named_parameters():
        p.requires_grad = (n in keep)
    return sorted(keep)


def bce_with_ranking_loss_numpy(pos_logits: Sequence[float],
                                neg_bits_matrix: Sequence[Sequence[int]],
                                pos_bits: Sequence[int],
                                neg_mask: Optional[Sequence[bool]] = None,
                                ranking_weight: float = 1.0,
                                ranking_temperature: float = 1.0
                                ) -> Dict[str, float]:
    """单样本 BCE + 同窗口 CE 排序损失的 numpy 参考实现（与 torch 版同式）。

    - bce：BCEWithLogits(pos_logits, pos_bits) 按位平均（只对真值指纹）。
    - s_pos = logits·pos_bits，s_neg_k = logits·neg_bits_k（同 ranking.py dot_score）。
    - rank = -log softmax([s_pos, s_neg...]/tau)[0]，neg_mask=False 的负例忽略；
      无有效负例时 rank=0。
    - total = bce + ranking_weight * rank。
    """
    import numpy as np
    lg = np.asarray(pos_logits, dtype=np.float64)
    pb = np.asarray(pos_bits, dtype=np.float64)
    if lg.shape != pb.shape:
        raise ValueError("logits/bits length mismatch")
    bce = float(np.mean(np.maximum(lg, 0) - lg * pb + np.log1p(np.exp(-np.abs(lg)))))
    nb = np.asarray(neg_bits_matrix, dtype=np.float64).reshape(-1, lg.shape[0])
    m = (np.ones(len(nb), bool) if neg_mask is None
         else np.asarray(neg_mask, dtype=bool).reshape(-1))
    nb = nb[m]
    if len(nb):
        sc = np.concatenate([[lg @ pb], nb @ lg]) / float(ranking_temperature)
        mx = sc.max()
        rank = float(mx + np.log(np.exp(sc - mx).sum()) - sc[0])
    else:
        rank = 0.0
    return {"bce": bce, "rank": rank,
            "total": bce + float(ranking_weight) * rank,
            "n_neg": int(len(nb))}


def unpack_bits_torch(packed, nbits: int):
    """uint8 packed（高位先行，同 np.unpackbits / candidates.pack_bits）在当前设备上解包。

    输入 (..., ceil(nbits/8)) uint8 -> (..., nbits) float32。用于把负例指纹以
    packed 形式搬到 GPU 后再展开，H2D 传输量是 float 的 1/32。
    """
    torch = _require_torch()
    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    return bits.reshape(*packed.shape[:-1], -1)[..., : int(nbits)].float()


def _batch_bits(batch: Dict, name: str, nbits: int, device: str):
    """取 bits_pos / bits_neg：float/bool 张量或 *_packed uint8 均可。"""
    if name in batch and batch[name] is not None:
        return batch[name].to(device, non_blocking=True).float()
    pk = batch.get(name + "_packed")
    if pk is not None:
        return unpack_bits_torch(pk.to(device, non_blocking=True), nbits)
    return None


def fingerprint_ranking_loss_torch(logits, bits_pos, bits_neg=None, neg_mask=None,
                                   ranking_weight: float = 1.0,
                                   ranking_temperature: float = 1.0) -> Dict:
    """float32 下的 BCE + CE 排序损失（返回张量 dict：total/bce/rank/n_neg）。

    logits/bits_pos: (B,D)；bits_neg: (B,K,D) 或 None；neg_mask: (B,K) bool，
    True = 有效负例。全部无效或无负例时 rank=0、total=bce。
    """
    torch = _require_torch()
    F = torch.nn.functional
    logits = logits.float()
    bits_pos = bits_pos.float()
    bce = F.binary_cross_entropy_with_logits(logits, bits_pos)
    zero = logits.new_zeros(())
    if bits_neg is None or bits_neg.numel() == 0 or float(ranking_weight) == 0.0:
        return {"total": bce, "bce": bce, "rank": zero, "n_neg": 0}
    bits_neg = bits_neg.float()
    B, K = bits_neg.shape[:2]
    if neg_mask is None:
        neg_mask = torch.ones(B, K, dtype=torch.bool, device=logits.device)
    neg_mask = neg_mask.to(logits.device).bool()
    s_pos = (logits * bits_pos).sum(-1, keepdim=True)
    s_neg = torch.einsum("bd,bkd->bk", logits, bits_neg)
    s_neg = s_neg.masked_fill(~neg_mask, float("-inf"))
    sc = torch.cat([s_pos, s_neg], 1) / float(ranking_temperature)
    valid = neg_mask.any(1)
    n_neg = int(neg_mask.sum().item())
    if not bool(valid.any()):
        return {"total": bce, "bce": bce, "rank": zero, "n_neg": 0}
    tgt = torch.zeros(int(valid.sum().item()), dtype=torch.long, device=logits.device)
    rank = F.cross_entropy(sc[valid], tgt)
    return {"total": bce + float(ranking_weight) * rank, "bce": bce,
            "rank": rank, "n_neg": n_neg}


def train_step_torch(model, optimizer, scaler, batch: Dict, config: TrainConfig,
                     device: str = "cuda") -> Dict[str, float]:
    """单步训练：前向（CUDA fp16 autocast）+ float32 BCE/CE 排序损失 + 裁剪.

    batch 键：mz/it/pad/prec/ad/ins/ce/mode（FPNet 输入），
    bits_pos (B,D) 真值指纹（旧键 target 作别名），
    bits_neg (B,K,D) 同窗口负例指纹（或 bits_neg_packed uint8 (B,K,ceil(D/8))，
    在设备上解包），neg_mask (B,K) bool。没有负例键时只算 BCE。
    scaler：CUDA + amp_fp16 时必须是 torch.amp.GradScaler；其余情况可为 None。
    返回 {"loss","bce","rank","n_neg","grad_norm"}（python float）。
    """
    torch = _require_torch()
    model.train()
    cfg = config
    dev = str(device)
    on_cuda = dev.startswith("cuda")
    keys = ("mz", "it", "pad", "prec", "ad", "ins", "ce", "mode")
    args = [batch[k].to(dev, non_blocking=True) for k in keys]
    bits_pos = _batch_bits(batch, "bits_pos", cfg.nbits, dev)
    if bits_pos is None:
        bits_pos = _batch_bits(batch, "target", cfg.nbits, dev)
    if bits_pos is None:
        raise KeyError("batch needs bits_pos (or legacy target / bits_pos_packed)")
    bits_neg = _batch_bits(batch, "bits_neg", cfg.nbits, dev)
    neg_mask = batch.get("neg_mask")
    if neg_mask is not None:
        neg_mask = neg_mask.to(dev, non_blocking=True)

    use_fp16 = bool(cfg.amp_fp16) and on_cuda
    use_bf16 = (not on_cuda) and bool(getattr(cfg, "cpu_bf16", False))
    if use_fp16 and scaler is None:
        raise ValueError("amp_fp16=True 需要传入 torch.amp.GradScaler('cuda')")
    optimizer.zero_grad(set_to_none=True)
    if use_fp16 or use_bf16:
        ctx = torch.autocast("cuda" if on_cuda else "cpu",
                             dtype=torch.float16 if use_fp16 else torch.bfloat16)
    else:
        import contextlib
        ctx = contextlib.nullcontext()
    with ctx:
        logits = model(*args)
    out = fingerprint_ranking_loss_torch(
        logits.float(), bits_pos, bits_neg, neg_mask,
        ranking_weight=cfg.ranking_weight,
        ranking_temperature=cfg.ranking_temperature)
    loss = out["total"]
    params = [p for p in model.parameters() if p.requires_grad]
    if use_fp16:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gn = torch.nn.utils.clip_grad_norm_(params, float(cfg.grad_clip))
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, float(cfg.grad_clip))
        optimizer.step()
    return {"loss": float(loss.detach().item()),
            "bce": float(out["bce"].detach().item()),
            "rank": float(out["rank"].detach().item()),
            "n_neg": int(out["n_neg"]),
            "grad_norm": float(gn.detach().item()) if hasattr(gn, "detach") else float(gn)}


def save_train_state(path: str, model_state: Dict, optimizer_state: Dict,
                     config: TrainConfig, split_ref: Dict) -> str:
    """保存训练状态：model + optimizer + config + split（torch 可选）。

    model_state/optimizer_state 为 state_dict（调用方传入 .state_dict() 结果
    或空占位）；config 为 TrainConfig；split_ref 至少含 manifest 路径、
    排除 key 数与 caliber 标注。torch 不可用时以 pickle 协议保存纯
    python 部分？不——统一用 torch.save（需 torch），本地只核对参数，
    不实际调用。返回写入路径。
    """
    torch = _require_torch()
    payload = {"model": model_state, "optimizer": optimizer_state,
               "config": asdict(config.validated()),
               "split": dict(split_ref), "saved_utc": _utc_now()}
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(payload, path)
    sidecar = os.path.splitext(path)[0] + ".config.json"
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump({"config": payload["config"], "split": payload["split"],
                   "saved_utc": payload["saved_utc"]},
                  f, ensure_ascii=False, indent=2)
    return str(path)

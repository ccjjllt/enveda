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

精度与损失：
- T4 首轮可用 FP16 AMP + GradScaler；m/z、正弦嵌入相关计算与质量过滤
  保持所需精度（float32/float64），不得照搬 TPU BF16 方案。
  AMP 只包前向/反向，质量窗口与评估 stays float64。
- 损失 = BCE（指纹多标签）+ 同质量窗口候选排序损失（margin ranking）：
  在训练结构内构造负例并排除真值等价结构，不从验证标签调负例。
  推理 checkpoint 没有已确认的完整优化器状态，加载只作微调初始化，
  不称精确续训；保存必须含 optimizer/config/split（见 save_train_state）。

产物：artifacts/004-research/<run-id>/ 下存 config/environment/metrics/
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
    ranking_weight: float = 1.0
    ranking_margin: float = 0.1
    # split 口径：训练排除的结构 key 集合名称与 manifest 路径（不内嵌 key）。
    split_manifest: str = "artifacts/004-research/np_manifest.json"
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
                                neg_logits_matrix: Sequence[Sequence[float]],
                                pos_bits: Sequence[int],
                                neg_bits_matrix: Sequence[Sequence[int]],
                                ranking_weight: float = 1.0,
                                ranking_margin: float = 0.1) -> Dict[str, float]:
    """BCE + 同质量窗口排序损失的 numpy 参考实现（本地可核对公式）。

    - BCE：对正例候选用 BCEWithLogits（按位平均），负例同样；返回 bce_pos/bce_neg。
    - ranking：每个正例分数 s+ = dot(sig? 此处用 logits 点积，与 ranking.py
      single 一致）应高于同窗口各负例 s-：loss_rank = mean(max(0, margin-(s+-s-)))。
    - total = bce + ranking_weight * loss_rank。负例必须已排除真值等价结构
      （调用方保证，此处只算数）。
    分数与 ranking.py dot_score 同口径，便于对照。
    """
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def bce_bits(logits: Sequence[float], bits: Sequence[int]) -> float:
        if len(logits) != len(bits):
            raise ValueError("logits/bits length mismatch")
        tot = 0.0
        for l, b in zip(logits, bits):
            p = sigmoid(float(l))
            p = min(max(p, 1e-7), 1.0 - 1e-7)
            t = 1.0 if b else 0.0
            tot += -(t * math.log(p) + (1.0 - t) * math.log(1.0 - p))
        return tot / max(1, len(bits))

    def dot(lg: Sequence[float], bt: Sequence[int]) -> float:
        return float(sum(float(l) for l, b in zip(lg, bt) if b))

    bce_p = bce_bits(list(pos_logits), list(pos_bits))
    bce_n_list = [bce_bits(list(nl), list(nb))
                  for nl, nb in zip(neg_logits_matrix, neg_bits_matrix)]
    bce_n = (sum(bce_n_list) / len(bce_n_list)) if bce_n_list else 0.0
    s_pos = dot(list(pos_logits), list(pos_bits))
    rank_terms = []
    for nl, nb in zip(neg_logits_matrix, neg_bits_matrix):
        s_neg = dot(list(nl), list(nb))
        rank_terms.append(max(0.0, float(ranking_margin) - (s_pos - s_neg)))
    rank = (sum(rank_terms) / len(rank_terms)) if rank_terms else 0.0
    total = (bce_p + bce_n) / (2.0 if neg_logits_matrix else 1.0) \
        + float(ranking_weight) * rank
    return {"bce_pos": float(bce_p), "bce_neg": float(bce_n),
            "rank": float(rank), "total": float(total)}


def train_step_torch(model, optimizer, scaler, batch: Dict, config: TrainConfig,
                     device: str = "cuda") -> Dict[str, float]:
    """单步 torch 训练（T4 上调用；本地不执行）。

    batch 键：mz/it/pad/prec/ad/ins/ce/mode/logits_pos/bits_pos/
    logits_neg/bits_neg（后两者为同窗口负例，已排除真值等价）。
    AMP：scaler 为 GradScaler 或 None；m/z 与正弦嵌入保持 float32，
    质量过滤在外层 float64，本函数内不降精度比较质量。
    返回当步 loss 字典（float）。
    """
    torch = _require_torch()
    model.train()
    ce = config
    mz = batch["mz"].to(device)
    it = batch["it"].to(device)
    pad = batch["pad"].to(device)
    prec = batch["prec"].to(device)
    ad = batch["ad"].to(device)
    ins = batch["ins"].to(device)
    ce_t = batch["ce"].to(device)
    mode = batch["mode"].to(device)
    use_amp = bool(ce.amp_fp16) and str(device).startswith("cuda")
    optimizer.zero_grad(set_to_none=True)
    if use_amp:
        # FP16 AMP（T4）；不用 TPU BF16 方案。scaler 由调用方创建。
        from torch.cuda.amp import autocast
        with autocast(dtype=torch.float16):
            logits = model(mz, it, pad, prec, ad, ins, ce_t, mode)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, batch["target"].to(device))
        if scaler is None:
            raise ValueError("amp_fp16=True 需要传入 GradScaler")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        logits = model(mz, it, pad, prec, ad, ins, ce_t, mode)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, batch["target"].to(device))
        loss.backward()
        optimizer.step()
    return {"loss": float(loss.detach().cpu().item())}


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

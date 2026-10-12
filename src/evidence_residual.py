"""008-A D2 门控残差评分（CPU 手动研究用模块，不是测试文件）。

背景（008-A 固定对照，单配置，不扫描）：
- D 的错配：`artifacts/007-r1d/run_D.py:193-205` 训练用全部候选未门控标准化残差
  `w.x_std`，`:97-100` 推理在 `has_lib==0` 时强制清零；`:117` 无 forward 候选永远按 v9
  续尾，忽略库残差（NP34/G1 21 个分子因此锁死，其中 NP 33 个真值实际有参考）。
- D2 只做一次固定对照：`α=0.25、τ=1.0`，特征沿用 D 13 维，不扫参。
  训练与推理共用同一门控评分函数：
    gated_residual = has_library * model(standardized_features)
  其中 model 为线性 `w.x_std`（无偏置），standardized 与 D 同口径
  （mu/sd 训练折内计算，sd<1e-12 列置零），门限 `has_library>0`。
- 分组排序保留 A/D 的跨组优先（scored 先于 unscored，先固定），组内一律按
  `final = v9 + clip(α*gated, -τ, +τ)` 重排。D 把 unscored 尾按 v9 锁死；
  D2 允许 unscored 在同一组内按 final 重排。α=0 时 final==v9，必须复现 A 全序
 （250/250 全序一致才算通过），tie 按 shortlist 序号（order 位置）。
- 跨组候选对（scored vs unscored）排序规则不允许改变，不应作为训练目标；
  D2 训练只保留同学组对。两端都无库（gated 差恒 0）的对也不提供梯度，训练中剔除。
  每分子每增强副本总权重 1/4（与 D 一致），在该副本内可学习对上均分；
  若某副本零可学习对则该副本权重落空（据实记录，不补）。
  μ/sd 与 D 一致：训练折全部 4 副本行（含增强分布），不读验证折。

仅依赖标准库 + numpy。零 GPU：本模块不导入 torch，不调用 cuda。
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

try:  # numpy 可用时用于训练/标准化；缺失不影响导入。
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None  # type: ignore

#: D2 沿用 D 13 维特征顺序（冻结，不扫描特征）。
D2_FEATURES_13: List[str] = [
    "has_lib",
    "lib_max",
    "lib_gap",
    "best_n_matched",
    "best_explained",
    "best_ent_sim",
    "best_same_adduct",
    "ce_missing",
    "neg_ce_diff",
    "ana_max",
    "ana_explained",
    "base_z",
    "has_lib_x_base",
]

#: has_library 门控列在特征向量中的下标（首列）。
HAS_LIB_INDEX = 0

#: D2 固定单配置（冻结，不扫描）。
D2_ALPHA = 0.25
D2_TAU = 1.0

#: 标准化零方差阈值（与 D / molecule_ranker 同口径）。
SD_EPS = 1e-12

#: 数值一致性容差。
CONSISTENCY_TOL = 1e-12

__all__ = [
    "D2_FEATURES_13",
    "HAS_LIB_INDEX",
    "D2_ALPHA",
    "D2_TAU",
    "feat_vec_d2",
    "standardize_fit",
    "standardize_apply",
    "standardize_row",
    "raw_linear_score",
    "gated_residual_row",
    "gated_residual_batch",
    "final_score",
    "rank_grouped_d2",
    "train_gated_pairwise_residual",
    "verify_gated_consistency",
]


def feat_vec_d2(row: Dict) -> List[float]:
    """D2 特征向量（与 D 同顺序，13 维）。

    输入：feature_table 行 dict（含 D2_FEATURES_13 全部键）。
    输出：13 维 float 列表。缺键抛错（不静默补零，避免口径漂移）。
    """
    try:
        return [float(row[k]) for k in D2_FEATURES_13]
    except KeyError as e:
        raise KeyError("D2 row missing feature %s" % (e,))


def _as_2d(X: Sequence[Sequence[float]]):
    """转 numpy 2D float 数组（内部用）。"""
    if _np is None:
        raise RuntimeError("need numpy for D2 residual")
    import numpy as np  # type: ignore
    Xa = np.asarray([[float(v) for v in r] for r in X], dtype=float)
    if Xa.ndim != 2 or Xa.shape[0] == 0 or Xa.shape[1] == 0:
        raise ValueError("need non-empty 2D X")
    return Xa


def standardize_fit(X: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    """训练折标准化拟合（与 D 同口径，训练/推理共用）。

    输入：训练折全部行（含 4 增强副本，不含验证折）。
    输出：(mu, sd)，列均值/总体标准差（ddof=0，与 run_D/molecule_ranker 一致）。
    """
    Xa = _as_2d(X)
    import numpy as np  # type: ignore
    mu = Xa.mean(axis=0)
    sd = Xa.std(axis=0)  # ddof=0，与 D 一致
    return [float(v) for v in mu.tolist()], [float(v) for v in sd.tolist()]


def standardize_apply(
    X: Sequence[Sequence[float]],
    mu: Sequence[float],
    sd: Sequence[float],
) -> List[List[float]]:
    """批量标准化（训练/推理共用同一函数）。

    零方差列（sd<1e-12）置零，与 D / molecule_ranker 同口径。
    """
    Xa = _as_2d(X)
    import numpy as np  # type: ignore
    mu_a = np.asarray([float(v) for v in mu], dtype=float)
    sd_a = np.asarray([float(v) for v in sd], dtype=float)
    if mu_a.shape[0] != Xa.shape[1] or sd_a.shape[0] != Xa.shape[1]:
        raise ValueError("mu/sd width mismatch")
    Xs = np.zeros_like(Xa)
    for j in range(Xa.shape[1]):
        if sd_a[j] < SD_EPS:
            Xs[:, j] = 0.0
        else:
            Xs[:, j] = (Xa[:, j] - mu_a[j]) / sd_a[j]
    return Xs.tolist()


def standardize_row(
    feat: Sequence[float],
    mu: Sequence[float],
    sd: Sequence[float],
) -> List[float]:
    """单行标准化（训练/推理共用，与 standardize_apply 同公式）。

    输入：单候选 13 维特征；输出：标准化后 13 维。
    """
    xs = standardize_apply([list(feat)], mu, sd)
    return [float(v) for v in xs[0]]


def raw_linear_score(x_std: Sequence[float], w: Sequence[float]) -> float:
    """线性 model(standardized_features) = w.x_std（无偏置）。

    pairwise 差分中偏置抵消，与 train_pairwise_residual 同口径。
    """
    if len(x_std) != len(w):
        raise ValueError("width mismatch %d vs %d" % (len(x_std), len(w)))
    s = 0.0
    for a, b in zip(x_std, w):
        s += float(a) * float(b)
    return float(s)


def gated_residual_row(feat: Sequence[float], model: Dict) -> float:
    """门控残差（训练/推理共用同一函数）。

    gated = has_library * model(standardized_features)，
    其中 has_library 取特征首列 `has_lib>0`（与行 has_lib 同源，不另传掩码，
    保证训练与推理门源一致），standardized 经 standardize_row（同 mu/sd）。
    """
    if len(feat) != len(model["w"]):
        raise ValueError("feature width must be %d" % len(model["w"]))
    has_lib = 1.0 if float(feat[HAS_LIB_INDEX]) > 0.0 else 0.0
    if has_lib <= 0.0:
        return 0.0
    xs = standardize_row(feat, model["mu"], model["sd"])
    return float(raw_linear_score(xs, model["w"]))


def gated_residual_batch(
    X: Sequence[Sequence[float]], model: Dict
) -> List[float]:
    """批量门控残差（与 gated_residual_row 同公式，用于一致性断言）。

    先经 standardize_apply 批量标准化，再逐行点积并乘门控。
    """
    Xs = standardize_apply(X, model["mu"], model["sd"])
    w = [float(v) for v in model["w"]]
    out: List[float] = []
    for i, r in enumerate(X):
        has_lib = 1.0 if float(r[HAS_LIB_INDEX]) > 0.0 else 0.0
        if has_lib <= 0.0:
            out.append(0.0)
        else:
            out.append(float(raw_linear_score(Xs[i], w)))
    return out


def final_score(v9: float, gated: float, alpha: float = D2_ALPHA,
                tau: float = D2_TAU) -> Tuple[float, bool]:
    """最终分 = v9 + clip(α*gated, -τ, +τ)（训练评估与推理共用）。

    返回 (final, clipped)，clipped 指 |α*gated|>τ+1e-12（与 run_D 同阈值）。
    α=0 时 final==v9（数值精确相等，不经浮点缩放）。
    """
    g = float(gated)
    a = float(alpha)
    t = float(tau)
    if a == 0.0:
        return float(v9), False
    prod = a * g
    clipped = bool(abs(prod) > t + 1e-12)
    if prod > t:
        prod = t
    elif prod < -t:
        prod = -t
    return float(float(v9) + prod), clipped


def rank_grouped_d2(
    order: Sequence[str],
    v9map: Dict[str, float],
    gated_map: Dict[str, float],
    scored_set: Sequence[str],
    alpha: float = D2_ALPHA,
    tau: float = D2_TAU,
) -> Tuple[List[str], Dict[str, float], int]:
    """D2 分组排序（训练评估与推理共用同一函数）。

    - 跨组优先固定：scored 组恒在前，unscored 组恒在尾（与 A/D 一致）。
    - 组内一律按 final = v9 + clip(α*gated) 排序，tie 按 shortlist 序号（order 位置）。
      D 把 unscored 尾按 v9 锁死；D2 允许 unscored 在组内按 final 重排。
    - α=0 时 final==v9，两组都退化为按 v9 + 序号排序，必须复现 A 全序。
    返回 (ranked, finals_all, n_clipped)，finals_all 含全部候选 final
    （unscored 也为 final，不再回退 v9）；n_clipped 为本分子触发 clip 的候选数。
    """
    scored = set(str(c) for c in scored_set)
    idx = {str(c): i for i, c in enumerate(order)}
    finals: Dict[str, float] = {}
    n_clipped = 0
    for c in order:
        cs = str(c)
        f, cl = final_score(float(v9map[cs]), float(gated_map.get(cs, 0.0)),
                            alpha, tau)
        finals[cs] = f
        if cl:
            n_clipped += 1
    scored_list = [str(c) for c in order if str(c) in scored]
    unscored_list = [str(c) for c in order if str(c) not in scored]
    scored_sorted = sorted(scored_list, key=lambda c: (-finals[c], idx[c]))
    unscored_sorted = sorted(unscored_list, key=lambda c: (-finals[c], idx[c]))
    return scored_sorted + unscored_sorted, finals, int(n_clipped)


def train_gated_pairwise_residual(
    X: Sequence[Sequence[float]],
    pair_index: Sequence[Tuple[int, int]],
    pair_weight: Sequence[float],
    base_diff: Sequence[float],
    l2: float = 1.0,
    lr: float = 0.5,
    iters: int = 500,
) -> Dict:
    """门控 pairwise 训练（Bradley-Terry，全批量，与推理共用标准化/门控）。

    模型：P(pos 胜) = sigmoid(base_diff + gated_pos - gated_neg)，
    其中 gated_c = has_lib_c * w.x_std_c（has_lib 取 X 首列 >0，与推理同源），
    base_diff 为固定底座差（v9_pos − v9_neg，不学习），L2 系数 l2。
    与未门控 train_pairwise_residual 的唯一区别是有效差分
      D_gated = hl_pos*x_std_pos − hl_neg*x_std_neg
    取代 D = x_std_pos − x_std_neg；mu/sd 仍按训练行计算，零方差列置零。
    调用方只传入可学习对（同组且至少一端有库）；两端都无库的对 D_gated恒0，
    不应传入（若传入则梯度为零，仅稀释权重，调用方负责剔除并记录）。

    超参冻结 l2=1.0、lr=0.5、iters=500（与 D 一致，不按验证标签调参）。
    输出 {w, mu, sd, n_pairs, l2, lr, iters, caliber}（无 bias，差分中抵消）。
    """
    if _np is None:
        raise RuntimeError("need numpy for D2 training")
    import numpy as np  # type: ignore
    Xa = _as_2d(X)
    pairs = [(int(a), int(b)) for a, b in pair_index]
    if not pairs:
        raise ValueError("need non-empty pairs")
    pw = np.asarray([float(v) for v in pair_weight], dtype=float)
    bd = np.asarray([float(v) for v in base_diff], dtype=float)
    if not (len(pairs) == pw.shape[0] == bd.shape[0]):
        raise ValueError("pairs/weights/base_diff length mismatch")
    # 门控与推理同源：取 X 首列 has_lib>0。
    hl = (Xa[:, HAS_LIB_INDEX] > 0.0).astype(float)
    mu = Xa.mean(axis=0)
    sd = Xa.std(axis=0)
    Xs = np.zeros_like(Xa)
    for j in range(Xa.shape[1]):
        if sd[j] < SD_EPS:
            Xs[:, j] = 0.0
        else:
            Xs[:, j] = (Xa[:, j] - mu[j]) / sd[j]
    d = Xa.shape[1]
    w = np.zeros(d, dtype=float)
    s = float(pw.sum())
    if s <= 0.0:
        raise ValueError("pair weights must sum positive")
    for _ in range(int(iters)):
        # 门控差分（与推理 gated_residual_batch 同公式的差分形式）。
        D = np.asarray([hl[a] * Xs[a] - hl[b] * Xs[b] for a, b in pairs])
        z = D.dot(w) + bd
        p = np.where(z >= 0.0, 1.0 / (1.0 + np.exp(-z)),
                     np.exp(z) / (1.0 + np.exp(z)))
        e = (p - 1.0) * (pw / s)
        gw = D.T.dot(e) + float(l2) * w
        w = w - float(lr) * gw
    return {"w": [float(v) for v in w],
            "mu": [float(v) for v in mu],
            "sd": [float(v) for v in sd],
            "n_pairs": int(len(pairs)),
            "l2": float(l2), "lr": float(lr), "iters": int(iters),
            "caliber": "008d2_gated_canonical_source_struct_v1"}


def verify_gated_consistency(
    X: Sequence[Sequence[float]],
    model: Dict,
    tol: float = CONSISTENCY_TOL,
) -> Dict:
    """数值一致性断言入口：同一候选训练打分 == 推理打分。

    批量路径（训练用 gated_residual_batch）与逐行路径（推理用
    gated_residual_row）必须在 tol 内一致。返回 {max_abs_diff, ok}，
    调用方在 mismatch 时判失败（不静默通过）。
    """
    batch = gated_residual_batch(X, model)
    per_row = [gated_residual_row(list(r), model) for r in X]
    md = 0.0
    for a, b in zip(batch, per_row):
        d = abs(float(a) - float(b))
        if d > md:
            md = d
    return {"max_abs_diff": float(md), "ok": bool(md <= float(tol)),
            "tol": float(tol), "n": int(len(batch))}

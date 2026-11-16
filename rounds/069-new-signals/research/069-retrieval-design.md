# 069 检索/图传播新信号设计（A/B/C）

整理日期：2026-11-16（北京）。状态：**设计文档，未执行任何新实验**。文中所有分数、比例、时长都是历史记录或量级预估，不是本轮结果；只有 Kaggle 公榜实际出分才算兑现。

---

## 0. 结论先行

### 0.1 三条实验的真实定位

先把结论说清楚，避免把工程量花在已经被管线覆盖的地方：

| 方向 | 表面说法 | 管线实际已有什么 | 真正的新东西 | 上限在哪 |
| --- | --- | --- | --- | --- |
| A 训练库 k-NN 检索 | 275k 结构谱库检索、投结构票 | Engine 2 已有中性质量 ±10 ppm 窗口、`lib`/`lib_mean`/`libsh`/`lib_cnt`、modified cosine、±200 Da top-80 analog、Tanimoto 传播；Engine 1 也做库检索 | **CE/仪器条件化的候选级证据**、**来源多样性**、**计数共识**（不是 max）、**候选级证据落盘** | 同分异构体排序（rank 2–25） |
| B 测试集内网络传播 | 从“重排 2–25”升级为跨分子分数传播 | N-1c 已有同 adduct 谱网、top-5 邻居、Tanimoto、`w^4`、`LAM=0.35` | **跨 adduct 边 + 中性质量门**、**共享候选身份传播**、**固定 top-1**、**学习权重** | 网络覆盖（可见集 114/400 有邻居） |
| C top-1 校准器 | 多信号一致才换 leader | 管线有碎片领跑、v8 质量领跑规则、extlib 领跑提升，但**没有学习式校准** | **交叉拟合的精度门控领跑切换** | rank-1 的 0.5–1.0 单分子收益 |

关键事实：**最终 25 个候选位绝大多数是满的**（可见集 400/400 都是 25；隐藏集预计相同；即使有空位，冻结管线末端还有 `assert_monotone` 与 DreamsNP 按质量补尾，见 068a 的冻结 cell 48）。候选池 710,701 是 train 结构 + COCONUT，Engine 2 运行时再并入 ChEBI/LIPID MAPS 与 PubChemLite。所以 A/B 的**核心**都**不是召回实验**：它们不能把池外结构变进来，只能改已有 25 个候选的顺序。真正的召回缺口在池外 de novo 生成，不在这三条里。

### 0.2 优先级（按“信息增益 / 成本”）

1. **A 的 Stage 1（纯 CPU，不占 GPU 配额）**：先在 3,000 个按结构分组、留出独立谱的结构上量“检索本身有没有同分异构体区分度”。这一步能在 0 GPU 的情况下杀掉或确认 A，投入产出最高。
2. **B0/B1（面板 + 最便宜的已核实通道升级）**：B0 在面板上原样复现 N-1c，校准面板灵敏度；B1 固定 top-1 后扫 `LAM`，检验 `LAM=0.5 → 0.484` 是否是 top-1 被改坏。B 的代码改动最小，且是对唯一已核实正增益（+0.001）的加固。
3. **C（同一面板，交叉拟合）**：上限最高（单分子 +0.5~1.0），但历史 top-1 改动多次退步；必须用“≥20 次切换、≥85% 精度、锁定 holdout 同号”三重门。
4. **A 的 Stage 2（端到端中名次重排）**：只有 Stage 1 过关才做。
5. **公榜探针**：每通过一个信号只推一次；先 B1，再 C，A 最后。

### 0.3 预期量级

| 组合 | 面板 ΔMRR | 折算公榜 Δ（按 Class1 30% / Class2 40% / Class3 30%） | 判断 |
| --- | ---: | ---: | --- |
| B1 单项 | +0.001 ~ +0.004 | +0.001 ~ +0.002 | 最可能先兑现 |
| B 全套 | +0.002 ~ +0.005 | +0.001 ~ +0.003 | 需要面板覆盖足够 |
| C 单项 | +0.004 ~ +0.012（Class 1 内） | +0.001 ~ +0.004 | 上限高、方差大 |
| A 中名次 | +0.001 ~ +0.006 | +0.001 ~ +0.003 | 取决于 Stage 1 |
| 三条全过 | +0.006 ~ +0.015 | +0.003 ~ +0.008 | 仍不足以单独进前 20 |

从 0.492 到前 20 线 0.503 需要 **+0.011**。三条实验全过也大概率只到 0.495–0.500；它们的价值是建立**第一批可复核的新信号**和一套能预测公榜的面板，而不是一次跳到前 20。真正要进前 20，还需要池外候选生成或一个当前通道没有的信息源（见 §8）。

---

## 1. 已核实的管线事实（本设计的事实基础）

### 1.1 数据与评分

| 项 | 已核实 |
| --- | --- |
| `train.parquet` | 2,539,608 行、18 列、21 row group、无 `molecule_id`/`spectrum_id`；官方 275,810 个结构；含 `normalized_smiles`、`inchikey14`、`molecular_formula`、`adduct`、`precursor_mz`、`ionization_mode`、`instrument_type`、`collision_energy_ev`、`ingest_lib`、峰数组 |
| 可见 `test.parquet` | 1,213 谱 / 400 分子 / 7 类 adduct，全部 timsTOF；是训练集的占位副本，同一条谱在 train 里存在 |
| 隐藏测试 | 约 400 分子、约 1,500 谱、10 类 adduct、timsTOF、157–1,159 Da、每分子 1–16 谱；同时含“有参考谱 / 库内无参考 / 库外 de novo”三类，比例未公开 |
| 指标 | MRR@25：首个正确候选排名 r 得 1/r，r>25 或缺失得 0，对分子平均 |
| 正确性 | 官方 RDKit 2026.03.3：去立体化学 → 互变异构规范化 → InChIKey 前 14 位。冻结节 cell 47 的 `metric_identity` 返回 `(ik14, heavy_atom_composition)`，`first_ranks` 用它对提交去重 |

### 1.2 当前管线（068a 底座）的通道与代码位置

| 通道 | 位置 | 已实现 | 本设计相关的缺口 |
| --- | --- | --- | --- |
| Engine 1 谱库 | 068a cell 36/39；`build_spec_cache`、`Library`、`Engine` | 从 `train.parquet` 建 2.5M 谱 cache；pool 710,701；FPNet/ranker；输出 `BASE[mid] = (smis, keys, lib_max, scs, forms)` | 只把**分子级** `lib_max` 导出；候选级库分数不落盘 |
| Engine 2（prvsiyan） | 068a cell 32 `casmi_engine.py` | `CFG.PPM_WIN=10`、`N_ANALOG=80`、`ANALOG_WIN=200`、`SIM_POWER=3`、`MZ_TOL=0.01`；`molecule_channels` 算 `lib`（plain max）、`lib_mean`、`libsh`（shifted/modified cosine；库内 shift = query precursor − ref precursor）、`lib_cnt`（参考行数）；`analog_search` 在目标中性质量 ±200 Da 内取参考、按 `target − ref_nm` 对齐后算 shifted cosine，取 top-80；`_analog_feats` 用 `max(Tanimoto × w^3)`；`blk_libx` 用 `lib_mean/libsh/lib_cnt`；`blk_anapol` 用正负极性 analog | 不分 CE、不分仪器、不数来源；analog 只取 max 不取计数；候选级证据不导出 |
| Engine 2 排序器 | 同上 `fit_rankers`/`fit_pv_rankers` | 31 个基础特征 + `base/mass/libx/anapol/fragx` 五组；`w_pv=0.88` 与自有 ranker 融合 | 特征是固定的；新证据只能后置重排，不能改 ranker（否则重演 044/045/049/067 的“离线好、公榜不兑现”） |
| PubChem 105M | 068a cell 33/34 | 按质量窗口检索、popularity prior、same-formula gate | 与本设计无关（A/B/C 不新增池外结构） |
| GLACIER/ICEBERG | 冻结 cell 41 | 前向谱预测，进入融合 | 与本设计无关 |
| 最终规则 | cell 21（Variant M 合并 + 文档先验 + IK14 去重）、cell 23（v8 质量领跑 + learned isomer reranker）、cell 24（N-1c）、cell 25（extlib rank-1 overlay） | 见下 | 见下 |
| 指标安全尾 | 冻结 cell 48 | `assert_monotone`；DreamsNP 按 ±10 ppm 补空位 | 列表是满的，补尾几乎不触发 |

### 1.3 N-1c 的精确行为（B 的基线）

068a cell 24，`_n1c(LAM=0.35, TAU=0.6)`：

- 只取 `[M+H]+` / `[M-H]-` 两类 adduct（第 19 行 `if str(ad) not in ('[M+H]+','[M-H]-')`）；可见集 7 类 adduct 中有 5 类（61/1,213 谱）完全没进网络。
- 同一分子同一 adduct 的多张谱：峰按 0.01 Da 分箱、取 top-120、强度开方后 L2 归一；不同 adduct 不混合。
- 边：只连**相同 adduct** 的两个分子，`mcos ≥ 0.6` 且匹配峰 ≥ 6；**没有中性质量一致性门**，也不允许跨 adduct。
- 传播：邻居只取它自己的前 5 名候选，权重 `1/rank` 归一；对本分子 25 个候选算 Morgan(r=2, 2048) Tanimoto；`N[i] += w^4 × Σ p × Tanimoto`。
- 重排：`z(-log rank) + LAM × z(N)`，对**全部 25 名**排序——代码上允许改 top-1（可见集实际 top1 changed = 0）。
- 可见集日志：molecules 400、with spectra 391、with neighbors 114、edges 109、changed lists 2、top1 changed 0、77s。
- 公榜：065a（LAM=0.25）0.488，068a（LAM=0.35）对应公开 0.492；`LAM=0.50` 公开消融为 0.484（−0.004）。

### 1.4 已被证伪或已中性的相邻尝试（不要重复）

| 尝试 | 结果 | 对本设计的含义 |
| --- | --- | --- |
| 066a 训练库连续余弦 mid-rank | 公榜 0.488，中性 | 066a 的索引只有 **182,941 个结构 / 2,539,608 行**（按官方 275,810 结构约 66% 覆盖，约 93k 结构在索引中没有参考谱）；且是 IK14 join 的分箱余弦，无 CE/来源/计数。A 必须修覆盖与条件化，不能是它的复跑 |
| 测试集内网络传播（早期，weight 0.25） | 离线面板 MRR 0.9706 → 0.9674（−0.0033） | B 的面板必须先复现 N-1c 的 +0.001，否则面板对小效应不可信 |
| 3 槽 tail rescue（0.477 底座） | −0.006；0.468 底座无变化 | 尾位注入的数学上限很低（见 §5.7），不作为 A/B 的主卖点 |
| 外部谱库弱命中提升 rank-1（052，TAU 0.55） | −0.010；TAU≥0.60 零命中 | A 的外部库部分只能做中名次/尾位，不能碰 top-1 |
| extlib overlay（TAU 0.80，K=4，LIB_TAU=0.90） | 065a 可见集 0 hits / 0 promoted | 外部库对当前测试覆盖极低，A 不能依赖它 |
| 自训 FPNet / 自有 ranker（044/045/049/067） | 离线 Tanimoto +26.5%，公榜 0.418–0.443；067 在 0.488 上中性 | 不训练新模型；只做后置重排 |
| top-1 规则改动（057/058/041） | −0.005 / −0.008 / −0.066 | C 必须用交叉拟合 + 精度门，不能拍规则 |
| `w_mid` 0.50→0.75（0.468 底座） | +0.004，是 59 臂里少数真正动榜的旋钮 | 中名次是唯一被证实的杠杆；A/B 的主战场正确 |

---

## 2. 方向 A：训练库检索证据表 + 中名次重排

### A.1 假设与为什么现有管线没覆盖

**假设**

- **H-A1（条件化）**：同一结构在不同 CE、不同仪器下的碎片模式差异很大。当前 `lib` 对候选的所有参考谱取 max，不分 CE、不分仪器；把参考谱限制到与 query 的 CE/仪器相近的一档，能提高同分异构体区分度。
- **H-A2（共识）**：单条最大余弦容易被一条“碰巧像”的参考谱抬高。要求**多个独立来源**（MassBank / MoNA / GNPS / MSnLib）或多个 CE 档同时支持，能降低假阳性；这正是当前 `lib_cnt`（只是行数）和 `ap`（只是 max Tanimoto）没有表达的。
- **H-A3（计数传播）**：Engine 2 的 analog 证据是 `max_t Tanimoto × w^3`；改成“有多少个独立 retrieved analog 在结构邻域内支持该候选”的计数，能区分“一个近邻”与“一群近邻”，对同分异构体更稳。
- **H-A4（落盘）**：候选级 `lib/libsh/lib_cnt` 现在只在 Engine 2 进程内使用，最终只导出分子级 `library_max`（`export_evidence.py` 对每个候选行写同一个 `base[2]`）。把候选级证据落盘后，才能在不重训 ranker 的前提下做中名次重排。

**为什么不是重复 066a**

| 维度 | 066a | A |
| --- | --- | --- |
| 索引覆盖 | 182,941 结构（约 66%） | `work/spec_*` + `train_structs.parquet`，275,810 全覆盖 |
| 相似度 | IK14 join 的分箱 sqrt 余弦 | plain + modified cosine（shift），CE/仪器软加权 |
| 聚合 | 单值余弦 | 直接分 + 软票数 + 来源多样性 + analog 计数 |
| 作用位置 | 只在 `library_max<0.90` 的分子上重排 2–25 | 同样只重排 2–25，但证据轴不同；`library_max` 只作为门控之一 |
| 结果 | 公榜中性 | 未知；Stage 1 先量检索本身 |

**明确不做的部分**：不再用 IK14 join 的分箱余弦；不把检索候选插到 rank 1；不替换尾部候选（列表已满，见 §5.7）；不改 Engine 2 的 ranker。

### A.2 算法

#### 输入

- `test.parquet` 的 query 谱（隐藏测试约 400 分子 / 1,500 谱）。
- 管线已经建好的 `work/spec_meta.parquet` + `work/spec_off.npy` + `work/spec_mz.npy` + `work/spec_it.npy`（Engine 1 cell 36 的 `build_spec_cache` 产物）。列含 `sid, lib, instr, adduct_ix, mode, precursor_mz, n_clean, nm, adduct, ce_mean, ce_min, ce_max, ce_n`。
- `work/train_structs.parquet`（275,810 行；`sid → smiles/inchikey14/mass`）。
- 最终 `submission.csv`（25 个候选/分子）。
- `frozen_candidate_evidence.parquet`（`baseline_rank`、`metric_key`、`library_max` 等）。

#### 索引

```text
rows = spec_meta where n_clean >= 3 and nm finite
# 1) 谱级去重：hash = sha1(round(mz,3) + round(intensity,3) + round(precursor,4) + adduct)
#    同一 hash 若标签冲突（stored ik14 不同）→ 整条丢弃；否则保留一条
# 2) 结构键：用 train_structs 的 smiles 做 metric_identity（只对进入候选集的 ~80k 结构算，不遍历 2.5M）
# 3) 按 nm 排序；保留 CSR 偏移，直接 memmap 峰数组
```

不在 A 里重新载入 3 GB 的 `train.parquet`；直接用管线 cache。这样索引构建从 20–30 min 降到 **1–3 min**，且与 Engine 1 的谱清洗口径一致。

#### 相似度

对 query 谱 `j`（clean 后 top-120、sqrt、L2）与参考谱 `r`：

```text
c_plain(j,r) = cosine_entropy(q_j, r)                  # 同 pv.entropy_sim
c_shift(j,r) = max(c_plain(q_j, r),
                   c_plain(q_j, shift(r, prec_j - prec_r)))   # modified cosine
g_ce(j,r)    = exp(-((CE_j - CE_r) / 10 eV)^2)  if both known else 1
g_ins(j,r)   = 1.0 if instr_family(j)==instr_family(r) else 0.9
w(j,r)       = max(c_plain, c_shift) * g_ce * g_ins
```

CE 用软权重而不是硬过滤：一个结构通常只有 1–3 张参考谱，硬过滤会把参考打空。`g_ins` 只做 0.9 的软折扣，不做 0。

#### 候选分数

对结构 `s`（其参考谱集合 `R_s`）：

```text
D_s    = max_j max_{r in R_s} w(j,r)                       # 直接证据（max，同 pv）
Dbar_s = mean_j max_{r in R_s} w(j,r)                      # 跨 query 谱平均，抗单谱偶然
V_s    = sum_j sum_{r in R_s} max(0, w(j,r) - tau0)        # 软票数（tau0=0.2）
Src_s  = |{ lib(r) : max_j w(j,r) >= tau1 }|               # 来源多样性（tau1=0.5）
K_s    = top-K structures by D (K=100)
A_s    = sum_{t in K_s, t != s} max(0, D_t - tau0) * 1[ Tanimoto(FP_s, FP_t) >= tau2 ]
Amax_s = max_{t in K_s, t != s} max(0, D_t - tau0) * Tanimoto(FP_s, FP_t)^alpha
```

`D_s/Dbar_s/V_s/Src_s` 是“直接检索 + 共识”，`A_s/Amax_s` 是“结构邻域传播”。`Amax_s` 对应 Engine 2 已有的 `ap`，**只作为对照**；新信号是 `A_s`（计数）和 `Src_s`（来源数）。

#### 重排公式

只对最终列表的 25 个候选、只动 rank 2–25，top-1 固定：

```text
rank_s = 当前列表名次（1..25）
R_s = z(-log rank_s)
    + l1 * z(Dbar_s)
    + l2 * z(V_s)
    + l3 * z(Src_s)
    + l4 * z(A_s)
new_order = argsort(-R_s) over positions 2..25
```

约束：

- 候选集合不变（纯置换），`assert set(new)==set(old)`。
- `new_order[0] == old_order[0]`（top-1 固定）。
- 若某分子 `library_max >= 0.95`，跳过（管线已经很强，别动）。
- 若某分子列表 < 5 个候选（smoke 占位），跳过。
- 任一步异常 → 原样保留 `submission.csv`。

#### 超参

| 参数 | 网格 | 说明 |
| --- | --- | --- |
| `K` | 50 / 100 / 200 | analog 近邻数 |
| `tau0` | 0.15 / 0.20 / 0.30 | 软票门槛 |
| `tau1` | 0.40 / 0.50 / 0.60 | 来源计数的单谱门槛 |
| `tau2` | 0.40 / 0.50 / 0.60 | 结构邻域 Tanimoto 门槛 |
| `alpha` | 1 / 2 / 3 | Tanimoto 幂 |
| `sigma_ce` | 10 / 20 / 40 eV | CE 软权重带宽 |
| `l1..l4` | 0 / 0.1 / 0.2 / 0.3 / 0.5（只选一组全局值） | 不在 400 分子上逐分子调 |

**只选一组全局超参**，在面板上按分子级 MRR 选；不做 per-molecule 权重（400 分子撑不住）。

#### 复杂度

- 索引：1–3 min，内存 ~4 GB（memmap + meta）。
- 打分：每 query 的候选参考谱数由中性质量窗口决定，通常 200–2,000 行；400 分子 × 1–4 谱 × 2,000 行 × 150 峰 ≈ 10^8–10^9 次比较，numba 下 **10–20 min**。
- 可选的 Stage 1（3,000 结构）：1–2 h CPU。
- 显存：0（纯 CPU）。

#### 伪代码

```python
# cell 25b (after extlib overlay)
meta = pq.read_table('work/spec_meta.parquet').to_pandas()
off  = np.load('work/spec_off.npy', mmap_mode='r')
mz   = np.load('work/spec_mz.npy',  mmap_mode='r')
it   = np.load('work/spec_it.npy',  mmap_mode='r')
struct = pq.read_table('work/train_structs.parquet').to_pandas()   # sid -> smiles, ik14

rows = dedupe_by_spectrum_hash(meta)                 # drop conflicting-label duplicates
nm_sorted = meta.nm.values[rows]
order = np.argsort(nm_sorted)
nm_sorted = nm_sorted[order]
sub = pd.read_csv('submission.csv', dtype=str, keep_default_na=False)
L = {m: s.split(';') for m, s in zip(sub.molecule_id, sub.smiles)}

for mid, q in queries.items():                       # q: list of spectra
    target = median(neutral_mass(q.precursor_mz, q.adduct))
    lo, hi = searchsorted(nm_sorted, target*(1-10e-6)), searchsorted(nm_sorted, target*(1+10e-6), 'right')
    cand_rows = rows[order[lo:hi]]
    D, Dbar, V, SRC = defaultdict(float), defaultdict(list), defaultdict(float), defaultdict(set)
    for j, (qm, qi, ce, ins) in enumerate(clean_spectra(q)):
        per_spectrum = {}
        for r in cand_rows:
            w = max(cos(qm,qi,r), mcos(qm,qi,r,shift=q.prec-r.prec)) \
                * g_ce(ce, r.ce_mean) * g_ins(ins, r.instr)
            s = r.sid
            per_spectrum[s] = max(per_spectrum.get(s, 0.0), w)
            D[s] = max(D[s], w); V[s] += max(0, w-0.2)
            if w >= 0.5: SRC[s].add(r.lib)
        for s, w in per_spectrum.items(): Dbar[s].append(w)
    K = topk(D, 100)
    cand = L[mid]
    for s in cand:
        A[s] = sum(max(0,D[t]-0.2) for t in K if t!=s and tanimoto(fp(s),fp(t))>=0.5)
        R[s] = z(-log(rank[s])) + l1*z(mean(Dbar[s] or [0])) + l2*z(V[s]) \
             + l3*z(len(SRC[s])) + l4*z(A[s])
    new = [cand[0]] + [cand[i] for i in argsort([-R[c] for c in cand[1:]])]
```

#### 消融

| 变体 | 相似度/聚合 | 作用 |
| --- | --- | --- |
| **A0** | 066a 的 IK14 join 分箱 sqrt 余弦 | 对照，确认“不是 066a 重演” |
| **A1** | plain + modified cosine，全覆盖索引 | 修 066a 的覆盖缺口 |
| **A2** | A1 + CE 软权重 | H-A1 |
| **A3** | A2 + 来源多样性 `Src_s` | H-A2 |
| **A4** | A3 + analog 计数 `A_s` | H-A3 |
| **A5** | A4 + `Amax_s` 对照 + 全局超参选择 | 完整版 |

Stage 1 在 P1 上跑 A0–A5；Stage 2 只把 Stage 1 胜出的 1–2 个配置带到 P3/P2。

### A.3 离线验证

#### 两条轨道（与 AGENTS 一致）

1. **未见结构（P2）**：按 `metric_identity` 整组留出；把该结构的所有谱从 library 删除，结构本身仍可留在 Engine 1 的 710k pool（这是“库内无参考”的 Class 2，也是 A/B/C 真正要解决的排序场景）。不要随机切谱。
2. **已见结构的独立谱（P3）**：结构留在 library，但留出 1–3 张 query 谱；这是“有参考谱但这条谱没被见过”的 Class 1。必须按谱 hash 去重，不能把同一条谱既放 library 又放 query。

#### 面板

| 面板 | 规模 | 用途 | 构建 |
| --- | --- | --- | --- |
| **P1** 检索面板 | 3,000 个结构（每结构留 1 张 query，至少 1 张留库） | A 的 Stage 1：只量直接检索相似度，不跑管线 | 从 train 按结构整组选；结构留库、留出 1 谱为 query；timsTOF 分层；质量 157–1,159 Da |
| **P3** Class 1 面板 | 500–800 个结构 | A 的 Stage 2、C 的主面板 | 结构留库，留 1–3 谱为 query；其余谱留库 |
| **P2** Class 2 面板 | 300 个结构 | A/B 的无参考场景、B 的覆盖诊断 | 结构所有谱移出 library；结构留在 Engine 1 pool |

P3 + P2 合在一个伪测试里（800 分子），一次 GPU 全管线跑完，省 GPU。

#### 指标

```text
r_i      = 真值在 25 列表中的首个名次（0 = 不在）
MRR@25   = mean(1/r_i)
recall25 = mean(1[r_i > 0])
top1     = mean(1[r_i == 1])

O_all  = mean(1[r_i>0] * (1 - 1/r_i))          # 完美重排的上限（含 top-1）
O_mid  = mean(1[r_i>=2] * (1/2 - 1/r_i))      # 只动 2–25 的上限（真值放到第 2）
O_lead = mean(1[r_i>=2] * (1 - 1/r_i))        # 只换 leader 的上限
```

`O_mid` 是 A/B 的硬上限：如果面板上 `O_mid < 0.02`，中名次工作不可能有大收益，直接停。`O_lead` 是 C 的硬上限。

#### 统计

- **按结构整组 bootstrap**（P1 一个结构有多谱，必须 cluster；P3/P2 一分子一结构，可按分子）；10,000 次；报 Δ 与 95% CI。
- 主指标是**配对 ΔMRR**，不是绝对 MRR（面板绝对分与公榜不可比）。
- 同时报：`#lists_changed`、`#top1_changed`、升级/降级的正确率（promotion precision = 被提前的候选里真值比例；demotion cost = 被推后的候选里真值比例）。
- 用 `O_mid` 报告“当前方法捕获了 oracle 的百分之几”。

#### 样本量与可检测效应

| 面板 | n | 稀疏改动的配对 Δ 可检测量级（80% power，粗略） |
| --- | ---: | ---: |
| P1 | 3,000 结构 | ~0.004 |
| P3 | 500 分子 | ~0.006 |
| P3 | 800 分子 | ~0.005 |
| C（60 次切换） | 800 分子 | ~0.008 |

所以门槛要设在可检测量级之上，且按面板规模分别设：

- P1 3,000 结构：Δ ≥ +0.005 才算 A 的检索相似度有真实增益。
- P3 800 分子：A Stage 2 / B 的 Δ ≥ +0.005；C 的 Δ ≥ +0.004 且切换 ≥ 20 次。
- P3 500 分子：只信 Δ ≥ +0.006；低于这个量级的中性结果不能区分“没用”与“面板太吵”，不要据此做方向决策。

这也是为什么 §7 的面板取 800（500 S1 + 300 S2）而不是 400：400 分子只够发现 +0.008 以上的效应，而我们要找的是 +0.002~+0.006。

#### 避免跨库重复 / 重复谱污染（A/B/C 共用清单）

1. **谱 hash 去重**：`sha1(round(mz,3), round(int,3), round(precursor,4), adduct)`；同 hash 不同标签（train 里已有记录的冲突）整条丢弃，并报数量。
2. **结构整组**：用 `metric_identity`（去立体 + 互变异构 + IK14），不用存储的 `inchikey14`；对同分子式的候选再抽查一遍，防止互变异构体以不同 stored IK14 留在 library。
3. **query 谱不得留在 library**：按 hash 删除；`assert` 查询谱的 hash 不在 cache。
4. **外部库**：extlib / MassBank 记录若与 train 的 hash 或 IK14 重复，从外部库删除；做外部注入实验时按**整来源留出**（例如留出全部 GNPS 记录），否则“外部库命中”只是 train 泄漏。
5. **重复来源**：同一结构在 MassBank/MoNA/GNPS 可能有多条；A 的 `Src_s` 天然按来源计数，但每个来源内部要 cap（每来源最多计 2 条），避免 10 条 MassBank 记录把一个候选抬起来。
6. **N-1c/B**：图只用谱，不用标签；S2 的 query 结构不能出现在 library（否则 1.0 余弦泄漏）；S1 只允许同结构的*其他*谱留库。
7. **COCONUT / DreamsNP**：面板结构若出现在 COCONUT 或 DreamsNP 补尾表里，从补尾表删除，或在面板运行关闭补尾。

### A.4 远端实现

| 项 | 方案 |
| --- | --- |
| 插入位置 | **068a cell 25（extlib overlay）之后**，新增 cell 25b；必须在 overlay 之后，才能看到最终列表；A 固定 top-1，不会动 overlay 提升的结构 |
| 输入 | `work/spec_meta.parquet`、`work/spec_off.npy`、`work/spec_mz.npy`、`work/spec_it.npy`、`work/train_structs.parquet`、`test.parquet`、`submission.csv`、`frozen_candidate_evidence.parquet` |
| 输出 | `retrieval_evidence.parquet`（molecule_id, smiles, D, Dbar, V, Src, A, Amax）、`retrieval_report.json` |
| 运行时间 | 索引 1–3 min + 打分 10–20 min（400 分子）；Stage 1 的 3,000 结构约 1–2 h（独立 CPU 内核） |
| 显存 | 0；RAM 峰值 4–6 GB（memmap；若不足，按 nm 窗口分片） |
| 失败保护 | 整段 `try/except`，异常时 `submission.csv` 原样保留；所有 assert 失败也只回退 |
| 是否需要新数据集 | 不需要；只用现有 competition + 068a 已有输入 |

### A.5 预期、失败模式、是否上公榜

| 失败模式 | 触发条件 | 缓解 |
| --- | --- | --- |
| CE 硬过滤把参考打空 | 每个结构只有 1–3 谱、CE 单一 | 只用软权重 `g_ce`，不硬删；`sigma_ce` 取 20–40 eV |
| 来源多 = 热门化合物 | 大天然产物有很多来源 | 与 `pop` 先验做偏相关；来源计数只在 `library_max` 中等的分子上启用 |
| analog 计数偏向 hub | 常见骨架被很多邻居支持 | 除以 `sqrt(deg(s))` 或用高 Tanimoto 门槛；与 `Amax` 对照 |
| 直接检索已饱和 | Stage 1 相对 A0 无增益 | 直接杀掉 A 的 Stage 2；不要用“公榜可能不一样”自我说服 |
| 中名次收益被稀释 | 面板 Δ 只在 S1 上，隐藏集 S1 只占 ~30% | 用 §0.3 的稀释公式折算；门槛按折算后设置 |
| 066a 重演 | 新轴没有超过 066a 的控制 | 把 066a 的 IK14 分箱余弦作为 A0 对照，必须显著超过它 |

**预期**：Stage 1 的检索 MRR 相对 A0 +0.00 ~ +0.05（同分异构体排序）；端到端中名次 +0.001 ~ +0.006；`P(≥+0.002) ≈ 0.3`。

**如果离线 recall@25 不涨，是否还上公榜？**

- 若 **recall 不涨但中名次 MRR 涨 ≥ +0.003**：值得一次探针。MRR 奖励名次，不奖励“在不在列表里”；把真值从第 8 名提到第 3 名是 +0.208 的单分子收益，recall 完全不变。历史上 `w_mid 0.5→0.75` 就是这种收益（+0.004）。
- 若 **recall 和中名次 MRR 都不涨**：不上。A 的价值就是排序；两者都不动说明新证据没有信息。
- 若 **recall 涨但 MRR 不涨**（真值从 30 名进到 25 名但仍在尾部）：先看 `O_mid`；如果 `O_mid` 很小，说明列表已满、尾部替换不值得，不上。

### A.6 预训练谱嵌入（本轮不做，但给出触发条件）

A 的第三种相似度是“预训练谱嵌入 + 向量检索”（DreaMS 或同类 encoder），而不是分箱余弦。它确实是当前管线没有的：Engine 2 只用 entropy cosine / modified cosine；DreamsFP 是“谱→指纹”预测器，不是检索嵌入。

**为什么本轮不做**

| 成本/风险 | 量级 |
| --- | --- |
| 2.5M 谱过 encoder | T4×2 上乐观 300–800 谱/s → 1–2.5 h；加预处理与批处理，约 3–4 h |
| 索引体积 | 1,024 维 fp16 × 2.5M ≈ 5 GB；必须作为私有 dataset 上传并在断网内核挂载、版本化 |
| 许可/复现 | DreaMS 权重可用，但需按主办方要求附原始训练代码与允许数据；衍生权重问题（讨论 749217）需单独核对 |
| 域移 | train 与隐藏 timsTOF 的仪器/CE 分布不同；嵌入可能反而不如分箱余弦 |
| 证据风险 | 仍是“离线更好、公榜不兑现”的高风险路径（044/045/049/067） |

**触发条件（只有满足才立项）**

在 P1 上量两个诊断：

```text
R_ceiling = # queries where true structure in top-100 by plain cosine but not top-5 / N
O_mid     = 只动 2–25 的 oracle 上限
```

- 若 `R_ceiling > 40%` 且 `O_mid > 0.03`：说明“真值能被检索到，但排序不对”，更好的相似度可能有价值，值得做嵌入检索。
- 若 `R_ceiling < 15%`：真值要么已经在 top-5，要么根本检索不到；瓶颈在 ranker 或候选池，不在相似度，**不要做嵌入**。
- 若 `O_mid < 0.02`：中名次本身没有空间，嵌入也不会有大收益。

即使触发，也先做“只对 P1 的 3,000 结构建嵌入索引”的小规模验证，不要一上来就嵌 2.5M 全库。

> 交叉阅读：`docs/research/069-frontier-models.md` 把 DreaMS 谱嵌入列为当前唯一同时满足“新信息、许可、9 h 断网”的模型路线。本节的触发条件回答的是另一个问题——**什么时候值得为它付出 5 GB 索引和 3–4 h GPU**；两者不冲突：前沿调研说“能做”，本节说“先证明检索天花板在相似度上再做”。

---

## 3. 方向 B：N-1d 跨分子候选分数传播

### B.1 假设与为什么现有管线没覆盖

**假设**

- **H-B1（跨 adduct 密度）**：可见集 7 类 adduct 中 N-1c 只用了 2 类；隐藏集有 10 类。用中性质量 + adduct shift 把 `[M+Na]+`、`[M+NH4]+`、`[M+CH2O2-H]-` 等谱接进网络，可把 114/400 的邻居覆盖提高。
- **H-B2（中性质量门）**：N-1c 只要求同 adduct + 碎片余弦，没有 `|nm_x−nm_y|` 门；加上 ±10 ppm 门可以去掉“碎片像但母体差很远”的假边，提高精度。
- **H-B3（共享候选身份）**：N-1c 只把邻居的 top-5 候选通过 Tanimoto 投给本分子的 25 个候选；如果邻居的列表里**直接出现**了本分子的真值（analog 候选），这条证据比 Tanimoto 更硬。用“共享候选 + 结构亲和”传播，精度更高。
- **H-B4（top-1 安全）**：N-1c 代码允许改 top-1；`LAM=0.50 → 0.484` 可能部分是 top-1 被改坏。固定 top-1 后，也许能把 `LAM` 提高到 0.5–0.7 而不退步。
- **H-B5（可学习权重）**：`LAM/TAU/w^4` 现在是手调的；在 800 分子面板上做分组交叉验证，选一组全局权重（不学 per-molecule 模型）。

**为什么现有管线没覆盖**：N-1c 的 5 个具体缺口（§1.3）就是 B 的 5 个改动；B 不新增候选，只改传播的边、消息和权重。早期“测试集内网络传播 weight 0.25”在离线面板是 −0.0033，而 N-1c 后来公榜 +0.001，说明这个方向**对实现细节极敏感**，必须用面板校准 + 小步提交。

### B.2 算法

#### 图

节点 = 分子（400–800）。对每对 `(x,y)`：

```text
nm_x = median_j neutral_mass(precursor_j, adduct_j)      # 用 30 个 adduct 表；未知 adduct 跳过
if |nm_x - nm_y| > 10 ppm * nm_x: skip                   # 硬门（B2）

w_xy = max_{j in x, k in y} mcos(q_j, q_k; shift=prec_j-prec_k)
       * g_ce(j,k) * g_ins(j,k)
if w_xy >= TAU and matched_peaks >= 6:
    E[x].append((y, w_xy)); E[y].append((x, w_xy))

E[x] = top_M(E[x])                                       # M=10，度上限，防 hub
```

`mcos` 允许跨 adduct：shift 用两张谱的 precursor 差，中性损失对齐；`g_ce` 用 10–40 eV 的软核；`g_ins` 同仪器 1.0、否则 0.9。

#### 消息与重排

```text
p_y(c)   = 1/(3 + rank_y(c)) if c in y's 25-list else 0
k(c,y)   = Tanimoto(FP_c, FP_{y,leader})                 # 邻居 leader 的结构亲和
M_x(c)   = sum_{y in E[x]} w_xy^p * p_y(c) * k(c,y)^alpha
           * 1[|mass_c - nm_x| <= 10 ppm]

S_x(c)   = z(-log rank_x(c)) + lam * z(M_x(c))
new      = argsort(-S_x) over positions 2..25, top-1 pinned
```

- 默认 **top-1 固定**：只对 2–25 排序。B1 变体专门检验“固定 top-1 + 更高 LAM”。
- 不做候选集合变化；如果 `c` 不在本分子列表里，`p_x(c)=0`，不参与（保持纯置换）。
- 学习超参：`TAU ∈ {0.5,0.6,0.7}`、`M ∈ {5,10,20}`、`p ∈ {1,2,4}`、`alpha ∈ {1,2}`、`lam ∈ {0,0.2,0.35,0.5,0.7}`；在面板上按分子级 MRR 选一组全局值，按结构分组 5 折。

#### 复杂度

- 中性质量预筛后，800 节点通常只剩 5k–20k 对；每对最多 4×4 张谱的 mcos。**3–5 min CPU**，无显存。
- 现有 N-1c 400 分子 77s；跨 adduct + 共享候选后预计 2–4×。

#### 伪代码

```python
# cell 24b: replace _n1c() with _n1d()
SP = load_query_spectra(test.parquet)          # all adducts, not just [M+H]+/[M-H]-
NM = {m: median(neutral_mass(s.prec, s.adduct) for s in SP[m])}

E = defaultdict(list)
for x, y in pairs(sorted_by_nm(NM)):            # nm-sorted, early stop
    if abs(NM[x]-NM[y]) > 10e-6*NM[x]: continue
    w = max(mcos(sx, sy, shift=sx.prec-sy.prec) * ce_w * ins_w
            for sx in SP[x] for sy in SP[y])
    if w >= TAU and peaks >= 6:
        E[x].append((y,w)); E[y].append((x,w))
for x in E: E[x] = sorted(E[x], key=-w)[:M]

for x in molecules:
    msg = defaultdict(float)
    for y, w in E[x]:
        for c in L[y][:25]:
            msg[c] += w**p * (1/(3+rank_y[c])) * tanimoto(fp[c], fp[L[y][0]])**alpha
    S = {c: z(-log(rank_x[c])) + lam*z(msg[c]) for c in L[x]}
    new = [L[x][0]] + sorted(L[x][1:], key=lambda c: -S[c])   # top-1 pinned
```

### B.3 离线验证

#### 变体矩阵

| 变体 | 改动 | 目的 |
| --- | --- | --- |
| **B0** | N-1c 原样（LAM=0.35, top-1 允许变） | 面板校准：必须与 068a 的 `submission.csv` 逐字节一致 |
| **B1** | 固定 top-1，扫 LAM 0.35/0.5/0.7 | 检验 LAM=0.5 的 −0.004 是否来自 top-1 |
| **B2** | B1 + 跨 adduct 边 + 中性质量门 | 密度与精度 |
| **B3** | B2 + 共享候选消息 `p_y(c)` | 新传播 |
| **B4** | B3 + 学习 `TAU/M/p/alpha/lam` | 全局权重 |

#### 指标

- 主指标：配对 ΔMRR@25（对 B0）；`#lists_changed`、`#top1_changed`（B1 起必须为 0）。
- 覆盖诊断：
  - `Cov_edges = #molecules with ≥1 edge / N`（可见集 114/400 = 28.5%）。
  - `Cov_truth = #molecules whose truth appears in some neighbor's 25-list / N`（B3 的可作用上限）。
  - `O_mid`（B 的硬上限）。
- 分组 5 折按 `metric_identity`；10k bootstrap。
- 分 adduct / 分 S1/S2 报告；跨 adduct 的增益应集中在非 `[M+H]+` 分子上。

#### 样本量

- 800 分子（P3 500 + P2 300）；B0 的校准必须在这 800 上做，不能只看可见集 400。
- 若 `Cov_edges < 20%` 或 `Cov_truth < 5%`，B 的天花板太低，直接停（不要用“隐藏集可能更密”自我安慰）。

#### 污染控制

- 图只用 query 谱；不得使用任何标签。
- S2 的 query 结构不能留 library（否则 1.0 余弦边泄漏）；S1 只留同结构的其他谱。
- 跨 adduct 边要防 adduct 误标：GNPS 的 adduct 误标有记录；用 `|nm_x−nm_y| ≤ 10 ppm` + `matched_peaks ≥ 8` 双重门。
- 同一条谱的重复（同 hash）在 query 内部去重，避免一个分子用 3 张相同谱刷高边权。

### B.4 远端实现

| 项 | 方案 |
| --- | --- |
| 插入位置 | **068a cell 24 之后**新增 cell 24b；或把 cell 24 的 `_n1c()` 调用换成 `_n1d()`（保留 `submission_pre_n1c.csv` 作为输入） |
| 输入 | `submission_pre_n1c.csv`、`test.parquet`；不需要 `train.parquet`（纯测试集内） |
| 输出 | `submission.csv`、`n1d_report.json`（edges、coverage、changed、top1_changed、per-adduct） |
| 运行时间 | 3–5 min CPU（400–800 分子） |
| 显存 | 0；RAM < 1 GB |
| 失败保护 | 异常时回退 `submission_pre_n1c.csv`；top-1 改动必须显式开关，默认关 |
| 新数据集 | 无 |

### B.5 预期、失败模式、是否上公榜

| 失败模式 | 触发 | 缓解 |
| --- | --- | --- |
| 图太稀 | 只有 114/400 有邻居 | 跨 adduct 提高密度；若仍 <20%，停 |
| 假边 | 同 adduct 碎片像但母体不同 | 中性质量门 + 更高匹配峰数 |
| hub 支配 | 常见骨架被很多邻居投票 | `top_M` + 度归一 + `alpha≥2` |
| top-1 被改坏 | 高 LAM | 默认固定 top-1；B1 单独检验 |
| 面板不可信 | B0 与 068a 不能逐字节一致，或 B0 的 Δ 与已知 +0.001 反号 | 先修面板，不进入 B2–B4 |
| 小效应被噪声淹没 | 500 分子、2 个列表变化 | 只看 ≥+0.003 的 OOF；否则只做一次探针 |

**预期**：B1 +0.001~+0.004；B2 +0.001~+0.002；B3 +0.000~+0.002；全套 +0.002~+0.005。`P(≥+0.002) ≈ 0.35`。

**如果离线 recall@25 不涨，是否还上公榜？**

B 是纯置换，recall@25 **不可能涨**（除非替换尾部候选，而列表是满的）。所以这个问题对 B 不成立；B 的唯一门槛是 MRR 和 top-1 安全。若 OOF ΔMRR ≥ +0.002 且 B1 的 top1_changed=0，值得一次探针；否则不上。

---

## 4. 方向 C：多信号 top-1 校准器

### C.1 假设与为什么现有管线没覆盖

**假设**

- **H-C1**：当前 leader 是**规则**决定的（cell 21 碎片领跑、cell 23 v8 质量领跑、cell 25 overlay 提升），没有“多通道同时同意”的学习式校准。把 `library_max`、质量 ppm、neural gap、engine agreement、邻居票数放在一起，可以在**高精度**下识别“leader 明显错、第二三名候选明显对”的少数分子。
- **H-C2**：历史 top-1 改动退步（057 −0.005、058 −0.008、041 −0.066）都是**拍规则**；交叉拟合 + 精度门 + 锁定 holdout 的校准器还没有试过。
- **H-C3**：MRR 的单分子收益不对称：换对 +0.5~1.0，换错 −1.0；所以只要精度 ≥85%、覆盖率 2–3%，就能有 +0.004~+0.01（面板）。

**为什么现有管线没覆盖**：`ranking.py` 里已经有 `BASE_FEATURES`/`make_features`（`engine_agreement`、`neural_top_gap`、`neural_agrees_top1`、`library_max`、`ice_z/gl_z` 等），但**最终决策没有用它**；它只是公共模块里的特征构造器。C 是第一次把这些通道放到一个**校准的领跑切换**里。

### C.2 算法

#### 特征（全部在推理时可得）

| 特征 | 来源 | 方向 |
| --- | --- | --- |
| `lib_L`, `lib_C` | A 的候选级证据；**无 A 时不可用**（分子级 `library_max` 不能当候选级分数用），C 退化为 C0 并去掉 `lib_C>=0.85` 门 | C 越高越好 |
| `ppm_L`, `ppm_C` | 068a cell 20/23 的 `MASS_PPM` | 绝对值越小越好 |
| `neu_L`, `neu_C`, `neu_gap` | `neural_candidate_features.parquet` 的 `neural_score` | C 越高越好 |
| `eng_L`, `eng_C`, `agree` | `frozen_candidate_evidence.parquet` 的 `main_rank/engine2_rank/pubchem_rank` | C 在更多 engine 前 3 越好 |
| `vote_L`, `vote_C` | A 的 `V`/`A`；无 A 时 0 | C 越高越好 |
| `msg_L`, `msg_C` | B 的 `M`；无 B 时 0 | C 越高越好 |
| `n_spectra`, `n_adducts`, `ce_span` | test.parquet | 上下文 |
| `src_L`, `src_C` | 候选来源（train / COCONUT / PubChem / NP） | 上下文 |
| `in_pool_L`, `in_pool_C` | IK14 是否在 275,810 train 结构里 | 上下文 |

#### 模型

优先**约束 logistic**（6–8 个系数，符号单调），而不是 GBM：

```text
P(C correct) = sigma(b0
    + b1 * (lib_C - lib_L)
    + b2 * (|ppm_L| - |ppm_C|)
    + b3 * (neu_C - neu_L)
    + b4 * (vote_C - vote_L)
    + b5 * (msg_C - msg_L)
    + b6 * (agree_C - agree_L))
b1..b6 >= 0
```

`lib_L/lib_C` 的门控用 A 的 `D_s`（max，表示“有没有强参考”）；连续项用 `Dbar_s`（跨谱平均，稳定）。训练集只取 **leader 不确定**的分子（`lib_L < 0.95` 且第 2/3 名有非零证据）；否则负样本压倒性，logistic 只会学到很低的基线。评估仍在全量分子上做，切换只在 eligible 子集里发生。

#### 切换规则

```text
switch  iff  P >= p*
        and  lib_C >= 0.85
        and  |ppm_C| <= 5
        and  neu_C - neu_L >= 0
        and  (vote_C - vote_L >= 1  or  A/B 特征缺失)
        and  lib_L < 0.95
        and  identity(C) valid and not duplicate
```

- 每个分子最多切一次；只把 C 移到 rank 1，其余顺序保持（或按原 rank 顺延）。
- 被 extlib overlay 提升的结构加白名单，不参与切换。
- 若 C 的候选不在当前 25 列表里，不允许（纯置换）。

#### 交叉拟合（严格防泄漏）

```text
groups = metric_identity(molecule)            # 结构整组，不按谱
outer = GroupKFold(5)
for train_idx, test_idx in outer:
    inner = GroupKFold(3) on train_idx
    fit beta on inner-train (constrained logistic)
    choose p* on inner-val maximizing ΔMRR s.t. precision>=0.85 and n_switch>=20
    refit beta on train_idx with chosen p*
    predict switches on test_idx
OOF = pooled test predictions
report ΔMRR, precision, n_switch, bootstrap CI
```

- **锁定 holdout**：从 800 里留 150 个结构，开发和选参都不碰；最后只跑一次，符号必须与 OOF 一致才提交。
- **leave-one-source-out**：按 `ingest_lib` 留出一个来源，检验规则不是只对某个库有效。
- **敏感性**：把 `p*`、`tau` 在 ±20% 内扰动，ΔMRR 不应变号。

#### 复杂度

秒级；无 GPU；无新数据。

### C.3 离线验证

- 面板：P3 500–800（Class 1，`lib` 有方差）+ P2 300（Class 2，`lib≈0`，规则应基本不触发）。
- 指标：**切换精度**（被切换的分子里 C 正确的比例）、**切换召回**（leader 错且 C 对的分子里被切的比例）、`ΔMRR`、`#switches`、`top1` 变化数。
- 分来源、分 adduct、分 `library_max` 档报告；如果 80% 的切换来自一个来源，规则不可信。
- 用 `O_lead` 报告上限：如果 `O_lead < 0.02`，C 不可能有大收益。
- 样本量：800 分子；eligible（leader 不确定且第 2/3 名有证据）预计 300–500。若 eligible < 150，不拟合（系数估计不稳）；若 OOF 切换事件 < 20，说明规则覆盖太低、精度估计不稳，**不做公榜探针**，只记录为“覆盖不足”。不要用 <20 次切换的精度去赌 top-1。

### C.4 远端实现

| 项 | 方案 |
| --- | --- |
| 插入位置 | **068a cell 25（或 A 的 25b）之后**，新增 cell 26；必须是最后一个改列表的步骤 |
| 输入 | `submission.csv`、`frozen_candidate_evidence.parquet`、`neural_candidate_features.parquet`、`MASS_PPM`、A/B 的 report（可选） |
| 输出 | `submission.csv`、`top1_report.json`（switches、precision proxy、features） |
| 运行时间 | 秒级 |
| 显存 | 0 |
| 失败保护 | 异常/白名单/重复/无效 identity → 原样保留 |
| 新数据集 | 无 |

### C.5 预期、失败模式、是否上公榜

| 失败模式 | 触发 | 缓解 |
| --- | --- | --- |
| 精度不够 | 规则在 OOF 只有 70–80% | 提高 `p*`、加 `lib_C≥0.9`、减少切换 |
| 过拟合面板 | 800 分子、10+ 特征 | 约束 logistic、6 个系数、锁定 holdout、leave-one-source-out |
| 来源混杂 | 切换集中在天然产物库 | 分来源报告；必要时按来源分层 |
| 与 v8 质量规则冲突 | leader 已被质量规则移动 | 只在 `lib_L<0.95` 且 `ppm_C≤5` 时切；v8 已移动的分子加标记 |
| 隐藏集类别比例不同 | Class 1 只占 20–30% | 用 §0.3 稀释公式；门槛按折算后设置 |
| 历史重演 | 公榜 top-1 改动再次退步 | 只推一次；若负，回退 068a，不再加大 |

**预期**：面板 Class 1 内 `ΔMRR +0.004~+0.012`；折算公榜 +0.001~+0.004。`P(≥+0.002 公榜) ≈ 0.25`。

**如果离线 recall@25 不涨，是否还上公榜？**

C 与 recall 无关（它不改候选集合，只换 leader）。门槛是 OOF ΔMRR、切换数和精度；recall 不涨不影响判断。但若 `O_lead` 很小（真值很少在第 2–25 名），C 的收益上限就低，不值得为它冒险。

---

## 5. 共用离线面板与验证协议

### 5.1 两条轨道

| 轨道 | 定义 | 用途 | 不能做什么 |
| --- | --- | --- | --- |
| **未见结构（谱层面）** | 按 `metric_identity` 整组留出；该结构所有谱移出 library；结构仍可能在 Engine 1 pool / 预训练权重中（已知泄漏边界，见 §5.6） | Class 2 排序；A/B 的无参考场景 | 不能把该结构的任何谱留在 library；不能随机切谱 |
| **已见结构独立谱** | 结构留 library；留 1–3 张谱为 query；其余留 library | Class 1；C 的主面板；A 的直接证据 | 不能把 query 的同 hash 谱留在 library；不能跨结构重复 |

### 5.2 面板构建步骤

```text
1. 读 train.parquet（只读需要的列；21 row group 流式）
2. 谱 hash 去重；冲突标签丢弃；报数量
3. 选结构：
   S1: >=2 谱、至少 1 张 timsTOF、157-1159 Da、排除可见 400 分子、排除外部库已有结构
   S2: >=1 谱、同质量范围、与 S1 不相交、排除外部库已有结构
   分层：分子式频次（常见/稀有各半）、adduct 组合、来源（NP/合成）
4. 每个 S1 结构留 1-3 张谱为 query；其余留 library
   每个 S2 结构所有谱移出 library；query 取 1-3 张
5. 生成 pseudo test.parquet（12 列同 schema；molecule_id=q000001...；spectrum_id=s...）
6. 生成 pseudo sample_submission.csv（同 molecule_id；25 个 CCO 占位）
7. 写 filtered train.parquet：删除 query 谱（S1）或全部谱（S2）；再删除与 query 同 hash 的行
8. labels.parquet：molecule_id, true_ik14, stored_ik14, smiles, class(S1/S2), source, formula, adducts, n_spectra
9. assert：query hash 不在 filtered train；S2 的 query stored ik14 不在 filtered train 的 spectrum 行里（结构本身可以留在 Engine 1 pool / train_structs，这是已知边界）
```

### 5.3 指标与统计

见 §A.3 的公式；统一用 `first_ranks`/`metric_identity`，按结构整组 bootstrap。所有面板结果报**配对 Δ**，不报绝对分。

### 5.4 跨库重复 / 重复谱污染清单

见 §A.3 的 7 条；A/B/C 共用。额外两条：

- **外部库整来源留出**：做外部注入时，留出整个 `ingest_lib` 来源（如 GNPS），不要只留出单条记录，否则“外部库命中”只是 train 泄漏。
- **同结构多来源 cap**：每个来源最多计 2 条参考谱；否则 MassBank 的 10 条记录会把一个候选抬起来。

### 5.5 面板校准（必须先做）

1. 在 P3+P2 上跑冻结管线（伪测试），保存 `submission.csv` 和 `submission_pre_n1c.csv`。
2. 用 `_n1c(LAM=0.35)` 从 `submission_pre_n1c.csv` 重算，必须与保存的 `submission.csv` **逐字节一致**。不一致 → N-1c 实现/输入有问题，先修。
3. 用 `_n1c(LAM=0.5)` 重算，记录 `top1_changed`；如果 top-1 变了且 ΔMRR 为负，支持“LAM=0.5 的 −0.004 来自 top-1”的假设，B1 的价值上升。
4. 面板对 N-1c 的 Δ 如果与已知公榜 +0.001 反号，说明面板对小效应不可信：**只信 ≥+0.005 的效应**，把 +0.001~+0.003 留给公榜探针判断。

### 5.6 泄漏边界

- 预训练 FPNet / GLACIER / ranker 在 train.parquet 上训练过，面板结构对模型权重是“见过”的；P2 的绝对分因此会高于真实 Class 3。**只用于同一 base 上的相对比较，不用于预测绝对公榜分**。
- Engine 2 的 ranker 用的是固定的 `rank_train.npz` 模拟行，不从面板 train 现训；所以过滤 train 不会改变 ranker，只改变 library/pool。
- 隐藏集的 Class 1/2/3 比例未知；面板按 30/40/30 与 40/40/20 两种情景各报一次折算。

### 5.7 尾位注入的数学上限（为什么 A/B 不做“新候选”）

最终列表通常是 25 个满位（可见集 400/400；隐藏集预计相同；空位还有 DreamsNP 补尾）。若把第 25 名替换成检索候选，设：

```text
p25 = P(当前第 25 名是正确结构)
q   = P(检索候选是正确结构)
ΔMRR_per_molecule = (q - p25) / 25
```

即使 `q = 0.05`、`p25 = 0.01`，每分子也只有 0.0016；`q = 0.02`、`p25 = 0.01` 只有 0.0004。要让尾位注入有 +0.003 的整体收益，需要 `q - p25 ≈ 0.075`，这在外部库弱命中（052 的余弦 0.55–0.57）上不现实。

所以：

- A/B **只做中名次置换**；不替换第 25 名，不新增候选。
- 面板先报 `recall25`；若 ≥0.93，尾位注入直接放弃（正确结构已经在列表里，尾部替换只会伤害）。
- 若 `recall25 < 0.85`，说明候选集合本身有缺口，应该转向池外生成，而不是在 25 名内做尾部交换。

---

## 6. 实验矩阵

| ID | 依赖 | 面板/数据 | 改动 | 主指标 | 通过门槛 | 计算 | 失败后动作 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **E0a** | — | P1 3,000 结构 | 建检索索引 + A0 066a 控制 + A1 plain/shift | recall@25 / MRR@25 | 控制可复现 pv 风格 | CPU 2 h | 修索引 |
| **E0b** | — | P3 500 + P2 300 | 伪测试全管线 base | 生成 base lists | 800 行合法、top1 稳定 | GPU 5 h | 缩面板 |
| **E0c** | E0b | 同上 | 落盘 A 证据 + B 消息 | 覆盖/诊断 | 无 | CPU 1 h | — |
| **E1** | E0a | P1 | A0→A5 消融（CE/来源/计数） | ΔMRR@25 | ≥ +0.01 且 CI>0（相对 A0） | CPU 3 h | 杀 A Stage 2 |
| **E2** | E1,E0c | P3/P2 | A 中名次重排 | ΔMRR@25 | ≥ +0.003 且 top1=0；`O_mid`≥0.02 | CPU 1 h | 不上 A |
| **E3** | E0b | P3/P2 | B0→B4 | ΔMRR@25 | B0 逐字节一致；B1+ ≥+0.002；top1=0 | CPU 1 h | 缩到 B1 |
| **E4** | E0c | P3/P2 | C0（无 A/B）/C+A/C+B/C+A+B | ΔMRR, precision | ≥+0.004；≥20 切换；≥85% 精度；锁定 holdout 同号 | CPU 1 h | 不推 C |
| **E5** | 通过项 | 隐藏测试 | 069a 单信号提交 | public MRR | 每次只改一个信号 | GPU 3.3 h | 回退 |

**E0b 的 800 分子是否太多？** 065a 隐藏 400 分子约 2.7 h；800 分子约 4–5 h，加伪数据准备 30–45 min，仍在 9 h 内。先跑 50 分子测吞吐；若 >0.5 min/分子，缩到 600（400 S1 + 200 S2）。

---

## 7. 优先级与排期

### 7.1 现在（无 GPU，11-21 之前）

1. **E0a + E1**：纯 CPU，不占 GPU 配额；在 3,000 结构上量 A 的检索。若 E1 不过，A 直接停，省下后面的 GPU。
2. 同时准备 **E0b 的伪数据构建脚本**（本地只写代码，不跑全量；用 50 分子小样本核对 schema 与 filter）。
3. 不推送任何内核；068a 仍按原计划等 11-21 08:00 刷新后推送。

### 7.2 11-21 之后（GPU 刷新，30 h/周）

| 顺序 | 动作 | GPU | 产出 |
| --- | --- | --- | --- |
| 1 | 推送 **068a**（既有计划） | 2.7 h | 0.492 复现 |
| 2 | 跑 **E0b** 800 分子伪测试 | 4–5 h | base lists + 证据 |
| 3 | **E3-B0/B1**（CPU） | 0 | 面板校准 + top-1 假设 |
| 4 | **E4-C**（CPU） | 0 | 校准器 OOF |
| 5 | 若 E1 过：**E2-A**（CPU） | 0 | A 中名次 |
| 6 | **E5**：B1 探针（2.8–3.3 h） | 2.8–3.3 h | public 分 |
| 7 | 若 B1 ≥ +0.002：C 探针；A 最后 | 2.8–3.3 h/次 | 逐信号归因 |

GPU 预算：2.7 + 5 + 3.3 + 3.3 ≈ 14.3 h，留 15 h 余量给失败重跑。CPU 工作不占 GPU 配额，但受 9 h/会话限制；E1/E3/E4 各自拆成独立 CPU 内核。

### 7.3 决策树

```text
E1 不过 → 杀 A（不再做 A 的 Stage 2）；保留 A 证据表给 C 用
E1 过 → E2；E2 不过 → 杀 A 端到端；A 证据仍可给 C
B0 不一致 → 修面板；不进入 B2–B4
B1 过 → 推 B1 探针；负 → 回退 068a，不再加 LAM
C OOF 不过 → 不推；过 → 推一次，锁定 holdout 同号
```

---

## 8. 与 0.492 / 0.503 的关系（战略判断）

### 8.1 稀释公式

隐藏集三类比例未知，面板结果不能直接当公榜：

```text
Δ_public ≈ f1 * Δ_S1 + f2 * Δ_S2 + f3 * Δ_S3
f1 + f2 + f3 = 1
```

- A 的直接证据主要作用在 S1（有参考谱），S2 靠 analog；B 只作用在有邻居的子集（可见 ~28%）；C 主要作用在 S1。
- 若面板 S1 上 Δ=+0.01，而 S1 只占 30%，公榜只有 +0.003。**任何“面板 +0.01”都要先除以 3 再判断值不值得推。**
- 报告时同时给 30/40/30 与 40/40/20 两种情景，避免用单一比例自证。

### 8.2 这三条能不能到前 20？

- 0.492 → 0.503 需要 +0.011，相当于面板 Class 1 上 +0.03~+0.04 的效应；目前没有任何单条信号有这个量级。
- 三条全过的乐观估计是 +0.003~+0.008，到 0.495–0.500，接近但不到 0.503。
- 真正能跳过 0.503 的，是**池外候选生成**（Class 3）或一个当前通道没有的信息源；A/B/C 是“把已有候选排得更好”，不是“找到新候选”。这一点必须写进下一轮计划，避免把 0.500 当成终点。

### 8.3 不要做的事（省 GPU）

- 不要重跑 066a 的 IK14 分箱余弦（中性）。
- 不要把 A 的检索候选插到 rank 1（052 的 −0.010）。
- 不要在未固定 top-1 时把 N-1c `LAM` 提到 0.5 以上（0.484）。
- 不要训练新 FPNet / ranker（044/045/049/067 的教训）。
- 不要用随机切谱或存储 IK14 做留出/分组。
- 不要相信 400–800 分子上面板 Δ < +0.004 的“正结果”。
- 不要在同一个内核版本里叠两个新信号再提交（无法归因）。

---

## 9. 附录

### 9.1 符号

| 符号 | 含义 |
| --- | --- |
| `nm` | 中性单同位素质量；优先用 `molecular_formula` 算术，缺失时用 precursor+adduct 表 |
| `D_s` / `Dbar_s` | 候选直接检索证据（max / 跨 query 谱平均） |
| `V_s` | 软票数（超过 `tau0` 的相似度之和） |
| `Src_s` | 来源多样性（不同 `lib` 的数量） |
| `A_s` / `Amax_s` | analog 计数 / analog 最大 Tanimoto |
| `M_x(c)` | 邻居对本分子候选 `c` 的传播消息 |
| `O_mid` / `O_lead` | 只动 2–25 / 只换 leader 的 oracle 上限 |
| `f1/f2/f3` | 隐藏集 Class 1/2/3 比例（未知） |

### 9.2 代码与文件索引

| 内容 | 位置 |
| --- | --- |
| N-1c | `notebooks/068a/enveda-068a-n1c-lam035.ipynb` cell 24 |
| extlib overlay | 同上 cell 25 |
| 最终合并 / 文档先验 / IK14 去重 | 同上 cell 21 |
| v8 质量领跑 + learned isomer reranker | 同上 cell 23 |
| Engine 2（prvsiyan） | 同上 cell 32（`casmi_engine.py`） |
| Engine 1 谱库 cache | 同上 cell 36/39 |
| `metric_identity` / `assert_monotone` | 冻结管线 cell 47/48（068a cell 12 的 pinned sources） |
| 068a 输入清单 | `notebooks/068a/kernel-metadata.json` |
| 比赛/远端工作流 | `docs/remote-workflow.md` |
| 已放弃方向 | `docs/research-archive.md` |
| 065/066/067/068 | `docs/plan/`、`docs/progress/` 对应文件 |

### 9.3 风险清单（提交前逐条核对）

- [ ] 面板 query 谱的 hash 不在 library；S2 的 query 结构不在 library。
- [ ] 分组用 `metric_identity`，不是 stored `inchikey14`。
- [ ] 外部库按整来源留出；同来源参考谱 cap 2 条。
- [ ] A/B 都是纯置换；`set(new)==set(old)`、25 个、无重复。
- [ ] A 固定 top-1；B 默认固定 top-1；C 的切换有白名单/精度门。
- [ ] 面板 B0 与 068a 逐字节一致。
- [ ] 面板 Δ 用结构整组 bootstrap 报 CI，不报单点。
- [ ] 折算公榜时用 `f1/f2/f3` 稀释，不用面板绝对值。
- [ ] 每个 069 内核只改一个信号；审计 `top1_changed`、`lists_changed`、运行时间、输出 sha256。
- [ ] 068a 的 `LAM=0.35` 底座未被新 cell 隐式改动（先算 `submission_pre_n1c` 的 sha256）。

### 9.4 许可与署名边界

- A/B/C 不引入新模型权重；只用 `train.parquet`、`test.parquet` 与 068a 已有公开资产。
- 外部谱库只使用 068a 已挂的 CC0 extlib 与 MassBank harmonized（丢弃 `is_casmi_subset`）；NIST 及其衍生资产不碰。
- 若后续做 DreaMS 嵌入检索（本设计不做），需按主办方确认的 DreaMS 许可与断网输入打包，单独评审。
- 当前 A 线含 sanpier 等许可 unknown 的资产，只冲奖牌/名次；最终奖金提交仍需 B 线许可干净版本，这一点不因 A/B/C 改变。
- 署名：A 的检索思路承接 066a（liuzewen123）与 Engine 2（prvsiyan）；B 承接 N-1c（analyticaobscura / Fable EK-18）；C 的特征框架承接 `ranking.py`（romanrozen / 公共 0.466 配方）。新增部分在进展文档中单独列出。

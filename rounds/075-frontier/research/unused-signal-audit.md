# 075-B 冻结管线「算过但没进最终排序」的逐候选信号审计

日期：2026-11-18。对象：0.492 配方 = `rounds/068-n1c-lam035/068a/enveda-068a-n1c-lam035.ipynb`（065a 底座 + N-1c `LAM=0.35`）。
只读审计：未改任何已有文件、未跑训练/推理、未推 Kaggle。

## 证据等级与行号约定

- **【已核实】**：能在解码后的源码、本地已下载产物（parquet/json/pkl）或远端日志里直接读到。
- **【已核实·运行】**：来自远端运行产物（`artifacts/kout/…`）。
- **【推断】**：由代码结构 + 已有分数反推，文中标 🔍。
- 行号约定：`F-cell N Lx` = 解码后 frozen notebook（`FROZEN_B64`，SHA-256 `d1e757c2…`）第 N 个 cell 内第 x 行；`N-cell N Lx` = 068a 自身第 N 个 cell 内第 x 行。F/N cell 序号均为 0-based，与 `nbdump/068a.py` 的 `##### CELL n` 一致。

## 0. 三个事实先钉住

1. **最终顺序只由极少数标量决定。** 25 名排序 = `consensus` 的 RRF（`1/(3+b)+w/(3+n)`，`w_top=0.25/w_mid=0.75`）→ 物理 overlay 的**阈值判定** → `_score_cand`（`1/(3+br)+0.75/(3+nr)+POP_W·pop/20+SPEC_W·spec+NP_W·np+MASS_PEN`）→ K14 去重/回填 → mass rank-1 → N-1c 网络 → extlib 升位。除 `_score_cand` 外，其余各层都只吃**名次**，不吃分数。
2. **绝大多数逐候选数值在生成候选的那一刻就被压缩成「名次」。** Engine 1 的 `scs`（逐候选 ranker 分）在 `F-cell 41 L1326` 被显式丢弃；Engine 2 的 blend 分只用于 `argsort`，子进程只回传 `smiles+keys`（`F-cell 32 L6`）；frozen 融合 `fuse2` 的 `sc` 只用于排序（`F-cell 45 L15-25`）。
3. **已有一批逐候选量其实落在磁盘上、但没进排序也未进分析。** `e004_candidate_inputs.json` 的 `pubchem[mid].pc_fz / promotion_evidence[].S,pop,raw_fz`（25 个/题）从未被当作特征用过；`frozen_candidate_evidence.parquet` 的 `ice/gl` 列同理（`ice` 在 0.492 配置里不参与主列表排序）。

---

## 1. 总表

「已用于排序」分三档：**是（权重项）** / **仅阈值/门控** / **否（只影响它自己的名次，值本身丢弃）**。
「能否区分同分异构」= 对同一分子式的两个候选，该量是否会不同（这是本任务唯一有用的性质）。

| # | 信号 | 定义位置 | 已用于排序 | 区分同分异构 | 导出成本 | 已被证伪 |
|---|---|---|---|---|---|---|
| 1 | Engine 1 逐候选 ranker 分 `scs`（LightGBM 4-booster 均值） | `F-cell 39 L29-35`；丢弃于 `F-cell 41 L1326` | 否（只保留顺序） | 能 | **E3**：1 行 frozen 补丁 + 重算 `FROZEN_SHA256` | 否（从未作为连续量测过） |
| 2 | Engine 1 逐候选特征矩阵 `X, F`（`FEATURES`+`names`，含库余弦/加合物/指纹/碎片） | `F-cell 39 L26`（`V.run` 返回），`F-cell 37 L26-31` | 否（进 ranker 后被丢掉） | 能 | **E4**：需改引擎包或冻结源里加钩子，成本最高 | 否（这些列从未被单独导出过） |
| 3 | Engine 1 逐候选分子式 `forms`（含 `\|gen` 标记） | `F-cell 39 L35, L47` | 仅阈值/分组（ICE/GL 分组、同式规则） | 否（同式相同） | **E3**：同 #1 一处补丁 | 否 |
| 4 | Engine 1 逐候选 pool id → `POOL_POP`（`log1p(SID)+log1p(PMID)`） | `F-cell 39 L7, L39-47`（`POOLPOP_MU=0.15`） | **是（权重项，base 列表层）** | 能 | E0（已生效） | 否 |
| 5 | `library_max`（分子级：max 库余弦） | `F-cell 39 L27`，导出 `N-cell 12 export_evidence` | **是（门控）**：`LIB_HIT_GATE=0.78`、PubChem 门 `LIB_TAU=0.9`、物理/质量规则 | **否（全列表同值）** | E0 | 否（但它是召回证据，不是排序证据） |
| 6 | Engine 1 逐候选库余弦（单结构级） | 引擎包内部（`casmi/library.py`，本地不可读） | 否（只经 `lib_max` 出场） | 能 | E4 | 部分：066a 的「连续余弦 mid-rank」0.488 中性 |
| 7 | Engine 2 逐候选融合分 `blend(pv, ours, 0.88)` | `F-cell 32 L6`（`eng_runner.py`） | 否（`argsort` 后只回传 smiles/keys） | 能 | **E2**：我们自己的 cell，无任何 pin | 否 |
| 8 | Engine 2 逐候选 ppm `(pool_mass−target)/target·1e6` | `casmi_engine.py L324-325`（`blk_mass`） | 否（只是 ranker 特征） | **否**（同式相同） | E2 | **是**：072c +0.00000；072d 47/48 同式 |
| 9 | Engine 2 逐候选碎片解释度 `F1/F2/F3/frag_pol`（含**加合物感知** `explain_score_adduct`） | `casmi_engine.py L401-404`；`pv.py L330-357` | 否（只是 ranker 特征） | 能 | E2（同上） | 否（与物理 overlay 不同源） |
| 10 | Engine 2 逐候选 analog Tanimoto / 库特征（`rank_features` 31 列） | `pv.py L210-255` | 否（特征） | 能 | E2 | 否 |
| 11 | PubChem 逐候选原始指纹分 `pc_fz` / `raw_fz` | `F-cell 34 RUNNER L80`；`probe_core2 probe_one` | 仅阈值（`rel = pc_fz[0] − best_pool_fz`）、ICE 分组重排 | 能 | **E0：已在 `e004_candidate_inputs.json`** | 否 |
| 12 | PubChem 逐候选 `S = z(f.z)+0.15·pop` 与 `pop` | 同上；`promotion_evidence[k]` | 仅顶层一个候选（升位门 `S>6 & pop≥5`） | 能 | E0（已导出） | 否 |
| 13 | `best_pool_fz` / `fz_top` / `target`（分子级） | `F-cell 34 RUNNER L70,L80` | 仅阈值 | 否 | E0 | 否 |
| 14 | ICEBERG 逐候选分 `ICE_SCORES` | `F-cell 41 L105`；导出 `ice` 列 | **否**（主列表 `BASE_ICE=False`；`CELL19_ICE=False`）；仅 PC 列表同式组重排 + 物理 overlay 的 ±1 票 | 能 | E0（已导出） | 部分：083a ICE×DreaMS 门 0.488 |
| 15 | GLACIER 逐候选分 `GL_SCORES` | `F-cell 41 L114-126`；`gl_of` L1298 | **是（同式组内重排，`GL_LAM=1.0`）**；物理 overlay 的 ±1 票 | 能 | E0（已导出） | 否（未单独消融过） |
| 16 | 双引擎融合 `fuse2` 的逐候选 RRF 分 `sc` | `F-cell 45 L15-25` | 仅排序（`sorted` 后丢分） | 能（但已被名次吸收） | E3 或 E1 | 否 |
| 17 | 物理 overlay 逐候选 `frag_new`（优化 cfg）、`frag_old`（基线 cfg）、逐谱 energy 矩阵 | `N-cell 14 _physical` + `_post` | **仅阈值**（0.095/0.085/0.065、votes 0.70/0.65），只算前 12 + ≤8 outsider | 能 | **E1**：改 `_post` 字符串 + 更新其 SHA 断言 | 否（这是作者称 +0.065 MRR 的信号） |
| 18 | 文档先验三分量 `pop / src_spectral_library / np`（按 InChIKey14） | `N-cell 21 L62-70`（`_score_cand`） | **是（权重项，2–25 名）** | 能 | E0/E1（可加导出） | 否（作者称 +0.0083，未独立复现） |
| 19 | K14 去重 / LOTUS 回填 / DreaMS 回填 | `N-cell 21 L200-260`；`F-cell 48 L43-56` | 仅空槽 | **否**（mass window + 身份） | E0 | 不适用（LOTUS/DreaMS 未挂载，结构性死代码） |
| 20 | 神经 `neural_cos / neural_fp / neural_score / neural_std` | `N-cell 12 neural_scores.py` | `neural_score` **是**；`cos/fp` 仅门控；`std` **否** | 能 | E0（parquet 已有） | RR8 家族见 072a/b/e |
| 21 | 双神经模型（public vs v80）**分歧** | `N-cell 15 L69-74`（0.5·z+0.5·z） | 否（被平均掉） | 能 | E1（1 行，保存两路分量） | 否 |
| 22 | RR8 102 维特征（49 谱特征 + 观察中性丢失 + 候选事实）与模型分 | `N-cell 23 L4`；`probe/rr8mod/rr8_features.py` | **否（`V8='MASS'` ⇒ RR8=False）** | 能 | E1（特征已能离线算） | **是**：072a/072b/072e |
| 23 | N-1c 逐候选网络分 `N[i]`、边 `(neighbor, cos)` | `N-cell 24 L44-61` | **是（权重项）**，但只改 2/400 列表 | 能 | E1 | 部分：072b 说无法在 120 分子面板判定 |
| 24 | APEX 外谱最佳余弦 `best_cos` / 命中峰数 `n_matched` | `N-cell 25 L150-209` | 仅阈值（`TAU=0.80`, `K=4`） | 能（但只给 1 个结构） | E1 | 否，但 0 覆盖（065a/066a/067a 均 0 hits） |
| 25 | `MASS_PPM` / `MASS_PEN` | `N-cell 20 L31-40`；`N-cell 23 L34-60` | **是（权重项 0–0.15，2–25 名）** | **否**（同式相同） | E0 | **是**：072c 网格 +0.00000 |
| 26 | 同位素包络一致性 | **不存在**（全文无 isotope 代码，只有 `N-cell 2` 的营销文案） | 否 | （若实现则能） | 需新写 | — |
| 27 | `n_spectra` / `n_adducts`（分子级） | `N-cell 12 neural_scores.py` | 否 | **否** | E0 | 否 |
| 28 | `rel`（PubChem 通道相对池的 margin）、槽位选择 | `F-cell 43 L71-72` | 仅阈值（`>600` 换激进槽位） | 分子级 | E0 | 否 |

> 成本分级：**E0** 无需改动（已在产物里）；**E1** 改我们自己的代码/字符串 + 更新本地 SHA 常量（不动 frozen）；**E2** 改 `N-cell 32` 的 `ENG_FILES`（无任何 pin）；**E3** 改 frozen cell + 更新 `FROZEN_B64`/`FROZEN_SHA256`（有先例，见 §3）；**E4** 需改外部引擎包（不可行/高成本）。

---

## 2. 逐条细节

### 2.1 候选生成：Engine 1（`F-cell 36/37/39`）

```
F-cell 39 L26:  C, X, F, names, info = V.run(spectra, target)
F-cell 39 L27:  lib_max = float(info.get('lib_max', 0.0))
F-cell 39 L29:  score = rank_score(np.concatenate([X, F], 1), list(FEATURES) + list(names))
F-cell 39 L30:  order = np.argsort(-score, kind='stable')
F-cell 39 L35:  ... scs.append(float(score[i])); ... forms.append(C.formula[i]); pids_.append(int(C.pid[i]))
```

- **`scs` 是逐候选分数，`X/F` 是逐候选特征，`forms` 是逐候选分子式**——三者在 `F-cell 41 L1326` 一起被丢掉：
  ```
  BASE[m] = (smis[:25], keys[:25], lib_max)
  ```
  之后所有层（含 `export_evidence.py`）看到的 `BASE` 只有 3 元组。**这是全管线最大的一处「算了就扔」。**
- `POOL_POP` 重排（`L39-47`）**是**唯一在 base 列表层生效的结构级先验：`z(score) + 0.15·(log1p(SID)+log1p(PMID))`，生成候选 `pid<0` 保持原位。它已经在排序里，但**没有单独消融过**（59 臂表里没有它）。
- `lib_max` 是**分子级**：`info['lib_max']` 是全部候选 × 全部库谱的最大值。证据：`072b/frozen_candidate_evidence.parquet` 里同一 `molecule_id` 的 25 行 `library_max` 完全相同（如 `m_005e53` 全为 1.0）。所以它**不能**用来在候选之间选。

### 2.2 候选生成：Engine 2（`F-cell 32`）

`ENG_FILES` 是 4 个内嵌源文件（`pv.py` 357 行、`pv_fp.py` 182 行、`casmi_engine.py` 511 行、`eng_runner.py`）。逐候选量：

| 量 | 位置 | 定义 |
|---|---|---|
| ppm | `casmi_engine.py L323-325` | `ppm = (POOL["mass"][cand] − target)/target·1e6`，再拼 `[ppm, |ppm|, |ppm+1.5|]` 三列 |
| 碎片解释度 | `L401-404` | `F1=explain_score(mode)`、`F2=explain_score_adduct(adduct)`、`F3=mean(F3)`、`frag_pol=max` |
| 加合物感知的碎片匹配 | `pv.py L330-341` | `CARRIERS[adduct]`（`[M+NH4]+` 允许 `NH4−E`，`[M+Na]+` 允许 Na…），`dh∈{-2..+2}`，`tol=0.01`，按 `sqrt(intensity)` 计权 |
| ≤2 键断裂枚举 | `pv.py L298-313` | 拆 1 或 2 根键后各连通块质量集合（`max_bonds=34`） |
| 库/类似物特征 | `pv.py L210-255` | `lv`（库余弦）、`ap`（analog Tanimoto^3 加权）、`best_tan`、`mean_tan`、与 model logits 的一致性等 31 列 |
| 融合分 | `eng_runner.py`（`N-cell 32 L6`） | `bl = blend_scores(S['pv'], S['ours'], 0.88)`；`order=np.argsort(-bl[mid])[:80]` |

**输出只有 `{mid: {smiles:[...], keys:[...]}}`**（`eng_runner.py` 末段），40 个候选、**没有任何分数**。所以：ppm/F1/F2/F3/库余弦全部只通过 ranker 影响「40 个是谁、什么顺序」，值本身不可追溯。
注：`frag_ad`（加合物感知）与 068a 物理 overlay 的 `frag_rescore`（也支持 carriers，但 cfg 不同：linear/0.005/w2=0.6）**不是同一实现**，两者的一致性本身是一个未用过的信号。

### 2.3 PubChem 通道（`F-cell 34`）

```
RUNNER L5-7 (docstring): {molecule_id: {pc, pc_fz, pc_keys, best_pool_fz, target}}
probe_one:  sc = zf + POP_LAM·pop ; order = argsort(-sc)
            promotion_evidence[k] = dict(smiles, raw_fz=fz[j], S=sc[j], pop=pop[j])
            diag['S']=out_sc[0]; diag['top_pop']=out_p[0]
F-cell 43 L71: rel = (fz_top or pc_fz[0]) − best_pool_fz    # 只为选槽位/升位
```

- **逐候选 `raw_fz / S / pop` 对全部 25 个提案都存在**，并且**已经导出**：`artifacts/kout/073a/e004_candidate_inputs.json` 的 `pubchem[mid]` 有 `pc`(25)、`pc_fz`(25)、`pc_keys`(25)、`promotion_evidence`(25)、`S`、`top_pop`、`best_pool_fz`、`target`。本次实测：12 个可见分子共 300 个 PC 候选，`promotion_evidence` 的 `pop` 范围 **0.693–6.885，中位 2.139**。
- 排序里只用了 `S/pop` 的**第 1 名**（`S>6 且 pop≥5`，或 `rel>150`）。第 2–25 名的 `S` 完全没被读。
- `ICE_PC=True`（`F-cell 26 L13`）时 `pc_fz` 还会进 `ice_fuse.rerank`（`F-cell 41 L1328-1334`），但只在**同分子式组内**，且 ICE 本身在主列表被关掉（见 2.4）。

### 2.4 ICEBERG / GLACIER（`F-cell 41`、`F-cell 45`）

- `F-cell 41 L9: BASE_ICE = False`，`F-cell 45 L2: CELL19_ICE = False`。
- 因此 `F-cell 41 L1316-1325` 的主列表 ICE 重排**整段不执行**；`F-cell 45 L40/L43` 传给 `rerank_multi` 的 ice 是 `{}`。
- ICE 的实际用途只剩两处：(a) `ICE_PC=True` → PC 列表同式组重排；(b) `N-cell 14 _post` 的 `simulator_vote`（`ice>=gl` ±1 票）。
- 代价是实打实的：`073a/candidate_audit.json` 记录 `ice: n_cands 814, n_scored 793, seconds 77.86`（11 分子）。按 400 题外推 ≈ 47 min GPU，主产物被丢弃。🔍
- GLACIER **是真用的**：`F-cell 45 L43` 的 `gl_fuse.rerank_multi(..., [ICE_LAM, GL_LAM]=[1.0,1.0])`，作用于融合后 40 个候选的**同分子式组**，再 `vs = fsm[:25]`。`gl` 列也导出，物理 overlay 再用一次 ±1 票。
- **未被用过的是**：`gl` 作为 2–25 名的**连续项**（`_score_cand` 里没有 gl）；以及 `gl` 在**跨分子式**上的用法（作者包内部是否分组，见 069 管线图 §L2 的读法）。

### 2.5 融合与「同式组内重排」（`F-cell 45`）

```
L15-25 fuse2: sc[k] = 1/(3+r_main) + 0.6/(3+r_eng2)  →  sorted → top 40
L27       PREFUSION_LISTS   = 融合前的 25 名（导出）
L28/L49   FUSION_OVERFLOW   = fsm[25:]（10–15 个候选，**只存 smiles，不存 sc**）
L50       vs = fsm[:25]
```
`fuse2` 的 `sc` 是逐候选的，返回 `fsm, fk, fsc`；但 `fsc` 只在 ICE/GL 重排里当输入，重排后即丢。`FUSION_OVERFLOW` 里的 15 个候选（本地实测 `len=15`）没有任何分数——它们只能靠 `N-cell 12 outsiders.py` 重新走神经打分。

### 2.6 物理 overlay（`N-cell 14`、`N-cell 16`）

`frag_rescore.score_molecule(spectra, candidates, cfg)` 返回**逐候选**解释度：

- `frag_new`：`DEFAULT_CFG`（linear 强度、`w2=0.6`、`tol=0.005`、`h_shifts −2..+3`、`max_breaks=2`）
- `frag_old`：`BASELINE_CFG`（sqrt、`w2=1.0`、`tol=0.01`、`−2..+2`）
- `energy`：按 2 张谱一组切块后的逐块得分矩阵（用于 `votes`）

但这些只喂给 `_post.decision()` 的**阈值**：`frag_new[j] ≥ .27`、`margin ≥ .095`、`oldmargin ≥ .028`、`votes ≥ .70`、`nmargin ≥ .23`、`neural_cos ≥ leader+0.003`、`neural_fp ≥ leader−0.012`；邻位交换 `≥.085/.015/.65`；尾部 `≥.28/.065/.02/.65`。**只有前 12 个候选 + ≤8 个 outsider 被评分**，且分数不出 `spectral_parallax_audit.csv`（那里只有 `top1_frag_margin` 一个标量）。这是「连续信号被压成布尔」的典型。

### 2.7 文档先验（`N-cell 21 L62-70`）

```python
def _score_cand(s):
    br, nr, k14 = cinfo[s]
    return (1.0/(3.0+br) + 0.75/(3.0+nr)
            + POP_W*(pop_map.get(k14,0.0)/20.0)
            + (SPEC_W if lib_hit else 0.0)*spec_map.get(k14,0.0)
            + NP_W*np_map.get(k14,0.0))
rest.sort(key=lambda _s: _score_cand(_s) + MASS_PEN.get(mid,{}).get(_s,0.0), reverse=True)
```
- `pop/spec/np` 按 **InChIKey14** 查 `pool_popularity.csv`（列：`inchikey14, pop, src_spectral_library, np_coconut, np_lotus, src_natural_product`）。**这是结构级、与质量无关**的量——唯一已经在排序里、且能区分同分异构的现成先验。
- 量级问题（本次实测）：`pop` 0.693–6.885 → `POP_W·pop/20` 仅 **0.0042–0.0413**；而 RRF 项 `1/(3+br)+0.75/(3+nr)` 在相邻名次间的差在前段约 0.06–0.25、后段 <0.01。⇒ **先验在列表前段几乎不可能改变顺序，只在后段（RRF 已很小）才有话语权。** 🔍 这意味着它的「+0.0083」若真实，来源应是 15–25 名。
- `lib_hit` 是**分子级**门（`LIB_MAX[mid] ≥ 0.78`），`spec` 才是逐候选。
- 缺失候选的 fallback 是 `1/(3+idx+1)`（纯名次），说明该层对 `cinfo` 之外的结构没有任何证据。

### 2.8 N-1c 网络（`N-cell 24`）

```
L44-47  两两 modified cosine（0.0101 Da，precursor shift），n≥6 且 s≥TAU=0.6 连边
L55-58  N[i] += w**4 * Σ_top5 p_k · Tanimoto(cand_i, neighbor_cand_k)
L59-61  o = argsort(-(z(-log rank) + LAM·z(N)))
```
逐候选 `N[i]` 与全部边 `(neighbor, cos)` 都是可用的结构级信号，**只有排序结果被保留**。远端实测（`072b` v2 日志、`072/progress.md` L249）：`with neighbors 114, edges 109, changed lists 2, top1 changed 0, 81s`。⇒ 这条信号在 400 题上只覆盖 28.5% 的分子，且只改 0.5% 的列表。

### 2.9 神经打分（`N-cell 12`、`N-cell 15`）

`neural_score.py` 逐候选产出：`neural_cos = mean_spec(Z·Mᵀ)`、`neural_fp = mean_spec(P·Fᵀ/fp_bits)`、`neural_score = mean_spec(cos/T + w·fp)`、`neural_std = std_spec(combined)`、`n_spectra`、`n_adducts`。

用途盘点：
- `neural_score`：consensus RRF（`w_mid=0.75`）+ `_score_cand` 的 `0.75/(3+nr)`。**是主排序项**。
- `neural_cos` / `neural_fp`：只在物理 overlay 当门槛（`+0.003` / `−0.012`）。**值本身不进排序**。
- `neural_std` / `n_adducts`：**全链路无人读**（只在 `ranking.py` 的 `BASE_FEATURES` 里出现，那是 072a 实验用的特征表，不在出货路径）。
- `N-cell 15 L69-74`：两个模型（public bundle 与 popwenman v80）各自打分后 `0.5·z(a)+0.5·z(b)`，`neural_cos/fp/std` 直接线性平均。⇒ **两模型的逐候选分歧被平均掉**，没有留下来。

### 2.10 RR8（`N-cell 23 L4`）

`_RR8_FEATURES` 定义 49 个逐候选特征（16 个谱解释度含 2 个紧容差 + `log_n_k1/k2` + 5 个候选事实 + 13 个中性丢失 × agree/clash = 26）与 102 维模型输入；`rr8mod/rr8_features.py` 里 `full_features = X + centre_within(X)`。

- 关键：`V8='MASS'` ⇒ `RR8=False`（`N-cell 5 L40-42`），**整段是死代码**（072b 结论，`072/progress.md` L231）。
- 即便打开，它在 `lib_max≥0.9` 的题上跳过；`artifacts/rr8x/features.pkl` 是 360 条记录、每条 `X` 25×53（49 逐候选特征 + 4 个跨谱一致性列，072a 的 B 臂），另有 `frozen`/`y`/`libmax`/`n_spec`。这是 072a 的离线特征表，不是出货路径。

### 2.11 APEX 外谱 overlay（`N-cell 25`）

`calc_cosine(tol=0.015 Da, sqrt 归一, 贪心一对一)` 返回 `(cos, n_matched)`；precursor 窗口 `10 ppm + 0.002 Da`；只保留每个 query 的 `best_match`（一个结构），`cos≥0.80 且 n≥4` 才升位。⇒ **逐候选余弦/命中峰数没有算**（只算「查询 vs 外库参考谱」的最佳值），且 `nm` 是死代码（072 加合物审计结论）。远端 3 次运行均 `0 hits / 0 promoted`（069 管线图 §2.4B）。

### 2.12 被 `sorted/argsort` 丢掉的中间量清单

| 位置 | 中间量 | 丢掉的方式 |
|---|---|---|
| `F-cell 39 L30-35` | `score`（逐候选） | 只存 `scs`，`F-cell 41 L1326` 再丢 |
| `F-cell 45 L24` | `sc[k]`（融合分） | `sorted` 后只留 keys |
| `F-cell 45 L49` | 溢出候选的分数 | `FUSION_OVERFLOW` 只存 smiles |
| `F-cell 43 L71` | `rel`（通道 margin） | 只用于槽位选择 |
| `F-cell 41 L1318-1334` | ICE/GL 重排的 z 分 | 只留 permuted list |
| `N-cell 21 L73` | `_score_cand(s)` 逐候选合成值 | `sort` 后丢 |
| `N-cell 24 L59` | `z(base)+LAM·z(N)` | `argsort` 后丢 |
| `N-cell 25 L194-209` | `best_cos`、`n_matched` | 只留 `match_smi` |
| `N-cell 20 L31-40` | `MASS_PPM` | 只用于阈值/惩罚，未导出 |

---

## 3. 最值得导出的 3 个信号，以及怎么在不破坏 SHA 锁的前提下导出

先明确「锁」有三层（`N-cell 13 L68-74`）：

```python
L71: assert sha256((CODE/name).read_bytes()) == MODULE_SHA256[name]      # 8 个 helper 模块
L74: assert sha256(FROZEN.read_bytes())      == FROZEN_SHA256            # 冻结 0.465 notebook
# 另有 cell 14 L5-6 的两条字符串锁：_physical / _post
# 另有 cell 23 L28-29 的两条字符串锁：_RR8_FEATURES / _RR8_OVERLAY
```
`FROZEN_SHA256` 的算法已核实：**它等于解码后 notebook 字节的 sha256**（本次两种算法都比对过：`sha256(base64decode(FROZEN_B64))` 命中 `d1e757c2…`；对 base64 文本求哈希不命中）。072d 已经用过这个手法：解冻 → 改 3 个 cell（`INSTRUMENT_COMP` 环境变量）→ 重新序列化 → 换 `FROZEN_B64` + `FROZEN_SHA256`

### 推荐 1：物理 overlay 的逐候选碎片分（`frag_new` / `frag_old` / `energy`）

- **为什么**：它已经在跑、只覆盖 12+8 个候选、且是目前唯一被作者报告过「同式异构体 +0.065 MRR」的物理信号；现在是阈值化的布尔量。把连续值导出后，可以做的不是「再调阈值」，而是**跨候选归一化**（同一题内 z-score / 与 leader 的 margin 分布 / 与 Engine 2 的 `frag_ad` 的相关性）。
- **改哪里**：`N-cell 14` 的 `_post` 字符串里 `decision()` 已经算出 `frag_new/frag_old/energy` 与 `pos`。在 `apply()` 的 `audits.append(meta)` 处补一个逐候选 CSV/parquet（`mid, smiles, frag_new, frag_old, votes, neural_score, neural_cos, neural_fp, is_leader`）。只改 `_post` 一处 + 更新 `N-cell 14 L6` 的 `assert sha256(_post)== 'cd1594b3…'`。**不动 frozen，不动 MODULE_SHA256。**
- **额外成本**：0（这些值已算），文件约 400×20 行。

### 推荐 2：Engine 1 的逐候选 ranker 分 `scs` 与分子式 `forms`

- **为什么**：这是全管线唯一「已在内存、被显式丢弃」的 60 候选分数。有了它才能回答「榜首领先第 2 名多少」「同式组内的分数分布是否重叠」——这是判断「名次是否可信」的唯一现成量；072e 的失败可读作「拿不到可信度就用固定 margin」🔍。
- **改哪里**：`F-cell 41 L1326`
  ```python
  BASE[m] = (smis[:25], keys[:25], lib_max)            # 现状
  BASE[m] = (smis[:25], keys[:25], lib_max, scs[:25], forms[:25])   # 建议
  ```
  然后 `N-cell 12 export_evidence.py` 里 `base=scope['BASE'].get(mid,([],[],0.))` 已按索引取 `base[0..2]`，向后兼容；再加 `base[3]/base[4]` 两列即可。
- **pin 更新**：解冻 `FROZEN_B64` → 改这一个 cell → `json.dumps` 重新序列化 → `FROZEN_SHA256 = sha256(new_bytes)` → base64 回填。**先例：072d 用同样流程改了 3 个 cell 并成功运行**（`rounds/072-scale/072d/…ipynb` 的 `FROZEN_SHA256=1e8db790…`）。风险：`BASE` 变 5 元组后，`F-cell 50 L21` 的 `base, eng, pc = scope['BASE'], …` 与 `export_evidence` 的 `base[2]` 都仍然成立；但 `N-cell 21` 不读 `BASE`，无影响。**对齐风险（本项唯一真实风险）**：`F-cell 41 L1313` 是 5 元组解包 `smis, keys, lib_max, scs, forms = BASE[m]`，而 `L1320-1323` 在 ICE 分支里会 `smis, keys = s2, k2`（按 `order` 重排）。当前 `BASE_ICE=False` 使该分支不执行，所以直接保留 `scs` 不会错位；但**一旦以后打开 `BASE_ICE`，必须同时把 `scs/forms` 按 `order` 重排**，否则导出的是错位分数。
- 成本：1 行 patch + 1 次 smoke 审计（可见 12 分子）。

### 推荐 3：Engine 2 的逐候选分数（`blend` / `pv` / `ours` / `ppm` / `F1..F3`）

- **为什么**：它覆盖**最终 25 名之外**的 40 个候选（含被 rank-2 规则硬插的那一个），并且 `frag_ad` 是**加合物感知**的独立实现，可与推荐 1 构成「双碎片评分器一致性」——不训任何模型就能得到的 isomer 判别信号。ppm 列虽然对同式无效，但可以用来把「不同式」的候选排除在一致性统计之外。
- **改哪里**：`N-cell 32 L6` 的 `eng_runner.py` 字符串里，`res[str(mid)] = dict(smiles=smis, keys=keys)` 改成附带 `scores=[float(bl[mid][c]) for c in order_kept]`、`pv/ours` 分量、以及 `casmi_engine` 里 `r['frag']/r['frag_ad']/r['frag_mean']/r['frag_pol']/ppm`。`ENG_FILES` **没有任何 SHA 断言**（本次核对：cell 32 内无 `sha256` 字符串），所以这是**零 pin 成本**的改动。
- 成本：改 1 个字符串 + 输出体积（400×40×8 float ≈ 0.5 MB）。

### 零成本的「第 0 步」

在动上面三项之前，先把**已经导出**的东西做成离线表：
- 本地已有：`e004_candidate_inputs.json` 的 `pc_fz / S / pop / raw_fz`（逐候选 25 个）、`frozen_candidate_evidence.parquet` 的 `ice / gl / library_max`（`artifacts/kout/073a`、`072b`）。
- 本地**没有**：`neural_candidate_features.parquet`（含 `neural_cos/fp/score/std`）。它只在 Kaggle `/kaggle/working` 里；需要一次 `kaggle kernels output` 下载（几 MB，不耗 GPU、不耗提交），或下次运行顺手带走。

在 072d 的 150 分子床上（`artifacts/kout/072d_v3/per_query_ranks.json` 有 150×25 的候选顺序）直接测「这些量对 47 个同式错误是否有 AUC」——若 AUC≈0.5，就不必为它们改代码；若有信号，再决定导出哪一路。**这一步 0 GPU、0 提交。**

> 注意：072d 是「留出谱」，类似物谱与结构目录仍在，MRR 0.8286 高于隐藏集；按 075 计划，最终判据要落在「骨架整组留出」（075b）上。

---

## 4. 明确不能区分同分异构体的信号（不要再往这里投预算）

以下量对**同一分子式的两个候选**取值相同（或只差加合物/质子化态），在 47/48 同式错误的场景里信息量为 0：

1. **`MASS_PPM` / `MASS_PEN`**（`N-cell 20 L31-40`）：`ppm=(ExactMolWt(s)−nm)/nm·1e6`，同式 ⇒ 完全相同。072c 的 500 组网格得到 `+0.00000` 是这条的机制性证明；072d 的 `47/48` 是同一事实的另一面。
2. **中性质量、加合物假设、±10/30 ppm 窗口、`_ADD` 七类映射**（`N-cell 20 L11`）：全部是分子级常量。072 的加合物审计：`_ADD` 只覆盖 **84.72%** 的 train 行（其余静默失去 mass prior），而 `_apex_extlib_overlay` 自算的 `nm` 对二聚体/甲酸/氯加合物 0% 正确（但 `nm` 是死代码）。两者都**不影响同式排序**。
3. **`library_max`**（`F-cell 39 L27`）：分子级 max，25 行同值（072b parquet 实测）。它只能回答「这题有没有库命中」，不能回答「哪个异构体」。
4. **`best_pool_fz` / `fz_top` / `target` / `rel`**（`F-cell 34/43`）：分子级参考量。
5. **LOTUS 回填 / DreaMS 回填**（`N-cell 21`、`F-cell 48 L43-56`）：mass window + 空槽，且都是死代码（未挂数据）。
6. **K14 去重**：只处理「同一骨架的立体/互变异构体重复」，对**不同骨架的同式异构体**无效；`dedup_swaps=0`（074a 日志）。
7. **`n_spectra` / `n_adducts`**：分子级。
8. **同位素包络一致性**：**管线里根本没有实现**。全文 `isotope` 只出现在 `N-cell 2` 的一句营销文案（"ambiguous precursor isotopic envelopes"）。要做得从零写（理论同位素分布 vs 观测 M+1/M+2），成本是「新信号」而不是「导出」。
9. **`Engine 2` 的 ppm 三列**（`blk_mass`）：同式相同；它只在「跨分子式」时有用，而 072d 证明跨式错误只有 1/48。

**结论**：任何只依赖分子式/精确质量的量（mass prior、加合物推理、同位素、同式池扩写、库命中门）在这一层**没有可回收空间**；能动的只有「结构级」与「谱-结构级」的量：物理碎片解释度、Engine 2 的加合物感知碎片分、文档先验三分量、ICE/GL 的连续分、N-1c 的邻居分、以及两个神经模型的分歧。

---

## 5. 与既有负结果的对账（避免重复劳动）

| 方向 | 证据 | 对本审计的含义 |
|---|---|---|
| RR8 学习重排（含 53/102 维特征） | 072a：freeze off `−0.0071`；重训 B `−0.0485`；072b：`V8='MASS'` 下 RR8 恒 off（死代码）；072e：`V8='BOTH'` + margin 1.0 → 公榜 **0.487（−0.005）** | #22 不必再导出/重训；`features.pkl` 只作历史 |
| mass prior / ppm | 072c：500 组网格 `+0.00000`；072d：`47/48` 同式 | #8/#25 关闭 |
| 加候选源 | 072d：`recall@25 = 150/150` | 扩池/扩通道不再是排序问题 |
| 库连续余弦 mid-rank | 066a：公榜 0.488（中性） | #6 的低成本变体已试过 |
| 自研 model3 mid-rank | 067a：公榜 0.488（中性） | 神经分数的「再加工」空间很小 |
| ICE 外部对齐门 | 083a：0.488（−0.004） | #14 的门控式用法已失败 |
| APEX 外谱升位 | 065a/066a/067a：0 hits / 0 promoted | #24 无覆盖，不算「证伪」但无收益 |

**没有被上述任何一次实验覆盖的**（因此值得先做零成本 AUC 体检）：#1（Engine 1 分数 gap）、#9（Engine 2 加合物感知碎片度）、#17（物理 overlay 连续分）、#18（文档先验三分量的边际）、#21（双神经模型分歧）、#23（N-1c 的邻居分本身）。

---

## 6. 本次核对记录（只读）

- 解码 `FROZEN_B64`（`nbdump/068a.py` 的 `N-cell 12`）→ 59 个 cell；`sha256(decoded)` = `d1e757c2b71b46e6…`，与 `FROZEN_SHA256` 一致。
- 读 068a 全部 32 个 cell 的源码；`MODULES` 8 个字符串、`ENG_FILES` 4 个字符串逐行展开。
- 本地产物：`artifacts/kout/073a/{candidate_audit.json,e004_candidate_inputs.json,experiment_audit.json}`、`artifacts/kout/072b/frozen_candidate_evidence.parquet`（688×13）、`artifacts/kout/072d_v3/{per_query_ranks.json,instrument_report_corrected.json,rank2_formula.json}`、`artifacts/rr8x/features.pkl`（360 条记录，`X` 25×53）。
- 072d 的 frozen 补丁 diff：仅 `F-cell 4/26/28` 三处 `INSTRUMENT_COMP`，`FROZEN_SHA256` 换为 `1e8db7906ecc3328…`；验证了「解冻→改→重算 pin」的可行路径。
- 未运行任何训练/推理；未下载任何新文件；未修改任何已有文件。

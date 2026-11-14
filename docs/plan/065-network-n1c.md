# 065 计划：在 063a（0.487）底座上复现公开 N-1c 网络重排（A 线）

## 背景
- 当前自有最佳仍是 063a/063b 公榜 **0.487**（ref 58233824 / 58233993）。A 轨底座为 063a：062a 代码 + popwenman scorer v80、不挂 LOTUS、`V8='MASS'`、`COMMIT_MODE='smoke'`、guhongbin docker。
- 2026-11-14 06:05 北京公榜快照（CSV 时间戳 2026-11-13T22:05:46Z）：5248 队，团队 ccjjllt 第 **559** 名；恰好 0.487 的平台为第 178–825 名共 648 队；**≥0.488 共 177 队**。相对约 6 小时前（2026-11-13T16:00:03Z，5228 队、第 536 名、≥0.488 共 112 队），≥0.488 新增 65 队，多数从 0.487 升到 0.488。
- 其中作者本人已核实跳出平台：alex chilton 0.487→**0.488**（最后提交 2026-11-13 16:17:51 UTC），Ertuğrul Özer（ozertuu）0.487→**0.488**（2026-11-13 12:05:02 UTC）。

## 已核实的公开 0.488 配方
- 公开 notebook：[alexchilton/casmi26-lb-0-488-peaks-to-structures-network](https://www.kaggle.com/code/alexchilton/casmi26-lb-0-488-peaks-to-structures-network)（2026-11-13 22:01:46 UTC 跑完，GPU T4，与 063a 相同的 10 个 dataset + `popwenman/casmi26-scorer-train-v80` + 同一 docker）。
- 作者说明：该版本是在其上一版 0.487 上，于外部库 overlay **之前**多跑一格 **N-1c**（称来自 [analyticaobscura/casmi-2026](https://www.kaggle.com/code/analyticaobscura/casmi-2026)）。作者账户提交 **58314720**（2026-11-13）公榜 **0.488**，相对其 0.487 为 +0.001。
- 与本地 063a 逐格核对：除页首说明、配置格 `COMMIT_MODE`（公开为 `'fast'`，063a 为 `'smoke'`）以及新增的 N-1c 代码格外，其余代码格哈希一致。因此 063a + N-1c 即该公开配方，只把 commit 审计留在 smoke。
- N-1c 行为（按公开代码，异常则 submission 原样保留）：
  - 测试谱用 modified cosine 相连：相似度 ≥ `TAU` 0.6 且匹配峰数 ≥ 6；
  - 每个分子 top-25 候选叠加邻域 top-5 候选的 `w^4 × Tanimoto`；
  - 新次序为 `z(-log rank) + LAM 0.25 × z(neighbour score)`。
- 不要把 ozertuu 标题「0.500+」且 `V8='BOTH'` 的 notebook 当作本轮配方；作者现已到 0.488，但与已核实的 N-1c 路径不是同一件事。064a 的 `LIB_HIT_GATE` 0.78→0.75 仍是未提交的单旋钮猜测，本轮让路给已核实的 +0.001。

## 实验
1. **065a**（已在 `notebooks/065a` 备好，2026-11-14 08:17 北京已推 **v1**，RUNNING）：063a 底座只插入上述 N-1c 代码格（外部库 overlay 之前），其余常量不动（`LIB_HIT_GATE=0.78`、`V8='MASS'`、`COMMIT_MODE='smoke'`）。内核 slug：`ccjjllt/enveda-065a-network-n1c`。
2. 等 2026-11-14 08:00 北京每周 GPU 配额刷新后再推；完整隐藏重跑约 6–9 小时。刷新前不推。064a 延后到 065a 出分之后。
3. 审计/提交沿用 `scripts/063/watch.sh`（单文件拉取、退避、smoke 门槛、通过则 `-k -v`）。

## 许可
A 线（公开 notebook 差异移植，sanpier 等模型许可 unknown），仅冲奖牌/名次。

## 署名
N-1c / 公开 0.488：analyticaobscura、alex chilton。底座沿用 063a 已列出处（guhongbin、popwenman、ozertuu、Takumu Hata、samartalwar、heliosli、romanrozen、zz3t3rn41、sanpier、Dmitrii Gluzdov 等）。

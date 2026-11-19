# 075 进展

更新日期：2026-11-19（北京时间）。

## 已完成与实测结果

### 1. 前沿与管线调研（2026-11-18）
- 结合 `071` 深度审计，收敛为两份精简文档：[模型与资产清单](research/models-and-assets.md)、[管线信号与验证口径](research/pipeline-signals-and-eval.md)。
- 公榜复核（2026-11-19 10:42 北京）：第 1 名 `0.550`（Shehab Alshehabi）、第 20 名金牌线 `0.506`、第 30 名 `0.502`；前 30/40 名无任何公开 notebook。公开可复现前沿仍为 `0.492–0.494`（我方 `068a` 稳定 **0.492**）。

### 2. 本地制备核查（2026-11-18）
- **RDKit Murcko 兼容问题**：本机 `rdkit 2026.09.2` 调用 `MurckoScaffold.GetScaffoldForMol` / `MurckoScaffoldSmiles` 6/6 分子均报 `RuntimeError: bad any cast`。化学型/骨架分组必须在 Kaggle 远端（`RDKit 2026.03.3`）计算。
- **结构聚类稀疏度**：对 20,167 个 ≥2 谱结构抽样 5,000（Morgan2 2048-bit，Tanimoto ≥0.65 leader 聚类）得 **4,921 簇**（单结构 4,844、2–4 簇 77、≥5 簇 0，最大簇 3），结构库整体高度分散。

### 3. NX 同分异构判别器验证 → **公榜崩溃，已终止**
- `075c`（`jiyujia/enveda-075c-nx-instrument` v2 `COMPLETE`）：在 `072d` 150 分子面板上用 `ahmedberatozer/casmi26-nx-model` 做同式组重排，MRR `0.8286 → 0.9394`（`fold0 w=1.0`）。因 150 个结构均来自 `train.parquet`（`fold0` 仅留出 1 折），存在训练泄漏，只证明集成路径可用。
- **`075d`（`jiyujia/enveda-075d-nx-rerank` v3，ref `58519267`，2026-11-18 17:20 UTC 提交）**：将 `NX fold0 w=1.0` + `NX_LIB_GATE=0.9` 接入 `068a`（允许同式组内换 rank-1，commit smoke `active 0` 符合预期）。
  - **2026-11-19 读回：`COMPLETE`，public = `0.381`（相对 068a 0.492 暴跌 `−0.111`）**。
  - 结论：NX 在未见结构上替换 rank-1 严重误排（与 `075d` 原笔记历史注记 `e6 0.472→0.414` 同类崩塌）。**NX 路线正式关闭**，相关本地产物与脚本已归档清理。

## 资源与下一步

- **配额（2026-11-19 MCP 核实）**：当前账号周 GPU 已用 `121,537s / 108,000s`（0h 剩余），刷新时间 **`2026-11-21T00:00:00Z`**（北京 11-21 08:00）；全部历史提交均已 `COMPLETE`，无在途任务。
- **下一步**：GPU 刷新后若继续推进，按 [plan.md](plan.md) 先建 `075b` 化学型整组留出面板与零成本信号审计，再单臂评估 `MIST MSG` 候选打分，不恢复已关闭方向。

## 2026-11-19 晚间 深度研究增量

本次读回最近10项提交，075d=0.381、074a=0.492、072e=0.487、083a=0.488、082d/f3/g=0.492均COMPLETE；配额仍为已用121536.588s/允许108000s、reserved0，刷新11-21 08:00北京。本次未启动GPU、训练、推理或新提交。

### 新证据和判断更新

1. **重建旧面板的选样范围**：仅读train第0个rowgroup的身份/来源/adduct等标量列，未读取峰数组。072d150身份681谱；元数据模拟075c每极性前8谱得到839条原始选择、59身份改变、额外158谱（不等于清峰后有效输入）。原始857条匹配谱中63身份同极性多adduct。072d的244/681谱为二聚体，150/150不能外推池外召回已饱和。未由这些风险认定NX掉分原因。
2. **多CE可用接口**：可见test400分子中279在同adduct有≥2不同单值CE，915谱单值、298多值；只作接口覆盖，不作泛化证明，多值谱不拆观测。
3. **新模型FLARE**：作者HF发布83.2MB checkpoint、MSG训练配置、局部peak↔atom late interaction，README声明MIT。预测公式缺失时回填标签公式、负离子预处理疑点和checkpoint训练manifest未核均明确记录；首轮正离子，真实公式仅作单独诊断。
4. **0.495公开配方新证据**：Taeyang平台公开submission58449454=0.495；具体notebook关联依作者说明，Alex/Arsen复现ref是公开自报。Taeyang v2与Alex v3的19 code cells相同，Zohrab复制文案不另计独立复现；FR三项训练派生输入生成链尚未定位。
5. **前沿排除边界**：MARASON原MSG流程使用NIST模型补CE，不是现成干净MSG资产；AIMe/DeepMS2Reasoner由NIST训练、MSG仅评估。ICEBERG2.1加速不自动等于新信息，需区别msg_all与msg_simulation。

### 产物和人工核对

- 本次研究并入当前075：新增[框架决策](research/next-frameworks.md)，更新计划、两份专题、README/轮次索引及归档中的过度归因；不恢复已清理方向。
- 机器证据：[旧面板和CE复算](../../data/references/075-panel-recheck-2026-11-19.json)、[公开前沿](../../data/references/075-public-frontier-2026-11-19.json)、[作者模型资产](../../data/references/075-model-frontier-2026-11-19.json)。原始小源码哈希/版本在JSON，未下载模型大权重。
- 工具与环境：本机Python3/pyarrow读取元数据；`git show pre-cleanup-2026-11-19:...`只读历史源码；Kaggle MCP公开数据/代码/提交/配额；GitHub、作者HF与论文主源。手动使用validate-data的比较口径核查，未创建测试文件/CI。
- 两个他人复现ref的私有提交详情被接口权限拒绝，已停止该路径；仅使用公共团队提交接口确认Taeyang0.495，不将自报视作平台逐ref确认。
- 交叉审阅修正了raw谱数、N1c并未冻结leader、union包含final25、pilot不能证明大幅收益等限定。已核算0.028净提升的条件尺度：无伤害时rank2→1需5.6%分子，rank5→1需3.5%；这不是本赛预期收益。
- 最后手动执行`git diff --check`、本轮Markdown本地链接存在性检查和3份JSON解析，均无异常；校核上述MRR算术。没有运行模型或把这些文档检查写成测试体系。

### 下一步

按更新计划先固定等输入的075b与训练曝光清单，再单臂测FLARE/MIST及CE条件前向评分，量候选oracle空间与实际逐分子ΔRR。原“零成本”仅指导出改动轻，运行底座仍消耗GPU。研究报告完成、新方案效果未测，当前最佳仍是068a的0.492。

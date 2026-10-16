# 011 — 固定MassBank扩库：集成与提交记录

日期：2026-10-14。对应[011计划](../plan/011-massbank-expansion-submission.md)。用户已明确授权由助手执行运行和一次验证性submission。保留v9r2/public0.323；不变更其代码或已有最终选择。

## 当前实际状态

- 固定来源离线资产已生成并上传到私有Kaggle Dataset：`ccjjllt/casmi26-massbank539-expansion`，v1，状态READY。
- v10本地notebook已生成并完成550例原生集成核查；GPU运行被Kaggle配额拒绝，CPU两次兼容性小样本完整成功但时长估算超限。**尚未提交比赛，无新public分数。** 当前状态`READY_LOCALLY_GPU_QUOTA_BLOCKED`。
- Kaggle配额实查GPU已用108618.608秒/总108000秒，reserved0，刷新2026-10-17 00:00 UTC。TPU剩72000秒与本次CUDA管线不同，不能直接替代。准备保留原GPU版，并在GPU被拒绝时做CPU兼容性/时长小样本，不能用部分评分结果冒充完整提交。

## 固定539中剔除1个无有效身份的条目

生产逐条核对发现原539包含1个`canonical_key=''`的通配/Markush组，原索引位置35；代表SMILES含`*`和R1/R2变量，不能作为确定化学结构。实际资产保留538个有效结构，原顺序不变；该条在既有NP/G1/R3窗口均未命中，剔除不改变010报告指标。原539→538映射和排除理由完整保存，不依据query真值筛选。

另有新增canonical `GIGFIWJRTMBSRP`与base raw key同字符串，但base规范化后为`FETOMZZQYPFKRV`。新索引统一使用`MB539_<canonical>`，与化学身份分离，避免碰撞；原notebook仍从SMILES重新生成canonical身份。

资产在`artifacts/011-expansion-asset/`，不进入Git。生成入口[scripts/build_011_expansion_asset.py](../../scripts/build_011_expansion_asset.py)：

- keys Unicode[538]，masses float64[538]，FP uint8[538,867]，同6930选位与big位序、6尾位填0；SMILES和canonical同序。
- `mb539_manifest.json` SHA256 `cf95c455dbc1ebe1ab91b3a9e6050f7279ec9de2333498532cb7db92a205d3b4`，每个输入文件记录bytes/SHA。
- 538结构对应985条公开来源记录；结构许可CC BY479、CC BY-SA49、CC0 10。集合保守标CC-BY-SA-4.0，逐记录原许可和出处不变。原MassBank ZIP临时副本已清理，依据008保存的唯一许可证集合、009完整accession/贡献者证据恢复，未捏造缺失的作者/原记录哈希。
- 包约1.89MB，只含公开结构及其衍生指纹、来源信息；无比赛谱或模型权重。`kaggle datasets create -p artifacts/011-expansion-asset --keep-tabular`实际成功，默认private，未使用`--public`。

## v10实现与保留的算法

构建入口[scripts/build_011_submission.py](../../scripts/build_011_submission.py)，主文件[enveda-submission-v10.ipynb](../../notebooks/submission-v10/enveda-submission-v10.ipynb)。原712199候选追加538，合计712737。`run_groups`由原train/COCO两分支改为train/COCO/新增三分支，COCO继续使用原cidx，新候选独立读取packed FP。先对完整质量窗评分，再原canonical-max/top100，不尾接已截断列表。

GPU主版本的FP模型、质量换算、ICE cell与v9逐字一致。原代表谱、10ppm、权重、fused_merge、ddof1、缺失续后与Top25均保留。另补原2000分子之外的动态尾部分块及实际test ID完整性校验，避免固定长度假设。输入使用唯一mb539前缀，避免原find('fp.npy')抓错文件。

既有5个dataset版本固定为FP1/COCO2/train-pool1/ICE1/cp313-wheelhouse1，新资产v1；保留v9镜像digest `2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f`，互联网关闭，notebook私有。

CPU兼容性入口另置`artifacts/011-cpu-probe/`：同模型和评分，只改device为CPU/4线程、按metadata选2个具有ICE支持谱的分子，输出`cpu_probe.csv`，**不得提交该小样本入口**。主GPU版本未因此改变。

本轮不创建测试文件、框架或CI；手动编译各cell、核对数据/代码与实际运行日志，运行成功、提交成功和得分分别记录。

## 远端实际执行

- GPU主版本第一次CLI push被Kaggle明确拒绝：`Maximum weekly GPU quota of 30.00 hours reached.` 未取得可运行的GPU版本，不能称运行中或提交成功；这是Kaggle配额限制，非自动审批拒绝。
- 将title与slug对齐后，独立CPU小样本入口push成功：`ccjjllt/enveda-v10-cpu-probe` **v1**，关闭互联网、CPU、最多900s，仅2个metadata支持分子，输出`cpu_probe.csv`。该版本禁止作为比赛提交。
- CPU v1完整成功：2分子/149候选、447个预测任务，status=ok、n_pred_fail=0、149条fused_merge有效、两行各25个候选；RDKit2026.03.6。预测243.34s、整段253.6s、1.84任务/s，notebook日志约322s结束。按旧v9占位运行72333任务仅作量级估算需10.92h，超过比赛9h，不能直接按当前配置全量提交。
- 追加**同样2分子**的CPU v2，唯一执行配置变化为torch线程4→1，模型与评分保持相同；已push成功，最多900s。目的是检查CPU线程开销，不调模型/权重/评价规则。原v1输出保留在`artifacts/011-cpu-probe-output/`，v2另存目录，均不得提交。
- CPU v2完整成功：447/447预测，预测487.09s、整段496.4s、0.92任务/s，无预测失败；两行最终CSV与v1逐字节相同。按同72333任务估算21.84h，单线程没有加速。产物`artifacts/011-cpu-probe-output-v2/`。CPU两次使用Python3.13.15、torch2.11.0+cu128（cuda不可用）、RDKit2026.03.6及同离线镜像。
- 量级估算仅来自2分子，不是实跑完整CPU计时，也不声称所有可能CPU优化都不行；但已试配置无法支持按9小时上限提交，因此没有启动预计超限的全量CPU任务，没有减候选/砍ICE/提交截断结果。两个CPU作业均COMPLETE，无本轮遗留运行作业。

## 阻塞与恢复步骤

阻塞来自Kaggle实际GPU周配额拒绝，不是缺用户授权或自动审批拦截。用户已授权运行/提交，不需例行再次确认；当前未创建未来自动任务，不能声称已安排配额恢复后自动执行。

GPU刷新时间2026-10-17 00:00 UTC，即北京时间**10月17日08:00**。恢复时复查额度，现有数据集v1可直接复用；按以下步骤续接：

```bash
kaggle kernels push -p notebooks/submission-v10 --accelerator NvidiaTeslaT4
# 记录实际push返回版本号，等该版本COMPLETE后人工核对输出与ICE完整状态。
kaggle competitions submit enveda-CASMI26-molecule-id-mass-spectra \
  -k ccjjllt/enveda-submission-v10 -v ACTUAL_COMPLETED_VERSION \
  -f submission.csv -m "v10 fixed MassBank expansion, 538 valid new structures"
```

提交前须按真实test核对全部ID、原FP异常计数、ICE status=ok、文件格式和实际耗时；再查看隐藏重跑和publicScore，不用占位测试命中率声称有效。v9r2保留，不改变原最终选择。完整恢复状态在`artifacts/011-execution/run-state.json`，精简[执行记录](../../data/references/submission-execution-2026-10-14.json)。

## 实际集成路径人工核对

运行[scripts/check_011_integration.py](../../scripts/check_011_integration.py)，从实际v9/v10 notebook AST提取`run_groups`，只把执行设备换成本地CPU、写出路径改为`artifacts/011-check/`；复用550个代表谱logits（NP250/G1研究250/R3-50），随后执行原生RDKit canonical-max/top100。无额外神经前向、无新保留集评价、无参数选择。

- shipped候选对61826→61865，canonical对61390→61429，新增39对；无exception或canonical fallback。空窗5→4，扩库救回1窗，4个原未知质量窗口仍空。
- 原候选顺序550/550不变。仅R3 ROTFFYYCARWRHQ一条FP分因同组padding宽度变动相差1个float32 ULP（6.1035e-05），排序不变，不声称所有浮点分数逐bit相同。
- G1完整top100与原扩池缓存250/250一致，R3完整top100与010重新计算的float64参考50/50一致。
- NP一例WYJAPUKIYAZSEM第84/85名因float64两分在float32舍入为同值而按canonical排序互换；原v9和v10都发生，真值rank21、Top25与Top100集合不变，不是扩库回归。
- GPU主版本model cell4、mass cell5、ICE cell32逐字与v9一致；改动仅候选追加/指纹读取、动态尾部分块、输出ID完整性及说明。产物`artifacts/011-check/{integration,raw_score_comparison,r3_full_reference}.json`和日志保留。

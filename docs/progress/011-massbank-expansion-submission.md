# 011 — 固定MassBank扩库：集成与提交记录

日期：2026-10-14；执行续接：2026-10-17。对应[011计划](../plan/011-massbank-expansion-submission.md)。用户已明确授权由助手执行运行和一次验证性submission。保留v9r2/public0.323；不变更其代码或已有最终选择。

## 当前实际状态

- 固定来源离线资产已生成并上传到私有Kaggle Dataset：`ccjjllt/casmi26-massbank539-expansion`，v1，状态READY。
- 10月17日用户提示配额恢复后实查GPU used0/reserved0/allowed108000秒，下一次刷新10月24日00:00 UTC；私有资产v1仍READY。v9r2/public0.323仍为当前最好已评分结果。
- 主GPU入口代码SHA与10月14日已核查版本一致；**v10版本1已COMPLETE，完整输出核查通过，并于北京时间10月17日11:43提交比赛，ref57293821，scriptVersionId359192425。12:51复查提交已COMPLETE/public0.323，与v9r2显示同分。** T4、private、internet关闭、原镜像未变。当前状态`SUBMISSION_COMPLETE_PUBLIC_0_323`。精简证据见[10月17日执行记录](../../data/references/submission-execution-2026-10-17.json)。
- 10月14日的GPU拒绝及CPU两次小样本结果保留为历史记录。CPU均完整成功，但预计全量超时；恢复后直接复用已核对的GPU主版本，没有重复创建数据集。

## 固定539中剔除1个无有效身份的条目

生产逐条核对发现原539包含1个`canonical_key=''`的通配/Markush组，原索引位置35；代表SMILES含`*`和R1/R2变量，不能作为确定化学结构。实际资产保留538个有效结构，原顺序不变；该条在既有NP/G1/R3窗口均未命中，剔除不改变010报告指标。原539→538映射和排除理由完整保存，不依据query真值筛选。

另有新增canonical `GIGFIWJRTMBSRP`与base raw key同字符串，但base规范化后为`FETOMZZQYPFKRV`。新索引统一使用`MB539_<canonical>`，与化学身份分离，避免碰撞；原notebook仍从SMILES重新生成canonical身份。

资产在`artifacts/runs/011-expansion-asset/`，不进入Git。生成入口[scripts/011/build_011_expansion_asset.py](../../scripts/011/build_011_expansion_asset.py)：

- keys Unicode[538]，masses float64[538]，FP uint8[538,867]，同6930选位与big位序、6尾位填0；SMILES和canonical同序。
- `mb539_manifest.json` SHA256 `cf95c455dbc1ebe1ab91b3a9e6050f7279ec9de2333498532cb7db92a205d3b4`，每个输入文件记录bytes/SHA。
- 538结构对应985条公开来源记录；结构许可CC BY479、CC BY-SA49、CC0 10。集合保守标CC-BY-SA-4.0，逐记录原许可和出处不变。原MassBank ZIP临时副本已清理，依据008保存的唯一许可证集合、009完整accession/贡献者证据恢复，未捏造缺失的作者/原记录哈希。
- 包约1.89MB，只含公开结构及其衍生指纹、来源信息；无比赛谱或模型权重。`kaggle datasets create -p artifacts/runs/011-expansion-asset --keep-tabular`实际成功，默认private，未使用`--public`。

## v10实现与保留的算法

构建入口[scripts/011/build_011_submission.py](../../scripts/011/build_011_submission.py)，主文件[enveda-submission-v10.ipynb](../../notebooks/submissions/submission-v10/enveda-submission-v10.ipynb)。原712199候选追加538，合计712737。`run_groups`由原train/COCO两分支改为train/COCO/新增三分支，COCO继续使用原cidx，新候选独立读取packed FP。先对完整质量窗评分，再原canonical-max/top100，不尾接已截断列表。

GPU主版本的FP模型、质量换算、ICE cell与v9逐字一致。原代表谱、10ppm、权重、fused_merge、ddof1、缺失续后与Top25均保留。另补原2000分子之外的动态尾部分块及实际test ID完整性校验，避免固定长度假设。输入使用唯一mb539前缀，避免原find('fp.npy')抓错文件。

既有5个dataset版本固定为FP1/COCO2/train-pool1/ICE1/cp313-wheelhouse1，新资产v1；保留v9镜像digest `2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f`，互联网关闭，notebook私有。

CPU兼容性入口另置`artifacts/runs/011-cpu-probe/`：同模型和评分，只改device为CPU/4线程、按metadata选2个具有ICE支持谱的分子，输出`cpu_probe.csv`，**不得提交该小样本入口**。主GPU版本未因此改变。

本轮不创建测试文件、框架或CI；手动编译各cell、核对数据/代码与实际运行日志，运行成功、提交成功和得分分别记录。

## 远端实际执行

- 10月17日01:45 UTC，配额恢复后同一GPU入口push成功为`ccjjllt/enveda-submission-v10` **v1**，kernel id139133853。MCP latest metadata确认current_version_number=1、private/T4/offline、原镜像。拉回的35个cell类型及拼接后source逐个相同；远端source字段为字符串、本地是行列表，不能以JSON字段形式差异误判代码变化。远端副本在`artifacts/runs/011-execution/remote-source-v1.ipynb`。
- `versionLabel="1"`接口返回找不到标签，latest接口可读且明确版本1；本轮不再次push，避免latest指向漂移。API `last_run_time`仍出现10月14日的过时日期，耗时以本轮实际观察时间和完成日志为准。
- GPU主版本第一次CLI push被Kaggle明确拒绝：`Maximum weekly GPU quota of 30.00 hours reached.` 未取得可运行的GPU版本，不能称运行中或提交成功；这是Kaggle配额限制，非自动审批拒绝。
- 将title与slug对齐后，独立CPU小样本入口push成功：`ccjjllt/enveda-v10-cpu-probe` **v1**，关闭互联网、CPU、最多900s，仅2个metadata支持分子，输出`cpu_probe.csv`。该版本禁止作为比赛提交。
- CPU v1完整成功：2分子/149候选、447个预测任务，status=ok、n_pred_fail=0、149条fused_merge有效、两行各25个候选；RDKit2026.03.6。预测243.34s、整段253.6s、1.84任务/s，notebook日志约322s结束。按旧v9占位运行72333任务仅作量级估算需10.92h，超过比赛9h，不能直接按当前配置全量提交。
- 追加**同样2分子**的CPU v2，唯一执行配置变化为torch线程4→1，模型与评分保持相同；已push成功，最多900s。目的是检查CPU线程开销，不调模型/权重/评价规则。原v1输出保留在`artifacts/runs/011-cpu-probe-output/`，v2另存目录，均不得提交。
- CPU v2完整成功：447/447预测，预测487.09s、整段496.4s、0.92任务/s，无预测失败；两行最终CSV与v1逐字节相同。按同72333任务估算21.84h，单线程没有加速。产物`artifacts/runs/011-cpu-probe-output-v2/`。CPU两次使用Python3.13.15、torch2.11.0+cu128（cuda不可用）、RDKit2026.03.6及同离线镜像。
- 量级估算仅来自2分子，不是实跑完整CPU计时，也不声称所有可能CPU优化都不行；但已试配置无法支持按9小时上限提交，因此没有启动预计超限的全量CPU任务，没有减候选/砍ICE/提交截断结果。10月14日的两个CPU作业均COMPLETE，没有遗留CPU探针作业；10月17日GPU作业状态另见上文。

## 阻塞与恢复步骤

10月14日阻塞来自Kaggle实际GPU周配额拒绝，不是缺用户授权或自动审批拦截。10月17日已解除并实际启动版本1。用户已授权运行/提交，不需例行再次确认；本轮是用户返回后的续接，没有创建未来自动任务。

GPU刷新时间2026-10-17 00:00 UTC，即北京时间**10月17日08:00**。恢复时复查额度，现有数据集v1可直接复用；按以下步骤续接：

```bash
kaggle kernels push -p notebooks/submissions/submission-v10 --accelerator NvidiaTeslaT4
# 记录实际push返回版本号，等该版本COMPLETE后人工核对输出与ICE完整状态。
kaggle competitions submit enveda-CASMI26-molecule-id-mass-spectra \
  -k ccjjllt/enveda-submission-v10 -v ACTUAL_COMPLETED_VERSION \
  -f submission.csv -m "v10 fixed MassBank expansion, 538 valid new structures"
```

提交前须按真实test核对全部ID、原FP异常计数、ICE status=ok、文件格式和实际耗时；再查看隐藏重跑和publicScore，不用占位测试命中率声称有效。v9r2保留，不改变原最终选择。完整恢复状态在`artifacts/runs/011-execution/run-state.json`，精简[执行记录](../../data/references/submission-execution-2026-10-14.json)。

## 实际集成路径人工核对

运行[scripts/011/check_011_integration.py](../../scripts/011/check_011_integration.py)，从实际v9/v10 notebook AST提取`run_groups`，只把执行设备换成本地CPU、写出路径改为`artifacts/runs/011-check/`；复用550个代表谱logits（NP250/G1研究250/R3-50），随后执行原生RDKit canonical-max/top100。无额外神经前向、无新保留集评价、无参数选择。

- shipped候选对61826→61865，canonical对61390→61429，新增39对；无exception或canonical fallback。空窗5→4，扩库救回1窗，4个原未知质量窗口仍空。
- 原候选顺序550/550不变。仅R3 ROTFFYYCARWRHQ一条FP分因同组padding宽度变动相差1个float32 ULP（6.1035e-05），排序不变，不声称所有浮点分数逐bit相同。
- G1完整top100与原扩池缓存250/250一致，R3完整top100与010重新计算的float64参考50/50一致。
- NP一例WYJAPUKIYAZSEM第84/85名因float64两分在float32舍入为同值而按canonical排序互换；原v9和v10都发生，真值rank21、Top25与Top100集合不变，不是扩库回归。
- GPU主版本model cell4、mass cell5、ICE cell32逐字与v9一致；改动仅候选追加/指纹读取、动态尾部分块、输出ID完整性及说明。产物`artifacts/runs/011-check/{integration,raw_score_comparison,r3_full_reference}.json`和日志保留。

## 10月17日运行期间的信息复查

- 官方[重评分公告749430](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/749430)说明早期提交已重评分；本轮重新读取v9r2/ref57028399仍为COMPLETE/public0.323。即时排名2096/3451仅是当时快照，不是本次运行结果。
- 当前官方公开文件的bytes和创建日期与原记录一致。最终输出仍须结合实际运行读取的分子数核对，不能将文件元数据相同代替产物检查。
- [官方10月16日回复749217#3539870](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/749217#3539870)重申第三方衍生权重需原训练代码和允许训练数据，上下文针对无公开recipe/记录的DreaMS衍生版本，没有点名禁止本项目s2。010已取得原作者完整训练源码和competition train声明；具体s2输入、划分、CLI的关联仍待补齐，不将“尚未核全”等同禁用，也不声称奖金材料已完整。
- [Ahmed公开v11](https://www.kaggle.com/code/ahmedberatozer/casmi26-v11-e2rank2-rel150)新增第二引擎首选去重后插入第2位、调整PubChem提升阈值。作者[实验说明](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/749412)的收益未由本地独立验证；不混入当前单一扩库提交。
- 对s2做了有界溯源补查：现用checkpoint为step20000/6930bits/d512/6层/36,020,242参数；原作者baseline v34 cell41明确s2单谱、m1合并谱，支持输入视图来源，但不证明实际seed/CLI。发布checkpoint只有model/step/nbits/d/layers，公开训练器save还写opt，发布包装步骤缺失。作者另一份[评估说明](https://raw.githubusercontent.com/cubres/enveda-casmi-2026/main/datasets/casmi-2026-fingerprint-model-eval/README.md)的clean_s2是step24000另一轮，不将其lr/split移植到现用权重。仍需补spec.npz构建/six/is_val、FP预处理、实际CLI以及训练输入至发布权重的关联；未重训练。

## 10月17日研究现状梳理

北京时间约10:22，按用户“了解当前的研究现状”请求阅读历史计划、进展及010/011产物记录，并通过Kaggle MCP做只读状态核对：

- `get_notebook_session_status`（`ccjjllt/enveda-submission-v10`，latest）返回`RUNNING`；`list_notebook_session_output`仅返回空`log`，本次未取得可核查输出，不能确认完成、实际吞吐或成功生成submission。
- `search_competition_submissions`（本比赛、Date、pageSize=10）仍返回5笔历史提交。最好有效结果为v9r2/ref57028399，`COMPLETE`、public **0.323**；没有v10比赛提交或新分数。历史有效提升为v7 **0.206** → v8 **0.307** → v9r2 **0.323**。
- 当前执行入口仍是011：只增加538个有效MassBank结构，保留v9排序算法，等待完整GPU输出核查后推进原已授权的一次验证性提交。010的R3本地扩库增益受同源和人工分层限制，不能预测隐藏分数；NP/G1的零指标变化属于已有真值场景的干扰检查。
- 后续研究继承010：优先补现有模拟器不支持条件下的结构碎片证据；完整MIST/FRIGID谱条件链仍未跑通。条件多谱、D2、FIORA与F4a当前配置不部署；未见结构泛化与已见结构独立谱检索继续分开评价，原保留250已用于确认，不再作为未使用保留集。

本次仅阅读、远端状态查询与文档更新；另明确README中010“尚未集成/运行”和本页“无遗留CPU作业”的历史范围，避免与011当前GPU运行状态混淆。未新增训练、推理、push或submission，未创建测试文件或自动任务。既有v10作业仍在运行。

## 10月17日完整运行核查与实际提交

用户告知运行完成后，MCP重新读取status、metadata、session output及提交列表：v10为COMPLETE、current_version_number=1。完整日志最后时间2487.03秒（约41.45分钟），写出CSV在2476.50秒；不使用metadata中仍过时的10月14日last_run_time计算时长。原源码和评分规则未修改，没有再次push。

通过以下命令拉回明确版本的四个产物（沙箱DNS受限后以允许联网的同命令成功执行）：

```bash
kaggle kernels output ccjjllt/enveda-submission-v10/1 -p artifacts/runs/011-v10-output --file-pattern '^(submission\.csv|ice_meta_f1\.json|iceberg_scores_f1\.json|canon100\.json)$' --quiet
```

本地用`/opt/homebrew/bin/python3.11`手动读取CSV、实际test ID、候选缓存及ICE分数，RDKit2026.03.6解析全部9357种输出SMILES；按原ddof1 z融合、已评分候选优先及缺失候选续后重建400行排序，逐行与CSV完全一致。结果和文件bytes/SHA在`artifacts/runs/011-v10-output/verification.json`，日志和原始产物同目录。无标签命中率计算、无测试文件或框架。

- 实际读取400分子/1213谱，输出400行×25；ID无缺失、重复或额外项，SMILES解析失败0，首选裸CCO为0。
- FP空窗/质量回退/异常均0；canonical fallback为0。完整窗57302条，canonical57149条，Top100总候选31044。
- ICE完整meta：`status=ok`、`errors=[]`，覆盖371/400分子且371全部评分；`n_pred=unique_jobs=72336`、`n_pred_fail=0`，有效merged候选29017。ICE段1990.9秒，纯预测1854.98秒、39.0任务/秒，峰值显存4.6GB，未触及7200秒预算。29个条件不支持分子沿用FP排序。
- `pad_cco=35`表示35行短名单不足25时在末尾共补408个CCO，非35个预测失败；与v9相同，原候选无空行，未为本次扩库修改该行为。Python3.13.15、torch2.11.0+cu128、RDKit2026.03.6，设备cuda/T4；纯推理未新增训练seed或权重。
- 与本地保存的v9可见输出相比，2行候选序列改变、首选改变0；该比较只核查输出差异，不能预测隐藏收益。CSV为391970字节，SHA256 `fe459dfd89192870339255793f551d793e1e53780faeb1c199dc84484c775833`。

核查通过后按既有授权调用`create_code_competition_submission`，指定owner=ccjjllt、slug=enveda-submission-v10、kernelVersion=1、fileName=submission.csv；服务返回**ref57293821**，提交时间2026-10-17T03:43:22.103Z（北京时间11:43:22）。随后提交列表确认指向**scriptVersionId359192425**。当前API省略status与public_score、total_bytes为0，仅能确认已接收，不能据此称隐藏重跑成功或得到0分；等待终态和分数。保留v9r2/ref57028399/public0.323与原最终选择，不重复提交同版。

## 10月17日评分终态

北京时间约12:51再次读取ref57293821，已为**COMPLETE/public0.323**，total_bytes508585；v9r2仍COMPLETE/public0.323，508590字节。上节“等待”是11:43的历史状态。538结构扩库本次未观察到显示public提升，不能据此倒推隐藏类别比例或逐行变化；同分原因、官方更新与后续实际实验转入[012进展](012-score-plateau-and-complementary-evidence.md)。没有重复提交。

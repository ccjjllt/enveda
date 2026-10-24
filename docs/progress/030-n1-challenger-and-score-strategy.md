# 030 — N1 联合证据提交与冲奖研究路线：实际进展

日期：**2026-10-22 21:15 左右（北京时间）**。对应[计划](../plan/030-n1-challenger-and-score-strategy.md)。

## 2026-10-24 最终结果：v15 public 0.464，提升0.008

用户通知成绩已出后，**18:16（北京时间）**用ccjjllt认证查询确认：submission **57583178 = COMPLETE / public 0.464**；旧最佳v14/ref57462581为0.456，**绝对提升0.008**。030已完成一次实际提分提交，v15作为新的已验证公榜基线；v14/v8保留作对照。

同次完整排行榜快照：**171 / 3940**，得分0.464；第1/第5/第20分别为 **0.504 / 0.489 / 0.471**，距第20名0.007、距第5名0.025。这些是当时公榜位置，不是最终奖金或金牌线。10月22日记录的515/3802只是历史快照，不作同一时点名次对照。

本次结果支持030整套提交优于此前最佳，不能把全部0.008精确归因于最后joint一行：关闭complement同时改变前面的PubChem重排与中间融合，且v14/v15挂载资产未逐版本证明完全一致；没有取得隐藏集N1分层日志，不能从总分推断获益分子数量或类别比例。**代码完成、可见运行成功、正式评分成功、公榜提高均已分别确认；离线泛化增益与私榜成绩仍未知。**

下一步沿用v15，先固定资产版本，在P250开发诊断集一次缓存完整GPU证据，重放“模拟器有/无 × rel-promotion开/关”，解释旧CPU评估方向反转；随后单独验证028扩池。P250已多次用于决策，不称独立留出。新结构泛化评估仍需核实模型训练排除与跨库重复。本次查分没有启动新实验、push或重复提交。

查询依据：`artifacts/submissions/030-v15-kernel-v9/submission-score.json`、`submissions-scored.csv`、`leaderboard-scored/summary.json`；精简结果已同步 `data/references/research-results-030.json`。

### 本轮完成与提交记录

用户通知跑完后，北京时间13:03查得 kernel **v9 COMPLETE**，currentVersionNumber仍为9。日志至结束 **1533.50秒（25分33秒）**，环境与源码正确，没有发现提交阻断。

- ICE：status ok，12个输入分子中11个可覆盖，11个实际评分；815个输入候选中794个有分数，1601次预测，失败0，耗时106.45秒。
- 原生GL：status ok，11个可覆盖分子全部评分，794个候选有分数，1601次预测，预测/特征失败均0，耗时79.73秒。
- N1：当前smoke12个分子中11个有真实[M+H]+，剩余1个没有NH4/脱水谱，因此真实N1适用数及joint双覆盖数均0。合成路径实际完成11个分子、39张代理谱（其中8张多CE合并）、1940次预测、失败0，耗时39.03秒。这仅证明代理推理路径跑通，不证明真实joint提分。
- 最终输出400行，ID唯一且与输入完整对应，没有空SMILES，每行1–25个候选；12行真实推理、388行占位填充符合基线smoke逻辑。SHA256 `5c098f153b748a651757c5aea2f078a7d1d82cfb3ae89bd656c5cc81a203ce76` 与最终E2R2报告一致。正式比赛提交使用代码重跑，不是上传此占位CSV。
- 日志中 `CLAW STAGE HEALTH ... NOT RUN` 是原代码成功后未更新标签，实际ICE/GL meta均ok；中途 `NO-OP` 是旧E004阶段提示，后续E006/E2R2已完成。均不作为模型失败或阻止本次提交的依据。

**13:05:38（北京时间）已将kernel v9正式提交一次，命名v15，ref `57583178`。13:06查询为PENDING，publicScore空；新分数尚未返回，最佳仍0.456。** 当日剩余4次提交。用户再次表示**“跑完告诉你”**，停止主动查分，等待用户通知；没有安排自动任务，也不重复push/提交。

关键输出下载自明确的 `version_number=9`，保存于 `artifacts/submissions/030-v15-kernel-v9/`，含submission、人工格式核对、ICE/GL运行报告、最终排序统计、提交回执及状态。完整日志位于 `artifacts/runs/030/execution-2026-10-24/completed.log` 和 `completed-readable.log`。CLI元信息的lastRunTime返回旧日期，不将其作为本轮执行时间；以本轮push/submit本地时间与实际日志耗时记录。

上述13:06待评分状态已由本节顶部18:16的COMPLETE/0.464结果更新。不要将可见运行日志的N1覆盖数当作隐藏重跑覆盖数。

### 本轮启动与跟踪记录

用户明确要求继续。北京时间12:27实查 ccjjllt GPU **0/30h已用、30h可用**，原kernel COMPLETE，最佳提交仍0.456。随后手动push `notebooks/030`，服务端返回 **kernel v9**，状态 RUNNING；尚未完成或提交。

031目录迁移未改notebook字节，SHA256仍为 `d64c97cf657e5916fb7ece3ac4b7a2e69d8cf4e51f8018dbcf558d5107e5a5b3`。服务端currentVersionNumber=9，下载回来的36个cell源码与030一致，私有/断网/T4、15数据源及镜像digest保持。原始执行记录：`artifacts/runs/030/execution-2026-10-24/`。

实际运行日志已经确认 **Python 3.12.13、Tesla T4×2、metric RDKit 2026.03.3**；64个基础资产文件匹配MANIFEST。可见占位数据进入12分子smoke，最后已返回阶段为候选引擎子进程，尚未看到ICE/GL/N1结果或最终输出。日志读取连接偶发中断，远端状态仍RUNNING且无失败消息；读取连接中断不等于远端失败。

北京时间约12:40，用户说**“跑完我通知你”**。已停止本地只读日志流和主动查询，**没有取消Kaggle运行、没有提交、没有安排自动任务**。下次用户通知后先检查v9当前状态与完整日志，确认实际输出后再提交v9，不能重复push。若期间别人更新了同一kernel，应先核实v9，不能默认最新输出就是本次输出。

本轮沿用预训练模型和原排序器配置，不新增训练实验；基线运行本身会在已有训练行上拟合GBM，种子为Engine2 `(0,1)`、PV `(0,1,2,3)`。完成后补实际耗时、输出和提交结果。15项资产的当时可见版本保存在 `asset-versions-observed.json`，是push后只读快照，不冒充服务端精确挂载版本证明。CLI 2.2.4的部分版本参数未正确传递，SDK显式 `version_label='9'/'8'` 也分别遇到404；本轮状态使用已核实currentVersionNumber=9的当前会话，源码通过同一kernel最新版本取回。

等待期间的只读研究定位：旧验证器的快照在模拟器重排之后，无法恢复原始BASE/PC；后续最小开发诊断可在P250上把快照提前至simulator排名尾部之前，一次GPU缓存后重放“模拟器开/关 × rel>150提升开/关”。它检验PC第一名被模拟器替换、而rel门值不变的交互机制。P250全是[M+H]+、已反复使用，只能解释通用promotion机制，不能评价N1收益或称独立留出。当前所谓promo_off只关闭rel分支，仍保留S/pop分支。此处仅记录后续假设，本轮未实施该实验。

下面10月22日内容为准备阶段历史记录。

## 实查研究位置

使用 ccjjllt 认证读取提交记录与私有 notebook，并下载完整排行榜：

| 项目 | 本轮实际结果 |
|---|---|
| 最佳提交 | v14，ref `57462581`，COMPLETE，public **0.456** |
| 旧对照 | v11 **0.446**；v12直接参考 **0.434**；v13关闭promotion **0.432** |
| 当前名次 | **515 / 3802**，队伍 ccjjllt |
| 榜单参照 | 第1 **0.504**；第5 **0.484**；第20 **0.469** |
| 与第20/第5差距 | **0.013 / 0.028**；它们不是最终金牌/奖金分数线 |
| 现有 kernel | `ccjjllt/enveda-020-e006-repro`，COMPLETE；027记录得分版本为v8 |
| ccjjllt GPU | 已用 **30.10h / 30h**，剩余 **0**；API刷新时间 UTC 10-24 00:00，即北京时间 **10-24 08:00** |
| TPU | 剩余20h；当前CUDA模拟器不能直接当作TPU推理运行 |

本机 CLI 默认身份是 qingyu1232，起初对 ccjjllt 私有 kernel/提交记录返回403。用户指出认证在 `.kaggle`，后在内存中使用已有 `access_token_ccjjllt_check` 并确认身份为 **ccjjllt**。没有修改默认认证，也没有输出、复制或提交密钥；其他账号的额度不属于本轮运行依据。

本轮用户明确选择：**先备好，届时手动继续**。没有新建自动任务，没有 GPU push、训练、比赛提交或新分数。

## 确认的缺陷及研究解释

1. **029a漏移植关键开关。** 其 simulator 仍 `N1_COMPLEMENT=True`，`n1_complement` 删除所有有有限 ICE 分数的 N1 GL 结果，之后才设置 `N1_GL_MIDS`。最终增加 `[ice, gl]` 不能挽回已删除的证据。最新 lavinwins 源码明确是 False。此结论是源码路径核对，不是已经跑出的指标。
2. **现有本地验证省略模拟器，不能验证完整融合。** P/Q上关闭promotion分别 +0.0374/+0.0253，真实提交却 −0.014。仪器、来源和抽样偏差仍可能存在，模拟器交互也尚待实测；不能仅靠继续改变抽样或扫阈值消除这个问题。
3. **召回与排序风险被混合。** 029a把028扩池、joint、20 analog parents和DreaMS放在一起；既往大生成在C改善、B退化，不能假定增大候选池必然提分。本轮拆出有外部实分先例的完整 N1 路径。
4. **旧结论需限缩。** 022/024中由单个public总分推断隐藏集“主要缺参考”或“大多在PubChem”不成立；full1 的 NP 0.918 与 fp0 0.578 不能单独证明训练成员身份。030不继承这些强推断。

## 实际修改与产物

- [提交入口](../../notebooks/030/enveda-030-n1-challenger.ipynb)：以实际已得分源码为基线，新增说明并修改原cell21、33（新cell22、34，零起始编号）。
- `N1_COMPLEMENT=False`；最终只对 N1 代理覆盖列表 joint ICE＋GL；增加覆盖与有限分数计数。
- 其余 **33个原cell源码逐字节不变**，维持原15数据源、候选池、默认生成规模、POP=0.15。
- [metadata](../../notebooks/030/kernel-metadata.json)：私有、断网、T4、既有Py3.12镜像，`code_file` 已对应真实文件；本地目录可直接作为 CLI push 入口。
- [改动说明](../../notebooks/030/PATCHES.md)、[源文件哈希](../../notebooks/030/source-manifest.json)、[手动构建脚本](../../scripts/030/build_030_n1_challenger.py)。
- README首部改为030入口，并在029记录明确旧开关问题和新的实验顺序。原029 notebook保留作历史产物，不再直接推送它。

来源：基线 [guhongbin/v465](https://www.kaggle.com/code/guhongbin/casmi26-v465-v15-pop015)，联合路径 [lavinwins](https://www.kaggle.com/code/lavinwins/casmi26-top-1-gold) / [flexonafft](https://www.kaggle.com/code/flexonafft/casmi26-closing-the-evidence-gap)。下载的当前自有源码与v465 **SHA256完全一致**：`8ce4f69e6f8a6e38c0d7ee4ffec151cc2b7454699abb0f53b1e16fd592852e29`。保留原署名及各资产许可。

## 人工核对与实际结果边界

- 手动解析32个Python cell语法成功；清除了notebook已有输出；人工逐项看过与自有基线的源码diff及metadata。
- 独立只读复核没有发现新增变量未初始化或joint路径被再次阻断。自有kernel的API镜像前缀为 `kaggle-private-byod`，030继承此前metadata的 `kaggle-gpu-images`，两者digest相同；服务端是否实际采用Py3.12镜像仍须看首次运行日志，不能以本地配置代替运行确认。
- upstream lavinwins 与030所用行为相符；保留N1 GL也会影响已有PubChem重排和中间融合。因此同次运行E004对照只用于说明当前运行，不能称为原v465的完整A/B。
- 没有真实N1 proxy运行缓存可供本地重放。旧005缓存中有19个N1适用输入，其中3个有ICE有限分数，但旧GL来自原生支持路径，不可伪装成代理GL。可供后续极小试点的结构为 `FBWWXAGANVJTLU`、`IECPWNUMDGFDKC`、`KLLGGGQNRTVBSU`。
- **代码已准备；远端未运行；离线增益未测；新submission未提交；公榜仍0.456。** 没有新增训练或随机种子；沿用上游推理配置。精确挂载数据/权重版本、实际设备、时长及新kernel版本待远端运行后填写。
- 本轮没有创建测试文件、测试框架或CI；语法与源码核对均为人工触发。

安全的原始查询快照位于 `artifacts/runs/030/`，包括 `quota.json`、`submissions.csv`、`leaderboard/`、上游及当前kernel源码；可携带的摘要见 `data/references/research-results-030.json`。

## 下次直接执行

用户说“继续030提交”后，从本目录产物开始，不必重新设计方案。先检查ccjjllt配额是否实际恢复，然后在当前进程内选用已核实凭据；不要用默认的其他账号。例行CLI调用可使用如下模式，凭据只进入进程环境：

```python
import os, subprocess
from pathlib import Path
os.environ['KAGGLE_API_TOKEN'] = (
    Path.home() / '.kaggle/access_token_ccjjllt_check'
).read_text().strip()
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
assert api.config_values.get('username') == 'ccjjllt'
api.quota_view_cli(output_format='json')
# 确认额度可用后，手动执行：
subprocess.run(['kaggle', 'kernels', 'push', '-p', 'notebooks/030'], check=True)
```

使用现有 Kaggle CLI 的 Python 环境执行，保持项目为当前目录。收到版本号后观察同一版本完成情况，查看日志与产物；随后 `kaggle competitions submit ... -k ccjjllt/enveda-020-e006-repro -v <实际完成版本> -f submission.csv -m 'v15 030 v465 + N1 challenger joint ICE-GL'`。不要上传本地CSV代替代码提交，不猜版本号，不把可见smoke当泛化结果。

完成一次实分对照后，研究优先级为：完整GPU留出流程 → 单独028召回增量 → 根据同式/缺参考/负离子错例引入新的候选或谱证据。目标是能在独立数据上成立的自有改进，而非反复适配public榜。

## 2026-10-24 12:05 巡检附注

ccjjllt GPU 配额已恢复为剩余 **30.00h**（已用 0）。远端仍未 push/运行 030；公榜仍 0.456。按用户「届时手动继续」约定，本巡检**未**执行 kernels push。手动顺序见上文「下次直接执行」。


## 2026-10-24 18:10 巡检：v15实际分数

- ref `57583178`（v15，030 kernel v9）状态COMPLETE，**public 0.464**，较v14 0.456 **+0.008**，为当前最佳。
- 全量公榜快照：名次170/3939（0.464同分约170–352名）；#1 0.504，#10 0.475，#20 0.471，#50 0.467。金牌线约需0.471以上（以当时名次估计，非官方线）。
- 结论仅限public子集；N1联合ICE-GL路径在公榜上为正向，但未做独立留出验证。下一步按上文优先级：完整GPU留出流程 → 单独028召回增量 → 同式/缺参考/负离子错例的新候选或谱证据。

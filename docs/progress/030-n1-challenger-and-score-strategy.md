# 030 — N1 联合证据提交与冲奖研究路线：实际进展

日期：**2026-10-22 21:15 左右（北京时间）**。对应[计划](../plan/030-n1-challenger-and-score-strategy.md)。

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

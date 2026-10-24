# 030：v465 + 完整 N1 challenger

入口：`enveda-030-n1-challenger.ipynb`。10月22日完成准备；10月24日 `ccjjllt/enveda-020-e006-repro` v9已COMPLETE，Python3.12.13/T4×2，可见运行耗时25分33秒；正式提交v15/ref57583178已COMPLETE，**public 0.464，比原基线0.456提高0.008**。结果及解释边界见030进展。

## 改动

基线为下载的 `guhongbin/casmi26-v465-v15-pop015`，与本账号当前
`ccjjllt/enveda-020-e006-repro` 下载源码逐字节相同。
基线 SHA256：`8ce4f69e6f8a6e38c0d7ee4ffec151cc2b7454699abb0f53b1e16fd592852e29`。

仅修改两个原代码 cell（编号从 0 开始）：

- 原 cell 21 → 新 cell 22：`N1_COMPLEMENT=False`，保留有有限 ICE 分数的 N1 代理 GL 分数，补充实际覆盖计数。
- 原 cell 33 → 新 cell 34：只在 `N1_GL_MIDS` 的最终排序使用联合 ICE + GL，保留 joint 分支及有限分数计数。

另加中文说明，清除输出和执行编号。其余 33 个原 cell 源码逐字节不变。
没有引入 028 扩池、20 analog parents 或额外 DreaMS 数据源；数据源仍为原 v465 的 15 项。
metadata 指向本目录实际文件，保留私有、断网、T4 和既有 Python 3.12 镜像配置。

## 证据与边界

路径来自 [lavinwins v16](https://www.kaggle.com/code/lavinwins/casmi26-top-1-gold)，
其原始来源为 [flexonafft v47](https://www.kaggle.com/code/flexonafft/casmi26-closing-the-evidence-gap)。
029a 只移植 joint 分支但遗漏 complement 开关；有 ICE 的 N1 分子在此前已被过滤，不能得到真正的双模型融合。

关闭 complement 也会使保留的代理 GL 分数进入原有 PubChem 重排和中间融合。
因此本次改动是完整 N1 challenger 路径；同次 E004 对照不代表原 v465 全流程基线。
可见占位数据只用于运行核对，不能用来判断泛化或提分。

## 人工核对

2026-10-22：逐 cell 源码差异核对，仅上述两处代码 cell 改变；32 个 Python cell 均通过 `ast.parse`。
metadata 的 15 个数据源与 v465 完全相同，`code_file` 存在，所有输出已清除。
未启动训练或 Kaggle 运行，也未创建测试文件。

手动重建：`python3 scripts/030/build_030_n1_challenger.py`。
构建输入、输出 SHA256 和变更位置记录在 `source-manifest.json`。

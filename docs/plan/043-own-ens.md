# 043 计划：自研双模型集成重排
- 042 续训（warm start）最终 held-out 同分异构 MRR 0.7636@fp_w1.0（初始 0.7606@0.5），小幅提升。
- 040 重排单元改为两模型：039 模型（fp_w0.5）+ 042 模型 model2.pt（fp_w1.0），每个查询内候选分数 z-score 后相加，再按 W0.25 与基线秩融合。
- 数据集 ccjjllt/enveda-own-spec 新版本加入 model2.pt / train_result2.json（原文件保留）。
- 内核 v20 → k_watch 要求日志含 "043 ENS 2" 与 "NEURAL OK" 才提交 v23。

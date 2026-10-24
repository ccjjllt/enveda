# 035 计划：v15 + 仅 034 负离子代理
v16（033 注入）0.450、v17（033+034）0.446，均低于 v15 0.464。033 注入可能是退化主因，单独验证 034：
以 030 notebook（v15）为底，只在 cell 22 的 build_ice_input 后插入 notebooks/034/negproxy_snippet.py，不含 033 注入。GPU 跑一次、提交一次。

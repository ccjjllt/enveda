"""Build the round-030 N1 challenger submission from the downloaded v465 source.

Run manually from the repository root. This only writes the submission notebook,
metadata and provenance; it does not execute or push a Kaggle notebook.
"""

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/030/upstream-v465/casmi26-v465-v15-pop015.ipynb"
CHALLENGER = ROOT / "artifacts/030/upstream-lavinwins/casmi26-top-1-gold.ipynb"
OUT = ROOT / "notebooks/030"
NAME = "enveda-030-n1-challenger.ipynb"


def source(cell):
    return "".join(cell.get("source", []))


def unique_cell(notebook, marker):
    matches = [(i, c) for i, c in enumerate(notebook["cells"]) if marker in source(c)]
    if len(matches) != 1:
        raise ValueError(f"Expected one cell containing {marker!r}, found {len(matches)}")
    return matches[0]


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one occurrence of {old!r}")
    return text.replace(old, new, 1)


base = json.loads(BASE.read_text())
challenger = json.loads(CHALLENGER.read_text())
notebook = copy.deepcopy(base)
sim_i, sim = unique_cell(notebook, "N1_COMPLEMENT =")
sim_source = replace_once(
    source(sim),
    "N1_COMPLEMENT = True\n",
    "# N1 challenger: flexonafft v47, as integrated in lavinwins v16.\n"
    "N1_COMPLEMENT = False\n"
    "print('[N1 challenger] configuration: N1_COMPLEMENT=False; real [M+H]+ stays on H1', flush=True)\n",
)
_, challenger_sim = unique_cell(challenger, "N1_COMPLEMENT =")
coverage_start = "# Counts distinguish selected proxies from completed, retained forward scores.\n"
coverage_end = "# ---- /N1A_MCE"
coverage = source(challenger_sim).split(coverage_start, 1)[1].split(coverage_end, 1)[0]
sim_source = replace_once(sim_source, coverage_end, coverage_start + coverage + coverage_end)
sim["source"] = sim_source.splitlines(keepends=True)

# The upstream factorial diff contains only the N1 joint branch and its counters.
factorial_i, factorial = unique_cell(notebook, "def _factorial_entry(scope):")
_, challenger_factorial = unique_cell(challenger, "def _factorial_entry(scope):")
factorial_source = source(challenger_factorial)
factorial_source = replace_once(
    factorial_source,
    "# ice_fallback=False still rebuilds the exact E004 GL-only control.",
    "# ice_fallback=False rebuilds this run's E004 GL-only control; upstream N1 inputs already changed.",
)
factorial_source = replace_once(
    factorial_source,
    '"Visible smoke is not independent validation.",',
    '"Visible smoke is not independent validation.",\n'
    '                "This same-run E004 control is not the original v465 full-pipeline baseline.",',
)
factorial["source"] = factorial_source.splitlines(keepends=True)

header = """# 030 — v465 + 完整 N1 challenger

以已复现 public 0.456 的 guhongbin/v465 为基线，本轮仅引入 N1 challenger：
保留已有 ICEBERG 证据的 N1 代理 GLACIER 分数，并在 N1 最终排序中联合使用两种证据。
`N1_COMPLEMENT=False` 是必要部分；仅添加最终 joint 分支会被旧的 complement 筛选抵消。

候选池、默认生成规模、人气权重和数据源保持 v465 配置。
本改动也会使 N1 代理分进入已有的 PubChem 重排和中间融合，
因此同次运行的 E004 对照不等同于原 v465 全流程对照。
可见占位数据只检查运行；提分须以本账号的实际提交结果为准。

来源与署名：
- [Hongbin Gu / guhongbin v465](https://www.kaggle.com/code/guhongbin/casmi26-v465-v15-pop015)
- [lavinwins v16 N1 challenger 路径](https://www.kaggle.com/code/lavinwins/casmi26-top-1-gold)
- [flexonafft v47](https://www.kaggle.com/code/flexonafft/casmi26-closing-the-evidence-gap)

保留以下原始 notebook 的全部署名；基线代码 Apache-2.0，外部数据与模型遵循各自原许可。
"""
notebook["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": header.splitlines(keepends=True)})
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

metadata = json.loads((ROOT / "notebooks/029/kernel-metadata.json").read_text())
base_metadata = json.loads((BASE.parent / "kernel-metadata.json").read_text())
metadata["dataset_sources"] = base_metadata["dataset_sources"]
metadata["code_file"] = NAME
OUT.mkdir(parents=True, exist_ok=True)
(OUT / NAME).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
(OUT / "kernel-metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
manifest = {
    "baseline": "guhongbin/casmi26-v465-v15-pop015",
    "baseline_sha256": hashlib.sha256(BASE.read_bytes()).hexdigest(),
    "challenger_source": "lavinwins/casmi26-top-1-gold",
    "challenger_source_sha256": hashlib.sha256(CHALLENGER.read_bytes()).hexdigest(),
    "notebook_sha256": hashlib.sha256((OUT / NAME).read_bytes()).hexdigest(),
    "changed_source_cell_indices_zero_based": [sim_i, factorial_i],
    "output_cell_indices_zero_based": [sim_i + 1, factorial_i + 1],
    "dataset_count": len(metadata["dataset_sources"]),
    "changes": [
        "N1_COMPLEMENT=False retains proxy GL scores where finite ICE scores exist",
        "Final joint ICE+proxy GL rerank only on N1_GL_MIDS",
        "N1 coverage and finite joint-score counters; same-run control limitation",
    ],
}
(OUT / "source-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(manifest, ensure_ascii=False, indent=2))

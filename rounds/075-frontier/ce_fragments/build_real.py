"""Package the matched-CE cosine comparison for the real candidate lists."""

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SLUG = "enveda-075b-real-ce"
PANEL = "jiyujia/enveda-075b-queries"
CANDIDATES = "jiyujia/enveda-075b-pipeline"


def build(destination, cpu, panel_kernel=PANEL, candidate_kernel=CANDIDATES, slug=SLUG,
          measured_ev_only=False):
    scorer = (HERE / "score.py").read_text()
    scorer = scorer.rsplit('\nif __name__ == "__main__":', maxsplit=1)[0]
    evaluator = (HERE.parent / "panel/evaluation/evaluate_scores.py").read_text()
    device = "cpu" if cpu else "cuda"
    batch_size = "24" if cpu else "96"
    panel_slug = panel_kernel.split("/")[1]
    candidate_slug = candidate_kernel.split("/")[1]
    run = f'''

def mounted_from(notebook, filename):
    paths = [path for path in Path("/kaggle/input").rglob(filename) if notebook in path.parts]
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one {{notebook}}/{{filename}}, found {{paths}}")
    return paths[0]


if __name__ == "__main__":
    queries = mounted_from({panel_slug!r}, "queries.parquet")
    candidates = mounted_from({candidate_slug!r}, "candidates.parquet")
    labels = mounted_from({panel_slug!r}, "labels.parquet")
    scores = Path("/kaggle/working/scores.parquet")
    args = [
        "--queries", str(queries), "--candidates", str(candidates),
        "--cosine-only", "--device", {device!r}, "--batch-size", {batch_size!r},
        "--output", str(scores),
    ]
    if {measured_ev_only!r}:
        args.append("--measured-ev-only")
    main(args)
    comparison = {{"__name__": "ce_evaluation"}}
    exec(compile({evaluator!r}, "evaluate_scores.py", "exec"), comparison)
    spectrum_labels = pd.read_parquet(labels)
    measured = comparison["evaluate"](spectrum_labels, pd.read_parquet(candidates), pd.read_parquet(scores))
    measured["cohort"] = "all"
    if "spectrum_group" in spectrum_labels:
        multi_source = spectrum_labels["spectrum_group"].eq("same_source_multi_ce")
        multi_ids = spectrum_labels.loc[multi_source, "molecule_id"]
        multi = measured[measured["molecule_id"].isin(multi_ids)]
        if not multi.empty:
            measured = pd.concat([measured, multi.assign(
                panel="same_source_multi_ce", cohort="same_source_multi_ce",
            )], ignore_index=True)
    summary = comparison["summarize"](measured)
    summary.to_csv("/kaggle/working/ce_comparison.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    paired = paired_ce_summary(measured)
    paired.to_csv("/kaggle/working/ce_paired.csv", index=False)
    print(paired.to_string(index=False), flush=True)
'''
    metadata = json.loads((HERE / "kernel-metadata.json").read_text())
    metadata.update(
        id=f"jiyujia/{slug}", title=slug, enable_gpu=not cpu,
        kernel_sources=[panel_kernel, candidate_kernel],
    )
    if metadata["enable_gpu"]:
        metadata["machine_shape"] = "NvidiaTeslaT4"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / metadata["code_file"]).write_text(scorer + run)
    (destination / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--panel-kernel", default=PANEL)
    parser.add_argument("--candidate-kernel", default=CANDIDATES)
    parser.add_argument("--slug", default=SLUG)
    parser.add_argument("--measured-ev-only", action="store_true")
    args = parser.parse_args()
    build(args.destination, args.cpu, args.panel_kernel, args.candidate_kernel, args.slug,
          args.measured_ev_only)

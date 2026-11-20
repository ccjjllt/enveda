"""Compare each score on fixed candidate lists, including missing answers."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def reorder_formula_group(candidates, scores, acquisitions=None, fuse_ranks=False):
    original = candidates.sort_values("rank")
    candidate_ids = original["candidate_id"].drop_duplicates().tolist()
    if len(candidate_ids) < 2:
        return original, 0
    model_scores = scores[scores["candidate_id"].isin(candidate_ids)].copy()
    if model_scores.empty:
        return original, 0
    model_scores["query_id"] = model_scores["query_id"].fillna("__molecule__")
    matrix = model_scores.pivot_table(index="query_id", columns="candidate_id", values="score")
    matrix = matrix.reindex(columns=candidate_ids)
    complete = matrix.dropna()
    if complete.empty:
        return original, 0
    aggregate = complete.loc["__molecule__"] if "__molecule__" in complete.index else complete.mean()
    if fuse_ranks:
        old_position = pd.Series(range(1, len(candidate_ids) + 1), index=candidate_ids)
        learned_position = aggregate.reindex(candidate_ids).rank(method="min", ascending=False)
        aggregate = 1 / old_position + 1 / learned_position
    reranked = original.assign(score=original["candidate_id"].map(aggregate))
    reranked = reranked.sort_values(["score", "rank"], ascending=[False, True]).drop(columns="score")
    first, proposed = original.iloc[0], reranked.iloc[0]
    if acquisitions is not None and first["rank"] == 1 and first["candidate_id"] != proposed["candidate_id"]:
        margins = (matrix[proposed["candidate_id"]] - matrix[first["candidate_id"]]).dropna()
        observed = acquisitions.reindex(margins.index).dropna()
        distinct_energy = observed.groupby("adduct")["collision_energy"].nunique()
        if not distinct_energy.ge(2).any() or not margins.loc[observed.index].gt(0).all():
            reranked = pd.concat([original.iloc[:1], reranked.drop(index=original.index[0])])
    reranked["rank"] = original["rank"].to_numpy()
    return reranked, len(complete)


def evaluate(labels, candidates, scores, acquisitions=None):
    lists = dict(tuple(candidates.groupby("molecule_id", sort=False)))
    score_groups = dict(tuple(scores.groupby(["model", "molecule_id"], sort=False)))
    empty_scores = scores.iloc[:0]
    policies = [(name, name, None, False) for name in sorted(scores["model"].unique())]
    if "flare" in scores["model"].values:
        policies.append(("flare_rrf", "flare", None, True))
        if acquisitions is not None:
            policies.append(("flare_ce_consensus", "flare", acquisitions, False))
    rows = []
    for label in labels.itertuples(index=False):
        original = lists.get(label.molecule_id, candidates.iloc[:0])
        truth = original.loc[original["candidate_id"].eq(label.candidate_id), "rank"]
        before = int(truth.min()) if len(truth) else None
        for model, source, measured, fuse_ranks in policies:
            local_scores = score_groups.get((source, label.molecule_id), empty_scores)
            groups = original.groupby("formula", dropna=False, sort=False)
            ranked_groups = [
                reorder_formula_group(group, local_scores, measured, fuse_ranks)
                if pd.notna(candidate_formula) else (group, 0)
                for candidate_formula, group in groups
            ]
            records = pd.concat([group for group, _ in ranked_groups]) if ranked_groups else original
            truth = records.loc[records["candidate_id"].eq(label.candidate_id), "rank"]
            after = int(truth.min()) if len(truth) else None
            rows.append(dict(
                model=model, molecule_id=label.molecule_id, panel=label.panel, scaffold=label.scaffold,
                before=before, after=after,
                scored_formula_groups=sum(count > 0 for _, count in ranked_groups),
                rr_before=1 / before if before else 0.0,
                rr_after=1 / after if after else 0.0,
            ))
    if not rows:
        raise ValueError("No model scores available")
    ranks = pd.DataFrame(rows)
    ranks["delta_rr"] = ranks["rr_after"] - ranks["rr_before"]
    if "msg_folds" in labels:
        ranks = ranks.merge(labels[["molecule_id", "msg_folds"]], on="molecule_id", how="left")
    return ranks


def summarize(ranks):
    rng = np.random.default_rng(7502)
    rows = []
    groups = [(panel, model, "all", group) for (panel, model), group in ranks.groupby(["panel", "model"])]
    if "msg_folds" in ranks:
        groups.extend(
            (panel, model, "absent_from_public_msg", group)
            for (panel, model), group in ranks[ranks["msg_folds"].eq("")].groupby(["panel", "model"])
        )
    for panel, model, structures, group in groups:
        clusters = group.groupby("scaffold", dropna=False)["delta_rr"].agg(["sum", "size"])
        size = len(clusters)
        sample = rng.integers(0, size, size=(2000, size))
        boot = clusters["sum"].to_numpy()[sample].sum(axis=1) / clusters["size"].to_numpy()[sample].sum(axis=1)
        low, high = np.quantile(boot, [0.025, 0.975])
        rows.append(dict(
            panel=panel, model=model, structures=structures, molecules=len(group),
            coverage=int(group["scored_formula_groups"].gt(0).sum()),
            recall_at_25=group["before"].notna().mean(),
            mrr_before=group["rr_before"].mean(), mrr_after=group["rr_after"].mean(),
            delta_mrr=group["delta_rr"].mean(),
            gain_rr=group["delta_rr"].clip(lower=0).sum(),
            loss_rr=-group["delta_rr"].clip(upper=0).sum(),
            moved_to_first=int((group["before"].gt(1) & group["after"].eq(1)).sum()),
            lost_first=int((group["before"].eq(1) & group["after"].gt(1)).sum()),
            delta_low=low, delta_high=high,
        ))
    return pd.DataFrame(rows)


def compare_flare_and_mist(ranks):
    models = sorted(ranks["model"].unique())
    flares = [model for model in models if model.lower().startswith("flare")]
    mists = [model for model in models if model.lower().startswith("mist")]
    flare_panels = ranks.loc[ranks["model"].isin(flares), ["panel", "model"]].drop_duplicates()
    for panel, flare in flare_panels.itertuples(index=False, name=None):
        deltas = ranks[ranks["panel"].eq(panel)].pivot(
            index="molecule_id", columns="model", values="delta_rr"
        )
        for mist in mists:
            left, right = deltas[flare], deltas[mist]
            print(
                f"{panel} {flare} × {mist}: "
                f"FLARE-only improvements {int((left.gt(0) & right.le(0)).sum())}, "
                f"MIST-only improvements {int((right.gt(0) & left.le(0)).sum())}, "
                f"joint improvements {int((left.gt(0) & right.gt(0)).sum())}",
                flush=True,
            )


def summarize_formula_proposals(inputs, labels, candidates):
    proposed_path = next(inputs.rglob("formula_predictions.parquet"), None)
    query_path = next(inputs.rglob("queries.parquet"), None)
    if proposed_path is None or query_path is None:
        return
    truth = labels.merge(candidates, on=["molecule_id", "candidate_id"])
    truth = truth[["molecule_id", "formula"]].drop_duplicates("molecule_id")
    queries = pd.read_parquet(query_path, columns=["molecule_id", "query_id"]).merge(truth, on="molecule_id")
    proposed = pd.read_parquet(proposed_path)
    checked = queries.merge(proposed, on=["molecule_id", "query_id"], how="left")
    inferred = int(checked["inferred_formula"].notna().sum())
    matches = int(checked["inferred_formula"].eq(checked["formula"]).sum())
    print(f"Formula matches on spectra whose structure is in the candidates: {matches} of {inferred} inferred, {len(checked)} spectra", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--scores", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args()
    inputs = Path("/kaggle/input")
    if args.labels is None:
        label_paths = list(inputs.rglob("labels.parquet"))
        (args.labels,) = label_paths
    if args.candidates is None:
        candidate_paths = list(inputs.rglob("candidates.parquet"))
        (args.candidates,) = candidate_paths
    if args.scores is None:
        args.scores = sorted(inputs.rglob("scores.parquet"))
    labels = pd.read_parquet(args.labels)
    msg_table = next(inputs.rglob("massspecgym_derived.parquet"), None)
    if msg_table is not None:
        folds = pd.read_parquet(msg_table, columns=["metric_key14", "fold"]).drop_duplicates().dropna()
        folds = folds.groupby("metric_key14")["fold"].agg(lambda values: ",".join(sorted(set(values))))
        labels["msg_folds"] = labels["candidate_id"].map(folds).fillna("")
    candidates = pd.read_parquet(args.candidates)
    if not args.scores:
        raise FileNotFoundError("No scores.parquet inputs")
    scores = pd.concat(map(pd.read_parquet, args.scores), ignore_index=True)
    scores = scores[np.isfinite(scores["score"])]
    query_path = args.labels.with_name("queries.parquet")
    acquisitions = None
    if query_path.is_file():
        acquisitions = pd.read_parquet(query_path, columns=["query_id", "adduct", "collision_energy"])
        acquisitions["collision_energy"] = pd.to_numeric(acquisitions["collision_energy"], errors="coerce")
        acquisitions = acquisitions.set_index("query_id")
    ranks = evaluate(labels, candidates, scores, acquisitions)
    comparison = summarize(ranks)
    ranks.to_parquet(args.output / "molecule_ranks.parquet", index=False)
    comparison.to_csv(args.output / "comparison.csv", index=False)
    print(comparison.to_string(index=False), flush=True)
    summarize_formula_proposals(inputs, labels, candidates)
    compare_flare_and_mist(ranks)


if __name__ == "__main__":
    main()

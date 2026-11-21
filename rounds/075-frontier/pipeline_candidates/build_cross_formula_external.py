"""Run the fixed cross-formula rule on a second MassBank panel."""

import argparse
from inspect import getsource
import json
from pathlib import Path

from build_kernel import build, code_cell
from build_taeyang_cross_formula import promote_cross_formula


SLUG = 'enveda-075b-massbank-202603-cross-formula'
PANEL = 'jiyujia/enveda-075b-external-massbank-202603/1'


def compare_cross_formula(root, panel):
    from tempfile import TemporaryDirectory

    original = pd.read_csv(root / 'submission.csv', dtype=str, keep_default_na=False)
    with TemporaryDirectory(dir='/tmp') as work:
        destination = Path(work) / 'cross_formula.csv'
        status = promote_cross_formula(root, True, destination)
        report = {'rule': status}
        if status['status'] == 'ok':
            revised = pd.read_csv(destination, dtype=str, keep_default_na=False)
            truth = pd.read_parquet(panel / 'labels.parquet', columns=['molecule_id', 'candidate_id'])
            paired = truth.merge(original, on='molecule_id', validate='one_to_one').merge(
                revised, on='molecule_id', validate='one_to_one', suffixes=('_old', '_new'))
            if len(paired) != len(truth) or len(truth) != len(original) or len(original) != len(revised):
                raise ValueError('The two rankings must cover the complete second panel')

            def truth_rank(answer, identity):
                for rank, smiles in enumerate(answer.split(';'), 1):
                    candidate = describe(smiles)
                    if candidate is not None and candidate[0] == identity:
                        return rank
                return 0

            results = [(row.molecule_id, truth_rank(row.smiles_old, row.candidate_id),
                        truth_rank(row.smiles_new, row.candidate_id)) for row in paired.itertuples()]
            n = len(results)
            old_mrr = sum(1 / old for _, old, _ in results if old) / n
            new_mrr = sum(1 / new for _, _, new in results if new) / n
            report.update(molecules=n, mrr_old=old_mrr, mrr_new=new_mrr, mrr_delta=new_mrr - old_mrr,
                          top1_old=sum(old == 1 for _, old, _ in results), top1_new=sum(new == 1 for _, _, new in results),
                          recall_count_old=sum(bool(old) for _, old, _ in results),
                          recall_count_new=sum(bool(new) for _, _, new in results),
                          changed_truth_ranks=[{'molecule_id': mid, 'old': old, 'new': new}
                                               for mid, old, new in results if old != new])
    (root / 'cross_formula_external.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    build(args.destination, PANEL, SLUG)
    path = args.destination / f'{SLUG}.ipynb'
    notebook = json.loads(path.read_text())
    comparison = getsource(promote_cross_formula) + '\n\n' + getsource(compare_cross_formula)
    comparison += "\ncompare_cross_formula(ROOT, Path(PANEL_AUDIT['query_source']))\n"
    compile(comparison, 'cross_formula_comparison', 'exec')
    notebook['cells'].append(code_cell(comparison))
    path.write_text(json.dumps(notebook, ensure_ascii=False) + '\n')

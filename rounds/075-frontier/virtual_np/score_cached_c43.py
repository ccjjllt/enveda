"""Score sealed prefusion C43 comparisons using the authorized private panel."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path

import pyarrow.parquet as pq
from rdkit import RDLogger, rdBase

from evaluate import ORIGINAL_LISTS, baseline
from retrieval_pilot_score import CAP, digest, identified_rank, metric_identity, reciprocal, summarize


ARMS = ('original_slots', 'protected', 'open')
BASELINE_SHA = '05509b32cfb9bcbe8ab563244f91c97e89f878eba3cc40263e3f48fd4a1f372d'
ATTRIBUTION_ARMS = ('historical','native_only','original_slots')
ORIGINAL_SLOTS = {'main':'0c5bb6fbc60dbdebc002045285d1f9e30e2a2ad9ac0f46dba9b797bbd918b874',
                  'c43':'f9bddb8082ea263bbb29fbd95c4548a9de3d0216a6a15aae18011ae103c2940d'}


def checked_json(path, expected):
    value = path.read_bytes()
    if sha256(value).hexdigest() != expected:
        raise ValueError(f'Blind comparison file changed: {path.name}')
    return json.loads(value)


def replacement_inputs(folder, seal, base):
    if (seal['status'] != 'paired_c43_sealed' or seal['native_no_shadow_controls_exact'] != 60
            or not seal['equal_finite_counts']):
        raise ValueError('The matched-budget replacement did not complete')
    observed = checked_json(folder/'c41_forward_observed.json', seal['effective_observed_sha256'])
    delta = checked_json(folder/'forward_delta'/'delta_receipt.json', seal['delta_receipt_sha256'])
    if (set(observed) != set(base['predictor_budget']) or delta['plan_sha256'] != seal['plan_sha256']
            or not delta['all_deltas_observed'] or not delta['equal_finite_counts']):
        raise ValueError('Forward execution differs from the sealed 60-query replacement')
    recomputed = Counter()
    for mid, result in observed.items():
        expected = base['predictor_budget'][mid]
        if set(expected) != {channel+'_candidate_spectra' for channel in result}:
            raise ValueError(f'Mismatched prediction channels for {mid}')
        for channel, lane in result.items():
            if (lane['candidate_spectra'] != expected[channel+'_candidate_spectra']
                    or lane['delta_candidate_spectra'] != lane['observed_delta_candidate_spectra']
                    or not lane['finite_control'] == lane['finite_treatment'] == lane['finite_effective']):
                raise ValueError(f'Unequal candidate-spectra budget for {mid} {channel}')
            recomputed[channel] += lane['observed_delta_candidate_spectra']
    lists = {arm:checked_json(folder/f'{arm}_c43.json',seal[arm]['c43_sha256']) for arm in ('original_slots','replacement')}
    return lists, dict(recomputed_candidate_spectra=dict(recomputed), attempted_slots=seal['attempted'],
                       effective_slots=seal['effective'], equal_finite_counts=True)


def native_only_inputs(folder, seal, base, sources):
    if (seal['status']!='c43_sealed' or seal['query_count']!=60 or seal['queries_sha256']!=base['queries_sha256']
            or seal['candidate_budget']!=CAP or seal['original_t4']!='jiyujia/enveda-075-np-recall-base/2'
            or not seal['original_slots_reference_shadow_archived']
            or any(seal[k]!=60 for k in ('historical_c43_exact','original_slots_c43_exact','original_slots_main_exact'))
            or seal['new_predictor_requests_in_this_job']!=0 or seal['native_only_incremental_forward_requests_vs_public_v29']!=0):
        raise ValueError('Expected the complete native-only attribution with no new forward predictions')
    files=seal['files']
    names={'native_only_request_evidence.json',*(f'{arm}_{kind}.json' for arm in ATTRIBUTION_ARMS for kind in ('main','c43'))}
    if set(files)!=names or any(files['original_slots_'+kind+'.json']!=sha for kind,sha in ORIGINAL_SLOTS.items()):
        raise ValueError('The attribution does not use the independently verified old-slot control')
    arms={}
    ids=set(sources)
    for arm in ATTRIBUTION_ARMS:
        main=checked_json(folder/f'{arm}_main.json',files[f'{arm}_main.json'])
        ranks=checked_json(folder/f'{arm}_c43.json',files[f'{arm}_c43.json'])
        if set(main)!=ids or set(ranks)!=ids or any(not 0<len(ranks[mid])<=CAP for mid in ids):
            raise ValueError(f'Incomplete C43 lists for {arm}')
        if arm=='historical' and any(main[mid][0]!=sources[mid]['main'] or ranks[mid]!=sources[mid]['prefusion'] for mid in ids):
            raise ValueError('Historical source rankings differ from the original T4 archive')
        arms[arm]=ranks
    if any(len(arms['historical'][mid])!=len(arms[arm][mid]) for arm in ATTRIBUTION_ARMS[1:] for mid in ids):
        raise ValueError('Per-query C43 list length differs across attribution arms')
    evidence=checked_json(folder/'native_only_request_evidence.json',files['native_only_request_evidence.json'])
    if set(evidence['cases'])!=ids or set(base['predictor_budget'])!=ids:
        raise ValueError('Prediction evidence covers a different population')
    totals=Counter()
    for mid in ids:
        row=evidence['cases'][mid]
        old=base['predictor_budget'][mid]
        if old['gl_proxy_candidate_spectra']!=0:
            raise ValueError('Unexpected original proxy request')
        for model in ('ice','gl'):
            for mode in ('native','shadow'):
                lane=f'{model}_{mode}'
                if row[lane+'_historical_pairs']!=old[lane+'_candidate_spectra']:
                    raise ValueError(f'Prediction budget differs for {mid}/{lane}')
            if not 0<=row[model+'_shadow_readable_top25_formula_pairs']<=row[model+'_shadow_historical_pairs']:
                raise ValueError('Impossible shadow attribution footprint')
        if (not any(row[m+'_shadow_readable_top25_formula_pairs'] for m in ('ice','gl'))
                and arms['native_only'][mid]!=arms['original_slots'][mid]):
            raise ValueError(f'Shadow changed a C43 list with no readable top-25 formula group for {mid}')
        totals.update(row)
    if dict(totals)!=evidence['totals'] or evidence['totals']!=seal['request_counts']:
        raise ValueError('Prediction totals differ from the blind receipt')
    return arms, dict(new_predictor_requests_in_this_job=0,native_only_incremental_forward_requests_vs_public_v29=0,
                      archived_candidate_spectra=dict(totals)), evidence['cases']


def attribution_measures(group):
    steps={}
    for before,after in (('historical','native_only'),('native_only','original_slots'),('historical','original_slots')):
        paired=[dict(row,baseline_rank=row[f'{before}_rank']) for row in group]
        steps[f'{before}_to_{after}']=summarize(paired,after)
    a=steps['historical_to_native_only']['delta_mrr']
    b=steps['native_only_to_original_slots']['delta_mrr']
    total=steps['historical_to_original_slots']['delta_mrr']
    if abs(a+b-total)>1e-12:
        raise ValueError('Cumulative paired MRR differs from the end-to-end C43 change')
    return dict(mrr25={arm:sum(reciprocal(row[arm+'_rank']) for row in group)/len(group) for arm in ATTRIBUTION_ARMS},
                step_a_delta_mrr=a,step_b_delta_mrr=b,cumulative_delta_mrr=total,comparisons=steps)


def score(args):
    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError('Expected the official competition RDKit')
    RDLogger.DisableLog('rdApp.*')
    native_only = getattr(args,'native_only',False)
    seal_name = ('c41_native_only_receipt.json' if native_only else
                 'c41_study_receipt.json' if args.replacement else 'c41_existing_receipt.json')
    seal_path = args.blind / seal_name
    if digest(seal_path) != args.seal_sha256 or digest(args.baseline) != BASELINE_SHA:
        raise ValueError('The exact blind or historical source changed')
    seal = json.loads(seal_path.read_bytes())
    base, final, sources = baseline(args.baseline)
    if base['queries_sha256'] != args.query_sha256:
        raise ValueError('Query population differs from the official frozen panel')
    if native_only:
        arm_lists,budget,requests_by_query=native_only_inputs(args.blind,seal,base,sources)
    elif args.replacement:
        arm_lists, budget = replacement_inputs(args.blind, seal, base)
    else:
        if (seal['status'] != 'c43_sealed' or seal['molecules'] != 60
                or seal['baseline_zip_sha256'] != BASELINE_SHA or seal['new_predictor_requests'] != 0
                or not seal['original_forward_inputs_unchanged'] or seal['full_final_fusion_scored']
                or seal['original_no_shadow_c43_exact'] != 60 or seal['existing_slots_control_c43_exact'] != 60):
            raise ValueError('The exact-budget C43 experiment did not complete')
        if checked_json(args.blind/'c41_existing_budgets.json',seal['files']['c41_existing_budgets.json']) != base['predictor_budget']:
            raise ValueError('The prediction requests differ from the frozen budget')
        arm_lists = {arm:checked_json(args.blind/f'{arm}_c43.json',seal['files'][f'{arm}_c43.json']) for arm in ARMS}
        budget = dict(recomputed_candidate_spectra={},original_predictor_requests_unchanged=True)
    arms = tuple(arm_lists)
    historical = {mid: value['prefusion'] for mid,value in sources.items()}
    reference = arm_lists[ARMS[0]]
    for arm, candidate_lists in arm_lists.items():
        if set(candidate_lists) != set(sources) or any(len(value) > CAP for value in candidate_lists.values()):
            raise ValueError(f'Unexpected C43 population or slot count for {arm}')
        if native_only or arm == ARMS[0] or args.replacement:
            continue
        changed = {mid for mid, values in candidate_lists.items() if values != reference[mid]}
        top1 = {mid for mid, values in candidate_lists.items() if values[:1] != reference[mid][:1]}
        top5 = {mid for mid, values in candidate_lists.items() if values[:5] != reference[mid][:5]}
        reported = seal['arms'][arm]
        if (changed != set(reported['c43_changed']) or top1 != set(reported['c43_top1_changed'])
                or top5 != set(reported['c43_top5_changed'])):
            raise ValueError(f'Declared C43 changes differ from published lists for {arm}')
    if digest(args.labels) != args.label_sha256:
        raise ValueError('The authorized panel labels changed')
    answers = pq.read_table(args.labels, columns=['molecule_id','truth_smiles','truth_key','source']).to_pylist()
    if len(answers) != 60 or {row['molecule_id'] for row in answers} != set(sources):
        raise ValueError('The authorized panel labels differ from the sealed population')
    rows = []
    for answer in sorted(answers, key=lambda row:row['molecule_id']):
        mid = answer['molecule_id']
        key = metric_identity(answer['truth_smiles'])
        if key is None or key[0] != answer['truth_key']:
            raise ValueError(f'Held-out identity drifted: {mid}')
        missed = not any(identified_rank(seq, key) for seq in [final[mid], *(sources[mid][name] for name in ORIGINAL_LISTS)])
        ranks = {f'{arm}_rank':identified_rank(seq[mid], key) for arm,seq in arm_lists.items()}
        old = identified_rank(historical[mid], key)
        row=dict(molecule_id=mid, source=answer['source'], baseline_rank=old, baseline_sources_missed=missed,
                 historical_final_rank=identified_rank(final[mid], key), **ranks)
        if native_only:
            if old!=row['historical_rank']:raise ValueError(f'Historical rank mismatch for {mid}')
            h,n,s=(reciprocal(row[a+'_rank']) for a in ATTRIBUTION_ARMS)
            row.update(step_a_delta_rr=n-h,step_b_delta_rr=s-n,cumulative_delta_rr=s-h)
            for model in ('ice','gl'):
                name=model+'_shadow_readable_top25_formula_pairs'
                row[name]=requests_by_query[mid][name]
        rows.append(row)

    def measures(group):
        if native_only:
            return attribution_measures(group)
        paired = [dict(row, baseline_rank=row['original_slots_rank']) for row in group]
        return dict(versus_historical_c43={arm:summarize(group, arm) for arm in arms},
                    versus_reused_original_slots={arm:summarize(paired, arm) for arm in arms[1:]})

    summary = dict(scope='C43 prefusion, no final fusion ranking evaluated', queries_sha256=args.query_sha256,
                   blind_seal_sha256=args.seal_sha256, prediction_budget=budget,
                   historical_c43_mrr=sum(reciprocal(row['baseline_rank']) for row in rows)/len(rows),
                   original_sources_misses=sum(row['baseline_sources_missed'] for row in rows),
                   empty_slots_by_arm={arm:sum(CAP-len(values) for values in rankings.values()) for arm,rankings in arm_lists.items()},
                   total=measures(rows), by_source={source:measures([r for r in rows if r['source'] == source])
                                                    for source in sorted({r['source'] for r in rows})})
    if native_only:
        coverage={}
        for row in rows:
            bucket=('shadow_readable' if any(row[m+'_shadow_readable_top25_formula_pairs'] for m in ('ice','gl'))
                    else 'no_readable_shadow')
            coverage.setdefault(bucket,[]).append(row)
        summary['by_shadow_coverage']={bucket:measures(group) for bucket,group in coverage.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'cached_c43_metrics.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (args.out/'cached_c43_per_query.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary['total']),flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('blind','baseline','labels','out'):
        cli.add_argument('--'+name, type=Path, required=True)
    for name in ('seal-sha256','label-sha256','query-sha256'):
        cli.add_argument('--'+name, required=True)
    modes=cli.add_mutually_exclusive_group()
    modes.add_argument('--replacement', action='store_true')
    modes.add_argument('--native-only', action='store_true')
    score(cli.parse_args())

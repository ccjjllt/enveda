"""Apply the fixed top-five rule on the Enveda-180 baseline's own archived native requests."""

from collections import Counter
import csv
from io import StringIO
import json
from pathlib import Path

import pyarrow.parquet as pq
from rdkit import rdBase

from frozen_top5 import apply, dump, hash_value, verify


PREDICTION = 'jiyujia/enveda-075-first-formula-baseline-e180/2'
SUBMISSION_SHA = '2675b3cec0607a93b447c4bfbcac256812bc796cf8794b81fc8414c24fe1f55a'
EVIDENCE_SHA = '3b0895aee03299edda67599e255ef77dbd2311d5e82823a2418165b000065f59'
RANKING_SHA = '4311f3d653ed29dd598a4c55297a73c49c1ee38f170ee238bb5b6e459dc14a9f'
QUERY_SHA = '6fe782c57481b6fea95d1197da05ce4968c393881e3d962bcf8cdaed09b3bf7b'
RULE_SHA = '2a3b4ee7c93355eed8835ac3755a3d045601646e4ca80a58dd71b30368fcec8b'


def run(root, out, native_hashes):
    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError('The frozen rule requires contest RDKit 2026.03.3')
    hashes = {'submission.csv':SUBMISSION_SHA, 'e180_ranked_input.json':RANKING_SHA,
              'frozen_candidate_evidence.parquet':EVIDENCE_SHA} | native_hashes
    for name, expected in hashes.items():
        verify((root/name).read_bytes(), expected)
    receipt = json.loads((root/'e180_ranked_input.json').read_bytes())
    inference = receipt['inference']
    if (receipt['arm']!='baseline' or receipt['submission_sha256']!=SUBMISSION_SHA
            or inference['sha256']['test.parquet']!=QUERY_SHA
            or (inference['molecules'], inference['spectra'])!=(121,299)
            or any(x!='Tesla T4' for x in receipt['cuda_devices']) or not receipt['cuda_devices']):
        raise ValueError('Archived query and original final run do not agree')
    rows = list(csv.DictReader(StringIO((root/'submission.csv').read_text())))
    ranks = {row['molecule_id']:row['smiles'].split(';') for row in rows}
    if len(rows)!=121 or len(ranks)!=121 or any(len(v)!=25 for v in ranks.values()):
        raise ValueError('Expected the original 121 final 25-candidate lists')
    evidence = pq.read_table(root/'frozen_candidate_evidence.parquet',
                             columns=['molecule_id','library_max','processed','proxy_gl']).to_pylist()
    ledgers = {}
    for row in evidence:
        mid = row['molecule_id']
        value = row['library_max']
        if (row['processed'] is not True or row['proxy_gl'] is not False or value is None
                or ledgers.setdefault(mid, {'library_max':value})['library_max']!=value):
            raise ValueError('Ambiguous native E1 library strength, skipped query or proxy GL score')
    if set(ledgers)!=set(ranks):
        raise ValueError('Incomplete E1 library coverage for the frozen baseline')
    request, scores, model_meta = {}, {}, {}
    budget = {'cases': {m:{'predictor_budget':{}} for m in ranks}}
    for model in ('ice','gl'):
        def read(suffix):
            return json.loads((root/f'{model}_work'/(model+suffix)).read_bytes())
        items = read('_in.json')
        request[model] = {r['mid']:r for r in items}
        scores[model] = read('_out.json')
        meta = read('_out.json.meta.json')
        if (len(items)!=len(request[model]) or set(request[model])-ranks.keys()
                or set(scores[model])-request[model].keys() or meta['status']!='ok' or meta['device']!='cuda'):
            raise ValueError(f'{model} original request/population does not match its native output')
        for mid in ranks:
            item = request[model].get(mid,{})
            reps, scans = item.get('cands',[]), item.get('spectra',[])
            if len(reps)!=len(set(reps)) or set(scores[model].get(mid,{}))-set(reps):
                raise ValueError(f'{model} scored a different SMILES representation or duplicated requests')
            budget['cases'][mid]['predictor_budget'][model+'_native_candidate_spectra'] = len(reps)*len(scans)
        model_meta[model] = dict(device=meta['device'], contexts=len(items),
                                supplied_smiles=sum(len(r['cands']) for r in items),
                                archived_candidate_spectra=sum(case['predictor_budget'][model+'_native_candidate_spectra']
                                                              for case in budget['cases'].values()))
    outputs, cases = apply(ranks, ledgers, request, scores, budget)
    out.mkdir(parents=True,exist_ok=True)
    files = {f'{arm}_final.json':dump(out/f'{arm}_final.json',mapping) for arm,mapping in outputs.items()}
    files['final_native_top5_e180_cases.json'] = dump(out/'final_native_top5_e180_cases.json',cases)
    arms = {}
    for arm in ('protected','open'):
        changed = outputs[arm]
        if any(Counter(ranks[m])!=Counter(changed[m]) or Counter(ranks[m][:5])!=Counter(changed[m][:5])
               or ranks[m][5:]!=changed[m][5:] or (arm=='protected' and ranks[m][0]!=changed[m][0]) for m in ranks):
            raise ValueError('Final membership, fixed head, or ranks 6–25 changed')
        arms[arm] = dict(eligible=[m for m,r in cases.items() if r['arms'][arm]['comparable_pairs']],
                         changed=[m for m in ranks if ranks[m]!=changed[m]],
                         first_changed=[m for m in ranks if ranks[m][0]!=changed[m][0]],
                         total_comparable_pairs=sum(r['arms'][arm]['comparable_pairs'] for r in cases.values()),
                         total_adjacent_swaps=sum(len(r['arms'][arm]['changes']) for r in cases.values()))
    seal = dict(status='sealed', arm_names=list(outputs), molecules=121, measured_spectra=299,
                candidate_budget=25, actual_slots=sum(map(len,ranks.values())), new_predictor_requests=0,
                baseline=PREDICTION, base_submission_sha256=SUBMISSION_SHA, queries_sha256=QUERY_SHA,
                source_ranking_receipt_sha256=RANKING_SHA, source_library_threshold=0.9,
                source_60_rule_sha256=RULE_SHA, input_file_sha256=hashes, files=files, arms=arms,
                historical_final_membership_sha256=hash_value({m:sorted(v) for m,v in ranks.items()}),
                native_forward_archives=model_meta, weak_library_cases=sum(r['library_max']<0.9 for r in cases.values()),
                common_nonempty_request_cases=sum(r['same_nonempty_native_spectra'] for r in cases.values()),
                rdkit=rdBase.rdkitVersion)
    dump(out/'final_native_top5_e180_receipt.json',seal)
    print('121 original final native comparison',
          {a:{k:len(v) if isinstance(v,list) else v for k,v in d.items()} for a,d in arms.items()},flush=True)

"""Permute same-formula slots in the original final five using only exact archived native scores."""

from collections import Counter, defaultdict
import csv
from functools import lru_cache
from hashlib import sha256
from io import StringIO
import json
import math
from pathlib import Path
from zipfile import ZipFile

from rdkit import Chem, rdBase
from rdkit.Chem.rdMolDescriptors import CalcMolFormula


BASE_REF = 'jiyujia/enveda-075-np-recall-reseal/2'
SNAP_REF = 'jiyujia/enveda-075-c39-snapshot-t4/1'
ORIGINAL = 'jiyujia/enveda-075-np-recall-base/2'
BASE_SHA = '05509b32cfb9bcbe8ab563244f91c97e89f878eba3cc40263e3f48fd4a1f372d'
SNAP_SHA = '30ebaeda090adbabd48f0b60ef4a3a857275c12be84f0c37a58f1bd8fd6ed7dc'
CONTROL_SHA = '7aa006825a0f8654c4239f25e89ddb490ea04575b9a195d6b744955688d5d7c5'
QUERY_SHA = '2a8bf747f50e6920fcd4f24e04302b721281f8794c947cc58fe545264b12d0d4'
LIBRARY_THRESHOLD = 0.9


def verify(data, expected):
    if sha256(data).hexdigest() != expected:
        raise ValueError('The input SHA-256 differs from its original seal')
    return data


def dump(path, obj):
    blob = (json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n').encode()
    path.write_bytes(blob)
    return sha256(blob).hexdigest()


def hash_value(obj):
    return sha256(json.dumps(obj, ensure_ascii=False, allow_nan=False, sort_keys=True,
                             separators=(',', ':')).encode()).hexdigest()


@lru_cache(maxsize=None)
def formula(smiles):
    try:
        molecule = Chem.MolFromSmiles(smiles)
        return CalcMolFormula(molecule) if molecule is not None else None
    except (ValueError, RuntimeError, TypeError):
        return None


def original_inputs(snapshot, baseline):
    seal = json.loads(verify((snapshot/'c39_snapshot_receipt.json').read_bytes(), SNAP_SHA))
    if (seal['status'] != 'exact' or not seal['cache_reuse_supported'] or seal['device'] != 'cuda'
            or seal['original'] != ORIGINAL):
        raise ValueError('The T4 rerun did not reproduce the original C39 and C41 inputs')
    def snap_file(name):
        return json.loads(verify((snapshot/name).read_bytes(), seal['hashes'][name]))
    ledgers = snap_file('c39_native_ledgers.json')
    proof = snap_file('c39_request_verification.json')
    if not proof['exact'] or proof['original'] != ORIGINAL:
        raise ValueError('Original native predictor inputs did not match')
    hashes = {}
    def member(archive, name):
        data = archive.read(name)
        hashes[name] = sha256(data).hexdigest()
        return data
    cache = snapshot/'c39_original_control.zip'
    verify(cache.read_bytes(), CONTROL_SHA)
    with ZipFile(cache) as z:
        request, score = {}, {}
        for model in ('ice','gl'):
            read = proof['request_files'][model+'_native']
            if not read['exact'] or read['original_canonical_sha256'] != read['rebuilt_sha256']:
                raise ValueError(f'{model} request provenance mismatch')
            rows = json.loads(verify(member(z, f'{model}_work/{model}_in.json'), read['file_sha256']))
            request[model] = {row['mid']:row for row in rows}
            score[model] = json.loads(member(z, f'{model}_work/{model}_out.json'))
            if len(request[model]) != len(rows) or set(score[model])-set(request[model]):
                raise ValueError(f'{model} has duplicate/missing original request contexts')
            for mid, row in request[model].items():
                reps = row['cands']
                if len(set(reps)) != len(reps) or set(score[model].get(mid,{}))-set(reps):
                    raise ValueError(f'{model} has duplicated representatives or a score outside its request')
        budget = json.loads(member(z, 'pilot_forward_budget.json'))
    verify(baseline.read_bytes(), BASE_SHA)
    with ZipFile(baseline) as z:
        historical = json.loads(member(z,'pilot_arm_receipt.json'))
        if (historical['arm'] != 'baseline' or historical['queries_sha256'] != QUERY_SHA
                or historical['molecules'] != 60 or historical['candidate_slots'] != 25):
            raise ValueError('Expected the 60-query original final-fusion seal')
        csv_data = verify(member(z,'submission.csv'), historical['hashes']['submission.csv'])
        reader = csv.DictReader(StringIO(csv_data.decode()))
        if reader.fieldnames != ['molecule_id','smiles']:
            raise ValueError('Invalid final submission columns')
        records = list(reader)
        ranks = {row['molecule_id']:row['smiles'].split(';') if row['smiles'] else [] for row in records}
    if (len(records) != 60 or set(ranks) != set(ledgers) or set(ranks) != set(budget['cases'])
            or budget['arm'] != 'baseline' or budget['device'] != 'Tesla T4'
            or any(len(v)>25 or len(v)!=len(set(v)) for v in ranks.values())):
        raise ValueError('Historical exact candidate lists or original predictor budget disagree')
    return ranks, ledgers, request, score, budget, hashes | {n:seal['hashes'][n] for n in
                  ('c39_native_ledgers.json','c39_request_verification.json')}


def native_evidence(mid, head, request, score, budget):
    evidence = {}
    for model in ('ice','gl'):
        item = request[model].get(mid, {})
        reps, scans = item.get('cands',[]), item.get('spectra',[])
        if len(reps)*len(scans) != budget[model+'_native_candidate_spectra']:
            raise ValueError(f'{mid} {model} archived request budget differs')
        raw = score[model].get(mid, {})
        values = {}
        for s in head:
            value = raw.get(s)
            if value is not None:
                if s not in reps or isinstance(value, bool) or not math.isfinite(value):
                    raise ValueError('Invalid score or nonliteral candidate in the native archive')
                values[s] = value
        evidence[model] = dict(spectrum_sha256=hash_value(scans) if scans else None,
                               historical_candidate_spectra=len(reps)*len(scans), values=values)
    paired = (evidence['ice']['spectrum_sha256'] is not None and
              evidence['ice']['spectrum_sha256'] == evidence['gl']['spectrum_sha256'])
    return evidence, paired


def permutation(original, groups, scores, allow_first):
    ranked, moves = list(original), []
    comparable_pairs = dominant_pairs = 0
    for name, all_positions in groups.items():
        positions = [i for i in all_positions if allow_first or i != 0]
        for j, i in enumerate(positions):
            for k in positions[:j]:
                a, b = original[i], original[k]
                if a in scores and b in scores:
                    comparable_pairs += 1
                    dominant_pairs += int(all(new > old for new, old in zip(scores[a], scores[b], strict=True)))
            # Stable insertion across same-formula positions. A missing score, tie or model disagreement
            # keeps that pair in its incoming order and prevents crossing the same-formula incumbent.
            for cursor in range(j, 0, -1):
                left, right = positions[cursor - 1], positions[cursor]
                challenger, incumbent = ranked[right], ranked[left]
                if challenger not in scores or incumbent not in scores or not all(
                        x > y for x,y in zip(scores[challenger],scores[incumbent],strict=True)):
                    break
                ranked[left], ranked[right] = challenger, incumbent
                moves.append(dict(formula=name, earlier_rank=left+1, later_rank=right+1,
                                  promoted=challenger, displaced=incumbent))
    if (len(original) != len(ranked) or set(original[:5]) != set(ranked[:5]) or original[5:] != ranked[5:]
            or (not allow_first and original[:1] != ranked[:1])):
        raise ValueError('Local swap changed an excluded rank or changed a candidate')
    return ranked, dict(comparable_pairs=comparable_pairs, strictly_dominant_pairs_in_original_order=dominant_pairs,
                        changes=moves)


def run(snapshot, baseline, out):
    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError('The original competition formula calculation requires RDKit 2026.03.3')
    ranks, ledgers, request, raw_scores, budget, input_hashes = original_inputs(snapshot, baseline)
    outputs = {'historical':ranks, 'protected':{}, 'open':{}}
    cases = {}
    for mid, original in ranks.items():
        stage = ledgers[mid]
        strength = stage['library_max']
        if not math.isfinite(strength):
            raise ValueError(f'Nonfinite E1 library strength: {mid}')
        evidence, paired = native_evidence(mid, original[:5], request, raw_scores, budget['cases'][mid]['predictor_budget'])
        head = [dict(rank=i+1,smiles=s,formula=formula(s),ice=evidence['ice']['values'].get(s),
                     gl=evidence['gl']['values'].get(s)) for i,s in enumerate(original[:5])]
        groups = defaultdict(list)
        scores = {}
        for row in head:
            if row['formula'] and strength < LIBRARY_THRESHOLD and paired:
                groups[row['formula']].append(row['rank']-1)
                if row['ice'] is not None and row['gl'] is not None:
                    scores[row['smiles']] = (row['ice'],row['gl'])
        info = {}
        for arm in ('protected','open'):
            ranked, info[arm] = permutation(original, groups, scores, allow_first=arm=='open')
            outputs[arm][mid] = ranked
        cases[mid] = dict(library_max=strength, same_nonempty_native_spectra=paired, raw_native_head=head,
                          archived_spectra={m:{k:v for k,v in summary.items() if k!='values'}
                                            for m,summary in evidence.items()}, arms=info)
    out.mkdir(exist_ok=True, parents=True)
    files = {f'{name}_final.json':dump(out/f'{name}_final.json', rows) for name,rows in outputs.items()}
    files['final_native_top5_cases.json'] = dump(out/'final_native_top5_cases.json', cases)
    arms = {}
    for arm in ('protected', 'open'):
        after = outputs[arm]
        arms[arm] = dict(eligible=[mid for mid,r in cases.items() if r['arms'][arm]['comparable_pairs']],
                         total_comparable_pairs=sum(r['arms'][arm]['comparable_pairs'] for r in cases.values()),
                         total_adjacent_swaps=sum(len(r['arms'][arm]['changes']) for r in cases.values()),
                         changed=[m for m in ranks if ranks[m]!=after[m]],
                         first_changed=[m for m in ranks if ranks[m][:1]!=after[m][:1]],
                         top5_members_changed=[m for m in ranks if set(ranks[m][:5])!=set(after[m][:5])],
                         final_members_changed=[m for m in ranks if set(ranks[m])!=set(after[m])])
    seal = dict(status='sealed', arm_names=list(outputs), molecules=len(ranks), queries_sha256=QUERY_SHA,
                source_original=ORIGINAL, baseline=BASE_REF, baseline_archive_sha256=BASE_SHA,
                snapshot=SNAP_REF, snapshot_receipt_sha256=SNAP_SHA, original_cache_sha256=CONTROL_SHA,
                files=files, input_file_sha256=input_hashes, candidate_budget=25,
                actual_slots=sum(map(len,ranks.values())), original_lengths=dict(Counter(map(len,ranks.values()))),
                historical_final_membership_sha256=hash_value({m:sorted(v) for m,v in ranks.items()}),
                new_predictor_requests=0, scores_used='literal native C41 ICEBERG and GLACIER outputs',
                source_library_threshold=LIBRARY_THRESHOLD, weak_library_cases=sum(r['library_max']<LIBRARY_THRESHOLD for r in cases.values()),
                common_nonempty_request_cases=sum(r['same_nonempty_native_spectra'] for r in cases.values()),
                rdkit=rdBase.rdkitVersion, arms=arms)
    dump(out/'final_native_top5_receipt.json',seal)
    print('Final-five native order', {arm:{k:len(v) if isinstance(v,list) else v for k,v in a.items()}
                                      for arm,a in arms.items()}, flush=True)

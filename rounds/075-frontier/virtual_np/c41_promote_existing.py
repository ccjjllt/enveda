"""One blind, same-formula C41 position swap using only archived E1 structures and forward predictions."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math


MODELS = ('ice','gl')
PROTECTED_SLOTS = (5,4,3,2)


def spectrum_group(spectra):
    raw = json.dumps(spectra, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    return sha256(raw).hexdigest()


def forward_index(requests, scores, mids):
    """Index by predictor/query/literal representative/complete ordered spectrum group."""
    result = {model:defaultdict(dict) for model in MODELS}
    budgets = {mid:{} for mid in mids}
    for model in MODELS:
        for stage in ('native','shadow'):
            channel = model+'_'+stage
            requested = requests[channel]
            by_id = {r['mid']:r for r in requested}
            scored = scores[model][stage]
            if len(by_id)!=len(requested) or set(by_id)-mids or set(scored)-set(by_id):
                raise ValueError(f'{channel} cache disagrees with the original requests')
            for mid in mids:
                item = by_id.get(mid,{})
                smiles, spectra = item.get('cands',[]), item.get('spectra',[])
                row = scored.get(mid,{})
                if len(set(smiles))!=len(smiles) or set(row)-set(smiles):
                    raise ValueError(f'Unrequested prediction or duplicate {channel} representative for {mid}')
                budgets[mid][channel] = len(smiles)*len(spectra)
                if not spectra:
                    continue
                group = spectrum_group(spectra)
                for s in smiles:
                    value = row.get(s)
                    if value is None:
                        continue
                    if isinstance(value,bool) or not math.isfinite(value):
                        raise ValueError(f'Invalid archived {channel} score for {mid}')
                    entries = result[model][mid,s]
                    if group in entries and entries[group] != value:
                        raise ValueError(f'Conflicting {model} results for identical literal inputs in {mid}')
                    entries[group] = value
    return result,budgets


def candidate_row(obj):
    if hasattr(obj,'_asdict'):
        return obj._asdict()
    return obj


def compare_slot(mid, challenger, incumbent, evidence):
    comparison = {}
    for model in MODELS:
        new = evidence[model].get((mid,challenger),{})
        old = evidence[model].get((mid,incumbent),{})
        common = new.keys() & old.keys()
        # Multiple experimental groups without a prespecified single common group are ambiguous.
        if len(common)!=1:
            return None
        group, = common
        comparison[model] = dict(spectrum_group_sha256=group,challenger=new[group],incumbent=old[group],
                                 strictly_higher=new[group]>old[group])
    return comparison


def slot_opportunity(mid, positions, sm, formulas, challenger, formula, evidence):
    details = []
    for rank in positions:
        if rank>len(sm) or formulas[rank-1]!=formula:
            continue
        scores = compare_slot(mid,challenger,sm[rank-1],evidence)
        passed = scores is not None and all(v['strictly_higher'] for v in scores.values())
        details.append(dict(slot=rank,incumbent_smiles=sm[rank-1],physical=scores,passes=passed))
    winner = next((r['slot'] for r in details if r['passes']),None)
    return winner,details


def inspect_candidate(row,mid,stage,old_slate,stock,lookup,control_ids,formulas,evidence,identity,allow_rank1):
    tagged = row['smiles']
    ident = identity(tagged)
    if ident is None or (row.get('score_key') is not None and row['score_key']!=ident[0]):
        raise ValueError(f'Unexpected shortlist metric identity for {mid}')
    matches = lookup.get(ident,[])
    positions = control_ids.get(ident,[])
    item = dict(fpnet_rank=int(row['rank']),fpnet_score=float(row['fpnet_a_fz']),tag_smiles=tagged,
                old_e1_matches=len(matches),control_ranks=positions,weak_library=stage['library_max']<0.9,
                old_e1_rank=None,deep_only=False,representative=None,tag_alias=False,
                finite_models=[],first_slot=None,protected_slot=None,first_checks=[],protected_checks=[])
    if len(matches)!=1:
        item['reason']='absent_from_old_e1' if not matches else 'ambiguous_old_e1_identity'
        return item
    ix = matches[0]
    rec = stock[ix]
    sm = rec['smiles']
    present = [name for name in MODELS if evidence[name].get((mid,sm))]
    item.update(old_e1_rank=ix+1,deep_only=ix>=len(stage['main']),representative=sm,
                tag_alias=sm!=tagged,finite_models=present,native_formula=rec['native_formula'])
    if not item['weak_library']:
        item['reason']='strong_library'
    elif len(positions)>1 or (positions and old_slate[0][positions[0]-1]!=sm):
        item['reason']='metric_identity_already_present_as_other_representative'
    elif positions==[1]:
        item['reason']='already_first'
    elif len(present)!=len(MODELS):
        item['reason']='missing_archived_physical_score_for_exact_e1_representative'
    else:
        original_rank = positions[0] if positions else 0
        if allow_rank1:
            item['first_slot'],item['first_checks'] = slot_opportunity(
                mid,(1,),old_slate[0],formulas,sm,rec['native_formula'],evidence)
        if not original_rank or original_rank>5:
            item['protected_slot'],item['protected_checks'] = slot_opportunity(
                mid,PROTECTED_SLOTS,old_slate[0],formulas,sm,rec['native_formula'],evidence)
        checks = item['first_checks'] + item['protected_checks']
        if item['first_slot'] or item['protected_slot']:
            item['reason']='eligible'
        elif original_rank and original_rank<=5 and not allow_rank1:
            item['reason']='already_in_top5'
        elif not checks:
            item['reason']='no_same_formula_reserved_position'
        elif not any(r['physical'] is not None for r in checks):
            item['reason']='no_pair_with_matching_physical_observations'
        else:
            item['reason']='not_strictly_better_in_both_models'
    return item


def promote(control_main,native,shortlist,requests,scores,identity,*,allow_rank1=False):
    """Return post-C41 main lists and a blind case/row audit. The input control and caches are immutable.

    Each control value is (smiles, native_keys, library_max). Each native ledger has main, shadow_pool
    and library_max. Shortlist rows use the original FPNet A rank/score/SMILES from read_shortlist.
    Requests map ice_native/gl_native/ice_shadow/gl_shadow to their original runner input lists.
    Scores map model -> native/shadow -> query -> exact input SMILES -> measured score or None.
    identity is the competition score-key plus heavy-element composition function.
    """
    mids = set(native)
    if mids!=set(shortlist) or mids!=set(control_main):
        raise ValueError('The old-slot control and blind shortlist must cover the same queries')
    evidence,budgets = forward_index(requests,scores,mids)
    treated,cases,swaps = {},{},[]
    totals = Counter()
    for mid in control_main:
        old_slate = control_main[mid]
        old_sm,old_keys,strength = old_slate
        stage = native[mid]
        stock = stage['main'] + stage['shadow_pool']
        if (strength!=stage['library_max'] or not math.isfinite(strength)
                or len(old_sm)!=len(old_keys) or len(old_sm)>25 or len(set(old_keys))!=len(old_keys)):
            raise ValueError(f'Invalid control slate or library score for {mid}')
        by_rep = {(r['smiles'],r['native_key']):r for r in stock}
        if len(by_rep)!=len(stock) or not all((s,k) in by_rep for s,k in zip(old_sm,old_keys)):
            raise ValueError(f'Some old main entries do not match original E1 representatives and keys: {mid}')
        formulas = [by_rep[s,k]['native_formula'] for s,k in zip(old_sm,old_keys)]
        lookup,control_ids = defaultdict(list),defaultdict(list)
        for ix,rec in enumerate(stock):
            ident=identity(rec['smiles'])
            if ident is not None:
                lookup[ident].append(ix)
        for i,s in enumerate(old_sm,1):
            control_ids[identity(s)].append(i)
        ranked = [candidate_row(r) for r in shortlist[mid]]
        if ([int(r['rank']) for r in ranked]!=list(range(1,len(ranked)+1)) or len(ranked)>24
                or any(not math.isfinite(r['fpnet_a_fz']) for r in ranked)
                or len({identity(r['smiles']) for r in ranked})!=len(ranked)):
            raise ValueError(f'Invalid rank order, scores or duplicate identities in frozen shortlist: {mid}')
        examined = [inspect_candidate(r,mid,stage,old_slate,stock,lookup,control_ids,formulas,evidence,identity,allow_rank1)
                    for r in ranked]
        # Open arm first considers all candidates for #1; otherwise it exactly recovers the protected selection.
        chosen = next((r for r in examined if r['first_slot'] is not None),None)
        if chosen is None:
            chosen = next((r for r in examined if r['protected_slot'] is not None),None)
        new_sm,new_keys = list(old_sm),list(old_keys)
        if chosen:
            target = chosen['first_slot'] or chosen['protected_slot']
            oldrank = chosen['control_ranks'][0] if chosen['control_ranks'] else None
            rec = stock[chosen['old_e1_rank']-1]
            j = target-1
            if oldrank is not None:
                new_sm[oldrank-1],new_keys[oldrank-1] = old_sm[j],old_keys[j]
            new_sm[j],new_keys[j] = rec['smiles'],rec['native_key']
            if (len(set(new_keys))!=len(new_keys)
                    or sum(identity(s)==identity(rec['smiles']) for s in new_sm)!=1
                    or [by_rep[s,k]['native_formula'] for s,k in zip(new_sm,new_keys)]!=formulas):
                raise ValueError(f'Promotion changed a formula slot or duplicated an identity for {mid}')
            swaps.append(dict(molecule_id=mid,slot=target,fpnet_rank=chosen['fpnet_rank'],
                              native_e1_rank=chosen['old_e1_rank'],prior_control_rank=oldrank,
                              challenger=rec['smiles'],incumbent=old_sm[j],formula=rec['native_formula']))
            chosen['reason']='promoted'
            for other in examined:
                if other['reason']=='eligible':
                    other['reason']='eligible_but_one_position_already_used'
        treated[mid]=[new_sm,new_keys,strength]
        counts = Counter(r['reason'] for r in examined)
        covers = dict(shortlisted=len(examined),matched_unique_e1=sum(r['old_e1_matches']==1 for r in examined),
                      alias_tags=sum(r['tag_alias'] for r in examined),deep_only=sum(r['deep_only'] for r in examined),
                      both_finite=sum(len(r['finite_models'])==2 for r in examined),
                      original_top5=sum(any(i<=5 for i in r['control_ranks']) for r in examined),
                      first_opportunities=sum(r['first_slot'] is not None for r in examined),
                      protected_opportunities=sum(r['protected_slot'] is not None for r in examined),
                      changed=int(chosen is not None))
        cases[mid] = dict(candidate_counts=covers,reasons=dict(counts),predictor_budget=budgets[mid],rows=examined)
        totals.update(covers)
    audit = dict(allow_rank1=allow_rank1,queries=len(mids),new_forward_requests=0,
                 positions=([1] if allow_rank1 else [])+list(PROTECTED_SLOTS),
                 reserve_first_before_protected=allow_rank1,counts=dict(totals),cases=cases,swaps=swaps,
                 changed_molecule_ids=[s['molecule_id'] for s in swaps])
    return treated,audit

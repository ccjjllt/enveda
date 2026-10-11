"""Manual fixed-rule same-pool missing-evidence diagnostic; no training or parameter scan."""
import json,math,statistics,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
paths={'fiora':'artifacts/006-fiora/fiora_scores_22.json','ice':'artifacts/f1-scores/scores/f1e-4_ice_top100_split.json','inputs':'artifacts/f1-bundle/inputs.json','orig':'artifacts/007/fiora22-evidence.json','remove':'artifacts/007-r2/fiora_tailrule_compare.json'}
d={k:json.loads((root/p).read_text()) for k,p in paths.items()}
ms={m['key']:m for m in d['inputs']['molecules']}
orig={m['key']:m for m in d['orig']['per_molecule']}
def valid(v):return isinstance(v,(int,float)) and math.isfinite(v)
def z(a):
 f=[v for v in a if valid(v)]; sd=statistics.stdev(f) if len(f)>1 else 0; mu=statistics.mean(f) if f else 0
 return [(v-mu)/sd if valid(v) and sd>0 else 0.0 for v in a]
def rr(r):return 1/r if r and r<=25 else 0
rows=[];mismatch=[];zero_bad=[]
for q,fc in d['fiora'].items():
 m=ms[q];ic=d['ice'][q];truth=m['truth_canon'];fm={ic[s]['canon']:v for s,v in fc.items()};keys=[k for k in m['order_top100'] if k in fm];assert keys==m['order_top100'][:50]
 b=[m['base_scores_full_window'][k] for k in keys];s=[fm[k]['similarity'] if fm[k]['status']=='scored' and valid(fm[k]['similarity']) else None for k in keys];bz=z(b);sz=z(s);score=[x+y for x,y in zip(bz,sz)]
 neutral=[keys[i] for i in sorted(range(len(keys)),key=lambda i:(-score[i],i))]
 ok=[i for i in range(len(keys)) if valid(s[i])];bad=[i for i in range(len(keys)) if not valid(s[i])]
 tail=[keys[i] for i in sorted(ok,key=lambda i:(-score[i],i))+sorted(bad,key=lambda i:(-b[i],i))];rem=tail[:len(ok)]
 def rk(o):return o.index(truth)+1 if truth in o else None
 ranks={'base':rk(keys),'tail':rk(tail),'remove':rk(rem),'neutral':rk(neutral)}
 if ranks['tail']!=orig[q]['ranks']['base_plus_fiora50']:mismatch.append(q)
 rows.append({'key':q,'truth_status':fm[truth]['status'],'n_candidates':len(keys),'n_scored':len(ok),'ranks':ranks,'rr':{k:rr(v) for k,v in ranks.items()},'neutral_top25':neutral[:25]})
 for c in fc.values():
  if c['status']=='scored' and ((c['similarity']==0)!=(c['n_matched']==0)):zero_bad.append(q)
def metrics(a):
 return {m:{'n':len(a),'mrr25':statistics.mean(r['rr'][m] for r in a) if a else None,'top1':sum(r['ranks'][m]==1 for r in a),'top25':sum(r['rr'][m]>0 for r in a)} for m in ['base','tail','remove','neutral']}
def delta(a,cmp):
 ds=[r['rr']['neutral']-r['rr'][cmp] for r in a];return {'delta_mrr25':statistics.mean(ds),'improved':sum(v>1e-12 for v in ds),'worsened':sum(v< -1e-12 for v in ds),'tied':sum(abs(v)<=1e-12 for v in ds)}
out={'caliber':'One fixed same-candidate-pool diagnostic: score=z(base over all candidates)+z(FIORA over finite scored candidates); failed candidate FIORA contribution=0; rank all original candidates together, no scored-first promotion or deletion. ddof=1; ties preserve original base order. No training or weight scan.','sources':{k:{'path':p,'sha256':hashlib.sha256((root/p).read_bytes()).hexdigest()} for k,p in paths.items()},'checks':{'mapped_candidates':sum(r['n_candidates'] for r in rows),'tail_rank_mismatches':mismatch,'zero_similarity_vs_zero_matched_mismatch':zero_bad},'metrics':metrics(rows),'neutral_vs_base':delta(rows,'base'),'neutral_vs_tail':delta(rows,'tail'),'subsets':{},'per_molecule':rows}
for name,a in [('truth_scored',[r for r in rows if r['truth_status']=='scored']),('truth_failed',[r for r in rows if r['truth_status']!='scored'])]:out['subsets'][name]={'metrics':metrics(a),'neutral_vs_base':delta(a,'base')}
p=root/'artifacts/007/review-r1/fiora-neutral-missing.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['per_molecule','sources']},ensure_ascii=False,indent=2));print(p)

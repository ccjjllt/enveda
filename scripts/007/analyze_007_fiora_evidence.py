"""Manual fixed-rule FIORA22 audit; no model execution, training or weight search."""
import json, math, statistics, hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[2]
paths={'fiora':'artifacts/runs/006-fiora/fiora_scores_22.json','ice':'artifacts/legacy/f1-scores/scores/f1e-4_ice_top100_split.json','inputs':'artifacts/legacy/f1-bundle/inputs.json','filed':'artifacts/runs/006-f1/f1_per_molecule.json'}
d={k:json.loads((root/p).read_text()) for k,p in paths.items()}
ms={m['key']:m for m in d['inputs']['molecules']}; filed={m['key']:m for m in d['filed']}
def valid(x): return isinstance(x,(int,float)) and math.isfinite(x)
def z(a):
 b=[float(x) for x in a if valid(x)]
 sd=statistics.stdev(b) if len(b)>1 else 0
 mu=statistics.mean(b) if b else 0
 return [(float(x)-mu)/sd if valid(x) and sd>0 else 0.0 for x in a]
def rank(order,truth): return order.index(truth)+1 if truth in order else None
def rr(r): return 1/r if r and r<=25 else 0.0
def order(keys,base,sims,alone=False):
 bz=z(base); sz=z(sims); ok=[i for i,s in enumerate(sims) if valid(s)]; missing=[i for i,s in enumerate(sims) if not valid(s)]
 if not ok: return keys[:]
 fused=[sz[i] if alone else bz[i]+sz[i] for i in range(len(keys))]
 return [keys[i] for i in sorted(ok,key=lambda i:(-fused[i],i))+sorted(missing,key=lambda i:(-base[i],i))]
def met(rows,name):
 rs=[r['ranks'][name] for r in rows]; return {'n':len(rs),'mrr25':statistics.mean(rr(r) for r in rs) if rs else None,'top1':sum(r==1 for r in rs)/len(rs) if rs else None,'top25':sum(r is not None and r<=25 for r in rs)/len(rs) if rs else None}
rows=[];prefix=[];mismatches=[]
for q,fc in d['fiora'].items():
 m=ms[q]; ic=d['ice'][q]; truth=m['truth_canon']; fm={ic[s]['canon']:v for s,v in fc.items()}; keys=[k for k in m['order_top100'] if k in fm]
 assert len(keys)==len(fc)==len(fm)
 if keys!=m['order_top100'][:50]:prefix.append(q)
 base=[m['base_scores_full_window'][k] for k in keys]
 sims=[fm[k]['similarity'] if fm[k]['status']=='scored' and valid(fm[k]['similarity']) else None for k in keys]
 orders={'base50':keys,'fiora_only50':order(keys,base,sims,True),'base_plus_fiora50':order(keys,base,sims)}
 ik=m['order_top100']; im={v['canon']:v for v in ic.values()}; ib=[m['base_scores_full_window'][k] for k in ik]; iss=[im[k].get('fused_merge') for k in ik]
 orders['ice100merge_reference']=order(ik,ib,iss)
 ranks={n:rank(o,truth) for n,o in orders.items()}
 if abs(rr(ranks['ice100merge_reference'])-filed[q]['F1e_ice100_merge_rr'])>1e-12:mismatches.append(q)
 tv=fm.get(truth); status=tv['status'] if tv else 'not_in_candidates';ti=keys.index(truth) if truth in keys else None; ts=sims[ti] if ti is not None else None
 tiebounds=None
 if ts is not None:
  n_gt=sum(s is not None and s>ts for s in sims); n_eq=sum(s is not None and s==ts for s in sims);tiebounds=[n_gt+1,n_gt+n_eq]
 r={'key':q,'truth':truth,'n_candidates':len(keys),'n_scored':sum(valid(s) for s in sims),'n_positive_sim':sum(valid(s) and s>0 for s in sims),'n_ice_scored':sum(valid(s) for s in iss),'truth_fiora_status':status,'truth_fiora_similarity':ts,'truth_similarity_tie_rank_interval':tiebounds,'ranks':ranks,'rr25':{n:rr(r) for n,r in ranks.items()},'top25':{n:o[:25] for n,o in orders.items()},'query_modes':sorted(set(s.get('mode','') for s in m['spectra'])),'candidates':[{'canon':k,'base':base[i],'fiora':sims[i],'fiora_status':fm[k]['status']} for i,k in enumerate(keys)]}
 rows.append(r)
methods=list(rows[0]['ranks']); summary={n:met(rows,n) for n in methods}
def compare(sub):
 ds=[r['rr25']['base_plus_fiora50']-r['rr25']['base50'] for r in sub]
 return {'n':len(sub),'delta_mrr25':statistics.mean(ds) if ds else None,'improved':sum(v>1e-12 for v in ds),'worsened':sum(v< -1e-12 for v in ds),'tied':sum(abs(v)<=1e-12 for v in ds)}
out={'scope':'22 selected NP250 pilot molecules, exploratory; no training or parameter scan; includes positive controls; not deployment evidence','rule':'Single fixed rule: z(base)+z(FIORA), ddof=1; z finite entries, missing z=0; scored candidates sorted by fused score, failed candidates appended by base; ties preserve FP base order; FIORA-only same tie/missing rule. Candidate pool is exact FIORA top50 prefix of F1 top100. ICE100merge shown only as current reference, not a same-candidate-pool ablation.','source_hashes':{k:{'path':p,'sha256':hashlib.sha256((root/p).read_bytes()).hexdigest()} for k,p in paths.items()},'checks':{'mapped_candidates':sum(r['n_candidates'] for r in rows),'unmapped':0,'nonprefix_queries':prefix,'ice_reference_reconstruction_mismatches':mismatches,'truth_in_fiora_pool':sum(r['truth_fiora_status']!='not_in_candidates' for r in rows),'truth_scored':sum(r['truth_fiora_status']=='scored' for r in rows)},'summary':summary,'delta':compare(rows),'subsets':{},'per_molecule':rows}
for name,sub in [('truth_scored',[r for r in rows if r['truth_fiora_status']=='scored']),('truth_failed',[r for r in rows if r['truth_fiora_status']=='inference_failed']),('ice_unsupported',[r for r in rows if r['n_ice_scored']==0]),('ice_supported',[r for r in rows if r['n_ice_scored']>0]),('negative_only',[r for r in rows if r['query_modes']==['negative']])]:
 out['subsets'][name]={'metrics':{n:met(sub,n) for n in methods},'delta':compare(sub)}
p=root/'artifacts/runs/007/fiora22-evidence.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,ensure_ascii=False,indent=2)); print(json.dumps({k:v for k,v in out.items() if k not in ['per_molecule','source_hashes','rule']},ensure_ascii=False,indent=2)); print('OUTPUT',p)
for r in rows:print(r['key'],r['query_modes'],'truth',r['truth_fiora_status'],r['truth_fiora_similarity'],'rank',r['ranks'],'ties',r['truth_similarity_tie_rank_interval'])

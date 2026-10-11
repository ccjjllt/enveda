"""Read-only manual audit of saved R1 features and OOF models; no training/heldout access."""
import json, hashlib
from pathlib import Path
from collections import Counter
import numpy as np
R=Path(__file__).resolve().parents[1]/'artifacts/007-r1'
SOURCES={}
def load(f):
 raw=(R/f).read_bytes()
 SOURCES[f]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 return json.loads(raw)
def z(v):
 a=np.array([np.nan if x is None else x for x in v],float);ok=np.isfinite(a);o=np.zeros(len(a))
 if ok.sum()>1 and a[ok].std(ddof=1)>0:o[ok]=(a[ok]-a[ok].mean())/a[ok].std(ddof=1)
 return o
def ranked_rr(order,scores,truth):
 idx=sorted(range(len(order)),key=lambda i:(-float(scores[i]),i));keys=[order[i] for i in idx];rk=keys.index(truth)+1 if truth in keys else None
 return (1/rk if rk is not None and rk<=25 else 0),rk,keys
zero_fields=['has_lib','lib_max','lib_second','lib_gap','lib_n_spectra','lib_n_sources','best_n_matched','best_explained','best_ent_sim','best_same_adduct','neg_ce_diff','ana_max','ana_explained','has_lib_x_libmax']
def masked(rows,truth,scenario):
 out=[dict(x) for x in rows]
 if scenario=='seen':return out
 for row in out:
  if scenario=='nolib_all' or row['c']==truth:
   for field in zero_fields:row[field]=0.
   row['ce_missing']=1.
 leader=max(r['lib_max'] for r in out)
 for row in out:
  row['lib_leader_gap']=leader-row['lib_max'];row['has_lib_x_base']=row['has_lib']*row['base_z'];row['mol_nspec_x_libmax']=row['mol_n_spec']*row['lib_max']
 return out
def stats(a,b):
 a,b=np.array(a),np.array(b);d=b-a;idx=np.random.default_rng(20261004).integers(0,len(a),size=(20000,len(a)))
 return {'n':len(a),'A_mrr':float(a.mean()),'C_mrr':float(b.mean()),'delta':float(d.mean()),'A_top1':float((a==1).mean()),'C_top1':float((b==1).mean()),'A_top25':float((a>0).mean()),'C_top25':float((b>0).mean()),'improved':int((d>1e-12).sum()),'worsened':int((d<-1e-12).sum()),'tied':int((abs(d)<=1e-12).sum()),'w2r':int(((a<1)&(b==1)).sum()),'r2w':int(((a==1)&(b<1)).sum()),'ci95':np.quantile(d[idx].mean(axis=1),[.025,.975]).tolist()}
report={'scope':'Saved NP250/G1 research250 features and OOF models; no training; no heldout reads','bootstrap_seed':20261004,'bootstrap_draws':20000,'datasets':{}}
config=load('config.json'); feats=config['C_FEATURES']
for ds, ftname in [('np','feature_table_np250.json'),('g1','feature_table_g1.json')]:
 FT=load(ftname);AF=load('A_full_'+ds+'.json');AR=load('A_ranks_'+ds+'.json');folds=load('folds_'+ds+'.json')['folds'];models=load('models_C_'+ds+'.json');filed=load('metrics_C_'+ds+'.json')
 keys=list(FT);A=[];mismatch=Counter();v9diff=0.;seen_truth_library=[];truth_missing=[];all_scores={};base_records={}
 for k in keys:
  m=AF[k];order=m['order_top100'];v9=z([m['base'][c] for c in order])+z([m['fused_merge'][c] for c in order]);v9diff=max(v9diff,max(abs(v9[i]-m['v9'][c]) for i,c in enumerate(order)))
  rr,rk,ranked=ranked_rr(order,v9,m['truth']);A.append(rr);base_records[k]={'A_rr':rr,'A_rank':rk}
  mismatch['saved_A_full_rank']+=rk!=m['rank'];mismatch['saved_A_rr']+=abs(rr-m['rr'])>1e-12;mismatch['saved_A_ranks_rank']+=rk!=AR[k]['rank']
  tr=[r for r in FT[k]['rows'] if r['c']==FT[k]['truth']]
  if not tr:truth_missing.append(k)
  elif tr[0]['has_lib']>0:seen_truth_library.append(k)
 res={'A_mrr':float(np.mean(A)),'A_top1_count':sum(a==1 for a in A),'A_top25_count':sum(a>0 for a in A),'A_mismatches':dict(mismatch),'maximum_v9_score_difference':v9diff,'truth_has_library_count':len(seen_truth_library),'truth_outside_shortlist':truth_missing,'scenarios':{},'metric_discrepancies':[],'weight_signs':{},'changed_no_library':{}}
 for j,feat in enumerate(feats):
  ws=[m['w'][j] for m in models.values()];res['weight_signs'][feat]={'negative_folds':sum(w<0 for w in ws),'positive_folds':sum(w>0 for w in ws),'min':min(ws),'max':max(ws)}
 for alpha in [.25,.5]:
  for scenario in ['seen','nolib_all','pressure_truth_off']:
   C=[];cranks={};maxres=0.;nclip=0
   for k in keys:
    mol=FT[k];order=mol['order'];model=models[str(folds[mol['truth']])];lookup={r['c']:r for r in masked(mol['rows'],mol['truth'],scenario)}
    X=np.array([[lookup[c][f] for f in feats] for c in order],float);mu=np.array(model['mu']);sd=np.array(model['sd']);w=np.array(model['w']);std=np.zeros_like(X);ok=sd>=1e-12;std[:,ok]=(X[:,ok]-mu[ok])/sd[ok]
    residual=std@w;maxres=max(maxres,float(np.max(np.abs(residual))));nclip+=int((np.abs(alpha*residual)>1).sum());final=np.array([AF[k]['v9'][c] for c in order])+np.clip(alpha*residual,-1,1)
    rr,rk,_=ranked_rr(order,final,mol['truth']);C.append(rr);cranks[k]=rk
   name='a%.2f_%s'%(alpha,scenario);summary=stats(A,C);res['scenarios'][name]={**summary,'n_candidate_scores_clipped':nclip,'maximum_absolute_residual':maxres}
   for metric,val in summary.items():
    expected=filed[name][metric]
    if not np.allclose(val,expected,rtol=0,atol=1e-12):res['metric_discrepancies'].append({'scenario':name,'metric':metric,'actual':val,'filed':expected})
   all_scores[name]=dict(zip(keys,C))
   if scenario=='nolib_all':res['changed_no_library'][str(alpha)]=[{'key':k,**base_records[k],'C_rr':rr,'C_rank':cranks[k]} for k,a,rr in zip(keys,A,C) if abs(a-rr)>1e-12]
 if ds=='g1':
  k='ZNNDCTWCOYGODJ';res['xfer_single_observation']={str(alpha):{'key':k,'A_rr':base_records[k]['A_rr'],'C_rr':all_scores['a%.2f_seen'%alpha][k]} for alpha in [.25,.5]}
 res['source_holdout_scenario']={'available':False,'reason':'masks_source.json says definition built; evaluation deferred. run_C.py has only seen/nolib_all/pressure and a one-molecule xfer subset of seen.'}
 res['seen_equals_pressure_all250']={str(alpha):all_scores['a%.2f_seen'%alpha]==all_scores['a%.2f_pressure_truth_off'%alpha] for alpha in [.25,.5]}
 report['datasets'][ds]=res
combined=load('metrics_by_scenario.json');combined_mismatches=[]
for ds,r in report['datasets'].items():
 for alpha in [.25,.5]:
  for scenario in ['seen','nolib_all','pressure_truth_off']:
   a=r['scenarios']['a%.2f_%s'%(alpha,scenario)];b=combined[ds.upper()][scenario]['C_alpha%.2f'%alpha]
   for k,k2 in [('C_mrr','MRR'),('C_top1','Top1'),('C_top25','Top25'),('delta','Delta_vs_A'),('improved','improved'),('worsened','worsened'),('w2r','w2r'),('r2w','r2w')]:
    if abs(a[k]-b[k2])>1e-12:combined_mismatches.append({'dataset':ds,'alpha':alpha,'scenario':scenario,'metric':k})
report['metrics_by_scenario_mismatches']=combined_mismatches
report['sources']=SOURCES
out=R.parent/'007/review-r1/r1-metrics-audit.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
for ds,r in report['datasets'].items():
 print(ds,'A',r['A_mrr'],'truthlib',r['truth_has_library_count'],'mismatch',r['metric_discrepancies'],'no_lib_changes',r['changed_no_library'])
 for s,v in r['scenarios'].items():print(s,round(v['C_mrr'],9),v['improved'],v['worsened'],v['w2r'],v['r2w'],'clipped',v['n_candidate_scores_clipped'])
 print('WEIGHTS',r['weight_signs'])
print('output',out)

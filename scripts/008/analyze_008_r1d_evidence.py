"""Read-only manual recalculation of saved R1-D scores/models; no fitting or heldout query reads."""
import json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'artifacts/runs';D=R/'007-r1d'
SOURCES={}
def load(p):
 raw=p.read_bytes();SOURCES[str(p.relative_to(ROOT))]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 return json.loads(raw)
F=load(D/'config_d.json')['D_FEATURES_13']
def zero(r):
 q=dict(r)
 for k in ['has_lib','lib_max','lib_second','lib_gap','lib_n_spectra','lib_n_sources','best_n_matched','best_explained','best_ent_sim','best_same_adduct','neg_ce_diff','ana_max','ana_explained','has_lib_x_base']:q[k]=0.
 q['ce_missing']=1.;return q
def rows_for(ds,k,sc,ft,src):
 if sc=='seen':return ft[k]['rows']
 if sc=='source_transfer':return src[k]['rows']
 return [zero(r) if sc=='nolib_all' or r['c']==ft[k]['truth'] else dict(r) for r in ft[k]['rows']]
def feature(rows,order):
 lookup={r['c']:r for r in rows};return np.array([[lookup[c][f] for f in F] for c in order],float)
def random_rows(k,rows,order):
 seed=int(hashlib.sha256(('20261004:'+k).encode()).hexdigest()[:8],16);rng=np.random.default_rng(seed);drop={c:rng.random()<.5 for c in order}
 return [zero(r) if drop[r['c']] else dict(r) for r in rows]
def rr(keys,t):
 rank=keys.index(t)+1 if t in keys else None;return (1/rank if rank is not None and rank<=25 else 0),rank
def metrics(a,b):
 a,b=np.array(a),np.array(b);d=b-a;ind=np.random.default_rng(20261004).integers(0,len(a),(20000,len(a)))
 return {'A_mrr':float(a.mean()),'C_mrr':float(b.mean()),'delta':float(d.mean()),'C_top1':float((b==1).mean()),'C_top25':float((b>0).mean()),'improved':int((d>1e-12).sum()),'worsened':int((d<-1e-12).sum()),'w2r':int(((a<1)&(b==1)).sum()),'r2w':int(((a==1)&(b<1)).sum()),'ci95':np.quantile(d[ind].mean(axis=1),[.025,.975]).tolist()}
report={'scope':'Saved NP/G1 research query feature tables, models, OOF results only. No fitting.','datasets':{}}
for ds in ['np','g1']:
 ft=load(D/('feature_table_'+ds+'_d.json'));src=load(D/('feature_table_'+ds+'_d_source.json'));models=load(D/('models_D_'+ds+'.json'));folds=load(D/('folds_'+ds+'_d.json'))['folds'];oof=load(D/('oof_scores_D_'+ds+'.json'));af=load(R/'007-r1'/('A_full_'+ds+'.json'));saved=load(D/('metrics_tmp_'+ds+'.json'))['scenarios']
 check=Counter();out={'metrics':{},'training_standardization':{},'source_table_check':{},'gate_train_eval_mismatch':{},'forward_uncovered':{}};max_res_diff=0.;max_final_diff=0.;keys=list(ft);av=[rr(af[k]['ranked'],ft[k]['truth'])[0] for k in keys];uncovered=[];ungated_abs=[];truth_nolib_ungated=[]
 for k in keys:
  order=ft[k]['order'];pos={c:i for i,c in enumerate(order)};scored={c for c in order if af[k]['fused_merge'].get(c) is not None};v=af[k]['v9'];baseline=sorted(scored,key=lambda c:(-v[c],pos[c]))+sorted(set(order)-scored,key=lambda c:(-v[c],pos[c]));check['alpha0_order_mismatch']+=baseline!=af[k]['ranked'];check['fold_mismatch']+=oof[k]['fold']!=folds[ft[k]['truth']]
  if not scored:uncovered.append(k)
 for alpha in [.25,.5]:
  for sc in ['seen','nolib_all','pressure_truth_off','source_transfer']:
   bv=[];name='a%.2f_%s'%(alpha,sc)
   for k in keys:
    order=ft[k]['order'];pos={c:i for i,c in enumerate(order)};scored={c for c in order if af[k]['fused_merge'].get(c) is not None};model=models[str(folds[ft[k]['truth']])];rows=rows_for(ds,k,sc,ft,src);lookup={r['c']:r for r in rows};X=feature(rows,order);mu=np.array(model['mu']);sd=np.array(model['sd']);w=np.array(model['w']);std=np.zeros_like(X);ok=sd>=1e-12;std[:,ok]=(X[:,ok]-mu[ok])/sd[ok];raw=std@w;res=np.where(X[:,0]>0,raw,0.);final={c:float(af[k]['v9'][c]+np.clip(alpha*res[i],-1,1)) if c in scored else float(af[k]['v9'][c]) for i,c in enumerate(order)};ranking=sorted(scored,key=lambda c:(-final[c],pos[c]))+sorted(set(order)-scored,key=lambda c:(-af[k]['v9'][c],pos[c]));val,rank=rr(ranking,ft[k]['truth']);bv.append(val);rec=oof[k]['scenarios'][name]
    check['OOF_full_order_mismatch']+=ranking!=rec['ranked'];check['OOF_rank_mismatch']+=rank!=rec['rank'];check['OOF_rr_mismatch']+=abs(val-rec['rr'])>1e-12
    max_res_diff=max(max_res_diff,max(abs(res[i]-rec['residuals'][c]) for i,c in enumerate(order)));max_final_diff=max(max_final_diff,max(abs(final[c]-rec['finals'][c]) for c in order))
    if sc=='nolib_all':check['nolib_A_order_mismatch']+=ranking!=af[k]['ranked']
    if alpha==.25 and sc=='seen':
     ungated_abs.extend(abs(raw[X[:,0]==0]).tolist())
     if ft[k]['truth'] in order and lookup[ft[k]['truth']]['has_lib']==0:truth_nolib_ungated.append(float(raw[pos[ft[k]['truth']]]))
   met=metrics(av,bv);out['metrics'][name]=met
   for metric,val in met.items():check['filed_metric_mismatch']+=not np.allclose(val,saved[name][metric],atol=1e-12,rtol=0)
 for fold in range(5):
  xs=[];train=[]
  for k in keys:
   mol=ft[k];order=mol['order']
   if folds[mol['truth']]==fold or mol['truth'] not in order:continue
   train.append(k)
   for rows in [mol['rows'],rows_for(ds,k,'pressure_truth_off',ft,src),src[k]['rows'],random_rows(k,mol['rows'],order)]:xs.append(feature(rows,order))
  X=np.concatenate(xs);m=models[str(fold)];out['training_standardization'][str(fold)]={'n_train_molecules':len(train),'n_rows':len(X),'mu_max_abs_error':float(np.max(abs(X.mean(axis=0)-np.array(m['mu'])))),'sd_max_abs_error':float(np.max(abs(X.std(axis=0)-np.array(m['sd'])))),'n_train_validation_overlap':sum(folds[ft[k]['truth']]==fold for k in train)}
 target='riken' if ds=='np' else 'enveda-180';out['source_table_check']={'removed_source_present_rows':sum(target in r['sources_sorted'] for m in src.values() for r in m['rows']),'truth_with_lib_seen':sum(any(r['c']==m['truth'] and r['has_lib']>0 for r in m['rows']) for m in ft.values()),'truth_with_lib_source':sum(any(r['c']==m['truth'] and r['has_lib']>0 for r in m['rows']) for m in src.values()),'all_candidates_with_lib_seen':sum(r['has_lib']>0 for m in ft.values() for r in m['rows']),'all_candidates_with_lib_source':sum(r['has_lib']>0 for m in src.values() for r in m['rows'])}
 out['gate_train_eval_mismatch']={'raw_ungated_residual_of_seen_nolib_candidates_count':len(ungated_abs),'nonzero_count':sum(v>1e-12 for v in ungated_abs),'max_abs':max(ungated_abs),'truth_nolib_raw_residual_min':min(truth_nolib_ungated) if truth_nolib_ungated else None,'truth_nolib_raw_residual_max':max(truth_nolib_ungated) if truth_nolib_ungated else None,'code_finding':'Training optimizes w.x_std for every candidate; inference sets it to zero when has_lib=0. Training and inference functions differ.'}
 out['forward_uncovered']={'n_molecules':len(uncovered),'n_with_any_library':sum(any(r['has_lib']>0 for r in ft[k]['rows']) for k in uncovered),'n_with_truth_library':sum(any(r['has_lib']>0 and r['c']==ft[k]['truth'] for r in ft[k]['rows']) for k in uncovered),'all_scenarios_ranking_unchanged':all(oof[k]['scenarios'][s]['ranked']==af[k]['ranked'] for k in uncovered for s in oof[k]['scenarios']),'note':'D ignores residuals of forward-unscored candidates, even when they have library evidence.'}
 out['checks']=dict(check);out['maximum_residual_difference']=max_res_diff;out['maximum_final_score_difference']=max_final_diff;report['datasets'][ds]=out
report['sources']=SOURCES
path=R/'008/r1d-audit.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
for ds,o in report['datasets'].items():
 print(ds,'checks',o['checks'],'resdiff',o['maximum_residual_difference']);print('source',o['source_table_check']);print('gate',o['gate_train_eval_mismatch']);print('uncovered',o['forward_uncovered']);print('normalization',o['training_standardization'])
 for k,m in o['metrics'].items():print(k,round(m['C_mrr'],9),m['improved'],m['worsened'],m['w2r'],m['r2w'])
print(path)

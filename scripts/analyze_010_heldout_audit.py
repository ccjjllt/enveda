#!/usr/bin/env python3
"""手动复核009已用验证250：谱/FP缓存/固定路由/完整排序/配对CI；不调参、不训练。

不会覆盖009产物。这个集合已于2026-10-13用于验证，复算不产生第二个独立验证集。
默认重新运行1400谱CPU FP前向，以核对逐谱缓存；不运行GPU或ICE前向。
"""
from __future__ import annotations
import argparse,collections,hashlib,json,pickle,shutil,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
def load(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def z(x):
 a=np.array([np.nan if v is None else float(v) for v in x]);ok=np.isfinite(a);b=np.zeros(len(a))
 if ok.sum()>1 and a[ok].std(ddof=1)>0:b[ok]=(a[ok]-a[ok].mean())/a[ok].std(ddof=1)
 return b
def score_rr(order,truth):return 1/(order.index(truth)+1) if truth in order[:25] else 0.
def stats(rows):
 a=np.array([r['v9_rr'] for r in rows]);b=np.array([r['final_rr'] for r in rows]);d=b-a;rng=np.random.default_rng(20261012);boot=d[rng.integers(0,len(rows),(20000,len(rows)))].mean(axis=1)
 return {'n':len(rows),'mrr_a':float(a.mean()),'mrr_b':float(b.mean()),'delta_mrr':float(d.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'improved':int((d>1e-12).sum()),'worsened':int((d< -1e-12).sum()),'tied':int((abs(d)<=1e-12).sum()),'top1_a':int((a==1).sum()),'top1_b':int((b==1).sum()),'top1_wrong_to_right':int(((a<1)&(b==1)).sum()),'top1_right_to_wrong':int(((a==1)&(b<1)).sum()),'top25_a':int((a>0).sum()),'top25_b':int((b>0).sum())}

def verify_spectra(molecules,manifest):
 import pyarrow as pa
 import pyarrow.parquet as pq
 structs={s['canonical_key']:s for s in manifest['structures'] if s['split']=='heldout'};want={};checks=collections.Counter()
 for m in molecules:
  st=structs[m['key']]
  if len(st['rows_sorted'])!=len(m['spectra']):raise ValueError('manifest spectrum count mismatch')
  for ri,s in zip(st['rows_sorted'],m['spectra']):want[ri]=s
 cols=['ms2_mzs','ms2_normalized_intensities','precursor_mz','adduct','ionization_mode','instrument_type','collision_energy_ev','collision_energy_orig','collision_energy_orig_units']
 dest=['mz','it','prec','adduct','mode','instrument','ce_ev','ce_orig','ce_units'];pf=pq.ParquetFile(ROOT/'data/raw/train.parquet');off=0
 for gi in range(pf.num_row_groups):
  n=pf.metadata.row_group(gi).num_rows;indices=sorted(r for r in want if off<=r<off+n)
  if indices:
   table=pf.read_row_group(gi,columns=cols).take(pa.array([r-off for r in indices],type=pa.int64())).to_pydict()
   for i,ri in enumerate(indices):
    for src,dst in zip(cols,dest):
     a=table[src][i];b=want[ri][dst]
     if dst in ['mz','it','ce_ev']:same=np.array_equal(np.asarray(a if a is not None else [],float),np.asarray(b,float))
     elif dst in ['adduct','mode','instrument','ce_orig','ce_units']:same=str(a)==str(b)
     else:same=float(a)==float(b)
     checks[dst+'_mismatches']+=not same
   checks['spectra_checked']+=len(indices)
   print('raw spectra checked',checks['spectra_checked'],flush=True)
  off+=n
 return dict(checks)

def verify_logits(molecules,saved):
 import torch
 from src.models.fingerprint import FPNet,prep_peaks,ADDUCT_IX,instr_family
 torch.set_num_threads(4);p=ROOT/'artifacts/001-baseline/weights/fp_single_s2.pt';ck=torch.load(p,map_location='cpu',weights_only=True);net=FPNet(nbits=int(ck['nbits']),d=int(ck['d']),layers=int(ck['layers']));net.load_state_dict(ck['model'],strict=True);net.eval();specs=[s for m in molecules for s in m['spectra']];maximum=0.;sq=0.;different=0;rows=[];t=time.monotonic()
 for start in range(0,len(specs),32):
  ss=specs[start:start+32];n=len(ss);mz=np.zeros((n,128),np.float32);it=np.zeros_like(mz);pad=np.ones((n,128),bool);ce=[]
  for i,s in enumerate(ss):
   a,b=prep_peaks(np.asarray(s['mz'],float),np.asarray(s['it'],float),float(s['prec']));ll=min(len(a),128);mz[i,:ll]=a[:ll];it[i,:ll]=b[:ll];pad[i,:ll]=False;cv=np.asarray(s['ce_ev'],float).reshape(-1);ce.append(float(cv.mean()) if len(cv) and np.isfinite(cv).all() else 25.)
  args=[mz,it,pad,np.array([s['prec'] for s in ss],np.float32),np.array([ADDUCT_IX.get(s['adduct'],ADDUCT_IX['<unk>']) for s in ss]),np.array([instr_family(s['instrument']) for s in ss]),np.array(ce,np.float32),np.array([1. if s['mode']=='positive' else -1. for s in ss],np.float32)]
  with torch.no_grad():out=net(*[torch.as_tensor(a) for a in args]).numpy()
  err=abs(out-saved[start:start+n]);maximum=max(maximum,float(err.max()));sq+=float((err.astype(float)**2).sum());different+=int(np.count_nonzero(err));rows.extend(np.max(err,axis=1).tolist())
  if start%320==0:print('CPU forward checked',start+n,flush=True)
 return {'n_spectra':len(specs),'n_bits':6930,'maximum_absolute_difference':maximum,'different_scalar_count':different,'rmse':float(np.sqrt(sq/saved.size)),'spectra_with_any_difference':sum(e>0 for e in rows),'seconds':time.monotonic()-t,'checkpoint_sha256':sha(p),'torch':torch.__version__,'device':'CPU'}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'artifacts/010-heldout-audit');p.add_argument('--skip-forward',action='store_true');p.add_argument('--skip-raw-spectra',action='store_true');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);t=time.monotonic();d=ROOT/'artifacts/009-heldout';molecules=load(d/'inputs.json')['molecules'];cand=load(d/'heldout_candidate_scores.json');routes={r['key']:r for r in load(d/'routing.json')['rows']};split=load(d/'scores/scores/v9heldout_ice_top100_split.json');old=load(d/'heldout_score.json');old_rows={r['key']:r for r in old['per_molecule']};manifest=load(ROOT/'artifacts/006/f0_new_timstof500_manifest.json');logits=np.load(d/'heldout_logits.npy',mmap_mode='r');summary={'scope':'Audit of already-used validation250, not a new validation exercise; fixed rule unchanged; no training/GPU/ICE forward.'}
 recovered=args.output/'recovered-build';recovered.mkdir(exist_ok=True);summary['recovered_build_scripts']={}
 for source in sorted(Path('/private/tmp').glob('build_heldout_step*.py')):
  target=recovered/source.name;shutil.copyfile(source,target);summary['recovered_build_scripts'][source.name]={'source':str(source),'sha256':sha(source)}
 summary['raw_spectra_check']=None if args.skip_raw_spectra else verify_spectra(molecules,manifest)
 summary['logits_check']=None if args.skip_forward else verify_logits(molecules,logits)
 import pyarrow.parquet as pq
 from candidates import query_neutral_mass
 maps=[]
 for name in ['train','coco']:
  mm=pq.read_table(ROOT/f'artifacts/005/structure_identity_{name}.parquet',columns=['raw_key','canonical_group_key']).to_pydict();maps.append(dict(zip(mm['raw_key'],mm['canonical_group_key'])))
 keys=np.load(ROOT/'artifacts/pool-train/keys.npy',allow_pickle=True).tolist();ntr=len(keys);cm=pickle.load(open(ROOT/'artifacts/005/coco/coco_meta.pkl','rb'));keys+=list(cm['keys']);mass=np.concatenate([np.load(ROOT/'artifacts/pool-train/mass.npy'),np.load(ROOT/'artifacts/005/coco/coco_mass.npy')]);tfp=np.load(ROOT/'artifacts/pool-train/fp.npy',mmap_mode='r');cfp=np.load(ROOT/'artifacts/005/coco/coco_fp.npy',mmap_mode='r');windows=np.load(d/'heldout_windows.npz');checks=collections.Counter();maximums=collections.defaultdict(float);rows=[];off=0;meta_modes=collections.Counter();meta_adducts=collections.Counter()
 for m in molecules:
  k=m['key'];ss=m['spectra'];rep=max(range(len(ss)),key=lambda i:len(ss[i]['mz']));checks['qidx_mismatch']+=m['qidx']!=off+rep;qrep=np.asarray(logits[m['qidx']],float);qmean=np.asarray(logits[off:off+len(ss)],float).mean(axis=0);off+=len(ss);qm,known=query_neutral_mass(ss[rep]['prec'],ss[rep]['adduct']);wi=np.flatnonzero(abs(mass-qm)<=abs(qm)*10e-6);checks['window_mismatch']+=not np.array_equal(wi,windows['union|single|10.0|'+k]);packs=np.stack([tfp[j] if j<ntr else cfp[j-ntr] for j in wi]);bits=np.unpackbits(packs,axis=1,bitorder='big')[:,:6930].astype(float);dots=bits@qrep;best={}
  for pos,(j,score) in enumerate(zip(wi,dots)):
   canonical=maps[0 if j<ntr else 1].get(keys[j],keys[j]);previous=best.get(canonical)
   if previous is None or score>previous[0]:best[canonical]=(float(score),pos,int(j))
  order=sorted(best,key=lambda c:(-best[c][0],c));entry=cand[k];checks['full_keys_mismatch']+=set(order)!=set(entry['full_keys']);checks['shortlist_mismatch']+=order[:100]!=entry['frozen_top100'];repbits=bits[[best[c][1] for c in order]];mean_scores=dict(zip(order,(repbits@qmean).tolist()));base={c:best[c][0] for c in order}
  for c in order:
   maximums['representative_fp']=max(maximums['representative_fp'],abs(base[c]-entry['fp']['representative'][c]));maximums['all_mean_fp']=max(maximums['all_mean_fp'],abs(mean_scores[c]-entry['fp']['all_mean'][c]));checks['representative_pool_index_mismatch']+=best[c][2]!=entry['representative_pool_indices'][c]
  support=any(str(s['mode']).lower() in {'positive','pos','+','1'} and s['adduct'] in {'[M+H]+','[M+Na]+'} for s in ss);checks['metadata_route_mismatch']+=support!=routes[k]['metadata_has_supported_record'];short=order[:100];bycanon={}
  for smi,rec in split.get(k,{}).items():bycanon.setdefault(rec['canon'],[]).append(rec)
  sims=[bycanon[c][0]['fused_merge'] if c in bycanon else None for c in short];finite=np.array([v is not None and np.isfinite(v) for v in sims]);checks['observed_support_mismatch']+=support!=bool(finite.any());f=z([base[c] for c in short])+z(sims);ix=np.arange(len(short));ranked_ix=sorted(ix[finite],key=lambda i:(-f[i],i))+sorted(ix[~finite],key=lambda i:(-base[short[i]],i));v9=[short[i] for i in ranked_ix];mean_order=sorted(order,key=lambda c:(-mean_scores[c],c));final=v9 if support else mean_order;truth=m['truth_canon'];rec={'key':k,'supported':support,'n_spectra':len(ss),'adducts':dict(collections.Counter(s['adduct'] for s in ss)),'v9_rr':score_rr(v9,truth),'final_rr':score_rr(final,truth),'v9_rank':v9.index(truth)+1 if truth in v9 else None,'final_rank':final.index(truth)+1 if truth in final else None,'v9_top25':v9[:25],'final_top25':final[:25]}
  checks['v9_top25_mismatch']+=v9[:25]!=old_rows[k]['v9_top25'];checks['final_top25_mismatch']+=final[:25]!=old_rows[k]['final_top25'];checks['rr_mismatch']+=abs(rec['v9_rr']-old_rows[k]['v9_rr'])>1e-12 or abs(rec['final_rr']-old_rows[k]['final_rr'])>1e-12;rows.append(rec);meta_modes.update(s['mode'] for s in ss);meta_adducts.update(s['adduct'] for s in ss)
 summary['candidate_checks']=dict(checks);summary['maximum_score_errors']=dict(maximums);summary['spectra_counts']={'n':off,'logits_shape':list(logits.shape),'logits_finite':bool(np.isfinite(logits).all()),'modes':dict(meta_modes),'adducts':dict(meta_adducts)};summary['metrics']={'all':stats(rows),'supported':stats([r for r in rows if r['supported']]),'unsupported':stats([r for r in rows if not r['supported']])};summary['filed_metric_mismatches']=[]
 for label,metric in summary['metrics'].items():
  filed=old['summary_final_vs_v9'] if label=='all' else old['summary_final_vs_v9']['strata_'+label]
  for name,value in metric.items():
   if not np.allclose(value,filed[name],rtol=0,atol=1e-12):summary['filed_metric_mismatches'].append([label,name])
 held={s['canonical_key'] for s in manifest['structures'] if s['split']=='heldout'};research={s['canonical_key'] for s in manifest['structures'] if s['split']=='research'};summary['split_identity']={'used_validation_keys':len(held),'input_manifest_key_symmetric_difference':len(held^{m['key'] for m in molecules}),'research_key_overlap':len(held&research),'pretrained_exposure':'unknown; manifest holds research tuning labels out, not proof of unseen structures for fixed pretrained weights'}
 summary['freeze_evidence']={'config_sha256':sha(ROOT/'artifacts/009-conditional/frozen_config.json'),'conditional_recorded_created_utc':load(ROOT/'artifacts/009-conditional/run_meta.json')['created_utc'],'routing_recorded_created_utc':load(d/'routing.json')['created_utc'],'heldout_score_recorded_wall_clock':old['wall_clock_local'],'interpretation':'Recorded metadata supports freezing after development selection and before this validation; no claim of a new independent prior or external trusted timestamp.'};summary['changed_cases']=[r for r in rows if r['v9_rr']!=r['final_rr']];summary['seconds']=time.monotonic()-t;summary['script_sha256']=sha(Path(__file__));write(args.output/'audit.json',summary);write(args.output/'per_molecule.json',rows);print(json.dumps({k:v for k,v in summary.items() if k not in ['changed_cases','recovered_build_scripts']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()

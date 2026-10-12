#!/usr/bin/env python3
"""手动CPU研究：008扩库的真实干扰与贡献者来源敏感性；不训练、不读heldout。

固定008的539个新增结构。先用NP/G1研究缓存测全部新增干扰，再重新计算R3-50
FP原head排序及同记录/同贡献者移除索引。来源移除是压力诊断，不是部署过滤规则。
候选枚举只用观测质量与固定索引；truth仅在排序结束后评价，不注入候选。
"""
from __future__ import annotations
import argparse
import importlib.metadata
import platform
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import re
import sys
import time
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from candidate_expansion import fp_bits_for_smiles, query_neutral_mass_strict
from candidates import query_neutral_mass


def read(p): return json.loads(p.read_text())
def write(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def ranked(scores):return sorted(scores,key=lambda k:(-scores[k],k))
def rank(order,truth):return order.index(truth)+1 if truth in order else None
def rr(r):return 1/r if r is not None and r<=25 else 0.
def summary(rows,base='base_rank',expanded='expanded_rank'):
 a=np.array([rr(x[base]) for x in rows]);b=np.array([rr(x[expanded]) for x in rows]);d=b-a
 return {'n':len(rows),'base_mrr25':float(a.mean()) if len(a) else None,'expanded_mrr25':float(b.mean()) if len(b) else None,
 'delta_mrr25':float(d.mean()) if len(d) else None,'improved':int((d>1e-12).sum()),'worsened':int((d< -1e-12).sum()),
 'base_top1':int((a==1).sum()),'expanded_top1':int((b==1).sum()),'base_top25':int((a>0).sum()),'expanded_top25':int((b>0).sum()),
 'w2r':int(((a<1)&(b==1)).sum()),'r2w':int(((a==1)&(b<1)).sum()),'lost_top25':int(((a>0)&(b==0)).sum()),
 'base_top100':sum(x[base] is not None and x[base]<=100 for x in rows),'expanded_top100':sum(x[expanded] is not None and x[expanded]<=100 for x in rows),
 'rank_worsened':sum(x[base] is not None and x[expanded] is not None and x[expanded]>x[base] for x in rows)}


def load_index():
 import pyarrow.parquet as pq
 maps=[]
 for name in ['train','coco']:
  d=pq.read_table(ROOT/f'artifacts/005/structure_identity_{name}.parquet',columns=['raw_key','canonical_group_key']).to_pydict()
  maps.append(dict(zip(d['raw_key'],d['canonical_group_key'])))
 union=set(maps[0].values())|set(maps[1].values())
 entries=[json.loads(x) for x in (ROOT/'artifacts/008-expansion/candidate_index_records.jsonl').read_text().splitlines()]
 new=[x for x in entries if x['canonical_key'] not in union]
 return entries,new,maps,union


def fingerprint_cache(bits):
 cache={}
 def get(smiles):
  if smiles not in cache:cache[smiles]=np.asarray(fp_bits_for_smiles(smiles,bits)[0],dtype=np.float64)
  return cache[smiles]
 return get


def interference(out,new,fp):
 result={};full={}
 nm=np.array([e['mass_theo'] for e in new],float)
 for ds,folder in [('np','f1-bundle'),('g1','g1-validation')]:
  folder=ROOT/'artifacts'/folder;inp=read(folder/'inputs.json');logits=np.load(folder/'inputs_e0_logits.npy',mmap_mode='r');rows=[];offset=0;score_check=0.;ordering_mismatch=0;full[ds]={}
  for m in inp['molecules']:
   local=m['qidx']-offset;offset+=len(m['spectra'])
   if local!=max(range(len(m['spectra'])),key=lambda i:len(m['spectra'][i]['mz'])):raise RuntimeError('representative query index mismatch')
   s=m['spectra'][local];qm,known=query_neutral_mass(float(s['prec']),s['adduct']);q=np.asarray(logits[m['qidx']],float)
   base={k:float(v) for k,v in m['base_scores_full_window'].items()};bo=ranked(base);ordering_mismatch+=bo[:100]!=m['order_top100']
   # Verify selected cached logits against the saved best representative's score.
   score_check=max(score_check,abs(float(fp(m['rep_smiles'][bo[0]])@q)-base[bo[0]]))
   selected=np.flatnonzero(abs(nm-qm)<=abs(qm)*10e-6).tolist();added={new[j]['canonical_key']:float(fp(new[j]['smiles_rep'])@q) for j in selected}
   overlap=set(base)&set(added)
   if overlap:raise RuntimeError('supposedly new structure overlaps base')
   expanded={**base,**added};eo=ranked(expanded);truth=m['truth_canon']
   rows.append({'key':m['key'],'truth':truth,'query_mass':float(qm),'known_adduct':bool(known),'base_rank':rank(bo,truth),'expanded_rank':rank(eo,truth),'base_window':len(bo),'expanded_window':len(eo),'n_added':len(added),'added':added,
     'new_in_top100':[k for k in eo[:100] if k in added],'new_in_top25':[k for k in eo[:25] if k in added],
     'old_top100_displaced':[k for k in bo[:100] if k not in eo[:100]],'old_top25_displaced':[k for k in bo[:25] if k not in eo[:25]],'base_top1':bo[0],'expanded_top1':eo[0]})
   full[ds][m['key']]={'order_expanded':eo,'scores_expanded':expanded,'original_order':bo}
  result[ds]={'summary':summary(rows),'baseline_order_mismatches':ordering_mismatch,'cached_logit_vs_saved_top1_score_max_error':score_check,'queries_with_additions':sum(r['n_added']>0 for r in rows),'n_new_candidate_query_pairs':sum(r['n_added'] for r in rows),'new_top100_pairs':sum(len(r['new_in_top100']) for r in rows),'new_top25_pairs':sum(len(r['new_in_top25']) for r in rows),'rows':rows}
  print('CACHE',ds,json.dumps({k:v for k,v in result[ds].items() if k!='rows'}),flush=True)
 write(out/'research_interference.json',result);write(out/'research_full_fp_scores.json',full)
 return result


def provenance(out,entries):
 path=out/'massbank_complete_provenance.json'
 if path.exists():return read(path)
 from chemistry import canonical_group_key
 from rdkit import Chem
 from rdkit import RDLogger
 RDLogger.DisableLog('rdApp.warning');RDLogger.DisableLog('rdApp.error')
 meta=read(ROOT/'artifacts/008-expansion/candidate_index_meta.json');bulk=Path(meta['snapshot']['bulk_zip']);smiles_keys={e['smiles_rep']:e['canonical_key'] for e in entries};known={e['canonical_key'] for e in entries}
 by_smiles={};acc_source={};n_allowed=0
 with zipfile.ZipFile(bulk) as z:
  for name in z.namelist():
   if not name.endswith('.txt'):continue
   text=z.read(name).decode('utf-8',errors='ignore')
   def field(pattern):
    m=re.search(pattern,text,re.M);return m.group(1).strip() if m else ''
   lic=field(r'^LICENSE:\s*(.+)$')
   if lic not in {'CC0','CC BY','CC BY-SA','MIT'}:continue
   smi=field(r'^CH\$SMILES:\s*(.+)$');acc=field(r'^ACCESSION:\s*(.+)$');source=name.split('/')[-2]
   if not smi or smi=='N/A':continue
   by_smiles.setdefault(smi,[]).append((acc,source));acc_source[acc]=source;n_allowed+=1
 print('PROVENANCE scan',len(by_smiles),'SMILES',n_allowed,'records',flush=True)
 prov={k:{'sources':set(),'accessions':set()} for k in known};failed=0
 for i,(smi,recs) in enumerate(by_smiles.items()):
  key=smiles_keys.get(smi)
  if key is None:
   if Chem.MolFromSmiles(smi) is None:failed+=1;continue
   try:key=canonical_group_key(smi)
   except Exception:failed+=1;continue
   smiles_keys[smi]=key
  if key not in prov:raise RuntimeError('canonical index mismatch '+key)
  for acc,src in recs:prov[key]['sources'].add(src);prov[key]['accessions'].add(acc)
  if (i+1)%3000==0:print('PROVENANCE canonical',i+1,flush=True)
 result={'bulk_sha256':sha(bulk),'n_allowed_records_with_smiles':n_allowed,'n_distinct_smiles':len(by_smiles),'failed_smiles':failed,'source_by_accession':acc_source,
  'canonical':{k:{'sources':sorted(v['sources']),'accessions':sorted(v['accessions'])} for k,v in prov.items()}}
 write(path,result);return result


def r3_logits(out,obs):
 path=out/'r3_50_logits.npy'
 if path.exists():return np.load(path)
 import torch
 from src.models.fingerprint import FPNet,prep_peaks,ADDUCT_IX,instr_family
 torch.set_num_threads(4)
 ck=torch.load(ROOT/'artifacts/001-baseline/weights/fp_single_s2.pt',map_location='cpu',weights_only=True)
 net=FPNet(nbits=int(ck['nbits']),d=int(ck['d']),layers=int(ck['layers']));net.load_state_dict(ck['model'],strict=True);net.eval()
 specs=[s for m in obs for s in m['spectra']];n=len(specs);mz=np.zeros((n,128),np.float32);it=np.zeros_like(mz);pad=np.ones((n,128),bool);ce=[]
 for i,s in enumerate(specs):
  p=np.load(ROOT/s['peak_file_ref']);a,b=prep_peaks(p['mz'],p['it'],s['precursor_mz']);length=min(len(a),128);mz[i,:length]=a[:length];it[i,:length]=b[:length];pad[i,:length]=False
  cv=s.get('collision_energy_ev');a=np.asarray([] if cv is None else cv,float).reshape(-1);ce.append(float(a.mean()) if len(a) and np.isfinite(a).all() else 25.)
 args=[mz,it,pad,np.array([s['precursor_mz'] for s in specs],np.float32),np.array([ADDUCT_IX.get(s.get('adduct'),ADDUCT_IX['<unk>']) for s in specs]),np.array([instr_family(s['instrument_raw']) for s in specs]),np.array(ce,np.float32),np.array([1. if s['ionization_mode']=='positive' else -1. for s in specs],np.float32)]
 with torch.no_grad():logits=net(*[torch.as_tensor(a) for a in args]).cpu().numpy()
 if logits.shape!=(59,6930) or not np.isfinite(logits).all():raise RuntimeError('invalid logits')
 np.save(path,logits);print('R3 CPU logits',logits.shape,flush=True);return logits


def evaluate_sources(out,new,maps,union,prov,fp):
 obs=read(ROOT/'artifacts/007-r3/observations_50.json')['molecules'];filed={r['canonical_key']:r for r in read(ROOT/'artifacts/008-expansion/expansion_eval_50.json')['per_molecule']};logits=r3_logits(out,obs)
 tp=ROOT/'artifacts/pool-train';cp=ROOT/'artifacts/005/coco';mass=np.concatenate([np.load(tp/'mass.npy'),np.load(cp/'coco_mass.npy')]);tfp=np.load(tp/'fp.npy',mmap_mode='r');cfp=np.load(cp/'coco_fp.npy',mmap_mode='r');tkeys=np.load(tp/'keys.npy',allow_pickle=True).tolist();cmeta=pickle.load(open(cp/'coco_meta.pkl','rb'));keys=tkeys+list(cmeta['keys']);ntr=len(tkeys);nm=np.array([x['mass_theo'] for x in new]);rows=[];full={};offset=0;mismatch=[]
 for m in obs:
  ss=m['spectra'];q=np.mean(logits[offset:offset+len(ss)].astype(float),axis=0);offset+=len(ss);qacc={s['spectrum_id'] for s in ss};sources={prov['source_by_accession'][a] for a in qacc};qms=[query_neutral_mass_strict(s['precursor_mz'],s.get('adduct')) for s in ss];uncertain=any(x[2] for x in qms);qm=None if uncertain else float(np.median([x[0] for x in qms]));base={};added={}
  if qm is not None:
   ix=np.flatnonzero(abs(mass-qm)<=abs(qm)*10e-6)
   if len(ix):
    pack=np.stack([tfp[j] if j<ntr else cfp[j-ntr] for j in ix]);dots=np.unpackbits(pack,axis=1,bitorder='big')[:,:6930].astype(float)@q
    for j,score in zip(ix,dots):
     k=maps[0 if j<ntr else 1].get(keys[j],keys[j]);base[k]=max(base.get(k,-np.inf),float(score))
   for j in np.flatnonzero(abs(nm-qm)<=abs(qm)*10e-6):added[new[j]['canonical_key']]=float(fp(new[j]['smiles_rep'])@q)
  pools={'base':base,'expanded':{**base,**added},'exclude_query_records':{**base,**{k:v for k,v in added.items() if set(prov['canonical'][k]['accessions'])-qacc}},'exclude_query_contributors':{**base,**{k:v for k,v in added.items() if set(prov['canonical'][k]['sources'])-sources}}}
  orders={label:ranked(scores) for label,scores in pools.items()};truth=m['canonical_key'];r={'key':truth,'class_AB':m['class_AB'],'sources':sorted(sources),'query_accessions':sorted(qacc),'mass_uncertain':uncertain,'query_mass':qm,'n_base':len(base),'n_added':len(added),'new_candidates':list(added),'truth_external_sources':prov['canonical'].get(truth,{}).get('sources',[]),'truth_external_records':len(prov['canonical'].get(truth,{}).get('accessions',[]))}
  for label,order in orders.items():r[label+'_rank']=rank(order,truth);r[label+'_window']=len(order)
  r['full_expanded_coverage']=truth in union or truth in {e['canonical_key'] for e in new}
  r['full_records_excluded_coverage']=truth in union or (truth in prov['canonical'] and bool(set(prov['canonical'][truth]['accessions'])-qacc))
  r['full_contributor_excluded_coverage']=truth in union or (truth in prov['canonical'] and bool(set(prov['canonical'][truth]['sources'])-sources))
  old=filed[truth]
  for field,oldfield in [('base_rank','final_top25_base_rank_oracle'),('expanded_rank','final_top25_expanded_rank_oracle')]:
   if r[field]!=old[oldfield]:mismatch.append({'key':truth,'field':field,'new':r[field],'filed':old[oldfield]})
  rows.append(r);full[truth]={'query_sources':sorted(sources),'scores':pools,'orders':orders}
 output={'caliber':'FP orig-head only; reproduce008 mean-of-spectrum logits; 10ppm; full canonical window competition; no ICE','reproduction_mismatches':mismatch,'strata':{},'rows':rows,'boundary':'Leave-contributor-out removes all external candidate structures with no remaining allowed independent contributor, using source metadata only. It is a transfer stress observation, not evidence against legally including those sources in deployment; base712K unchanged.'}
 for label,rs in [('all',rows),('A',[r for r in rows if r['class_AB']=='A']),('B_all',[r for r in rows if r['class_AB']=='B']),('B_known',[r for r in rows if r['class_AB']=='B' and not r['mass_uncertain']])]:
  output['strata'][label]={variant:summary(rs,'base_rank',variant+'_rank') for variant in ['expanded','exclude_query_records','exclude_query_contributors']}
 for label,rs in [('all',rows),('A',[r for r in rows if r['class_AB']=='A']),('B_all',[r for r in rows if r['class_AB']=='B']),('B_known',[r for r in rows if r['class_AB']=='B' and not r['mass_uncertain']])]:
  for variant,full_field in [('expanded','full_expanded_coverage'),('exclude_query_records','full_records_excluded_coverage'),('exclude_query_contributors','full_contributor_excluded_coverage')]:
   output['strata'][label][variant]['full_index_coverage']=sum(r[full_field] for r in rs)
   output['strata'][label][variant]['mass_window_coverage']=sum(r[variant+'_rank'] is not None for r in rs)
 output['new_index_sizes_by_excluded_contributor']={source:sum(bool(set(prov['canonical'][e['canonical_key']]['sources'])-{source}) for e in new) for source in sorted({source for row in rows for source in row['sources']})}
 write(out/'r3_source_sensitivity.json',output);write(out/'r3_50_full_candidate_scores.json',full)
 print('R3 mismatches',mismatch,flush=True)
 for k,v in output['strata'].items():print('R3',k,json.dumps(v),flush=True)
 return output


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/009-expansion');args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True);t=time.monotonic();entries,new,maps,union=load_index();bits=np.load(ROOT/'artifacts/005/coco/fp_bits.npy');fp=fingerprint_cache(bits)
 print('external',len(entries),'new',len(new),'base_canonical',len(union),flush=True)
 inter=interference(out,new,fp);pro=provenance(out,entries);r3=evaluate_sources(out,new,maps,union,pro,fp)
 write(out/'run_meta.json',{'device':'local CPU; numpy caches and PyTorch CPU only','seconds':time.monotonic()-t,'seed':'no stochastic candidate selection; fixed prior008 method','new_candidates':len(new),'inputs':{str(x.relative_to(ROOT)):sha(x) for x in [ROOT/'artifacts/008-expansion/candidate_index_records.jsonl',ROOT/'artifacts/007-r3/observations_50.json',ROOT/'artifacts/f1-bundle/inputs.json',ROOT/'artifacts/g1-validation/inputs.json']},'script_sha256':sha(Path(__file__)),'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':importlib.metadata.version('torch'),'rdkit':importlib.metadata.version('rdkit')},'fp_checkpoint_sha256':sha(ROOT/'artifacts/001-baseline/weights/fp_single_s2.pt'),'fp_bits_sha256':sha(ROOT/'artifacts/005/coco/fp_bits.npy'),'boundary':'No heldout queries, no GPU, no truth-injected candidates. NP/G1 only FP-stage interference, not full-v9 metric. R3 source removal is a transfer stress, not deployed source filtering.'})
 print('DONE',time.monotonic()-t,flush=True)
if __name__=='__main__':main()

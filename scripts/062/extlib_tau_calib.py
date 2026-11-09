"""062: 离线标定外部谱库 rank-1 覆盖阈值 TAU。
查询 = train 中 enveda-180 timsTOF 谱（按 inchikey14 抽样分子，模拟隐藏测试），
参考 = takumuhata extlib + MassBank harmonized（去 is_casmi_subset），匹配逻辑与 alexchilton 0.484 overlay 完全一致。
输出每个分子的最佳命中 (cos, k, hit_ik14, truth_ik14)，用于估计各 TAU 下的触发率与精确率。"""
import sys, numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
EXT='/workspace/k062/ext'; TRAIN='/workspace/kin/competitions/enveda-CASMI26-molecule-id-mass-spectra/train.parquet'
N_MOL=int(sys.argv[1]) if len(sys.argv)>1 else 3000
MODE=sys.argv[2] if len(sys.argv)>2 else 'enveda'  # enveda=误触发率；cross=真命中余弦分布（非 enveda 库、结构在外部库中）
rng=np.random.default_rng(62)

def ik14(smi):
    try:
        m=Chem.MolFromSmiles(smi); return Chem.MolToInchiKey(m)[:14] if m else ''
    except Exception: return ''

refs=[]
meta=pd.read_parquet(f'{EXT}/ext_meta.parquet'); mass=pd.read_parquet(f'{EXT}/ext_mass.parquet')
mz_all=np.load(f'{EXT}/ext_mz.npy',mmap_mode='r'); in_all=np.load(f'{EXT}/ext_in.npy',mmap_mode='r')
# ext_mass.parquet 无 exact_mass 列（mono_mass 按结构去重，行数不同）→ 与 kernel 一致，不过滤
for idx,row in zip(meta.index, meta.itertuples(index=False)):
    mode=1 if 'pos' in str(row.ionization_mode).lower() else -1
    mz=np.asarray(mz_all[idx],float); it=np.asarray(in_all[idx],float); k=(mz>0)&(it>0); mz,it=mz[k],it[k]
    if len(mz)<3: continue
    o=np.argsort(mz); refs.append((mode,float(row.precursor_mz),mz[o],it[o],str(row.inchikey14),'A'))
print('A',len(refs),flush=True)
mb=pd.read_parquet(f'{EXT}/spectra.parquet',columns=['smiles','inchikey','exact_mass','ion_mode','mzs','intensities','is_casmi_subset'])
mb=mb[(~mb.is_casmi_subset.astype(bool))&mb.smiles.notna()]
for row in mb.itertuples(index=False):
    mode=1 if 'pos' in str(row.ion_mode).lower() else -1
    mz=np.asarray(row.mzs,float); it=np.asarray(row.intensities,float); k=(mz>0)&(it>0); mz,it=mz[k],it[k]
    if len(mz)<3: continue
    key=str(row.inchikey)[:14] if isinstance(row.inchikey,str) and len(row.inchikey)>=14 else ik14(row.smiles)
    o=np.argsort(mz); refs.append((mode,float(row.exact_mass)+1.007276*mode,mz[o],it[o],key,'B'))
del mb
print('A+B',len(refs),flush=True)
order=np.argsort([r[1] for r in refs],kind='stable'); refs=[refs[i] for i in order]; P=np.array([r[1] for r in refs])

def cosine(qm,qi,rm,ri,tol=0.015):
    qa,ra=np.sqrt(qi),np.sqrt(ri); qa=qa/(np.linalg.norm(qa)+1e-12); ra=ra/(np.linalg.norm(ra)+1e-12)
    j0=np.searchsorted(rm,qm-tol,'left'); j1=np.searchsorted(rm,qm+tol,'right')
    pairs=[(qa[i]*ra[j],i,j) for i in range(len(qm)) for j in range(j0[i],j1[i])]
    if not pairs: return 0.0,0
    pairs.sort(reverse=True); ui,uj,s,n=set(),set(),0.0,0
    for p,i,j in pairs:
        if i in ui or j in uj: continue
        ui.add(i);uj.add(j);s+=p;n+=1
    return float(s),n

# 抽样 enveda-180 分子
t=pq.read_table(TRAIN,columns=['inchikey14','ingest_lib','instrument_type']).to_pandas()
if MODE=='enveda':
    sel=(t.ingest_lib=='enveda-180')&(t.instrument_type=='timsTOF')
else:
    ref_keys=set(r[4] for r in refs)
    sel=t.ingest_lib.isin(['pluskal_ms2','enveda-np-examples'])&t.inchikey14.isin(ref_keys)
keys=t.loc[sel,'inchikey14'].unique()
pick=set(rng.choice(keys,size=min(N_MOL,len(keys)),replace=False))
LIBS=['enveda-180'] if MODE=='enveda' else ['pluskal_ms2','enveda-np-examples']
del t
print('enveda mols',len(keys),'pick',len(pick),flush=True)
rows=[]
pf=pq.ParquetFile(TRAIN)
cols=['inchikey14','ingest_lib','instrument_type','ionization_mode','precursor_mz','ms2_mzs','ms2_normalized_intensities']
for g in range(pf.num_row_groups):
    d=pf.read_row_group(g,columns=cols).to_pandas()
    d=d[d.ingest_lib.isin(LIBS)&d.inchikey14.isin(pick)&((d.instrument_type=='timsTOF')|(MODE!='enveda'))]
    rows.append(d)
q=pd.concat(rows); del rows
q=q.groupby('inchikey14').head(4)
print('query spectra',len(q),flush=True)
best={}
for r in q.itertuples(index=False):
    m=np.asarray(r.ms2_mzs,float); it=np.asarray(r.ms2_normalized_intensities,float); prec=float(r.precursor_mz)
    mode=1 if 'pos' in str(r.ionization_mode).lower() else -1
    ok=(m>0)&(it>0)&(m<=prec+0.5); m,it=m[ok],it[ok]; o=np.argsort(m); m,it=m[o],it[o]
    tol=prec*10e-6+0.002; lo=np.searchsorted(P,prec-tol,'left'); hi=np.searchsorted(P,prec+tol,'right')
    b=best.get(r.inchikey14,(0.0,0,'',''))
    for ri in range(lo,hi):
        ref=refs[ri]
        if ref[0]!=mode: continue
        c,n=cosine(m,it,ref[2],ref[3])
        if MODE!='enveda' and c>=0.995: continue
        if n>=4 and c>b[0]: b=(c,n,ref[4],ref[5])
    best[r.inchikey14]=b
out=pd.DataFrame([(k,)+v for k,v in best.items()],columns=['truth','cos','k','hit','src'])
out['correct']=out.truth==out.hit
out.to_csv(f'/workspace/k062/calib_{MODE}_{N_MOL}.csv',index=False)
ext_keys=set(meta.inchikey14)
print('truth in extlib frac',out.truth.isin(ext_keys).mean())
for tau in [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]:
    f=out[out.cos>=tau]
    print(f'TAU {tau:.2f} fire {len(f)/len(out):.4f} n={len(f)} precision {f.correct.mean() if len(f) else float("nan"):.3f}')
for lo_,hi_ in [(0.70,0.75),(0.75,0.80),(0.80,0.85),(0.85,0.90),(0.90,1.01)]:
    f=out[(out.cos>=lo_)&(out.cos<hi_)]; print(f'band {lo_}-{hi_} n={len(f)} prec {f.correct.mean() if len(f) else float("nan"):.3f}')

# 045 offline w_mid sweep (proxy). baseline = prvsiyan FPNet loglik, neural = model3.
import sys, os, random, json, numpy as np, pyarrow.parquet as pq, torch, importlib.util as iu, types
sys.path.insert(0,'.'); torch.set_num_threads(os.cpu_count())
from rdkit import Chem; from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.MolStandardize import rdMolStandardize
B='/workspace/k044out/own_spec'; m_=types.ModuleType('own'); m_.__path__=[B]; sys.modules['own']=m_
for n in ['feats','model']:
    sp=iu.spec_from_file_location('own.'+n,f'{B}/{n}.py'); md=iu.module_from_spec(sp); sys.modules[sp.name]=md; sp.loader.exec_module(md)
feat=sys.modules['own.feats']; OM=sys.modules['own.model']
import pv_fp; _pv=types.ModuleType("pv"); _pv.CFG=None; sys.modules["pv"]=_pv; import casmi_engine as ce
ce.find=lambda n: '/workspace/kin/datasets/prvsiyan/coconut-casmi26-candidates/'+n
P='/workspace/kin/competitions/enveda-CASMI26-molecule-id-mass-spectra/train.parquet'
k14=pq.read_table(P,columns=['inchikey14']).column(0).to_pylist()
ul=sorted(set(k14)); hold=np.random.default_rng(20261025).random(len(ul))<0.02
hk=set(np.array(ul)[hold]); del k14
import pyarrow as pa, pyarrow.compute as pc
_pf=pq.ParquetFile(P); _hv=pa.array(sorted(hk)); _parts=[]
for _i in range(_pf.num_row_groups):
    _t=_pf.read_row_group(_i,columns=['normalized_smiles','inchikey14','ionization_mode','instrument_type','adduct','precursor_mz','ms2_mzs','ms2_normalized_intensities','collision_energy_ev'])
    _parts.append(_t.filter(pc.is_in(_t['inchikey14'],value_set=_hv)))
T=pa.concat_tables(_parts).to_pandas(); del _parts
print('held spectra',len(T),flush=True)
Z=np.load(f'{B}/ext_pool_meta.npz'); byf={}
for s,f in zip(Z['smiles'],Z['formula']): byf.setdefault(str(f),[]).append(str(s))
TE=rdMolStandardize.TautomerEnumerator()
def form(s):
    m=Chem.MolFromSmiles(s); Chem.RemoveStereochemistry(m); return CalcMolFormula(TE.Canonicalize(m))
dev='cpu'; ck=torch.load(f'{B}/model3.pt',map_location=dev,weights_only=True)
net=OM.OwnSpec(**ck['cfg']); net.load_state_dict(ck['state_dict']); net.eval()
r3=json.load(open(f'{B}/train_result3.json'))['val']; fw=float(max(r3['isomer'],key=lambda k:(r3['isomer'][k] or 0)+(r3['pool'][k] or 0)))
D='/workspace/kin/datasets/prvsiyan/casmi26-fp-models-v2/'
single,merged,_=pv_fp.load_fp_models([D+'fp_merged_m1.pt',D+'fp_single_s2.pt'],dev); models=(single,merged,dev)
NMAX=int(sys.argv[1]) if len(sys.argv)>1 else 400
groups=list(T.groupby('inchikey14')); random.Random(0).shuffle(groups)
res=[]
with torch.inference_mode():
  for k,g in groups:
    if len(res)>=NMAX: break
    try:
      s0=g.normalized_smiles.iloc[0]; f=form(s0); pe=byf.get(f,[])
      if len(pe)<5: continue
      cs=[s0]+random.Random(len(res)+1).sample(pe,min(300,len(pe)))
      g=g.head(8); rec=[];specs=[]
      for r in g.itertuples():
        pk,nn=feat.spectrum(r.ms2_mzs,r.ms2_normalized_intensities,r.precursor_mz)
        if nn<3: continue
        me,ad,ins=feat.meta(r.precursor_mz,r.adduct,r.collision_energy_ev,r.instrument_type); rec.append((pk,nn,me,ad,ins))
      if not rec: continue
      for r in g.itertuples():
        a=np.asarray(r.ms2_mzs,np.float32); b=np.asarray(r.ms2_normalized_intensities,np.float32); kk=a<=r.precursor_mz+2
        specs.append((a[kk],b[kk]/max(b[kk].max(),1e-9)))
      modes=np.where(g.ionization_mode.astype(str).values=='positive',1.0,-1.0)
      ces=np.array([ce.parse_ce(v) for v in g.collision_energy_ev.values],np.float32)
      z=pv_fp.molecule_logits(models,specs,g.precursor_mz.values.astype(np.float32),list(g.adduct.astype(str)),list(g.instrument_type.astype(str)),ces,modes)
      fps=np.stack([np.unpackbits(ce.fp_and_mass(s)[0])[:len(z)] for s in cs]).astype(np.float32)
      ls=lambda x:-np.logaddexp(0,-x); bsc=(fps*ls(z)+(1-fps)*ls(-z)).sum(1)
      bsc=np.nan_to_num(bsc,nan=-1e9); keep=[0]+[int(i) for i in np.argsort(-bsc[1:],kind='stable')[:24]+1]; cs=[cs[i] for i in keep]; bsc=bsc[keep]
      F=torch.tensor(np.stack([feat.mol_fp(s)[1] for s in cs])); M=net.molecule(F)
      zz,pp=net.spec(torch.tensor(np.stack([v[0] for v in rec])),torch.tensor([v[1] for v in rec]),torch.tensor(np.stack([v[2] for v in rec])),torch.tensor([v[3] for v in rec]),torch.tensor([v[4] for v in rec]))
      nsc=net.score(zz,pp,M,F,fw).float().numpy().mean(0)
      res.append((bsc,nsc))
    except Exception as e: print('skip',repr(e)[:100]); continue
    if len(res)%50==0: print(len(res),flush=True)
def rk(s): o=np.argsort(-s,kind='stable'); r=np.empty(len(s),int); r[o]=np.arange(1,len(s)+1); return r
def mrr(orders): return round(float(np.mean([1/(1+list(o).index(0)) for o in orders])),4), round(float(np.mean([o[0]==0 for o in orders])),4)
out={}
for name in ['base','neural','V1_0.25','LS_0.5','LS_0.75','LS_1.0']:
    ords=[]
    for bsc,nsc in res:
        b=rk(bsc); n=rk(nsc)
        if name=='base': ords.append(np.argsort(b)); continue
        if name=='neural': ords.append(np.argsort(n)); continue
        v1=1/(3+b)+0.25/(3+n); lead=int(np.argmax(v1))
        if name=='V1_0.25': ords.append(np.argsort(-v1,kind='stable')); continue
        w=float(name.split('_')[1]); s=1/(3+b)+w/(3+n); s[lead]=1e9
        ords.append(np.argsort(-s,kind='stable'))
    out[name]=mrr(ords)
print('N',len(res)); print(json.dumps(out)); json.dump(dict(N=len(res),mrr_top1=out),open('sweep045.json','w'),indent=1)

import pyarrow.parquet as pq, pandas as pd, numpy as np, json, random
random.seed(0)
f=pq.ParquetFile('/workspace/kin/competitions/enveda-CASMI26-molecule-id-mass-spectra/train.parquet')
cols=['normalized_smiles','inchikey14','molecular_formula','ionization_mode','instrument_type','adduct','precursor_mz','ms2_mzs','ms2_normalized_intensities','collision_energy_ev','collision_energy_orig','collision_energy_orig_units']
parts=[]
for i in range(0,f.num_row_groups,max(1,f.num_row_groups//12)):
    d=f.read_row_group(i,columns=cols).to_pandas(); parts.append(d[d.adduct.isin(['[M-H]-','[M+H]+'])])
d=pd.concat(parts); print(len(d), f.num_row_groups)
smi=d.groupby('inchikey14').normalized_smiles.first(); fm=d.groupby('inchikey14').molecular_formula.first()
byf=fm.groupby(fm).groups
g=d.groupby('inchikey14').adduct.agg(set)
both=[k for k,s in g.items() if s=={'[M-H]-','[M+H]+'} and len(byf[fm[k]])>=6]
random.shuffle(both); both=both[:40]; print('mols',len(both))
P=2*1.007276
def sp(r,shift=0.0,add=None):
    return dict(mz=[float(x)+shift for x in r.ms2_mzs], it=[float(x) for x in r.ms2_normalized_intensities],
      prec=float(r.precursor_mz)+shift, adduct=add or r.adduct, mode=1, instrument=r.instrument_type,
      ce_ev=[float(x) for x in (r.collision_energy_ev if r.collision_energy_ev is not None else [])],
      ce_orig=str(r.collision_energy_orig), ce_units=str(r.collision_energy_orig_units))
A,B=[],[]; meta={}
for k in both:
    c=[x for x in byf[fm[k]] if x!=k]; random.shuffle(c); c=[smi[x] for x in c[:15]]+[smi[k]]
    meta[k]=smi[k]; rows=d[d.inchikey14==k]
    neg=rows[rows.adduct=='[M-H]-'].head(3); pos=rows[rows.adduct=='[M+H]+'].head(3)
    A.append(dict(mid=k,cands=c,spectra=[sp(r,P,'[M+H]+') for r in neg.itertuples()]))
    B.append(dict(mid=k,cands=c,spectra=[sp(r) for r in pos.itertuples()]))
json.dump(A,open('negproxy.json','w')); json.dump(B,open('posctrl.json','w')); json.dump(meta,open('truth.json','w'))

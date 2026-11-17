import subprocess,sys
subprocess.run([sys.executable,'-m','pip','install','-q','rdkit'],check=False)
import os, glob, json, time, math, random
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import torch, torch.nn as nn, torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

t0 = time.time()
comp = None
for c in glob.glob('/kaggle/input/**/train.parquet', recursive=True):
    if 'casmi' in c.lower() or 'mass-spectra' in c.lower(): comp = c; break
print('train path', comp, flush=True)
NB = 4000; BIN = 0.25; MAXP = 80
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', device, flush=True)

# ---------- load spectra (timsTOF, [M+H]+) ----------
cols = ['inchikey14','normalized_smiles','adduct','instrument_type','ms2_mzs','ms2_normalized_intensities']
pf = pq.ParquetFile(comp)
frames = []
for b in pf.iter_batches(batch_size=200000, columns=cols):
    d = b.to_pandas()
    d = d[(d.instrument_type == 'timsTOF') & (d.adduct == '[M+H]+')]
    if len(d): frames.append(d)
df = pd.concat(frames, ignore_index=True)
print('rows', len(df), 'structures', df.inchikey14.nunique(), flush=True)

bins_l, ints_l, key_l = [], [], []
for ik, smi, mz, it in zip(df.inchikey14.astype(str), df.normalized_smiles.astype(str),
                           df.ms2_mzs, df.ms2_normalized_intensities):
    if mz is None or it is None: continue
    a = np.asarray(mz, np.float32); b = np.asarray(it, np.float32)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    k = (a > 10) & (b > 0) & np.isfinite(a) & np.isfinite(b)
    a, b = a[k], b[k]
    if len(a) < 3: continue
    bi = np.clip((a / BIN).astype(np.int32), 0, NB - 1)
    if len(bi) > MAXP:
        o = np.argpartition(-b, MAXP)[:MAXP]; bi, b = bi[o], b[o]
    o = np.argsort(bi); bi, b = bi[o], b[o]
    bins_l.append(bi.astype(np.int16)); ints_l.append(b.astype(np.float16)); key_l.append(ik)
print('kept spectra', len(key_l), flush=True)

keys = sorted(set(key_l)); kidx = {k: i for i, k in enumerate(keys)}
row_key = np.array([kidx[k] for k in key_l], np.int32)
print('structures in data', len(keys), flush=True)

# ---------- fingerprints ----------
smi_of = {}
for ik, smi in zip(df.inchikey14.astype(str), df.normalized_smiles.astype(str)):
    if ik not in smi_of: smi_of[ik] = smi
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
FP = np.zeros((len(keys), 2048), np.float16)
for i, k in enumerate(keys):
    m = Chem.MolFromSmiles(smi_of[k])
    if m is None: continue
    b = gen.GetFingerprint(m).ToBitString()
    FP[i] = np.frombuffer(b.encode(), np.uint8).astype(np.float16) - 48.0
print('fps', FP.shape, round(time.time() - t0, 1), flush=True)

# formula for grouping
from rdkit.Chem import rdMolDescriptors
formula = {}
for k in keys:
    m = Chem.MolFromSmiles(smi_of[k])
    formula[k] = rdMolDescriptors.CalcMolFormula(m) if m is not None else None

# ---------- split by structure ----------
rng = random.Random(20261117)
shuf = keys[:]; rng.shuffle(shuf)
nval = max(1, int(0.06 * len(shuf)))
val_keys = set(shuf[:nval]); train_keys = set(shuf[nval:])
tr_rows = np.array([i for i, r in enumerate(row_key) if keys[r] in train_keys], np.int64)
print('train rows', len(tr_rows), 'val structures', len(val_keys), flush=True)

# ---------- model ----------
class Net(nn.Module):
    def __init__(self, din=2048, dh=1024, dout=NB):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, dh), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(dh, dh), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(dh, dout))
    def forward(self, x): return self.net(x)

model = Net().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
FPT = torch.tensor(FP.astype(np.float32))
scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

def make_target(idx_rows):
    y = torch.zeros(len(idx_rows), NB, device=device)
    for j, r in enumerate(idx_rows):
        y[j].index_put_((torch.tensor(bins_l[r].astype(np.int64), device=device),),
                        torch.tensor(ints_l[r].astype(np.float32), device=device), accumulate=True)
    y = y / y.sum(dim=1, keepdim=True).clamp_min(1e-9)
    return y

EPOCHS = 8; BS = 1024
steps = 0; ttrain = time.time()
for ep in range(EPOCHS):
    perm = np.random.permutation(tr_rows)
    tot = 0.0; nb = 0
    for s in range(0, len(perm), BS):
        if time.time() - ttrain > 4200: 
            print('[align] time cap in epoch', ep, flush=True); break
        idx = perm[s:s + BS]
        x = FPT[torch.tensor(row_key[idx], dtype=torch.long)].to(device)
        y = make_target(idx)
        with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
            p = torch.softmax(model(x), dim=1)
            loss = 1.0 - (p * y).sum(dim=1).div(p.norm(dim=1) * y.norm(dim=1) + 1e-9).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tot += float(loss); nb += 1; steps += 1
        if nb % 100 == 0: print(f'[align] ep{ep} step{nb} loss {tot/nb:.4f} {round(time.time()-ttrain,1)}s', flush=True)
    print(f'[align] epoch {ep} done loss {tot/max(nb,1):.4f} elapsed {round(time.time()-t0,1)}s', flush=True)

# ---------- eval helper ----------
@torch.no_grad()
def predict_for(ids):
    x = FPT[torch.tensor(ids, dtype=torch.long)].to(device)
    return torch.softmax(model(x), dim=1).cpu()

def binned_target(row):
    y = torch.zeros(NB)
    y.index_put_((torch.tensor(bins_l[row].astype(np.int64)),), torch.tensor(ints_l[row].astype(np.float32)), accumulate=True)
    return y / y.sum().clamp_min(1e-9)

def eval_groups(cand_pool_keys, k_list, n_query=200, seed=7):
    r = random.Random(seed)
    by_f = {}
    for k in cand_pool_keys: by_f.setdefault(formula.get(k), []).append(k)
    out = {k: dict(hit=0, n=0) for k in k_list}
    qs = [k for k in cand_pool_keys if len(by_f.get(formula.get(k), [])) >= 5]
    r.shuffle(qs); qs = qs[:n_query]
    for qk in qs:
        rows_q = [i for i, rk in enumerate(row_key) if keys[rk] == qk]
        if not rows_q: continue
        qy = binned_target(rows_q[0])
        peers = [x for x in by_f[formula[qk]] if x != qk]
        for K in k_list:
            if len(peers) < K - 1: continue
            cand = [qk] + r.sample(peers, K - 1)
            P = predict_for([kidx[c] for c in cand])
            sc = (P * qy.unsqueeze(0)).sum(dim=1) / (P.norm(dim=1) * qy.norm() + 1e-9)
            out[K]['n'] += 1
            if int(torch.argmax(sc)) == 0: out[K]['hit'] += 1
    return {k: dict(n=v['n'], R1=round(v['hit'] / v['n'], 4) if v['n'] else None) for k, v in out.items()}

va = eval_groups(list(val_keys), [5, 10, 25], n_query=250)
print('[align] VALID in-distribution ' + json.dumps(va), flush=True)

# out-of-distribution: e180
e180 = sorted(glob.glob('/kaggle/input/**/e180_not_in_train.parquet', recursive=True), key=len)
res = dict(val=va, steps=steps, elapsed_s=round(time.time() - t0, 1))
if e180:
    ref = pd.read_parquet(e180[0])
    seen = {}
    for r_ in ref.itertuples(index=False):
        smi = str(r_.smiles); ik = str(r_.ik14)
        m = Chem.MolFromSmiles(smi)
        if m is None or ik in seen: continue
        mz = np.asarray(r_.mzs, float); it = np.asarray(r_.intensities, float)
        k = (mz > 0) & (it > 0); mz, it = mz[k], it[k]
        if len(mz) < 5: continue
        bi = np.clip((mz / BIN).astype(np.int32), 0, NB - 1)
        if len(bi) > MAXP:
            o = np.argpartition(-it, MAXP)[:MAXP]; bi, it = bi[o], it[o]
        y = torch.zeros(NB); y.index_put_((torch.tensor(bi.astype(np.int64)),), torch.tensor(it.astype(np.float32)), accumulate=True)
        fp = np.frombuffer(gen.GetFingerprint(m).ToBitString().encode(), np.uint8).astype(np.float32) - 48.0
        seen[ik] = dict(f=rdMolDescriptors.CalcMolFormula(m), y=y / y.sum().clamp_min(1e-9), fp=fp)
    print('[align] e180 structures', len(seen), flush=True)
    r = random.Random(11)
    by_f = {}
    for ik, v in seen.items(): by_f.setdefault(v['f'], []).append(ik)
    ood = {}
    for K in (5, 10):
        hit = 0; n = 0
        cands = [ik for ik, v in seen.items() if len(by_f.get(v['f'], [])) >= K]
        r.shuffle(cands)
        for qk in cands[:200]:
            peers = [x for x in by_f[seen[qk]['f']] if x != qk]
            cand = [qk] + r.sample(peers, K - 1)
            X = torch.tensor(np.stack([seen[c]['fp'] for c in cand])).to(device)
            with torch.no_grad(): P = torch.softmax(model(X), dim=1).cpu()
            qy = seen[qk]['y']
            sc = (P * qy.unsqueeze(0)).sum(dim=1) / (P.norm(dim=1) * qy.norm() + 1e-9)
            n += 1
            if int(torch.argmax(sc)) == 0: hit += 1
        ood[K] = dict(n=n, R1=round(hit / n, 4) if n else None)
    res['ood_e180'] = ood
    print('[align] E180 5/10-way ' + json.dumps(ood), flush=True)

verdict = None
try:
    r5 = res.get('ood_e180', {}).get(5, {}).get('R1')
    if r5 is not None: verdict = 'PASS' if r5 > 0.44 else 'FAIL(gate >0.44)'
except Exception: pass
res['verdict_vs_ICE_0.4395'] = verdict
json.dump(res, open('/kaggle/working/class2_ranker_result.json', 'w'), indent=2)
torch.save({'state': model.state_dict(), 'nb': NB, 'bin': BIN}, '/kaggle/working/fp2spec.pt')
print('[align] RESULT ' + json.dumps(res), flush=True)

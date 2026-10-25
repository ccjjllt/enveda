# ===== own_spec training (Kaggle GPU). Inputs: competition train.parquet. Output: /kaggle/working/own_spec/
import os, sys, time, json, glob, random, math
import numpy as np, torch, pyarrow.parquet as pq
from multiprocessing import Pool
T0 = time.time()
BUDGET_H = float(os.environ.get('OWN_BUDGET_H', '7.3'))
SMOKE = os.environ.get('OWN_SMOKE') == '1'
OUT = os.environ.get('OWN_OUT', '/kaggle/working/own_spec'); os.makedirs(OUT, exist_ok=True)
TRAIN = (glob.glob('/kaggle/input/**/train.parquet', recursive=True) + [os.environ.get('OWN_TRAIN', '')])[0]
def log(*a): print(f'[{(time.time()-T0)/60:7.1f}m]', *a, flush=True)
COLS = ['normalized_smiles', 'inchikey14', 'molecular_formula', 'adduct', 'precursor_mz', 'ms2_mzs',
        'ms2_normalized_intensities', 'collision_energy_ev', 'instrument_type']

def prep_rg(i):
    t = pq.ParquetFile(TRAIN).read_row_group(i, columns=COLS)
    if SMOKE: t = t.slice(0, 4000)
    d = t.to_pydict(); n = len(d['inchikey14'])
    X = np.zeros((n, K, 3), np.float32); N = np.zeros(n, np.int16); M = np.zeros((n, 3), np.float32)
    A = np.zeros(n, np.int8); I = np.zeros(n, np.int8)
    for j in range(n):
        X[j], N[j] = spectrum(d['ms2_mzs'][j] or [], d['ms2_normalized_intensities'][j] or [], d['precursor_mz'][j] or 0)
        M[j], A[j], I[j] = meta(d['precursor_mz'][j] or 0, d['adduct'][j], d['collision_energy_ev'][j], d['instrument_type'][j])
    keep = N >= 3
    return (X[keep], N[keep], M[keep], A[keep], I[keep], [d['inchikey14'][j] for j in np.nonzero(keep)[0]],
            {d['inchikey14'][j]: (d['normalized_smiles'][j], d['molecular_formula'][j]) for j in range(n)})

def fp_one(s):
    try: return mol_fp(s)[1]
    except Exception: return None

if __name__ == '__main__':
    log('train', TRAIN, 'cuda', torch.cuda.device_count())
    nrg = pq.ParquetFile(TRAIN).num_row_groups
    if SMOKE: nrg = 2
    with Pool(min(4, os.cpu_count())) as p: parts = p.map(prep_rg, range(nrg))
    X = np.concatenate([q[0] for q in parts]); N = np.concatenate([q[1] for q in parts]).astype(np.int64)
    MM = np.concatenate([q[2] for q in parts]); A = np.concatenate([q[3] for q in parts]).astype(np.int64)
    I = np.concatenate([q[4] for q in parts]).astype(np.int64)
    keys = [k for q in parts for k in q[5]]; info = {}
    for q in parts: info.update(q[6])
    del parts
    ulist = sorted(info); kid = {k: i for i, k in enumerate(ulist)}
    Y = np.array([kid[k] for k in keys], np.int64); del keys
    log('spectra', len(Y), 'molecules', len(ulist))
    with Pool(os.cpu_count()) as p: fps = p.map(fp_one, [info[k][0] for k in ulist], chunksize=512)
    good = np.array([f is not None for f in fps])
    FP = np.stack([f if f is not None else np.zeros(FP_BITS, np.uint8) for f in fps]); del fps
    sel = good[Y]; X, N, MM, A, I, Y = X[sel], N[sel], MM[sel], A[sel], I[sel], Y[sel]
    form = [info[k][1] for k in ulist]
    log('fp done', FP.shape, 'spectra kept', len(Y))
    rng = np.random.default_rng(20261025)
    hold = rng.random(len(ulist)) < (0.2 if SMOKE else 0.02)
    by_mol = {}
    for i, y in enumerate(Y.tolist()): by_mol.setdefault(y, []).append(i)
    tr_mols = np.array([m for m in by_mol if not hold[m]]); va_mols = [m for m in by_mol if hold[m]]
    by_form = {}
    for m in tr_mols.tolist(): by_form.setdefault(form[m], []).append(m)
    multi = [v for v in by_form.values() if len(v) >= 2]
    all_form = {}
    for m in range(len(ulist)):
        if good[m]: all_form.setdefault(form[m], []).append(m)
    va = []
    for m in va_mols[:3000]:
        c = [x for x in all_form[form[m]] if x != m][:63]
        if len(c) < (1 if SMOKE else 5): continue
        va.append((random.Random(m).choice(by_mol[m]), [m] + c))
    log('train mols', len(tr_mols), 'val isomer sets', len(va), 'multi-formula groups', len(multi))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = OwnSpec(fp_bits=FP_BITS).to(dev)
    BS = 64 if SMOKE else 1024
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.02)
    scaler = torch.amp.GradScaler(enabled=dev == 'cuda')
    tX, tN, tM, tA, tI = map(torch.from_numpy, (X, N, MM, A, I)); tFP = torch.from_numpy(FP)

    class Both(torch.nn.Module):
        def __init__(s, n): super().__init__(); s.n = n
        def forward(s, x, n, m, a, i, fp): z, fl = s.n.spec(x, n, m, a, i); return z, fl, s.n.molecule(fp)
    both = Both(net); bothp = torch.nn.DataParallel(both) if torch.cuda.device_count() > 1 else both

    def batch():
        mols = []
        while len(mols) < BS // 2:
            g = multi[rng.integers(len(multi))]
            mols += list(rng.choice(g, min(len(g), 8), replace=False))
        mols += list(rng.choice(tr_mols, BS - len(mols)))
        mols = list(dict.fromkeys(int(m) for m in mols))[:BS]
        idx = np.array([by_mol[m][rng.integers(len(by_mol[m]))] for m in mols])
        return idx, np.array(mols)

    def evaluate(fp_ws=(0.0, 0.5, 1.0, 2.0, 4.0)):
        net.eval(); rr = {w: [] for w in fp_ws}
        with torch.no_grad():
            for j in range(0, len(va), 256):
                ch = va[j:j + 256]; ii = torch.tensor([c[0] for c in ch])
                z, fl = net.spec(tX[ii].to(dev), tN[ii].to(dev), tM[ii].to(dev), tA[ii].to(dev), tI[ii].to(dev))
                for k, (_, cands) in enumerate(ch):
                    f = tFP[cands].to(dev); Mz = net.molecule(f)
                    for w in fp_ws:
                        s = net.score(z[k:k + 1], fl[k:k + 1], Mz, f, w)[0]
                        rr[w].append(1.0 / (1 + int((s[1:] > s[0]).sum())))
        net.train(); return {w: round(float(np.mean(v)), 4) for w, v in rr.items()}

    def save(tag):
        torch.save({'state_dict': net.state_dict(), 'cfg': net.cfg}, f'{OUT}/model.pt')
        json.dump(dict(tag=tag, steps=step, val=last_val, n_val=len(va), elapsed_h=(time.time() - T0) / 3600), open(f'{OUT}/train_result.json', 'w'), indent=1)

    step, last_val, lr0 = 0, None, 3e-4
    t_train0 = time.time(); total = BUDGET_H * 3600 - (t_train0 - T0) - 900
    if SMOKE: total = 60
    next_eval = t_train0 + (20 if SMOKE else 1800)
    last_val = evaluate(); log('VAL init (random)', last_val)
    while True:
        frac = (time.time() - t_train0) / total
        if frac >= 1: break
        lr = lr0 * min(1.0, step / 2000 + 1e-3) * 0.5 * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups: g['lr'] = lr
        idx, mols = batch(); ii = torch.from_numpy(idx)
        x = tX[ii].to(dev, non_blocking=True).clone()
        x[..., 1] *= torch.exp(0.15 * torch.randn_like(x[..., 1])).clamp(0.5, 2)
        n = tN[ii].to(dev); f = tFP[torch.from_numpy(mols)].to(dev)
        with torch.autocast(dev, enabled=dev == 'cuda'):
            z, fl, mz = bothp(x, n, tM[ii].to(dev), tA[ii].to(dev), tI[ii].to(dev), f)
        lg = z.float() @ mz.float().T * net.logit_scale.exp().clamp(max=100)
        lab = torch.arange(len(lg), device=dev)
        loss_c = 0.5 * (torch.nn.functional.cross_entropy(lg, lab) + torch.nn.functional.cross_entropy(lg.T, lab))
        loss_f = torch.nn.functional.binary_cross_entropy_with_logits(fl.float(), (f > 0).float())
        loss = loss_c + 20.0 * loss_f
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt); scaler.update(); step += 1
        if step % (20 if SMOKE else 500) == 0: log(f'step {step} lr {lr:.2e} loss_c {loss_c.item():.3f} loss_f {loss_f.item():.4f}')
        if time.time() > next_eval:
            last_val = evaluate(); log('VAL isomer MRR', last_val); save('periodic'); next_eval = time.time() + (20 if SMOKE else 1800)
    last_val = evaluate(); log('FINAL VAL isomer MRR', last_val); save('final')
    import shutil
    for fn in ['feats.py', 'model.py']:
        src = os.environ.get('OWN_SRC', '/kaggle/working') + '/' + fn
        if os.path.exists(src): shutil.copy(src, OUT)
    log('OWN_SPEC DONE')

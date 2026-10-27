import subprocess as _sp, sys as _sys
try:
    import rdkit
except ImportError:
    _sp.run([_sys.executable, '-m', 'pip', 'install', '-q', 'rdkit==2026.3.3'], check=True)
import rdkit; print('rdkit', rdkit.__version__, flush=True)
"""own_spec features (zhan zhao / enveda project, own implementation).
Spectrum -> K peak tokens; molecule -> Morgan count fingerprint. Pure numpy/RDKit."""
import numpy as np
try:
    from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
except Exception:
    pass
K = 64
FP_BITS = 4096
ADDUCTS = ['[M+H]+', '[M-H]-', '[M+Na]+', '[M+K]+', '[M+NH4]+', '[M+Cl]-', '[M+FA-H]-', '[M+H-H2O]+',
           '[M+2H]2+', '[M-2H]2-', '[M-H2O+H]+', '[M+H-NH3]+']
INSTR = ['timstof', 'orbitrap', 'qtof', 'tof', 'itft', 'qft', 'hcd']

def adduct_id(a):
    return ADDUCTS.index(a) + 1 if a in ADDUCTS else 0

def instr_id(s):
    s = str(s or '').lower().replace(' ', '').replace('-', '')
    for i, k in enumerate(INSTR):
        if k in s:
            return i + 1
    return 0

def ce_value(ce):
    try:
        a = np.asarray(ce if ce is not None else [], dtype=np.float64).reshape(-1)
        a = a[np.isfinite(a)]
        return (float(a.mean()), 1.0) if len(a) else (0.0, 0.0)
    except (TypeError, ValueError):
        return 0.0, 0.0

def spectrum(mz, it, prec):
    """-> (K,3) float32 [mz, intensity(0..1), prec-mz] sorted by mz, n peaks."""
    mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
    ok = np.isfinite(mz) & np.isfinite(it) & (mz > 0) & (it > 0) & (mz < float(prec) + 1.0)
    mz, it = mz[ok], it[ok]
    out = np.zeros((K, 3), np.float32)
    if not len(mz):
        return out, 0
    ix = np.argsort(-it, kind='stable')[:K]; ix = ix[np.argsort(mz[ix], kind='stable')]
    mz, it = mz[ix], it[ix] / it[ix].max()
    n = len(mz)
    out[:n, 0] = mz; out[:n, 1] = it; out[:n, 2] = float(prec) - mz
    return out, n

def meta(prec, adduct, ce, instr):
    v, known = ce_value(ce)
    return np.array([float(prec) / 1000.0, np.clip(v, 0, 300) / 100.0, known], np.float32), adduct_id(adduct), instr_id(instr)

def mol_fp(smiles):
    """-> (key14, uint8 count fp FP_BITS) using stereo-stripped canonical tautomer (metric identity)."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.MolStandardize import rdMolStandardize
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return None, None
    Chem.RemoveStereochemistry(m)
    try:
        m = rdMolStandardize.TautomerEnumerator().Canonicalize(m)
    except Exception:
        pass
    g = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
    c = np.minimum(g.GetCountFingerprintAsNumPy(m), 255).astype(np.uint8)
    return Chem.MolToInchiKey(m)[:14], c

"""own_spec dual encoder (own implementation)."""
import math, torch
from torch import nn
from torch.nn import functional as F

def fourier(x, n=16, lo=0.01, hi=2000.0):
    f = torch.exp(torch.linspace(math.log(2 * math.pi / hi), math.log(2 * math.pi / lo), n, device=x.device))
    a = x.unsqueeze(-1) * f
    return torch.cat([a.sin(), a.cos()], -1)

class OwnSpec(nn.Module):
    def __init__(self, d=384, layers=6, heads=8, emb=256, fp_bits=4096, drop=0.1):
        super().__init__()
        self.cfg = dict(d=d, layers=layers, heads=heads, emb=emb, fp_bits=fp_bits, drop=drop)
        self.tok = nn.Sequential(nn.Linear(16 * 2 * 2 + 2, d), nn.GELU(), nn.Linear(d, d))
        self.add = nn.Embedding(16, d); self.ins = nn.Embedding(8, d)
        self.met = nn.Sequential(nn.Linear(3 + 32, d), nn.GELU(), nn.Linear(d, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        L = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(L, layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.zs = nn.Linear(d, emb); self.fph = nn.Linear(d, fp_bits)
        self.mol = nn.Sequential(nn.Linear(fp_bits, 1024), nn.GELU(), nn.Dropout(drop), nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, emb))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def spec(self, x, n, m, a, ins):
        it = x[..., 1:2]
        t = torch.cat([fourier(x[..., 0]), fourier(x[..., 2].clamp(min=0)), it.sqrt(), torch.log1p(100 * it) / math.log(101)], -1)
        t = self.tok(t)
        c = self.cls.expand(len(x), -1, -1) + self.add(a)[:, None] + self.ins(ins)[:, None] + \
            self.met(torch.cat([m, fourier(m[:, 0] * 1000)], -1))[:, None]
        h = torch.cat([c, t], 1)
        pad = torch.arange(x.shape[1] + 1, device=x.device)[None] > n[:, None]
        h = self.ln(self.enc(h, src_key_padding_mask=pad)[:, 0])
        return F.normalize(self.zs(h), dim=-1), self.fph(h)

    def molecule(self, fp):  # fp: uint8 counts
        return F.normalize(self.mol(torch.log1p(fp.float())), dim=-1)

    def score(self, z, fplog, M, fp, fp_w=1.0):
        """z (S,E), fplog (S,B) logits, M (C,E), fp (C,B) counts -> (S,C) score."""
        s = z @ M.T * self.logit_scale.exp()
        b = (fp > 0).float()
        ll = F.logsigmoid(fplog) @ b.T + F.logsigmoid(-fplog) @ (1 - b).T
        return s + fp_w * ll / b.shape[1] * 100.0

import os as _os
if _os.path.isdir('/kaggle/working'):
    for _n, _s in {'feats.py': '"""own_spec features (zhan zhao / enveda project, own implementation).\nSpectrum -> K peak tokens; molecule -> Morgan count fingerprint. Pure numpy/RDKit."""\nimport numpy as np\ntry:\n    from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")\nexcept Exception:\n    pass\nK = 64\nFP_BITS = 4096\nADDUCTS = [\'[M+H]+\', \'[M-H]-\', \'[M+Na]+\', \'[M+K]+\', \'[M+NH4]+\', \'[M+Cl]-\', \'[M+FA-H]-\', \'[M+H-H2O]+\',\n           \'[M+2H]2+\', \'[M-2H]2-\', \'[M-H2O+H]+\', \'[M+H-NH3]+\']\nINSTR = [\'timstof\', \'orbitrap\', \'qtof\', \'tof\', \'itft\', \'qft\', \'hcd\']\n\ndef adduct_id(a):\n    return ADDUCTS.index(a) + 1 if a in ADDUCTS else 0\n\ndef instr_id(s):\n    s = str(s or \'\').lower().replace(\' \', \'\').replace(\'-\', \'\')\n    for i, k in enumerate(INSTR):\n        if k in s:\n            return i + 1\n    return 0\n\ndef ce_value(ce):\n    try:\n        a = np.asarray(ce if ce is not None else [], dtype=np.float64).reshape(-1)\n        a = a[np.isfinite(a)]\n        return (float(a.mean()), 1.0) if len(a) else (0.0, 0.0)\n    except (TypeError, ValueError):\n        return 0.0, 0.0\n\ndef spectrum(mz, it, prec):\n    """-> (K,3) float32 [mz, intensity(0..1), prec-mz] sorted by mz, n peaks."""\n    mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)\n    ok = np.isfinite(mz) & np.isfinite(it) & (mz > 0) & (it > 0) & (mz < float(prec) + 1.0)\n    mz, it = mz[ok], it[ok]\n    out = np.zeros((K, 3), np.float32)\n    if not len(mz):\n        return out, 0\n    ix = np.argsort(-it, kind=\'stable\')[:K]; ix = ix[np.argsort(mz[ix], kind=\'stable\')]\n    mz, it = mz[ix], it[ix] / it[ix].max()\n    n = len(mz)\n    out[:n, 0] = mz; out[:n, 1] = it; out[:n, 2] = float(prec) - mz\n    return out, n\n\ndef meta(prec, adduct, ce, instr):\n    v, known = ce_value(ce)\n    return np.array([float(prec) / 1000.0, np.clip(v, 0, 300) / 100.0, known], np.float32), adduct_id(adduct), instr_id(instr)\n\ndef mol_fp(smiles):\n    """-> (key14, uint8 count fp FP_BITS) using stereo-stripped canonical tautomer (metric identity)."""\n    from rdkit import Chem\n    from rdkit.Chem import rdFingerprintGenerator\n    from rdkit.Chem.MolStandardize import rdMolStandardize\n    m = Chem.MolFromSmiles(str(smiles))\n    if m is None:\n        return None, None\n    Chem.RemoveStereochemistry(m)\n    try:\n        m = rdMolStandardize.TautomerEnumerator().Canonicalize(m)\n    except Exception:\n        pass\n    g = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)\n    c = np.minimum(g.GetCountFingerprintAsNumPy(m), 255).astype(np.uint8)\n    return Chem.MolToInchiKey(m)[:14], c\n', 'model.py': '"""own_spec dual encoder (own implementation)."""\nimport math, torch\nfrom torch import nn\nfrom torch.nn import functional as F\n\ndef fourier(x, n=16, lo=0.01, hi=2000.0):\n    f = torch.exp(torch.linspace(math.log(2 * math.pi / hi), math.log(2 * math.pi / lo), n, device=x.device))\n    a = x.unsqueeze(-1) * f\n    return torch.cat([a.sin(), a.cos()], -1)\n\nclass OwnSpec(nn.Module):\n    def __init__(self, d=384, layers=6, heads=8, emb=256, fp_bits=4096, drop=0.1):\n        super().__init__()\n        self.cfg = dict(d=d, layers=layers, heads=heads, emb=emb, fp_bits=fp_bits, drop=drop)\n        self.tok = nn.Sequential(nn.Linear(16 * 2 * 2 + 2, d), nn.GELU(), nn.Linear(d, d))\n        self.add = nn.Embedding(16, d); self.ins = nn.Embedding(8, d)\n        self.met = nn.Sequential(nn.Linear(3 + 32, d), nn.GELU(), nn.Linear(d, d))\n        self.cls = nn.Parameter(torch.zeros(1, 1, d))\n        L = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True, activation=\'gelu\')\n        self.enc = nn.TransformerEncoder(L, layers, enable_nested_tensor=False)\n        self.ln = nn.LayerNorm(d)\n        self.zs = nn.Linear(d, emb); self.fph = nn.Linear(d, fp_bits)\n        self.mol = nn.Sequential(nn.Linear(fp_bits, 1024), nn.GELU(), nn.Dropout(drop), nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, emb))\n        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))\n\n    def spec(self, x, n, m, a, ins):\n        it = x[..., 1:2]\n        t = torch.cat([fourier(x[..., 0]), fourier(x[..., 2].clamp(min=0)), it.sqrt(), torch.log1p(100 * it) / math.log(101)], -1)\n        t = self.tok(t)\n        c = self.cls.expand(len(x), -1, -1) + self.add(a)[:, None] + self.ins(ins)[:, None] + \\\n            self.met(torch.cat([m, fourier(m[:, 0] * 1000)], -1))[:, None]\n        h = torch.cat([c, t], 1)\n        pad = torch.arange(x.shape[1] + 1, device=x.device)[None] > n[:, None]\n        h = self.ln(self.enc(h, src_key_padding_mask=pad)[:, 0])\n        return F.normalize(self.zs(h), dim=-1), self.fph(h)\n\n    def molecule(self, fp):  # fp: uint8 counts\n        return F.normalize(self.mol(torch.log1p(fp.float())), dim=-1)\n\n    def score(self, z, fplog, M, fp, fp_w=1.0):\n        """z (S,E), fplog (S,B) logits, M (C,E), fp (C,B) counts -> (S,C) score."""\n        s = z @ M.T * self.logit_scale.exp()\n        b = (fp > 0).float()\n        ll = F.logsigmoid(fplog) @ b.T + F.logsigmoid(-fplog) @ (1 - b).T\n        return s + fp_w * ll / b.shape[1] * 100.0\n'}.items():
        open('/kaggle/working/' + _n, 'w').write(_s)

# ===== 044 own_spec hard-negative fine-tune (Kaggle GPU). Inputs: train.parquet + PubChemLite/COCONUT pools + ccjjllt/enveda-own-spec.
# 新增损失：每个锚点在「同分子式候选池」(训练库 + PubChemLite + COCONUT，与测试候选同源) 中在线挖掘当前模型排得最高的异构体，
# 用推理时的完整打分 (cos*scale + fp_w*指纹对数似然) 做 listwise CE，直接优化重排时真正使用的分数。
import os, sys, time, json, glob, random, math, pickle
import numpy as np, torch, pyarrow.parquet as pq
from multiprocessing import Pool
from torch.nn import functional as F
T0 = time.time()
BUDGET_H = float(os.environ.get('OWN_BUDGET_H', '10.3'))
SMOKE = os.environ.get('OWN_SMOKE') == '1'
OUT = os.environ.get('OWN_OUT', '/kaggle/working/own_spec'); os.makedirs(OUT, exist_ok=True)
TRAIN = (glob.glob('/kaggle/input/**/train.parquet', recursive=True) + [os.environ.get('OWN_TRAIN', '')])[0]
IN = os.environ.get('OWN_IN', '/kaggle/input')
EXT_CAP_MIN = 3.0 if SMOKE else float(os.environ.get('OWN_EXT_CAP_MIN', '110'))
HN_A = 32 if SMOKE else 384      # anchors per step with hard-neg loss
HN_P = 64                        # pool candidates sampled per anchor (mining)
HN_K = 24                        # mined hard negatives kept per anchor
HN_W = float(os.environ.get('OWN_HN_W', '1.0')); HN_FPW = 1.0
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
            {d['inchikey14'][j]: d['normalized_smiles'][j] for j in range(n)})

def fp_sparse(s):
    """-> (key14, rdkit formula, nz idx uint16, counts uint8) or None. Same canonicalisation as mol_fp (inference)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula
        from rdkit.Chem.MolStandardize import rdMolStandardize
        m = Chem.MolFromSmiles(str(s))
        if m is None: return None
        Chem.RemoveStereochemistry(m)
        try: m = rdMolStandardize.TautomerEnumerator().Canonicalize(m)
        except Exception: pass
        g = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
        c = np.minimum(g.GetCountFingerprintAsNumPy(m), 255).astype(np.uint8); nz = np.nonzero(c)[0]
        return Chem.MolToInchiKey(m)[:14], CalcMolFormula(m), nz.astype(np.uint16), c[nz]
    except Exception:
        return None

def fp_many(smiles, cap_min=None, tag=''):
    res = [None] * len(smiles); t = time.time(); done = 0
    with Pool(os.cpu_count()) as p:
        for i, r in enumerate(p.imap(fp_sparse, smiles, chunksize=256)):
            res[i] = r; done = i + 1
            if done % 100000 == 0: log(tag, 'fp', done, '/', len(smiles))
            if cap_min and time.time() - t > cap_min * 60:
                log(tag, 'fp time cap hit at', done); p.terminate(); break
    return res, done

def dense(rs):
    o = np.zeros((len(rs), FP_BITS), np.uint8)
    for i, r in enumerate(rs):
        if r is not None: o[i, r[2].astype(np.int64)] = r[3]
    return o

if __name__ == '__main__':
    log('train', TRAIN, 'cuda', torch.cuda.device_count(), 'cpus', os.cpu_count())
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
    U = len(ulist); log('spectra', len(Y), 'molecules', U)
    rs, _ = fp_many([info[k] for k in ulist], tag='train')
    good = np.array([r is not None for r in rs]); assert good.mean() > 0.9, ('fingerprint failure', good.mean())
    form = [r[1] if r is not None else '' for r in rs]
    FP = dense(rs); del rs
    sel = good[Y]; X, N, MM, A, I, Y = X[sel], N[sel], MM[sel], A[sel], I[sel], Y[sel]
    log('train fp done', FP.shape, 'spectra kept', len(Y))
    # ---- external candidate pools (same sources the test-time candidate generator uses)
    ext_smi, ext_src = [], []
    for nm in ['pubchemlite_meta.pkl', 'coco_meta.pkl']:
        fs = glob.glob(f'{IN}/**/{nm}', recursive=True)
        if not fs: log('missing pool', nm); continue
        m = pickle.load(open(fs[0], 'rb')); s = [str(x) for x in m['smiles']]
        if SMOKE: s = s[:3000]
        ext_smi += s; ext_src += [nm] * len(s); log('pool', nm, len(s))
    assert SMOKE or len(ext_smi) > 300000, 'external pools not mounted'
    ers, edone = fp_many(ext_smi, EXT_CAP_MIN, 'ext')
    seen = set(ulist); keep_e = []
    for i, r in enumerate(ers[:edone]):
        if r is not None and r[0] not in seen:
            seen.add(r[0]); keep_e.append(i)
    ers = [ers[i] for i in keep_e]; E = len(ers)
    EFP = dense(ers); eform = [r[1] for r in ers]
    np.savez_compressed(f'{OUT}/ext_pool_meta.npz', key=np.array([r[0] for r in ers]), formula=np.array(eform),
                        smiles=np.array([ext_smi[i] for i in keep_e]))
    del ers, ext_smi
    log('ext pool kept', E, 'of', edone)
    ALLFP = np.concatenate([FP, EFP]); del EFP
    allform = form + eform
    rng = np.random.default_rng(20261025)  # fixed: same holdout as 039/042
    hold = rng.random(U) < (0.2 if SMOKE else 0.02)
    by_mol = {}
    for i, y in enumerate(Y.tolist()): by_mol.setdefault(y, []).append(i)
    tr_mols = np.array([m for m in by_mol if not hold[m]]); va_mols = [m for m in by_mol if hold[m]]
    by_form = {}
    for m in tr_mols.tolist(): by_form.setdefault(form[m], []).append(m)
    multi = [v for v in by_form.values() if len(v) >= 2]
    all_form = {}
    for m in range(U):
        if good[m]: all_form.setdefault(form[m], []).append(m)
    # pool per formula for hard negatives: train mols (not held out) + external
    pool = {}
    for m in tr_mols.tolist(): pool.setdefault(form[m], []).append(m)
    for j, f in enumerate(eform): pool.setdefault(f, []).append(U + j)
    pool = {f: np.array(v, np.int64) for f, v in pool.items()}
    va, vp = [], []
    for m in va_mols[:3000]:
        c = [x for x in all_form[form[m]] if x != m][:63]
        if len(c) >= (1 if SMOKE else 5): va.append((random.Random(m).choice(by_mol[m]), [m] + c))
        pe = [x for x in pool.get(form[m], np.zeros(0, np.int64)).tolist() if x >= U]
        if len(pe) >= (1 if SMOKE else 5):
            pe = random.Random(m + 1).sample(pe, min(63, len(pe)))
            vp.append((random.Random(m).choice(by_mol[m]), [m] + pe))
    has_pool = np.array([len(pool.get(form[m], ())) >= 2 for m in range(U)])
    log('train mols', len(tr_mols), 'val isomer sets', len(va), 'val pool sets', len(vp), 'multi groups', len(multi),
        'anchors w/ pool', int(has_pool[tr_mols].sum()))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(int(os.environ.get('OWN_SEED', '2')))
    net = OwnSpec(fp_bits=FP_BITS).to(dev)
    _init = sorted(glob.glob(f'{IN}/**/model2.pt', recursive=True))
    if os.environ.get('OWN_INIT', '1') == '1':
        assert _init or SMOKE, 'warm start model2.pt not found'
        if _init:
            net.load_state_dict(torch.load(_init[0], map_location=dev)['state_dict']); log('warm start from', _init[0])
    rng2 = np.random.default_rng(int(os.environ.get('OWN_SEED', '2')) + 7)
    BS = 64 if SMOKE else 1024
    lr0 = float(os.environ.get('OWN_LR', '2e-4'))
    opt = torch.optim.AdamW(net.parameters(), lr=lr0, weight_decay=0.02)
    scaler = torch.amp.GradScaler(enabled=dev == 'cuda')
    tX, tN, tM, tA, tI = map(torch.from_numpy, (X, N, MM, A, I))
    gFP = torch.from_numpy(ALLFP).to(dev); del ALLFP   # (U+E, 4096) uint8 on GPU0 (~3.5GB)
    log('fp table on', dev, tuple(gFP.shape))

    class Both(torch.nn.Module):
        def __init__(s, n): super().__init__(); s.n = n
        def forward(s, x, n, m, a, i, fp): z, fl = s.n.spec(x, n, m, a, i); return z, fl, s.n.molecule(fp)
    class Mol(torch.nn.Module):
        def __init__(s, n): super().__init__(); s.n = n
        def forward(s, fp): return s.n.molecule(fp)
    ngpu = torch.cuda.device_count()
    both = Both(net); bothp = torch.nn.DataParallel(both) if ngpu > 1 else both
    molp = torch.nn.DataParallel(Mol(net)) if ngpu > 1 else Mol(net)

    def batch():
        rng = rng2; mols = []
        while len(mols) < BS // 2:
            g = multi[rng.integers(len(multi))]
            mols += list(rng.choice(g, min(len(g), 8), replace=False))
        mols += list(rng.choice(tr_mols, BS - len(mols)))
        mols = list(dict.fromkeys(int(m) for m in mols))[:BS]
        idx = np.array([by_mol[m][rng.integers(len(by_mol[m]))] for m in mols])
        return idx, np.array(mols)

    def pair_score(z, fl, Mz, b, fpw):
        """z (A,E), fl (A,B), Mz (A,H,E), b (A,H,B) binary -> (A,H) — same formula as OwnSpec.score."""
        s = torch.einsum('ae,ahe->ah', z, Mz) * net.logit_scale.exp().clamp(max=100)
        ls1, ls0 = F.logsigmoid(fl), F.logsigmoid(-fl)
        ll = torch.einsum('ab,ahb->ah', ls1 - ls0, b) + ls0.sum(-1, keepdim=True)
        return s + fpw * ll / b.shape[-1] * 100.0

    def evaluate(sets, fp_ws=(0.0, 0.5, 1.0, 2.0, 4.0)):
        net.eval(); rr = {w: [] for w in fp_ws}
        with torch.no_grad():
            for j in range(0, len(sets), 256):
                ch = sets[j:j + 256]; ii = torch.tensor([c[0] for c in ch])
                z, fl = net.spec(tX[ii].to(dev), tN[ii].to(dev), tM[ii].to(dev), tA[ii].to(dev), tI[ii].to(dev))
                for k, (_, cands) in enumerate(ch):
                    f = gFP[torch.tensor(cands, device=dev)]; Mz = net.molecule(f)
                    for w in fp_ws:
                        s = net.score(z[k:k + 1], fl[k:k + 1], Mz, f, w)[0]
                        rr[w].append(1.0 / (1 + int((s[1:] > s[0]).sum())))
        net.train(); return {w: round(float(np.mean(v)), 4) if v else None for w, v in rr.items()}

    def ev():
        return dict(isomer=evaluate(va), pool=evaluate(vp))

    hist = []
    def save(tag):
        torch.save({'state_dict': net.state_dict(), 'cfg': net.cfg}, f'{OUT}/model3.pt')
        json.dump(dict(tag=tag, steps=step, val=last_val, init_val=init_val, hist=hist, n_val=len(va), n_val_pool=len(vp),
                       ext_pool=E, hn=dict(A=HN_A, P=HN_P, K=HN_K, W=HN_W, fpw=HN_FPW), lr=lr0,
                       elapsed_h=(time.time() - T0) / 3600), open(f'{OUT}/train_result3.json', 'w'), indent=1)

    def hard_negs(mols):
        """for anchors (train mol ids) -> (anchor positions, cand matrix (A, P) global ids or -1)."""
        pos = [i for i, m in enumerate(mols.tolist()) if has_pool[m]]
        rng2.shuffle(pos); pos = pos[:HN_A]
        C = np.full((len(pos), HN_P), -1, np.int64)
        for r, i in enumerate(pos):
            m = int(mols[i]); pl = pool[form[m]]; pl = pl[pl != m]
            if len(pl) > HN_P: pl = pl[rng2.choice(len(pl), HN_P, replace=False)]
            C[r, :len(pl)] = pl
        return np.array(pos, np.int64), C

    step = 0
    t_train0 = time.time(); total = BUDGET_H * 3600 - (t_train0 - T0) - 1200
    if SMOKE: total = 60
    next_eval = t_train0 + (20 if SMOKE else 1800)
    init_val = last_val = ev(); log('VAL init', last_val)
    while True:
        frac = (time.time() - t_train0) / total
        if frac >= 1: break
        lr = lr0 * min(1.0, step / 1000 + 1e-3) * 0.5 * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups: g['lr'] = lr
        idx, mols = batch(); ii = torch.from_numpy(idx)
        x = tX[ii].to(dev, non_blocking=True).clone()
        x[..., 1] *= torch.exp(0.15 * torch.randn_like(x[..., 1])).clamp(0.5, 2)
        n = tN[ii].to(dev); f = gFP[torch.from_numpy(mols).to(dev)]
        hp, C = hard_negs(mols)
        with torch.autocast(dev, enabled=dev == 'cuda'):
            z, fl, mz = bothp(x, n, tM[ii].to(dev), tA[ii].to(dev), tI[ii].to(dev), f)
        lg = z.float() @ mz.float().T * net.logit_scale.exp().clamp(max=100)
        lab = torch.arange(len(lg), device=dev)
        loss_c = 0.5 * (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab))
        loss_f = F.binary_cross_entropy_with_logits(fl.float(), (f > 0).float())
        loss_h = torch.zeros((), device=dev)
        if len(hp):
            tC = torch.from_numpy(C).to(dev); valid = tC >= 0; hpt = torch.from_numpy(hp).to(dev)
            za, fla = z[hpt].float(), fl[hpt].float()
            with torch.no_grad(), torch.autocast(dev, enabled=dev == 'cuda'):   # mining with current model
                cf = gFP[tC.clamp(min=0).reshape(-1)]
                Mc = molp(cf).float().reshape(len(hp), HN_P, -1)
                sc = pair_score(za.detach(), fla.detach(), Mc, (cf > 0).float().reshape(len(hp), HN_P, -1), HN_FPW)
                sc = sc.masked_fill(~valid, -1e9)
                top = sc.topk(min(HN_K, HN_P), dim=1).indices
            sel_ids = tC.gather(1, top); sel_ok = valid.gather(1, top)
            ids = torch.cat([torch.from_numpy(mols).to(dev)[hpt][:, None], sel_ids.clamp(min=0)], 1)   # col 0 = positive
            cf2 = gFP[ids.reshape(-1)]
            with torch.autocast(dev, enabled=dev == 'cuda'):
                M2 = molp(cf2).float().reshape(len(hp), ids.shape[1], -1)
            s2 = pair_score(za, fla, M2, (cf2 > 0).float().reshape(len(hp), ids.shape[1], -1), HN_FPW)
            s2 = s2.masked_fill(~torch.cat([torch.ones_like(sel_ok[:, :1]), sel_ok], 1), -1e4)
            loss_h = F.cross_entropy(s2, torch.zeros(len(hp), dtype=torch.long, device=dev))
        loss = loss_c + 20.0 * loss_f + HN_W * loss_h
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt); scaler.update(); step += 1
        if step % (20 if SMOKE else 500) == 0:
            log(f'step {step} lr {lr:.2e} loss_c {loss_c.item():.3f} loss_f {loss_f.item():.4f} loss_h {loss_h.item():.3f} hnA {len(hp)}')
        if time.time() > next_eval:
            last_val = ev(); hist.append((step, last_val)); log('VAL', last_val); save('periodic'); next_eval = time.time() + (20 if SMOKE else 1800)
    last_val = ev(); log('FINAL VAL', last_val); save('final')
    import shutil
    for fn in ['feats.py', 'model.py']:
        src = os.environ.get('OWN_SRC', '/kaggle/working') + '/' + fn
        if os.path.exists(src): shutil.copy(src, OUT)
    log('OWN_SPEC_HN DONE')

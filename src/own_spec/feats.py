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

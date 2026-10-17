"""015 候选级参考谱证据的纯 CPU 计算，不包含标签或训练逻辑。

谱余弦使用固定 0.01 Da 容差的 direct/precursor-shift 联合匹配边，
按强度乘积贪心一对一；shift只允许同极性、单电荷、单体条件。Morgan 为去立体、互变异构规范结构的
radius=2 / 2048 bit 指纹。参考/候选选择由调用方冻结。
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import math

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

MASS = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196,
            Na=22.9897692809, K=38.96370668, Cl=34.96885268)
ELECTRON = 0.00054857990
PROTON = MASS['H'] - ELECTRON
WATER = 2 * MASS['H'] + MASS['O']
AMMONIUM = MASS['N'] + 4 * MASS['H'] - ELECTRON
FORMIC = MASS['C'] + 2 * MASS['H'] + 2 * MASS['O']
ACETIC = 2 * MASS['C'] + 4 * MASS['H'] + 2 * MASS['O']
ADDUCT_NZD = {
    '[M+H]+': (1, 1, PROTON), '[M+NH4]+': (1, 1, AMMONIUM),
    '[M+Na]+': (1, 1, MASS['Na'] - ELECTRON), '[M+K]+': (1, 1, MASS['K'] - ELECTRON),
    '[M-H2O+H]+': (1, 1, PROTON - WATER), '[M-2H2O+H]+': (1, 1, PROTON - 2 * WATER),
    '[M+2H]2+': (1, 2, 2 * PROTON), '[M]+': (1, 1, -ELECTRON),
    '[M-H2O]+': (1, 1, -ELECTRON - WATER),
    '[M+CH3OH+H]+': (1, 1, PROTON + MASS['C'] + 4 * MASS['H'] + MASS['O']),
    '[M+CH3CN+H]+': (1, 1, PROTON + 2 * MASS['C'] + 3 * MASS['H'] + MASS['N']),
    '[M-H]-': (1, 1, -PROTON), '[M-H2O-H]-': (1, 1, -PROTON - WATER),
    '[M+CH2O2-H]-': (1, 1, FORMIC - PROTON),
    '[M+C2H4O2-H]-': (1, 1, ACETIC - PROTON),
    '[M+Cl]-': (1, 1, MASS['Cl'] + ELECTRON), '[M]-': (1, 1, ELECTRON),
    '[M-2H]-': (1, 2, -2 * PROTON), '[M-2H]2-': (1, 2, -2 * PROTON),
    '[M+Na-2H]-': (1, 1, MASS['Na'] - 2 * PROTON),
    '[2M+H]+': (2, 1, PROTON), '[2M+Na]+': (2, 1, MASS['Na'] - ELECTRON),
    '[2M-H]-': (2, 1, -PROTON), '[2M+NH4]+': (2, 1, AMMONIUM),
    '[2M+K]+': (2, 1, MASS['K'] - ELECTRON),
    '[2M+CH2O2-H]-': (2, 1, FORMIC - PROTON),
    '[2M+C2H4O2-H]-': (2, 1, ACETIC - PROTON),
    '[2M+Na-2H]-': (2, 1, MASS['Na'] - 2 * PROTON),
    '[3M+H]+': (3, 1, PROTON), '[3M-H]-': (3, 1, -PROTON),
}
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
TAUTOMER = rdMolStandardize.TautomerEnumerator()


def neutral_mass(precursor, adduct):
    values = ADDUCT_NZD.get(adduct)
    if values is None or precursor is None or not math.isfinite(float(precursor)):
        return math.nan
    n, z, delta = values
    return (float(precursor) * z - delta) / n


def exact_peak_hash(mz, intensity):
    """Raw float64 byte identity; no sorting, scaling or rounding."""
    digest = hashlib.sha256()
    for values in (mz, intensity):
        arr = np.asarray(values, dtype='<f8')
        digest.update(np.asarray([len(arr)], dtype='<i8').tobytes())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def clean_peaks(mz, intensity, precursor, adduct, floor=0.002, topk=256):
    """Frozen 015 hypothesis: floor/topk then precursor/monomer removal and sqrt.

    Return a signal-retention audit as well as L2-normalized sqrt intensities.
    The retained fraction is relative to positive finite raw signal and to
    floor/topk signal separately; it is not a training feature.
    """
    mz, intensity = np.asarray(mz, dtype=np.float64), np.asarray(intensity, dtype=np.float64)
    if len(mz) != len(intensity):
        raise ValueError('Peak length mismatch')
    n_original = len(mz)
    valid = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
    mz, intensity = mz[valid], intensity[valid]
    raw_signal = float(intensity.sum())
    if len(mz):
        keep = intensity >= floor * intensity.max()
        mz, intensity = mz[keep], intensity[keep]
    if len(mz) > topk:
        keep = np.argsort(-intensity, kind='stable')[:topk]
        mz, intensity = mz[keep], intensity[keep]
    n_clean = len(mz)
    clean_signal = float(intensity.sum())
    keep = np.abs(mz - float(precursor)) > 1.5
    n, z, delta = ADDUCT_NZD[adduct]
    monomer_mz = (neutral_mass(precursor, adduct) + delta) / z if n > 1 else None
    if monomer_mz is not None:
        keep &= np.abs(mz - monomer_mz) > 1.5
    mz, intensity = mz[keep], intensity[keep]
    remaining = float(intensity.sum())
    audit = {'n_original': n_original, 'n_before_removal': n_clean,
             'n_after_removal': len(mz), 'signal_retained_raw': remaining / raw_signal if raw_signal else 0.0,
             'signal_retained_clean': remaining / clean_signal if clean_signal else 0.0,
             'removed_intact_monomer_mz': monomer_mz}
    order = np.argsort(mz, kind='stable')
    mz, intensity = mz[order], np.sqrt(intensity[order])
    norm = np.linalg.norm(intensity)
    return mz, intensity / norm if norm else intensity, audit


def loss_shift_allowed(query_adduct, ref_adduct):
    a, b = ADDUCT_NZD.get(query_adduct), ADDUCT_NZD.get(ref_adduct)
    return bool(a and b and a[:2] == (1, 1) and b[:2] == (1, 1)
                and query_adduct[-1:] == ref_adduct[-1:])


def joint_cosine(qmz, qint, rmz, rint, shift=None, tolerance=0.01):
    """All direct/shift matching edges, product-descending greedy one-to-one.

    Shared edge is represented once (direct wins exact channel ties). This is
    a greedy union matcher; it is not a globally optimal assignment.
    """
    edges = {}
    for delta, channel in [(0.0, 'direct')] + ([(shift, 'shift')] if shift is not None else []):
        low = np.searchsorted(rmz, qmz - delta - tolerance, side='left')
        high = np.searchsorted(rmz, qmz - delta + tolerance, side='right')
        for i in np.flatnonzero(high > low):
            for j in range(int(low[i]), int(high[i])):
                edges.setdefault((int(i), j), (float(qint[i] * rint[j]), channel))
    used_query, used_ref, contributions, channels = set(), set(), [], []
    for (i, j), (product, channel) in sorted(edges.items(), key=lambda x: (-x[1][0], x[0][0], x[0][1])):
        if i not in used_query and j not in used_ref:
            used_query.add(i)
            used_ref.add(j)
            contributions.append(product)
            channels.append(channel)
    total = float(sum(contributions))
    cosine = min(1.0, max(0.0, total))
    fraction = min(1.0, max(0.0, max(contributions) / total)) if total else 0.0
    count = len(contributions)
    quality = cosine ** 3.2 * min(count / 3.0, 1.0) * max(0.0, 1.0 - fraction)
    return {'cosine': cosine, 'quality': quality, 'matched_count': count,
            'largest_pair_fraction': fraction, 'direct_matches': channels.count('direct'),
            'shift_matches': channels.count('shift')}


def spectral_evidence(query, reference):
    """Scores depend only on observed spectra and their metadata."""
    allowed = loss_shift_allowed(query['adduct'], reference['adduct'])
    shift = query['precursor_mz'] - reference['precursor_mz'] if allowed else None
    return {**joint_cosine(query['mz'], query['intensity'], reference['mz'], reference['intensity'], shift),
            'shift_available': allowed, 'shift_da': shift}


@lru_cache(maxsize=100000)
def canonical_morgan(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Invalid structure for Morgan fingerprint')
    Chem.RemoveStereochemistry(mol)
    mol = TAUTOMER.Canonicalize(mol)
    return MORGAN.GetFingerprint(mol)


def tanimoto_to_refs(candidate_smiles, reference_smiles):
    fp = canonical_morgan(candidate_smiles)
    refs = [canonical_morgan(s) for s in reference_smiles]
    return np.asarray(DataStructs.BulkTanimotoSimilarity(fp, refs), dtype=np.float64)

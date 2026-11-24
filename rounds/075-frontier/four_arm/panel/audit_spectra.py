"""Compare independent panel spectra to original competition training spectra in measured precursor windows."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def prepare_peaks(mz, values, precursor):
    mz = np.asarray(mz, dtype=float)
    values = np.asarray(values, dtype=float)
    good = (mz > 0) & (mz < precursor - 0.5) & np.isfinite(mz) & np.isfinite(values) & (values > 0)
    mz, values = mz[good], values[good]
    if not len(mz):
        return mz, values
    keep = values >= 0.005 * values.max()
    mz, values = mz[keep], values[keep]
    keep = np.argsort(-values, kind='stable')[:100]
    intensity = np.sqrt(values[keep])
    return mz[keep], intensity / np.linalg.norm(intensity)


def cosine(query, ref):
    qm, qi = query
    rm, ri = ref
    q, r = np.nonzero(np.abs(qm[:, None] - rm) <= np.maximum(.005, qm[:, None] * 10e-6))
    weights = qi[q] * ri[r]
    uq, ur, score = set(), set(), 0.
    for i in np.argsort(-weights, kind='stable'):
        a, b = q[i], r[i]
        if a not in uq and b not in ur:
            uq.add(a)
            ur.add(b)
            score += weights[i]
    return float(score), len(uq), len(qm), len(rm)


def audit(blind, training):
    queries = pq.read_table(blind).to_pylist()
    queries.sort(key=lambda row: row['precursor_mz'])
    precursors = np.array([r['precursor_mz'] for r in queries])
    peaks = [prepare_peaks(row['ms2_mzs'], row['ms2_normalized_intensities'], row['precursor_mz']) for row in queries]
    diagnostics = [dict(spectrum_id=row['spectrum_id'], molecule_id=row['molecule_id'], checked=0, maximum=None, strong=0) for row in queries]
    columns = ['ingest_lib','inchikey14','precursor_mz','ionization_mode','ms2_mzs','ms2_normalized_intensities']
    for batch in pq.ParquetFile(training).iter_batches(columns=columns, batch_size=16384):
        prec = np.array(batch[2])
        lo = np.searchsorted(precursors, prec / (1 + 20e-6), side='left')
        hi = np.searchsorted(precursors, prec / (1 - 20e-6), side='right')
        window = (hi > lo) & np.isfinite(prec)
        if not window.any():
            continue
        for row, a, b in zip(batch.filter(pa.array(window)).to_pylist(), lo[window], hi[window]):
            ref = prepare_peaks(row['ms2_mzs'],row['ms2_normalized_intensities'],row['precursor_mz'])
            for j in range(a,b):
                if row['ionization_mode'] != queries[j]['ionization_mode']:
                    continue
                d = diagnostics[j]
                d['checked'] += 1
                score, shared, nq, nr = cosine(peaks[j], ref)
                if d['maximum'] is None or score > d['maximum']['cosine']:
                    d['maximum'] = dict(cosine=score, matched_peaks=shared, query_peaks=nq, training_peaks=nr,
                                        train_key=row['inchikey14'], train_source=row['ingest_lib'])
                if score >= .97 and shared >= 4 and shared >= .8 * nq and shared >= .8 * nr:
                    d['strong'] += 1
    return dict(blind_sha256=sha256(blind.read_bytes()).hexdigest(), training_reference='Competition train.parquet',
                training_rows=pq.ParquetFile(training).metadata.num_rows,
                method='All train rows with same measured polarity and precursor within 20 ppm. Remove peaks within 0.5 Da of precursor and above, keep top 100 at >=0.5% of base peak; square-root intensity, L2 norm, greedy one-to-one cosine, mass tolerance max(5 mDa, 10 ppm). Flag cosine >=0.97 with >=4 matched peaks and >=80% coverage by peak count on both sides. Low-peak similarities listed separately by raw maxima. Does not inspect every external/pretraining library.',
                spectra=len(queries), pairs=sum(d['checked'] for d in diagnostics),
                flagged_pairs=sum(d['strong'] for d in diagnostics), results=diagnostics)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('--blind', type=Path, required=True)
    args.add_argument('--train', type=Path, required=True)
    args.add_argument('--output', type=Path, required=True)
    a = args.parse_args()
    data = audit(a.blind, a.train)
    a.output.write_text(json.dumps(data, indent=2)+'\n')
    print({k:v for k,v in data.items() if k in {'spectra','pairs','flagged_pairs'}})

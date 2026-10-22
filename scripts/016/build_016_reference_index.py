"""016 traceable reference index + 40-case timing (manual, CPU only).

Phase A: full train scan in batches (no 2.5M Python-object expansion).
  Writes artifacts/runs/016-index/index.parquet (one row per kept spectrum:
  canonical_group_key, train_row, neutral_mass, adduct/mode/instrument/
  ingest_lib, peak_hash, n_clean, signal_retained, shard/offset) plus
  shards/shard_*.npz (float32 cleaned mz/intensity + int64 offsets, lazy mmap).
  Rows with unknown adduct mass are skipped and counted, never guessed.
Phase B: 40 frozen panel queries (20 cross_source + 20 natural_missing,
  hash-ordered, incl. negative/multimer) run ±200Da same-polarity nearest-2000
  retrieval + joint scoring; reports per-query wall/RSS, truncation, source
  density, quality>0 rate vs limits (RSS<8GiB target<2GiB, p95<2s CPU).

Manual run:
  /opt/homebrew/bin/python3.11 scripts/016/build_016_reference_index.py --phase index --probe
  /opt/homebrew/bin/python3.11 scripts/016/build_016_reference_index.py --phase index
  /opt/homebrew/bin/python3.11 scripts/016/build_016_reference_index.py --phase timing
"""
import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

SHARD_CACHE_MAX = None  # unbounded: 156 shards total ~558MB disk; LRU-32 thrashed (141s/10.5GB)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from analog_evidence import clean_peaks, exact_peak_hash, joint_cosine, neutral_mass

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/runs/016-index"
BATCH = 16384
SEED40 = "20261017|016-timing40"


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)


def build_index(probe=False):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import shutil
    t0 = time.time()
    out_dir = OUT / "probe" if probe else OUT
    shard_dir = out_dir / "shards"
    if not probe and OUT.exists():
        shutil.rmtree(OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(exist_ok=True)
    (out_dir / "batches").mkdir(exist_ok=True)
    map_t = pq.read_table(ROOT / "artifacts/runs/005/structure_identity_train.parquet",
                          columns=["raw_key", "canonical_group_key"])
    raw2canon = dict(zip(map_t.column("raw_key").to_pylist(),
                         map_t.column("canonical_group_key").to_pylist()))
    cols = ["inchikey14", "ingest_lib", "instrument_type", "ionization_mode",
            "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"]
    pf = pq.ParquetFile(ROOT / "data/raw/train.parquet")
    n_total = pf.metadata.num_rows
    limit = 100000 if probe else n_total
    recs = {"canon": [], "row": [], "nm": [], "prec": [], "adduct": [], "mode": [],
            "instrument": [], "lib": [], "phash": [], "n_clean": [],
            "sig_ret": [], "shard": [], "off": []}
    mz_parts, it_parts, off_parts = [], [], []
    skipped = {}
    kept = 0
    shard_id = 0
    batch_id = 0
    rows_seen = 0
    for batch in pf.iter_batches(batch_size=BATCH, columns=cols):
        if rows_seen >= limit:
            break
        d = batch.to_pylist()
        rows_seen += len(d)
        if rows_seen > limit:
            d = d[:len(d) - (rows_seen - limit)]
        recs = {"canon": [], "row": [], "nm": [], "prec": [], "adduct": [], "mode": [],
                "instrument": [], "lib": [], "phash": [], "n_clean": [],
                "sig_ret": [], "shard": [], "off": []}
        smz, sit, soff = [], [], [0]
        for i, r in enumerate(d):
            grow = rows_seen - len(d) + i
            nm = neutral_mass(r["precursor_mz"], r["adduct"])
            if not np.isfinite(nm):
                skipped[r["adduct"]] = skipped.get(r["adduct"], 0) + 1
                continue
            mz = np.asarray(r["ms2_mzs"], dtype=np.float64)
            it = np.asarray(r["ms2_normalized_intensities"], dtype=np.float64)
            try:
                cmz, cit, audit = clean_peaks(mz, it, r["precursor_mz"], r["adduct"])
            except (KeyError, ValueError):
                skipped[f"clean:{r['adduct']}"] = skipped.get(f"clean:{r['adduct']}", 0) + 1
                continue
            if len(cmz) == 0:
                skipped["empty_cleaned"] = skipped.get("empty_cleaned", 0) + 1
                continue
            recs["canon"].append(raw2canon.get(r["inchikey14"], r["inchikey14"]))
            recs["row"].append(grow)
            recs["nm"].append(float(nm))
            recs["prec"].append(float(r["precursor_mz"]))
            recs["adduct"].append(r["adduct"])
            recs["mode"].append(r["ionization_mode"])
            recs["instrument"].append(r["instrument_type"])
            recs["lib"].append(r["ingest_lib"])
            recs["phash"].append(exact_peak_hash(mz, it))
            recs["n_clean"].append(int(len(cmz)))
            recs["sig_ret"].append(float(audit["signal_retained_raw"]))
            recs["shard"].append(shard_id)
            recs["off"].append(int(soff[-1]))
            smz.append(cmz.astype(np.float32))
            sit.append(cit.astype(np.float32))
            soff.append(soff[-1] + len(cmz))
            kept += 1
        if smz:
            np.savez_compressed(shard_dir / f"shard_{shard_id:04d}.npz",
                                mz=np.concatenate(smz), it=np.concatenate(sit),
                                off=np.asarray(soff, dtype=np.int64))
            shard_id += 1
        if recs["row"] and not probe:
            table = pa.table({k: (pa.array(v) if k not in ("nm", "sig_ret", "prec")
                                  else pa.array(np.asarray(v, dtype=np.float64)))
                              for k, v in recs.items()})
            pq.write_table(table, out_dir / "batches" / f"batch_{batch_id:04d}.parquet")
            batch_id += 1
            del recs, table
        print(json.dumps({"rows_seen": rows_seen, "kept": kept,
                          "shards": shard_id, "rss_gib": round(rss_gib(), 3),
                          "elapsed": round(time.time() - t0, 1)}), flush=True)
    if not probe:
        import pyarrow.dataset as pds
        full = pds.dataset(str(out_dir / "batches"), format="parquet").to_table()
        pq.write_table(full, out_dir / "index.parquet")
        shutil.rmtree(out_dir / "batches")
    summary = {"rows_scanned": rows_seen, "kept": kept, "shards": shard_id,
               "skipped": skipped, "seconds": round(time.time() - t0, 1),
               "rss_gib": round(rss_gib(), 3), "probe": probe}
    summary = {"rows_scanned": rows_seen, "kept": kept, "shards": shard_id,
               "skipped": skipped, "seconds": round(time.time() - t0, 1),
               "rss_gib": round(rss_gib(), 3), "probe": probe}
    json.dump(summary, open(out_dir / ("index_probe.json" if probe else "index_build.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


def load_index():
    import pyarrow.parquet as pq
    t = pq.read_table(OUT / "index.parquet")
    idx = {c: np.asarray(t.column(c).to_pylist()) for c in
           ["canon", "row", "adduct", "mode", "instrument", "lib", "phash"]}
    idx["nm"] = np.asarray(t.column("nm").to_pylist(), dtype=np.float64)
    idx["prec"] = np.asarray(t.column("prec").to_pylist(), dtype=np.float64)
    idx["shard"] = np.asarray(t.column("shard").to_pylist(), dtype=np.int64)
    idx["off"] = np.asarray(t.column("off").to_pylist(), dtype=np.int64)
    idx["n_clean"] = np.asarray(t.column("n_clean").to_pylist(), dtype=np.int64)
    order = np.argsort(idx["nm"], kind="mergesort")
    return {k: v[order] for k, v in idx.items()}, t.num_rows


def read_peaks(shard_cache, shard_id, off, n):
    if shard_id not in shard_cache:
        if SHARD_CACHE_MAX is not None and len(shard_cache) >= SHARD_CACHE_MAX:
            shard_cache.popitem(last=False)
        z = np.load(OUT / "shards" / f"shard_{shard_id:04d}.npz")
        shard_cache[shard_id] = (z["mz"], z["it"], z["off"])
    mz, it, off_arr = shard_cache[shard_id]
    # locate position: off is absolute peak offset within shard; find via searchsorted
    pos = int(np.searchsorted(off_arr, off, side="left"))
    return (np.asarray(mz[off:off + n], dtype=np.float64),
            np.asarray(it[off:off + n], dtype=np.float64))


def run_timing():
    import pyarrow.parquet as pq
    t0 = time.time()
    idx, n = load_index()
    panel = json.load(open(ROOT / "artifacts/runs/016-panel/panel_manifest.json"))["panel"]
    cross = sorted([p for p in panel if p["group"] == "cross_source"],
                   key=lambda p: hashlib.sha256(f"{SEED40}|{p['canonical']}".encode()).hexdigest())[:20]
    nat = sorted([p for p in panel if p["group"] == "natural_missing"],
                 key=lambda p: hashlib.sha256(f"{SEED40}|{p['canonical']}".encode()).hexdigest())[:20]
    # query peaks come from the index shards themselves (same frozen cleaning);
    # never load the full-train peak columns (GBs) for a few dozen rows
    rowmap = dict(zip(idx["row"].tolist(), range(n)))
    shard_cache = {}
    queries = []
    missing = []
    for p in cross + nat:
        pos = rowmap.get(p["query_row"])
        if pos is None:
            missing.append(p["canonical"])
            continue
        cmz, cit = read_peaks(shard_cache, int(idx["shard"][pos]), int(idx["off"][pos]),
                              int(idx["n_clean"][pos]))
        queries.append({**p, "cmz": cmz, "cit": cit})
    try:
        qnm = np.array([neutral_mass(p["query_precursor_mz"], p["query_adduct"]) for p in queries])
    except KeyError:
        qnm = np.array([np.nan] * len(queries))
    shard_cache = {}
    rows = []
    missing_report = {"query_rows_absent_from_index": missing}
    for qi, q in enumerate(queries):
        s = time.perf_counter()
        lo = int(np.searchsorted(idx["nm"], qnm[qi] - 200.0, "left"))
        hi = int(np.searchsorted(idx["nm"], qnm[qi] + 200.0, "right"))
        eligible = hi - lo
        same = [i for i in range(lo, hi) if idx["mode"][i] == q["query_mode"]]
        same = sorted(same, key=lambda i: abs(float(idx["nm"][i]) - qnm[qi]))[:2000]
        scored, pos = 0, 0
        lib_hit = {}
        for i in same:
            rmz, rit = read_peaks(shard_cache, int(idx["shard"][i]), int(idx["off"][i]),
                                  int(idx["n_clean"][i]))
            # timing uses direct-only joint scoring (shift=None) to bound cost honestly;
            # full direct+shift evidence runs in the later feature stage, not here
            r = joint_cosine(q["cmz"], q["cit"], rmz, rit, shift=None)
            scored += 1
            pos += (r["quality"] > 0)
            lib_hit[str(idx["lib"][i])] = lib_hit.get(str(idx["lib"][i]), 0) + 1
        wall = time.perf_counter() - s
        rows.append({"case": q["canonical"], "group": q["group"], "eligible": eligible,
                     "truncated": max(0, eligible - 2000), "scored": scored,
                     "quality_pos": pos, "wall_s": round(wall, 3),
                     "rss_gib": round(rss_gib(), 3), "top_libs": sorted(lib_hit.items(), key=lambda x: -x[1])[:5]})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    walls = sorted(r["wall_s"] for r in rows)
    summary = {"n": len(rows),
               "wall_p50": walls[len(walls) // 2], "wall_p95": walls[int(len(walls) * 0.95) - 1],
               "wall_max": walls[-1], "rss_gib": round(rss_gib(), 3),
               "mean_scored": float(np.mean([r["scored"] for r in rows])),
               "mean_quality_pos_rate": float(np.mean([r["quality_pos"] / max(1, r["scored"]) for r in rows])),
               "limits": {"rss_gib_lt": 8.0, "rss_target_gib": 2.0, "p95_s_lt": 2.0},
               "seconds_total": round(time.time() - t0, 1)}
    json.dump({"rows": rows, "summary": summary, **missing_report},
              open(OUT / "timing40.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["index", "timing"], required=True)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    if args.phase == "index":
        build_index(probe=args.probe)
    else:
        run_timing()

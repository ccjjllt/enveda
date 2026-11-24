"""Measure MSG GLACIER throughput on LOTUS structures without competition training spectra."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd


INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working")
TABLE_SHA = "6ccb833e47d88715fca1d614b05cae92fa333fa388ae1d64226f2695a62081aa"
WEIGHTS_SHA = "9192e4c2eeafe83317d5a2c2641e1fc4d68934488757a4faafeb66f2ab397a12"
RUNNER_SHA = "5c4f74313ac54263d05d987c39d012d81ec57c5f35e1095ecd49a5bcb6440fca"
ENERGIES = (20., 40., 60.)


def sample(table):
    df = pd.read_parquet(table)
    groups = df.groupby("scorer_key")
    in_training = groups.in_casmi_train.transform("any")
    in_pool = groups.in_casmi_pool.transform("any")
    keep = (df.in_lotus & ~in_training & df.scorer_key.str.len().eq(14)
            & df.n_heavy.between(8, 64) & df.mono_mass.between(120, 850) & df.smiles.notna())
    rows = df.loc[keep, ["scorer_key", "smiles", "formula", "mono_mass", "n_heavy"]].copy()
    rows["in_pool"] = in_pool[keep]
    rows = rows.sort_values(["scorer_key", "smiles"]).drop_duplicates("scorer_key")
    rows["sample_hash"] = rows.scorer_key.map(lambda s: sha256(("np-sim-v1/" + s).encode()).hexdigest())
    rows = rows.sort_values("sample_hash")
    outside = rows[~rows.in_pool].head(1000)
    return pd.concat([outside, rows[rows.in_pool].head(2000 - len(outside))]).sort_values("sample_hash")


def checked(path, checksum):
    if sha256(path.read_bytes()).hexdigest() != checksum:
        raise ValueError(f"Unexpected input digest for {path.name}")
    return path


def run():
    start = time.monotonic()
    table, = INPUT.rglob("natural_products.parquet")
    rows = sample(checked(table, TABLE_SHA))
    rows.to_parquet(OUT / "selected.parquet", index=False)
    runner, = INPUT.rglob("gl_runner.py")
    pkg = checked(runner, RUNNER_SHA).parent
    checked(pkg / "ckpt/glacier.pt", WEIGHTS_SHA)
    ice, = INPUT.rglob("ice_runner.py")
    sys.path.insert(0, str(pkg))
    import gl_runner as gl
    log = gl.Log()
    rdkit = gl.setup_env(SimpleNamespace(pkg=str(pkg), site="/tmp/np_glacier_site",
                                         wheels=str(ice.parent / "wheels"), any_rdkit=False), log)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU throughput requires CUDA")
    torch.set_num_threads(2)
    predictor = gl.Predictor(gl.load_model(str(pkg), "cuda:0", log), "cuda:0", 24, log)
    by_key = rows.set_index("scorer_key")
    valid_keys = [key for key in by_key.index if gl.smiles_ok(by_key.at[key, "smiles"])[0]]
    chunks = []
    diagnostics = {"selected": len(rows), "outside_main_pool": int((~rows.in_pool).sum()),
                   "supported_structures": len(valid_keys), "energy_inputs": ENERGIES,
                   "adduct": "[M+H]+", "instrument_token": "QTOF", "rdkit": rdkit,
                   "gpu": torch.cuda.get_device_name(0), "gpu_devices_used": 1, "teacher_sha256": WEIGHTS_SHA,
                   "status": "running", "chunks": chunks}
    begin_predictions = time.monotonic()
    # The kernel has 40 minutes. Leave time to write an accurate partial throughput receipt.
    deadline = time.time() + max(0., 1950. - (time.monotonic() - start))
    for offset in range(0, len(valid_keys), 64):
        keys = valid_keys[offset:offset+64]
        jobs = {(by_key.at[k, "smiles"], "[M+H]+", "QTOF", ce) for k in keys for ce in ENERGIES}
        with torch.inference_mode():
            finished = predictor.predict(jobs, deadline)
        records = []
        for k in keys:
            for ce in ENERGIES:
                pair = predictor.cache.get((by_key.at[k, "smiles"], "[M+H]+", "QTOF", ce))
                if pair is None:
                    continue
                mz, intensity = pair
                good = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
                if good.sum() >= 4:
                    records.append(dict(scorer_key=k, energy_input=ce, mz=mz[good].astype("float32"),
                                        intensity=intensity[good].astype("float32")))
        size = 0
        if records:
            path = OUT / f"spectra_{offset:04d}.parquet"
            pd.DataFrame(records).to_parquet(path, index=False)
            size = path.stat().st_size
        chunks.append(dict(input_structures=len(keys), jobs=len(jobs), usable_spectra=len(records),
                           all_energies=sum(n == len(ENERGIES) for n in pd.Series(
                               [r["scorer_key"] for r in records], dtype=str).value_counts()),
                           compressed_bytes=size, finished=finished))
        predictor.cache.clear()
        predictor.pcache.clear()
        diagnostics.update(prediction_seconds=round(time.monotonic()-begin_predictions, 2),
                           wall_seconds=round(time.monotonic()-start, 2),
                           prediction_errors=predictor.n_fail, feature_errors=predictor.n_feat_fail)
        (OUT / "simulation_summary.json").write_text(json.dumps(diagnostics, indent=2))
        if not finished:
            break
    diagnostics["status"] = "complete" if chunks and all(c["finished"] for c in chunks) else "partial"
    (OUT / "simulation_summary.json").write_text(json.dumps(diagnostics, indent=2))
    print("GLACIER SIMULATION", diagnostics["status"], sum(c["usable_spectra"] for c in chunks),
          "usable spectra from", len(rows), "selected structures", flush=True)


if __name__ == "__main__":
    run()

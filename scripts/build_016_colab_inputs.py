"""016 Colab inputs (manual, local CPU only).

Builds the Ice-only Colab bundle for the frozen 400-molecule panel:
  panel spectra (400 train rows) -> s2 forward on local CPU (torch, weights
  frozen v10) -> 10ppm full-window FP scoring (v10 caliber, torch CPU bmm) ->
  canonical-max merge (005 union map, no RDKit regrouping) -> top100 ->
  f1_worker inputs.json. Colab T4 then runs ONLY iceberg ICE predictions
  (proven 009 pattern), tars scores back. No submission from this step.

Frozen parity: model/prep/neutral_mass/fwd_batch sources are exec'd byte-
identical from notebooks/submission-v10/enveda-submission-v10.ipynb
(cells 4/5/6-fwd; SHA-recorded). One disclosed patch: cell-4 `.cuda()` ->
`.cpu()` for local forward (eval mode, math identical). Scoring chunk G=20
(per-molecule independent; chunking cannot change scores).

Manual run: /opt/homebrew/bin/python3.11 scripts/build_016_colab_inputs.py [--probe]
Probe: 5 queries, no tar. Full writes artifacts/016-colab/{inputs.json,manifest.json}
and panel-016-ice-bundle.tar.gz (inputs.json + code/ice + install.sh + f1_worker.py).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/016-colab"
NB = ROOT / "notebooks/submission-v10/enveda-submission-v10.ipynb"
W_SINGLE = ROOT / "artifacts/001-baseline/weights/fp_single_s2.pt"
PPM = 10.0
NTR = 275810
NCO = 436389
G = 20


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def exec_frozen():
    """Exec v10 cells 4/5 + fwd_batch; return namespace + cell SHAs."""
    import torch
    nb = json.load(open(NB))
    cells = ["".join(c.get("source", [])) for c in nb["cells"]]
    c4, c5 = cells[4], cells[5]
    fwd_src = [l for l in cells[6].split("\n") if l.startswith("def fwd_batch")]
    # fwd_batch is a contiguous def block: slice from its def to the next top-level line
    lines = cells[6].split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("def fwd_batch"))
    end = next((i for i in range(start + 1, len(lines))
                if lines[i] and not lines[i].startswith((" ", "\t"))), len(lines))
    fwd = "\n".join(lines[start:end])
    if "device='cuda'" in fwd:
        fwd = fwd.replace("device='cuda'", "device='cpu'")
        dev_patch2 = ";fwd device='cuda'->'cpu'"
    else:
        dev_patch2 = ""
    c4cpu = c4.replace(".cuda().eval()", ".cpu().eval()")
    assert c4cpu != c4, "cuda patch anchor missing"
    t_fp = np.load(ROOT / "datasets/train-fp-pool/fp.npy", mmap_mode="r")
    ns = {"torch": torch, "W_SINGLE": str(W_SINGLE), "t_fp": t_fp,
            "np": np, "__name__": "v10frozen"}
    exec(compile(c4cpu, "v10c4", "exec"), ns)
    exec(compile(c5, "v10c5", "exec"), ns)
    exec(compile(fwd, "v10fwd", "exec"), ns)
    shas = {"cell4": hashlib.sha256(c4.encode()).hexdigest(),
            "cell5": hashlib.sha256(c5.encode()).hexdigest(),
            "cell6": hashlib.sha256(cells[6].encode()).hexdigest(),
            "device_patch": ".cuda().eval()->.cpu().eval() one token" + dev_patch2}
    return ns, shas


def read_panel_spectra(panel):
    import pyarrow.parquet as pq
    want = sorted(p["query_row"] for p in panel)
    pf = pq.ParquetFile(ROOT / "data/raw/train.parquet")
    # row-group boundaries
    bounds, acc = [], 0
    for i in range(pf.metadata.num_row_groups):
        n = pf.metadata.row_group(i).num_rows
        bounds.append((acc, acc + n))
        acc += n
    cols = ["ms2_mzs", "ms2_normalized_intensities", "precursor_mz", "adduct",
            "instrument_type", "ionization_mode", "collision_energy_ev",
            "collision_energy_orig", "collision_energy_orig_units"]
    out = {}
    for gi, (lo, hi) in enumerate(bounds):
        sub = [r for r in want if lo <= r < hi]
        if not sub:
            continue
        t = pf.read_row_group(gi, columns=cols).to_pylist()
        for r in sub:
            out[r] = t[r - lo]
    assert len(out) == len(want), (len(out), len(want))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    import torch
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    from rdkit import Chem

    panel = json.load(open(ROOT / "artifacts/016-panel/panel_manifest.json"))["panel"]
    if args.probe:
        panel = panel[:5]
    ns, shas = exec_frozen()
    prep, fwd = ns["prep_peaks"], ns["fwd_batch"]
    net = ns["net"]
    spec = read_panel_spectra(panel)
    prows = []
    for p in panel:
        s = spec[p["query_row"]]
        prows.append((np.asarray(s["ms2_mzs"], float), np.asarray(s["ms2_normalized_intensities"], float),
                      float(s["precursor_mz"]), s["adduct"], s["instrument_type"],
                      s["collision_energy_ev"], s["ionization_mode"]))
    Zs = []
    with torch.no_grad():
        for i in range(0, len(prows), 64):
            Zs.append(fwd(prows[i:i + 64]).cpu())
            print("forward", min(i + 64, len(prows)), "/", len(prows), flush=True)
    Z = torch.cat(Zs).numpy().astype(np.float32)
    assert Z.shape == (len(panel), 6930) and np.isfinite(Z).all()
    fwd_sec = time.time() - t0

    # pool assembly in v10 order: train + coco + mb539
    import pickle
    t_fp = np.load(ROOT / "datasets/train-fp-pool/fp.npy", mmap_mode="r")
    t_mass = np.load(ROOT / "datasets/train-fp-pool/mass.npy", mmap_mode="r")
    t_keys = np.load(ROOT / "datasets/train-fp-pool/keys.npy", mmap_mode="r")
    t_smi = [l.rstrip("\n") for l in open(ROOT / "datasets/train-fp-pool/smiles.txt")]
    c_fp = np.load(ROOT / "artifacts/005/coco/coco_fp.npy", mmap_mode="r")
    c_mass = np.load(ROOT / "artifacts/005/coco/coco_mass.npy", mmap_mode="r")
    c_meta = pickle.load(open(ROOT / "artifacts/005/coco/coco_meta.pkl", "rb"))
    x_fp = np.load(ROOT / "artifacts/011-expansion-asset/mb539_fp.npy", mmap_mode="r")
    x_mass = np.load(ROOT / "artifacts/011-expansion-asset/mb539_masses.npy", mmap_mode="r")
    x_keys = np.load(ROOT / "artifacts/011-expansion-asset/mb539_keys.npy", mmap_mode="r")
    x_smi = [l.rstrip("\n") for l in open(ROOT / "artifacts/011-expansion-asset/mb539_smiles.txt")]
    allm = np.concatenate([t_mass, c_mass, x_mass]).astype(np.float64)
    NBASE = NTR + NCO
    assert allm.shape == (NBASE + 538,) and t_fp.shape == (NTR, 867) and c_fp.shape == (NCO, 867)
    c_keys = list(c_meta["keys"])
    c_smi = list(c_meta["smiles"])
    cidx = {k: i for i, k in enumerate(c_keys)}
    # union canonical map: train + coco identity tables, else raw key
    import pyarrow.parquet as pq
    umap = {}
    for path in ["artifacts/005/structure_identity_train.parquet",
                 "artifacts/005/structure_identity_coco.parquet"]:
        tt = pq.read_table(ROOT / path, columns=["raw_key", "canonical_group_key"])
        umap.update(dict(zip(tt.column("raw_key").to_pylist(), tt.column("canonical_group_key").to_pylist())))
    allk = list(t_keys) + c_keys + list(x_keys)
    alls = t_smi + c_smi + x_smi
    NALL = len(allk)
    assert NALL == NBASE + 538 == 712737

    QM = ns["neutral_mass"](np.array([r[2] for r in prows], float), [r[3] for r in prows])
    assert np.isfinite(QM).all()
    mols = []
    stat = {"empty_window": 0, "mass_fallback": 0, "exception": 0, "ok": 0}
    for g in range(0, len(panel), G):
        chunk = range(g, min(g + G, len(panel)))
        Ws = [np.where(np.abs(allm - QM[i]) <= QM[i] * PPM * 1e-6)[0] for i in chunk]
        L = max([len(w) for w in Ws] + [1])
        Fmat = np.zeros((len(chunk), L, 6930), np.float32)
        Fmat_np = np.ones((len(chunk), L), bool)
        for bi, w in enumerate(Ws):
            is_tr = w < NTR
            is_co = (w >= NTR) & (w < NBASE)
            is_ex = w >= NBASE
            if is_tr.any():
                Fmat[bi, np.where(is_tr)[0]] = np.unpackbits(
                    np.asarray(t_fp[w[is_tr]], dtype=np.uint8), axis=1)[:, :6930]
            if is_co.any():
                ci = np.array([cidx[allk[j]] for j in w[is_co]])
                Fmat[bi, np.where(is_co)[0]] = np.unpackbits(
                    np.asarray(c_fp[ci], dtype=np.uint8), axis=1)[:, :6930]
            if is_ex.any():
                Fmat[bi, np.where(is_ex)[0]] = np.unpackbits(
                    np.asarray(x_fp[w[is_ex] - NBASE], dtype=np.uint8), axis=1)[:, :6930]
            Fmat_np[bi, :len(w)] = False
        with torch.no_grad():
            S = torch.bmm(torch.as_tensor(Fmat),
                          torch.as_tensor(Z[list(chunk)]).unsqueeze(-1)).squeeze(-1).numpy()
        S[Fmat_np] = float("-inf")
        for bi, i in enumerate(chunk):
            w = Ws[bi]
            KLb = [allk[j] for j in w]
            SLb = [alls[j] for j in w]
            PLb = [abs(allm[j] - QM[i]) / QM[i] * 1e6 for j in w]
            if not len(KLb):
                stat["empty_window"] += 1
                mols.append((panel[i], [], [], [], "empty_window"))
                continue
            sc = [float(x) for x in S[bi][:len(KLb)].tolist()]
            pairs = sorted(zip(KLb, SLb, sc), key=lambda t: (-t[2], str(t[0])))
            stat["ok"] += 1
            mols.append((panel[i], KLb, SLb, [s for _, _, s in pairs], "ok",
                         [k for k, _, _ in pairs]))
        print("scored group", g // G + 1, round(time.time() - t0, 1), stat, flush=True)

    # canonical-max merge -> top100 (005 union map caliber)
    import collections
    molecules = []
    for item in mols:
        p = item[0]
        if item[-1] == "empty_window":
            molecules.append({"key": p["canonical"], "qidx": len(molecules),
                              "truth_canon": p["canonical"], "order_top100": [],
                              "scores_top100": [],
                              "rep_smiles": {}, "formulas": {},
                              "spectra": [spec[p["query_row"]]], "note": "empty_window"})
            continue
        _, KLb, SLb, sc_sorted, _, shipped_order = item
        sm_of = dict(zip(KLb, SLb))
        best = {}
        for k, s in zip(shipped_order, sc_sorted):
            c = umap.get(k, k)
            if c not in best or s > best[c][1]:
                best[c] = (sm_of[k], s)
        ranked = sorted(best.items(), key=lambda kv: (-kv[1][1], kv[0]))[:100]
        rep = {c: sm for c, (sm, _) in ranked}
        sc100 = [float(s) for _, (_, s) in ranked]
        forms = {}
        for c, sm in rep.items():
            try:
                forms[c] = CalcMolFormula(Chem.MolFromSmiles(sm))
            except Exception:
                forms[c] = None
        s0 = spec[p["query_row"]]
        molecules.append({"key": p["canonical"], "qidx": len(molecules),
                          "truth_canon": p["canonical"],
                          "order_top100": [c for c, _ in ranked],
                          "scores_top100": sc100,
                          "rep_smiles": rep, "formulas": forms,
                          "spectra": [{"mz": list(np.asarray(s0["ms2_mzs"], float)),
                                       "it": list(np.asarray(s0["ms2_normalized_intensities"], float)),
                                       "prec": str(s0["precursor_mz"]), "adduct": s0["adduct"],
                                       "mode": s0["ionization_mode"], "instrument": s0["instrument_type"],
                                       "ce_ev": s0["collision_energy_ev"],
                                       "ce_orig": s0["collision_energy_orig"],
                                       "ce_units": s0["collision_energy_orig_units"]}]})
    if args.probe:
        print(json.dumps({"molecules": len(molecules), "stat": stat,
                          "fwd_seconds": round(fwd_sec, 1),
                          "top100_lens": sorted([len(m["order_top100"]) for m in molecules])[:5]},
                         indent=2))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    bundle_src = OUT / "panel-016"
    import shutil
    if bundle_src.exists():
        shutil.rmtree(bundle_src)
    bundle_src.mkdir(parents=True)
    doc = {"bundle": "panel-016 ice inputs", "seed": 20261017, "ppm": PPM,
           "pool": "train275810+coco436389+mb538=712737",
           "n_molecules": len(molecules), "stat": stat,
           "truth_note": "truth_canon for final scoring only; worker never reads it",
           "sources": "panel_manifest.json frozen 016",
           "molecules": molecules}
    json.dump(doc, open(bundle_src / "inputs.json", "w"))
    for f in ["f1_worker.py", "install.sh"]:
        import shutil
        shutil.copy(ROOT / "artifacts/f1-bundle" / f, bundle_src / f)
    import shutil
    shutil.copytree(ROOT / "artifacts/f1-bundle/code", bundle_src / "code")
    manifest = {"cells": shas, "weights": {"fp_single_s2.pt": sha_file(W_SINGLE)},
                "pool": {"n": NALL}, "stat": stat,
                "torch": torch.__version__, "device": "cpu",
                "rdkit": "2026.03.6(local merge uses 005 map; formulas via local RDKit)"}
    json.dump(manifest, open(bundle_src / "MANIFEST.json", "w"), indent=2)
    tar = OUT / "panel-016-ice-bundle.tar.gz"
    subprocess.run(["tar", "czf", str(tar), "-C", str(OUT), "panel-016"], check=True)
    print("bundle", tar, tar.stat().st_size, "sha", sha_file(tar)[:16])
    print("DRIVE UPLOADS:")
    print(" ", tar.name, tar.stat().st_size, "(bundle: inputs.json + code/ice + install.sh + f1_worker.py)")
    for w, sz in [("artifacts/001-baseline/weights/fp_single_s2.pt", None),
                  ("artifacts/f1-bundle/weights/gen.pt", None),
                  ("artifacts/f1-bundle/weights/inten.pt", None)]:
        p = ROOT / w
        print(" ", Path(w).name, p.stat().st_size, f"({w})")
    print(json.dumps({**manifest, "seconds": round(time.time() - t0, 1)}, indent=2)[:1500])


if __name__ == "__main__":
    main()

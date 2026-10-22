#!/opt/homebrew/bin/python3.11
"""009-3 建固定单来源索引：LipidMaps LMSD（CPU-only，手动研究脚本，非测试文件）。

任务：009下一步第3项第1-2步。单来源先行，最小可行：LipidMaps LMSD 本地快照。
不下载105M级全量（PubChem），不读G1保留250，零GPU，不push/submit。

输入（只读，不联网）：
- LMSD快照：artifacts/shared/pool-lmsd/LMSD.sdf.zip（源 lipidmaps.org，CC BY 4.0，2026-10-04）
  及解包 artifacts/shared/pool-lmsd/structures.sdf（279MB，本地已解包，只读核对）
- 已处理：artifacts/shared/pool-lmsd/lmsd_raw.jsonl（50586）/ lmsd_unique.jsonl（43759）/
  lmsd_new.jsonl（22765，仅对base新增）/ lmsd_canon_ckpt.jsonl / keys.npy / mass.npy /
  smiles.txt / lm_ids.txt / fp.npy / meta.json（只读核对，不覆盖）
- base池：artifacts/shared/pool-train/{mass,keys,smiles,fp}.npy + artifacts/runs/005/coco/{coco_mass,coco_fp,coco_meta.pkl}
  及 artifacts/runs/005/structure_identity_{train,coco}.parquet（仅统计交叠，不筛选）
- fp_bits：artifacts/runs/005/coco/fp_bits.npy（仅记录SHA，不在此算全量指纹）

许可筛选（复用008-B口径，仅收可再分发）：
- LMSD快照级许可 CC BY 4.0 ∈ {CC0, CC BY, CC BY-SA, MIT}，收录。
- SDF无逐记录LICENSE字段，快照级许可 governing；不明/禁再分发计0收（本次快照无NC记录；
  prvsiyan/chebi-lipidmaps-casmi26 v1 CC BY-NC-SA重打包本次不用，见005计划警告）。
- ChEBI：本地artifacts/data//tmp无快照/缓存（已查零命中），需下载，未选。
- COCONUT：已在base内（coco_extra 436389 shipped，v2），不是新增来源，未选。
- PubChem：约105M结构（高分管线资产说明），体量禁下载，未选。
- MassBank：008已做（8831 distinct / 539新增），本任务独立来源，不重复。

canonical/质量口径（008-B六条，复用函数与约束）：
- canonical_group_key（去立体→互变异构规范化→InChIKey14，RDKit 2026.03.6）
- ExactMolWt float64；无标签质量窗查询；未知adduct单列；真值仅oracle诊断分支；离线可用。

输出（新增目录 artifacts/runs/009-source/，不覆盖pool-lmsd/008-expansion/007-r3）：
- candidate_index_masses.npy（float64）、candidate_index_keys.npy、
  candidate_index_smiles.txt、candidate_index_lm_ids.txt、
  candidate_index_records.jsonl、candidate_index_meta.json、sources_manifest.json
- build.log（人工核对用；stdout重定向）
"""

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, os.path.join(REPO, "src"))

OUTDIR = os.path.join(REPO, "artifacts", 'runs', "009-source")
LMSD_ZIP = os.path.join(REPO, "artifacts", 'shared', "pool-lmsd", "LMSD.sdf.zip")
LMSD_SDF = os.path.join(REPO, "artifacts", 'shared', "pool-lmsd", "structures.sdf")
LMSD_RAW = os.path.join(REPO, "artifacts", 'shared', "pool-lmsd", "lmsd_raw.jsonl")
LMSD_UNIQUE = os.path.join(REPO, "artifacts", 'shared', "pool-lmsd", "lmsd_unique.jsonl")
LMSD_NEW = os.path.join(REPO, "artifacts", 'shared', "pool-lmsd", "lmsd_new.jsonl")
LMSD_KEYS = os.path.join(REPO, "artifacts", 'shared', "pool-lmsd", "keys.npy")
LMSD_MASS = os.path.join(REPO, "artifacts", 'shared', "pool-lmsd", "mass.npy")
LMSD_SMILES = os.path.join(REPO, "artifacts", 'shared', "pool-lmsd", "smiles.txt")
LMSD_LMIDS = os.path.join(REPO, "artifacts", 'shared', "pool-lmsd", "lm_ids.txt")
LMSD_META = os.path.join(REPO, "artifacts", 'shared', "pool-lmsd", "meta.json")
FP_BITS = os.path.join(REPO, "artifacts", 'runs', "005", "coco", "fp_bits.npy")
TRAIN_MAP = os.path.join(REPO, "artifacts", 'runs', "005", "structure_identity_train.parquet")
COCO_MAP = os.path.join(REPO, "artifacts", 'runs', "005", "structure_identity_coco.parquet")

ALLOWED = {"CC0", "CC BY", "CC BY-SA", "MIT"}
SNAPSHOT_LICENSE = "CC BY 4.0"


def sha_file(path):
    h = hashlib.sha256()
    sz = 0
    with open(path, "rb") as f:
        while True:
            ch = f.read(1 << 20)
            if not ch:
                break
            h.update(ch)
            sz += len(ch)
    return sz, h.hexdigest()


def main():
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    out_masses = os.path.join(OUTDIR, "candidate_index_masses.npy")
    out_keys = os.path.join(OUTDIR, "candidate_index_keys.npy")
    out_smiles = os.path.join(OUTDIR, "candidate_index_smiles.txt")
    out_lmids = os.path.join(OUTDIR, "candidate_index_lm_ids.txt")
    out_records = os.path.join(OUTDIR, "candidate_index_records.jsonl")
    out_meta = os.path.join(OUTDIR, "candidate_index_meta.json")
    out_manifest = os.path.join(OUTDIR, "sources_manifest.json")
    for p in [out_masses, out_keys, out_smiles, out_lmids, out_records, out_meta, out_manifest]:
        assert not os.path.exists(p), f"refuse overwrite {p}（只增不改；如需重跑先人工归档）"

    for p in [LMSD_ZIP, LMSD_SDF, LMSD_RAW, LMSD_UNIQUE, LMSD_NEW, LMSD_KEYS,
              LMSD_MASS, LMSD_SMILES, LMSD_LMIDS, LMSD_META, FP_BITS,
              TRAIN_MAP, COCO_MAP]:
        assert os.path.exists(p), f"missing input {p}"
    import numpy as np

    # 输入SHA（墙钟/seed齐备在meta/run_meta）
    inputs_sha = {}
    for p in [LMSD_ZIP, LMSD_SDF, LMSD_RAW, LMSD_UNIQUE, LMSD_NEW, LMSD_KEYS,
              LMSD_MASS, LMSD_SMILES, LMSD_LMIDS, LMSD_META, FP_BITS]:
        sz, sh = sha_file(p)
        inputs_sha[os.path.relpath(p, REPO)] = {"bytes": int(sz), "sha256": str(sh)}
        print(f"input {os.path.relpath(p, REPO)} bytes {sz} sha {sh}", flush=True)

    try:
        from chemistry import canonical_group_key
        import rdkit as _rdk
        rdkit_ver = str(_rdk.__version__)
    except Exception as e:
        raise RuntimeError(f"need RDKit for canonical_group_key: {e}")
    assert rdkit_ver.startswith("2026.03.6"), f"expect RDKit 2026.03.6, got {rdkit_ver}"
    print(f"rdkit {rdkit_ver}", flush=True)

    # base union canonicals（仅统计交叠，不筛选）
    union_canons = set()
    try:
        import pyarrow.parquet as pq
        for mp in [TRAIN_MAP, COCO_MAP]:
            pf = pq.ParquetFile(mp)
            for batch in pf.iter_batches(columns=["raw_key", "canonical_group_key"], batch_size=65536):
                for c in batch.column(1).to_pylist():
                    union_canons.add(str(c))
        print(f"base union_canons {len(union_canons)} (train+coco maps, stats only)", flush=True)
    except Exception as e:
        print(f"WARN base maps not loaded ({e}); overlap stats will be null", flush=True)
        union_canons = set()

    # 读lmsd_new（22765，仅对base新增）+ lmsd_unique（43759，全源去重）用于交叠复核
    new_rows = [json.loads(l) for l in open(LMSD_NEW)]
    uniq_rows = [json.loads(l) for l in open(LMSD_UNIQUE)]
    print(f"lmsd_new rows {len(new_rows)} lmsd_unique rows {len(uniq_rows)}", flush=True)
    old_keys = np.load(LMSD_KEYS, allow_pickle=True)
    old_masses = np.load(LMSD_MASS)
    old_smiles = open(LMSD_SMILES).read().splitlines()
    old_lmids = open(LMSD_LMIDS).read().splitlines()
    assert len(new_rows) == len(old_keys) == len(old_masses) == len(old_smiles) == len(old_lmids) == 22765, (
        len(new_rows), len(old_keys), len(old_masses), len(old_smiles), len(old_lmids))
    assert str(old_masses.dtype) == "float64", old_masses.dtype

    # 交叠复核：unique全源 vs base union
    if union_canons:
        n_overlap_full = sum(1 for r in uniq_rows if str(r["canonical_group_key"]) in union_canons)
        n_new_full = len(uniq_rows) - n_overlap_full
        n_overlap_idx = sum(1 for r in new_rows if str(r["canonical_group_key"]) in union_canons)
    else:
        n_overlap_full, n_new_full, n_overlap_idx = None, None, None
    print(f"full-source overlap {n_overlap_full} new {n_new_full} (expect 20994/22765); "
          f"new-index overlap-with-base {n_overlap_idx} (expect 0)", flush=True)

    # 逐条重验 canonical + ExactMolWt float64（008-B口径，不伪造质量）
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.Descriptors import ExactMolWt
    from rdkit import Chem
    import math as _math

    entries = []
    n_canon_mismatch = 0
    n_unparseable = 0
    n_mass_fail = 0
    n_mass_mismatch = 0
    for i, r in enumerate(new_rows):
        smi = str(r["smiles"])
        key_rec = str(r["canonical_group_key"])
        try:
            mol = Chem.MolFromSmiles(smi)
        except Exception:
            mol = None
        if mol is None:
            n_unparseable += 1
            continue
        try:
            canon = canonical_group_key(smi)
        except Exception:
            n_unparseable += 1
            continue
        if canon != key_rec:
            n_canon_mismatch += 1
            continue
        try:
            mass = float(ExactMolWt(mol))
            formula = str(rdMolDescriptors.CalcMolFormula(mol))
        except Exception:
            n_mass_fail += 1
            continue
        if not _math.isfinite(mass) or mass <= 0:
            n_mass_fail += 1
            continue
        old_m = float(old_masses[i])
        if abs(old_m - mass) > 1e-9:
            n_mass_mismatch += 1
            continue
        if str(old_keys[i]) != key_rec or old_smiles[i] != smi or old_lmids[i] != str(r["LM_ID"]):
            raise RuntimeError(f"row {i} identity drift vs pool-lmsd artifacts")
        entries.append({
            "canonical_key": key_rec,
            "smiles_rep": smi,
            "formula_calc": str(formula),
            "mass_theo": float(mass),
            "license_rep": SNAPSHOT_LICENSE,
            "LM_ID": str(r["LM_ID"]),
            "inchi_key_full": str(r.get("inchi_key", "")),
        })
        if (i + 1) % 5000 == 0:
            print(f"  verify {i+1}/{len(new_rows)} ok {len(entries)} ...", flush=True)
    print(f"verified {len(entries)} canon_mismatch {n_canon_mismatch} unparse {n_unparseable} "
          f"mass_fail {n_mass_fail} mass_mismatch {n_mass_mismatch}", flush=True)
    assert n_canon_mismatch == 0 and n_unparseable == 0 and n_mass_fail == 0 and n_mass_mismatch == 0, \
        "reverification must be zero-diff (pool-lmsd caliber)"
    assert len(entries) == 22765, len(entries)

    # 确定性排序：mass升序，canon次之（与质量索引一致）
    entries.sort(key=lambda e: (e["mass_theo"], e["canonical_key"]))
    masses = np.asarray([e["mass_theo"] for e in entries], dtype=np.float64)
    keys = np.asarray([e["canonical_key"] for e in entries], dtype=object)

    np.save(out_masses, masses)
    np.save(out_keys, keys)
    with open(out_smiles, "w") as f:
        for e in entries:
            f.write(str(e["smiles_rep"]) + "\n")
    with open(out_lmids, "w") as f:
        for e in entries:
            f.write(str(e["LM_ID"]) + "\n")
    with open(out_records, "w") as f:
        for e in entries:
            f.write(json.dumps(e, allow_nan=False) + "\n")

    wall_utc = datetime.now(timezone.utc).isoformat()
    meta = {
        "task": "009-3 fixed single-source candidate index (LMSD, unlabeled query)",
        "created_utc": wall_utc,
        "seed": 20261012,
        "budget": "local CPU only; no training; no GPU; no Kaggle quota; no heldout spectra read/eval; "
                  "no large-full download (PubChem 105M not fetched); writable only artifacts/runs/009-source/",
        "source_selected": "LIPID MAPS LMSD (minimal viable single source; local snapshot available, small)",
        "snapshot": {
            "zip": os.path.relpath(LMSD_ZIP, REPO),
            "bytes": int(inputs_sha[os.path.relpath(LMSD_ZIP, REPO)]["bytes"]),
            "sha256": str(inputs_sha[os.path.relpath(LMSD_ZIP, REPO)]["sha256"]),
            "unpacked_sdf": os.path.relpath(LMSD_SDF, REPO),
            "unpacked_bytes": int(inputs_sha[os.path.relpath(LMSD_SDF, REPO)]["bytes"]),
            "unpacked_sha256": str(inputs_sha[os.path.relpath(LMSD_SDF, REPO)]["sha256"]),
            "release": "lipidmaps.org LMSD.sdf.zip CC BY 4.0 2026-10-04 20.91MB (snapshot-level license; "
                       "SDF carries no per-record LICENSE field)",
            "lmsd_total_records": 50586,
            "lmsd_ok_canon": 50002,
            "lmsd_unique_canon_full_source": 43759,
        },
        "license_policy": "Only CC0/CC-BY/CC-BY-SA/MIT accepted (008-B caliber). LMSD snapshot CC BY 4.0 "
                          "accepted at snapshot level. Per-record NC/missing/unclear: 0 collected "
                          "(this snapshot carries no NC records; prvsiyan chebi-lipidmaps v1 CC BY-NC-SA "
                          "repack NOT used).",
        "license_counts": {"snapshot_level": {SNAPSHOT_LICENSE: 50586},
                           "n_disallowed_collected": 0,
                           "n_unclear_collected": 0},
        "sources_not_selected": {
            "ChEBI": "officially confirmed usable (745841) but no local snapshot/cache in "
                     "artifacts/data//tmp (zero hits); requires download; not minimal-first; deferred",
            "COCONUT": "already inside base (coco_extra 436389 shipped, coconut-casmi26-candidates v2, "
                       "CC BY 4.0); not a new source; full re-download out of scope; not selected",
            "PubChem": "~105M structures per high-score pipeline asset note; full download banned by task "
                       "constraint; mixed licenses; not moved; deferred",
            "MassBank": "done in 008 (8831 distinct / 539 new vs base); this task is an independent source; "
                        "not repeated",
        },
        "n_canon_distinct_single_source_index": int(len(entries)),
        "full_source_overlap_with_base712K_union": n_overlap_full,
        "full_source_new_vs_base712K": n_new_full,
        "index_overlap_with_base712K_union": int(n_overlap_idx) if n_overlap_idx is not None else None,
        "index_new_vs_base712K": int(len(entries)) if n_overlap_idx == 0 else None,
        "base_union_canons_for_stats": int(len(union_canons)) if union_canons else None,
        "base_shipped_for_stats": 712199,
        "reverification": {"n_canon_mismatch": 0, "n_unparseable": 0, "n_mass_fail": 0,
                           "n_mass_mismatch_1e9": 0, "rows": 22765},
        "canonical_rule": "src.chemistry.canonical_group_key (RemoveStereochemistry -> "
                          "TautomerEnumerator.Canonicalize -> InChIKey14)",
        "rdkit_version": str(rdkit_ver),
        "mass_rule": "RDKit ExactMolWt float64 (v32 fp_and_mass verbatim, same as pool mass.npy)",
        "query_rule": "src.candidate_expansion.query_candidates_unlabeled only (peaks, precursor_mz, adduct, "
                      "instrument, CE; predicted_formula optional from spectral predictor only; truth formula "
                      "only in oracle_diagnose_truth_in_list). Unknown adduct -> mass_uncertain, no accurate window.",
        "adduct_rule": "retrieval.ADDUCT_TABLE 15 classes + [M-H2O-H]- explicit (delta -19.01784115344); "
                       "None/empty/unknown -> mass_uncertain True",
        "ppm": 10.0,
        "wmax_fp_top100_prefix": 100,
        "final_top25": 25,
        "homology_boundary": "R3-50 spectra are MassBank-derived evaluation queries; LMSD is an independent "
                             "lipid-structure source (no shared record provenance). Chemical overlap may still "
                             "exist; reported as observed, unknown exposure/temporal marked unknown, no "
                             "'guaranteed unseen' claim.",
        "temporal_note": "unknown; no guarantee-unseen claim",
        "local_B_note": "B means currently out-of-library (base 712K), not hidden-category inference",
        "inputs_sha256": inputs_sha,
        "forbidden_respected": ["no test files", "no training", "no GPU", "no Kaggle quota",
                                "no heldout spectra read/eval", "no truth SMILES extraction",
                                "no disallowed-license collection", "no network query",
                                "no large-full download", "no G1-reserve read"],
        "timing_s": float(time.time() - t0),
    }
    json.dump(meta, open(out_meta, "w"), indent=1, allow_nan=False)

    manifest = {
        "index_sources": [
            {"name": "LIPID MAPS LMSD snapshot (single source)",
             "zip": os.path.relpath(LMSD_ZIP, REPO),
             "bytes": int(inputs_sha[os.path.relpath(LMSD_ZIP, REPO)]["bytes"]),
             "sha256": str(inputs_sha[os.path.relpath(LMSD_ZIP, REPO)]["sha256"]),
             "snapshot_license": SNAPSHOT_LICENSE,
             "allowed_licenses_collected": sorted(list(ALLOWED)),
             "excluded_licenses_collected": [],
             "derived_verified_rows": [os.path.relpath(p, REPO) for p in
                                       [LMSD_RAW, LMSD_UNIQUE, LMSD_NEW, LMSD_KEYS, LMSD_MASS,
                                        LMSD_SMILES, LMSD_LMIDS]]},
            {"name": "fp_bits mapping (v32)",
             "path": os.path.relpath(FP_BITS, REPO),
             "bytes": int(inputs_sha[os.path.relpath(FP_BITS, REPO)]["bytes"]),
             "sha256": str(inputs_sha[os.path.relpath(FP_BITS, REPO)]["sha256"])},
            {"name": "base pool maps for overlap stats only (not for filtering)",
             "paths": [os.path.relpath(TRAIN_MAP, REPO), os.path.relpath(COCO_MAP, REPO)],
             "union_canons": int(len(union_canons)) if union_canons else None},
        ],
        "canonical_rule": meta["canonical_rule"],
        "rdkit_version": str(rdkit_ver),
        "mass_rule": meta["mass_rule"],
        "query_rule": meta["query_rule"],
        "outputs": {k: os.path.relpath(p, REPO) for k, p in
                    [("masses", out_masses), ("keys", out_keys), ("smiles", out_smiles),
                     ("lm_ids", out_lmids), ("records", out_records), ("meta", out_meta)]},
    }
    json.dump(manifest, open(out_manifest, "w"), indent=1, allow_nan=False)
    print(f"wrote {out_masses} masses {len(masses)}", flush=True)
    print(f"overlap_full {n_overlap_full} new_full {n_new_full}", flush=True)
    print(f"timing {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

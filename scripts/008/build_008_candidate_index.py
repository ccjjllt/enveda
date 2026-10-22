#!/opt/homebrew/bin/python3.11
"""008-B 建索引：从 MassBank 2026.03 bulk 快照建固定外部结构索引（CPU-only，手动研究脚本，非测试文件）。

输入（只读，不联网）：
- bulk zip：/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/massbank-2026.03.zip
  （MassBank-data 2026.03，Zenodo https://doi.org/10.5281/zenodo.19073053，
  release CC BY 4.0，但 per-record LICENSE  governing）。
- base 池 maps（仅用于统计 overlap，不用于筛选外部索引）：
  artifacts/runs/005/structure_identity_train.parquet + structure_identity_coco.parquet
- fp_bits：artifacts/runs/005/coco/fp_bits.npy（仅记录 SHA，不在此处算全量指纹）

筛选（独立完整来源筛选，不读 observations_*.json / truth SMILES）：
- LICENSE ∈ {CC0, CC BY, CC BY-SA, MIT} 才收；dl-de/by-2-0、CC BY-NC*、缺失/不明不收，
  计数为 discards（许可不明/禁再分发不收）。
- CH$SMILES 存在且非 N/A；RDKit 可解析；canonical_group_key 可算；
  ExactMolWt 有限；formula 可算。失败分别计数，不伪造质量。
- 去重：先按 SMILES 字符串去重降 RDKit 调用，再按 canonical_group_key 去重
  （保留首次 accession，记录 n_records / accessions）。
- 不按 class_AB / truth key / truth SMILES 过滤或分配预算；评价谱同源边界在
  输出 meta 同源声明中记录（评价 50 谱同样来自 MassBank，覆盖可能偏乐观）。

输出（新增目录 artifacts/runs/008-expansion/，不覆盖 007-r3）：
- candidate_index_masses.npy（float64）、candidate_index_keys.npy（canonical str）、
  candidate_index_smiles.txt、candidate_index_meta.json（含快照/SHA/许可/canonical规则/统计）、
  candidate_index_records.jsonl（每 canonical 一行：代表 SMILES/formula/mass/license/accessions）
- sources_manifest.json（索引来源清单：快照路径/SHA/n_txt/许可分布/canonical规则/fp_bits SHA）
- build.log（人工核对用）

零 GPU、CPU-only，不 push/submit，不读保留集。最终索引离线可用（.npy/.json/.txt，不联网查询）。
"""

import hashlib
import json
import os
import re
import sys
import time
import zipfile
from collections import Counter

from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, os.path.join(REPO, "src"))

OUTDIR = os.path.join(REPO, "artifacts", 'runs', "008-expansion")
BULK_ZIP = "/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/massbank-2026.03.zip"
FP_BITS = os.path.join(REPO, "artifacts", 'runs', "005", "coco", "fp_bits.npy")
TRAIN_MAP = os.path.join(REPO, "artifacts", 'runs', "005", "structure_identity_train.parquet")
COCO_MAP = os.path.join(REPO, "artifacts", 'runs', "005", "structure_identity_coco.parquet")

ALLOWED = {"CC0", "CC BY", "CC BY-SA", "MIT"}


def sha_file(path, limit=None):
    h = hashlib.sha256()
    sz = 0
    with open(path, "rb") as f:
        while True:
            ch = f.read(1 << 20)
            if not ch:
                break
            h.update(ch)
            sz += len(ch)
            if limit is not None and sz >= limit:
                break
    return sz, h.hexdigest()


def main():
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    out_masses = os.path.join(OUTDIR, "candidate_index_masses.npy")
    out_keys = os.path.join(OUTDIR, "candidate_index_keys.npy")
    out_smiles = os.path.join(OUTDIR, "candidate_index_smiles.txt")
    out_meta = os.path.join(OUTDIR, "candidate_index_meta.json")
    out_records = os.path.join(OUTDIR, "candidate_index_records.jsonl")
    out_manifest = os.path.join(OUTDIR, "sources_manifest.json")
    for p in [out_masses, out_keys, out_smiles, out_meta, out_records, out_manifest]:
        assert not os.path.exists(p), f"refuse overwrite {p}（只增不改；如需重跑先人工归档）"

    assert os.path.exists(BULK_ZIP), f"missing bulk zip {BULK_ZIP}"
    assert os.path.exists(FP_BITS), f"missing fp_bits {FP_BITS}"
    import numpy as np

    bulk_bytes, bulk_sha = sha_file(BULK_ZIP)
    bits_bytes, bits_sha = sha_file(FP_BITS)
    bits_arr = np.load(FP_BITS)
    assert bits_arr.shape == (6930,), bits_arr.shape
    print(f"bulk {BULK_ZIP} bytes {bulk_bytes} sha {bulk_sha}", flush=True)
    print(f"fp_bits {bits_arr.shape} sha {bits_sha} bytes {bits_bytes}", flush=True)

    try:
        from chemistry import canonical_group_key
        import rdkit as _rdk
        rdkit_ver = str(_rdk.__version__)
    except Exception as e:
        raise RuntimeError(f"need RDKit for canonical_group_key: {e}")

    # base union canonicals（仅统计 overlap，不筛选）
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

    z = zipfile.ZipFile(BULK_ZIP)
    names = [n for n in z.namelist() if n.endswith(".txt")]
    print(f"bulk n_txt {len(names)}", flush=True)

    lic_counter = Counter()
    # 第一遍：快照许可分布（不解析 SMILES，只读 LICENSE 行，快速）
    # 为省时，直接在第二遍中一并计数；此处跳过单独遍。

    # 第二遍：解析记录，按 SMILES 字符串去重
    smile_to_recs = {}
    n_no_smiles = 0
    n_disallowed_license = 0
    disallowed_counter = Counter()
    for n in names:
        try:
            txt = z.read(n).decode("utf-8", errors="ignore")
        except Exception:
            continue
        m_lic = re.search(r"^LICENSE:\s*(.+)$", txt, re.M)
        lic = m_lic.group(1).strip() if m_lic else ""
        lic_counter[lic] += 1
        if lic not in ALLOWED:
            n_disallowed_license += 1
            disallowed_counter[lic if lic else "MISSING"] += 1
            continue
        m_smi = re.search(r"^CH\$SMILES:\s*(.+)$", txt, re.M)
        smi = m_smi.group(1).strip() if m_smi else ""
        if not smi or smi == "N/A":
            n_no_smiles += 1
            continue
        m_acc = re.search(r"^ACCESSION:\s*(.+)$", txt, re.M)
        acc = m_acc.group(1).strip() if m_acc else n
        m_for = re.search(r"^CH\$FORMULA:\s*(.+)$", txt, re.M)
        formula_rec = m_for.group(1).strip() if m_for else ""
        ent = smile_to_recs.get(smi)
        if ent is None:
            smile_to_recs[smi] = {"smiles": smi, "accessions": [acc], "licenses": [lic],
                                  "formula_record": formula_rec, "zip_path": n}
        else:
            ent["accessions"].append(acc)
            ent["licenses"].append(lic)
    print(f"allowed SMILES distinct {len(smile_to_recs)} disallowed {n_disallowed_license} no_smiles {n_no_smiles}", flush=True)
    print(f"license dist (bulk all): {dict(lic_counter)}", flush=True)

    # 第三遍：RDKit canonical + mass + formula（distinct SMILES）
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.Descriptors import ExactMolWt
    from rdkit import Chem

    canon_to_entry = {}
    n_unparseable = 0
    n_canon_fail = 0
    n_mass_fail = 0
    for i, (smi, rec) in enumerate(smile_to_recs.items()):
        if (i + 1) % 5000 == 0:
            print(f"  rdkit {i+1}/{len(smile_to_recs)} canon_entries {len(canon_to_entry)} ...", flush=True)
        try:
            mol = Chem.MolFromSmiles(str(smi))
        except Exception:
            mol = None
        if mol is None:
            n_unparseable += 1
            continue
        try:
            canon = canonical_group_key(str(smi))
        except Exception:
            n_canon_fail += 1
            continue
        try:
            mass = float(ExactMolWt(mol))
            formula = str(rdMolDescriptors.CalcMolFormula(mol))
        except Exception:
            n_mass_fail += 1
            continue
        import math as _math

        if not _math.isfinite(mass) or mass <= 0:
            n_mass_fail += 1
            continue
        ent = canon_to_entry.get(canon)
        if ent is None:
            # 代表许可：取首条（均为允许类）；记录许可集合
            lic_set = sorted(set(rec["licenses"]))
            canon_to_entry[canon] = {
                "canonical_key": str(canon),
                "smiles_rep": str(smi),
                "formula_calc": str(formula),
                "mass_theo": float(mass),
                "license_rep": str(rec["licenses"][0]),
                "licenses_set": lic_set,
                "n_records": int(len(rec["accessions"])),
                "accessions": list(rec["accessions"][:5]),
                "n_accessions_total": int(len(rec["accessions"])),
            }
        else:
            ent["n_records"] += int(len(rec["accessions"]))
            ent["n_accessions_total"] += int(len(rec["accessions"]))
            for lic in rec["licenses"]:
                if lic not in ent["licenses_set"]:
                    ent["licenses_set"].append(lic)
            ent["licenses_set"] = sorted(ent["licenses_set"])

    print(f"canon distinct {len(canon_to_entry)} unparse {n_unparseable} canon_fail {n_canon_fail} mass_fail {n_mass_fail}", flush=True)

    # 排序：按 mass 升序（确定性，与质量索引一致）
    entries = sorted(canon_to_entry.values(), key=lambda e: (e["mass_theo"], e["canonical_key"]))
    masses = np.asarray([e["mass_theo"] for e in entries], dtype=np.float64)
    keys = np.asarray([e["canonical_key"] for e in entries], dtype=object)
    smiles_list = [e["smiles_rep"] for e in entries]

    # overlap 统计（仅报告，不筛选）
    if union_canons:
        n_overlap = sum(1 for e in entries if e["canonical_key"] in union_canons)
        n_new = len(entries) - n_overlap
    else:
        n_overlap = None
        n_new = None
    print(f"overlap with base712K union: overlap {n_overlap} new {n_new}", flush=True)

    np.save(out_masses, masses)
    np.save(out_keys, keys)
    with open(out_smiles, "w") as f:
        for s in smiles_list:
            f.write(str(s) + "\n")
    with open(out_records, "w") as f:
        for e in entries:
            f.write(json.dumps(e, allow_nan=False) + "\n")

    meta = {
        "task": "008-B external candidate index (fixed, MassBank snapshot, unlabeled query)",
        "created_utc": "2026-10-12T00:00:00Z",
        "budget": "local CPU only; no training; no GPU; no Kaggle quota; no heldout spectra read/eval; writable only artifacts/runs/008-expansion/",
        "snapshot": {
            "bulk_zip": BULK_ZIP,
            "bytes": int(bulk_bytes),
            "sha256": str(bulk_sha),
            "n_txt": int(len(names)),
            "release": "MassBank-data 2026.03 CC BY 4.0 https://doi.org/10.5281/zenodo.19073053 (release license noted; per-record LICENSE governs)",
            "zip_top_prefix": str(names[0].split("/")[0] if names else ""),
        },
        "license_policy": "Only CC0/CC-BY/CC-BY-SA/MIT accepted; dl-de/by-2-0, CC BY-NC*, missing/unclear excluded and counted (not collected). Per-record LICENSE verbatim governs.",
        "license_counts_bulk": {str(k): int(v) for k, v in lic_counter.items()},
        "license_counts_disallowed": {str(k): int(v) for k, v in disallowed_counter.items()},
        "n_disallowed_license": int(n_disallowed_license),
        "n_no_smiles": int(n_no_smiles),
        "n_smiles_distinct_allowed": int(len(smile_to_recs)),
        "n_unparseable_smiles": int(n_unparseable),
        "n_canon_fail": int(n_canon_fail),
        "n_mass_fail": int(n_mass_fail),
        "n_canon_distinct_external": int(len(entries)),
        "n_overlap_with_base712K_union": n_overlap,
        "n_new_vs_base712K": n_new,
        "base_union_canons_for_stats": int(len(union_canons)) if union_canons else None,
        "canonical_rule": "src.chemistry.canonical_group_key (RemoveStereochemistry -> TautomerEnumerator.Canonicalize -> InChIKey14)",
        "rdkit_version": str(rdkit_ver),
        "mass_rule": "RDKit ExactMolWt float64 (v32 fp_and_mass verbatim, same as pool mass.npy)",
        "fp_rule": "v32 concat 10407 -> fp_bits.npy select 6930 (sha %s); fingerprints NOT precomputed for all external here (on-demand for window candidates in evaluation, same caliber); index masses+smiles offline-usable, no network query" % str(bits_sha[:12]),
        "fp_bits": {"path": FP_BITS, "bytes": int(bits_bytes), "sha256": str(bits_sha)},
        "query_rule": "src.candidate_expansion.query_candidates_unlabeled only (peaks, precursor_mz, adduct, instrument, CE; predicted_formula optional from spectral predictor only; truth formula only in oracle_diagnose_truth_in_list). Unknown adduct -> mass_uncertain, no accurate window.",
        "adduct_rule": "retrieval.ADDUCT_TABLE 15 classes + [M-H2O-H]- explicit (delta -19.01784115344); None/empty/unknown -> mass_uncertain True",
        "ppm": 10.0,
        "wmax_fp_top100_prefix": 100,
        "final_top25": 25,
        "homology_boundary": "Evaluation 50 spectra are also from MassBank (same source as index); external coverage may be optimistic vs truly unseen structures. Recorded here, not corrected.",
        "forbidden_respected": ["no test files", "no training", "no GPU", "no Kaggle quota", "no heldout spectra read/eval", "no truth SMILES extraction", "no disallowed-license collection", "no network query"],
        "timing_s": float(time.time() - t0),
    }
    json.dump(meta, open(out_meta, "w"), indent=1, allow_nan=False)

    manifest = {
        "index_sources": [
            {
                "name": "MassBank-data 2026.03 bulk snapshot",
                "path": BULK_ZIP,
                "bytes": int(bulk_bytes),
                "sha256": str(bulk_sha),
                "n_txt": int(len(names)),
                "release_license": "CC BY 4.0 (release); per-record LICENSE governs",
                "doi": "https://doi.org/10.5281/zenodo.19073053",
                "allowed_licenses_collected": sorted(list(ALLOWED)),
                "excluded_licenses": sorted([str(k) for k in disallowed_counter.keys()]),
            },
            {
                "name": "fp_bits mapping (v32)",
                "path": FP_BITS,
                "bytes": int(bits_bytes),
                "sha256": str(bits_sha),
            },
            {
                "name": "base pool maps for overlap stats only (not for filtering)",
                "paths": [TRAIN_MAP, COCO_MAP],
                "union_canons": int(len(union_canons)) if union_canons else None,
            },
        ],
        "canonical_rule": meta["canonical_rule"],
        "rdkit_version": str(rdkit_ver),
        "mass_rule": meta["mass_rule"],
        "query_rule": meta["query_rule"],
        "outputs": {
            "masses": os.path.relpath(out_masses, REPO),
            "keys": os.path.relpath(out_keys, REPO),
            "smiles": os.path.relpath(out_smiles, REPO),
            "meta": os.path.relpath(out_meta, REPO),
            "records": os.path.relpath(out_records, REPO),
        },
    }
    json.dump(manifest, open(out_manifest, "w"), indent=1, allow_nan=False)
    print(f"wrote {out_masses} {os.path.getsize(out_masses)} masses {len(masses)}", flush=True)
    print(f"wrote {out_keys} {os.path.getsize(out_keys)}", flush=True)
    print(f"wrote {out_records} {os.path.getsize(out_records)}", flush=True)
    print(f"timing {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

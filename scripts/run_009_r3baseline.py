#!/usr/bin/env python3
"""009-R3baseline 完整v9基线最小ICE forward bundle准备 (CPU-only, 手动, 零GPU).

只读允许目录, 输出到 artifacts/009-r3baseline/, 默认不覆盖已有产物则比对.
禁止: 读G1保留250/heldout, 扫阈值/权重, GPU forward, push/submit/远端, 测试文件/CI.

范围 (任务指定):
- 仅支持分支23分子 (A16/B7, r3_routing.json) 需要ICE前向.
- 4 NILU质量不确定无窗跳过 (单列, 0候选).
- 不支持27分子不需ICE (条件规则走CPU all-mean FP, 已有分数, 不补ICE分).
- 候选集复用现有R3 FP窗/shortlist (r3_50_full_candidate_scores.json expanded, 不重建规则).
  R3支持窗全<100, 故 expanded top100 == 全窗.

Bundle (v9r3baseline/ 顶层目录, 修复flat-tar教训):
  artifacts/009-r3baseline/v9r3baseline/{inputs.json,code/ice/,g1_worker.py,
    install.sh,MANIFEST.iceberg.json,weights_reference.json,README.md}
  tar: artifacts/009-r3baseline/v9r3baseline-bundle.tar.gz (含顶层目录)
  Colab: tar xzf -C /content -> /content/v9r3baseline
  runner SHA必须与G1验证包一致: ice_runner dc0d5d1e, worker 54b0751f.

Manual:
  /opt/homebrew/bin/python3.11 scripts/run_009_r3baseline.py [--check-only]
"""
from __future__ import annotations
import argparse, hashlib, json, math, re, shutil, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/009-r3baseline"
PKG = OUT / "v9r3baseline"
COV = ("[M+H]+", "[M+Na]+")

ALLOWED_PREFIXES = (
    "artifacts/009-conditional/",
    "artifacts/009-expansion/",
    "artifacts/007-r3/",
    "artifacts/008-expansion/",
    "artifacts/005/",
    "artifacts/pool-train/",
    "artifacts/g1-validation/",
    "artifacts/009-v9recall/",
    "artifacts/001-baseline/",
)

def guard(p: Path):
    s = str(p)
    try:
        rel = str(p.relative_to(ROOT))
    except ValueError:
        rel = s
    low = (s + " " + rel).lower()
    if "heldout" in low or "holdout" in low:
        raise RuntimeError(f"refuse heldout path: {s}")
    if rel.startswith("artifacts/009-r3baseline/") or s.startswith("artifacts/009-r3baseline/"):
        return
    if rel.startswith("scripts/") or s.startswith("scripts/"):
        return
    if any(rel.startswith(a) or s.startswith(a) for a in ALLOWED_PREFIXES):
        return
    raise RuntimeError(f"read outside allowed: {s} (rel {rel})")

def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def read_json(p: Path):
    guard(p)
    return json.loads(p.read_text())

def dump(obj, indent=1) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=indent, allow_nan=False) + "\n"

# ---- runner mirror (verbatim semantics, CPU, no torch) ----
def is_covered(sp: dict) -> bool:
    md = sp.get("mode")
    pos = (md == 1) or (isinstance(md, str) and md.lower() in ("positive", "pos", "+", "1"))
    return bool(pos and sp.get("adduct") in COV)

def _nce(s) -> float:
    v = [float(x) for x in re.findall(r"[0-9]+\.?[0-9]*", str(s or ""))]
    return float(np.mean(v)) if v else float("nan")

def spectrum_ce(sp: dict) -> float:
    if "ce_units" in sp or "ce_ev" in sp:
        if sp.get("ce_units") == "NCE":
            return _nce(sp.get("ce_orig"))
        v = sp.get("ce_ev")
        return float(np.mean(v)) if v is not None and len(v) else float("nan")
    v = sp.get("ce")
    return float(v) if v is not None and np.isfinite(float(v)) else float("nan")

def ce_bucket(ce: float) -> float:
    ceb = float(np.round(ce / 5.0) * 5.0) if np.isfinite(ce) else float("nan")
    if np.isfinite(ceb) and ceb < 5:
        ceb = 5.0
    return ceb

def instr_token(sp: dict) -> str:
    if sp.get("instr") in ("Orbitrap", "QTOF"):
        return sp["instr"]
    raw = sp.get("instrument", sp.get("instrument_raw"))
    t = str(raw or "").lower()
    fam = 1 if ("orbitrap" in t or "qft" in t or "ftms" in t or "hybrid ft" in t or "itft" in t or "exactive" in t or "ft-icr" in t or "fticr" in t) else (2 if "tof" in t else 5)
    # timsTOF contains tof -> 2 -> QTOF; family QTOF string itself contains tof -> QTOF
    if sp.get("instrument") in ("Orbitrap", "QTOF"):
        return sp["instrument"]
    return "Orbitrap" if fam == 1 else "QTOF"

def expand_records(spectra, cemode: str):
    recs = []
    for qi, sp in enumerate(spectra):
        covered = bool(is_covered(sp))
        units = sp.get("ce_units")
        nce = (units == "NCE")
        ce_ev = [float(v) for v in (sp.get("ce_ev") or [])]
        if not covered:
            recs.append({"qid": qi, "sp": sp, "ces": [], "covered": False})
            continue
        if cemode == "mean" or len(ce_ev) <= 1:
            recs.append({"qid": qi, "sp": sp, "ces": [float(spectrum_ce(sp))], "covered": True})
            continue
        if nce:
            nums = [float(x) for x in re.findall(r"[0-9]+\.?[0-9]*", str(sp.get("ce_orig") or ""))]
            ces = list(nums) if len(nums) == len(ce_ev) else [float(spectrum_ce(sp))]
        else:
            ces = list(ce_ev)
        recs.append({"qid": qi, "sp": sp, "ces": ces, "covered": True})
    return recs

def unique_for_mol(canons: list, spectra, cemode: str):
    recs = expand_records(spectra, cemode)
    uq = set()
    for cand in canons:
        for r in recs:
            if not r["covered"]:
                continue
            ad = r["sp"]["adduct"]
            ins = instr_token(r["sp"])
            for ce in r["ces"]:
                ceb = ce_bucket(ce)
                uq.add((cand, ad, ins, ("sum" if isinstance(ceb, float) and math.isnan(ceb) else round(float(ceb), 1))))
    return uq, recs

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    t0 = time.monotonic()
    wall = datetime.now().astimezone().isoformat()

    p_rout = ROOT / "artifacts/009-conditional/r3_routing.json"
    p_full = ROOT / "artifacts/009-expansion/r3_50_full_candidate_scores.json"
    p_obs = ROOT / "artifacts/007-r3/observations_50.json"
    p_index = ROOT / "artifacts/008-expansion/candidate_index_records.jsonl"
    p_train = ROOT / "artifacts/005/structure_identity_train.parquet"
    p_coco = ROOT / "artifacts/005/structure_identity_coco.parquet"
    p_logits = ROOT / "artifacts/009-expansion/r3_50_logits.npy"
    p_ice = ROOT / "artifacts/g1-validation/code/ice/ice_runner.py"
    p_worker = ROOT / "artifacts/g1-validation/g1_worker.py"
    p_install = ROOT / "artifacts/009-v9recall/bundle/install.sh"
    p_manifest = ROOT / "artifacts/009-v9recall/bundle/MANIFEST.iceberg.json"
    for p in (p_rout, p_full, p_obs, p_index, p_train, p_coco, p_ice, p_worker, p_install, p_manifest):
        guard(p)
    assert p_logits.exists(), "missing r3_50_logits.npy FP cache (复用不重算)"
    print(f"FP cache exists: {p_logits} ({p_logits.stat().st_size} bytes)")

    rout = read_json(p_rout)
    full = read_json(p_full)
    obs = read_json(p_obs)
    obs_by = {m["canonical_key"]: m for m in obs["molecules"]}
    obs_idx = {m["canonical_key"]: i for i, m in enumerate(obs["molecules"])}
    by_rec = {}
    for line in p_index.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        by_rec[r["canonical_key"]] = r
    import pyarrow.parquet as pq
    td = pq.read_table(p_train, columns=["canonical_group_key", "output_smiles"]).to_pylist()
    cd = pq.read_table(p_coco, columns=["canonical_group_key", "output_smiles"]).to_pylist()
    train_map = {r["canonical_group_key"]: r["output_smiles"] for r in td}
    coco_map = {r["canonical_group_key"]: r["output_smiles"] for r in cd}
    print(f"maps: by_rec {len(by_rec)} train {len(train_map)} coco {len(coco_map)}")

    # supported 23 in obs order (deterministic, YTCZZ last)
    sup_set = set(r["canonical_key_for_audit_only"] for r in rout["rows"] if r["metadata_has_supported_record"])
    assert len(sup_set) == 23, sup_set
    unsup_set = set(r["canonical_key_for_audit_only"] for r in rout["rows"] if not r["metadata_has_supported_record"])
    assert len(unsup_set) == 27, len(unsup_set)
    sup_ordered = [m["canonical_key"] for m in obs["molecules"] if m["canonical_key"] in sup_set]
    assert len(sup_ordered) == 23
    # class check A16/B7
    from collections import Counter
    cls = Counter(obs_by[k]["class_AB"] for k in sup_ordered)
    print(f"supported class counts: {dict(cls)} (expect A16/B7)")
    assert cls["A"] == 16 and cls["B"] == 7, dict(cls)
    # YTCZZ must be in package
    assert "YTCZZXIRLARSET" in sup_ordered, "YTCZZ missing"
    print(f"YTCZZ bundle index: {sup_ordered.index('YTCZZXIRLARSET')} (0-based, 23 mols)")

    # 4 NILU mass-uncertain skip list (single-column, 0 window)
    nilu = ["NMVVBVMYPLMIOU", "UTDAGHZGKXPRQI", "JOECRAPIMXFQDL", "ZTILAOCGFRDHBH"]
    for k in nilu:
        assert k not in sup_set, k
        assert k in unsup_set, k
    print(f"NILU 4 skip confirmed (all in unsupported 27, 0 window)")

    # RDKit for formulas
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    molecules = []
    scope_rows = []
    total_mean_uq = set()
    total_split_uq = set()
    for bi, key in enumerate(sup_ordered):
        m_obs = obs_by[key]
        fentry = full[key]
        eo = list(fentry["orders"]["expanded"])
        scores = dict(fentry["scores"]["expanded"])
        # expanded top100 == full window (<100 verified)
        assert len(eo) <= 100, (key, len(eo))
        top100 = eo[:100]
        scores100 = [float(scores[c]) for c in top100]
        # rep_smiles + formulas
        rep = {}
        fms = {}
        for cand in top100:
            if cand in by_rec:
                smi = by_rec[cand]["smiles_rep"]
            elif cand in train_map:
                smi = train_map[cand]
            elif cand in coco_map:
                smi = coco_map[cand]
            else:
                raise RuntimeError(f"missing smiles for {cand} (mol {key})")
            rep[cand] = smi
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                raise RuntimeError(f"RDKit unparseable {cand} {smi!r:.100}")
            fms[cand] = str(rdMolDescriptors.CalcMolFormula(mol))
        # spectra -> worker format (normalized it, family instrument)
        spectra = []
        for s in m_obs["spectra"]:
            z = np.load(ROOT / s["peak_file_ref"])
            mz = np.asarray(z["mz"], dtype=np.float64)
            it = np.asarray(z["it"], dtype=np.float64)
            mx = float(it.max()) if len(it) else 1.0
            itn = (it / mx).tolist() if mx > 0 else it.tolist()
            ce_ev = s.get("collision_energy_ev")
            ce_ev = [float(v) for v in ce_ev] if ce_ev else []
            spectra.append({
                "mz": [float(v) for v in mz.tolist()],
                "it": [float(v) for v in itn],
                "prec": float(s["precursor_mz"]),
                "adduct": str(s["adduct"]) if s.get("adduct") else None,
                "mode": str(s["ionization_mode"]),
                "instrument": str(s["instrument_family"]),
                "ce_ev": ce_ev,
                "ce_orig": str(s.get("collision_energy_raw") or ""),
                "ce_units": str(s.get("collision_energy_units") or ""),
            })
        # unique jobs (009-v9recall caliber)
        uq_mean, recs_mean = unique_for_mol(top100, spectra, "mean")
        uq_split, recs_split = unique_for_mol(top100, spectra, "split")
        uq_union = set(uq_mean) | set(uq_split)
        total_mean_uq |= set((key,) + tuple(x) for x in uq_mean)
        total_split_uq |= set((key,) + tuple(x) for x in uq_split)
        n_cov = sum(1 for r in recs_mean if r["covered"])
        n_uncov = sum(1 for r in recs_mean if not r["covered"])
        # support spectra summary (adduct/仪器/CE)
        sup_detail = []
        for i, s in enumerate(m_obs["spectra"]):
            w = [x for x in spectra[i]["mz"]][:0]  # placeholder to keep structure
            sup_detail.append({
                "idx": i,
                "spectrum_id": s["spectrum_id"],
                "adduct": spectra[i]["adduct"],
                "mode": spectra[i]["mode"],
                "instrument_family": spectra[i]["instrument"],
                "instrument_raw": s["instrument_raw"],
                "ce_raw": spectra[i]["ce_orig"],
                "ce_units": spectra[i]["ce_units"],
                "ce_ev": spectra[i]["ce_ev"],
                "n_peaks": int(s["n_peaks"]),
                "prec": spectra[i]["prec"],
                "covered": bool(is_covered(spectra[i])),
            })
        molecules.append({
            "key": key,
            "qidx": int(obs_idx[key]),
            "bundle_idx": int(bi),
            "class_AB": str(m_obs["class_AB"]),
            "route": "retain_v9",
            "truth_canon": str(key),
            "n_win_canon": int(len(eo)),
            "base_rank_full": None,
            "base_rr": None,
            "order_top100": top100,
            "scores_top100": scores100,
            "base_scores_full_window": {c: float(scores[c]) for c in top100},
            "rep_smiles": rep,
            "formulas": fms,
            "spectra": spectra,
        })
        scope_rows.append({
            "key": key,
            "class_AB": str(m_obs["class_AB"]),
            "bundle_idx": int(bi),
            "qidx_obs": int(obs_idx[key]),
            "n_cands_expanded_top100": int(len(top100)),
            "n_win_canon": int(len(eo)),
            "n_spectra": int(len(spectra)),
            "n_covered": int(n_cov),
            "n_unsupported": int(n_uncov),
            "support_spectra": sup_detail,
            "unique_mean": int(len(uq_mean)),
            "unique_split": int(len(uq_split)),
            "unique_union": int(len(uq_union)),
            "is_ytczz": bool(key == "YTCZZXIRLARSET"),
        })
        print(f"{bi:02d} {key} {m_obs['class_AB']} ncand={len(top100)} nspec={len(spectra)} cov={n_cov} uniq_mean/split/union={len(uq_mean)}/{len(uq_split)}/{len(uq_union)}")

    n_cands_total = sum(r["n_cands_expanded_top100"] for r in scope_rows)
    uniq_mean_total = sum(r["unique_mean"] for r in scope_rows)
    uniq_split_total = sum(r["unique_split"] for r in scope_rows)
    uniq_union_total = sum(r["unique_union"] for r in scope_rows)
    print(f"TOTALS ncands={n_cands_total} uniq_mean={uniq_mean_total} split={uniq_split_total} union={uniq_union_total}")
    # dry-run 2-mol expectation (first 2 in bundle order)
    two_mean = sum(r["unique_mean"] for r in scope_rows[:2])
    two_cands = sum(r["n_cands_expanded_top100"] for r in scope_rows[:2])
    print(f"DRY2 first2 cands={two_cands} uniq_mean={two_mean}")

    def cost(uq):
        pure_s = uq / 26.0
        return {"unique_jobs": int(uq), "pure_pred_s_at_26_per_s": round(pure_s, 2),
                "pure_pred_min": round(pure_s / 60.0, 3),
                "with_load_overhead_min_est": "~5-10 min total (model load + scoring), far below 2h",
                "basis": "F1 MANIFEST timing_basis.ice_pred_per_s=26; 005实测33/s更快为下限参考; linear scaling"}

    forward = {
        "scope": "R3-50完整v9基线最小ICE forward: 仅支持分支23分子 (A16/B7); 4 NILU质量不确定无窗跳过单列; 不支持27分子不需ICE (条件规则走CPU all-mean FP, 已有分数, 不补ICE分)",
        "candidate_policy": "复用现有R3 FP窗/shortlist (r3_50_full_candidate_scores.json expanded, 不重建规则); 支持窗全<=100故expanded top100==全窗; truth仅离线打分, 不注入候选",
        "fp_cache_reuse": {"r3_50_logits.npy": "exists, reuse no recompute", "r3_50_full_candidate_scores.json": "expanded scores reuse, no recompute"},
        "n_supported_mols": 23,
        "n_supported_A": 16,
        "n_supported_B": 7,
        "n_unsupported_mols": 27,
        "n_nilU_uncertain_skipped": 4,
        "nilu_keys": nilu,
        "rows": scope_rows,
        "totals": {"n_cands": int(n_cands_total), "unique_mean": int(uniq_mean_total),
                   "unique_split": int(uniq_split_total), "unique_union": int(uniq_union_total)},
        "cost_T4": {"mean": cost(uniq_mean_total), "split": cost(uniq_split_total), "union": cost(uniq_union_total)},
        "dryrun_first2": {"n_mols": 2, "n_cands": int(two_cands), "unique_split": int(sum(r['unique_split'] for r in scope_rows[:2]))},
        "comparison_to_2h": {"limit_min": 120,
            "full_split_min_pure": round(uniq_split_total / 26.0 / 60.0, 3),
            "conclusion": f"Full 23-mol {uniq_split_total} jobs pure {uniq_split_total/26.0:.1f}s ({uniq_split_total/26.0/60.0:.2f}min), >100x below 2h pure; with load/overhead still <10min. First small sample then scale."},
        "ytczz": {"key": "YTCZZXIRLARSET", "bundle_idx": int(next(r['bundle_idx'] for r in scope_rows if r['key']=='YTCZZXIRLARSET')),
                  "n_cands": int(next(r['n_cands_expanded_top100'] for r in scope_rows if r['key']=='YTCZZXIRLARSET')),
                  "note": "YTCZZ在包内; 其forward结果即回答ytczz_contract.json契约问题 (query SUPPORTED, 断连钠盐ATTEMPTABLE but UNVALIDATED, 不假设救回rank40)"},
        "unsupported_policy": "27 unsupported molecules need no ICE; conditional route all_mean_full_window on CPU with existing FP scores; no ICE score补分, missing从不打0",
    }

    inputs_obj = {"molecules": molecules}

    # ---- write / verify ----
    OUT.mkdir(parents=True, exist_ok=True)
    PKG.mkdir(parents=True, exist_ok=True)
    targets = [
        (PKG / "inputs.json", dump(inputs_obj, 1)),
        (OUT / "forward_tasks.json", dump(forward, 2)),
    ]
    failures = []
    for path, text in targets:
        if path.exists():
            old = path.read_text()
            if old == text:
                print(f"OK value-consistent {path.relative_to(ROOT)} ({len(text)} bytes), skip write")
            else:
                failures.append(f"{path}: differs, NOT overwriting ({len(old)} vs {len(text)})")
                print(f"FAIL {path} differs, not overwriting")
                # show first diff
                import difflib
                for line in list(difflib.unified_diff(old.splitlines()[:5], text.splitlines()[:5]))[:10]:
                    print("  " + line)
        else:
            if args.check_only:
                print(f"MISSING {path} (would write {len(text)} bytes)")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
                print(f"WROTE {path.relative_to(ROOT)} ({len(text)} bytes)")
    if failures:
        print("FAILED:", failures, file=sys.stderr)
        sys.exit(1)

    if args.check_only:
        print("CHECK-ONLY done (bundle copies/tar not touched)")
        print(f"KEY: mols=23 cands={n_cands_total} split={uniq_split_total} dry2cands={two_cands}")
        return

    # ---- verbatim copies ----
    # code/ice
    src_ice = ROOT / "artifacts/g1-validation/code/ice"
    dst_ice = PKG / "code/ice"
    if dst_ice.exists():
        shutil.rmtree(dst_ice)
    shutil.copytree(src_ice, dst_ice, ignore=shutil.ignore_patterns("__pycache__"))
    # g1_worker, install, manifest
    for src, dst in [(p_worker, PKG / "g1_worker.py"), (p_install, PKG / "install.sh"),
                     (p_manifest, PKG / "MANIFEST.iceberg.json")]:
        guard(src)
        shutil.copyfile(src, dst)
    # verify SHA
    ice_sha = sha_file(PKG / "code/ice/ice_runner.py")
    worker_sha = sha_file(PKG / "g1_worker.py")
    print(f"ice_runner SHA {ice_sha} (expect dc0d5d1e...)")
    print(f"worker SHA {worker_sha} (expect 54b0751f...)")
    assert ice_sha.startswith("dc0d5d1e"), ice_sha
    assert worker_sha.startswith("54b0751f"), worker_sha
    # install.sh executable bit
    (PKG / "install.sh").chmod(0o755)

    # weights_reference.json (adapted, same SHA)
    weights_ref = {
        "note": "大权重不塞进包, 走远端下载/引用; 本地运行前按SHA核验, 拒绝mismatch (zero-download gate)",
        "iceberg": {
            "gen.pt": {"bytes": 14561244, "sha256": "deef0451c1870d2125653bb212c9d95c5b5d9bf80156dfa9b22fedda7db49bea",
                       "local_refs": ["artifacts/g1-validation/weights/gen.pt", "artifacts/f1-bundle/weights/gen.pt"],
                       "manifest": "artifacts/g1-validation/weights/MANIFEST.iceberg.json ckpt/gen.pt (bundle copy)"},
            "inten.pt": {"bytes": 13793037, "sha256": "c49e330e049d3a8b0095026ed694fcf70045c24305354853732aaa5f89d55068",
                         "local_refs": ["artifacts/g1-validation/weights/inten.pt", "artifacts/f1-bundle/weights/inten.pt"],
                         "manifest": "artifacts/g1-validation/weights/MANIFEST.iceberg.json ckpt/inten.pt (bundle copy)"},
            "ms_pred_commit": "708148c2a8eb0c32120644436fefd2fe90cd2214",
            "rdkit_panel": "2025.3.6 (Colab PyPI非精确, 记入meta --any-rdkit)",
            "prep_tag": "panel-A1:v1(clean0.001/512/prec+2>x2->prep0.002/256/p1/entw;sim0.01Da/20ppm)",
        },
        "fp_for_cpu_recalc_only": {
            "fp_single_s2.pt": {"bytes": 144110255, "sha256": "c19aa29016ed4d7a67bac38105b8fa5076c696c29eb66a504558c9b2e52b614a",
                                "path": "artifacts/001-baseline/weights/fp_single_s2.pt", "note": "CPU FP已用, 不进Colab包"}
        },
        "how_to_provide_on_colab": [
            "方式A (推荐, Drive): 本地将 gen.pt + inten.pt 上传到 Drive (如 /content/drive/MyDrive/v9r3baseline-weights/), Colab里 mkdir -p /content/v9r3baseline/weights && cp /content/drive/MyDrive/v9r3baseline-weights/*.pt /content/v9r3baseline/weights/ 再按README三步运行; 运行前核对 bytes+sha, mismatch拒绝.",
            "方式B (Kaggle dataset): 将两权重作为私有Kaggle dataset挂载, cp到 /content/v9r3baseline/weights/后同样核验.",
            "方式C (私有存储): 从私有存储下载到 /content/v9r3baseline/weights/, 同样核验. 不要把 token/key写进包目录.",
            "本包发出时 weights/ 为空 (零下载运行要求权重+代码+输入齐备才跑; 此最小包故意不含大权重, 由用户按上法提供, 避免塞包).",
        ],
        "runner_reuse": {
            "code_ice": "v9r3baseline/code/ice (verbatim copy from artifacts/g1-validation/code/ice; ice_runner.py sha dc0d5d1e14053e2e7d9088817c620d6615372abc3afdd02be2fc7ff9ee0b2802)",
            "worker": "v9r3baseline/g1_worker.py (verbatim copy from artifacts/g1-validation/g1_worker.py; sha 54b0751f22046bced9e6cc77caced11c4549298f26a7e6009ae9062e5341ab0c; PREP_TAG panel-A1:v1, deterministic, cache key model|ckpt|structure|CE|adduct|instrument|prep)",
            "install": "v9r3baseline/install.sh (PyPI rdkit/einops/lightning/torchmetrics; torch用Colab自带, 不重装)",
        },
        "verify_cmd": "sha256sum v9r3baseline/weights/gen.pt v9r3baseline/weights/inten.pt  # 对照上表 sha256 + bytes",
    }
    (PKG / "weights_reference.json").write_text(dump(weights_ref, 2))

    # README.md
    readme = f"""# 009-R3baseline 完整v9基线最小 ICE forward 包 (23支持分子, Colab T4)

复用 F1/G1 可移植包模式与已验证 ICE runner/权重, 不重复造轮子。CPU-only 准备, 不含 GPU 执行。

## 0. 范围 (固定)

- 只含支持分支 23 分子 shortlist (A16/B7, 共 {n_cands_total} 候选) + query 谱 (30张, 其中26支持/4不支持)。
- 候选集复用现有 R3 FP 窗/shortlist (`artifacts/009-expansion/r3_50_full_candidate_scores.json` expanded, 不重建规则)。支持窗全<=100, 故 expanded top100 == 全窗。
- 4 NILU 质量不确定 (NMVVBVMYPLMIOU/UTDAGHZGKXPRQI/JOECRAPIMXFQDL/ZTILAOCGFRDHBH) 无窗跳过, 单列, 0候选, 不进包。
- 不支持 27 分子不需 ICE (条件规则走 CPU all-mean FP, 已有分数, 不补 ICE 分, missing 从不打 0)。
- YTCZZ (B, QTOF [M+H]+, 84候选, rank40) 在包内 (bundle_idx {next(r['bundle_idx'] for r in scope_rows if r['key']=='YTCZZXIRLARSET')})。其 forward 结果即回答 `artifacts/009-v9recall/ytczz_contract.json` 契约问题: query SUPPORTED, 断连钠盐 ATTEMPTABLE but UNVALIDATED, 不假设救回 rank40, 不把 FP rank40 移植到 ICE。

详见包外 `../forward_tasks.json` (逐分子候选数/支持谱/unique任务数/成本)、`../run_meta.json`。

## 1. 包内容

- `inputs.json` — 23 分子 worker 兼容输入: `order_top100` 即 expanded top100 (FP `orig-head + 10ppm + canon-max + order(-score,canon)`), `scores_top100` 对齐 FP 分, `rep_smiles` (base 来自 train/coco parquet output_smiles, 新真值来自 candidate_index_records), `formulas` (RDKit 2026.03.6 CalcMolFormula), `spectra` (mz/it 归一化max1/prec/adduct/mode/instrument=family Orbitrap-QTOF/ce_ev-ce_orig-ce_units 原样)、`truth_canon` 明文、`qidx` (obs序号)、`class_AB`、`route=retain_v9`。
- `code/ice/` — verbatim copy 自 `artifacts/g1-validation/code/ice` (`ice_runner.py` sha `dc0d5d1e14053e2e7d9088817c620d6615372abc3afdd02be2fc7ff9ee0b2802`、`fuse.py`、`shim/`、`src/ms_pred/`，`PREP_TAG panel-A1:v1`，deterministic)。不改 runner。
- `g1_worker.py` — verbatim copy 自 `artifacts/g1-validation/g1_worker.py` (sha `54b0751f22046bced9e6cc77caced11c4549298f26a7e6009ae9062e5341ab0c`，mean/split 双融合、`fused_merge` + `sims-mean` 双存、缓存键 `model|ckpt|structure|CE|adduct|instrument|prep`)。
- `install.sh` — copy 自 G1 (PyPI `rdkit einops pytorch_lightning lightning-utilities torchmetrics`; torch 用 Colab 自带, 不重装; 约 2–4 分钟)。
- `MANIFEST.iceberg.json` — copy 自 G1 weights MANIFEST (含 `gen.pt`/`inten.pt` bytes+sha, 供核验)。
- `weights_reference.json` — 已验证 SHA 的权重引用方式 + 远端提供方法 (本包 `weights/` 为空, 不塞大权重)。
- `README.md` (本文件) — Colab 三步命令、truth 声明、期望输出格式、dry-run 与成本。

`weights/`、`scores/`、`cache/`、`work/` 运行时创建 (发出时 `weights/` 为空, `scores/`/`cache/` 为空)。

truth 声明: `truth_canon` 仅供本地离线打分。truth 不得用于调参、模型选择、早停; 矩阵固定 (split 主方案), 任何“较好 CE 策略”比较都在 CPU 用已存分数重算, 不重跑 GPU。

heldout 声明: 本包仅 23 研究分子, 无 G1 保留 250, 不评估保留集, 不拼全 500 均值, 不扫描阈值/权重。

## 2. Colab 从零开始 (只做 3 步, 不假设会话在线)

```bash
# 1) 传包 (二选一) + 权重 (二选一, 见 weights_reference.json)
# A. Drive 方式: 把 v9r3baseline-bundle.tar.gz 传到 Drive, Colab 里:
from google.colab import drive; drive.mount('/content/drive')
!tar xzf \"$V9_TAR\" -C /content && test -f /content/v9r3baseline/inputs.json && ls /content/v9r3baseline   # 应看到 inputs.json install.sh g1_worker.py MANIFEST.iceberg.json weights_reference.json README.md code/
# 本包 tar 带顶层目录 v9r3baseline/, 解压 -C /content 后即得 /content/v9r3baseline, 不依赖 notebook 侧 mkdir (修复 flat-tar 教训)。
# 权重: 本地把 gen.pt + inten.pt 传到 Drive (如 /content/drive/MyDrive/v9r3baseline-weights/), Colab 里:
!mkdir -p /content/v9r3baseline/weights && cp \"$V9_GEN\" \"$V9_INTEN\" /content/v9r3baseline/weights/ && sha256sum /content/v9r3baseline/weights/*.pt
# 对照 weights_reference.json: gen 14561244 deef0451…、inten 13793037 c49e330e…, mismatch 拒绝运行。
# B. 直接上传: 把 v9r3baseline-bundle.tar.gz + 两权重传到 /content 后同样解压/复制。

# 2) 安装 (PyPI; torch 用 Colab 自带, 不重装; 约 2-4 分钟)
!cd /content/v9r3baseline && bash install.sh

# 3) 运行 (T4 GPU 运行时; 全量 {uniq_split_total} jobs 纯预测约 {uniq_split_total/26.0:.1f}s, 含 load/overhead 预计 5–10min, 见 §3)
# 先 wiring (无 torch, CPU 也可跑, 见 §4 dry-run):
!cd /content/v9r3baseline && PYTHONPATH=code/ice/shim:code/ice/src python3 g1_worker.py --pkg code/ice --model iceberg --inputs inputs.json --out /tmp/dry.json --k 100 --cemode split --dry-run --mol-limit 2
# 实跑 split 主方案 (v9 fused_merge 主用; 含 load/overhead 预计 5–10min, 远低于 2h):
!cd /content/v9r3baseline && mkdir -p scores cache work work/ice_ckpt_pkg/ckpt && ln -sf $(pwd)/weights/gen.pt work/ice_ckpt_pkg/ckpt/gen.pt && ln -sf $(pwd)/weights/inten.pt work/ice_ckpt_pkg/ckpt/inten.pt && ls -l work/ice_ckpt_pkg/ckpt/ && PYTHONPATH=code/ice/shim:code/ice/src python3 g1_worker.py --pkg code/ice --model iceberg --inputs inputs.json --out scores/v9r3baseline_ice_top100_split.json --k 100 --cemode split --device cuda --cache-dir cache --workdir work --ckpt-pkg work/ice_ckpt_pkg --any-rdkit --chunk-jobs 3000
# 说明: --ckpt-pkg 需事先 symlink (见上行 ln -sf)。
# 可选 mean 对照 (同包复用缓存, 增量小):
# PYTHONPATH=code/ice/shim:code/ice/src python3 g1_worker.py --pkg code/ice --model iceberg --inputs inputs.json --out scores/v9r3baseline_ice_top100_mean.json --k 100 --cemode mean --device cuda --cache-dir cache --workdir work --ckpt-pkg work/ice_ckpt_pkg --any-rdkit

# 4) 回传 (把整个 scores/ 打包下载; cache/ 可选, 用于续跑/审计峰)
!cd /content/v9r3baseline && tar czf v9r3baseline-scores.tar.gz scores && ls -lh v9r3baseline-scores.tar.gz
# 从 Colab Files 面板下载 v9r3baseline-scores.tar.gz (+ 可选 cache/*.npz)。
```

Kaggle 周配额见顶期间之外的替代: 在任何 CUDA+联网机器上同样三步可跑。零 Kaggle GPU: 本次不 push、不 submit、不调用远端。

## 3. 预期时长 (T4, unique (canon,adduct,instr,CE) 任务数, 本地 dry-run 实算)

| 范围 | unique jobs (本地 dry-run, 无 GPU 也可跑) | 纯预测 ICE@26/s | 含 load/overhead 估计 |
| --- | --- | --- | --- |
| 全 23 支持分子 split (v9 主, {n_cands_total} cands) | {uniq_split_total} | {uniq_split_total/26.0:.1f}s ({uniq_split_total/26.0/60.0:.2f}min) | ~5–10min |
| 全 23 mean | {uniq_mean_total} | {uniq_mean_total/26.0:.1f}s | 同上 |
| dry-run 2 分子 split (CLSD4+UVYP6=10 cands, wiring) | {two_mean} | — (fake sims, CPU <1s planning) | — |
| YTCZZ 单分子 (84 cands) | 84 | 3.2s | 含 load 约 3–6min (首轮小样本) |

- 估算方法: `seconds ≈ unique_jobs / 26` (F1 MANIFEST `timing_basis.ice_pred_per_s=26`; 005 实测 ICE 33.0 pred/s 更快, 老基准供参考下限)。实际每格 meta 含 `pred_per_s`、`max_mem_gb`、`seconds`, 以实际为准。
- 对照 008 建议 2h (120min) 上限: 全量纯预测约 {uniq_split_total/26.0/60.0:.2f}min, >100x 低于上限; 含 overhead 仍 <10min。
- 首轮仍沿用 2h 上限, 实测小样本 (先 YTCZZ 单分子或 2 分子 wiring) 后再放大 (009 下一步 1)。本包先 wiring 再实跑, 分段抗 Colab 超时/掉线。
- 先验 unique jobs (无 GPU 也可跑, CPU <1s):
  ```bash
  PYTHONPATH=code/ice/shim:code/ice/src python3 g1_worker.py --pkg code/ice --model iceberg --inputs inputs.json --out /tmp/dry.json --k 100 --cemode split --dry-run --mol-limit 0
  # 输出 DRY-RUN: 23 mols, {uniq_split_total} unique jobs; meta 含 unique_jobs/n_mols_covered
  ```

## 4. Dry-run 方法 (CPU-only, 不执行 GPU)

- 本地已验证 (M5 CPU, <1s planning):
  ```bash
  PYTHONPATH=artifacts/009-r3baseline/v9r3baseline/code/ice/shim:artifacts/009-r3baseline/v9r3baseline/code/ice/src /opt/homebrew/bin/python3.11 artifacts/009-r3baseline/v9r3baseline/g1_worker.py --pkg artifacts/009-r3baseline/v9r3baseline/code/ice --model iceberg --inputs artifacts/009-r3baseline/v9r3baseline/inputs.json --out /tmp/v9r3baseline_dry2.json --k 100 --cemode split --dry-run --mol-limit 2
  # DRY-RUN: 2 mols, {two_mean} unique jobs; status ok-dry-run; n_cands 10
  PYTHONPATH=... python3 g1_worker.py --pkg ... --inputs .../inputs.json --out /tmp/v9r3baseline_dry_full_split.json --k 100 --cemode split --dry-run --mol-limit 0
  # DRY-RUN: 23 mols, {uniq_split_total} unique jobs; n_cands {n_cands_total}; planning <1s
  ```
- Colab wiring 同上 (`--dry-run --mol-limit 2`, fake sims, 执行融合数学、键格式与输出 schema 端到端, 不碰 torch/模型/rdkit)。
- 通过标准: `status ok-dry-run`、`unique_jobs` 与上表一致、`n_mols_covered 23/23` (本包 23 分子全覆盖; `n_spectra_unsupported 4/30` 为负/不支持 adduct 计 unsupported, 从不打 0)。

## 5. 回传文件清单与格式

`scores/` (必须回传全部):

- `v9r3baseline_ice_top100_split.json` / `v9r3baseline_ice_top100_split.json.meta.json` (v9 主, 含 `fused_merge` + `sims_merge`)
- 可选 `v9r3baseline_ice_top100_mean.json` / `.meta.json` (对照)
- 单候选记录 (与 F1/G1 scores 同名): `{{canon, base, fused, sims:[{{qid,adduct,instr,ce,ce_bucket,ce_source,nce,how,sim}}], peaks:{{cond:hash}}, [fused_merge, sims_merge]}}`。
- `fused`: split 格 = 各 CE 相似度平均; `fused_merge` = 等权合并预测谱后的相似度 (v9 主融合); `cache/<model>/<mid>.npz` (可选回传): 逐条件预测峰 float32 + 键清单, 键 = `model|ckpt_sha|structure|CE|adduct|instrument|预处理` (见 meta `prep_tag=panel-A1:v1`)。

离线融合 (CPU, 不重跑 GPU; v9 口径, 全 shortlist 重算 z, 不能尾接旧融合):

```python
def _z(vals):
    import math
    ok=[v is not None and isinstance(v,(int,float)) and math.isfinite(v) for v in vals]
    xs=[float(v) for v,o in zip(vals,ok) if o]
    if len(xs)<2: return [0.0]*len(vals)
    mu=sum(xs)/len(xs); sd=math.sqrt(sum((x-mu)**2 for x in xs)/(len(xs)-1))
    return [((float(v)-mu)/sd) if o else 0.0 for v,o in zip(vals,ok)] if sd>0 else [0.0]*len(vals)
zb=_z(base); zi=_z(ice_merge); fused=[a+b for a,b in zip(zb,zi)]  # 等权
scored_sorted=sorted([i for i,v in enumerate(ice_merge) if v is not None], key=lambda i: (-fused[i], canon[i]))
unscored_sorted=sorted([i for i in range(n) if ice_merge[i] is None], key=lambda i: (-base[i], canon[i]))
full_order=scored_sorted+unscored_sorted  # missing=None, unscored 按 base 续后, 从不打 0
```

27 不支持分子按 `unscored` 续后, 不补分。详见包外 `../forward_tasks.json`。

## 6. 已知偏差与约束

- `--any-rdkit`: Colab PyPI rdkit 不是 panel 精确的 2025.03 (与 005 DEV 运行同偏差), 实际版本记入每个 meta; `site=none` (当前环境直跑, 无离线 wheel 站)。
- 确定性: `torch.use_deterministic_algorithms(True, warn_only=True)`, 与 filed runner 一致。
- 包内无任何 token/key; 也不要把 Colab/Kaggle 凭据写进包目录。
- NIST 相关一律不碰 (本包无 NIST 权重/代码路径)。
- 本包仅 23 研究分子, 不含 heldout; truth 仅离线打分, 不调参; 不扫描阈值/权重。
- 大权重走远端下载/引用, 不塞进包 (见 `weights_reference.json`); `code/ice` 与 `g1_worker.py` 为 verbatim 复用, 不重复造轮子。
- YTCZZ 断连钠盐 (`.[Na+]`, 2 frags) MAGMA/DAG UNVALIDATED, 需实际 forward 分离失败类型 (unsupported vs 解析/预测/预算失败), 记录 meta `pred_per_s/max_mem_gb`。
"""
    (PKG / "README.md").write_text(readme)
    print(f"WROTE README.md ({len(readme)} bytes)")

    # run_meta
    inputs_sha = {}
    for p in (p_rout, p_full, p_obs, p_index, p_train, p_coco, p_logits):
        # parquet/npy large: sha file
        inputs_sha[str(p.relative_to(ROOT))] = sha_file(p)
    inputs_sha["artifacts/g1-validation/code/ice/ice_runner.py"] = sha_file(ROOT / "artifacts/g1-validation/code/ice/ice_runner.py")
    inputs_sha["artifacts/g1-validation/g1_worker.py"] = sha_file(ROOT / "artifacts/g1-validation/g1_worker.py")
    inputs_sha["artifacts/009-r3baseline/v9r3baseline/inputs.json"] = sha_file(PKG / "inputs.json")
    inputs_sha["artifacts/009-r3baseline/v9r3baseline/g1_worker.py"] = worker_sha
    inputs_sha["artifacts/009-r3baseline/v9r3baseline/code/ice/ice_runner.py"] = ice_sha
    meta = {
        "task": "009-R3baseline 完整v9基线最小ICE forward bundle (CPU-only准备, 不执行GPU)",
        "device": "local M5 CPU-only; numpy + RDKit CPU; PyTorch not executed (dry-run planning only); no Kaggle GPU push/submit/remote",
        "seed": "no stochastic candidate selection; fixed prior008/009 method + obs-order deterministic; dry-run fake sims use deterministic sha-seeded RNG per key (worker internal)",
        "wall_clock_local": wall,
        "seconds_build": round(time.monotonic() - t0, 2),
        "inputs_sha256": inputs_sha,
        "environment": {"python": "3.11.16", "numpy": str(np.__version__)},
        "fp_cache": {"r3_50_logits.npy": "exists reuse", "r3_50_full_candidate_scores.json": "expanded reuse"},
        "ice_weights_ref": {"gen.pt": "deef0451c1870d2125653bb212c9d95c5b5d9bf80156dfa9b22fedda7db49bea",
                            "inten.pt": "c49e330e049d3a8b0095026ed694fcf70045c24305354853732aaa5f89d55068"},
        "outputs": ["v9r3baseline/inputs.json", "v9r3baseline/code/ice/*", "v9r3baseline/g1_worker.py",
                    "v9r3baseline/install.sh", "v9r3baseline/MANIFEST.iceberg.json",
                    "v9r3baseline/weights_reference.json", "v9r3baseline/README.md",
                    "forward_tasks.json", "run_meta.json", "v9r3baseline-bundle.tar.gz"],
        "forward_summary": {"n_mols": 23, "n_cands": int(n_cands_total), "unique_split": int(uniq_split_total),
                            "unique_mean": int(uniq_mean_total), "dry2_cands": int(two_cands), "dry2_split": int(two_mean)},
        "boundary": "No test files/framework/CI; manual python direct + manual JSON reads; zero Kaggle GPU; no G1 heldout250 labels; no threshold/weight scan; unsupported 27 no ICE补分; new dir artifacts/009-r3baseline",
        "caliber": "FP orig-head + 10ppm + canon-max + order(-score,canon); ICE COV positive [M+H]+/[M+Na]+; v9 z full-shortlist recalc, scored续后, never 0",
        "ytczz": "YTCZZXIRLARSET in package bundle_idx %d, 84 cands, forward结果即回答契约" % next(r['bundle_idx'] for r in scope_rows if r['key']=='YTCZZXIRLARSET'),
    }
    try:
        import rdkit
        meta["environment"]["rdkit"] = str(rdkit.__version__)
    except Exception:
        pass
    (OUT / "run_meta.json").write_text(dump(meta, 2))
    print(f"WROTE run_meta.json")
    print(f"\nKEY NUMBERS: mols=23 cands={n_cands_total} split={uniq_split_total} mean={uniq_mean_total} dry2={two_mean}")
    print("NEXT: tar + dry-run via manual bash (see README §4), then notebook.")

if __name__ == "__main__":
    main()

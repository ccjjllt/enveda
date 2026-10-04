"""E4 MIST-CF 公式重排：20 分子可复现推理入口（研究脚本，非测试）。

背景与口径（接手必读）：
- E4 已实跑：MIST-CF 双权重 md5 已验（mist bff0db38…/fast 505eced4…，在
  artifacts/005/）；20 分子 formula top-k：质量基线 Top1 9/20，MIST-CF 重排
  Top1 15/20。本脚本把逐分子表与推理代码落盘，使其可复现。
- SIRIUS 缺席声明：本机无 SIRIUS/decomp（官方 decomp 需 SIRIUS 5.5.7 二进制与
  SIRIUS_PATH），因此不用 SIRIUS 枚举质量对应公式，改用 E0 窗内公式枚举代替
  decomp：对每分子取 E0 train|single|10ppm nocap 窗内结构去重后的 molecular_formula
  集合作为公式候选。输出 meta 与本 docstring 均声明此替代。
- 候选偏乐观局限：E0 窗内枚举保证真值在候选内（本 20 分子 train 窗召回 20/20，
  与 E0 全量 train|single|10ppm 250/250 一致），而真实 SIRIUS decomp 可能漏真值；
  因此本 formula top-k 偏乐观，不能当作真实 de novo 公式召回，只能衡量“窗内公式
  已给定时 MIST-CF 的重排能力”。此局限同时写入输出 JSON meta.limitations。
- MIST-CF 官方实现仅正模式（ION_LST 7 种正加合；still only positive mode）。
  本 20 分子均为正模式 single-selected 谱（[M+H]+/[M+NH4]+/[M+Na]+/[M-H2O+H]+），
  经 ion_remap 标准化后全部受支持；负谱在本脚本中记 unsupported（本 20 不含负谱，
  但代码保留此分支以防误用）。
- timsTOF→Unknown override：Enveda 本批 1,184 谱 instrument_type 全为 timsTOF，
  MIST-CF instrument_to_type 无 timsTOF 映射（仅 orbitrap/qtof/iontrap/fticr/
  unknown）。本脚本统一 override 为 "Unknown (LCMS)"→unknown，tolerance 取
  unknown 档（15ppm 仅用于记录，不参与 MIST 前向），逐分子记录
  instrument_original/instrument_mapped/instrument_override=true。
- 双 ckpt：artifacts/005/mist_cf_best.ckpt（MistNet hidden128/layers2/abs-sines，
  ion+instrument，max_subpeak20）与 fast_filter_best.ckpt（FastFFN hidden256/
  layers3/abs-sines），经 torch.load(map_location=cpu) + PL 2.6
  load_from_checkpoint 加载，CPU 推理，确定性单线程（torch.manual_seed，
  DataLoader shuffle=False，batch 内按公式名字典序决胜平局）。
- 上游 MIST-CF 代码来源：https://github.com/samgoldman97/mist-cf，MIT，
  pin commit 71c820b43825e0b0a58680a3690f29a7352290f4（2023-09-20）。
  本脚本运行时 clone（或复用已 clone）该 commit 的 src/mist_cf 原码做
  subformula 指派（internal subformula assignment，非 SIRIUS 树）与模型前向，
  不重写 transformer，保证与 ckpt 权重严格兼容。commit 与权重 md5 同时写入 meta。

用法（本地 M5/CPU，已在 torch 2.14.1 + PL 2.6.6 + RDKit 2026.03.6 下实跑）：
  /opt/homebrew/bin/python3.11 scripts/mist_cf_formula.py \\
    --output artifacts/005/mist_cf_formula_topk_20.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


MIST_CF_REPO_URL = "https://github.com/samgoldman97/mist-cf"
MIST_CF_COMMIT = "71c820b43825e0b0a58680a3690f29a7352290f4"
EXPECTED_MD5 = {
    "mist_cf_best.ckpt": "bff0db38c4ae1e68bd93bf3e3ec46094",
    "fast_filter_best.ckpt": "505eced478107e5c182b38e964a4add3",
}
MIST_ADDUCT_NOTE = "still only positive mode; 7 MIST ions; Enveda negatives unsupported"
INSTR_OVERRIDE_FROM = "timsTOF"
INSTR_OVERRIDE_TO = "Unknown (LCMS)"
MASS_DIFF_THRESH_PPM = 15
MAX_SUBPEAK = 20
ASSIGN_TOP_PEAKS = 200


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_mist_cf_src(dest: Path, repo_url: str, commit: str) -> Path:
    dest = Path(dest)
    if dest.exists():
        try:
            cur = subprocess.run(
                ["git", "-C", str(dest), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=30,
            ).stdout.strip()
            if cur != commit:
                subprocess.run(
                    ["git", "-C", str(dest), "fetch", "--depth", "1",
                     "origin", commit],
                    check=False, timeout=120,
                )
                subprocess.run(
                    ["git", "-C", str(dest), "checkout", commit],
                    check=True, timeout=60,
                )
        except Exception as e:
            raise RuntimeError(f"mist-cf repo at {dest} not at {commit}: {e}")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", repo_url, str(dest)], check=True, timeout=180,
        )
        subprocess.run(
            ["git", "-C", str(dest), "checkout", commit], check=True, timeout=60,
        )
    src = dest / "src"
    if not (src / "mist_cf").exists():
        raise RuntimeError(f"mist-cf src not found under {src}")
    return src


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=Path,
                    default=Path("artifacts/004-research/np_queries_1184.json"))
    ap.add_argument("--windows", type=Path,
                    default=Path("artifacts/005/e0_cq_windows.npz"))
    ap.add_argument("--train-parquet", type=Path, default=Path("data/raw/train.parquet"))
    ap.add_argument("--train-keys", type=Path,
                    default=Path("datasets/train-fp-pool/keys.npy"))
    ap.add_argument("--coco-meta", type=Path,
                    default=Path("artifacts/005/coco/coco_meta.pkl"))
    ap.add_argument("--mist-ckpt", type=Path,
                    default=Path("artifacts/005/mist_cf_best.ckpt"))
    ap.add_argument("--fast-ckpt", type=Path,
                    default=Path("artifacts/005/fast_filter_best.ckpt"))
    ap.add_argument("--mist-cf-dir", type=Path,
                    default=Path("/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/mist-cf-e4"))
    ap.add_argument("--window-prefix", type=str, default="train|single|10.0")
    ap.add_argument("--n-molecules", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--output", type=Path,
                    default=Path("artifacts/005/mist_cf_formula_topk_20.json"))
    ap.add_argument("--tmp-subform-dir", type=Path, default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        torch.set_num_threads(1)
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[1]
    def res(p: Path) -> Path:
        return p if p.is_absolute() else (repo_root / p)

    queries_p, windows_p = res(args.queries), res(args.windows)
    train_pq, train_keys_p = res(args.train_parquet), res(args.train_keys)
    coco_p = res(args.coco_meta)
    mist_p, fast_p = res(args.mist_ckpt), res(args.fast_ckpt)
    out_p = res(args.output)

    for p in [queries_p, windows_p, train_pq, train_keys_p, coco_p, mist_p, fast_p]:
        if not p.exists():
            raise FileNotFoundError(f"missing input {p}")

    mist_md5, fast_md5 = md5_of(mist_p), md5_of(fast_p)
    md5_ok = (mist_md5 == EXPECTED_MD5["mist_cf_best.ckpt"]
              and fast_md5 == EXPECTED_MD5["fast_filter_best.ckpt"])

    src = ensure_mist_cf_src(Path(args.mist_cf_dir), MIST_CF_REPO_URL, MIST_CF_COMMIT)
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import torch
    import pytorch_lightning as pl
    # PL2.6 CPU shim for 2023-era mist-cf (PL1.x logger API); inference only,
    # ConsoleLogger never instantiated. Stub must exist before mist_cf import.
    try:
        import pytorch_lightning.loggers as _pl_loggers
        if not hasattr(_pl_loggers, "LightningLoggerBase"):
            from pytorch_lightning.loggers.logger import Logger as _Logger
            _pl_loggers.LightningLoggerBase = _Logger
        import sys as _sys
        import types as _types
        _base = _sys.modules.get("pytorch_lightning.loggers.base")
        if _base is None:
            _base = _types.ModuleType("pytorch_lightning.loggers.base")
            def _ranks_zero_experiment(fn=None, **kwargs):
                if fn is None:
                    return lambda f: f
                return fn
            _base.rank_zero_experiment = _ranks_zero_experiment
            _sys.modules["pytorch_lightning.loggers.base"] = _base
        # stub ray (tune_utils only; training callback never used in inference)
        if "ray" not in _sys.modules:
            _ray = _types.ModuleType("ray")
            _tune = _types.ModuleType("ray.tune")
            _integ = _types.ModuleType("ray.tune.integration")
            _pl_integ = _types.ModuleType("ray.tune.integration.pytorch_lightning")
            class _DummyCb:
                def __init__(self, *a, **k):
                    pass
            _pl_integ.TuneCallback = _DummyCb
            _tune.report = lambda *a, **k: None
            _ray.tune = _tune
            _sys.modules["ray"] = _ray
            _sys.modules["ray.tune"] = _tune
            _sys.modules["ray.tune.integration"] = _integ
            _sys.modules["ray.tune.integration.pytorch_lightning"] = _pl_integ
        # stub plot_utils (needs matplotlib/seaborn; unused in inference)
        for _stub_name in ["mist_cf.common.tune_utils", "mist_cf.common.plot_utils"]:
            if _stub_name not in _sys.modules:
                _m = _types.ModuleType(_stub_name)
                _m.__all__ = []
                _sys.modules[_stub_name] = _m
    except Exception:
        pass
    import mist_cf.common as common
    from mist_cf.mist_cf_score import mist_cf_data, mist_cf_model
    from mist_cf.fast_form_score import fast_form_data, fast_form_model
    from torch.utils.data import DataLoader

    try:
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula
        from rdkit import Chem
        import rdkit
        rdkit_ver = rdkit.__version__
    except Exception:
        CalcMolFormula, Chem, rdkit_ver = None, None, "missing"
    import pyarrow.parquet as pq

    # ---- queries: single-selected per molecule, positive-only, sorted, first N ----
    queries = json.loads(queries_p.read_text())
    from collections import defaultdict
    grouped: dict[str, list] = defaultdict(list)
    for idx, q in enumerate(queries):
        grouped[q["inchikey14"]].append((idx, q))
    single = []
    for key, lst in grouped.items():
        best_idx, best_q = max(lst, key=lambda t: len(t[1]["ms2_mzs"]))
        single.append((key, best_q))
    single_sorted = sorted(single, key=lambda t: t[0])
    positives = [(k, q) for k, q in single_sorted if q.get("ionization_mode") == "positive"]
    selected = positives[: args.n_molecules]
    if len(selected) < args.n_molecules:
        raise RuntimeError(f"only {len(selected)} positives, need {args.n_molecules}")
    sel_keys = [k for k, _ in selected]

    # ---- windows ----
    windows = np.load(str(windows_p))
    train_keys = np.load(str(train_keys_p))
    import pickle
    coco = pickle.loads(coco_p.read_bytes())
    coco_smiles = coco["smiles"]

    needed_train_keys: set[str] = set(sel_keys)
    for k in sel_keys:
        arr = windows[f"{args.window_prefix}|{k}"]
        for idx in arr.tolist():
            if int(idx) < len(train_keys):
                needed_train_keys.add(str(train_keys[int(idx)]))
    pf = pq.ParquetFile(str(train_pq))
    train_form: dict[str, str] = {}
    for batch in pf.iter_batches(columns=["inchikey14", "molecular_formula"],
                                 batch_size=65536):
        iks = batch.column(0).to_pylist()
        fms = batch.column(1).to_pylist()
        for k, f in zip(iks, fms):
            if k in needed_train_keys and k not in train_form:
                train_form[k] = str(f)
                if len(train_form) == len(needed_train_keys):
                    break
        if len(train_form) == len(needed_train_keys):
            break
    missing = [k for k in needed_train_keys if k not in train_form]
    if missing:
        raise RuntimeError(f"train formulas missing for {len(missing)} keys")

    coco_cache: dict[int, str] = {}
    def coco_formula(union_idx: int) -> str:
        if union_idx in coco_cache:
            return coco_cache[union_idx]
        s = str(coco_smiles[union_idx - len(train_keys)])
        m = Chem.MolFromSmiles(s) if Chem is not None else None
        f = CalcMolFormula(m) if m is not None else ""
        coco_cache[union_idx] = f
        return f

    # ---- models (PL2.6 CPU) ----
    torch_model_mist = mist_cf_model.MistNet.load_from_checkpoint(str(mist_p), map_location="cpu")
    torch_model_fast = fast_form_model.FastFFN.load_from_checkpoint(str(fast_p), map_location="cpu")
    torch_model_mist.eval()
    torch_model_fast.eval()
    device = torch.device("cpu")
    torch_model_mist = torch_model_mist.to(device)
    torch_model_fast = torch_model_fast.to(device)

    # instrument override check (timsTOF not in mapping)
    tims_mapped = common.instrument_to_type.get(INSTR_OVERRIDE_FROM, "uknown")

    import tempfile
    tmpdir = Path(args.tmp_subform_dir) if args.tmp_subform_dir else Path(tempfile.mkdtemp(prefix="e4_mistcf_subform_"))
    tmpdir.mkdir(parents=True, exist_ok=True)

    per_molecule = []
    n_mass_top1 = 0
    n_mist_top1 = 0
    for key, q in selected:
        precursor_mz = float(q["precursor_mz"])
        obs_adduct = str(q["adduct"])
        obs_instr = str(q.get("instrument_type", ""))
        # ion standardize via upstream remap
        try:
            std_ion = common.standardize_adduct(obs_adduct)
            ion_supported = std_ion in common.ION_LST
        except Exception:
            std_ion = obs_adduct
            ion_supported = False
        # instrument override
        if obs_instr == INSTR_OVERRIDE_FROM:
            mapped_instr = INSTR_OVERRIDE_TO
            instr_override = True
        else:
            mapped_instr = obs_instr
            instr_override = (mapped_instr != obs_instr)
        try:
            _idx = common.get_instr_idx(mapped_instr)
            instr_supported = True
        except Exception:
            instr_supported = False

        truth_raw = train_form[key]
        try:
            truth_std = common.standardize_form(truth_raw)
        except Exception:
            truth_std = truth_raw

        # E0 window enumeration (train pool indices; union prefix would need offset but default is train)
        wkey = f"{args.window_prefix}|{key}"
        arr = windows[wkey].tolist()
        cand_raw_set: set[str] = set()
        n_win_structs = len(arr)
        for idx in arr:
            idx = int(idx)
            if args.window_prefix.startswith("train|"):
                f = train_form.get(str(train_keys[idx]), "")
            else:
                if idx < len(train_keys):
                    f = train_form.get(str(train_keys[idx]), "")
                else:
                    f = coco_formula(idx)
            if f:
                cand_raw_set.add(f)
        cand_raw_set.add(truth_raw)
        # standardize + keep valid els only for MIST (mass baseline uses same set for fair compare)
        cand_std_list: list[str] = []
        raw_to_std: dict[str, str] = {}
        n_invalid = 0
        for f in cand_raw_set:
            try:
                if not common.has_valid_els(f):
                    n_invalid += 1
                    continue
                s = common.standardize_form(f)
                raw_to_std[f] = s
                cand_std_list.append(s)
            except Exception:
                n_invalid += 1
        # dedupe standardized (different raw strings may map to same std)
        std_to_raw: dict[str, str] = {}
        for raw, s in raw_to_std.items():
            std_to_raw.setdefault(s, raw)
        cand_std = sorted(std_to_raw.keys())
        n_cands = len(cand_std)
        # mass baseline (same theoretical as MIST cls term)
        def theo_mass(form_std: str) -> float:
            fm = common.formula_mass(form_std)
            ion_m = common.ion_to_mass[std_ion]
            v = fm + ion_m
            v = common.electron_correct(v)
            return v
        mass_errs = []
        for f in cand_std:
            try:
                t = theo_mass(f)
                err_ppm = abs(precursor_mz - t) / precursor_mz * 1e6
            except Exception:
                err_ppm = float("inf")
            mass_errs.append((err_ppm, f))
        mass_sorted = sorted(mass_errs, key=lambda t: (t[0], t[1]))
        mass_rank = next((i + 1 for i, (_, f) in enumerate(mass_sorted) if f == truth_std), None)
        mass_top1 = mass_sorted[0][1] if mass_sorted else None
        if mass_rank == 1:
            n_mass_top1 += 1

        # MIST scoring (skip if ion/instrument unsupported)
        mist_scores: dict[str, float] = {}
        fast_scores: dict[str, float] = {}
        mist_rank = None
        mist_top1 = None
        mist_error = None
        if ion_supported and instr_supported and n_cands > 0:
            try:
                # spec array: mz<=precursor+1, top ASSIGN_TOP_PEAKS by intensity
                mzs = np.asarray(q["ms2_mzs"], dtype=float)
                its = np.asarray(q["ms2_normalized_intensities"], dtype=float)
                keep = mzs <= (precursor_mz + 1.0)
                mzs, its = mzs[keep], its[keep]
                if len(mzs) == 0:
                    raise RuntimeError("no peaks after precursor filter")
                order = np.argsort(its)[::-1][:ASSIGN_TOP_PEAKS]
                # assignment input in intensity-desc order (documented Enveda adaptation;
                # upstream [:max_subpeak] then keeps most intense peaks)
                spec = np.vstack([mzs[order], its[order]]).T
                # per-candidate subformula JSONs
                spec_name = f"E4_{key}"
                jobj: dict = {}
                for f in cand_std:
                    out = common.assign_subforms(f, spec, std_ion,
                                                 mass_diff_thresh=MASS_DIFF_THRESH_PPM)
                    tbl = out.get("output_tbl")
                    jobj[f] = {"cand_ion": std_ion, "cand_tbl": tbl}
                (tmpdir / f"{spec_name}.json").write_text(json.dumps(jobj))
                import pandas as pd
                rows = [{"spec": spec_name, "cand_form": f, "cand_ion": std_ion,
                         "instrument": mapped_instr, "parentmass": precursor_mz}
                        for f in cand_std]
                df = pd.DataFrame(rows)
                ds = mist_cf_data.PredDataset(df, subform_dir=str(tmpdir),
                                              max_subpeak=torch_model_mist.max_subpeak,
                                              num_workers=0,
                                              ablate_cls_error=not torch_model_mist.cls_mass_diff)
                loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                    collate_fn=ds.get_collate_fn(), num_workers=0)
                all_scores: list[float] = []
                all_forms: list[str] = []
                with torch.no_grad():
                    for batch in loader:
                        num_peaks = batch["num_peaks"].to(device)
                        peak_types = batch["types"].to(device)
                        form_vec = batch["form_vec"].to(device)
                        ion_vec = batch["ion_vec"].to(device)
                        instrument_vec = batch["instrument_vec"].to(device)
                        intens = batch["intens"].to(device)
                        rel = batch["rel_mass_diffs"].to(device)
                        out = torch_model_mist.forward(num_peaks, peak_types, form_vec,
                                                       ion_vec, instrument_vec, intens, rel)
                        sc = out.squeeze(-1).detach().cpu().numpy()
                        sc = np.atleast_1d(sc)
                        all_scores.extend([float(x) for x in sc])
                        all_forms.extend(list(batch["str_forms"]))
                mist_scores = dict(zip(all_forms, all_scores))
                # fast filter scores (formula-only prior, for record)
                fdf = pd.DataFrame([{"spec": spec_name, "cand_form": f, "cand_ion": std_ion}
                                    for f in cand_std])
                fds = fast_form_data.PredDataset(fdf, num_workers=0)
                floader = DataLoader(fds, batch_size=args.batch_size, shuffle=False,
                                     collate_fn=fds.get_collate_fn(), num_workers=0)
                with torch.no_grad():
                    for b in floader:
                        x = b["x"].to(device).float()
                        o = torch_model_fast.forward(x)
                        sc = np.atleast_1d(o.detach().cpu().numpy())
                        for f, s in zip(list(b["str_forms"]), sc.tolist()):
                            fast_scores[f] = float(s)
                mist_sorted = sorted(mist_scores.items(), key=lambda t: (-t[1], t[0]))
                mist_rank = next((i + 1 for i, (f, _) in enumerate(mist_sorted) if f == truth_std), None)
                mist_top1 = mist_sorted[0][0] if mist_sorted else None
                if mist_rank == 1:
                    n_mist_top1 += 1
            except Exception as e:
                mist_error = f"{type(e).__name__}: {e}"

        per_molecule.append({
            "key": key,
            "truth_formula_raw": truth_raw,
            "truth_formula_std": truth_std,
            "precursor_mz": precursor_mz,
            "adduct_observed": obs_adduct,
            "adduct_standardized": std_ion,
            "ion_supported": bool(ion_supported),
            "instrument_original": obs_instr,
            "instrument_mapped": mapped_instr,
            "instrument_override": bool(instr_override),
            "instrument_supported": bool(instr_supported),
            "n_window_structures": int(n_win_structs),
            "n_candidates": int(n_cands),
            "n_invalid_formulas_skipped": int(n_invalid),
            "std_to_raw": dict(sorted(std_to_raw.items())),
            "mass_rank": mass_rank,
            "mass_top1": mass_top1,
            "mass_top1_raw": std_to_raw.get(mass_top1) if mass_top1 else None,
            "mass_sorted_top5": [[f, float(e)] for e, f in mass_sorted[:5]],
            "mist_rank": mist_rank,
            "mist_top1": mist_top1,
            "mist_top1_raw": std_to_raw.get(mist_top1) if mist_top1 else None,
            "mist_scores": {k: float(v) for k, v in sorted(mist_scores.items())},
            "fast_scores": {k: float(v) for k, v in sorted(fast_scores.items())},
            "mist_error": mist_error,
        })

    summary = {
        "n": len(per_molecule),
        "mass_top1": int(n_mass_top1),
        "mist_top1": int(n_mist_top1),
    }
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "seed": args.seed,
        "torch_version": getattr(__import__("torch"), "__version__", "unknown"),
        "pl_version": getattr(__import__("pytorch_lightning"), "__version__", "unknown"),
        "rdkit_version": rdkit_ver,
        "mist_cf_repo": MIST_CF_REPO_URL,
        "mist_cf_commit": MIST_CF_COMMIT,
        "checkpoints": {
            "mist_cf_best.ckpt": {"path": str(mist_p), "md5": mist_md5,
                                  "expected": EXPECTED_MD5["mist_cf_best.ckpt"],
                                  "match": mist_md5 == EXPECTED_MD5["mist_cf_best.ckpt"]},
            "fast_filter_best.ckpt": {"path": str(fast_p), "md5": fast_md5,
                                      "expected": EXPECTED_MD5["fast_filter_best.ckpt"],
                                      "match": fast_md5 == EXPECTED_MD5["fast_filter_best.ckpt"]},
            "md5_all_match": bool(md5_ok),
        },
        "sirius_present": False,
        "sirius_statement": ("SIRIUS/decomp absent in this environment; formula enumeration "
                             "uses E0 window distinct formulas instead of SIRIUS decomp."),
        "candidate_source": (f"E0 {args.window_prefix} nocap window distinct molecular_formula "
                             "(train first-occurrence; coco via RDKit only for union prefixes)"),
        "candidate_limitation_optimistic": ("Window enumeration guarantees truth in candidates "
            f"(E0 train recall 250/250 at 10ppm; this run 20/20); true SIRIUS decomp may miss "
            "truth, so this formula top-k is optimistic vs true de novo and only measures "
            "reranking given window formulas."),
        "ion_note": MIST_ADDUCT_NOTE,
        "instrument_override_rule": f"{INSTR_OVERRIDE_FROM}->{INSTR_OVERRIDE_TO} (unknown); recorded per molecule",
        "spec_filter": (f"mz<=precursor+1, top {ASSIGN_TOP_PEAKS} by intensity desc for subformula "
                        "assignment; MIST max_subpeak 20 keeps most intense (Enveda noisy-spectra "
                        "adaptation; upstream slices [:20] in spec order)"),
        "subformula_thresh_ppm": MASS_DIFF_THRESH_PPM,
        "selection_rule": ("NP250 single-selected (max raw peaks) positive-only, sorted by "
                           "inchikey14, first N; no truth/rank used for selection"),
        "limitations": [
            "SIRIUS/decomp absent; E0 window enumeration replaces decomp.",
            "Window enumeration is optimistic: truth always in candidates, unlike true decomp.",
            "MIST-CF positive-only; negatives would be unsupported (this 20 are all positive).",
            "timsTOF mapped to Unknown; instrument evidence weaker than orbitrap/qtof.",
            "Peak prefilter (top200 intense) and max_subpeak20 truncation are Enveda adaptations.",
            "Public NPLIB1 checkpoint pretraining exposure unknown; NP250 is a reused dev set.",
        ],
    }
    report = {"meta": meta, "summary": summary, "per_molecule": per_molecule}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_p}")


if __name__ == "__main__":
    main()

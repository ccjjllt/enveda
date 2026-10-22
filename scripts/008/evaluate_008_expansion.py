#!/opt/homebrew/bin/python3.11
"""008-B 评估：无标签召回 + CPU FP 排序，四阶段记录（手动研究脚本，非测试文件）。

口径（与 v9/007-R3 同口径对照）：
- 查询仅用谱字段：peaks/precursor/adduct/instrument/CE，经
  src.candidate_expansion.query_candidates_unlabeled（10ppm float64 全窗枚举，
  未知 adduct → mass_uncertain 单列，不计准确窗覆盖）。
  predicted_formula=None（本次无谱预测式，不过滤）；真值公式仅 oracle 诊断。
- class_AB / truth key / truth SMILES 仅用于分层描述与 oracle 诊断（覆盖/MRR），
  不决定候选、不分配预算、不参与窗枚举/FP 截断。
- 全库：base 712199（train 275810 + coco 436389）vs expanded 712738（+ external_new 539，
  external 8831 中去 base union 706242 后新增；overlap 8292 不重复计）。
- 质量窗：10ppm float64 全窗（|m-QM|<=|QM|*10e-6），返回全部合格（不预截断）；
  FP top100 为预先固定无标签截断（按 FP 分取前 100，窗<100 则全留）；
  最终 top25 为 FP 前 25。现有 R3 top100 为 mass-error 排序，不得冒充 FP shortlist。
- FP：fp_single_s2.pt orig-head CPU 前向，prep_peaks v32 verbatim；
  多谱分子 logits 均值（与 retrieval_scoring_10 一致）；collapse shipped->canon max，
  order(-score,canon)；RR@25（rank<=25?1/rank:0），G1 口径。
- 5 张问题谱：4 NILU 无 PRECURSOR_TYPE（None）+ 1 Bruker [M-H2O-H]-（本次扩展表支持后
  转已知）。NILU 4 单列质量不确定（不计准确窗/FP）；Bruker 纠正后计准确窗。
  另附 oracle 宽窗诊断？本次仅单列，不做猜测性纠正（UTDAG/JOECR 等无标注不猜 adduct）。

输入只读：observations_10/50、peaks/*.npz、pool-train/*、005/coco/*、005/maps、
008-expansion/candidate_index_*、weights fp_single_s2.pt、fp_bits.npy。
输出新增：artifacts/runs/008-expansion/expansion_eval_10.json、expansion_eval_50.json
（不覆盖 007-r3）。零 GPU、CPU-only，不 push/submit，不读保留集。

四阶段每分子记录（oracle 诊断标 oracle）：
- full_base_covered_oracle / full_expanded_covered_oracle（真值 canonical 在库内？）
- window_base_* / window_expanded_*（size_shipped/size_canon/covered_oracle）
- fp_top100_*（rank_oracle/covered/RR/MRR 分层）
- final_top25_*（同上）
- 挤下损失：A 类 base top25 命中但 expanded 非 top25（lost）、rank 恶化数；
  B 类新增召回（new recall）分 top100/top25/top1。
"""

import json
import os
import sys
import time
import hashlib
import pickle

from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)
from workspace_paths import workspace_path

import numpy as np

OUTDIR = os.path.join(REPO, "artifacts", 'runs', "008-expansion")
OBS10 = os.path.join(REPO, "artifacts", 'runs', "007-r3", "observations_10.json")
OBS50 = os.path.join(REPO, "artifacts", 'runs', "007-r3", "observations_50.json")
EXT_MASSES = os.path.join(OUTDIR, "candidate_index_masses.npy")
EXT_KEYS = os.path.join(OUTDIR, "candidate_index_keys.npy")
EXT_SMILES = os.path.join(OUTDIR, "candidate_index_smiles.txt")
EXT_RECORDS = os.path.join(OUTDIR, "candidate_index_records.jsonl")
CKPT = os.path.join(REPO, "artifacts", 'runs', "001-baseline", "weights", "fp_single_s2.pt")
FP_BITS = os.path.join(REPO, "artifacts", 'runs', "005", "coco", "fp_bits.npy")
OUT10 = os.path.join(OUTDIR, "expansion_eval_10.json")
OUT50 = os.path.join(OUTDIR, "expansion_eval_50.json")

for p in [OUT10, OUT50]:
    assert not os.path.exists(p), f"refuse overwrite {p}"


def sha_file(p):
    import hashlib as _hl

    h = _hl.sha256()
    sz = 0
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
            sz += len(ch)
    return sz, h.hexdigest()


def ce_mean_fallback(ce_ev, fallback=25.0):
    try:
        arr = np.asarray(ce_ev, dtype=np.float64).reshape(-1)
        if ce_ev is None or len(arr) == 0:
            raise ValueError
        if not np.isfinite(arr).all():
            raise ValueError
        return float(arr.mean()), False
    except Exception:
        return float(fallback), True


def mrr_top(ranks, k=25):
    n = len(ranks)
    rr = sum((1.0 / r if (r is not None and r <= k) else 0.0) for r in ranks)
    top1 = sum(1 for r in ranks if r == 1)
    topk = sum(1 for r in ranks if (r is not None and r <= k))
    return {"n": int(n), "mrr_at_%d" % k: float(rr / n) if n else 0.0,
            "top1_n": int(top1), "top1": float(top1 / n) if n else 0.0,
            "top%d_n" % k: int(topk), "top%d" % k: float(topk / n) if n else 0.0,
            "ranks": list(ranks)}


def main():
    import torch
    from src.models.fingerprint import FPNet, prep_peaks, ADDUCT_IX, instr_family
    from candidate_expansion import (
        query_candidates_unlabeled,
        oracle_diagnose_truth_in_list,
        rr_at_k,
    )

    t0 = time.time()
    # ---- base pool mmap（不重建）----
    t_mass = np.load(os.path.join(REPO, "artifacts", 'shared', "pool-train", "mass.npy"), mmap_mode="r")
    c_mass = np.load(os.path.join(REPO, "artifacts", 'runs', "005", "coco", "coco_mass.npy"), mmap_mode="r")
    t_fp = np.load(os.path.join(REPO, "artifacts", 'shared', "pool-train", "fp.npy"), mmap_mode="r")
    c_fp = np.load(os.path.join(REPO, "artifacts", 'runs', "005", "coco", "coco_fp.npy"), mmap_mode="r")
    t_keys = np.load(os.path.join(REPO, "artifacts", 'shared', "pool-train", "keys.npy"), allow_pickle=True)
    cmeta = pickle.load(open(os.path.join(REPO, "artifacts", 'runs', "005", "coco", "coco_meta.pkl"), "rb"))
    c_keys = np.asarray(cmeta["keys"])
    c_smiles = np.asarray(cmeta["smiles"])
    t_smiles = open(os.path.join(REPO, "artifacts", 'shared', "pool-train", "smiles.txt")).read().splitlines()
    NTR = 275810
    N_BASE = 712199
    assert len(t_keys) == NTR and len(c_keys) == 436389
    base_masses = np.concatenate([np.asarray(t_mass, dtype=np.float64), np.asarray(c_mass, dtype=np.float64)])
    assert len(base_masses) == N_BASE
    base_keys = list(t_keys.tolist()) + [str(k) for k in c_keys.tolist()]
    base_smiles = list(t_smiles) + [str(s) for s in c_smiles.tolist()]
    print(f"base pool {N_BASE} mmap ok", flush=True)

    import pyarrow.parquet as pq

    train_map = {}
    for batch in pq.ParquetFile(os.path.join(REPO, "artifacts", 'runs', "005", "structure_identity_train.parquet")).iter_batches(columns=["raw_key", "canonical_group_key"], batch_size=65536):
        for r, c in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            train_map[str(r)] = str(c)
    coco_map = {}
    for batch in pq.ParquetFile(os.path.join(REPO, "artifacts", 'runs', "005", "structure_identity_coco.parquet")).iter_batches(columns=["raw_key", "canonical_group_key"], batch_size=65536):
        for r, c in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            coco_map[str(r)] = str(c)
    union_canons = set(train_map.values()) | set(coco_map.values())
    print(f"maps union_canons {len(union_canons)}", flush=True)

    # ---- external index（只读质量/keys/smiles/records；指纹按需算）----
    ext_masses = np.load(EXT_MASSES)
    ext_keys = np.load(EXT_KEYS, allow_pickle=True)
    ext_smiles = open(EXT_SMILES).read().splitlines()
    assert len(ext_masses) == len(ext_keys) == len(ext_smiles) == 8831
    ext_keys = [str(k) for k in ext_keys.tolist()]
    # external_new：canonical 不在 base union（539）
    is_new = np.array([k not in union_canons for k in ext_keys], dtype=bool)
    new_idx = np.where(is_new)[0].tolist()
    print(f"external 8831 new {len(new_idx)} overlap {8831-len(new_idx)}", flush=True)
    new_masses = np.asarray([float(ext_masses[i]) for i in new_idx], dtype=np.float64)
    new_keys = [ext_keys[i] for i in new_idx]
    new_smiles = [ext_smiles[i] for i in new_idx]
    # new key-> (mass, smiles, ext_pos)
    new_lookup = {k: (float(m), str(s), int(pos)) for k, m, s, pos in zip(new_keys, new_masses, new_smiles, new_idx)}
    # 全库量
    N_EXPANDED = N_BASE + len(new_idx)
    print(f"full base {N_BASE} expanded {N_EXPANDED} (+{len(new_idx)})", flush=True)

    BITS = np.load(FP_BITS)
    assert BITS.shape == (6930,)
    ckpt_bytes, ckpt_sha = sha_file(CKPT)
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    net = FPNet(nbits=int(ck["nbits"]), d=int(ck["d"]), layers=int(ck["layers"]))
    net.load_state_dict(ck["model"], strict=True)
    net.eval()
    print(f"ckpt sha {ckpt_sha[:12]} nbits {ck['nbits']} d {ck['d']}", flush=True)

    # RDKit 指纹按需（window 新候选）
    from candidate_expansion import fp_bits_for_smiles

    fp_cache = {}

    def fp_of_external(smiles):
        if smiles in fp_cache:
            return fp_cache[smiles]
        sel, _, _ = fp_bits_for_smiles(str(smiles), BITS)
        arr = np.asarray(sel, dtype=np.float64)
        fp_cache[smiles] = arr
        return arr

    def evaluate_observations(obs_path, out_path, label):
        obs = json.load(open(obs_path))
        mols = obs["molecules"]
        print(f"== {label}: {len(mols)} molecules from {obs_path}", flush=True)
        # 谱列表（仅谱字段，不读 truth SMILES）
        spec_list = []
        for mi, m in enumerate(mols):
            for s in m["spectra"]:
                d = np.load(workspace_path(s["peak_file_ref"]))
                mz = np.asarray(d["mz"], dtype=np.float64)
                it = np.asarray(d["it"], dtype=np.float64)
                spec_list.append({
                    "mi": int(mi),
                    "mol_canon": str(m["canonical_key"]),
                    "class_AB": str(m.get("class_AB", "")),
                    "spectrum_id": str(s["spectrum_id"]),
                    "mz": mz, "it": it,
                    "prec": float(s["precursor_mz"]) if s.get("precursor_mz") is not None else None,
                    "adduct": s.get("adduct"),
                    "instr_raw": str(s.get("instrument_raw", "")),
                    "mode": str(s.get("ionization_mode", "")),
                    "ce_ev": s.get("collision_energy_ev"),
                })
        print(f"  spectra {len(spec_list)}", flush=True)
        # FP 前向（CPU）
        P = []
        ce_fb = 0
        for dd in spec_list:
            if dd["prec"] is None:
                raise RuntimeError(f"missing precursor {dd['spectrum_id']} (should not happen in R3 set)")
            a, b = prep_peaks(np.asarray(dd["mz"], float), np.asarray(dd["it"], float), float(dd["prec"]))
            P.append((a, b, dd))
            _, fb = ce_mean_fallback(dd["ce_ev"], 25.0)
            if fb:
                ce_fb += 1
        Np = 128
        B = len(P)
        mz = np.zeros((B, Np), np.float32)
        it = np.zeros((B, Np), np.float32)
        pad = np.ones((B, Np), bool)
        for j, (a, b, _) in enumerate(P):
            n = min(len(a), Np)
            if n > 0:
                mz[j, :n] = a[:n]
                it[j, :n] = b[:n]
                pad[j, :n] = False
        prec_arr = np.array([float(dd["prec"]) for _, _, dd in P], dtype=np.float32)
        ad_arr = np.array([ADDUCT_IX.get(dd["adduct"], ADDUCT_IX["<unk>"]) for _, _, dd in P])
        ins_arr = np.array([instr_family(dd["instr_raw"]) for _, _, dd in P])
        ce_vals = []
        for _, _, dd in P:
            v, _ = ce_mean_fallback(dd["ce_ev"], 25.0)
            ce_vals.append(v)
        ce_arr = np.array(ce_vals, dtype=np.float32)
        mode_arr = np.array([1.0 if dd["mode"] == "positive" else -1.0 for _, _, dd in P], dtype=np.float32)
        with torch.no_grad():
            logits = net(torch.as_tensor(mz), torch.as_tensor(it), torch.as_tensor(pad),
                         torch.as_tensor(prec_arr), torch.as_tensor(ad_arr), torch.as_tensor(ins_arr),
                         torch.as_tensor(ce_arr), torch.as_tensor(mode_arr)).cpu().numpy()
        assert logits.shape == (len(spec_list), 6930) and bool(np.isfinite(logits).all())
        print(f"  logits {logits.shape} ce_fallback {ce_fb}/{len(spec_list)}", flush=True)
        mol_q = {}
        for mi, m in enumerate(mols):
            idxs = [i for i, dd in enumerate(spec_list) if dd["mi"] == mi]
            if len(idxs) == 1:
                q = np.asarray(logits[idxs[0]], dtype=np.float64)
                how = "single"
            else:
                q = np.mean([np.asarray(logits[i], dtype=np.float64) for i in idxs], axis=0)
                how = f"mean_of_{len(idxs)}"
            mol_q[str(m["canonical_key"])] = (q, how)

        per = []
        for mi, m in enumerate(mols):
            canon_truth = str(m["canonical_key"])
            # 无标签查询：每谱独立 QM，再按分子聚合（median of known；若有不确定则分子单列）
            qinfos = []
            for s in m["spectra"]:
                qi = query_candidates_unlabeled(
                    peaks_mz=np.load(workspace_path(s["peak_file_ref"]))["mz"],
                    peaks_it=np.load(workspace_path(s["peak_file_ref"]))["it"],
                    precursor_mz=float(s["precursor_mz"]),
                    adduct=s.get("adduct"),
                    instrument=s.get("instrument_raw"),
                    ce=s.get("collision_energy_ev"),
                    index_masses=base_masses,  # 仅为 QM 计算占位，窗另算 base/expanded
                    index_keys=base_keys,
                    ppm=10.0,
                    predicted_formula=None,
                )
                qinfos.append({"spectrum_id": str(s["spectrum_id"]), "QM": qi["QM"],
                               "known": bool(qi["known_adduct"]), "uncertain": bool(qi["mass_uncertain"]),
                               "adduct": s.get("adduct"), "prec": float(s["precursor_mz"])})
            known_qms = [q["QM"] for q in qinfos if (not q["uncertain"] and q["QM"] is not None)]
            mass_uncertain = len(known_qms) != len(qinfos)
            if mass_uncertain:
                QM_med = None
            else:
                QM_med = float(np.median(np.asarray(known_qms, dtype=np.float64))) if known_qms else None
            # 全库 oracle（diagnostic only）
            full_base_cov = canon_truth in union_canons
            # expanded full = union + new keys
            full_exp_cov = bool(full_base_cov or (canon_truth in new_lookup))
            # 窗（base vs expanded；uncertain 则空窗）
            if mass_uncertain or QM_med is None:
                w_base_shipped, w_base_canon_list = [], []
                w_exp_shipped_n, w_exp_canon_list = 0, []
                win_base_cov, win_exp_cov = False, False
                n_win_base_shipped, n_win_base_canon = 0, 0
                n_win_exp_shipped, n_win_exp_canon = 0, 0
            else:
                qm = float(QM_med)
                wb = np.where(np.abs(base_masses - qm) <= abs(qm) * 10e-6)[0].astype(np.int64)
                # base collapse 到 canon（for window coverage oracle）
                base_win_canons = set()
                for j in wb.tolist():
                    jj = int(j)
                    sk = str(base_keys[jj])
                    c = train_map.get(sk) if jj < NTR else coco_map.get(sk)
                    if c is None:
                        c = sk
                    base_win_canons.add(str(c))
                win_base_cov = canon_truth in base_win_canons
                n_win_base_shipped = int(len(wb))
                n_win_base_canon = int(len(base_win_canons))
                # expanded：base + new（new masses 内窗）
                we_new = [k for k, (mm, _, _) in new_lookup.items() if abs(float(mm) - qm) <= abs(qm) * 10e-6]
                exp_canons = set(base_win_canons) | set(we_new)
                win_exp_cov = canon_truth in exp_canons
                n_win_exp_shipped = int(len(wb) + len(we_new))
                n_win_exp_canon = int(len(exp_canons))
                w_base_shipped = wb.tolist()
                w_exp_new_keys = list(we_new)
            # FP 排序（uncertain 跳过，rank None）
            q, qhow = mol_q[canon_truth]
            if mass_uncertain or QM_med is None:
                base_rank100 = None
                base_rank25 = None
                exp_rank100 = None
                exp_rank25 = None
                base_rr100 = 0.0
                base_rr25 = 0.0
                exp_rr100 = 0.0
                exp_rr25 = 0.0
                base_top1 = None
                exp_top1 = None
                n_fp_base = 0
                n_fp_exp = 0
                truth_base_score = None
                truth_exp_score = None
            else:
                qm = float(QM_med)
                # base window dots
                wb = np.where(np.abs(base_masses - qm) <= abs(qm) * 10e-6)[0].astype(np.int64)
                # 为 FP 排序，按窗枚举全部（不预截断），再按分排序取 top100/top25
                dots_base = {}
                # 批量 unpack？窗小（<100），逐个 unpack 可接受
                for j in wb.tolist():
                    jj = int(j)
                    pack = t_fp[jj] if jj < NTR else c_fp[jj - NTR]
                    bits = np.unpackbits(np.asarray(pack, dtype=np.uint8).reshape(1, -1), axis=1, bitorder="big")[0, :6930].astype(np.float64)
                    dots_base[jj] = float(bits @ q)
                # collapse shipped->canon max
                best_base = {}
                for jj, sc in dots_base.items():
                    sk = str(base_keys[int(jj)])
                    c = train_map.get(sk) if int(jj) < NTR else coco_map.get(sk)
                    if c is None:
                        c = sk
                    c = str(c)
                    if c not in best_base or sc > best_base[c]:
                        best_base[c] = float(sc)
                order_base = sorted(best_base.items(), key=lambda kv: (-kv[1], kv[0]))
                ranked_base = [k for k, _ in order_base]
                diag100_b = oracle_diagnose_truth_in_list(canon_truth, ranked_base[:100])
                diag25_b = oracle_diagnose_truth_in_list(canon_truth, ranked_base[:25])
                base_rank100 = diag100_b["rank_1based"]
                # rank in full FP order（for MRR@25 用 top25 截断 RR；top100 覆盖另计）
                # base_rank for RR@25 = rank in full order if <=25 else None?与 G1 一致：rank in full collapsed order，RR 按 <=25。
                full_diag_b = oracle_diagnose_truth_in_list(canon_truth, ranked_base)
                r_full_b = full_diag_b["rank_1based"]
                base_rank25 = r_full_b if (r_full_b is not None and r_full_b <= 25) else None
                # 若真值不在窗，rank None（miss）
                base_rr25 = rr_at_k(r_full_b, 25)
                base_top1 = ranked_base[0] if ranked_base else None
                n_fp_base = int(len(ranked_base))
                truth_base_score = best_base.get(canon_truth)

                # expanded：base dots + new dots（new 指纹按需）
                best_exp = dict(best_base)
                for k in w_exp_new_keys if 'w_exp_new_keys' in locals() else []:
                    mm, smi, _ = new_lookup[k]
                    arr = fp_of_external(smi)
                    sc = float(arr @ q)
                    if k not in best_exp or sc > best_exp[k]:
                        best_exp[k] = sc
                # 注意：若 overlap 新键已在 base（不应发生，因 new 与 union 不交），直接 max 即可。
                order_exp = sorted(best_exp.items(), key=lambda kv: (-kv[1], kv[0]))
                ranked_exp = [k for k, _ in order_exp]
                full_diag_e = oracle_diagnose_truth_in_list(canon_truth, ranked_exp)
                r_full_e = full_diag_e["rank_1based"]
                exp_rank25 = r_full_e if (r_full_e is not None and r_full_e <= 25) else None
                diag100_e = oracle_diagnose_truth_in_list(canon_truth, ranked_exp[:100])
                exp_rank100 = diag100_e["rank_1based"]
                exp_rr25 = rr_at_k(r_full_e, 25)
                exp_top1 = ranked_exp[0] if ranked_exp else None
                n_fp_exp = int(len(ranked_exp))
                truth_exp_score = best_exp.get(canon_truth)
                # top100 覆盖 oracle（rank<=100）
                base_cov100 = bool(base_rank100 is not None)
                exp_cov100 = bool(exp_rank100 is not None)
            if mass_uncertain or QM_med is None:
                base_cov100, exp_cov100 = False, False
                base_rank_full_for_mrr = None
                exp_rank_full_for_mrr = None
            else:
                base_rank_full_for_mrr = r_full_b
                exp_rank_full_for_mrr = r_full_e

            # 损失/新增（oracle 比较，仅报告）
            # base top25 hit?
            base_hit25 = bool(base_rank25 is not None) if not (mass_uncertain or QM_med is None) else False
            exp_hit25 = bool(exp_rank25 is not None) if not (mass_uncertain or QM_med is None) else False
            lost25 = bool(base_hit25 and (not exp_hit25))
            # rank 恶化（both ranked 且 exp 更大）
            worsened = False
            if (not (mass_uncertain or QM_med is None)) and (r_full_b is not None and r_full_e is not None):
                worsened = bool(r_full_e > r_full_b)
            # B 新增（base miss → exp hit）按 top100/top25/top1
            new100 = bool((r_full_b is None) and (r_full_e is not None and r_full_e <= 100)) if not (mass_uncertain or QM_med is None) else False
            new25 = bool((base_rank25 is None) and (exp_rank25 is not None)) if not (mass_uncertain or QM_med is None) else False
            new1 = bool(r_full_e == 1 and r_full_b != 1) if not (mass_uncertain or QM_med is None) else False

            per.append({
                "canonical_key": canon_truth,
                "class_AB": str(m.get("class_AB", "")),
                "n_spectra": int(m.get("n_spectra", len(m.get("spectra", [])))),
                "spectrum_ids": [str(s.get("spectrum_id", "")) for s in m.get("spectra", [])],
                "q_how": str(mol_q[canon_truth][1]),
                "QM_med": None if QM_med is None else float(QM_med),
                "mass_uncertain": bool(mass_uncertain),
                "per_spectrum_QM": qinfos,
                "full_base_covered_oracle": bool(full_base_cov),
                "full_expanded_covered_oracle": bool(full_exp_cov),
                "window_base_shipped": int(n_win_base_shipped) if not (mass_uncertain or QM_med is None) else 0,
                "window_base_canon": int(n_win_base_canon) if not (mass_uncertain or QM_med is None) else 0,
                "window_base_covered_oracle": bool(win_base_cov) if not (mass_uncertain or QM_med is None) else False,
                "window_expanded_shipped": int(n_win_exp_shipped) if not (mass_uncertain or QM_med is None) else 0,
                "window_expanded_canon": int(n_win_exp_canon) if not (mass_uncertain or QM_med is None) else 0,
                "window_expanded_covered_oracle": bool(win_exp_cov) if not (mass_uncertain or QM_med is None) else False,
                "n_fp_base": int(n_fp_base) if not (mass_uncertain or QM_med is None) else 0,
                "n_fp_expanded": int(n_fp_exp) if not (mass_uncertain or QM_med is None) else 0,
                "fp_top100_base_rank_oracle": base_rank100,
                "fp_top100_base_covered_oracle": bool(base_cov100) if not (mass_uncertain or QM_med is None) else False,
                "fp_top100_expanded_rank_oracle": exp_rank100,
                "fp_top100_expanded_covered_oracle": bool(exp_cov100) if not (mass_uncertain or QM_med is None) else False,
                "final_top25_base_rank_oracle": base_rank_full_for_mrr,
                "final_top25_base_RR": float(base_rr25) if not (mass_uncertain or QM_med is None) else 0.0,
                "final_top25_expanded_rank_oracle": exp_rank_full_for_mrr,
                "final_top25_expanded_RR": float(exp_rr25) if not (mass_uncertain or QM_med is None) else 0.0,
                "base_top1": base_top1,
                "expanded_top1": exp_top1,
                "truth_base_score": None if truth_base_score is None else float(truth_base_score),
                "truth_expanded_score": None if truth_exp_score is None else float(truth_exp_score),
                "lost_top25_oracle": bool(lost25),
                "rank_worsened_oracle": bool(worsened),
                "new_recall_top100_oracle": bool(new100),
                "new_recall_top25_oracle": bool(new25),
                "new_recall_top1_oracle": bool(new1),
            })
            print(f"  {canon_truth} {m.get('class_AB','')} QM {QM_med} unc {mass_uncertain} full {full_base_cov}->{full_exp_cov} win {n_win_base_shipped if not (mass_uncertain or QM_med is None) else 0}->{n_win_exp_shipped if not (mass_uncertain or QM_med is None) else 0} base_rank {base_rank_full_for_mrr if not (mass_uncertain or QM_med is None) else None} exp_rank {exp_rank_full_for_mrr if not (mass_uncertain or QM_med is None) else None}", flush=True)

        # 分层汇总（A/B 描述，不估隐藏比例；uncertain 单列）
        def strat(rows):
            ranks_b = [r["final_top25_base_rank_oracle"] for r in rows if not r["mass_uncertain"]]
            ranks_e = [r["final_top25_expanded_rank_oracle"] for r in rows if not r["mass_uncertain"]]
            cov_full_b = sum(1 for r in rows if r["full_base_covered_oracle"])
            cov_full_e = sum(1 for r in rows if r["full_expanded_covered_oracle"])
            cov_win_b = sum(1 for r in rows if (not r["mass_uncertain"] and r["window_base_covered_oracle"]))
            cov_win_e = sum(1 for r in rows if (not r["mass_uncertain"] and r["window_expanded_covered_oracle"]))
            cov100_b = sum(1 for r in rows if (not r["mass_uncertain"] and r["fp_top100_base_covered_oracle"]))
            cov100_e = sum(1 for r in rows if (not r["mass_uncertain"] and r["fp_top100_expanded_covered_oracle"]))
            # MRR 按非 uncertain 分母？报告两种：all（uncertain 按 0）与 known-only
            m_all_b = mrr_top([r["final_top25_base_rank_oracle"] for r in rows], 25)
            m_all_e = mrr_top([r["final_top25_expanded_rank_oracle"] for r in rows], 25)
            m_known_b = mrr_top(ranks_b, 25)
            m_known_e = mrr_top(ranks_e, 25)
            n_unc = sum(1 for r in rows if r["mass_uncertain"])
            return {
                "n": int(len(rows)), "n_uncertain": int(n_unc),
                "full_base_covered": int(cov_full_b), "full_expanded_covered": int(cov_full_e),
                "window_base_covered": int(cov_win_b), "window_expanded_covered": int(cov_win_e),
                "fp_top100_base_covered": int(cov100_b), "fp_top100_expanded_covered": int(cov100_e),
                "mrr_all_base": m_all_b, "mrr_all_expanded": m_all_e,
                "mrr_known_only_base": m_known_b, "mrr_known_only_expanded": m_known_e,
                "lost_top25": int(sum(1 for r in rows if r["lost_top25_oracle"])),
                "rank_worsened": int(sum(1 for r in rows if r["rank_worsened_oracle"])),
                "new_top100": int(sum(1 for r in rows if r["new_recall_top100_oracle"])),
                "new_top25": int(sum(1 for r in rows if r["new_recall_top25_oracle"])),
                "new_top1": int(sum(1 for r in rows if r["new_recall_top1_oracle"])),
                "mean_window_base_shipped": float(sum(r["window_base_shipped"] for r in rows) / len(rows)) if rows else 0.0,
                "mean_window_expanded_shipped": float(sum(r["window_expanded_shipped"] for r in rows) / len(rows)) if rows else 0.0,
            }

        all_s = strat(per)
        A_s = strat([r for r in per if r["class_AB"] == "A"])
        B_s = strat([r for r in per if r["class_AB"] == "B"])
        print(f"[{label}] all {all_s}", flush=True)
        print(f"[{label}] A {A_s}", flush=True)
        print(f"[{label}] B {B_s}", flush=True)

        out = {
            "meta": {
                "task": f"008-B expansion eval {label} (unlabeled recall + CPU FP, four stages)",
                "created_utc": "2026-10-12T00:00:00Z",
                "budget": "local CPU only; no training; no GPU; no Kaggle quota; no heldout spectra read/eval; writable only artifacts/runs/008-expansion/",
                "inputs": {
                    "observations": os.path.relpath(obs_path, REPO) + f" ({len(mols)} molecules, read-only; class_AB/truth only for stratified oracle reporting, never for retrieval/budget)",
                    "peaks": "artifacts/runs/007-r3/peaks/*.npz",
                    "pool_base": "artifacts/shared/pool-train/ + artifacts/runs/005/coco/ union 712199 mmap",
                    "external_index": "artifacts/runs/008-expansion/candidate_index_* (8831 distinct, new 539; masses float64 ExactMolWt; query via candidate_expansion only)",
                    "weights": {"path": os.path.relpath(CKPT, REPO), "bytes": int(ckpt_bytes), "sha256": str(ckpt_sha), "head": "orig (fp_single_s2)"},
                    "prep": "src.models.fingerprint.prep_peaks v32 verbatim",
                    "window": "10ppm float64 full window (all qualified, no truncation); FP top100 pre-fixed WMAX=100 by FP score (not mass-error); final top25 by FP score. R3 mass-ranked top100 is NOT FP shortlist.",
                    "collapse": "shipped->canon max (rdkit_full), order(-score,canon)",
                    "truth": "canonical_group_key oracle only (diagnostic branches), G1 caliber RR@25",
                    "multi": "single->single logits; multi->mean logits (same as retrieval_scoring_10)",
                    "unknown_adduct": "strict: None/empty/unknown -> mass_uncertain True, no accurate window; [M-H2O-H]- explicit supported (delta -19.01784115344); 4 NILU remain uncertain, 1 Bruker corrected",
                },
                "pool_counts": {"base": int(N_BASE), "external_new": int(len(new_idx)), "expanded": int(N_EXPANDED)},
                "device": "cpu",
                "ce_fallback_25eV": f"{ce_fb}/{len(spec_list)}",
                "forbidden_respected": ["no test files", "no training", "no GPU", "no Kaggle quota", "no heldout spectra read/eval", "no truth SMILES extraction", "no disallowed-license collection", "no network query"],
            },
            "per_molecule": per,
            "summary": {"all": all_s, "A": A_s, "B": B_s},
            "timing_s": float(time.time() - t0),
        }
        json.dump(out, open(out_path, "w"), indent=1, allow_nan=False)
        print(f"wrote {out_path} bytes {os.path.getsize(out_path)}", flush=True)
        return out

    r10 = evaluate_observations(OBS10, OUT10, "10")
    r50 = evaluate_observations(OBS50, OUT50, "50")
    print(f"done timing {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

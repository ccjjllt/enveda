#!/opt/homebrew/bin/python3.11
"""009-3 召回与干扰实验：LMSD固定单来源在R3-50（CPU-only，手动研究脚本，非测试文件）。

口径（008-B六条 + evaluate_008_expansion.py FP caliber复用，不新起GPU）：
- 查询仅用谱字段：peaks/precursor/adduct/instrument/CE，经
  src.candidate_expansion.query_candidates_unlabeled（10ppm float64全窗枚举，
  未知adduct → mass_uncertain单列，不计准确窗覆盖）。
  predicted_formula=None（本次无谱预测式，不过滤）；真值公式仅oracle诊断。
- class_AB / truth key / truth SMILES仅用于分层描述与oracle诊断（覆盖/MRR），
  不决定候选、不分配预算、不参与窗枚举/FP截断。
- 全库：base 712199（train 275810 + coco 436389）vs expanded 734964
  （+ LMSD单来源新增22765；LMSD全源去重43759中去base union 706242后新增；
  overlap 20994不重复计）。
- 质量窗：10ppm float64全窗，返回全部合格（不预截断）；FP top100为预先固定
  无标签截断（按FP分取前100，窗<100则全留）；最终top25为FP前25。
  R3 mass-ranked top100不得冒充FP shortlist。
- FP：fp_single_s2.pt orig-head CPU前向，prep_peaks v32 verbatim；
  多谱分子logits均值（与retrieval_scoring_10一致）；collapse shipped->canon max，
  order(-score,canon)；RR@25（rank<=25?1/rank:0），G1口径。
- 4张NILU无PRECURSOR_TYPE（None）单列质量不确定（不计准确窗/FP）；
  1张Bruker [M-H2O-H]-扩展表支持后计准确窗。另UTDAG/JOECR等无标注不猜adduct。
- A/B只作分层描述，不估隐藏比例；本地B仅称当前712K库外；时间关系标unknown，
  不写“保证未见”。无参考谱分层：R3-50全部n_reference_spectra_in_L=0，
  故all=无参考谱总体，A=库内无参考谱，B=库外无参考谱（当前库外），另列B_known
  （B中质量已知21）。

输入只读：observations_50、peaks/*.npz、pool-train/*、005/coco/*、005/maps、
009-source/candidate_index_*、weights fp_single_s2.pt、fp_bits.npy。
输出新增：artifacts/009-source/recall_eval_50.json、run_meta.json
（不覆盖007-r3/008-expansion/pool-lmsd）。零GPU、CPU-only，不push/submit，不读保留集。
"""

import json
import os
import sys
import time
import hashlib
import pickle
from datetime import datetime, timezone

REPO = "/Users/zhaohuaxiang/ccjjllt/enveda"
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)

import numpy as np

OUTDIR = os.path.join(REPO, "artifacts", "009-source")
OBS50 = os.path.join(REPO, "artifacts", "007-r3", "observations_50.json")
SRC_MASSES = os.path.join(OUTDIR, "candidate_index_masses.npy")
SRC_KEYS = os.path.join(OUTDIR, "candidate_index_keys.npy")
SRC_SMILES = os.path.join(OUTDIR, "candidate_index_smiles.txt")
SRC_RECORDS = os.path.join(OUTDIR, "candidate_index_records.jsonl")
SRC_META = os.path.join(OUTDIR, "candidate_index_meta.json")
CKPT = os.path.join(REPO, "artifacts", "001-baseline", "weights", "fp_single_s2.pt")
FP_BITS = os.path.join(REPO, "artifacts", "005", "coco", "fp_bits.npy")
OUT50 = os.path.join(OUTDIR, "recall_eval_50.json")
OUTMETA = os.path.join(OUTDIR, "run_meta.json")

for p in [OUT50, OUTMETA]:
    assert not os.path.exists(p), f"refuse overwrite {p}"
for p in [OBS50, SRC_MASSES, SRC_KEYS, SRC_SMILES, SRC_RECORDS, SRC_META, CKPT, FP_BITS]:
    assert os.path.exists(p), f"missing input {p}"

SEED = 20261012


def sha_file(p):
    h = hashlib.sha256()
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
        fp_bits_for_smiles,
    )

    t0 = time.time()
    wall_start = datetime.now(timezone.utc).isoformat()
    torch.set_num_threads(4)

    # ---- base pool mmap（不重建）----
    t_mass = np.load(os.path.join(REPO, "artifacts", "pool-train", "mass.npy"), mmap_mode="r")
    c_mass = np.load(os.path.join(REPO, "artifacts", "005", "coco", "coco_mass.npy"), mmap_mode="r")
    t_fp = np.load(os.path.join(REPO, "artifacts", "pool-train", "fp.npy"), mmap_mode="r")
    c_fp = np.load(os.path.join(REPO, "artifacts", "005", "coco", "coco_fp.npy"), mmap_mode="r")
    t_keys = np.load(os.path.join(REPO, "artifacts", "pool-train", "keys.npy"), allow_pickle=True)
    cmeta = pickle.load(open(os.path.join(REPO, "artifacts", "005", "coco", "coco_meta.pkl"), "rb"))
    c_keys = np.asarray(cmeta["keys"])
    c_smiles = np.asarray(cmeta["smiles"])
    t_smiles = open(os.path.join(REPO, "artifacts", "pool-train", "smiles.txt")).read().splitlines()
    NTR = 275810
    N_BASE = 712199
    assert len(t_keys) == NTR and len(c_keys) == 436389
    base_masses = np.concatenate([np.asarray(t_mass, dtype=np.float64), np.asarray(c_mass, dtype=np.float64)])
    assert len(base_masses) == N_BASE
    base_keys = list(t_keys.tolist()) + [str(k) for k in c_keys.tolist()]
    print(f"base pool {N_BASE} mmap ok", flush=True)

    import pyarrow.parquet as pq
    train_map = {}
    for batch in pq.ParquetFile(os.path.join(REPO, "artifacts", "005", "structure_identity_train.parquet")).iter_batches(columns=["raw_key", "canonical_group_key"], batch_size=65536):
        for r, c in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            train_map[str(r)] = str(c)
    coco_map = {}
    for batch in pq.ParquetFile(os.path.join(REPO, "artifacts", "005", "structure_identity_coco.parquet")).iter_batches(columns=["raw_key", "canonical_group_key"], batch_size=65536):
        for r, c in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            coco_map[str(r)] = str(c)
    union_canons = set(train_map.values()) | set(coco_map.values())
    print(f"maps union_canons {len(union_canons)}", flush=True)

    # ---- LMSD single-source index（只读；指纹按需算，同008 caliber）----
    src_masses = np.load(SRC_MASSES)
    src_keys = np.load(SRC_KEYS, allow_pickle=True)
    src_smiles = open(SRC_SMILES).read().splitlines()
    assert len(src_masses) == len(src_keys) == len(src_smiles) == 22765
    assert str(src_masses.dtype) == "float64"
    src_keys = [str(k) for k in src_keys.tolist()]
    assert all(k not in union_canons for k in src_keys), "single-source index must be new-only vs base"
    new_lookup = {k: (float(m), str(s)) for k, m, s in zip(src_keys, src_masses, src_smiles)}
    N_EXPANDED = N_BASE + len(src_keys)
    print(f"single-source LMSD new {len(src_keys)} expanded {N_EXPANDED}", flush=True)

    BITS = np.load(FP_BITS)
    assert BITS.shape == (6930,)
    ckpt_bytes, ckpt_sha = sha_file(CKPT)
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    net = FPNet(nbits=int(ck["nbits"]), d=int(ck["d"]), layers=int(ck["layers"]))
    net.load_state_dict(ck["model"], strict=True)
    net.eval()
    print(f"ckpt sha {ckpt_sha[:12]} nbits {ck['nbits']} d {ck['d']}", flush=True)

    fp_cache = {}

    def fp_of_source(smiles):
        if smiles in fp_cache:
            return fp_cache[smiles]
        sel, _, _ = fp_bits_for_smiles(str(smiles), BITS)
        arr = np.asarray(sel, dtype=np.float64)
        fp_cache[smiles] = arr
        return arr

    obs = json.load(open(OBS50))
    mols = obs["molecules"]
    print(f"R3-50: {len(mols)} molecules", flush=True)
    spec_list = []
    for mi, m in enumerate(mols):
        for s in m["spectra"]:
            d = np.load(os.path.join(REPO, s["peak_file_ref"]))
            spec_list.append({
                "mi": int(mi), "mol_canon": str(m["canonical_key"]),
                "class_AB": str(m.get("class_AB", "")),
                "spectrum_id": str(s["spectrum_id"]),
                "mz": np.asarray(d["mz"], dtype=np.float64),
                "it": np.asarray(d["it"], dtype=np.float64),
                "prec": float(s["precursor_mz"]) if s.get("precursor_mz") is not None else None,
                "adduct": s.get("adduct"),
                "instr_raw": str(s.get("instrument_raw", "")),
                "mode": str(s.get("ionization_mode", "")),
                "ce_ev": s.get("collision_energy_ev"),
            })
    print(f"  spectra {len(spec_list)}", flush=True)
    P, ce_fb = [], 0
    for dd in spec_list:
        if dd["prec"] is None:
            raise RuntimeError(f"missing precursor {dd['spectrum_id']}")
        a, b = prep_peaks(np.asarray(dd["mz"], float), np.asarray(dd["it"], float), float(dd["prec"]))
        P.append((a, b, dd))
        _, fb = ce_mean_fallback(dd["ce_ev"], 25.0)
        if fb:
            ce_fb += 1
    Np, B = 128, len(P)
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
    ce_arr = np.array([ce_mean_fallback(dd["ce_ev"], 25.0)[0] for _, _, dd in P], dtype=np.float32)
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
        q = np.asarray(logits[idxs[0]], dtype=np.float64) if len(idxs) == 1 else \
            np.mean([np.asarray(logits[i], dtype=np.float64) for i in idxs], axis=0)
        mol_q[str(m["canonical_key"])] = (q, "single" if len(idxs) == 1 else f"mean_of_{len(idxs)}")

    per = []
    for mi, m in enumerate(mols):
        canon_truth = str(m["canonical_key"])
        qinfos = []
        for s in m["spectra"]:
            d = np.load(os.path.join(REPO, s["peak_file_ref"]))
            qi = query_candidates_unlabeled(
                peaks_mz=np.asarray(d["mz"]), peaks_it=np.asarray(d["it"]),
                precursor_mz=float(s["precursor_mz"]), adduct=s.get("adduct"),
                instrument=s.get("instrument_raw"), ce=s.get("collision_energy_ev"),
                index_masses=base_masses, index_keys=base_keys,
                ppm=10.0, predicted_formula=None)
            qinfos.append({"spectrum_id": str(s["spectrum_id"]), "QM": qi["QM"],
                           "known": bool(qi["known_adduct"]), "uncertain": bool(qi["mass_uncertain"]),
                           "adduct": s.get("adduct"), "prec": float(s["precursor_mz"])})
        known_qms = [q["QM"] for q in qinfos if (not q["uncertain"] and q["QM"] is not None)]
        mass_uncertain = len(known_qms) != len(qinfos)
        QM_med = None if mass_uncertain else (
            float(np.median(np.asarray(known_qms, dtype=np.float64))) if known_qms else None)
        full_base_cov = canon_truth in union_canons
        full_exp_cov = bool(full_base_cov or (canon_truth in new_lookup))
        if mass_uncertain or QM_med is None:
            n_wb_s, n_wb_c, n_we_s, n_we_c = 0, 0, 0, 0
            win_b_cov, win_e_cov = False, False
            base_rank100, exp_rank100, r_full_b, r_full_e = None, None, None, None
            base_rr25, exp_rr25, base_top1, exp_top1 = 0.0, 0.0, None, None
            n_fp_b, n_fp_e, ts_b, ts_e = 0, 0, None, None
            b_cov100, e_cov100 = False, False
            new_in_top100, new_in_top25 = [], []
            displaced_top100, displaced_top25 = [], []
        else:
            qm = float(QM_med)
            wb = np.where(np.abs(base_masses - qm) <= abs(qm) * 10e-6)[0].astype(np.int64)
            base_win_canons = set()
            for j in wb.tolist():
                jj = int(j)
                sk = str(base_keys[jj])
                c = train_map.get(sk) if jj < NTR else coco_map.get(sk)
                base_win_canons.add(str(c if c is not None else sk))
            win_b_cov = canon_truth in base_win_canons
            we_new = [k for k, (mm, _) in new_lookup.items()
                      if abs(float(mm) - qm) <= abs(qm) * 10e-6]
            exp_canons = set(base_win_canons) | set(we_new)
            win_e_cov = canon_truth in exp_canons
            n_wb_s, n_wb_c = int(len(wb)), int(len(base_win_canons))
            n_we_s, n_we_c = int(len(wb) + len(we_new)), int(len(exp_canons))
            q, _ = mol_q[canon_truth]
            dots_base = {}
            for j in wb.tolist():
                jj = int(j)
                pack = t_fp[jj] if jj < NTR else c_fp[jj - NTR]
                bits = np.unpackbits(np.asarray(pack, dtype=np.uint8).reshape(1, -1),
                                     axis=1, bitorder="big")[0, :6930].astype(np.float64)
                dots_base[jj] = float(bits @ q)
            best_base = {}
            for jj, sc in dots_base.items():
                sk = str(base_keys[int(jj)])
                c = train_map.get(sk) if int(jj) < NTR else coco_map.get(sk)
                c = str(c if c is not None else sk)
                if c not in best_base or sc > best_base[c]:
                    best_base[c] = float(sc)
            order_base = sorted(best_base.items(), key=lambda kv: (-kv[1], kv[0]))
            ranked_base = [k for k, _ in order_base]
            best_exp = dict(best_base)
            for k in we_new:
                mm, smi = new_lookup[k]
                sc = float(fp_of_source(smi) @ q)
                if k not in best_exp or sc > best_exp[k]:
                    best_exp[k] = sc
            order_exp = sorted(best_exp.items(), key=lambda kv: (-kv[1], kv[0]))
            ranked_exp = [k for k, _ in order_exp]
            d100b = oracle_diagnose_truth_in_list(canon_truth, ranked_base[:100])
            d100e = oracle_diagnose_truth_in_list(canon_truth, ranked_exp[:100])
            dfullb = oracle_diagnose_truth_in_list(canon_truth, ranked_base)
            dfulle = oracle_diagnose_truth_in_list(canon_truth, ranked_exp)
            base_rank100, exp_rank100 = d100b["rank_1based"], d100e["rank_1based"]
            r_full_b, r_full_e = dfullb["rank_1based"], dfulle["rank_1based"]
            base_rr25, exp_rr25 = rr_at_k(r_full_b, 25), rr_at_k(r_full_e, 25)
            base_top1 = ranked_base[0] if ranked_base else None
            exp_top1 = ranked_exp[0] if ranked_exp else None
            n_fp_b, n_fp_e = int(len(ranked_base)), int(len(ranked_exp))
            ts_b, ts_e = best_base.get(canon_truth), best_exp.get(canon_truth)
            b_cov100, e_cov100 = bool(base_rank100 is not None), bool(exp_rank100 is not None)
            newset = set(we_new)
            new_in_top100 = [k for k in ranked_exp[:100] if k in newset]
            new_in_top25 = [k for k in ranked_exp[:25] if k in newset]
            set_exp100, set_exp25 = set(ranked_exp[:100]), set(ranked_exp[:25])
            displaced_top100 = [k for k in ranked_base[:100] if k not in set_exp100]
            displaced_top25 = [k for k in ranked_base[:25] if k not in set_exp25]
        base_hit25 = bool(r_full_b is not None and r_full_b <= 25)
        exp_hit25 = bool(r_full_e is not None and r_full_e <= 25)
        lost25 = bool(base_hit25 and (not exp_hit25))
        worsened = bool(r_full_b is not None and r_full_e is not None and r_full_e > r_full_b)
        new100 = bool(r_full_b is None and r_full_e is not None and r_full_e <= 100)
        new25 = bool((r_full_b is None or r_full_b > 25) and exp_hit25)
        new1 = bool(r_full_e == 1 and r_full_b != 1)
        per.append({
            "canonical_key": canon_truth, "class_AB": str(m.get("class_AB", "")),
            "in_candidate_library_C": bool(m.get("in_candidate_library_C", False)),
            "n_reference_spectra_in_L": int(m.get("n_reference_spectra_in_L", 0)),
            "exposure": "unknown", "temporal_relation": "unknown",
            "n_spectra": int(m.get("n_spectra", len(m.get("spectra", [])))),
            "spectrum_ids": [str(s.get("spectrum_id", "")) for s in m.get("spectra", [])],
            "q_how": str(mol_q[canon_truth][1]),
            "QM_med": None if QM_med is None else float(QM_med),
            "mass_uncertain": bool(mass_uncertain),
            "per_spectrum_QM": qinfos,
            "full_base_covered_oracle": bool(full_base_cov),
            "full_expanded_covered_oracle": bool(full_exp_cov),
            "window_base_shipped": int(n_wb_s), "window_base_canon": int(n_wb_c),
            "window_base_covered_oracle": bool(win_b_cov),
            "window_expanded_shipped": int(n_we_s), "window_expanded_canon": int(n_we_c),
            "window_expanded_covered_oracle": bool(win_e_cov),
            "n_fp_base": int(n_fp_b), "n_fp_expanded": int(n_fp_e),
            "fp_top100_base_rank_oracle": base_rank100,
            "fp_top100_base_covered_oracle": bool(b_cov100),
            "fp_top100_expanded_rank_oracle": exp_rank100,
            "fp_top100_expanded_covered_oracle": bool(e_cov100),
            "final_top25_base_rank_oracle": r_full_b,
            "final_top25_base_RR": float(base_rr25),
            "final_top25_expanded_rank_oracle": r_full_e,
            "final_top25_expanded_RR": float(exp_rr25),
            "base_top1": base_top1, "expanded_top1": exp_top1,
            "truth_base_score": None if ts_b is None else float(ts_b),
            "truth_expanded_score": None if ts_e is None else float(ts_e),
            "lost_top25_oracle": bool(lost25), "rank_worsened_oracle": bool(worsened),
            "new_recall_top100_oracle": bool(new100), "new_recall_top25_oracle": bool(new25),
            "new_recall_top1_oracle": bool(new1),
            "n_new_in_top100": int(len(new_in_top100)), "n_new_in_top25": int(len(new_in_top25)),
            "new_in_top100": new_in_top100, "new_in_top25": new_in_top25,
            "n_displaced_top100": int(len(displaced_top100)),
            "n_displaced_top25": int(len(displaced_top25)),
            "displaced_top100": displaced_top100, "displaced_top25": displaced_top25,
        })
        print(f"  {canon_truth} {m.get('class_AB','')} QM {QM_med} unc {mass_uncertain} "
              f"full {full_base_cov}->{full_exp_cov} win {n_wb_s}->{n_we_s} "
              f"rank {r_full_b}->{r_full_e}", flush=True)

    def strat(rows):
        m_all_b = mrr_top([r["final_top25_base_rank_oracle"] for r in rows], 25)
        m_all_e = mrr_top([r["final_top25_expanded_rank_oracle"] for r in rows], 25)
        known = [r for r in rows if not r["mass_uncertain"]]
        m_kn_b = mrr_top([r["final_top25_base_rank_oracle"] for r in known], 25)
        m_kn_e = mrr_top([r["final_top25_expanded_rank_oracle"] for r in known], 25)
        return {
            "n": int(len(rows)), "n_uncertain": int(sum(1 for r in rows if r["mass_uncertain"])),
            "full_base_covered": int(sum(1 for r in rows if r["full_base_covered_oracle"])),
            "full_expanded_covered": int(sum(1 for r in rows if r["full_expanded_covered_oracle"])),
            "window_base_covered": int(sum(1 for r in rows if (not r["mass_uncertain"] and r["window_base_covered_oracle"]))),
            "window_expanded_covered": int(sum(1 for r in rows if (not r["mass_uncertain"] and r["window_expanded_covered_oracle"]))),
            "fp_top100_base_covered": int(sum(1 for r in rows if (not r["mass_uncertain"] and r["fp_top100_base_covered_oracle"]))),
            "fp_top100_expanded_covered": int(sum(1 for r in rows if (not r["mass_uncertain"] and r["fp_top100_expanded_covered_oracle"]))),
            "mrr_all_base": m_all_b, "mrr_all_expanded": m_all_e,
            "mrr_known_only_base": m_kn_b, "mrr_known_only_expanded": m_kn_e,
            "lost_top25": int(sum(1 for r in rows if r["lost_top25_oracle"])),
            "rank_worsened": int(sum(1 for r in rows if r["rank_worsened_oracle"])),
            "new_top100": int(sum(1 for r in rows if r["new_recall_top100_oracle"])),
            "new_top25": int(sum(1 for r in rows if r["new_recall_top25_oracle"])),
            "new_top1": int(sum(1 for r in rows if r["new_recall_top1_oracle"])),
            "mean_window_base_shipped": float(sum(r["window_base_shipped"] for r in rows) / len(rows)) if rows else 0.0,
            "mean_window_expanded_shipped": float(sum(r["window_expanded_shipped"] for r in rows) / len(rows)) if rows else 0.0,
            "mean_n_fp_base": float(sum(r["n_fp_base"] for r in rows) / len(rows)) if rows else 0.0,
            "mean_n_fp_expanded": float(sum(r["n_fp_expanded"] for r in rows) / len(rows)) if rows else 0.0,
            "total_new_in_top100": int(sum(r["n_new_in_top100"] for r in rows)),
            "total_new_in_top25": int(sum(r["n_new_in_top25"] for r in rows)),
            "total_displaced_top100": int(sum(r["n_displaced_top100"] for r in rows)),
            "total_displaced_top25": int(sum(r["n_displaced_top25"] for r in rows)),
        }

    summary = {
        "all_no_ref": strat(per),
        "A_in_lib_no_ref": strat([r for r in per if r["class_AB"] == "A"]),
        "B_out_of_lib_no_ref": strat([r for r in per if r["class_AB"] == "B"]),
        "B_known_mass": strat([r for r in per if (r["class_AB"] == "B" and not r["mass_uncertain"])]),
    }
    print(f"[all] {summary['all_no_ref']}", flush=True)
    print(f"[A] {summary['A_in_lib_no_ref']}", flush=True)
    print(f"[B] {summary['B_out_of_lib_no_ref']}", flush=True)
    print(f"[B_known] {summary['B_known_mass']}", flush=True)

    inputs_sha = {}
    for p in [OBS50, SRC_MASSES, SRC_KEYS, SRC_SMILES, SRC_RECORDS, SRC_META, CKPT, FP_BITS,
              os.path.join(REPO, "artifacts", "pool-train", "mass.npy"),
              os.path.join(REPO, "artifacts", "005", "coco", "coco_mass.npy"),
              os.path.join(REPO, "artifacts", "pool-train", "fp.npy"),
              os.path.join(REPO, "artifacts", "005", "coco", "coco_fp.npy")]:
        sz, sh = sha_file(p)
        inputs_sha[os.path.relpath(p, REPO)] = {"bytes": int(sz), "sha256": str(sh)}
    import torch as _t, platform
    import importlib.metadata as _md
    wall_end = datetime.now(timezone.utc).isoformat()
    out = {
        "meta": {
            "task": "009-3 LMSD single-source recall+interference on R3-50 (unlabeled + CPU FP, four stages)",
            "created_start_utc": wall_start, "created_end_utc": wall_end,
            "seed": SEED,
            "budget": "local CPU only; no training; no GPU; no Kaggle quota; no heldout spectra read/eval; "
                      "no large-full download; writable only artifacts/009-source/",
            "inputs": {
                "observations": os.path.relpath(OBS50, REPO) + " (50 molecules 25A25B, 59 spectra, read-only; "
                               "class_AB/truth only for stratified oracle reporting, never for retrieval/budget)",
                "peaks": "artifacts/007-r3/peaks/*.npz",
                "pool_base": "artifacts/pool-train/ + artifacts/005/coco/ union 712199 mmap",
                "single_source_index": "artifacts/009-source/candidate_index_* (LMSD new-only 22765; masses float64 "
                                       "ExactMolWt; query via candidate_expansion only)",
                "weights": {"path": os.path.relpath(CKPT, REPO), "bytes": int(ckpt_bytes),
                            "sha256": str(ckpt_sha), "head": "orig (fp_single_s2)"},
                "prep": "src.models.fingerprint.prep_peaks v32 verbatim",
                "window": "10ppm float64 full window (all qualified, no truncation); FP top100 pre-fixed WMAX=100 "
                          "by FP score (not mass-error); final top25 by FP score. R3 mass-ranked top100 is NOT FP shortlist.",
                "collapse": "shipped->canon max (rdkit_full), order(-score,canon)",
                "truth": "canonical_group_key oracle only (diagnostic branches), G1 caliber RR@25",
                "multi": "single->single logits; multi->mean logits (same as retrieval_scoring_10)",
                "unknown_adduct": "strict: None/empty/unknown -> mass_uncertain True, no accurate window; "
                                  "[M-H2O-H]- explicit supported (delta -19.01784115344); 4 NILU remain uncertain, 1 Bruker counted",
                "stratification": "all(50, all nref=0)=no-ref overall; A(25, in-lib no-ref); B(25, out-of-lib no-ref, "
                                  "currently out-of-library); B_known(21, B with known mass). No hidden-proportion estimation.",
            },
            "pool_counts": {"base": int(N_BASE), "single_source_new": int(len(src_keys)),
                            "expanded": int(N_EXPANDED)},
            "device": "cpu", "threads_torch": 4,
            "ce_fallback_25eV": f"{ce_fb}/{len(spec_list)}",
            "inputs_sha256": inputs_sha,
            "environment": {"python": platform.python_version(), "numpy": str(np.__version__),
                            "torch": str(_md.version("torch")), "rdkit": str(_md.version("rdkit"))},
            "forbidden_respected": ["no test files", "no training", "no GPU", "no Kaggle quota",
                                    "no heldout spectra read/eval", "no truth SMILES extraction",
                                    "no disallowed-license collection", "no network query",
                                    "no large-full download", "no G1-reserve read",
                                    "no MassBank-homolog 20/21 substitution", "no all-train NP substitution",
                                    "no public threshold tuning"],
        },
        "per_molecule": per,
        "summary": summary,
        "timing_s": float(time.time() - t0),
    }
    json.dump(out, open(OUT50, "w"), indent=1, allow_nan=False)
    print(f"wrote {OUT50} bytes {os.path.getsize(OUT50)}", flush=True)
    run_meta = {
        "task": "009-3 run_meta (index build + recall eval)",
        "seed": SEED, "device": "local CPU only",
        "created_start_utc": wall_start, "created_end_utc": wall_end,
        "timing_eval_s": float(time.time() - t0),
        "inputs_sha256": inputs_sha,
        "script_sha256": {"evaluate_009_source.py": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
                          "candidate_index_meta": json.load(open(SRC_META))},
        "environment": out["meta"]["environment"],
        "fp_checkpoint_sha256": str(ckpt_sha), "fp_bits_sha256": sha_file(FP_BITS)[1],
        "boundary": "No heldout queries, no GPU, no truth-injected candidates. R3-50 all nref=0; "
                    "B is currently out-of-library, not hidden-category inference. "
                    "MassBank 20/21 and all-train NP scores not substituted.",
    }
    json.dump(run_meta, open(OUTMETA, "w"), indent=1, allow_nan=False)
    print(f"wrote {OUTMETA}", flush=True)
    print(f"done timing {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

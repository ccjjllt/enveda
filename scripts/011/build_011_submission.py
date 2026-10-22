"""手动构建v10：仅追加有效MassBank结构；可生成独立CPU兼容性小样本入口。"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--cpu-threads', type=int, default=4)
    ap.add_argument('--probe', action='store_true', help='2 supported molecules only; NEVER submit this version')
    args = ap.parse_args()
    src = ROOT / 'notebooks/submissions/submission-v9/enveda-submission-v9.ipynb'
    nb = json.loads(src.read_text())
    asset = ROOT / 'artifacts/runs/011-expansion-asset'
    manifest_path = asset / 'mb539_manifest.json'
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    def get(i): return ''.join(nb['cells'][i]['source'])
    def put(i, s): nb['cells'][i]['source'] = s.splitlines(keepends=True)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell['execution_count'], cell['outputs'] = None, []
    put(0, '# Enveda v10: fixed MassBank expansion only\n\n'
        'v9r2 model and ranking rules are retained. Original 539-source selection contains one invalid Markush entry; '
        'the offline asset contains 538 usable structures. Original pool 712199 -> 712737. '
        'Raw candidate IDs are namespaced, canonical identities are recomputed from SMILES by the original pipeline. '
        'No multispectrum routing, unknown-CE fallback, LMSD, or model change. ICE budget remains 7200 seconds.\n'
        + ('CPU compatibility probe: only two metadata-selected supported molecules. NEVER SUBMIT THIS PROBE.\n' if args.probe else ''))
    put(2, get(2) + "\nX_MANIFEST = find('mb539_manifest.json')\nX_KEYS = find('mb539_keys.npy')\nX_MASS = find('mb539_masses.npy')\nX_FP = find('mb539_fp.npy')\nX_SMI = find('mb539_smiles.txt')\n")
    addition = '''
# Only candidate-source change: append fixed, validated external structures.
import hashlib, json
assert hashlib.sha256(Path(X_MANIFEST).read_bytes()).hexdigest() == MANIFEST_SHA
x_manifest = json.loads(Path(X_MANIFEST).read_text())
for _p in (X_KEYS, X_MASS, X_FP, X_SMI):
    _name = Path(_p).name
    assert hashlib.sha256(Path(_p).read_bytes()).hexdigest() == x_manifest['files'][_name]['sha256'], _name
x_keys = np.load(X_KEYS, allow_pickle=False)
x_mass = np.load(X_MASS, allow_pickle=False)
x_fp = np.load(X_FP, mmap_mode='r', allow_pickle=False)
x_smi = Path(X_SMI).read_text().splitlines()
NBASE = len(allk)
NEXP = len(x_keys)
assert NBASE == 712199 and NEXP == 538
assert len(x_smi) == len(x_mass) == x_fp.shape[0] == NEXP
assert x_fp.shape[1] == 867 and x_fp.dtype == np.uint8
assert x_mass.dtype == np.float64 and np.isfinite(x_mass).all() and (x_mass > 0).all()
assert len(set(x_keys.tolist())) == NEXP and not (set(allk) & set(x_keys.tolist()))
allk.extend(x_keys.tolist())
alls.extend(x_smi)
allm = np.concatenate([allm, x_mass])
assert len(allk) == len(alls) == len(allm) == 712737
print('v10 expansion:', NEXP, 'original source entries:', 539, 'invalid Markush excluded:', 1, 'total:', len(allk))
'''.replace('MANIFEST_SHA', repr(manifest_sha))
    put(3, get(3) + addition)
    old = '''                is_tr = w < NTR
                if is_tr.any():
                    Fmat[bi,np.where(is_tr)[0]] = np.unpackbits(np.asarray(t_fp[w[is_tr]], dtype=np.uint8), axis=1)[:, :6930]
                if (~is_tr).any():
                    ci = np.array([cidx[allk[j]] for j in w[~is_tr]])
                    Fmat[bi,np.where(~is_tr)[0]] = np.unpackbits(np.asarray(c_fp[ci], dtype=np.uint8), axis=1)[:, :6930]
'''
    new = '''                is_tr = w < NTR
                is_co = (w >= NTR) & (w < NBASE)
                is_ex = w >= NBASE
                if is_tr.any():
                    Fmat[bi,np.where(is_tr)[0]] = np.unpackbits(np.asarray(t_fp[w[is_tr]], dtype=np.uint8), axis=1)[:, :6930]
                if is_co.any():
                    ci = np.array([cidx[allk[j]] for j in w[is_co]])
                    Fmat[bi,np.where(is_co)[0]] = np.unpackbits(np.asarray(c_fp[ci], dtype=np.uint8), axis=1)[:, :6930]
                if is_ex.any():
                    Fmat[bi,np.where(is_ex)[0]] = np.unpackbits(np.asarray(x_fp[w[is_ex] - NBASE], dtype=np.uint8), axis=1)[:, :6930]
'''
    assert get(6).count(old) == 1
    put(6, get(6).replace(old, new))
    # Existing 25 chunks cover 2000 molecules; preserve them and handle any tail.
    put(31, get(31) + '\nfor _g0 in range(500, NG, 20):\n    run_groups(_g0, min(_g0 + 20, NG))\n')
    s = get(33)
    marker = "with open('/kaggle/working/submission.csv','w',newline='') as f:"
    assert marker in s
    s = s.replace(marker, "assert len(out_rows) == len(mids) and {r[0] for r in out_rows} == set(TQ['molecule_id']), 'output IDs incomplete'\n" + marker)
    put(33, s)
    put(34, '输出submission.csv。研究与生产记录见011进展；仅扩库，其他评分规则保留v9。\n')
    if args.cpu:
        assert args.cpu_threads >= 1
        put(1, get(1).replace("assert torch.cuda.is_available(), 'need GPU'", f"DEVICE = 'cpu'\ntorch.set_num_threads({args.cpu_threads})\nprint('CPU execution, unchanged FP/ICE weights and scoring; threads', torch.get_num_threads())"))
        put(4, get(4).replace('.cuda().eval()', '.to(DEVICE).eval()'))
        put(6, get(6).replace("device='cuda'", 'device=DEVICE'))
        put(32, get(32).replace("dev = 'cuda' if torch.cuda.is_available() else 'cpu'\nassert dev == 'cuda', 'need GPU for ICE forward'", "dev = DEVICE"))
    if args.probe:
        assert args.cpu, 'probe is CPU-only'
        selector = '''
# Manual CPU compatibility/timing sample; metadata only, no structures/labels.
_probe_ids = [mid for mid, ids in mol.items() if any(TQ['ionization_mode'][i] == 'positive' and TQ['adduct'][i] in {'[M+H]+', '[M+Na]+'} for i in ids)][:2]
assert len(_probe_ids) == 2
_probe_keep = [i for i, mid in enumerate(TQ['molecule_id']) if mid in set(_probe_ids)]
TQ = {k: [v[i] for i in _probe_keep] for k, v in TQ.items()}
mol = {}
for i, mid in enumerate(TQ['molecule_id']):
    mol.setdefault(mid, []).append(i)
print('CPU_PROBE_ONLY', _probe_ids, 'spectra', len(TQ['molecule_id']))
'''
        put(3, get(3) + selector)
        # Distinct filename prevents accidental competition submission of probe.
        put(33, get(33).replace('/kaggle/working/submission.csv', '/kaggle/working/cpu_probe.csv'))
    slug = 'enveda-v10-cpu-probe' if args.probe else 'enveda-submission-v10'
    outdir = ROOT / ('artifacts/runs/011-cpu-probe' if args.probe else 'notebooks/submissions/submission-v10')
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f'{slug}.ipynb'
    for i, c in enumerate(nb['cells']):
        assert c['cell_type'] in {'code', 'markdown', 'raw'}
        if c['cell_type'] == 'code': compile(''.join(c['source']), f'cell-{i}', 'exec')
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
    md = json.loads((ROOT / 'notebooks/submissions/submission-v9/kernel-metadata.json').read_text())
    md.update(id=f'ccjjllt/{slug}', title='Enveda v10 CPU probe' if args.probe else 'Enveda Submission v10',
              code_file=path.name, enable_gpu=not args.cpu, enable_internet=False, is_private=True)
    md['dataset_sources'] = [
        'prvsiyan/casmi26-fp-models-v2/1', 'prvsiyan/coconut-casmi26-candidates/2',
        'ccjjllt/casmi26-train-fp-pool/1', 'ahmedberatozer/casmi26-iceberg/1',
        'ccjjllt/casmi26-cp313-wheelhouse/1', 'ccjjllt/casmi26-massbank539-expansion/1']
    md['docker_image'] = 'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f'
    md['docker_image_pinning_type'] = 'original'
    if args.cpu: md.pop('machine_shape', None)
    (outdir / 'kernel-metadata.json').write_text(json.dumps(md, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'cpu': args.cpu, 'probe': args.probe, 'asset_manifest_sha': manifest_sha}))


if __name__ == '__main__': main()

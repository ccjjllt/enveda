"""Mine the exact original C34 high-ranked structures for an independent source."""
import os
import subprocess
import sys


COMPETITION = 'enveda-CASMI26-molecule-id-mass-spectra'
POPULARITY = 'casmi26-pubchem-popularity-prior'
ASSETS = 'casmi26-attributed-v4g-reproduction-assets'
TRAIN_LIMIT = 192
DEV_LIMIT = 48


def asset(name, components):
    result, = (path for path in ROOT.rglob(name) if set(components).issubset(path.parts))
    return result


def make_mining_panel():
    describe = chemistry()
    held_keys, held_scaffolds = heldout_groups(describe)
    is_pan = pan_family()
    source = pq.ParquetFile(find(COMPETITION, 'train.parquet'))
    meta = ['ingest_lib','inchikey14','normalized_smiles','instrument_type','adduct','ionization_mode','num_peaks',
            'precursor_mz','collision_energy_ev','collision_energy_orig','collision_energy_orig_units']
    best = {}
    for group in range(source.num_row_groups):
        frame = source.read_row_group(group,columns=meta).to_pandas()
        eligible = (frame.ingest_lib.eq('enveda-180') & frame.instrument_type.eq('timsTOF') &
                    frame.adduct.eq('[M-H]-') & frame.ionization_mode.eq('negative') &
                    frame.collision_energy_orig_units.eq('eV') & frame.collision_energy_orig.isin(('40','-40')) &
                    frame.num_peaks.ge(4) & frame.precursor_mz.gt(30) & frame.inchikey14.notna() & frame.normalized_smiles.notna())
        for row in frame.loc[eligible].itertuples():
            if row.collision_energy_ev is None or list(row.collision_energy_ev) != [40.]:
                continue
            if row.inchikey14 not in best or row.num_peaks > best[row.inchikey14][1].num_peaks:
                best[row.inchikey14] = (group, row)
    sizes = {'train':0, 'dev':0}
    caps = {'train':TRAIN_LIMIT,'dev':DEV_LIMIT}
    requests, used = {}, set()
    for key in sorted(best, key=hash_order):
        group, row = best[key]
        if key in held_keys or is_pan(row.normalized_smiles):
            continue
        chemical = describe(row.normalized_smiles, scaffold=True)
        if chemical is None:
            continue
        normalized, scaffold, exact_mass, _ = chemical
        if normalized in held_keys or normalized in used or scaffold in held_scaffolds:
            continue
        observed = row.precursor_mz + PROTON
        if abs(exact_mass-observed) > observed*1e-5:
            continue
        split = 'dev' if hash_order('hard-negative-dev/'+(scaffold or 'acyclic'))[0] < 51 else 'train'
        if sizes[split] == caps[split]:
            continue
        requests.setdefault(group,[]).append((row, normalized, scaffold, split))
        used.add(normalized)
        sizes[split] += 1
        if sizes == caps:
            break
    if sizes != caps:
        raise ValueError(f'Unexpected mining set sizes {sizes}')
    lines = []
    peaks = ['ms2_mzs','ms2_normalized_intensities']
    for group, order in requests.items():
        selected = source.read_row_group(group,columns=peaks).take([row.Index for row,*_ in order]).to_pylist()
        for spectrum,(row,key,scaffold,split) in zip(selected,order,strict=True):
            lines.append(dict(**spectrum,molecule_id=key,smiles=row.normalized_smiles,group=scaffold or 'acyclic',
                              split=split,source=row.ingest_lib,adduct=row.adduct,ionization_mode=row.ionization_mode,
                              precursor_mz=float(row.precursor_mz),collision_energy_ev=[40.]))
    panel = pd.DataFrame(lines).sort_values('molecule_id')
    assert panel.groupby('group').split.nunique().max() == 1
    panel.to_parquet(OUTPUT/'hard_mining_queries.parquet',index=False)
    print('Mining frozen measured spectra', sizes, 'source enveda-180',flush=True)
    return panel


def main():
    started = time.monotonic()
    panel = make_mining_panel()
    code = asset('fpnet.py', (ASSETS,'casmi26-v3-models','code','casmi')).parent.parent
    models = asset('ranker_0.pkl', (ASSETS,'casmi26-v3-models')).parent
    pool = asset('pool_meta.parquet', (ASSETS,'casmi26-v2-pool')).parent
    pc = find(POPULARITY, 'pc_mass.npy').parent
    popularity = find(POPULARITY, 'pc_src_mask.npy').parent
    fold = find('casmi26-fold-safe-fpnet','fpnet_fold0.pt')
    settings = dict(CASMI_FPNET_FOLD0=str(fold), CASMI_POP_DIR=str(popularity), CASMI_POP_LAM='0.15',
                    CASMI_POP_UNION='200', CASMI_SRC_BONUS='0.5', CASMI_SRC_BITS='1', CASMI_NP_BONUS='1.0',
                    CASMI_LOTUS_W='0.5', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                    OMP_NUM_THREADS='1', PYTHONHASHSEED='0')
    (OUTPUT/'probe_core2.py').write_text(C34_CORE)
    runner = OUTPUT/'pc_runner.py'
    runner.write_text(C34_RUNNER)
    output = OUTPUT/'hard_mining_c34.json'
    command = [sys.executable,'-u',str(runner),str(code),str(models),str(pool),str(pc),
               str(OUTPUT/'hard_mining_queries.parquet'),str(output),'4']
    subprocess.run(command,env={**os.environ,**settings},check=True)
    rows = json.loads(output.read_text())
    if set(rows) != set(panel.molecule_id):
        raise ValueError('C34 did not return the fixed mining panel.')
    report = dict(queries=len(rows), source='enveda-180', candidates=sum(len(r['pc']) for r in rows.values()),
                  window_ppm=10, limits={'train':TRAIN_LIMIT,'dev':DEV_LIMIT},
                  device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
                  seconds=round(time.monotonic()-started,1))
    (OUTPUT/'hard_mining_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()

# P0a: MIST(MSG)+MolForge chain validation on train timsTOF holdout (N spectra)
import os, sys, json, time, subprocess, hashlib
import numpy as np, pandas as pd
t0=time.time(); work='/kaggle/working/p0a'; os.makedirs(work,exist_ok=True)
audit=dict(stages={})
def stage(name,fn):
    try:
        r=fn(); audit['stages'][name]=dict(ok=True,result=r); print(f'[p0a] STAGE {name} OK {r}',flush=True)
    except Exception as e:
        import traceback; audit['stages'][name]=dict(ok=False,error=repr(e)[:500])
        print(f'[p0a] STAGE {name} FAIL {repr(e)[:500]}',flush=True)
        traceback.print_exc()
def done():
    audit['elapsed_s']=round(time.time()-t0,1)
    json.dump(audit,open(os.path.join(work,'p0a_audit.json'),'w'),indent=2)
    print('[p0a] AUDIT '+json.dumps({k:(v if k=='stages' else v) for k,v in audit.items() if k!='stages'}),flush=True)

N=int(os.environ.get('P0A_N','150'))
# 1. deps
def s_deps():
    r=subprocess.run([sys.executable,'-m','pip','install','-q','rdkit','sentencepiece','pyyaml','tqdm','joblib'],
                     capture_output=True,text=True)
    import torch
    return dict(pip_rc=r.returncode,torch=torch.__version__,cuda=torch.cuda.is_available())
stage('deps',s_deps)
# 2. code
def s_code():
    if not os.path.exists('/kaggle/working/mmf'):
        r=subprocess.run(['git','clone','--depth','1','https://github.com/harrylaucngd/MIST-MolForge.git','/kaggle/working/mmf'],
                         capture_output=True,text=True)
        if r.returncode!=0: raise RuntimeError(r.stderr[-500:])
        r=subprocess.run(['git','-C','/kaggle/working/mmf','submodule','update','--init','--recursive','--depth','1'],
                         capture_output=True,text=True)
    sys.path.insert(0,'/kaggle/working/mmf/src')
    if not os.path.exists('/kaggle/working/mist'):
        r=subprocess.run(['git','clone','--depth','1','--branch','main_v2','https://github.com/samgoldman97/mist.git','/kaggle/working/mist'],
                         capture_output=True,text=True)
        if r.returncode!=0: raise RuntimeError(r.stderr[-500:])
    return dict(mmf=True,mist=True)
stage('code',s_code)
# 3. weights
import urllib.request
def s_weights():
    V='view_only=17065170fb0e4a98a64f6d1ae6a5bd8f'
    F='https://files.osf.io/v1/resources/dqcwm/providers/osfstorage/'
    urls={'mist_fp.ckpt':'https://zenodo.org/api/records/11580401/files/mist_fp.ckpt/content',
     'decoder_molforge.pth':F+'69cb68be71f5a3841ec5b7ff?'+V,
     'combined_morgan4096_vocab_sp.model':F+'69cb6772d4a88369f793e13e?'+V,
     'combined_smiles_vocab_sp_morgan4096.model':F+'69cb6752ed5b5556b893e67e?'+V}
    dd=os.path.join(work,'ckpt'); os.makedirs(dd,exist_ok=True)
    got={}
    for name,url in urls.items():
        p=os.path.join(dd,name)
        if not os.path.exists(p):
            req=urllib.request.Request(url,headers={'User-Agent':'kaggle-p0a/1.0'})
            with urllib.request.urlopen(req,timeout=1200) as r, open(p,'wb') as f:
                h=hashlib.sha256(); n=0
                while True:
                    b=r.read(4<<20)
                    if not b: break
                    f.write(b); h.update(b); n+=len(b)
            got[name]=dict(bytes=n,sha256=h.hexdigest())
        else: got[name]=dict(bytes=os.path.getsize(p),cached=True)
    return got
stage('weights',s_weights)
# 4. mini dataset from train
def s_data():
    import glob
    tp=sorted(glob.glob('/kaggle/input/**/train.parquet',recursive=True),key=len)[0]
    df=pd.read_parquet(tp,columns=['normalized_smiles','inchikey14','molecular_formula','instrument_type','adduct','precursor_mz','ms2_mzs','ms2_normalized_intensities'])
    df=df[(df.instrument_type=='timsTOF')&(df.adduct=='[M+H]+')&(df.normalized_smiles.astype(str).str.len()>5)].copy()
    df=df.drop_duplicates('inchikey14').head(N).reset_index(drop=True)
    dd=os.path.join(work,'data'); os.makedirs(dd+'/spec_files',exist_ok=True)
    rows=[]; mgf=[]
    for i,r in df.iterrows():
        name=f's{i:05d}'
        mz=np.asarray(r.ms2_mzs,float); it=np.asarray(r.ms2_normalized_intensities,float)
        keep=(mz>0)&(it>0); mz,it=mz[keep],it[keep]
        o=np.argsort(mz); mz,it=mz[o],it[o]
        if len(mz)<5: continue
        with open(f'{dd}/spec_files/{name}.ms','w') as f:
            f.write(f'>compound {name}\n>formula {r.molecular_formula}\n>parentmass {float(r.precursor_mz)}\n>ionization [M+H]+\n>num peaks {len(mz)}\n\n>ms2peaks\n')
            f.write('\n'.join(f'{a} {b}' for a,b in zip(mz,it))+'\n')
        mgf.append(f'BEGIN IONS\nTITLE={name}\nPEPMASS={float(r.precursor_mz)}\nCHARGE=1+\n'+'\n'.join(f'{a} {b}' for a,b in zip(mz,it))+ '\nEND IONS\n')
        rows.append(dict(spec=name,formula=str(r.molecular_formula),smiles=str(r.normalized_smiles),inchikey=str(r.inchikey14),ionization='positive',instrument='timsTOF'))
    pd.DataFrame(rows).to_csv(f'{dd}/labels.tsv',sep='\t',index=False)
    pd.DataFrame([dict(name=r['spec'],split='test') for r in rows]).to_csv(f'{dd}/split.tsv',sep='\t',index=False)
    open(f'{dd}/spec.mgf','w').write('\n'.join(mgf))
    return dict(n=len(rows))
stage('data',s_data)
# 5. subformulae
def s_subform():
    dd=os.path.join(work,'data')
    import os as _os
    _env=dict(_os.environ); _env['PYTHONPATH']='/kaggle/working/mist/src:/kaggle/working/mmf/src'
    r=subprocess.run([sys.executable,'/kaggle/working/mist/src/mist/subformulae/assign_subformulae.py',
        '--spec-files',f'{dd}/spec.mgf','--labels-file',f'{dd}/labels.tsv','--output-dir',f'{dd}/subformulae',
        '--max-formulae','50','--num-workers','4'],capture_output=True,text=True,env=_env)
    found=[]
    for dp,dn,fn in os.walk(dd+'/subformulae'):
        found+=[os.path.join(dp,f) for f in fn if f.endswith('.json')]
    if r.returncode!=0 and not found: raise RuntimeError(r.stderr[-800:]+r.stdout[-800:])
    import glob as gb
    cands=sorted(gb.glob(dd+'/subformulae/**/*.json',recursive=True),key=len)
    subf=os.path.dirname(cands[0]) if cands else dd+'/subformulae'
    open(os.path.join(work,'subform_dir.txt'),'w').write(subf)
    return dict(n_json=len(cands),dir=subf,rc=r.returncode)
stage('subform',s_subform)
# 6. run benchmark core, batch_size=1, t=0.172
def s_run():
    import torch
    sys.path.insert(0,'/kaggle/working/mist/src')
    from mist_molforge.benchmark import load_mist_encoder, load_spec_data, MolForgeDecoder, get_paired_loader, normalize_formula, canonicalize_smiles, dedup_keep_order, compute_morgan_fingerprint, compute_tanimoto_similarity
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dd=os.path.join(work,'data'); ck=os.path.join(work,'ckpt')
    subf=open(os.path.join(work,'subform_dir.txt')).read().strip()
    cfg=dict(data=dict(labels_file=f'{dd}/labels.tsv',split_file=f'{dd}/split.tsv',spec_folder=f'{dd}/spec_files',subform_folder=subf),
        mist_encoder=dict(checkpoint=f'{ck}/mist_fp.ckpt',hidden_size=512,magma_modulo=2048,peak_attn_layers=2,num_heads=8,spectra_dropout=0.1,output_size=4096,form_embedder='pos-cos',set_pooling='cls',refine_layers=4,pairwise_featurization=True,embed_instrument=False,inten_transform='float',inten_prob=0.1,remove_prob=0.5,cls_type='ms1',spec_features='peakformula',mol_features='fingerprint',top_layers=1),
        molforge=dict(checkpoint=f'{ck}/decoder_molforge.pth',src_sp_model=f'{ck}/combined_morgan4096_vocab_sp.model',trg_sp_model=f'{ck}/combined_smiles_vocab_sp_morgan4096.model',root_dir='/kaggle/working/mmf/MolForge',src_seq_len=104,trg_seq_len=130,beam_size=10,append_src_eos=False),
        evaluation=dict(split='test'))
    try:
        enc=load_mist_encoder(cfg['mist_encoder'],device)
    except Exception as e:
        # fallback: DiffMS tarball mist_msg.pt
        import urllib.request
        tb=os.path.join(ck,'diffms_checkpoints.tar.gz')
        if not os.path.exists(tb):
            req=urllib.request.Request('https://zenodo.org/api/records/15122968/files/diffms_checkpoints.tar.gz/content',headers={'User-Agent':'kaggle-p0a/1.0'})
            with urllib.request.urlopen(req,timeout=1800) as r, open(tb,'wb') as f:
                while True:
                    b=r.read(4<<20)
                    if not b: break
                    f.write(b)
        import tarfile
        with tarfile.open(tb) as t:
            m=[x for x in t.getnames() if x.endswith('mist_msg.pt')]
            t.extract(m[0],ck)
            import shutil; shutil.move(os.path.join(ck,m[0]),os.path.join(ck,'mist_msg.pt'))
        cfg['mist_encoder']['checkpoint']=os.path.join(ck,'mist_msg.pt')
        enc=load_mist_encoder(cfg['mist_encoder'],device)
    dataset,pairs=load_spec_data(cfg['data'],cfg['mist_encoder'],split='test',max_spectra=None)
    dec=MolForgeDecoder(checkpoint_path=cfg['molforge']['checkpoint'],src_sp_model_path=cfg['molforge']['src_sp_model'],trg_sp_model_path=cfg['molforge']['trg_sp_model'],device=device,molforge_root=cfg['molforge']['root_dir'],src_seq_len=104,trg_seq_len=130,beam_size=10,append_src_eos=False)
    loader=get_paired_loader(dataset,shuffle=False,batch_size=1,num_workers=0)
    top1=top10=ntot=0; tans=[]; ex=[]
    import torch as T
    for bi,batch in enumerate(loader):
        batch={k:(v.to(device) if isinstance(v,T.Tensor) else v) for k,v in batch.items()}
        with T.no_grad():
            probs,_=enc(batch); probs=probs.detach().cpu().numpy()[0]
        spec,mol=pairs[bi]
        true=str(mol.get_smiles())
        on=np.where(probs>=0.172)[0].tolist()
        cands,_=dec.generate_topk_smiles(on,top_k=10)
        cands=dedup_keep_order([canonicalize_smiles(s) for s in cands])[:10]
        try:
            tc=canonicalize_smiles(true)
        except Exception: tc=true
        hit1=int(len(cands)>0 and cands[0]==tc); hit10=int(tc in cands)
        top1+=hit1; top10+=hit10; ntot+=1
        try:
            from rdkit.Chem import DataStructs
            from rdkit.Chem import rdFingerprintGenerator
            gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
            mt=Chem.MolFromSmiles(tc); f1=gen.GetFingerprint(mt) if mt else None
            bt=0.0
            for c in cands[:1]:
                mc=Chem.MolFromSmiles(c)
                if mc and f1: bt=DataStructs.TanimotoSimilarity(f1,gen.GetFingerprint(mc))
            tans.append(bt)
        except Exception: pass
        if len(ex)<10: ex.append(dict(spec=spec.get_spec_name(),true=true[:80],top1=cands[0][:80] if cands else None,hit1=hit1,hit10=hit10,nbits=len(on)))
        print(f'[p0a] {bi+1} onbits={len(on)} hit1={hit1} hit10={hit10}',flush=True)
    res=dict(n=ntot,top1=top1,top10=top10,top1_rate=round(top1/max(ntot,1),4),top10_rate=round(top10/max(ntot,1),4),tan_mean=round(float(np.mean(tans)),4) if tans else 0,elapsed_s=round(time.time()-t0,1),examples=ex)
    json.dump(res,open(os.path.join(work,'p0a_result.json'),'w'),indent=2)
    audit['result']=res
    return res
stage('run',s_run)
done()

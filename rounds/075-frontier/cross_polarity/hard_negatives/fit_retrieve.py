"""Fit on the real C34 mistakes and rank the complete blinded negative-ion windows."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import sha256
from multiprocessing import get_context


MINING = 'enveda-075-negative-hard-mine'
ARCHIVE = 'enveda-075b-massbank-adducts-pipeline'
CONTROL = 'enveda-075-negative-window-control-cpu'
INITIAL = 'enveda-075-cross-polarity-pilot'
SEED = 7523
NEGATIVES = 16
EPOCHS = 8


def examples(describe):
    panel = pd.read_parquet(find(MINING, 'hard_mining_queries.parquet'))
    raw = json.loads(find(MINING, 'hard_mining_c34.json').read_text())
    invariant = panel.groupby('group').split.nunique().max()
    if invariant != 1 or set(panel.source) != {'enveda-180'}:
        raise ValueError('Unexpected source or structures spanning the development fold.')
    banned_keys, banned_groups = heldout_groups(describe)
    is_pan = pan_family()
    dev = panel[panel.split.eq('dev')]
    dev_keys, dev_groups = set(dev.molecule_id), set(dev.group)

    @lru_cache(maxsize=16000)
    def candidate(smiles):
        return None if is_pan(smiles) else describe(smiles, scaffold=True)

    spec, fp, mask, train, sample_audit = [], [], [], [], []
    for row in panel.itertuples(index=False):
        truth = candidate(row.smiles)
        if truth is None or truth[0] != row.molecule_id or truth[0] in banned_keys or truth[1] in banned_groups:
            raise ValueError(f'Training panel includes an invalid or held-out positive {row.molecule_id}')
        bank = [truth[-1]]
        seen = {truth[0]}
        mined = raw[row.molecule_id]
        native_ranks = []
        for rank, (smiles, native_key) in enumerate(zip(mined['pc'],mined['pc_keys'],strict=True),1):
            chem = candidate(smiles)
            if chem is None or chem[0] in seen or chem[1] in banned_groups or {chem[0], native_key} & banned_keys:
                continue
            if row.split == 'train' and (chem[0] in dev_keys or (chem[1] or 'acyclic') in dev_groups):
                continue
            bank.append(chem[-1])
            seen.add(chem[0])
            native_ranks.append(rank)
            if len(native_ranks) == NEGATIVES:
                break
        if len(native_ranks) < 4:
            continue
        present = np.arange(NEGATIVES+1) < len(bank)
        pad = np.zeros((NEGATIVES+1,2048),dtype=np.uint8)
        pad[:len(bank)] = np.stack(bank)
        spec.append(spectrum_vector(row.ms2_mzs,row.ms2_normalized_intensities,row.precursor_mz,1))
        fp.append(pad)
        mask.append(present)
        train.append(row.split == 'train')
        sample_audit.append(dict(key=row.molecule_id, split=row.split, scaffold=row.group, c34_ranks=native_ranks))
    train = np.asarray(train, dtype=bool)
    if train.sum() < 64 or (~train).sum() < 16:
        raise ValueError(f'Too few genuine C34 contrasts: {(int(train.sum()),int((~train).sum()))}')
    return [np.stack(array) for array in (spec,fp,mask)], train, sample_audit


def scores(model, x, fingerprints, eligible):
    query = model.encode_query(x)
    molecules = model.encode_candidate(fingerprints)
    return (24*(molecules*query[:,None,:]).sum(-1)).masked_fill(~eligible,-torch.inf)


@torch.inference_mode()
def diagnostic(model, tensors, idx):
    values = torch.cat([scores(model, *(t[block] for t in tensors)) for block in np.array_split(idx,4)])
    ranks = 1 + (values[:,1:] > values[:,:1]).sum(1)
    return dict(n=len(ranks), mrr=float((1/ranks).mean()), top1=int((ranks==1).sum()), top5=int((ranks<=5).sum()))


def fit(device, describe):
    x, mask, audit = examples(describe)
    tensors = [torch.from_numpy(t).to(device) for t in x]
    model = Retriever().to(device)
    original = find(INITIAL,'paired.pt')
    model.load_state_dict(torch.load(original,map_location=device,weights_only=True))
    model.requires_grad_(False)
    model.query[-1].requires_grad_(True)
    model.candidate[-1].requires_grad_(True)
    training, development = np.flatnonzero(mask), np.flatnonzero(~mask)
    # Development is a holdout for this adaptation. Keep its labels out of model selection.
    previous = diagnostic(model,tensors,development)
    optimized = [x for x in model.parameters() if x.requires_grad]
    starting = [x.detach().clone() for x in optimized]
    optim = torch.optim.AdamW(optimized,lr=3e-4,weight_decay=0.01)
    generator = np.random.default_rng(SEED)
    losses = []
    for epoch in range(EPOCHS):
        unpermuted = generator.permutation(training)
        batch_losses = []
        for ids in np.array_split(unpermuted,max(1,len(unpermuted)//16)):
            logits = scores(model,*(t[ids] for t in tensors))
            positives = torch.zeros(len(ids),dtype=torch.long,device=device)
            anchor = sum(F.mse_loss(param, start) for param,start in zip(optimized,starting,strict=True))
            loss = F.cross_entropy(logits,positives) + 10*anchor
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            batch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(batch_losses)))
        print('hard-negative adaptation epoch',epoch+1,'loss',losses[-1],flush=True)
    model.eval()
    data = dict(train=int(mask.sum()), dev=int((~mask).sum()), source='enveda-180',
                trainable_parameters=sum(x.numel() for x in optimized),
                training_negative_pairs=sum(len(r['c34_ranks']) for r in audit if r['split']=='train'),
                dev_before=previous, dev_after=diagnostic(model,tensors,development), losses=losses,
                seed=SEED, epochs=EPOCHS, candidate_cap=NEGATIVES,
                heldout_contributors=['Eawag','BAFG'], heldout_homologs='PAN-Sulfonate C4-C13')
    (OUTPUT/'hard_adaptation_examples.json').write_text(json.dumps(audit,indent=2)+'\n')
    torch.save(model.state_dict(),OUTPUT/'hard_adapted.pt')
    print(json.dumps(data),flush=True)
    return model, data


def score_windows(device, model):
    queries = pd.read_parquet(find(PANEL,'pipeline_queries.parquet'))
    queries = queries[queries.adduct.eq('[M-H]-') & queries.ionization_mode.eq('negative')]
    if len(queries) != 47 or not queries.molecule_id.is_unique:
        raise ValueError('Expected the frozen 47-query blind panel.')
    frozen_names = ('candidates.parquet','source_candidates.parquet')
    references = [pd.read_parquet(find(slug,name),columns=['molecule_id','candidate_id'])
                  for slug in (ARCHIVE,CONTROL) for name in frozen_names]
    old = pd.concat(references).groupby('molecule_id').candidate_id.agg(set).to_dict()
    masses = (queries.precursor_mz.to_numpy() + PROTON).tolist()
    counts, proposals, frames = {}, {}, []
    with ProcessPoolExecutor(max_workers=4,mp_context=get_context('fork'),initializer=init_window) as pool:
        pending = pool.map(enumerate_window,masses,chunksize=1)
        with torch.inference_mode():
            for query,mass,(per_store,keys,smiles,fp) in zip(queries.itertuples(),masses,pending,strict=True):
                mid = str(query.molecule_id)
                spectrum = spectrum_vector(query.ms2_mzs,query.ms2_normalized_intensities,query.precursor_mz,1)
                vector = model.encode_query(torch.as_tensor(spectrum,device=device))
                raw = [(model.encode_candidate(torch.as_tensor(fp[i:i+BLOCK],device=device)) @ vector).cpu().numpy()
                       for i in range(0,len(fp),BLOCK)]
                similarity = np.concatenate(raw) if raw else np.empty(0,dtype=float)
                if not np.all(np.isfinite(similarity)):
                    raise ValueError(f'Invalid model scores for {mid}')
                order = np.argsort(-similarity,kind='stable')
                novel = [i for i in order if keys[i] not in old.get(mid,())]
                proposals[mid] = dict(target=mass, entries=[dict(smiles=smiles[i],key=keys[i],score=float(similarity[i]))
                                                           for i in novel[:PROPOSALS]])
                frames.append(pd.DataFrame(dict(molecule_id=mid,candidate_id=np.asarray(keys,dtype=str)[order],
                                                rank=np.arange(1,len(order)+1))))
                counts[mid] = dict(store_rows=list(map(int,per_store)),normalized_candidates=len(keys),
                                   outside_original_union=len(novel))
                if len(proposals)%4==0 or len(proposals)==47:
                    print('hard-negative complete-window ranking',len(proposals),'/',len(queries),flush=True)
    pd.concat(frames,ignore_index=True).to_parquet(OUTPUT/'negative_window_ranks.parquet',index=False)
    path = OUTPUT/'negative_window_proposals.json'
    path.write_text(json.dumps(proposals,allow_nan=False)+'\n')
    return dict(queries=len(proposals),ppm=PPM,proposal_cap=PROPOSALS,by_query=counts,
                proposals_sha256=sha256(path.read_bytes()).hexdigest())


def main():
    started = time.monotonic()
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, adaptation = fit(device,chemistry())
    fit_seconds = time.monotonic()-started
    (OUTPUT/'hard_adaptation_receipt.json').write_text(json.dumps({**adaptation,'seconds':round(fit_seconds,1)},indent=2)+'\n')
    report = score_windows(device,model)
    report.update(seconds=round(time.monotonic()-started-fit_seconds,1),
                  device=torch.cuda.get_device_name() if device=='cuda' else 'CPU')
    (OUTPUT/'negative_window_receipt.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()

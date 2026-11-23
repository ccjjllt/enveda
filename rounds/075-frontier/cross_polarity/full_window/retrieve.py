"""Rank all stored structures in each observed negative-ion PubChem mass window."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

PANEL = 'enveda-075b-external-massbank-adducts'
ORIGINAL = 'enveda-075b-massbank-adducts-pipeline'
MODELS = 'enveda-075-cross-polarity-pilot'
POPULARITY = 'casmi26-pubchem-popularity-prior'
PROPOSALS = 64
PPM = 10
BLOCK = 4096


def index_files():
    base = find(POPULARITY, 'pc_mass.npy')
    delta = find(POPULARITY, 'delta_pc_mass.npy').parent
    stores = []
    for directory, prefix in ((base.parent, 'pc'), (delta, 'delta_pc'), (delta, 'growth_pc')):
        mass, offsets, encoded = (np.load(directory / f'{prefix}_{kind}.npy', mmap_mode='r')
                                  for kind in ('mass', 'off', 'smiles'))
        if offsets.shape != (len(mass) + 1,) or offsets[0] != 0 or offsets[-1] != len(encoded):
            raise ValueError(f'Misaligned PubChem store {prefix}')
        stores.append((mass, offsets, encoded))
    return stores


def init_window():
    global _describe, _indices
    _describe = chemistry()
    _indices = index_files()


def enumerate_window(observed_mass):
    tolerance = observed_mass * PPM * 1e-6
    counts, structures = [], {}
    for masses, offsets, blob in _indices:
        low = int(np.searchsorted(masses, observed_mass - tolerance, side='left'))
        high = int(np.searchsorted(masses, observed_mass + tolerance, side='right'))
        counts.append(high - int(low))
        encoded = bytes(blob[offsets[low]:offsets[high]])
        local_offsets = offsets[low:high + 1] - offsets[low]
        for start, end in zip(local_offsets[:-1], local_offsets[1:]):
            text = encoded[start:end].decode('ascii')
            chemical = _describe(text)
            if chemical is not None and chemical[0]:
                structures.setdefault(chemical[0], (text, chemical[-1]))
    keys = list(structures)
    smiles = [structures[key][0] for key in keys]
    vectors = np.stack([structures[key][1] for key in keys]) if keys else np.empty((0, 2048), dtype=np.uint8)
    return counts, keys, smiles, vectors


def main():
    start = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError('Expected a Kaggle GPU for full-window scoring.')
    # Install the pinned wheel once before starting any scoring workers.
    chemistry()
    queries = pd.read_parquet(find(PANEL, 'pipeline_queries.parquet'))
    queries = queries[queries.adduct.eq('[M-H]-') & queries.ionization_mode.eq('negative')]
    if len(queries) != 47 or not queries.molecule_id.is_unique:
        raise ValueError('Expected the fixed 47 negative-ion queries.')
    old_candidates = [pd.read_parquet(find(ORIGINAL, name), columns=['molecule_id', 'candidate_id'])
                      for name in ('candidates.parquet', 'source_candidates.parquet')]
    old = pd.concat(old_candidates).groupby('molecule_id').candidate_id.agg(set).to_dict()
    observed = (queries.precursor_mz.to_numpy() + PROTON).tolist()
    torch.set_num_threads(1)
    options = dict(max_workers=4, mp_context=get_context('fork'), initializer=init_window)
    with ProcessPoolExecutor(**options) as pool:
        computed = pool.map(enumerate_window, observed, chunksize=1)
        model = Retriever().cuda().eval()
        weight = find(MODELS, 'paired.pt')
        model.load_state_dict(torch.load(weight, map_location='cuda', weights_only=True))
        proposals, audit, rank_tables = {}, {}, []
        with torch.inference_mode():
            for rec, mass, (counts, keys, smiles, fp) in zip(queries.itertuples(), observed, computed, strict=True):
                mids = str(rec.molecule_id)
                qs = spectrum_vector(rec.ms2_mzs, rec.ms2_normalized_intensities, rec.precursor_mz, 1)
                q = model.encode_query(torch.as_tensor(qs, device='cuda'))
                blocks = [(model.encode_candidate(torch.from_numpy(fp[i:i+BLOCK]).cuda()) @ q).cpu().numpy()
                          for i in range(0, len(fp), BLOCK)]
                similarity = np.concatenate(blocks) if blocks else np.empty(0, dtype=float)
                if not np.all(np.isfinite(similarity)):
                    raise ValueError(f'Nonfinite learned retrieval scores for {mids}')
                order = np.argsort(-similarity, kind='stable')
                outside = [i for i in order if keys[i] not in old.get(mids, ())]
                proposals[mids] = {'target': mass, 'entries': [dict(smiles=smiles[i], key=keys[i],
                    score=float(similarity[i])) for i in outside[:PROPOSALS]]}
                # A complete identity ranking remains available for the independent truth audit.
                rank_tables.append(pd.DataFrame(dict(molecule_id=mids,
                    candidate_id=np.asarray(keys, dtype=str)[order], rank=np.arange(1, len(order) + 1),
                    outside_original_union=[keys[i] not in old.get(mids, ()) for i in order])))
                audit[mids] = {'store_rows': list(map(int, counts)), 'normalized_candidates': len(keys),
                               'outside_original_union': len(outside)}
                if len(audit) % 4 == 0 or len(audit) == len(queries):
                    print('Full-window GPU ranking', len(audit), '/', len(queries), flush=True)
    pd.concat(rank_tables, ignore_index=True).to_parquet(OUTPUT / 'negative_window_ranks.parquet', index=False)
    (OUTPUT / 'negative_window_proposals.json').write_text(json.dumps(proposals, allow_nan=False) + '\n')
    receipt = dict(queries=len(proposals), ppm=PPM, proposal_cap=PROPOSALS, by_query=audit,
                   device=torch.cuda.get_device_name(), seconds=round(time.monotonic() - start, 1))
    (OUTPUT / 'negative_window_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('Scored', sum(x['normalized_candidates'] for x in audit.values()), 'normalized candidates in',
          receipt['seconds'], 'seconds', flush=True)


if __name__ == '__main__':
    main()

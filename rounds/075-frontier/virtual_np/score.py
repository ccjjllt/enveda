"""Blind CPU FPNet A dot scores using the released C34 query preprocessing."""

import argparse
import ast
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from hashlib import file_digest, sha256
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ARCHIVE_SHA = 'd7b28154b1bdf132528300bcf8faa1211984c843d5340ac81191472a452f564b'
# Pablo's sorted C34 A+B bank names the original fe_A checkpoint fpnet_0.pt.
# The digest was verified in the five-miss C34 audit and the v4b release manifest.
NATIVE_MODEL_A = 'casmi26-v3-models/fpnet_0.pt'
MODEL_A_SHA256 = 'edccf479a2a2e53461a06bc5e1aee4341b22abb32302c4f6d2c45281000d742a'
V3_HASHES = {
    'chem.py': '99474f23d8a67040b0c0f8c70e73592a6f7f656f50103e350ca473451e2169f1',
    'fpnet.py': '0ea6e372d1f045a877d350234c2ccf5d93fdd98cb23bd4c22376bf7e99547608',
    'spectra.py': 'e2481761cdfb2515e8a945751f85f6b9e885680a8e974d7ac35c78bbf279ae74',
}
SPECTRUM_FIELDS = ('molecule_id', 'precursor_mz', 'adduct', 'ionization_mode',
                   'collision_energy_ev', 'ms2_mzs', 'ms2_normalized_intensities')
SCHEMA = pa.schema([
    ('molecule_id', pa.string()), ('arm', pa.string()), ('native_position', pa.int32()),
    ('smiles', pa.string()), ('fpnet_a_fz', pa.float64()), ('score_key', pa.string()),
])
TOP_SCHEMA = SCHEMA.append(pa.field('rank', pa.int32()))


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def c34_snapshot(path):
    document = json.loads(path.read_text())
    if 'molecules' in document:
        archive_sha = document['archive_sha256']
        original = document['molecules']
    else:
        archive_sha = digest(path)
        original = document
    if archive_sha != ARCHIVE_SHA:
        raise ValueError('The comparison expects the archived original Pablo C34 /2 lists')
    for mid, row in original.items():
        if (not isinstance(row['target'], (int, float)) or not np.isfinite(row['target'])
                or not isinstance(row['pc'], list) or len(row['pc']) > 25
                or any(not isinstance(s, str) or not s for s in row['pc'])):
            raise ValueError(f'Invalid archived direct-native candidates: {mid}')
    return original


def native_helper(path):
    text = path.read_text()
    tree = ast.parse(text)
    if not any(isinstance(node, ast.FunctionDef) and node.name == 'molecule_logits'
               for node in tree.body):
        literals = [node.value for node in tree.body if isinstance(node, ast.Assign)
                    and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == 'CORE']
        if len(literals) != 1:
            raise ValueError('Expected Pablo inner cell 34 or its extracted probe_core2.py')
        text = ast.literal_eval(literals[0])
        tree = ast.parse(text)
    function, = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == 'molecule_logits']
    # Execute only the author's query helper, never the worker or notebook runner.
    scoped = {'np': np}
    standalone = ast.Module(body=[function], type_ignores=[])
    exec(compile(standalone, str(path), 'exec'), scoped)
    return scoped['molecule_logits'], sha256(ast.get_source_segment(text, function).encode()).hexdigest()


def observations(path, archive, chem):
    table = pd.read_parquet(path, columns=list(SPECTRUM_FIELDS))
    if set(table['molecule_id']) != set(archive):
        raise ValueError('Archived C34 lists and blind spectra have different molecule IDs')
    groups = {}
    for mid, group in table.groupby('molecule_id', sort=False):
        neutral = chem.neutral_mass(group.precursor_mz.to_numpy(dtype=np.float64), group.adduct.to_numpy())
        finite = neutral[np.isfinite(neutral)]
        if not len(finite) or not np.isclose(np.median(finite), archive[mid]['target'], atol=1e-8, rtol=0.):
            raise ValueError(f'C34 did not use the same measured mass observations for {mid}')
        spectra = []
        for row in group.itertuples(index=False):
            energy = (np.atleast_1d(np.asarray(row.collision_energy_ev, dtype=float))
                      if row.collision_energy_ev is not None else np.zeros(0))
            spectra.append(dict(mz=np.asarray(row.ms2_mzs, np.float64),
                                it=np.asarray(row.ms2_normalized_intensities, np.float64),
                                mode=1 if row.ionization_mode == 'positive' else -1,
                                adduct=row.adduct, prec=float(row.precursor_mz),
                                ce=float(energy.mean()) if len(energy) else np.nan,
                                ce_n=int(len(energy)) if len(energy) else 1))
        groups[mid] = spectra[:16]
    return groups


def query_cache(args, archive, chem, core, provenance):
    groups = observations(args.queries, archive, chem)
    mids = list(groups)
    if args.logits:
        prior = json.loads(args.logits.with_suffix('.json').read_text())
        if prior['inputs'] != provenance or prior['logits_sha256'] != digest(args.logits):
            raise ValueError('A-only cached logits describe different scoring inputs')
        with np.load(args.logits, allow_pickle=False) as saved:
            if saved['molecule_id'].tolist() != mids:
                raise ValueError('A-only cache molecule order differs')
            result = saved['logits']
    else:
        import torch
        from casmi import fpnet
        torch.set_num_threads(args.workers)
        bank = fpnet.ModelBank([str(args.model_a)], device='cpu')
        if bank.nbits != provenance['n_bits']:
            raise ValueError('FPNet A and the published pool have different fingerprint widths')
        result = np.stack([core(bank, groups[mid]) for mid in mids])
    if result.shape != (len(mids), provenance['n_bits']) or not np.isfinite(result).all():
        raise ValueError('Incomplete or nonfinite FPNet A representation')
    path = args.outdir / 'fpnet_a_query_logits.npz'
    np.savez_compressed(path, molecule_id=np.array(mids), logits=result)
    path.with_suffix('.json').write_text(json.dumps(
        dict(inputs=provenance, logits_sha256=digest(path), read_cache=bool(args.logits)), indent=2) + '\n')
    return dict(zip(mids, result, strict=True))


def initialize(root, bits, logits):
    global CHEM, BITS, LOGITS
    sys.path.insert(0, str(root))
    from casmi import chem
    CHEM, BITS, LOGITS = chem, bits, logits


def score_batch(request):
    scores = []
    for mid, arm, position, smiles, key in request:
        raw = CHEM.raw_fingerprint(smiles)
        value = None if raw is None else float(raw[BITS].astype(np.float32) @ LOGITS[mid])
        if value is not None and not np.isfinite(value):
            value = None
        scores.append(dict(molecule_id=mid, arm=arm, native_position=position,
                           smiles=smiles, fpnet_a_fz=value, score_key=key))
    return scores


def requests(args, archive, chem):
    # C34 chooses these positions before the A-only score is read.
    originals = [(mid, 'native_c34', i, smiles, chem.score_key(smiles))
                 for mid, row in archive.items() for i, smiles in enumerate(row['pc'], 1)]
    for start in range(0, len(originals), args.batch):
        yield originals[start:start + args.batch]
    columns = ['molecule_id', 'smiles', 'score_key']
    if args.candidate_source == 'known_main':
        columns.append('in_main_pool')
    for batch in pq.ParquetFile(args.proposals).iter_batches(batch_size=args.batch, columns=columns):
        parsed = batch.to_pylist()
        if any(row['molecule_id'] not in archive for row in parsed):
            raise ValueError('A proposal refers to a molecule outside the C34 baseline')
        if args.candidate_source == 'known_main' and any(row['in_main_pool'] is not True for row in parsed):
            raise ValueError('The known-main FPNet arm contains a non-main-pool proposal')
        yield [(row['molecule_id'], 'generated', None, row['smiles'], row['score_key'])
               for row in parsed]


def bounded_results(pool, tasks, workers):
    tasks = iter(tasks)
    pending = deque()
    while len(pending) < 2 * workers and (todo := next(tasks, None)) is not None:
        pending.append(pool.submit(score_batch, todo))
    while pending:
        yield pending.popleft().result()
        todo = next(tasks, None)
        if todo is not None:
            pending.append(pool.submit(score_batch, todo))


def score(args):
    from rdkit import rdBase
    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError(f'Expected competition RDKit 2026.03.3, found {rdBase.rdkitVersion}')
    if not 1 <= args.cap <= 25 or args.workers < 1 or args.batch < 1:
        raise ValueError('Use a candidate cap of 1–25 and a positive worker/batch count')
    if args.candidate_source == 'known_main' and (args.proposals.name != 'known_main.parquet' or args.cap != 24):
        raise ValueError('The known-main FPNet arm requires the sealed known_main.parquet and cap 24')
    model_a_sha256 = digest(args.model_a)
    if args.model_a.name != Path(NATIVE_MODEL_A).name or model_a_sha256 != MODEL_A_SHA256:
        raise ValueError('Expected the exact official native C34 FPNet A checkpoint')
    modules = {name: digest(args.code_root / 'casmi' / name) for name in V3_HASHES}
    if modules != V3_HASHES:
        raise ValueError('FPNet v3 source differs from the archived C34 source')
    start = time.monotonic()
    args.outdir.mkdir(parents=True, exist_ok=True)
    archive = c34_snapshot(args.c34)
    sys.path.insert(0, str(args.code_root))
    from casmi import chem
    core, core_hash = native_helper(args.core)
    bits = np.load(args.bits, allow_pickle=False)
    if (bits.ndim != 1 or not len(bits) or not np.issubdtype(bits.dtype, np.integer)
            or min(bits) < 0 or max(bits) >= chem.FP_RAW_BITS or len(set(bits)) != len(bits)):
        raise ValueError('The input must be C34’s ordered selected raw fingerprint bit indices')
    provenance = dict(query_sha256=digest(args.queries), original_c34_archive_sha256=ARCHIVE_SHA,
                      model_a_artifact=NATIVE_MODEL_A, model_a_sha256=model_a_sha256,
                      bits_sha256=digest(args.bits),
                      native_c34_helper_sha256=core_hash, v3_source_sha256=modules,
                      n_bits=len(bits), rdkit=rdBase.rdkitVersion,
                      logits='A-only under Pablo C34 query representation, native 1.5 scaling', device='cpu')
    logits = query_cache(args, archive, chem, core, provenance)
    print(f'Cached {len(logits)} FPNet A query vectors in {time.monotonic() - start:.1f} seconds', flush=True)
    best, controls, counts, missing = defaultdict(list), defaultdict(list), Counter(), Counter()
    all_scores = args.outdir / 'fpnet_a_scores.parquet'
    with pq.ParquetWriter(all_scores, SCHEMA) as writer, ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context('spawn'), initializer=initialize,
        initargs=(args.code_root, bits, logits),
    ) as pool:
        for batch in bounded_results(pool, requests(args, archive, chem), args.workers):
            writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
            updated = set()
            for row in batch:
                mid, arm = row['molecule_id'], row['arm']
                counts[mid, arm] += 1
                if row['fpnet_a_fz'] is None or not row['score_key']:
                    missing[mid, arm] += 1
                elif arm == 'native_c34':
                    controls[mid].append(row)
                else:
                    best[mid].append(row)
                    updated.add(mid)
            for mid in updated:
                best[mid] = sorted(best[mid], key=lambda row: (-row['fpnet_a_fz'], row['smiles']))[:args.cap]
    top, cases = [], {}
    for mid in archive:
        budget = min(args.cap, len(best[mid]), len(controls[mid]))
        cases[mid] = dict(candidate_budget=budget,
                          generated_total=counts[mid, 'generated'], generated_invalid=missing[mid, 'generated'],
                          native_total=counts[mid, 'native_c34'], native_invalid=missing[mid, 'native_c34'])
        for ordered in (best[mid][:budget], controls[mid][:budget]):
            top.extend(dict(row, rank=i) for i, row in enumerate(ordered, 1))
    shortlist = args.outdir / 'fpnet_a_ranked25.parquet'
    pq.write_table(pa.Table.from_pylist(top, schema=TOP_SCHEMA), shortlist)
    receipt = dict(inputs=provenance, compact_snapshot_sha256=digest(args.c34),
                   proposals_sha256=digest(args.proposals), cap=args.cap, cases=cases,
                   output_sha256={path.name: digest(path) for path in (all_scores, shortlist)},
                   elapsed_seconds=round(time.monotonic() - start, 2))
    if args.candidate_source == 'known_main':
        receipt['proposal_artifact'] = 'known_main.parquet'
    (args.outdir / 'fpnet_a_score_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'Blind A-only scoring done {sum(counts.values())} candidates in {receipt["elapsed_seconds"]} seconds',
          flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('queries', 'c34', 'proposals', 'code-root', 'model-a', 'bits', 'core', 'outdir'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--logits', type=Path)
    parser.add_argument('--candidate-source', choices=('proposals', 'known_main'), default='proposals')
    parser.add_argument('--cap', type=int, default=24)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch', type=int, default=512)
    score(parser.parse_args())

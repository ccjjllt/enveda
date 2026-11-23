"""Trace absent identities across the exact C34 stores and alternate normalization."""
import sys
from collections import Counter


PANEL = 'enveda-075b-external-massbank-adducts'
ARCHIVE = 'enveda-075b-massbank-adducts-pipeline'
WINDOW = 'enveda-075-negative-full-window'
STORE_NAMES = ('tier', 'delta', 'growth')


def native_code():
    matches = [path.parent.parent for path in ROOT.rglob('fpnet.py')
               if {'casmi26-v3-models', 'code', 'casmi'}.issubset(path.parts)]
    code, = matches
    return code


def string_hits(masses, offsets, blob, variants):
    """Find the given complete SMILES representations at any indexed mass."""
    block_size = 32 * 1024 * 1024
    width = max(map(len, variants))
    matches = {}
    for start in range(0, len(blob), block_size):
        data = bytes(blob[start:start + block_size + width])
        for text in variants:
            pos = data.find(text)
            while 0 <= pos < block_size:
                offset = start + pos
                row = int(np.searchsorted(offsets, offset))
                if row < len(masses) and offsets[row] == offset and offsets[row + 1] == offset + len(text):
                    matches.setdefault(text, []).append(float(masses[row]))
                pos = data.find(text, pos + 1)
    return matches


def main():
    started = time.monotonic()
    chemistry()
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize as standard

    sys.path.insert(0, str(native_code()))
    from casmi import chem

    tautomer = standard.TautomerEnumerator()
    uncharger = standard.Uncharger()

    def signature(text):
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            return None
        raw_key = Chem.MolToInchiKey(molecule)
        raw_formula = rdMolDescriptors.CalcMolFormula(molecule)
        variants = {text, Chem.MolToSmiles(molecule)}
        Chem.RemoveStereochemistry(molecule)
        variants.add(Chem.MolToSmiles(molecule))
        canonical = tautomer.Canonicalize(molecule)
        variants.add(Chem.MolToSmiles(canonical))
        parent = tautomer.Canonicalize(uncharger.uncharge(standard.FragmentParent(molecule)))
        variants.add(Chem.MolToSmiles(parent))
        return dict(native=chem.score_key(text), raw=raw_key, tautomer=Chem.MolToInchiKey(canonical)[:14],
                    parent=Chem.MolToInchiKey(parent)[:14], formula=raw_formula,
                    exact_mass=rdMolDescriptors.CalcExactMolWt(molecule), variants=sorted(variants))

    queries = pd.read_parquet(find(PANEL, 'pipeline_queries.parquet'))
    labels = pd.read_parquet(find(PANEL, 'labels.parquet'))
    panel = queries.loc[queries.adduct.eq('[M-H]-')].merge(labels, on='molecule_id', validate='one_to_one')
    old = pd.concat([pd.read_parquet(find(ARCHIVE, name), columns=['molecule_id','candidate_id'])
                     for name in ('candidates.parquet','source_candidates.parquet')]).drop_duplicates()
    ranked = pd.read_parquet(find(WINDOW, 'negative_window_ranks.parquet'), columns=['molecule_id','candidate_id'])
    seen = set(old.itertuples(index=False, name=None)) | set(ranked.itertuples(index=False, name=None))
    targets = {r.molecule_id: r for r in panel.itertuples(index=False) if (r.molecule_id,r.candidate_id) not in seen}
    if len(targets) != 11:
        raise ValueError(f'Expected the 11 archived misses, found {len(targets)}')
    identities = {mid: signature(r.truth_smiles) for mid,r in targets.items()}
    variants = {value.encode('ascii') for item in identities.values() for value in item['variants']}
    stores = index_files()
    global_hits = [string_hits(*store, variants) for store in stores]
    print('Scanned indexed SMILES bytes', [len(store[2]) for store in stores], flush=True)

    rows = []
    for mid, r in targets.items():
        identity = identities[mid]
        truth_atoms = Counter(atom.GetAtomicNum() for atom in Chem.MolFromSmiles(r.truth_smiles).GetAtoms()
                              if atom.GetAtomicNum() > 1)
        mass = float(chem.neutral_mass(np.asarray([r.precursor_mz]), np.asarray([r.adduct]))[0])
        windows = {}
        for store_id, (store_name,(masses,offsets,blob)) in enumerate(zip(STORE_NAMES,stores,strict=True)):
            low, high = np.searchsorted(masses, [mass*(1-1e-4),mass*(1+1e-4)])
            narrow_start, narrow_end = np.searchsorted(masses,[mass*(1-1e-5),mass*(1+1e-5)])
            # Check canonical connectivity even if a charge, tautomer or salt representation differs.
            off = np.asarray(offsets[low:high+1], dtype=np.int64)
            packed = bytes(blob[off[0]:off[-1]])
            hits, same_formula = [], 0
            parsed = 0
            for i, (start,end) in enumerate(zip(off[:-1]-off[0],off[1:]-off[0],strict=True), int(low)):
                text = packed[start:end].decode('ascii')
                molecule = Chem.MolFromSmiles(text)
                if molecule is None:
                    continue
                parsed += 1
                if rdMolDescriptors.CalcMolFormula(molecule)==identity['formula']:
                    same_formula += 1
                # A normalized fragment cannot contain atoms absent from this molecule.
                atoms = Counter(atom.GetAtomicNum() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1)
                if ('.' not in text and atoms != truth_atoms) or truth_atoms - atoms:
                    continue
                raw_key = Chem.MolToInchiKey(molecule)
                Chem.RemoveStereochemistry(molecule)
                canonical = tautomer.Canonicalize(molecule)
                if len(Chem.GetMolFrags(molecule)) == 1 and sum(a.GetFormalCharge() for a in molecule.GetAtoms()) == 0:
                    parent = canonical
                else:
                    parent = tautomer.Canonicalize(uncharger.uncharge(standard.FragmentParent(molecule)))
                matches = dict(raw_connectivity=raw_key[:14]==identity['raw'][:14],
                               tautomer=Chem.MolToInchiKey(canonical)[:14]==identity['tautomer'],
                               parent=Chem.MolToInchiKey(parent)[:14]==identity['parent'])
                if any(matches.values()):
                    hits.append(dict(smiles=text,index_ppm=float((masses[i]-mass)/mass*1e6),
                                     native=chem.score_key(text),**matches))
            windows[store_name] = dict(rows_10ppm=int(narrow_end-narrow_start), rows_100ppm=int(high-low),
                parsed_100ppm=parsed, same_formula_100ppm=same_formula, normalization_matches=hits,
                exact_strings_any_mass={s:global_hits[store_id][s.encode('ascii')] for s in identity['variants']
                                        if s.encode('ascii') in global_hits[store_id]})
        rows.append(dict(molecule_id=mid, accession=r.spectrum_id, label_key=r.candidate_id,
                         measured_mass=mass, record_mass=r.truth_neutral_mass, truth=identity, stores=windows))
        print(mid, 'native identity', identity['native'], 'matches',
              sum(len(v['normalization_matches']) for v in windows.values()), flush=True)
    report = dict(queries=len(rows), elapsed_seconds=round(time.monotonic()-started,1),
                  stores=[dict(store=name, rows=len(data[0]), smiles_bytes=len(data[2]))
                          for name,data in zip(STORE_NAMES,stores,strict=True)], per_query=rows)
    (OUTPUT/'negative_index_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()

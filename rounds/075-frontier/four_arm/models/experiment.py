"""Train on source/scaffold-disjoint real MS2, rank fixed public-v7 lists, then score offline."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch
from torch import nn
from torch.nn import functional as F


SEED = 241125
SOURCES = ('enveda-180', 'riken')
ADDUCTS = {'[M+H]+': 1.0072764666, '[M-H]-': -1.0072764666,
           '[M+Na]+': 22.989218, '[M+NH4]+': 18.033823, '[M+K]+': 38.963158}
ELEMENTS = ('C', 'H', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I')
ATOMIC_MASS = np.array([Chem.GetPeriodicTable().GetMostCommonIsotopeMass(x) for x in ELEMENTS])
TAUT = rdMolStandardize.TautomerEnumerator()
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, includeChirality=False)
TOKEN_VOCAB, TOKENS, PEAKS = 8192, 48, 32
ARMS = ('original', 'formula', 'structure', 'joint')
FORM_WEIGHT, STRUCT_WEIGHT = .45, .45
MAX_STRUCTURES, MAX_FORMULA_PAIRS, UPDATES = 8000, 1300, 600


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


@lru_cache(maxsize=100000)
def formula_counts(formula):
    members = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    if not members or ''.join(a+b for a, b in members) != formula or any(a not in ELEMENTS for a, _ in members):
        return None
    out = np.zeros(len(ELEMENTS), dtype=np.int16)
    for symbol, amount in members:
        out[ELEMENTS.index(symbol)] += int(amount or 1)
    return out


@lru_cache(maxsize=24000)
def molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        return None
    Chem.RemoveStereochemistry(mol)
    mol = TAUT.Canonicalize(mol)
    key = Chem.MolToInchiKey(mol).split('-')[0]
    if len(key) != 14:
        return None
    skeleton = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(skeleton)) if skeleton.GetNumAtoms() else ''
    raw = MORGAN.GetSparseCountFingerprint(mol).GetNonzeroElements()
    # Frequent local atom environments first, with deterministic tie and hash collisions removed.
    tokens = list(dict.fromkeys(i % TOKEN_VOCAB + 1 for i in sorted(raw, key=lambda i: (-raw[i], i))))[:TOKENS]
    padded = np.zeros(TOKENS, dtype=np.int64)
    padded[:len(tokens)] = tokens
    heavy = Counter(a.GetAtomicNum() for a in mol.GetAtoms() if a.GetAtomicNum()>1)
    signature = ','.join(f'{z}:{heavy[z]}' for z in sorted(heavy))
    return key, scaffold, rdMolDescriptors.CalcMolFormula(mol), padded, signature


def scan_features(row):
    adduct = row.adduct
    if adduct not in ADDUCTS or not np.isfinite(row.precursor_mz):
        return None
    prec = float(row.precursor_mz)
    mz = np.asarray(row.ms2_mzs, dtype=float)
    it = np.asarray(row.ms2_normalized_intensities, dtype=float)
    good = np.isfinite(mz) & np.isfinite(it) & (mz > 0) & (mz < prec - .5) & (it > 0)
    if not good.any():
        return None
    mz, it = mz[good], it[good]
    keep = np.flatnonzero(it >= it.max() * .005)
    take = keep[np.argsort(-it[keep], kind='stable')[:PEAKS]]
    if len(take) < 3:
        return None
    mz, weight = mz[take], np.sqrt(it[take] / it[take].max())
    ev = row.collision_energy_ev
    valid_ev = [float(x) for x in ev if x is not None and np.isfinite(x)] if isinstance(ev, (np.ndarray, list, tuple)) else []
    unit = str(getattr(row, 'collision_energy_orig_units', '')).lower()
    measured_ev = valid_ev[0] if len(valid_ev) == 1 and unit == 'ev' else None
    instr = str(row.instrument_type or '').lower()
    family = 'orbitrap' if 'orbitrap' in instr or 'qft' in instr else ('tims' if 'tims' in instr else ('tof' if 'tof' in instr else 'other'))
    ce_group = int(round(measured_ev / 10)) if measured_ev is not None else -1
    peak = np.zeros((PEAKS, 19), dtype=np.float32)
    loss = prec - mz
    base = [mz / 500, loss / 500, mz / prec, weight, np.full_like(mz, prec/700)]
    for axis in (mz, loss):
        for scale in (1., 20.):
            base.extend([np.sin(2*np.pi*axis/scale), np.cos(2*np.pi*axis/scale)])
    base.extend(np.full_like(mz, float(adduct == a)) for a in ADDUCTS)
    base.append(np.full_like(mz, min(measured_ev, 150)/100 if measured_ev is not None else -1.))
    peak[:len(mz)] = np.stack(base, axis=1)
    return dict(adduct=adduct, prec=prec, mz=mz, w=weight, peak=peak, family=family, ce=ce_group)


SPEC_COLUMNS = ['ingest_lib', 'inchikey14', 'normalized_smiles', 'molecular_formula',
                'adduct', 'precursor_mz', 'instrument_type', 'collision_energy_ev',
                'collision_energy_orig_units', 'ms2_mzs', 'ms2_normalized_intensities']


def filtered_batches(file, fields, sources, raw_keys=None):
    allowed_sources = pa.array(sources, type=pa.string())
    allowed_adducts = pa.array(list(ADDUCTS), type=pa.string())
    for batch in pq.ParquetFile(file).iter_batches(columns=fields, batch_size=16384):
        ok = pc.and_(pc.is_in(batch.column('ingest_lib'), value_set=allowed_sources),
                     pc.is_in(batch.column('adduct'), value_set=allowed_adducts))
        if raw_keys is not None:
            ok = pc.and_(ok, pc.is_in(batch.column('inchikey14'), value_set=raw_keys))
        if pc.any(ok).as_py():
            yield batch.filter(ok).to_pandas()


def train_cohort(file, blocked, seed):
    sources = sorted(set(SOURCES) - {s.lower() for s in blocked['blocked_sources']})
    if not sources:
        raise ValueError('All permitted training sources are held out.')
    fields = ['ingest_lib','adduct','inchikey14','normalized_smiles','molecular_formula']
    distinct = set()
    for tab in filtered_batches(file, fields, sources):
        distinct.update(tab[['inchikey14','normalized_smiles','molecular_formula']].dropna().drop_duplicates().itertuples(index=False, name=None))
    by_formula = defaultdict(list)
    for raw, smi, f in sorted(distinct):
        if len(raw) == 14 and f and raw not in blocked['blocked_keys'] and formula_counts(f) is not None:
            by_formula[f].append((raw, smi))
    formulas = sorted(by_formula, key=lambda f: (formula_counts(f) @ ATOMIC_MASS, f))
    mass = np.array([formula_counts(f) @ ATOMIC_MASS for f in formulas])
    rng = np.random.default_rng(seed)
    # Sample real isobaric formula competitors and multiple identities per formula so both losses have support.
    ordered = []
    seen = set()
    def add(f):
        if f not in seen:
            seen.add(f)
            records = by_formula[f]
            ordered.extend(records[i] for i in rng.permutation(len(records))[:4])
    n_pairs = 0
    for i in rng.permutation(len(formulas)):
        lo, hi = np.searchsorted(mass, [mass[i]*(1-1e-5), mass[i]*(1+1e-5)], side='left')
        others = [j for j in range(lo, hi) if j != i]
        if others:
            add(formulas[i])
            add(formulas[rng.choice(others)])
            n_pairs += 1
        if n_pairs >= MAX_FORMULA_PAIRS:
            break
    for i in rng.permutation(len(formulas)):
        if len(ordered) >= 2*MAX_STRUCTURES:
            break
        if len(by_formula[formulas[i]]) > 1:
            add(formulas[i])
    excluded_keys, excluded_scaffolds = set(blocked['blocked_keys']), set(blocked['blocked_scaffolds']) - {''}
    exclude_acyclic = bool(blocked.get('block_acyclic', True))
    raw_to_smi, chemical, skipped = {}, {}, Counter()
    for raw,smi in ordered:
        if len(chemical) >= MAX_STRUCTURES:
            break
        chem = molecule(smi)
        if (not chem or chem[0] in excluded_keys or chem[1] in excluded_scaffolds
                or (exclude_acyclic and not chem[1]) or formula_counts(chem[2]) is None):
            skipped['excluded_or_unsupported'] += 1
            continue
        if raw in raw_to_smi:
            continue
        chemical[chem[0]] = chem
        raw_to_smi[raw] = smi
    raw_keys = pa.array(sorted(raw_to_smi))
    retained, view_counts = defaultdict(dict), Counter()
    for batch in filtered_batches(file, SPEC_COLUMNS, sources, raw_keys):
        for r in batch.itertuples(index=False):
            # Check the actual structural representation on every input row before using its spectrum.
            if raw_to_smi.get(r.inchikey14) != r.normalized_smiles:
                continue
            chem = molecule(r.normalized_smiles)
            key = chem[0]
            destination = retained[key]
            if view_counts[key, r.ingest_lib, r.adduct] >= 3:
                continue
            scan = scan_features(r)
            if scan is None:
                continue
            group = (r.ingest_lib, r.adduct, scan['family'], scan['ce'])
            if group not in destination:
                destination[group] = scan
                view_counts[key, r.ingest_lib, r.adduct] += 1
    chemical = {k:v for k,v in chemical.items() if retained.get(k)}
    assert not (set(chemical) & excluded_keys)
    assert not ({c[1] for c in chemical.values() if c[1]} & excluded_scaffolds)
    info = dict(source_allowlist=sources, actual_sources=sorted({g[0] for d in retained.values() for g in d}),
                rejected_counts=skipped, identities=len(chemical), spectra=sum(map(len,retained.values())),
                nonempty_generic_scaffolds=len({c[1] for c in chemical.values() if c[1]}),
                zero_scaffold_identities=sum(not c[1] for c in chemical.values()), exclude_all_acyclic=exclude_acyclic,
                excluded_source_labels=sorted(blocked['blocked_sources']), excluded_scaffold_count=len(excluded_scaffolds),
                excluded_key_count=len(excluded_keys), source_overlap=0, score_key_overlap=0, nonempty_scaffold_overlap=0)
    return chemical, retained, info


def member_digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class FragmentLibrary:
    """Small neutral subformula mass catalogue, used for fragment and neutral-loss evidence."""
    def __init__(self):
        blocks = []
        for c in range(41):
            for n in range(7):
                oh = np.array(list(itertools.product(range(13), range(min(76,2*c+n+6)))),dtype=np.int16)
                for s,p in itertools.product(range(2), repeat=2):
                    for halogen in (None,6,7,8,9):
                        a = np.zeros((len(oh),len(ELEMENTS)),dtype=np.int16)
                        a[:,0]=c; a[:,1]=oh[:,1]; a[:,2]=n; a[:,3]=oh[:,0]; a[:,4]=s; a[:,5]=p
                        if halogen is not None:
                            a[:,halogen] = 1
                        blocks.append(a)
        rows = np.concatenate(blocks)
        mass = np.sum(rows * ATOMIC_MASS, axis=1)
        order = np.flatnonzero((mass > 1) & (mass <= 700))
        order = order[np.argsort(mass[order],kind='stable')]
        self.mass, self.rows = mass[order], rows[order].astype(np.int8)

    def annotation(self, masses):
        out = []
        for mass in masses:
            if mass < 1 or mass > 700:
                out.append(self.rows[:0])
                continue
            tol = max(.002, mass*1e-5)
            i,j = np.searchsorted(self.mass, [mass-tol, mass+tol])
            out.append(self.rows[i:j])
        return out

    def observe(self, scan):
        adduct = scan['adduct']
        proton = ADDUCTS['[M+H]+']
        mz = scan['mz']
        primary = self.annotation(mz-(proton if adduct!='[M-H]-' else -proton))
        if adduct in ('[M+Na]+','[M+K]+'):
            alt = self.annotation(mz-ADDUCTS[adduct])
            primary = [np.concatenate((a,b)) for a,b in zip(primary,alt)]
        return primary, self.annotation(scan['prec']-mz)


def formula_features(counts, scan, annotation):
    weights = scan['w'] / scan['w'].sum()
    out = []
    # One-H slack allows radical/odd-electron fragments. Other atoms cannot exceed the candidate.
    limits = counts.copy()
    limits[1] += 1
    for rows in annotation:
        hits = np.array([np.all(f<=limits,axis=1).sum() if len(f) else 0 for f in rows])
        present = hits>0
        out.extend([present.mean(), weights @ present, present[:3].mean(),
                    weights @ (1/np.maximum(hits,1)*present), np.log1p(hits).mean()])
    mass = counts @ ATOMIC_MASS
    obs = scan['prec'] - ADDUCTS[scan['adduct']]
    ppm = (obs-mass)*1e6 / mass
    out.extend([min(abs(ppm),40)/10, min(abs(ppm),10)/10, len(weights)/PEAKS,
                float(scan['adduct']=='[M-H]-')])
    return np.asarray(out, dtype=np.float32)


class FormulaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(14,32), nn.SiLU(), nn.Linear(32,1))

    def forward(self,x):
        return self.score(x).squeeze(-1)


class PairModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.peak = nn.Sequential(nn.Linear(19,96),nn.SiLU(),nn.Linear(96,48))
        self.local = nn.Embedding(TOKEN_VOCAB+1,48,padding_idx=0)

    def forward(self, peaks, tokens):
        q = F.normalize(self.peak(peaks),dim=-1)
        m = F.normalize(self.local(tokens),dim=-1)
        sim = torch.bmm(q,m.transpose(1,2))
        support = sim.masked_fill(tokens[:,None,:].eq(0),-2.).amax(dim=-1)
        w = peaks[:,:,3]
        return 12*(support*w).sum(dim=1)/w.sum(dim=1).clamp_min(1e-6)


def training_examples(chemical, scans, fragments):
    formulas = sorted({c[2] for c in chemical.values()}, key=lambda f: (formula_counts(f) @ ATOMIC_MASS,f))
    masses = np.array([formula_counts(f) @ ATOMIC_MASS for f in formulas])
    token_rows, peak_rows, eligible = [], [], defaultdict(list)
    positive_features, negative_features = [], []
    rng = np.random.default_rng(SEED+1)
    keys = sorted(chemical)
    for ci,key in enumerate(keys):
        chem = chemical[key]
        token_rows.append(chem[3])
        formula = chem[2]
        counts = formula_counts(formula)
        for condition, scan in sorted(scans[key].items()):
            observed = scan['prec'] - ADDUCTS[scan['adduct']]
            if abs(observed-counts@ATOMIC_MASS) > observed*2e-5:
                continue
            si = len(peak_rows)
            peak_rows.append(scan['peak'])
            eligible[(formula,condition)].append((ci,si))
            lo, hi = np.searchsorted(masses,[observed*(1-1e-5), observed*(1+1e-5)])
            negatives = [i for i in range(lo,hi) if formulas[i]!=formula]
            if not negatives or abs(observed-counts@ATOMIC_MASS)>observed*1e-5:
                continue
            obs = fragments.observe(scan)
            pos = formula_features(counts,scan,obs)
            for i in rng.permutation(negatives)[:2]:
                positive_features.append(pos)
                negative_features.append(formula_features(formula_counts(formulas[i]),scan,obs))
    groups = [g for g in eligible.values() if len(g)>=2]
    if not groups or not positive_features:
        raise ValueError('No real matched structure or near-isobaric formula contrasts.')
    return (np.array(token_rows), np.array(peak_rows), groups,
            np.stack(positive_features), np.stack(negative_features), keys)


def fit(args):
    started = time.monotonic()
    args.out.mkdir(parents=True,exist_ok=True)
    holdout = json.loads(args.exclude.read_text())
    chem, scans, info = train_cohort(args.train, holdout, SEED)
    print('Retained real training structures and spectra:',len(chem),info['spectra'],flush=True)
    fragments = FragmentLibrary()
    tokens, peaks, groups, positives, negatives, keys = training_examples(chem, scans, fragments)
    rng = np.random.default_rng(SEED+2)
    # Choose two different measured structures with the same formula, source, instrument, adduct and CE group.
    # Draws are precomputed on CPU. The four pair scores and both optimizers run on the requested accelerator.
    batch_size=96
    draws=np.empty((UPDATES,batch_size,4),dtype=np.int64)
    for dst in draws:
        for row, g_idx in zip(dst, rng.integers(len(groups),size=batch_size)):
            g=groups[g_idx]
            a=rng.integers(len(g)); b=(a+rng.integers(1,len(g)))%len(g)
            row[:2]=g[a]; row[2:]=g[b]
    fm, sm = FormulaNet().to(args.device), PairModel().to(args.device)
    fo,so = torch.optim.AdamW(fm.parameters(),lr=.002), torch.optim.AdamW(sm.parameters(),lr=.002)
    arrays=[torch.as_tensor(a,device=args.device) for a in (tokens,peaks,positives,negatives,draws)]
    mt,pt,pf,nf,samples=arrays
    for indices in samples:
        c1,q1,c2,q2=indices.T
        q=torch.cat([q1,q1,q2,q2]); c=torch.cat([c1,c2,c2,c1])
        pos1,neg1,pos2,neg2=sm(pt[q],mt[c]).chunk(4)
        ls=(F.softplus(neg1-pos1)+F.softplus(neg2-pos2)).mean()/2
        idx=torch.randint(len(pf),(batch_size,),device=args.device)
        lf=F.softplus(fm(nf[idx])-fm(pf[idx])).mean()
        fo.zero_grad(); so.zero_grad(); (ls+lf).backward(); fo.step(); so.step()
    torch.save({'formula':fm.cpu().state_dict(),'structure':sm.cpu().state_dict()},args.out/'checkpoint.pt')
    pd.DataFrame([dict(key=k,generic_scaffold=chem[k][1],formula=chem[k][2],
                       sources=';'.join(sorted({g[0] for g in scans[k]}))) for k in keys]).to_parquet(args.out/'training_identities.parquet',index=False)
    info.update(seed=SEED, train_file_sha256=digest(args.train), holdout_sha256=digest(args.exclude),
                formula_examples=len(positives), matched_structure_groups=len(groups), updates=UPDATES,batch_size=batch_size,
                device=args.device, device_name=torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'CPU',
                rdkit=rdBase.rdkitVersion, elapsed_seconds=round(time.monotonic()-started,2), checkpoint_sha256=digest(args.out/'checkpoint.pt'))
    info['excluded_membership'] = {field: sorted(member_digest(v) for v in set(holdout[field]))
                                   for field in ('blocked_keys','blocked_scaffolds','blocked_sources')}
    write_json(args.out/'fit.json',info)
    print('Fitted',info,flush=True)


def parse_list(text):
    return [s.strip() for s in str(text).split(';') if s.strip()]


def pool_spectra(scores, scans):
    # Multiple collision conditions within one ion type form one view, then observed adducts vote equally.
    buckets=defaultdict(list)
    for score,scan in zip(scores,scans):
        buckets[scan['adduct']].append(score)
    return np.mean([np.mean(v,axis=0) for v in buckets.values()],axis=0)


def fixed_rank(smiles, raw_scans, fragments, fm, sm, device):
    scans=[q for q in raw_scans if q is not None]
    n=len(smiles)
    original=np.arange(n)
    chemicals=[molecule(s) for s in smiles]
    valid=[i for i,c in enumerate(chemicals) if c and formula_counts(c[2]) is not None]
    forms=list(dict.fromkeys(chemicals[i][2] for i in valid))
    ranked={a:original.copy() for a in ARMS}
    if not scans or not forms:
        return ranked, chemicals, {}, np.full(n,np.nan)
    form_scores,struct_scores=[],[]
    mt=torch.as_tensor(np.stack([chemicals[i][3] for i in valid]),device=device)
    with torch.inference_mode():
        for scan in scans:
            obs=fragments.observe(scan)
            x=np.stack([formula_features(formula_counts(f),scan,obs) for f in forms])
            form_scores.append(fm(torch.tensor(x,device=device)).cpu().numpy())
            st=torch.as_tensor(scan['peak'],device=device).unsqueeze(0).expand(len(mt),-1,-1)
            struct_scores.append(sm(st,mt).cpu().numpy())
    fscore=dict(zip(forms,pool_spectra(form_scores,scans)))
    sscore=np.full(n,np.nan)
    sscore[valid]=pool_spectra(struct_scores,scans)
    # Keep invalid/unsupported slots fixed. Formula is a common shift for each group;
    # the original relative order within every formula is therefore preserved in this arm.
    adjusted=[-np.log(i+1)+FORM_WEIGHT*fscore[chemicals[i][2]] for i in valid]
    order=np.argsort(-np.asarray(adjusted),kind='stable')
    ranked['formula'][valid]=np.asarray(valid)[order]
    # For a given formula, sort only its original members; the chosen arm sets the formula of each slot.
    for name,parent in [('structure','original'),('joint','formula')]:
        out=ranked[parent].copy()
        for f in forms:
            members=[i for i in valid if chemicals[i][2]==f]
            if len(members)<2:
                continue
            slots=[i for i,c in enumerate(out) if c in members]
            adj=-np.log1p(np.arange(len(members)))+STRUCT_WEIGHT*sscore[members]
            out[slots]=np.asarray(members)[np.argsort(-adj,kind='stable')]
        ranked[name]=out
    for arm in ranked:
        assert np.array_equal(np.sort(ranked[arm]),original)
    return ranked, chemicals, fscore, sscore


def rank(args):
    args.out.mkdir(parents=True,exist_ok=True)
    artifact=args.model/'checkpoint.pt'
    fit_info=json.loads((args.model/'fit.json').read_text())
    if digest(artifact)!=fit_info['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from fit receipt')
    model=torch.load(artifact,map_location=args.device,weights_only=True)
    fm,sm=FormulaNet().to(args.device).eval(), PairModel().to(args.device).eval()
    fm.load_state_dict(model['formula']); sm.load_state_dict(model['structure'])
    scans=pd.read_parquet(args.queries)
    base=pd.read_csv(args.baseline,dtype=str,keep_default_na=False)
    if not base.molecule_id.is_unique or set(base.molecule_id)!=set(scans.molecule_id.astype(str)):
        raise ValueError('Frozen baseline and measured query identities do not match.')
    fragments=FragmentLibrary()
    by_id={str(q):grp for q,grp in scans.groupby('molecule_id')}
    records=[]
    for row in base.itertuples(index=False):
        candidates=parse_list(row.smiles)
        if not 1<=len(candidates)<=25:
            raise ValueError(f'Unexpected candidate count for {row.molecule_id}')
        query=[scan_features(r) for r in by_id[row.molecule_id].itertuples(index=False)]
        ranks,chems,fscore,sscore=fixed_rank(candidates,query,fragments,fm,sm,args.device)
        for arm,order in ranks.items():
            for pos,idx in enumerate(order,1):
                ch=chems[idx]
                records.append(dict(molecule_id=row.molecule_id,arm=arm,rank=pos,original_rank=int(idx)+1,
                                    smiles=candidates[idx],score_key=ch[0] if ch else '',formula=ch[2] if ch else '',
                                    heavy_signature=ch[4] if ch else '',
                                    candidate_formula_score=fscore.get(ch[2],np.nan) if ch else np.nan,
                                    candidate_structure_score=sscore[idx], usable_scans=sum(s is not None for s in query)))
    output=args.out/'ranks.parquet'
    pd.DataFrame(records).to_parquet(output,index=False)
    write_json(args.out/'ranking.json',dict(arms=ARMS,formula_shift=FORM_WEIGHT,structure_shift=STRUCT_WEIGHT,
        baseline_sha256=digest(args.baseline),queries_sha256=digest(args.queries),checkpoint_sha256=digest(artifact),
        excluded_membership=fit_info['excluded_membership'],exclude_all_acyclic=fit_info['exclude_all_acyclic'],
        ranks_sha256=digest(output),molecules=len(base),input_spectra=len(scans),slots=len(records)//4,
        short_lists=int(base.smiles.map(lambda t:len(parse_list(t))<25).sum()),
        methodology='Formula arm stably sorts valid candidates by log(original reciprocal rank) plus a shared formula score. Structure arm reorders identities within each incumbent formula’s slots using within-formula log reciprocal rank plus local similarity. Joint uses formula-only slots and structure-only relative order. Unsupported molecular slots remain in place; scans are pooled within and across adducts.'))
    print('Locked four ranked arms',len(base),'questions',len(records)//4,'slots per arm',flush=True)


def metrics(rs):
    return dict(n=len(rs),mrr25=float(np.mean([1/v if v else 0 for v in rs])),
                top1=sum(v==1 for v in rs),top5=sum(0<v<=5 for v in rs),top25=sum(v>0 for v in rs))


def interval(data,arm,group):
    groups=[g[f'delta_rr_{arm}'].to_numpy() for _,g in data.groupby(group,sort=True)]
    if len(groups)<2:
        return None
    rng=np.random.default_rng(SEED+18)
    samples=rng.integers(len(groups),size=(4000,len(groups)))
    v=np.array([v.sum() for v in groups])[samples].sum(1)/np.array([len(v) for v in groups])[samples].sum(1)
    return np.quantile(v,[.025,.975]).tolist()


def score(args):
    args.out.mkdir(parents=True, exist_ok=True)
    receipt=json.loads((args.ranks/'ranking.json').read_text())
    path=args.ranks/'ranks.parquet'
    if digest(path)!=receipt['ranks_sha256']:
        raise ValueError('Rank lists changed since inference')
    df=pd.read_parquet(path)
    labels=pd.read_parquet(args.labels)
    labels['molecule_id']=labels.molecule_id.astype(str)
    if not labels.molecule_id.is_unique or set(labels.molecule_id)!=set(df.molecule_id):
        raise ValueError('Missing, duplicate, or unexpected evaluation label.')
    by_id={k:v for k,v in df.groupby('molecule_id',sort=True)}
    records=[]
    for r in labels.itertuples(index=False):
        chem=molecule(r.truth_smiles)
        if not chem or (hasattr(r,'truth_key') and chem[0]!=r.truth_key):
            raise ValueError(f'Evaluation identity mismatch {r.molecule_id}')
        source=str(r.source).strip()
        if not source or source.lower() in ('nan','none'):
            raise ValueError(f'Missing source group {r.molecule_id}')
        excluded=receipt['excluded_membership']
        if (member_digest(chem[0]) not in excluded['blocked_keys'] or member_digest(source) not in excluded['blocked_sources']
                or (chem[1] and member_digest(chem[1]) not in excluded['blocked_scaffolds'])
                or (not chem[1] and not receipt['exclude_all_acyclic'])):
            raise ValueError(f'Validation identity/source/scaffold was not blocked at training: {r.molecule_id}')
        item=dict(molecule_id=r.molecule_id,source=source,truth_key=chem[0],scaffold=chem[1],
                  challenges=getattr(r,'challenges',''),natural_origin=getattr(r,'natural_origin',''),
                  scaffold_group=chem[1] or 'ACYCLIC')
        group=by_id[r.molecule_id]
        base=group[group.arm=='original'].sort_values('rank')
        if list(base['rank'])!=list(base.original_rank):
            raise ValueError('Original rank changed')
        expected=base.smiles.tolist()
        item.update(slots=len(expected),usable_scans=int(base.usable_scans.iloc[0]))
        for arm in ARMS:
            ordered=group[group.arm==arm].sort_values('rank')
            if sorted(ordered.original_rank)!=list(range(1,len(base)+1)) or ordered.sort_values('original_rank').smiles.tolist()!=expected:
                raise ValueError(f'Changed candidate slots in {arm} {r.molecule_id}')
            matches=ordered[(ordered.score_key==chem[0]) & (ordered.heavy_signature==chem[4])]
            n=int(matches['rank'].min()) if len(matches) else 0
            item[arm]=n
            item['head_'+arm]=ordered.score_key.iloc[0]
            item['rr_'+arm]=float(1/n) if n else 0.
            item['delta_rr_'+arm]=item['rr_'+arm]-item['rr_original']
        records.append(item)
    report=pd.DataFrame(records).sort_values('molecule_id')
    report.to_csv(args.out/'per_query.csv',index=False)
    comparison={}
    for arm in ARMS[1:]:
        comp={}
        for k in (1,5):
            new=report[arm].between(1,k); old=report.original.between(1,k)
            comp[f'gained_top{k}']=report.loc[new & ~old,'molecule_id'].tolist()
            comp[f'lost_top{k}']=report.loc[old & ~new,'molecule_id'].tolist()
        comp.update(delta_mrr25=float(report[f'delta_rr_{arm}'].mean()),delta_rr_positive_sum=float(report[f'delta_rr_{arm}'].clip(lower=0).sum()),
                    delta_rr_negative_sum=float(report[f'delta_rr_{arm}'].clip(upper=0).sum()),
                    bootstrap95_by_scaffold=interval(report,arm,'scaffold_group'),bootstrap95_by_source=interval(report,arm,'source'))
        comparison[arm]=comp
    groups=[]
    for col in ('source','scaffold_group'):
        for k,g in report.groupby(col,sort=True):
            for arm in ARMS:
                groups.append(dict(grouping=col,group=k,arm=arm,**metrics(g[arm]),delta_mrr25=float(g[f'delta_rr_{arm}'].mean())))
    pd.DataFrame(groups).to_csv(args.out/'group_metrics.csv',index=False)
    result=dict(arms={a:metrics(report[a]) for a in ARMS},paired=comparison,source_groups=int(report.source.nunique()),
                generic_scaffold_groups=int(report.scaffold_group.nunique()),eligible_questions=int(report.usable_scans.gt(0).sum()),
                ranks_sha256=digest(path),labels_sha256=digest(args.labels),
                notes='Rank 0 denotes a miss. All questions remain in the denominator. Acyclic structures form one shared group for uncertainty; see holdout manifest for source/family exposure.')
    write_json(args.out/'metrics.json',result)
    print('Paired natural-product metrics',json.dumps(result['arms']),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='stage',required=True)
    for stage,fields in [('fit',('train','exclude')),('rank',('model','queries','baseline')),('score',('ranks','labels'))]:
        p=sub.add_parser(stage)
        for f in (*fields,'out'):
            p.add_argument('--'+f,type=Path,required=True)
        if stage!='score':
            p.add_argument('--device',default='cuda')
    args=parser.parse_args()
    if rdBase.rdkitVersion!='2026.03.3':
        raise RuntimeError('Use the competition RDKit 2026.03.3 for matching, splits and candidate features.')
    torch.manual_seed(SEED)
    torch.set_num_threads(3)
    RDLogger.DisableLog('rdApp.*')
    if getattr(args,'device','').startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Kaggle CUDA is unavailable')
    {'fit':fit,'rank':rank,'score':score}[args.stage](args)


if __name__=='__main__':
    main()

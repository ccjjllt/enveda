"""Score the same positive-mode spectra and candidate structures with FLARE and MIST."""

import argparse
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ADDUCT_ALIASES = {'M+H': '[M+H]+', '[M+NH4]+': '[M+H3N+H]+',
                 '[M+H-H2O]+': '[M-H2O+H]+', '[M-2H2O+H]+': '[M-H4O2+H]+'}
SCORE_COLUMNS = ['molecule_id', 'query_id', 'candidate_id', 'model', 'score']


def mounted(name):
    paths = sorted(Path('/kaggle/input').rglob(name))
    if len(paths) != 1:
        raise FileNotFoundError(f'Expected one mounted {name}, found {paths}')
    return paths[0]


def extract(path, source_dir):
    source_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, 'r:gz') as archive:
        archive.extractall(source_dir, filter='data')


def import_rdkit():
    global Chem
    try:
        installed = version('rdkit')
    except PackageNotFoundError:
        installed = None
    if installed not in ('2026.3.3', '2026.03.3'):
        abi = f'cp{sys.version_info.major}{sys.version_info.minor}'
        wheels = sorted(Path('/kaggle/input').rglob(f'rdkit-2026.3.3-{abi}-*.whl'))
        if not wheels:
            wheels = sorted(Path('/kaggle/input').rglob(f'rdkit-2026.03.3-{abi}-*.whl'))
        if not wheels:
            raise FileNotFoundError(f'RDKit 2026.03.3 wheel for {abi} is missing')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-index',
                        '--no-deps', str(wheels[0])], check=True)
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog('rdApp.error')


def processed_peaks(query):
    mz = np.asarray(query.mz, dtype=np.float64)
    intensity = np.asarray(query.intensity, dtype=np.float64)
    valid = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
    valid &= mz <= float(query.precursor_mz) + 1.0
    mz, intensity = mz[valid], intensity[valid]
    if not len(mz):
        return np.empty((0, 2), dtype=np.float64)
    # Apply the releases' relative-intensity and sqrt transforms to one observed spectrum.
    order = np.argsort(-intensity, kind='stable')
    mz, intensity = mz[order], intensity[order]
    _, first = np.unique(np.round(mz, 4), return_index=True)
    selected = np.sort(first)[:60]
    peaks = np.column_stack([mz[selected], np.sqrt(intensity[selected] / intensity[0])])
    return peaks[peaks[:, 1] > 0.001]


@lru_cache(maxsize=4096)
def formula_properties(formula):
    from mist.utils import chem_utils

    try:
        counts = chem_utils.formula_to_dense(formula)
    except ValueError:
        return None
    # FLARE's released assignment expands the full subformula product.
    product = 1
    for count in counts:
        product *= int(count) + 1
    return float(counts @ chem_utils.VALID_MONO_MASSES), product


def eligible_formulas(query, formulas):
    from mist.utils.chem_utils import ion_to_mass

    adduct = str(query.adduct)
    ion = ADDUCT_ALIASES.get(adduct, adduct)
    if ion not in ion_to_mass or str(query.polarity).lower() not in ('positive', 'pos', '+'):
        return ion, []
    precursor = float(query.precursor_mz)
    if not np.isfinite(precursor) or not 0 < precursor <= 1000:
        return ion, []
    hypotheses = []
    for formula in formulas:
        properties = formula_properties(formula)
        if properties is None:
            continue
        mass, product = properties
        ppm = abs(mass + ion_to_mass[ion] - precursor) * 1e6 / precursor
        if ppm <= 20.0 and product <= 200_000:
            hypotheses.append((ppm, formula))
    return ion, hypotheses


def formula_hypothesis(query, candidates, peaks):
    from mist.data.featurizers import assign_subformulae

    # For all structures and both models, use a single formula selected from the
    # candidate pool by fragment coverage and precursor error alone.
    ion, hypotheses = eligible_formulas(query, candidates.formula.dropna().astype(str).unique())
    output = {'mz': peaks[:, 0], 'ms2_inten': peaks[:, 1]}
    fragment_weight = float(peaks[:, 1].sum())
    best = None
    for ppm, formula in sorted(hypotheses)[:8]:
        assigned = assign_subformulae({'cand_form': formula, 'cand_ion': ion, 'output_tbl': output})
        table = assigned['output_tbl']
        if table is None:
            continue
        explained = sum(table['ms2_inten']) / fragment_weight
        objective = explained - 0.20 * (ppm / 20.0) ** 2
        key = (-objective, ppm, formula)
        if best is None or key < best[0]:
            best = key, formula, ion, table
    return best


@torch.inference_mode()
def score_mist(checkpoint, work, device):
    from mist.data.atlas import morgan4096_packed, unpack_fingerprints
    from mist.models.scaled_mist import ScaledMIST
    from mist.utils import chem_utils

    release = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = release['config']
    if config['data']['dataset_name'] != 'msg_all':
        raise ValueError('Expected the authors’ MassSpecGym checkpoint')
    model = ScaledMIST(**config['model'])
    model.load_state_dict(release['ema_model'], strict=True)
    model.to(device).eval()
    del release

    rows = []
    for candidates, spectra in work:
        fingerprints = np.stack([morgan4096_packed(mol) for mol in candidates.mol])
        fingerprints = torch.as_tensor(unpack_fingerprints(fingerprints), device=device)
        fingerprints = F.normalize(fingerprints, dim=-1)
        for query, _, hypothesis in spectra:
            if hypothesis is None:
                continue
            _, formula, ion, assigned = hypothesis
            selected = np.argsort(-np.asarray(assigned['ms2_inten']), kind='stable')[:10]
            forms = [assigned['formula'][i] for i in selected] + [formula]
            form_vec = np.stack([chem_utils.formula_to_dense(f) for f in forms])
            intensity = [assigned['ms2_inten'][i] for i in selected] + [1.0]
            instrument = str(getattr(query, 'instrument_type', 'Unknown'))
            instrument = {'orbitrap': 'Orbitrap', 'qtof': 'QTOF'}.get(instrument.lower(), instrument)
            inputs = {
                'form_vec': torch.tensor(form_vec[None, :], dtype=torch.float32, device=device),
                'ion_vec': torch.full((1, len(forms)), chem_utils.get_ion_idx(ion), device=device),
                'intens': torch.tensor([intensity], dtype=torch.float32, device=device),
                'types': torch.tensor([[0] * len(selected) + [3]], device=device),
                'num_peaks': torch.tensor([len(forms)], device=device),
                'instruments': torch.tensor([chem_utils.get_instr_idx(instrument)], device=device),
            }
            probabilities, _ = model(inputs)
            scores = (fingerprints @ F.normalize(probabilities.float(), dim=-1).T).flatten()
            rows.extend(score_rows(query, candidates, 'mist_msg', scores))
    return rows


def load_flare(assets, source_dir, device):
    wheels = sorted(assets.rglob('torch_geometric-*.whl'))
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-index', '--no-deps',
                    '--target', '/tmp/enveda-flare-deps',
                    *map(str, wheels)], check=True)
    sys.path.insert(0, '/tmp/enveda-flare-deps')
    # Load the authors' network, formula assignment and scoring modules without
    # initializing their training, visualization or distributed graph wrappers.
    for name in ('flare.models', 'flare.utils', 'flare.subformula_assign.utils'):
        namespace = ModuleType(name)
        namespace.__path__ = [str(source_dir.joinpath(*name.split('.')))]
        sys.modules[name] = namespace
    from flare.models.spec_encoder import SpecFormulaTransformer

    release = torch.load(assets / 'flare.ckpt', map_location='cpu', weights_only=False)
    params = dict(release['hyper_parameters'])
    params['element_list'] = [element for element in params['element_list'] if element is not None]
    if (params['spectra_view'], params['model'], params['gnn_type'], params['atom_feature']) != (
            'SpecFormula', 'filipContrastive', 'gcn', 'full'):
        raise ValueError('Expected released FLARE formula-token and full atom GCN configuration')
    spectral_model = SpecFormulaTransformer(SimpleNamespace(**params))
    spectral_prefix = 'spec_enc_model.'
    state = release['state_dict']
    spectral_model.load_state_dict({key.removeprefix(spectral_prefix): value
                                    for key, value in state.items() if key.startswith(spectral_prefix)}, strict=True)
    spectral_model.to(device).eval()
    graph_prefix = 'mol_enc_model.GNN.gnn_layers.'
    layers = []
    for i in range(len(params['gnn_channels'])):
        prefix = f'{graph_prefix}{i}.'
        layers.append({key.removeprefix(prefix): value.to(device)
                       for key, value in state.items() if key.startswith(prefix)})
    required = {'graph_conv.weight', 'graph_conv.bias', 'res_connection.weight',
                'res_connection.bias', 'bn_layer.weight', 'bn_layer.bias',
                'bn_layer.running_mean', 'bn_layer.running_var'}
    if any(not required.issubset(layer) for layer in layers):
        raise ValueError(f'Unexpected FLARE GCN state: {[list(layer) for layer in layers]}')
    print('FLARE published checkpoint training inputs:', params.get('dataset_pth'),
          params.get('split_pth'), params.get('formula_source'), flush=True)
    return spectral_model, layers, params


def atom_features(atom, elements):
    """FLARE MolToGraph full feature order using DGLLife 0.3.2 vocabularies."""
    def hot(value, choices):
        return [value == option for option in choices]

    bond = Chem.rdchem.BondType
    bond_types = {edge.GetBondType() for edge in atom.GetBonds()}
    hybrid = Chem.rdchem.HybridizationType
    chiral = Chem.rdchem.ChiralType
    symbol = atom.GetSymbol()
    return [atom.GetMass() * 0.01,
            *hot(symbol, elements), symbol not in elements,
            *(option in bond_types for option in (bond.SINGLE, bond.DOUBLE, bond.TRIPLE, bond.AROMATIC)),
            *hot(atom.GetDegree(), range(11)), *hot(atom.GetTotalDegree(), range(6)),
            *hot(atom.GetValence(Chem.ValenceType.EXPLICIT), range(1, 7)),
            *hot(atom.GetValence(Chem.ValenceType.IMPLICIT), range(7)),
            *hot(atom.GetHybridization(), (hybrid.SP, hybrid.SP2, hybrid.SP3, hybrid.SP3D, hybrid.SP3D2)),
            *hot(atom.GetTotalNumHs(), range(5)), *hot(atom.GetFormalCharge(), range(-2, 3)),
            *hot(atom.GetNumRadicalElectrons(), range(5)), *hot(atom.GetIsAromatic(), (False, True)),
            *hot(atom.IsInRing(), (False, True)),
            *hot(atom.GetChiralTag(), (chiral.CHI_UNSPECIFIED, chiral.CHI_TETRAHEDRAL_CW,
                                      chiral.CHI_TETRAHEDRAL_CCW, chiral.CHI_OTHER))]


def flare_atoms(molecules, layers, elements, device):
    """Apply the released GCN weights with PyTorch in the absence of a DGL wheel."""
    features, sources, targets, counts = [], [], [], []
    for molecule in molecules:
        offset = len(features)
        counts.append(molecule.GetNumAtoms())
        for bond in molecule.GetBonds():
            left = offset + bond.GetBeginAtomIdx()
            right = offset + bond.GetEndAtomIdx()
            sources.extend((left, right))
            targets.extend((right, left))
        self_loops = range(offset, offset + counts[-1])
        sources.extend(self_loops)
        targets.extend(self_loops)
        features.extend(atom_features(atom, elements) for atom in molecule.GetAtoms())

    features = torch.tensor(features, dtype=torch.float32, device=device)
    source = torch.tensor(sources, dtype=torch.long, device=device)
    target = torch.tensor(targets, dtype=torch.long, device=device)
    for layer in layers:
        # DGLLife default GraphConv is an unnormalized neighbor sum with self-loops.
        weight = layer['graph_conv.weight']
        if weight.shape[0] > weight.shape[1]:
            projected = features @ weight
            messages = torch.zeros_like(projected).index_add_(0, target, projected[source])
        else:
            neighborhood = torch.zeros_like(features).index_add_(0, target, features[source])
            messages = neighborhood @ weight
        messages = (messages + layer['graph_conv.bias']).relu()
        residual = F.linear(features, layer['res_connection.weight'], layer['res_connection.bias']).relu()
        features = F.batch_norm(messages + residual, layer['bn_layer.running_mean'],
                                layer['bn_layer.running_var'], layer['bn_layer.weight'],
                                layer['bn_layer.bias'], training=False)
    padded = torch.nn.utils.rnn.pad_sequence(features.split(counts), batch_first=True)
    mask = torch.arange(padded.shape[1], device=device)[None, :] < torch.tensor(counts, device=device)[:, None]
    return padded, mask


def flare_spectrum(query, peaks, formula, ion, params):
    from flare.definitions import MSGYM_FORMULA_VECTOR_NORM, PRECURSOR_INTENSITY
    from flare.subformula_assign.utils.chem_utils import element_to_ind, formula_to_dense
    from flare.subformula_assign.utils.spectra_utils import assign_subforms

    table = assign_subforms(formula, peaks.copy(), ion, mass_diff_thresh=20)['output_tbl']
    if table is None:
        return None
    mz = np.asarray(table['mz'], dtype=np.float64)
    intensity = np.asarray(table['ms2_inten'], dtype=np.float64)
    formulas = np.asarray(table['formula'], dtype=str)
    precursor = formulas == formula
    if precursor.any():
        intensity[precursor] = PRECURSOR_INTENSITY
    else:
        mz = np.append(mz, float(query.precursor_mz))
        intensity = np.append(intensity, PRECURSOR_INTENSITY)
        formulas = np.append(formulas, formula)
    order = np.argsort(mz)
    order = order[mz[order] <= params['max_mz']]
    if not len(order):
        return None
    elements = [element_to_ind[e] for e in params['element_list']]
    vector = np.stack([formula_to_dense(formulas[i])[elements] for i in order])
    vector = vector / MSGYM_FORMULA_VECTOR_NORM
    if params['add_intensities']:
        vector = np.column_stack([vector, intensity[order]])
    return torch.as_tensor(vector, dtype=torch.float32)


def contrasting_peak_similarity(spectra, atoms, peak_mask, atom_mask):
    """Weight peaks by their matching-score range within one isomer group."""
    pair = torch.bmm(F.normalize(spectra, dim=-1), F.normalize(atoms, dim=-1).transpose(1, 2))
    valid = peak_mask[:, :, None] & atom_mask[:, None, :]
    pair = pair.masked_fill(~valid, -torch.inf)
    per_peak = pair.amax(dim=2).masked_fill(~peak_mask, 0)
    per_atom = pair.amax(dim=1).masked_fill(~atom_mask, 0)
    weights = per_peak.amax(dim=0) - per_peak.amin(dim=0)
    weights = weights * peak_mask[0]
    weights = torch.where(weights.sum() > 0, weights, peak_mask[0].float())
    spectrum_similarity = (per_peak * weights).sum(dim=1) / weights.sum().clamp_min(1e-8)
    atom_similarity = per_atom.sum(dim=1) / atom_mask.sum(dim=1).clamp_min(1)
    return 0.5 * (spectrum_similarity + atom_similarity)


@torch.inference_mode()
def score_flare(assets, source_dir, work, device, model_name):
    spectral_model, layers, params = load_flare(assets, source_dir, device)
    from flare.utils.general import filip_similarity_batch

    rows = []
    for candidates, spectra in work:
        group_indices = {}
        if model_name in ('flare_group_formula', 'flare_peak_spread'):
            group_indices = {formula: indices for formula, indices in candidates.groupby('formula').indices.items()
                             if candidates.iloc[indices].candidate_id.nunique() >= 2}
            if not group_indices:
                continue
        atoms, atom_mask = flare_atoms(candidates.mol, layers, params['element_list'], device)
        for query, peaks, hypothesis in spectra:
            if model_name == 'flare_group_formula':
                ion, formulas = eligible_formulas(query, group_indices)
                assignments = [(formula, group_indices[formula]) for _, formula in formulas]
            elif hypothesis is not None:
                _, formula, ion, _ = hypothesis
                assignments = [(formula, slice(None))]
            else:
                assignments = []
            for formula, indices in assignments:
                spectrum = flare_spectrum(query, peaks, formula, ion, params)
                if spectrum is None:
                    continue
                encoded = spectral_model(spectrum[None, :].to(device))
                group_atoms = atoms[indices]
                encoded = encoded.expand(len(group_atoms), -1, -1)
                peak_mask = ~torch.all(encoded == -5, dim=-1)
                if model_name == 'flare_peak_spread':
                    for isomers in group_indices.values():
                        scores = contrasting_peak_similarity(
                            encoded[isomers], group_atoms[isomers], peak_mask[isomers], atom_mask[isomers],
                        )
                        rows.extend(score_rows(query, candidates.iloc[isomers], model_name, scores))
                    continue
                scores = filip_similarity_batch(encoded, group_atoms, peak_mask, atom_mask[indices])
                rows.extend(score_rows(query, candidates.iloc[indices], model_name, scores))
    return rows


def score_rows(query, candidates, model, scores):
    return [(query.molecule_id, query.query_id, candidate_id, model, score)
            for candidate_id, score in zip(candidates.candidate_id, scores.cpu().tolist(), strict=True)
            if np.isfinite(score)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path)
    parser.add_argument('--candidates', type=Path)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working'))
    parser.add_argument('--models', nargs='+',
                        choices=['mist_msg', 'flare', 'flare_group_formula', 'flare_peak_spread'],
                        default=['mist_msg', 'flare'])
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    import_rdkit()
    args.output.mkdir(parents=True, exist_ok=True)
    assets = args.assets or mounted('mist_msg_noaug.ckpt').parent
    source_dir = Path('/tmp/enveda-flare-mist')
    extract(assets / 'mist.tgz', source_dir / 'mist')
    sys.path.insert(0, str(source_dir / 'mist' / 'src'))
    if any(name.startswith('flare') for name in args.models):
        extract(assets / 'flare.tgz', source_dir / 'flare')
        sys.path.insert(0, str(source_dir / 'flare'))

    query_path = args.queries or mounted('queries.parquet')
    candidate_path = args.candidates or mounted('candidates.parquet')
    queries = pd.read_parquet(query_path)
    candidates = pd.read_parquet(candidate_path).dropna(subset=['smiles'])
    print(f'Queries ({len(queries)}): {query_path}\nCandidates ({len(candidates)}): {candidate_path}', flush=True)
    candidates = candidates[candidates.smiles.str.len().gt(0)].copy()
    candidates['mol'] = candidates.smiles.map(Chem.MolFromSmiles)
    candidates = candidates[candidates.mol.notna()].sort_values('rank', kind='stable')
    groups = dict(tuple(candidates.groupby('molecule_id', sort=False)))
    work = []
    formula_rows = []
    use_global_formula = any(name != 'flare_group_formula' for name in args.models)
    for molecule_id, group in queries.groupby('molecule_id', sort=False):
        if molecule_id not in groups:
            continue
        spectra = []
        for query in group.itertuples(index=False):
            peaks = processed_peaks(query)
            if not len(peaks):
                continue
            hypothesis = formula_hypothesis(query, groups[molecule_id], peaks) if use_global_formula else None
            spectra.append((query, peaks, hypothesis))
            if hypothesis is not None:
                formula_rows.append((molecule_id, query.query_id, hypothesis[1], hypothesis[0][1]))
        if spectra:
            work.append((groups[molecule_id], spectra))
    if use_global_formula:
        pd.DataFrame(formula_rows, columns=['molecule_id', 'query_id', 'inferred_formula', 'formula_error_ppm']).to_parquet(
            args.output / 'formula_predictions.parquet', index=False)
        print(f'Formula matches: {len(formula_rows)}/{len(queries)} spectra', flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scores = []
    for name in args.models:
        started = time.monotonic()
        try:
            if name == 'mist_msg':
                rows = score_mist(assets / 'mist_msg_noaug.ckpt', work, device)
            else:
                rows = score_flare(assets, source_dir / 'flare', work, device, name)
        except Exception:
            print(f'{name} failed', flush=True)
            traceback.print_exc()
            continue
        print(f'{name}: {len(rows)} scores, {len({row[1] for row in rows})} spectra, '
              f'{time.monotonic() - started:.1f}s on {device}', flush=True)
        scores.extend(rows)
    if not scores:
        raise RuntimeError('No candidate scores were produced')
    pd.DataFrame(scores, columns=SCORE_COLUMNS).to_parquet(args.output / 'scores.parquet', index=False)


if __name__ == '__main__':
    main()

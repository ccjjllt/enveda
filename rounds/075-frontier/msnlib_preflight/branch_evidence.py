"""Inventory source-held-out structural support from observed MS2 to native MS3 products."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
import gzip
from hashlib import sha256
import json
from pathlib import Path
import tempfile

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from audit import FILES, RELEASE, download, inventory, matching_parents


PROTON = 1.00727646662
HYDROGEN = 1.00782503223
CARRIERS = {'[M+H]+': PROTON, '[M-H]-': -PROTON}
TAUTOMER = rdMolStandardize.TautomerEnumerator()
H_TRANSFERS = (-1, 0, 1)
MAX_ATOMS = 96
MAX_CHILD_PEAKS = 256


@lru_cache(maxsize=3000)
def parent_structure(smiles):
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    if (molecule is None or molecule.GetNumAtoms() > MAX_ATOMS or len(Chem.GetMolFrags(molecule)) != 1
            or any(atom.GetFormalCharge() or atom.GetIsotope() for atom in molecule.GetAtoms())):
        return None
    Chem.RemoveStereochemistry(molecule)
    try:
        canonical = TAUTOMER.Canonicalize(molecule)
        identity = Chem.MolToInchiKey(canonical).split('-')[0]
        normalized = Chem.MolFromSmiles(Chem.MolToSmiles(canonical, isomericSmiles=False))
        scaffold = MurckoScaffold.GetScaffoldForMol(normalized)
        generic = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(scaffold)) if scaffold.GetNumAtoms() else ''
    except (RuntimeError, ValueError):
        return None
    if not identity:
        return None
    return molecule, identity, generic


def rooted_smiles(molecule, indices, root):
    return Chem.MolFragmentToSmiles(molecule, atomsToUse=sorted(indices), rootedAtAtom=root,
                                    isomericSmiles=False)


def local_environment(molecule, indices, root):
    local = {root}
    for _ in range(2):
        local.update(neighbor.GetIdx() for idx in tuple(local)
                     for neighbor in molecule.GetAtomWithIdx(idx).GetNeighbors() if neighbor.GetIdx() in indices)
    return rooted_smiles(molecule, local, root)


@lru_cache(maxsize=3000)
def fragment_sites(smiles):
    """Bracket a hydrogen-capped single-cut piece as in the measured anchor probe."""
    state = parent_structure(smiles)
    if state is None:
        return ()
    molecule = state[0]
    proposals = set()
    # RDKit fragment index maps preserve the parent cut-site atom on each side.
    for bond in molecule.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        broken = Chem.FragmentOnBonds(molecule, [bond.GetIdx()], addDummies=False)
        atom_map = []
        try:
            fragments = Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=True, fragsMolAtomMapping=atom_map)
        except (RuntimeError, ValueError):
            continue
        if len(fragments) != 2:
            continue
        endpoints = {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}
        for fragment, indices in zip(fragments, atom_map):
            if len(indices) < 2 or Chem.GetFormalCharge(fragment):
                continue
            kept_indices = set(indices)
            (kept,), (lost,) = (endpoints & kept_indices), (endpoints - kept_indices)
            leaving_indices = set(range(molecule.GetNumAtoms())) - kept_indices
            descriptor = rooted_smiles(molecule, kept_indices, kept)
            retained_site = local_environment(molecule, kept_indices, kept)
            leaving_site = local_environment(molecule, leaving_indices, lost)
            formula = rdMolDescriptors.CalcMolFormula(fragment)
            proposals.add((rdMolDescriptors.CalcExactMolWt(fragment), descriptor, retained_site, leaving_site, formula))
    return tuple(sorted(proposals))


def assign_fragment(smiles, adduct, observed):
    proposals = fragment_sites(smiles)
    tolerance = max(0.002, observed * 10e-6)
    match = {(exact, kept, lost, formula, shift)
             for capped_mass, exact, kept, lost, formula in proposals
             for shift in H_TRANSFERS
             if abs(capped_mass + CARRIERS[adduct] + shift * HYDROGEN - observed) <= tolerance}
    if not match:
        return 'no_one_cut_mass_fit' if proposals else 'no_valid_one_cut_proposal', None
    if len(match) > 1:
        return 'ambiguous_cut_site', None
    return 'unique', next(iter(match))


@dataclass
class Support:
    scaffolds: set = field(default_factory=set)
    identities: set = field(default_factory=set)
    files: set = field(default_factory=set)
    channels: dict = field(default_factory=lambda: defaultdict(lambda: (set(), set(), set())))
    raw_verified: set = field(default_factory=set)

    def add(self, identity, scaffold, raw_name, channel, direct):
        self.identities.add(identity)
        self.files.add(raw_name)
        channel_identities, channel_files, channel_scaffolds = self.channels[channel]
        channel_identities.add(identity)
        channel_files.add(raw_name)
        if scaffold:
            self.scaffolds.add(scaffold)
            channel_scaffolds.add(scaffold)
        if direct:
            self.raw_verified.add(raw_name)


def published_proofs(path):
    if path is None:
        return set(), None
    raw = path.read_bytes()
    report = json.loads(raw)
    if report['author_release'] != RELEASE:
        raise ValueError('The parent-scan provenance comes from a different author release')
    proof = report['raw_direct_reference_audit']
    rows = {(row['parent_export_usi'], row['child_export_usi'], row['raw_file']) for row in proof['examples']
            if row['direct_parent_refs']}
    if len(rows) != proof['counts']['candidate_links_directly_verified']:
        raise ValueError('Raw verification ledger does not agree with its summary')
    return rows, sha256(raw).hexdigest()


def key_id(kind, values):
    return sha256(json.dumps([kind, values], separators=(',', ':')).encode()).hexdigest()


def project(path, out, proofs, support):
    parents, children, _, _, _, _, coverage = inventory(path)
    counts = Counter()
    rows, raw_rows = set(), set()
    for tree, child_rows in children.items():
        for child in child_rows:
            matched = matching_parents(parents.get(tree, ()), child)
            shared = [(name, root) for root in matched for name in root.files.keys() & child.files.keys()]
            if not shared:
                continue
            counts['ms3_selected_peak_and_shared_raw_file'] += 1
            product = (tree, round(child.child_mass, 4))
            rows.add(product)
            direct = {name for name, root in shared if (root.spectrum, child.spectrum, name) in proofs}
            counts['linked_records_directly_raw_verified'] += bool(direct)
            status, assigned = assign_fragment(child.smiles, child.adduct, child.child_mass)
            counts[status] += 1
            if status != 'unique':
                continue
            exact, retained, leaving, formula, shift = assigned
            _, identity, scaffold = parent_structure(child.smiles)
            raw_rows.add(product)
            channel = (child.instrument, child.root_ce, child.ce)
            exact_key = key_id('exact_fragment_and_cut_site', (exact, retained, leaving, formula, shift))
            local_key = key_id('local_cut_site_and_fragment_formula', (retained, leaving, formula, shift))
            raw_names = sorted({name for name, _ in shared})
            tolerance = max(.002, child.child_mass * 10e-6)
            observed_products = sorted({mass for _, root in shared for mass in root.peaks
                                        if abs(mass - child.child_mass) <= tolerance})
            for raw_name in raw_names:
                support[(child.adduct, 'exact', exact_key)].add(identity, scaffold, raw_name, channel, raw_name in direct)
                support[(child.adduct, 'local', local_key)].add(identity, scaffold, raw_name, channel, raw_name in direct)
            row = dict(adduct=child.adduct, identity=identity, generic_scaffold=scaffold,
                       raw_author_identity=child.identity, instrument=child.instrument,
                       root_ce_raw=child.root_ce, ms3_ce_raw=child.ce, root_mz=child.root_mass,
                       isolated_product_mz=child.child_mass, observed_parent_product_mzs=observed_products,
                       peak_mzs=child.peaks[:MAX_CHILD_PEAKS],
                       peaks_truncated=len(child.peaks) > MAX_CHILD_PEAKS,
                       selected_fragment_formula_h_capped=formula, hydrogen_shift=shift,
                       rooted_fragment_smiles=exact, retained_cut_environment=retained,
                       leaving_cut_environment=leaving, exact_key=exact_key, local_key=local_key,
                       author_ms3_usi=child.spectrum, parent_ms2_usis=sorted({root.spectrum for _, root in shared}),
                       raw_file_ids=raw_names, directly_verified_raw_file_ids=sorted(direct))
            out.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
    return dict(counts=counts, distinct_feature_products=len(rows), unique_graph_feature_products=len(raw_rows),
                covered_raw_author_identities=len({entry[0] for entry in coverage}))


def summarize(support, out):
    summary = defaultdict(Counter)
    with gzip.open(out, 'wt', encoding='utf-8') as stream:
        for (adduct, tier, key), value in sorted(support.items()):
            totals = summary[f'{adduct}/{tier}']
            totals['unique_keys'] += 1
            totals['keys_with_raw_direct_proof'] += bool(value.raw_verified)
            totals['keys_with_multiple_identities'] += len(value.identities) >= 2
            totals['keys_with_multiple_generic_scaffolds'] += len(value.scaffolds) >= 2
            totals['keys_with_three_files_and_two_scaffolds'] += len(value.files) >= 3 and len(value.scaffolds) >= 2
            totals['keys_with_three_identities_files_and_two_scaffolds'] += (
                len(value.identities) >= 3 and len(value.files) >= 3 and len(value.scaffolds) >= 2)
            supported_channels = sum(
                len(ids) >= 3 and len(files) >= 3 and len(scaffolds) >= 2
                for ids, files, scaffolds in value.channels.values())
            totals['keys_with_three_identities_files_and_two_scaffolds_same_ce_channel'] += bool(supported_channels)
            row = dict(adduct=adduct, descriptor_tier=tier, key=key, identities=len(value.identities),
                       generic_scaffolds=len(value.scaffolds), raw_files=len(value.files),
                       directly_verified_files=len(value.raw_verified), instrument_ce_channels=len(value.channels),
                       instrument_ce_channels_with_diverse_support=supported_channels)
            stream.write(json.dumps(row, separators=(',', ':')) + '\n')
    return dict(summary)


def run(args):
    RDLogger.DisableLog('rdApp.*')
    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError('Use RDKit 2026.03.3 to compute comparable structures and scaffolds')
    if parent_structure('Cc1ccccc1') is None or not parent_structure('Cc1ccccc1')[2]:
        raise RuntimeError('The RDKit Murcko implementation did not return a known scaffold')
    args.out.mkdir(parents=True, exist_ok=True)
    proofs, proof_sha = published_proofs(args.direct_proofs)
    report = dict(author_release=RELEASE, usage='public MSnLib branch-support audit, no model fitting or evaluation',
                  rdkit=rdBase.rdkitVersion, hydrogen_transfers=H_TRANSFERS,
                  available_direct_reference_proofs=len(proofs), published_proofs_sha256=proof_sha, assets={})
    support = defaultdict(Support)
    branches = args.out / 'observed_branches.jsonl.gz'
    with tempfile.TemporaryDirectory() as tmp, gzip.open(branches, 'wt', encoding='utf-8') as out:
        scratch = Path(tmp)
        for name in ('targetmol_pos', 'targetmol_neg'):
            path, provenance = download(RELEASE, FILES[name], scratch)
            results = project(path, out, proofs, support)
            report['assets'][name] = dict(provenance=provenance, **results)
            path.unlink()
            print(name, results, flush=True)
    report['descriptor_support'] = summarize(support, args.out / 'descriptor_support.jsonl.gz')
    (args.out / 'branch_evidence.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--direct-proofs', type=Path)
    run(parser.parse_args())

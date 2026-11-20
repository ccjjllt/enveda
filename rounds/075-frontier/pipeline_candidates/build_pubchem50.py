"""Build the held-out MassBank pipeline with a 50-entry PubChem retrieval window."""

import argparse
import ast
from pathlib import Path

from build_kernel import build, literal_value, replace, source, update


SLUG = 'enveda-075b-massbank-pubchem50'
PANEL = 'jiyujia/enveda-075b-external-massbank-202611/1'


PC_PREPARE = """
_PC_HEAD = 25
_PC_FIELDS = ('pc', 'pc_keys', 'pc_fz')
_pc_staged, _pc_extended, _pc_extra = {}, {}, {}
for _mid, _proposal in PC.items():
    if ICE_PC and _mid in BASE and BASE[_mid][2] < LIB_TAU and len(_proposal['pc']) > _PC_HEAD:
        _pc_staged[_mid] = {name: _proposal[name] for name in _PC_FIELDS}
    for _field in _PC_FIELDS:
        _proposal[_field] = _proposal[_field][:_PC_HEAD]
"""


PC_INPUT = """        from functools import lru_cache
        from rdkit.Chem.MolStandardize import rdMolStandardize
        _pc_tautomers = rdMolStandardize.TautomerEnumerator()

        @lru_cache(maxsize=10000)
        def _pc_identity(smiles):
            mol = _Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            _Chem.RemoveStereochemistry(mol)
            return _Chem.MolToInchiKey(_pc_tautomers.Canonicalize(mol))[:14]

        for m in mids:
            p = PC.get(m) if isinstance(PC, dict) else None
            if p and p.get('pc') and BASE[m][2] < LIB_TAU:
                p['pc_form'] = [_form(x) for x in p['pc']]
                original = ice_fuse.ice_candidates(p['pc'], p['pc_form'], _PC_HEAD)
                baseline = list(dict.fromkeys([*cands.get(m, []), *original]))
                cands[m] = baseline
                if m not in _pc_staged:
                    continue
                deep = _pc_staged[m]
                slots = set(p['pc_form'])
                seen = {_pc_identity(smiles) for smiles in p['pc']}
                extended = {name: list(p[name]) for name in (*_PC_FIELDS, 'pc_form')}
                for i in range(_PC_HEAD, len(deep['pc'])):
                    smiles = deep['pc'][i]
                    formula = _form(smiles)
                    if formula not in slots:
                        continue
                    identity = _pc_identity(smiles)
                    if not identity or identity in seen:
                        continue
                    seen.add(identity)
                    for name in _PC_FIELDS:
                        extended[name].append(deep[name][i])
                    extended['pc_form'].append(formula)
                if len(extended['pc']) == len(p['pc']):
                    continue
                _pc_extended[m] = extended
                full = ice_fuse.ice_candidates(extended['pc'], extended['pc_form'], len(extended['pc']))
                baseline_set = set(baseline)
                _pc_extra[m] = [smiles for smiles in full if smiles not in baseline_set]
                cands[m] = baseline + _pc_extra[m]
    print('PC50 extension:', len(_pc_extended), 'molecules,',
          sum(len(x['pc']) - _PC_HEAD for x in _pc_extended.values()), 'eligible tail entries,',
          sum(map(len, _pc_extra.values())), 'extra predictions per eligible model', flush=True)
"""


PC_SCORES = """
# Keep extra predictions available to PubChem without changing frozen N1/E004 score coverage.
def _split_pc50_scores(scores):
    supplemental = {}
    for mid, smiles in _pc_extra.items():
        scores_for_molecule = scores.get(str(mid))
        if scores_for_molecule is None:
            continue
        supplemental[str(mid)] = {smi: scores_for_molecule.pop(smi) for smi in smiles
                                  if smi in scores_for_molecule}
        if not scores_for_molecule:
            scores.pop(str(mid))
    return supplemental

_pc50_ice = _split_pc50_scores(ICE_SCORES)
_pc50_gl = _split_pc50_scores(GL_SCORES)

def _pc50_valid(score):
    return score is not None and np.isfinite(score)
"""


PC_RERANK = """    p = PC.get(m) if (ICE_PC and isinstance(PC, dict)) else None
    if p and p.get('pc_form'):
        try:
            current = p
            pc_ice, pc_gl = ice, gl
            if m in _pc_extended:
                expanded = _pc_extended[m]
                all_ice = {**ice, **_pc50_ice.get(str(m), {})}
                all_gl = {**gl, **_pc50_gl.get(str(m), {})}
                used = [i for i, smi in enumerate(expanded['pc'])
                        if i < _PC_HEAD or _pc50_valid(all_ice.get(smi)) or _pc50_valid(all_gl.get(smi))]
                if len(used) > len(p['pc']) and all_ice:
                    current = {name: [expanded[name][i] for i in used]
                               for name in (*_PC_FIELDS, 'pc_form')}
                    pc_ice, pc_gl = all_ice, all_gl if any(_pc50_valid(v) for v in all_gl.values()) else {}
            if pc_ice:
                order = ice_fuse.rerank(current['pc'], current['pc_keys'], current['pc_fz'], current['pc_form'],
                                        pc_ice, lam=ICE_LAM, top_n=len(current['pc']))
                order = gl_rerank(order, current['pc'], current['pc_keys'], current['pc_fz'], current['pc_form'],
                                  pc_ice, pc_gl, len(current['pc']), 'pc_')[:_PC_HEAD]
                fz0 = p['pc_fz'][0]  # Frozen slot gate uses the raw channel leader's score.
                for field in (*_PC_FIELDS, 'pc_form'):
                    p[field] = [current[field][i] for i in order]
                p['pc_fz'][0] = fz0
                ice_stats['pc_changed'] = ice_stats.get('pc_changed', 0) + int(order[:1] != [0])
        except Exception as e:
            print('pc rerank failed', m, repr(e))
"""


def extend_pubchem(frozen):
    code = source(frozen, 34)
    core, = [node.value for node in ast.parse(code).body if isinstance(node, ast.Assign)
             and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'CORE']
    update(frozen, 34, ast.get_source_segment(code, core),
           repr(replace(literal_value(code, 'CORE'), 'K = 25', 'K = 50')))

    update(frozen, 41, '\ntry:\n    ICE_PKG =', PC_PREPARE + '\ntry:\n    ICE_PKG =')
    code = source(frozen, 41)
    start = code.index("        for m in mids:\n            p = PC.get(m)")
    stop = code.index("    cands = {m: c for m, c in cands.items() if c}", start)
    update(frozen, 41, code[start:stop], PC_INPUT)
    update(frozen, 41, '    GL_SCORES = {}\n', '    GL_SCORES = {}\n' + PC_SCORES)
    code = source(frozen, 41)
    start = code.index('    p = PC.get(m) if (ICE_PC')
    stop = code.index("print('ICE rerank stats'", start)
    update(frozen, 41, code[start:stop], PC_RERANK)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    build(args.destination, PANEL, SLUG, extend_pubchem)

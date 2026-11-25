"""Enumerate a separate channel of formula- and topology-restricted O-glycan losses."""

from collections import Counter
from functools import partial


LINK = '[O;D2;+0;!R;!$(OC=O);!$(O[!#6]):1]'
HYDROXYL = '[O;H1;+0]'
SUBSTITUTION = '[O;D2;+0]'


def native_ring(derive, rule):
    right = derive._TRANSFORMS[rule][0].split('>>')[1]
    anchor = '[O:1]'
    if not right.startswith(anchor):
        raise ValueError(f'Expected a published O-linked pyranose: {rule}')
    return right.removeprefix(anchor)


def hydroxyl_positions(ring):
    oxygen = ring.index('O')
    return tuple(i for i, symbol in enumerate(ring)
                 if symbol == 'O' and i > oxygen and ring[i - 1] != '=')


def restrict_hydroxyls(ring, position=None, substituent=''):
    hydroxyls = hydroxyl_positions(ring)
    if position is not None and position not in hydroxyls:
        raise ValueError('The glycan decoration does not follow a hydroxyl oxygen.')
    for i in reversed(hydroxyls):
        replacement = SUBSTITUTION + substituent if i == position else HYDROXYL
        ring = ring[:i] + replacement + ring[i + 1:]
    return ring


def disaccharide(root, terminal):
    # The second ring number keeps both pyranose ring closures local.
    branch = restrict_hydroxyls(terminal.replace('1', '2'))
    return tuple(restrict_hydroxyls(root, i, branch) for i in hydroxyl_positions(root))


def enumerate_family(reactions, reactants, max_products):
    for reaction in reactions:
        yield from reaction.RunReactants(reactants, max_products)


def build_loss_rules(derive):
    from rdkit.Chem import AllChem

    hexose = native_ring(derive, 'hexose_O')
    pentose = native_ring(derive, 'pentose_O')
    deoxyhexose = native_ring(derive, 'deoxyhexose_O')
    glucuronyl = native_ring(derive, 'glucuronyl_O')
    specs = {
        'loss_O_GlcA': (dict(C=6, H=8, O=6), (restrict_hydroxyls(glucuronyl),)),
        'loss_O_HexHex': (dict(C=12, H=20, O=10), disaccharide(hexose, hexose)),
        'loss_O_HexPent': (dict(C=11, H=18, O=9),
                           disaccharide(hexose, pentose) + disaccharide(pentose, hexose)),
        'loss_O_HexdHex': (dict(C=12, H=20, O=9),
                           disaccharide(hexose, deoxyhexose) + disaccharide(deoxyhexose, hexose)),
        'loss_O_acetylHex': (dict(C=8, H=12, O=6),
                            tuple(restrict_hydroxyls(hexose, i, 'C(=O)C')
                                  for i in hydroxyl_positions(hexose))),
    }
    if specs['loss_O_GlcA'][0] != derive._TRANSFORMS['glucuronyl_O'][1]:
        raise ValueError('The published glucuronyl_O formula changed.')
    rules = {}
    for name, (loss, rings) in specs.items():
        smarts = tuple(f'{LINK}{ring}>>[O:1]' for ring in rings)
        reactions = tuple(AllChem.ReactionFromSmarts(text) for text in smarts)
        for reaction in reactions:
            if reaction is None or reaction.GetNumReactantTemplates() != 1 or reaction.GetNumProductTemplates() != 1:
                raise ValueError(f'Expected a single-parent and single-product loss: {name}')
            reaction.Initialize()
        rules[name] = dict(delta=-derive._delta_mass(loss), loss=Counter(loss), smarts=smarts,
                           run=partial(enumerate_family, reactions))
    return rules


if __name__ == '__main__':
    from generate import arguments, generate

    args = arguments()
    if args.only_rule:
        raise ValueError('Use the five licensed-source-derived O-glycan rules as one channel.')
    generate(args, build_loss_rules=build_loss_rules)

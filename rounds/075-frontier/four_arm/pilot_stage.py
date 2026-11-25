"""Add blind retention and a shared forward-scoring plan to Pablo v29's inner engine."""
import ast
from pathlib import Path


ARMS = ('baseline', 'early_score', 'late_dedupe', 'joint')
HERE = Path(__file__).resolve().parent


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Pinned Pablo source differs at {old!r}')
    return source.replace(old, new, 1)


def cell_text(book, index):
    return ''.join(book['cells'][index]['source'])


def write_cell(book, index, source):
    compile(source, f'pablo-pilot-inner-{index}', 'exec')
    book['cells'][index]['source'] = source.splitlines(keepends=True)


def literal_source(source, name):
    assignments = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                   and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id == name]
    if len(assignments) != 1:
        raise ValueError(f'Archived source is missing the unique {name} literal')
    lines = source.splitlines(keepends=True)
    node = assignments[0]
    return ast.literal_eval(node.value), lambda value: ''.join(
        lines[:node.lineno - 1] + [f'{name} = {value!r}\n'] + lines[node.end_lineno:])


def patch_pubchem(source, depth):
    worker, write = literal_source(source, 'V17_PATCH')
    anchor = "    diag['n_listed'] = len(out)\n"
    collector = f"""    # Export a wider evidence ledger without altering the original channel order or threshold.
    pilot_depth = {depth}
    pilot_any, pilot_outside, pilot_seen = [], [], set()
    pilot_keyed = diag['n_keyed']
    for j in order:
        k = key(j)
        if k is None or k in pilot_seen:
            continue
        pilot_seen.add(k)
        in_pool = k in _W['pool_keys']
        rec = dict(key=k, smiles=smis[idx[j]], raw_fz=float(fz[j]),
                   S=float(sc[j]), pop=float(pop[idx[j]]))
        if len(pilot_any) < pilot_depth:
            pilot_any.append(rec)
        if not in_pool and len(pilot_outside) < pilot_depth:
            pilot_outside.append(rec)
        if len(pilot_any) == len(pilot_outside) == pilot_depth:
            break
    diag.update(pilot_any=pilot_any, pilot_outside=pilot_outside, n_keyed=pilot_keyed)
"""
    worker = replace_once(worker, anchor, collector + anchor)
    compile(worker, 'pablo-pilot-c34-worker', 'exec')
    source = write(worker)
    runner, write = literal_source(source, 'RUNNER')
    anchor = "            if (k + 1) % 25 == 0 or k + 1 == len(tasks):\n"
    runner = replace_once(runner, anchor, "            res[mid].update(pilot_any=d.get('pilot_any', []),\n"
                          "                            pilot_outside=d.get('pilot_outside', []))\n" + anchor)
    compile(runner, 'pablo-pilot-c34-runner', 'exec')
    return write(runner)


def patch_engine1(source, depth):
    anchor = 'BASE, n_err = {}, 0\n'
    source = replace_once(source, anchor, anchor + 'PILOT_E1_DEEP = {}\n')
    anchor = '    pids_ = []\n'
    source = replace_once(source, anchor, anchor + '    deeper = []\n')
    anchor = '                    if len(smis) >= TOPN: break\n'
    source = replace_once(source, anchor, anchor + f"""                for i in order:
                    if len(deeper) >= max(0, {depth} - len(smis)):
                        break
                    if C.key[i] in seen:
                        continue
                    seen.add(C.key[i])
                    deeper.append((C.smiles[i], C.key[i], float(score[i]), C.formula[i], int(C.pid[i])))
""")
    anchor = '    if POOLPOP_MU > 0 and len(scs) > 1:'
    source = replace_once(source, anchor, '    native_scores = tuple(scs)\n' + anchor)
    anchor = '    BASE[mid] = (smis, keys, lib_max, scs, forms)\n'
    return replace_once(source, anchor, anchor + """    if deeper:
        location, scale = float(np.mean(native_scores)), float(np.std(native_scores)) + 1e-9
        PILOT_E1_DEEP[mid] = [(sm, k, (rs - location) / scale + (POOLPOP_MU * POOL_POP[pid] if pid >= 0 else 0.0),
                               form if pid >= 0 else form + '|gen') for sm, k, rs, form, pid in deeper]
""")


def patch_inner(book, arm, *, depth=100, extra_e1=24, extra_pc=24):
    if arm not in ARMS or depth < 60 or extra_e1 < 0 or extra_pc < 0:
        raise ValueError((arm, depth, extra_e1, extra_pc))
    write_cell(book, 34, patch_pubchem(cell_text(book, 34), depth))
    write_cell(book, 39, patch_engine1(cell_text(book, 39), depth))
    source = cell_text(book, 41)
    parameters = dict(arm=arm, depth=depth, extra_e1=extra_e1, extra_pc=extra_pc)
    supplement = 'PILOT_PARAMETERS = ' + repr(parameters) + '\n' + (HERE / 'pilot_stage.source').read_text()
    anchor = 'for m in list(BASE):\n'
    source = replace_once(source, anchor, supplement + '\n' + anchor)
    anchor = '    BASE[m] = (smis[:25], keys[:25], lib_max)\n'
    source = replace_once(source, anchor, anchor + """    if PILOT_EARLY or PILOT_LATE:
        pilot_set_pubchem(m, keys[:25], smis[:25], lib_max)
""")
    anchor = "    if p and p.get('pc_form') and ice:\n"
    source = replace_once(source, anchor, "    if p and p.get('pc_form') and ice and not PILOT_EARLY:\n")
    anchor = '    if BASE_ICE and ice and any(v is not None for v in ice.values()):\n'
    source = replace_once(source, anchor, """    if PILOT_EARLY and lib_max < LIB_TAU:
        smis, keys = pilot_rank_main(m, smis, keys)
""" + anchor)
    write_cell(book, 41, source + '\npilot_export_budget()\n')
    return book

"""Build v6 submission notebook: split v5 single giant loop cell into
forward cell + run_groups() + 5 block cells (20 groups each) + assemble cell.
G 8 -> 4 (smaller H2D transfers, smaller Fmat). No logic change otherwise.
"""
import json
from pathlib import Path

P = Path('notebooks/submissions/submission/enveda-submission.ipynb')
nb = json.loads(P.read_text())
cells = nb['cells']
assert len(cells) == 8, len(cells)
loop_src = ''.join(cells[6]['source'])

marker = "t0=time.perf_counter()\n"
assert marker in loop_src
head, tail = loop_src.split(marker, 1)
assert tail.startswith('for g in range(0, len(mids), G):'), tail[:60]

body = tail[len('for g in range(0, len(mids), G):'):]
body_loop, body_asm = body.split('import csv\n', 1)
assert 'w.writerows(out_rows)' in body_asm and 'assert len(out_rows)==len(mids)' in body_asm
body = body_loop  # run_groups gets the group loop only; assemble is rebuilt below


def indent(src, n=4):
    pad = ' ' * n
    return ''.join(pad + ln if ln.strip() else ln for ln in src.splitlines(keepends=True))


head = head.replace('G = 8\n', 'G = 4\n')
assert '\nG = 4\n' in head
cell_fwd = (
    head.replace('out_rows = []\n', '')
    + "print('forward+prep done', tuple(Z.shape))\n"
    + "NG = (len(mids) + G - 1)//G\n"
    + "print('groups:', NG, 'G=', G)\n"
    + "stat = {'empty_window':0,'mass_fallback':0,'cco':0,'exception':0}\n"
    + "PARTS = {}\n"
    + "def run_groups(g0, g1):\n"
    + "    import pickle\n"
    + "    tB = time.perf_counter()\n"
    + "    rows = []\n"
    + f"    for g in range(g0*G, min(g1*G, len(mids)), G):\n"
    + indent(
        body.replace('out_rows.append(', 'rows.append(').replace(
            'time.perf_counter()-t0', 'time.perf_counter()-tB'),
        4,
    )
    + "    PARTS[g0] = rows\n"
    + "    pickle.dump({'rows': rows}, open(f'/kaggle/working/partial_{g0}.pkl','wb'))\n"
    + "    print('block done', g0, g1, 'rows', len(rows), round(time.perf_counter()-tB,1), stat)\n"
)

# sanity: no stray out_rows refs or t0 refs inside the new function part
fn_part = cell_fwd.split('PARTS = {}', 1)[1]
assert 'out_rows' not in fn_part and '-t0' not in fn_part, 'stray refs in run_groups'

block_cells = []
NB = 5
for b in range(NB):
    g0, g1 = b * 20, (b + 1) * 20
    block_cells.append(f"run_groups({g0}, {g1})\n")

cell_asm = (
    "import csv\n"
    "out_rows = []\n"
    "for g0 in sorted(PARTS):\n"
    "    out_rows.extend(PARTS[g0])\n"
    "print('assembled', len(out_rows), 'from blocks', sorted(PARTS))\n"
    "with open('/kaggle/working/submission.csv','w',newline='') as f:\n"
    "    w = csv.writer(f); w.writerow(['molecule_id','smiles'])\n"
    "    w.writerows(out_rows)\n"
    "ncco = sum(1 for _, s in out_rows if s.startswith('CCO'))\n"
    "print('wrote', len(out_rows), 'cco_rows', ncco)\n"
    "print('coverage:', stat)\n"
    "assert len(out_rows)==len(mids)\n"
)


def code(src):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {},
            'outputs': [], 'source': src.splitlines(keepends=True)}


new_cells = cells[:6] + [code(cell_fwd)] + [code(s) for s in block_cells] + [code(cell_asm)] + cells[7:]
nb['cells'] = new_cells
P.write_text(json.dumps(nb))
print('cells:', len(new_cells))
print('fwd len:', len(cell_fwd), '| asm len:', len(cell_asm))

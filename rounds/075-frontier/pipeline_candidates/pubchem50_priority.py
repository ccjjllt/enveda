"""Reserve the original simulator queue before scoring the PubChem-50 tail."""

from build_kernel import source, update


RUNTIME = '''
def _pc50_runner_source(code):
    from textwrap import indent

    parse_args = '    return ap.parse_args(argv)'
    loop_start = '        stopped = False\\n        for mol in mols:\\n'
    loop_end = '        meta.update(n_pred=pred.n_pred'
    return_status = "        return done('budget' if stopped else 'ok')"
    if any(code.count(marker) != 1 for marker in (parse_args, loop_start, loop_end, return_status)):
        raise ValueError('Unexpected simulator runner source')
    code = code.replace(parse_args, "    ap.add_argument('--tail-input')\\n" + parse_args)
    start, end = code.index(loop_start), code.index(loop_end)
    loop = code[start:end].removeprefix('        stopped = False\\n')
    flush = '        if not stopped and not flush_chunk():\\n            stopped = True\\n'
    stop = '                    stopped = True\\n                    break\\n'
    if not loop.endswith(flush) or loop.count(stop) != 1:
        raise ValueError('Unexpected simulator chunk loop')
    loop = loop.removesuffix(flush).replace(stop, '                    return False\\n')
    code = (code[:start] + '        def run_phase(mols):\\n            nonlocal chunk_jobs\\n'
            + indent(loop, '    ') + '            return flush_chunk()\\n\\n'
            + '        stopped = not run_phase(mols)\\n' + code[end:])
    return code.replace(return_status, _PC50_TAIL_PHASE + return_status)


def _pc50_run_tail(run, package, runner, items, tail_items, **options):
    from tempfile import TemporaryDirectory

    if not tail_items:
        return run(package, items, **options)
    with TemporaryDirectory(prefix='pc50-runner-') as work:
        try:
            package_path, shadow = Path(package), Path(work)
            code = _pc50_runner_source((package_path / runner).read_text())
            compile(code, runner, 'exec')
            for child in package_path.iterdir():
                if child.name != runner:
                    (shadow / child.name).symlink_to(child)
            (shadow / runner).write_text(code)
            tail = shadow / 'pc50_tail_input.json'
            tail.write_text(json.dumps(tail_items))
        except (OSError, SyntaxError, ValueError) as error:
            print(f'PC50 tail runner unavailable; scoring original candidates: {error!r}', flush=True)
            return run(package, items, **options)
        options['extra_args'] = [*options.get('extra_args', ()), '--tail-input', str(tail)]
        return run(str(shadow), items, **options)
'''


TAIL_PHASE = '''        if a.tail_input:
            base_meta = meta
            meta = dict(status='base_incomplete' if stopped else 'no_time', errors=[],
                        n_mols_covered=0, n_mols_scored=0, n_scored=0)
            tail_started = time.time()
            pred_before = pred.n_pred
            try:
                if not stopped and time.time() < deadline:
                    with open(a.tail_input) as handle:
                        tail = json.load(handle)
                    for mol in tail:
                        mid = str(mol.get('mid'))
                        if mid not in plans:
                            plans[mid] = molecule_plan(mol)
                        scores = result.setdefault(mid, {})
                        mol['cands'] = [candidate for candidate in mol['cands'] if candidate not in scores]
                        meta['n_mols_covered'] += bool(plans[mid] and mol['cands'])
                    meta['status'] = 'ok' if run_phase(tail) else 'budget'
            except Exception as error:
                meta['status'] = 'error'
                meta['errors'].append(repr(error)[:500])
            finally:
                meta['n_pred'] = pred.n_pred - pred_before
                meta['seconds'] = round(time.time() - tail_started, 2)
                base_meta['pc50_tail'] = meta
                meta = base_meta
                log('PC50 tail', meta['pc50_tail'])
'''


def prioritize_original_predictions(frozen):
    ice_start = '    ICE_SCORES = ice_fuse.run_ice(ICE_PKG, items,'
    gl_start = '    GL_SCORES = gl_fuse.run_gl(GL_PKG, gl_items,'
    prepare_input = '''    pc50_tail_items = ice_fuse.build_ice_input(te, {m: c for m, c in _pc_extra.items() if c})
'''
    prepare_gl = '''    pc50_gl_tail = [dict(it, spectra=[s for s in it['spectra'] if s.get('adduct') == '[M+H]+'])
                    for it in pc50_tail_items]
    pc50_gl_tail = [it for it in pc50_gl_tail if it['spectra']]
'''
    # Keep the original candidate queue and patch the mounted runners only for these two calls.
    runtime = f'\n_PC50_TAIL_PHASE = {TAIL_PHASE!r}\n' + RUNTIME
    update(frozen, 41, '\ntry:\n    ICE_PKG =', runtime + '\ntry:\n    ICE_PKG =')
    update(frozen, 41, '                cands[m] = baseline + _pc_extra[m]\n', '')
    update(frozen, 41, ice_start, prepare_input +
           "    ICE_SCORES = _pc50_run_tail(ice_fuse.run_ice, ICE_PKG, 'ice_runner.py', items, pc50_tail_items,")
    update(frozen, 41, gl_start, prepare_gl +
           "    GL_SCORES = _pc50_run_tail(gl_fuse.run_gl, GL_PKG, 'gl_runner.py', gl_items, pc50_gl_tail,")

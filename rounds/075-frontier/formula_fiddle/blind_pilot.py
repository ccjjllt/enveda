"""Adapt fixed external spectra to FIDDLE v2.4 without mounting candidate truth."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from runpy import run_path
from tempfile import TemporaryDirectory


SLUG = 'enveda-075-fiddle-v240-blind-formula'
STAGE = 'jiyujia/enveda-075-fiddle-v240-timstof-stage/2'
SOURCES = {
    'mb110': ('jiyujia/enveda-075b-external-massbank-202611/1', 'pipeline_queries.parquet',
              '74929665fcceed3eb50e38f1bba500f7d208e21f7874c397b94df9071b6e75d5', 110, 170),
    'np60': ('jiyujia/enveda-075-np-recall-base/2', 'effective_query_spectra.parquet',
             '1b1431a7ee13652fc037781dc05aa5284a0e52292ee57d5fa531764eb1511531', 60, 98),
    'e180': ('jiyujia/enveda-075-e180-full-blind-input/1', 'test.parquet',
             '6fe782c57481b6fea95d1197da05ce4968c393881e3d962bcf8cdaed09b3bf7b', 121, 299),
}
COLUMNS = ('molecule_id', 'spectrum_id', 'instrument_type', 'ionization_mode', 'adduct', 'precursor_mz',
           'collision_energy_orig', 'collision_energy_orig_units', 'collision_energy_ev',
           'ms2_mzs', 'ms2_normalized_intensities')
ADDUCT_ALIASES = dict.fromkeys(('[M+FA-H]-', '[M+HCOO]-', '[M+HCO2]-', '[M+HCOOH-H]-'), '[M+CH2O2-H]-')
FILES = ('fiddle_blind_input.mgf', 'fiddle_blind_scans.jsonl', 'fiddle_blind_candidates.jsonl')
RECEIPT = 'fiddle_blind_formula_receipt.json'


def checksum(path):
    return sha256(path.read_bytes()).hexdigest()


def mounted(inputs, reference, name):
    slug = reference.split('/')[1]
    path, = (p for p in inputs.rglob(name) if slug in p.parts)
    return path


def measured_scan(row, alias):
    instrument = (row['instrument_type'] or '').strip().lower()
    if instrument != 'timstof':
        return None, 'other_instrument' if instrument else 'instrument_unknown'
    energies = row['collision_energy_ev'] or []
    if (row['collision_energy_orig_units'] != 'eV' or not energies
            or not all(type(x) in (float, int) and isfinite(x) and x != 0 for x in energies)):
        return None, 'measured_energy_unavailable'
    adduct = re.sub(r'\]1([+-])$', r']\1', re.sub(r'\s+', '', row['adduct'] or ''))
    adduct = ADDUCT_ALIASES.get(adduct, adduct)
    suffix = {'positive': '+', 'negative': '-'}.get(row['ionization_mode'])
    if suffix is None or not re.fullmatch(r'\[[^\]\r\n]+\]\d*[+-]', adduct) or not adduct.endswith(suffix):
        return None, 'inconsistent_adduct_or_polarity'
    precursor = row['precursor_mz']
    mz, intensity = row['ms2_mzs'] or [], row['ms2_normalized_intensities'] or []
    if (type(precursor) not in (float, int) or not isfinite(precursor) or precursor <= 0
            or not mz or len(mz) != len(intensity)
            or any(not isfinite(m) or not isfinite(i) or m <= 0 or i < 0 for m, i in zip(mz, intensity))):
        return None, 'unusable_peaks_or_precursor'
    energy = ','.join(str(abs(float(value))) for value in energies)
    return dict(id=alias, mz_array=mz, intensity_array=intensity, precursor_mz=precursor,
                adduct=adduct, collision_energy=energy), 'submitted'


def prepare(inputs, panel_names=tuple(SOURCES)):
    import pyarrow.parquet as pq

    cases, scans, audit = {}, [], {}
    for panel in panel_names:
        reference, filename, digest, n_mols, n_scans = SOURCES[panel]
        path = mounted(inputs, reference, filename)
        if checksum(path) != digest:
            raise ValueError(f'Frozen source spectra changed for {panel}')
        rows = pq.read_table(path, columns=list(COLUMNS)).to_pylist()
        if (len(rows) != n_scans or len({r['spectrum_id'] for r in rows}) != n_scans
                or len({r['molecule_id'] for r in rows}) != n_mols):
            raise ValueError(f'Unexpected or duplicated source scans for {panel}')
        summary = dict(reference=reference, filename=filename, input_sha256=digest, molecules=n_mols, scans=n_scans,
                       instrument_scan_counts=Counter(), instrument_case_counts=defaultdict(set))
        for ordinal, row in enumerate(rows, 1):
            alias = f'formula_{panel}_{ordinal:04d}'
            scan, status = measured_scan(row, alias)
            summary['instrument_scan_counts'][row['instrument_type']] += 1
            summary['instrument_case_counts'][row['instrument_type']].add(row['molecule_id'])
            cases[alias] = dict(panel=panel, molecule_id=row['molecule_id'], spectrum_id=row['spectrum_id'], id=alias,
                                instrument=row['instrument_type'], adduct=row['adduct'], precursor_mz=row['precursor_mz'],
                                energy_ev=row['collision_energy_ev'], energy_original=row['collision_energy_orig'],
                                energy_original_units=row['collision_energy_orig_units'], status=status)
            if scan is not None:
                scans.append(scan)
        summary['instrument_case_counts'] = {name: len(ids) for name, ids in summary['instrument_case_counts'].items()}
        audit[panel] = summary
    return cases, scans, audit


def write_mgf(path, scans):
    with path.open('w', encoding='ascii') as handle:
        for scan in scans:
            precursor = format(scan['precursor_mz'], '.17g')
            lines = ('BEGIN IONS', f"TITLE={scan['id']}", f'PEPMASS={precursor}', f'PRECURSOR_MZ={precursor}',
                     f"PRECURSOR_TYPE={scan['adduct']}", f"COLLISION_ENERGY={scan['collision_energy']}")
            handle.write('\n'.join(lines) + '\n')
            for mz, intensity in zip(scan['mz_array'], scan['intensity_array']):
                handle.write(f'{mz:.17g} {intensity:.17g}\n')
            handle.write('END IONS\n\n')


def emit_jsonl(path, values):
    path.write_text(''.join(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n' for row in values))


def run(inputs, out, stage_ref=STAGE, panel_names=('e180',)):
    import torch
    from stage import COMMIT, RECEIPT as STAGE_RECEIPT, TAG, load_bundle, make_predictor

    torch.set_num_threads(2)
    out.mkdir(exist_ok=True, parents=True)
    cases, scans, audit = prepare(inputs, panel_names)
    if not scans:
        raise ValueError('No source-confirmed timsTOF spectrum has explicit measured eV')
    mgf, scan_file, candidates_file = (out / name for name in FILES)
    write_mgf(mgf, scans)
    stage = mounted(inputs, stage_ref, STAGE_RECEIPT)
    pinned = json.loads(stage.read_bytes())
    if any(pinned.get(k) != v for k, v in dict(status='sealed', release_tag=TAG, upstream_commit=COMMIT,
                                                 instrument='timstof', device='cpu').items()):
        raise ValueError('Expected the sealed author v2.4.0 CPU stage')
    with TemporaryDirectory(prefix='fiddle-blind-formula-') as tmp:
        _, config, checkpoints = load_bundle(stage.parent, Path(tmp))
        predictor = make_predictor(config, checkpoints)
        validated = predictor.validate_mgf(mgf).to_dict(orient='records')
        if [r['id'] for r in validated] != [scan['id'] for scan in scans]:
            raise ValueError('Author validation changed the submitted alias population')
        accepted = []
        for row, scan in zip(validated, scans, strict=True):
            if row['status'] not in ('ok', 'warning', 'rejected'):
                raise ValueError('Unknown FIDDLE validation status')
            case = cases[scan['id']]
            case.update(status='accepted' if row['status'] != 'rejected' else 'author_rejected',
                        author_validation_status=row['status'], author_validation_reason=row['reason'])
            if row['status'] != 'rejected':
                accepted.append(scan)
        predictions = predictor.predict_batch(accepted, top_k=5) if accepted else []
    if [row['id'] for row in predictions] != [scan['id'] for scan in accepted]:
        raise ValueError('FIDDLE prediction population disagrees with the accepted source scans')
    rows = []
    for prediction, scan in zip(predictions, accepted, strict=True):
        case = cases[scan['id']]
        metadata = prediction['metadata']
        if metadata['instrument_type'] != 'timstof' or metadata['adduct'] != scan['adduct']:
            raise ValueError('Author API did not preserve the observed instrument or adduct')
        candidate_rows = []
        for item in prediction['candidates']:
            raw = item['metadata']['raw_rescore']
            if (not item['formula'] or item['mass'] is None or not isfinite(item['mass'])
                    or not isfinite(raw) or item['score'] != min(1., max(0., raw))):
                raise ValueError('Unusable author formula output')
            candidate_rows.append(dict(formula=item['formula'], mass=item['mass'], raw_rescore=raw))
        scores = [item['raw_rescore'] for item in candidate_rows]
        if len(scores) > 5 or scores != sorted(scores, reverse=True):
            raise ValueError('Expected the original author top-five formula ordering')
        case['status'] = 'predicted' if candidate_rows else 'no_formula_candidates'
        rows.append(dict(id=scan['id'], instrument=metadata['instrument_type'], adduct=metadata['adduct'],
                         ce_encoded=metadata['collision_energy'], candidates=candidate_rows))
    emit_jsonl(scan_file, cases.values())
    emit_jsonl(candidates_file, rows)
    for panel, summary in audit.items():
        group = [r for r in cases.values() if r['panel'] == panel]
        summary['scan_status'] = Counter(r['status'] for r in group)
        summary['eligible_instrument_and_ce_queries'] = len({r['molecule_id'] for r in group if
            r['status'] in ('predicted', 'no_formula_candidates', 'author_rejected')})
        summary['predicted_queries'] = len({r['molecule_id'] for r in group if r['status'] == 'predicted'})
    result = dict(status='sealed', formula_budget_per_scan=5, stage=stage_ref, stage_receipt_sha256=checksum(stage),
                  original_precursor=True, explicit_eV_only=True, site_instrument='timstof', panels=audit,
                  files={name: checksum(out / name) for name in FILES})
    (out / RECEIPT).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['panels'], allow_nan=False), flush=True)


def build(out, stage_version=2, all_panels=False):
    here = Path(__file__).resolve().parent
    common = run_path(str(here.parent / 'virtual_np/build.py'))
    stage_ref = f"{STAGE.rsplit('/', 1)[0]}/{stage_version}"
    panels = tuple(SOURCES) if all_panels else ('e180',)
    modules = {name: (here / name).read_bytes() for name in ('blind_pilot.py', 'stage.py')}
    for name, raw in modules.items():
        compile(raw, name, 'exec')
    payload = {name: common['pack'](raw) for name, raw in modules.items()}
    code = f'''
import base64, sys, tempfile, zlib
import subprocess
from hashlib import sha256
from pathlib import Path
inputs = Path('/kaggle/input')
{common['METRIC_BOOTSTRAP']}
with tempfile.TemporaryDirectory(prefix='fiddle-formula-') as temp:
    for name, (blob, expected) in {payload!r}.items():
        raw = zlib.decompress(base64.b64decode(blob))
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f'Embedded implementation changed: {{name}}')
        Path(temp, name).write_bytes(raw)
    sys.path.insert(0, temp)
    from blind_pilot import run
    run(inputs, Path('/kaggle/working'), {stage_ref!r}, {panels!r})
'''
    compile(code, SLUG, 'exec')
    meta = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='run.py', language='python', kernel_type='script',
                is_private=True, enable_gpu=False, enable_internet=False, docker_image=common['IMAGE'],
                dataset_sources=[common['WHEEL']], competition_sources=[],
                kernel_sources=[stage_ref, *(SOURCES[panel][0] for panel in panels)])
    out.mkdir(exist_ok=True, parents=True)
    (out / 'run.py').write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return checksum(out / 'run.py')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out', type=Path, required=True)
    cli.add_argument('--stage-version', type=int, default=2)
    cli.add_argument('--all-panels', action='store_true')
    print(build(**vars(cli.parse_args())))

"""Stage a dated COCONUT November 2026 archive without reading labels."""

import csv
import hashlib
import io
import json
from pathlib import Path
import time
from zipfile import ZipFile

import requests


RELEASE = 'https://coconut.naturalproducts.net/download'
VERSION = '2026-11'
UPSTREAM = f'https://coconut.s3.uni-jena.de/prod/downloads/{VERSION}'
VARIANTS = {
    'lite': dict(archive='coconut_csv_lite-11-2026.zip', receipt='coconut_2026_source_receipt.json',
                 size=199_108_039, etag='ff30cff5276dfe8880fb3c8ce539b7ef-24'),
    'full': dict(archive='coconut_csv-11-2026.zip', receipt='coconut_2026_full_source_receipt.json',
                 size=247_958_739, etag='0700d839a9aa3896e2827600abc3b52e-30'),
}
DEFAULT_VARIANT = 'lite'
MAX_DOWNLOAD = 256 * 1024 * 1024
MAX_UNCOMPRESSED = 4 * 1024 * 1024 * 1024
OUT = Path('/kaggle/working')


def download(path, upstream, expected_bytes, expected_etag):
    wanted_headers = ('Content-Type', 'Content-Length', 'ETag', 'Last-Modified', 'x-amz-version-id')
    for attempt in range(3):
        sha = hashlib.sha256()
        size = 0
        try:
            with requests.get(upstream, stream=True, timeout=(30, 90), allow_redirects=False,
                              headers={'Accept-Encoding': 'identity'}) as response:
                response.raise_for_status()
                if response.status_code != 200:
                    raise ValueError(f'Expected the dated full archive, got HTTP {response.status_code}')
                headers = {key: response.headers[key] for key in wanted_headers if key in response.headers}
                declared = int(headers['Content-Length']) if 'Content-Length' in headers else None
                if declared is not None and not 0 < declared <= MAX_DOWNLOAD:
                    raise ValueError('Archive exceeds the download ceiling or has no content')
                if declared != expected_bytes or headers.get('ETag', '').strip('"') != expected_etag:
                    raise ValueError('Dated object metadata changed from the recorded official HEAD')
                with path.open('wb') as stream:
                    for block in response.iter_content(chunk_size=4 * 1024 * 1024):
                        size += len(block)
                        if size > MAX_DOWNLOAD:
                            raise ValueError('Archive exceeded the download ceiling during transfer')
                        stream.write(block)
                        sha.update(block)
                if not size or (declared is not None and size != declared):
                    raise requests.RequestException('Incomplete archive download')
            return {'sha256': sha.hexdigest(), 'bytes': size, 'http_headers': headers}
        except requests.RequestException:
            path.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))


def inspect_archive(path):
    with ZipFile(path) as archive:
        entries = [info for info in archive.infolist() if not info.is_dir()]
        if sum(info.file_size for info in entries) > MAX_UNCOMPRESSED:
            raise ValueError('Archive would exceed the uncompressed ceiling')
        members = [info for info in entries if info.filename.lower().endswith('.csv')
                   and not any(part.startswith(('.', '__MACOSX')) for part in info.filename.split('/'))]
        if len(members) != 1:
            raise ValueError(f'Expected one CSV member, found {len(members)}')
        member = members[0]
        with archive.open(member) as record:
            columns = next(csv.reader(io.TextIOWrapper(record, encoding='utf-8-sig')), [])
        if not columns or len(columns) != len(set(columns)) or any(not key.strip() for key in columns):
            raise ValueError('Missing, empty or duplicate CSV header fields')
        structures = [key for key in columns if 'smiles' in key.lower()]
        if not structures:
            raise ValueError('The lite CSV has no recognizable SMILES column')
        metadata = [key for key in columns if any(word in key.lower() for word in (
            'source', 'collection', 'organism', 'taxonomy', 'species', 'doi', 'citati', 'reference'))]
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f'ZIP CRC check failed for {bad_member}')
        return dict(csv_member=member.filename, csv_bytes=member.file_size, csv_crc32=f'{member.CRC:08x}',
                    columns=columns, structure_columns=structures, provenance_columns=metadata)


def stage(out=OUT, variant=DEFAULT_VARIANT):
    spec = VARIANTS[variant]
    archive = spec['archive']
    upstream = f'{UPSTREAM}/{archive}'
    out.mkdir(parents=True, exist_ok=True)
    partial = out / f'{archive}.part'
    try:
        received = download(partial, upstream, spec['size'], spec['etag'])
        inspected = inspect_archive(partial)
        if variant == 'full' and not {'collections', 'organisms', 'dois'} <= set(inspected['columns']):
            raise ValueError('The full CSV lacks its expected provenance fields')
        report = dict(source=f'COCONUT 2.0 {variant} CSV', version=VERSION, release=RELEASE,
                      upstream=upstream, filename=archive, license='CC0-1.0',
                      expected_object_bytes=spec['size'], expected_multipart_etag=spec['etag'],
                      recorded_download_limit_bytes=MAX_DOWNLOAD, zip_crc_checked=True,
                      **received, **inspected,
                      usage='Public raw archive staged untouched. Corpus overlap has not been assessed.')
        partial.replace(out / archive)
        (out / spec['receipt']).write_text(json.dumps(report, indent=2) + '\n')
        print('COCONUT_2026_SOURCE_READY', json.dumps(report, sort_keys=True), flush=True)
    finally:
        partial.unlink(missing_ok=True)


if __name__ == '__main__':
    stage()

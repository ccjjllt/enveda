"""Stage the seven public NP-MRD SMILES archives released on 2025-08-01."""

import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import NamedTuple

import requests


RELEASE_DATE = '2025-08-01'
RELEASE_URL = 'https://np-mrd.org/downloads'
LICENSE_URL = 'https://np-mrd.org/about'
UPSTREAM = 'https://np-mrd.org/system/downloads/current'
LAST_MODIFIED = 'Fri, 01 Aug 2025 22:19:41 GMT'
OUT = Path('/kaggle/working')
RECEIPT = 'npmrd_2025_source_receipt.json'
FINGERPRINT_BYTES = 4096
MAX_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
COLUMNS = ('Natural_Products_Name', 'NP_MRD_ID', 'SMILES')


class Archive(NamedTuple):
    id_range: str
    size: int
    etag: str
    head_sha256: str
    tail_sha256: str

    @property
    def filename(self):
        return f'smiles_{self.id_range}.csv.gz'


ARCHIVES = (
    Archive('NP0000001_NP0050000', 1_539_234, '688d3d7d-177ca2',
            'f0fe11827a251c2e2dfddfff314f202a54eb59822fe271d37d8ab9dd88c44b60',
            '30cd8cc301657005477f07fca29cc0ed27e08942f975c379c1c6b67a95db2782'),
    Archive('NP0050001_NP0100000', 972_512, '688d3d7d-ed6e0',
            '16fd328063e8c32c1be9840118d2dfb379ead0d4ebcf9484048ada1a45c1022c',
            '1c8acfa615e8939d575e39f11dae5786c0b5e9d1967c80bd686d7ef1003baf19'),
    Archive('NP0100001_NP0150000', 609_576, '688d3d7d-94d28',
            'a8bab037e2c51571561623947cff12150a2931303ff53587c7ace1e522e08603',
            '48449ee2e56163a20e5cfe06e76ced38dcf7f8c30e9670ac5a23205ce52ed422'),
    Archive('NP0150001_NP0200000', 2_724_667, '688d3d7d-29933b',
            '6e496551f063b442ba5b9f7553edda1f8bbcd14455de783c293f31d01911afe7',
            'd7999ead2173aa927c85caec67fb68107be00b7b8f8d553c8125dd783df379cd'),
    Archive('NP0200001_NP0250000', 2_734_147, '688d3d7d-29b843',
            'a518057152d05b890f2c9f9d5e98521e1d785174430b10145038efcd428561ae',
            '9b6da4ef094898e604355d6813f3c16943cb00dee880bd7a90eb73f9b67a6a7e'),
    Archive('NP0250001_NP0300000', 2_733_262, '688d3d7d-29b4ce',
            'e6e6dc7646defc8a4ac994d5dbbbb44fc1941e0883065a089a323b2658d42213',
            '68fb7fb61ebde6d807b4330c241b1e21f6a7f5097b8bc9621b4e2b447aa7e44e'),
    Archive('NP0300001_NP0350000', 1_929_558, '688d3d7d-1d7156',
            '04eac1d465daf74ad77f23e8a136dd40fb81b2e491a70d0a8d7a4abb1369f055',
            '6b15d9f2b55b110052737b3846cd01ddf3342c878011766aa2293267d9b01170'),
)


def download(session, url, path, spec):
    for attempt in range(3):
        sha = hashlib.sha256()
        size = 0
        try:
            with session.get(url, stream=True, timeout=(20, 60), allow_redirects=False,
                             headers={'Accept-Encoding': 'identity'}) as response:
                response.raise_for_status()
                if response.status_code != 200:
                    raise ValueError(f'{spec.filename} returned HTTP {response.status_code}')
                header_names = ('Content-Type', 'Content-Length', 'ETag', 'Last-Modified')
                headers = {key: response.headers[key] for key in header_names if key in response.headers}
                if 'Content-Length' in headers and int(headers['Content-Length']) != spec.size:
                    raise ValueError(f'Content-Length changed for {spec.filename}')
                etag = headers.get('ETag')
                if etag is not None and etag.strip('"') != spec.etag:
                    raise ValueError(f'ETag changed for {spec.filename}')
                modified = headers.get('Last-Modified')
                if modified != LAST_MODIFIED and (modified is not None or etag is None):
                    raise ValueError(f'Expected release timestamp or ETag missing for {spec.filename}')
                with path.open('wb') as stream:
                    for block in response.iter_content(chunk_size=256 * 1024):
                        size += len(block)
                        if size > spec.size:
                            raise ValueError(f'{spec.filename} exceeded its pinned compressed size')
                        stream.write(block)
                        sha.update(block)
                if size != spec.size:
                    raise requests.RequestException(f'Incomplete download of {spec.filename}')
            return size, sha.hexdigest(), headers
        except requests.RequestException:
            path.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))


def inspect_archive(path, spec):
    with path.open('rb') as stream:
        head = hashlib.sha256(stream.read(FINGERPRINT_BYTES)).hexdigest()
        stream.seek(-FINGERPRINT_BYTES, 2)
        tail = hashlib.sha256(stream.read(FINGERPRINT_BYTES)).hexdigest()
    if (head, tail) != (spec.head_sha256, spec.tail_sha256):
        raise ValueError(f'Compressed byte fingerprints changed for {spec.filename}')

    with gzip.open(path, 'rb') as stream:
        header = stream.readline(1024)
        if tuple(header.decode('utf-8-sig').rstrip('\r\n').split(',')) != COLUMNS:
            raise ValueError(f'Structure CSV column mismatch in {spec.filename}')
        uncompressed = len(header)
        # Reading the complete GZIP payload verifies its internal checksum and length.
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            uncompressed += len(block)
            if uncompressed > MAX_UNCOMPRESSED_BYTES:
                raise ValueError(f'{spec.filename} exceeded its decompression limit')
    return uncompressed


def stage(out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='npmrd-', dir=out) as temporary, requests.Session() as session:
        scratch = Path(temporary)
        artifacts = []
        for spec in ARCHIVES:
            url = f'{UPSTREAM}/{spec.filename}'
            path = scratch / spec.filename
            size, sha, headers = download(session, url, path, spec)
            uncompressed = inspect_archive(path, spec)
            artifacts.append(dict(filename=spec.filename, upstream=url, bytes=size, sha256=sha,
                                  uncompressed_bytes=uncompressed, http_headers=headers,
                                  expected_etag=spec.etag, head_sha256=spec.head_sha256,
                                  tail_sha256=spec.tail_sha256))
            print('NP_MRD_ARCHIVE_VERIFIED', spec.filename, size, sha, flush=True)

        receipt = dict(source='Natural Products Magnetic Resonance Database (NP-MRD)',
                       release_date=RELEASE_DATE, release_url=RELEASE_URL,
                       license='CC-BY-NC-4.0', license_url=LICENSE_URL,
                       expected_last_modified=LAST_MODIFIED, fingerprint_bytes=FINGERPRINT_BYTES,
                       format='gzip-compressed structure CSV', columns=COLUMNS,
                       gzip_integrity_checked=True, total_bytes=sum(spec.size for spec in ARCHIVES),
                       artifacts=artifacts)
        (scratch / RECEIPT).write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        for spec in ARCHIVES:
            (scratch / spec.filename).replace(out / spec.filename)
        (scratch / RECEIPT).replace(out / RECEIPT)
    print('NP_MRD_2025_SOURCE_READY', RECEIPT, len(artifacts), receipt['total_bytes'], flush=True)


if __name__ == '__main__':
    stage()

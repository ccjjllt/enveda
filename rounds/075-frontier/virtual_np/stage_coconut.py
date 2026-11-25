"""Stage the raw COCONUT snapshot on Kaggle for reproducible parent reconstruction."""

import csv
import hashlib
import io
import json
from pathlib import Path
import time
from zipfile import ZipFile

import requests


RECORD = 'https://zenodo.org/records/13692394'
ARCHIVE = 'coconut-09-2024.csv.zip'
UPSTREAM = f'https://zenodo.org/api/records/13692394/files/{ARCHIVE}/content'
MD5 = '439fca469248b2eb5194ffb75ddf62ed'
OUT = Path('/kaggle/working')


def stage():
    temporary = OUT / f'{ARCHIVE}.part'
    for attempt in range(3):
        digest = hashlib.md5(usedforsecurity=False)
        sha = hashlib.sha256()
        try:
            with requests.get(UPSTREAM, stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                with temporary.open('wb') as stream:
                    for block in response.iter_content(chunk_size=4 * 1024 * 1024):
                        stream.write(block)
                        digest.update(block)
                        sha.update(block)
            if digest.hexdigest() != MD5:
                raise ValueError('The archived COCONUT download did not match the release checksum')
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
        finally:
            if digest.hexdigest() != MD5:
                temporary.unlink(missing_ok=True)

    with ZipFile(temporary) as archive:
        member, = [name for name in archive.namelist() if name.endswith('.csv')
                   and not name.startswith('__MACOSX/')]
        with archive.open(member) as record:
            columns = next(csv.reader(io.TextIOWrapper(record, encoding='utf-8-sig')))
        if not any('smiles' in name.lower() for name in columns):
            raise ValueError('The public CSV lacks a recognizable structure column')
    asset = temporary.replace(OUT / ARCHIVE)
    report = dict(source='COCONUT 2.0 September 2024', doi='10.5281/zenodo.13692394',
                  release=RECORD, upstream=UPSTREAM, author_md5=MD5,
                  sha256=sha.hexdigest(), bytes=asset.stat().st_size, csv_member=member, columns=columns,
                  usage='Pinned raw structure input for an independent candidate-parent inventory')
    (OUT / 'coconut_parent_source.json').write_text(json.dumps(report, indent=2) + '\n')
    print('COCONUT_RAW_SOURCE_READY', report['bytes'], report['sha256'], flush=True)


if __name__ == '__main__':
    stage()

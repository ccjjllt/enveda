"""Download the sealed Kaggle arm outputs; keep the validation labels local."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time

import requests
from kagglesdk.kaggle_client import KaggleClient
from kagglesdk.kernels.types.kernels_api_service import (
    ApiDownloadKernelOutputRequest,
    ApiGetKernelSessionStatusRequest,
)


SCORER = Path(__file__).resolve().parents[1] / 'retrieval_pilot_score.py'
OWNER = 'jiyujia'


def retry(callback):
    for attempt in range(4):
        try:
            return callback()
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def api_call(token, method, request):
    with KaggleClient(api_token=token) as client:
        return getattr(client.kernels.kernels_api_client, method)(request)


def download(token, slug, version, path):
    request = ApiDownloadKernelOutputRequest()
    request.owner_slug = OWNER
    request.kernel_slug = slug
    request.version_number = version
    request.file_path = path.name

    def fetch():
        response = api_call(token, 'download_kernel_output', request)
        if hasattr(response, 'location'):
            response = requests.get(response.location, timeout=45)
        response.raise_for_status()
        path.write_bytes(response.content)

    retry(fetch)


def collect_arm(token, arm, job, output):
    slug, version = job['slug'], job['version']
    query = ApiGetKernelSessionStatusRequest()
    query.user_name = OWNER
    query.kernel_slug = slug
    query.version_label = f'v{version}'
    status = retry(lambda: api_call(token, 'get_kernel_session_status', query))
    print(arm, slug, f'v{version}', status.status.name, status.failure_message, flush=True)
    if status.status.name != 'COMPLETE':
        return False

    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / 'pilot_arm_receipt.json'
    download(token, slug, version, receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if receipt['arm'] != arm:
        raise ValueError(f'{slug} finished as the wrong intervention')
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, token, slug, version, output / name)
                   for name in receipt['hashes']]
        for future in futures:
            future.result()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('manifest', 'labels', 'out', 'metric-python'):
        parser.add_argument('--' + argument, required=True, type=Path)
    args = parser.parse_args()
    runs = args.out / 'arms'
    manifest = json.loads(args.manifest.read_text())
    secret = Path.home() / '.kaggle/access_token_jiyujia'
    token = os.getenv('KAGGLE_API_TOKEN') or secret.read_text().strip()
    worked = [collect_arm(token, arm, job, runs / arm) for arm, job in manifest['jobs'].items()]
    if len(worked) != 4 or not all(worked):
        raise SystemExit('Four complete and sealed Kaggle arms are required before scoring')
    subprocess.run([args.metric_python, SCORER, 'score', '--runs', runs,
                    '--labels', args.labels, '--out', args.out], check=True)


if __name__ == '__main__':
    main()

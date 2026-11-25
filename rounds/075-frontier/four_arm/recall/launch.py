"""Push the four blinded research arms and record the accepted Kaggle versions."""
import argparse
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import time

from build_recall import ARMS, FOUR_ARM, RUN_SLUG, build


OWNER = 'jiyujia'


def write_receipt(path, receipt):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def launch(args):
    args.out.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    panel = json.loads((args.panel / 'panel_manifest.json').read_text())
    for name in ('blind/test.parquet', 'blind/sample_submission.csv', 'private/holdout.json'):
        if sha256((args.panel / name).read_bytes()).hexdigest() != panel['output_sha256'][name]:
            raise ValueError(f'Frozen panel file changed: {name}')
    receipt = dict(owner=OWNER, timeout_seconds_per_arm=args.timeout,
                   device='Tesla T4 ×2', local_panel=str(args.panel.resolve()), panel=panel, jobs={})
    if args.receipt.exists():
        receipt = json.loads(args.receipt.read_text())
        if receipt['panel'] != panel or receipt['timeout_seconds_per_arm'] != args.timeout:
            raise ValueError('An existing launch receipt belongs to a different panel or runtime')

    token_path = Path.home() / '.kaggle/access_token_jiyujia'
    env = {**os.environ, 'KAGGLE_API_TOKEN': token_path.read_text().strip()}
    for key in ('KAGGLE_KEY', 'KAGGLE_USERNAME'):
        env.pop(key, None)

    packages = {
        arm: build(args.source, args.metadata, args.panel, args.out / arm, arm,
                   FOUR_ARM / 'pilot_stage.py', FOUR_ARM / 'retrieval_pilot_score.py')
        for arm in ARMS if args.replace_changed or arm not in receipt['jobs']
    }
    if args.replace_changed:
        for arm, prior in list(receipt['jobs'].items()):
            if prior['notebook_sha256'] != packages[arm]['notebook_sha256']:
                receipt.setdefault('superseded_jobs', []).append(prior)
                del receipt['jobs'][arm]
        write_receipt(args.receipt, receipt)

    pending = {arm: outcome for arm, outcome in packages.items() if arm not in receipt['jobs']}
    deadline = time.monotonic() + args.wait_for_slot
    for arm, outcome in pending.items():
        assert outcome['slug'] == f'{RUN_SLUG}-{ARMS[arm]}'
        output = args.out / arm
        command = [args.kaggle, 'kernels', 'push', '--path', str(output), '--timeout', str(args.timeout),
                   '--accelerator', 'NvidiaTeslaT4']
        announced_wait = False
        auth_transport_retries = 0
        while True:
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
            detail = result.stdout + '\n' + result.stderr
            found = re.search(r'Kernel version (\d+) successfully pushed', detail)
            # The CLI failed during authentication, before any notebook write was sent.
            auth_transport_failed = ('/IntrospectToken' in detail and
                                     any(error in detail for error in ('requests.exceptions.SSLError',
                                                                       'requests.exceptions.ConnectionError',
                                                                       'requests.exceptions.Timeout')))
            if not found and auth_transport_failed and auth_transport_retries < 6:
                auth_transport_retries += 1
                print(arm, 'retrying after a Kaggle authentication connection error', flush=True)
                time.sleep(2 * auth_transport_retries)
                continue
            auth_transport_retries = 0
            if found or time.monotonic() >= deadline or 'Maximum batch GPU session count' not in detail:
                break
            if not announced_wait:
                print(arm, 'waiting for a batch GPU slot', flush=True)
                announced_wait = True
            time.sleep(min(60, max(0, deadline - time.monotonic())))
        if not found:
            raise RuntimeError(f'{arm}: Kaggle did not return a version\n{detail.strip()}')
        version = int(found[1])
        receipt['jobs'][arm] = dict(**outcome, version=version, accepted_beijing=datetime.now().astimezone().isoformat(),
                                    url=f"https://www.kaggle.com/code/{OWNER}/{outcome['slug']}/versions/{version}")
        write_receipt(args.receipt, receipt)
        print(arm, outcome['slug'], version, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('source', 'metadata', 'panel', 'out', 'receipt'):
        parser.add_argument('--' + argument, type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=10_800)
    parser.add_argument('--wait-for-slot', type=int, default=0, metavar='SECONDS')
    parser.add_argument('--replace-changed', action='store_true')
    parser.add_argument('--kaggle', default=str(Path.home() / '.local/bin/kaggle'))
    launch(parser.parse_args())


if __name__ == '__main__':
    main()

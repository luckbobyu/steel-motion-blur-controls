"""Read-only CPU check of original DM seed-2 validation prediction caches.

Uses the existing frozen cutoffs; never refits a cutoff or invokes a detector.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import zipfile

from evaluate_predictions import counts


def collect_runtime(root, out):
    """Private snapshot of known experiment runtime locations; do not import."""
    folders = ('code/DM-YOLO', 'scripts/_blocksys113_lib', 'scripts/_blocksys113_official')
    result = {'found': [], 'missing': [], 'files': []}
    for relative in folders:
        source = safe_path(root, relative)
        if not source.is_dir():
            result['missing'].append(relative)
            continue
        result['found'].append(relative)
        for path in sorted(source.rglob('*')):
            rel = path.relative_to(root)
            if any(part in ('.git', '__pycache__', 'runs', 'wandb', '.venv') for part in rel.parts):
                continue
            if not path.is_file() or not (path.suffix.lower() in ('.py', '.yaml', '.yml') or
                                         path.name in ('LICENSE', 'LICENSE.txt', 'COPYING', 'requirements.txt', 'MANIFEST_SHA256.json')):
                continue
            path.resolve().relative_to(root.resolve())
            data = path.read_bytes()
            dest = out / 'private_runtime' / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            result['files'].append({'path': rel.as_posix(), 'sha256': hashlib.sha256(data).hexdigest()})
    return result


def safe_path(root, relative):
    target = (root / relative).resolve()
    target.relative_to(root.resolve())
    return target


def record_path(root, archived):
    text = str(archived).replace('\\', '/')
    if '/records/' in text:
        text = 'records/' + text.split('/records/', 1)[1]
    if not text.startswith('records/'):
        raise ValueError('Expected a records path: ' + text)
    return safe_path(root, text)


def verify(root, bundle):
    manifest = json.loads((bundle / 'configs/server_cpu_validation_inputs.json').read_text(encoding='utf-8'))
    checks = []
    for dataset in ('neu', 'gc10'):
        locks = json.loads((bundle / ('configs/dm_' + dataset + '_seed2_cutoffs.json')).read_text(encoding='utf-8'))
        for arm in ('low', 'high'):
            matches = [a for a in manifest['validation_jobs'] if a['dataset'] == dataset and a['arm'] == arm]
            keys = ['unseen_27_' + angle for angle in ('0', '45', '90', '135')]
            if sorted(a['condition'] for a in matches) != sorted(keys):
                raise ValueError('Expected four linear27 validation jobs for ' + dataset + '/' + arm)
            data = {}
            fingerprints = []
            for asset in matches:
                report_path = record_path(root, asset['report'])
                if hashlib.sha256(report_path.read_bytes()).hexdigest() != asset['sha256']:
                    raise ValueError('Validation job report hash mismatch')
                job = json.loads(report_path.read_text(encoding='utf-8'))
                if job.get('status') != 'PASS':
                    raise ValueError('Archived validation job is not PASS')
                if any(lock['checkpoint']['sha256'] != job['checkpoint']['sha256'] for lock in locks[arm].values()):
                    raise ValueError('Validation checkpoint differs from frozen cutoff checkpoint')
                original = job['prediction_file']
                path = record_path(root, original)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != job['artifacts'][original]:
                    raise ValueError('Prediction cache hash mismatch: ' + str(path))
                items = json.loads(path.read_text(encoding='utf-8'))
                if set(items) != {asset['condition']}:
                    raise ValueError('Unexpected validation condition; expected linear27')
                data.update(items)
                fingerprints.append({'condition':asset['condition'], 'sha256':digest})
            rows = [row for key in keys for row in data[key]]
            # Match each direction separately. The same image filename recurs
            # across directions; matching is always within an individual row.
            for budget, lock in locks[arm].items():
                actual = counts(rows, lock['threshold'])
                expected = lock['validation_counts']
                equal = all(actual[k] == expected[k] for k in ('tp', 'fp', 'gt', 'views'))
                checks.append({'dataset': dataset, 'arm': arm, 'budget': float(budget),
                               'threshold': lock['threshold'], 'actual': actual,
                               'expected': expected, 'match': equal,
                               'prediction_fingerprints': fingerprints})
                print(dataset, arm, budget, 'PASS' if equal else 'MISMATCH', flush=True)
    return {'status': 'PASS' if all(x['match'] for x in checks) else 'FAIL',
            'checks': checks, 'python': sys.version,
            'scope': 'Original archived DM seed-2 validation cache counts at frozen cutoffs; '
                     'no training, inference, AP, threshold refit or bootstrap.',
            'GPU_used': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--collect-runtime', action='store_true', help='Optional private source snapshot; already collected in the first run')
    args = parser.parse_args()
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = args.project_root / 'records' / ('repro_cpu_acceptance_' + stamp)
    out.mkdir(parents=True, exist_ok=False)
    try:
        report = verify(args.project_root.resolve(), args.bundle.resolve())
    except Exception as exc:
        report = {'status': 'ERROR', 'error': str(exc), 'GPU_used': False}
    if args.collect_runtime:
        try:
            report['private_runtime_collection'] = collect_runtime(args.project_root.resolve(), out)
        except Exception as exc:
            report['private_runtime_collection'] = {'error': str(exc)}
    (out / 'REPORT.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    archive = out.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(out).as_posix())
    print('feedback_zip=' + str(archive), flush=True)
    if report['status'] != 'PASS':
        print(json.dumps(report, indent=2))
        raise SystemExit(1)


if __name__ == '__main__':
    main()

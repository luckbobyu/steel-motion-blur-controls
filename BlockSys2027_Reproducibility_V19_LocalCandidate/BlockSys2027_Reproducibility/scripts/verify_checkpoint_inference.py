"""Re-evaluate existing DM seed-2 checkpoints; no training entry point.

Requires the author's pinned lab project. Runs one dataset on one physical GPU.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import zipfile

from evaluate_predictions import counts
from verify_server_predictions import record_path, safe_path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def compare_predictions(old, new):
    def indexed(rows):
        result = {r['image']: r for r in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate image within a condition')
        return result
    a, b = indexed(old), indexed(new)
    if set(a) != set(b):
        raise ValueError('Inference image set differs from archived reference')
    changed = 0
    for name in a:
        if a[name]['targets'] != b[name]['targets'] or a[name]['shape'] != b[name]['shape']:
            raise ValueError('Inference ground truth or image geometry differs')
        x, y = a[name]['predictions'], b[name]['predictions']
        close = len(x) == len(y) and all(len(p) == len(q) and
            all(math.isclose(v, w, rel_tol=1e-5, abs_tol=1e-4) for v, w in zip(p, q))
            for p, q in zip(x, y))
        changed += not close
    return {'images':len(a), 'images_with_prediction_difference_above_tolerance':changed,
            'diagnostic_only':'Ordered coordinates/scores atol=1e-4, rtol=1e-5; counts at all frozen budgets are the acceptance criterion.'}


def run(args, out, report):
    root = args.project_root.resolve()
    bundle = args.bundle.resolve()
    pins = json.loads((bundle/'configs/inference_source_pins.json').read_text())
    for row in pins['files']:
        if digest(safe_path(root,row['path'])) != row['sha256']:
            raise ValueError('Historical source differs: '+row['path'])
    report['source_files_verified'] = len(pins['files'])
    sys.dont_write_bytecode = True
    sys.path.insert(0,str(root/'scripts'))
    # This support module contains historical functions, but only staging,
    # boot_legacy and shape_test are invoked below, never its queue runners.
    import blocksys113_common as common
    if Path(common.__file__).resolve() != (root/'scripts/blocksys113_common.py').resolve():
        raise ValueError('Wrong project helper loaded')
    common.configure(args.gpu,out)
    os.chdir(out)
    import numpy as np
    locks = json.loads((bundle/f'configs/dm_{args.dataset}_seed2_cutoffs.json').read_text())
    manifest = json.loads((bundle/'configs/server_cpu_validation_inputs.json').read_text())
    reference = {}
    for arm in ('low','high'):
        reference[arm] = {}
        for row in manifest['validation_jobs']:
            if (row['dataset'],row['arm']) != (args.dataset,arm):
                continue
            path = record_path(root,row['report'])
            if digest(path) != row['sha256']:
                raise ValueError('Reference job report changed')
            item = json.loads(path.read_text(encoding='utf-8'))
            if item.get('status') != 'PASS':
                raise ValueError('Reference job is not PASS')
            if item['checkpoint']['sha256'] != locks[arm]['0.25']['checkpoint']['sha256']:
                raise ValueError('Reference checkpoint mismatch')
            original = item['prediction_file']
            pred = record_path(root,original)
            if digest(pred) != item['artifacts'][original]:
                raise ValueError('Reference prediction changed')
            data = json.loads(pred.read_text(encoding='utf-8'))
            if set(data) != {row['condition']}:
                raise ValueError('Expected one linear27 reference condition')
            reference[arm].update(data)
    keys = ['unseen_27_'+angle for angle in ('0','45','90','135')]
    if any(set(reference[arm]) != set(keys) for arm in reference):
        raise ValueError('Four linear27 directions required for each arm')
    report.update(dataset=args.dataset, seed=2, physical_gpu=args.gpu,
                  training_started=False, inference_started=False, checks=[], prediction_diagnostics=[])
    with common.reserve_gpu(root,args.gpu):
        torch, yolo = common.boot_legacy(root)
        # Fail closed if a future helper accidentally tries to start training.
        def reject_training(*a,**kw):
            raise RuntimeError('Training is disabled in this inference acceptance script')
        yolo.train = reject_training
        with common.staged(root,args.dataset,out) as (_,_,_,val,names,_,_):
            kernels = common.line_kernels(np,(27,))
            for arm in ('low','high'):
                checkpoint = locks[arm]['0.25']['checkpoint']
                weight = record_path(root,checkpoint['path'])
                if digest(weight) != checkpoint['sha256']:
                    raise ValueError('Frozen checkpoint changed')
                current = {}
                for key in keys:
                    dest = out/'inference'/arm/key
                    dest.mkdir(parents=True,exist_ok=False)
                    print('INFERENCE',args.dataset,arm,key,flush=True)
                    report['inference_started'] = True
                    item = common.core.shape_test(torch,yolo,0,weight,val,names,{key:kernels[key]},dest)
                    result = json.loads(Path(item['prediction_file']).read_text(encoding='utf-8'))
                    if set(result) != {key}:
                        raise ValueError('Unexpected inferred condition')
                    current.update(result)
                    audit = compare_predictions(reference[arm][key],result[key])
                    report['prediction_diagnostics'].append(dict(arm=arm,condition=key,**audit))
                    torch.cuda.empty_cache()
                pooled = [r for key in keys for r in current[key]]
                for budget,lock in locks[arm].items():
                    actual = counts(pooled,lock['threshold'])
                    expected = lock['validation_counts']
                    match = all(actual[k] == expected[k] for k in ('tp','fp','gt','views'))
                    report['checks'].append(dict(arm=arm,budget=float(budget),threshold=lock['threshold'],
                                                 actual=actual,expected=expected,match=match))
                    print('COUNT',args.dataset,arm,budget,'PASS' if match else 'MISMATCH',flush=True)
                (out/'progress.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    report['status'] = 'PASS' if all(r['match'] for r in report['checks']) and len(report['checks']) == 6 else 'NEEDS_REVIEW'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',required=True,type=Path)
    p.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--dataset',required=True,choices=('neu','gc10'))
    p.add_argument('--gpu',required=True,type=int,choices=(0,1))
    args=p.parse_args()
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out=args.project_root.resolve()/'records'/f'repro_inference_{args.dataset}_gpu{args.gpu}_{stamp}'
    out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    report={'status':'RUNNING','training_started':False,
            'scope':'DM seed2 fixed-checkpoint linear27 validation inference only; no training, cutoff refit, test inference or bootstrap.'}
    try:
        run(args,out,report)
    except Exception as exc:
        report.update(status='ERROR',error=str(exc),traceback=traceback.format_exc())
        print(report['traceback'],flush=True)
    report['elapsed_seconds']=time.perf_counter()-start
    (out/'REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            rel=path.relative_to(out)
            if path.is_file() and path.suffix.lower() in ('.json','.csv','.yaml','.txt') and not any(x in ('yolo_settings','matplotlib') for x in rel.parts):
                z.write(path,rel.as_posix())
    print('STATUS='+report['status'],flush=True)
    print('feedback_zip='+str(archive),flush=True)
    if report['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()

"""Read-only collection of missing reproduction inputs from the lab project.

Does not train, install packages, load checkpoints, modify historical records,
or publish anything. Checkpoints/prediction caches are hashed by default but
copied only when explicitly requested. The returned ZIP is private feedback,
not a public release candidate.
"""
import argparse,ast,csv,datetime,hashlib,importlib.metadata,json,os,platform,shutil,subprocess,sys,zipfile
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def inside(root,relative):
    rel=Path(relative)
    if rel.is_absolute() or '..' in rel.parts or ':' in str(rel):raise ValueError('Unsafe manifest path: '+str(relative))
    path=(root/rel).resolve();path.relative_to(root.resolve())
    return path

def collect(project,manifest,out,include_predictions=False,include_weights=False,inventory=True):
    project=project.resolve();out=out.resolve()
    if not project.is_dir():raise ValueError('Project directory does not exist')
    if out.exists():raise ValueError('Output already exists; choose a new directory')
    if out==project or out in project.parents:raise ValueError('Output cannot replace project or its ancestors')
    out.mkdir(parents=True)
    report={'purpose':'Private author feedback. Not a public artifact.','published':False,'modifies_original_records':False,'GPU_training':False,'python':sys.version,'platform':platform.platform(),'checked_assets':[],'copied_sources':[],'errors':[]}
    def cp(path,rel):
        dst=inside(out,rel);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dst)
    for a in json.loads(manifest.read_text(encoding='utf-8'))['assets']:
        row={k:a[k] for k in ('path','kind','sha256')};report['checked_assets'].append(row)
        try:
            path=inside(project,a['path'])
            if not path.is_file():row['status']='MISSING';continue
            row['bytes']=path.stat().st_size;row['actual_sha256']=digest(path)
            row['status']='MATCH' if row['actual_sha256']==a['sha256'] else 'HASH_MISMATCH'
            if row['status']=='MATCH' and ((a['kind']=='checkpoint' and include_weights) or (a['kind']=='predictions' and include_predictions) or a['kind']=='record'):
                cp(path,a['path']);row['copied']=True
        except (ValueError,OSError) as e:row['status']='ERROR';row['error']=str(e)
    # Find the import closure of the original study helpers, without executing them.
    roots=[project/'scripts',project/'scripts/_blocksys113_lib']
    todo=['blocksys113_common','blocksys113_legacy_worker','blocksys113_yolo11_worker','run_blocksys113_supplement_v1','run_blocksys113_yolo11_v1','run_blocksys115_eval_v1'];seen=set()
    while todo:
        module=todo.pop()
        if module in seen:continue
        seen.add(module)
        source=next((p/(module+'.py') for p in roots if (p/(module+'.py')).is_file()),None)
        if source is None:continue
        try:
            text=source.read_text(encoding='utf-8-sig');tree=ast.parse(text)
            rel=source.relative_to(project).as_posix();cp(source,rel)
            report['copied_sources'].append({'path':rel,'sha256':digest(source)})
            for node in ast.walk(tree):
                names=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and not node.level and node.module else []
                for name in names:
                    head=name.split('.')[0]
                    if any((r/(head+'.py')).is_file() for r in roots):todo.append(head)
        except (OSError,ValueError,SyntaxError) as e:report['errors'].append({'source':str(source.relative_to(project)),'error':str(e)})
    # Source runtime is collected privately for later provenance/license inspection.
    runtime=project/'ultralytics'
    if runtime.is_dir():
        for p in runtime.rglob('*'):
            if p.is_file() and p.suffix.lower() in ('.py','.yaml','.yml') and '__pycache__' not in p.parts:
                cp(p,p.relative_to(project));report['copied_sources'].append({'path':p.relative_to(project).as_posix(),'sha256':digest(p)})
    for name in ('LICENSE','LICENSE.txt','COPYING','requirements.txt'):
        p=project/name
        if p.is_file():cp(p,'project_metadata/'+name)
    packages=sorted(({'name':p.metadata.get('Name','unknown'),'version':p.version} for p in importlib.metadata.distributions()),key=lambda x:x['name'].lower())
    (out/'installed_packages.json').write_text(json.dumps(packages,indent=2),encoding='utf-8')
    if inventory:
        rows=[]
        for dataset in ('NEU_DET_author','GC10_DET_author'):
            folder=project/'datasets'/dataset
            if not folder.is_dir():report['errors'].append({'dataset':dataset,'error':'Expected dataset directory missing'});continue
            for p in sorted(folder.rglob('*')):
                if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.txt','.yaml','.yml'):
                    rows.append({'dataset':dataset,'path':p.relative_to(project).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)})
        if rows:
            with (out/'dataset_inventory.csv').open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        report['dataset_files_indexed']=len(rows)
    try:
        q=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True,timeout=20)
        report['gpu_description']=q.stdout.strip() if q.returncode==0 else 'unavailable'
    except (OSError,subprocess.TimeoutExpired):report['gpu_description']='unavailable'
    # Include original JSON reports beside referenced cached assets/checkpoints.
    folders=set()
    for row in report['checked_assets']:
        if row['status']=='MATCH':
            p=inside(project,row['path'])
            for parent in p.parents:
                if parent==project or parent==project/'records':break
                if (parent/'report.json').is_file():folders.add(parent)
    for folder in folders:
        cp(folder/'report.json',(folder/'report.json').relative_to(project))
        for name in ('audit.json','training_audit.json','args.yaml','protocol_before_training.json','protocol_before_test.json'):
            p=folder/name
            if p.is_file():cp(p,p.relative_to(project))
    counts={s:sum(x['status']==s for x in report['checked_assets']) for s in ('MATCH','MISSING','HASH_MISMATCH','ERROR')}
    report['asset_status_counts']=counts
    report['status']='COLLECTED' if not any(counts[s] for s in ('MISSING','HASH_MISMATCH','ERROR')) and not report['errors'] else 'NEEDS_REVIEW'
    report['scope']='Discovery and file integrity only. No checkpoint inference, threshold search, bootstrap or training was rerun.'
    (out/'COLLECTION_REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with zipfile.ZipFile(out.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(out.parent).as_posix())
    return report,out.with_suffix('.zip')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',type=Path,required=True);p.add_argument('--manifest',type=Path,default=Path(__file__).resolve().parent.parent/'configs/server_assets.json');p.add_argument('--output',type=Path);p.add_argument('--include-predictions',action='store_true');p.add_argument('--include-weights',action='store_true');p.add_argument('--skip-dataset-inventory',action='store_true')
    a=p.parse_args();out=a.output or a.project_root/'records'/('repro_assets_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    report,archive=collect(a.project_root,a.manifest,out,a.include_predictions,a.include_weights,not a.skip_dataset_inventory)
    print(json.dumps({'status':report['status'],'checked':report['asset_status_counts'],'feedback_zip':str(archive),'GPU_training':False},ensure_ascii=True,indent=2))

if __name__=='__main__':main()

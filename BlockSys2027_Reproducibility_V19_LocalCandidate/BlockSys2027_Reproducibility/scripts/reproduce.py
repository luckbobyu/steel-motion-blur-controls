"""Rebuild reported tables from exported seed metrics and archived path counts.

This entry does not train a detector, run inference, refit thresholds or regenerate
image-cluster confidence intervals. Python standard library only.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics as st
import sys

ROOT=Path(__file__).resolve().parents[1]

def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))

def write(path,rows):
    if not rows:raise ValueError('Refusing an empty result: '+str(path))
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def close(a,b,context,tolerance=1e-8):
    if abs(float(a)-float(b))>tolerance:
        raise ValueError(f'{context}: got {a}, expected {b}, tolerance {tolerance}')

def seeds012(rows):
    if len(rows)!=3 or {int(x['seed']) for x in rows}!={0,1,2}:
        raise ValueError('Expected exactly paired seeds 0,1,2')

def verify_manifest(root):
    manifest=json.loads((root/'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
    for name,digest in manifest.items():
        path=root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Release file missing or changed: '+name)
    return len(manifest)

def rebuild(root,output):
    d=root/'data/exported';output.mkdir(parents=True,exist_ok=True)
    seed=read(d/'core_by_seed.csv');absolute=read(d/'core_absolute_by_seed.csv')
    source=read(d/'T2_strength.csv');core=[]
    if len(seed)!=36 or len(source)!=12:raise ValueError('Expected 36 core seed-family endpoints and 12 summaries')
    for ref in source:
        keys=('detector','dataset','condition')
        rr=[x for x in seed if all(x[k]==ref[k] for k in keys)];seeds012(rr)
        aa=[x for x in absolute if all(x[k]==ref[k] for k in keys)];seeds012(aa)
        r={k:ref[k] for k in keys}
        for raw,mean,sd in [('recall_gain_pp','gain_mean','gain_SD'),('mAP_gain_pp','mAP_gain_mean_pp','mAP_gain_sample_SD_pp')]:
            r[mean]=st.mean(float(x[raw]) for x in rr);r[sd]=st.stdev(float(x[raw]) for x in rr)
            close(r[mean],ref[mean],mean);close(r[sd],ref[sd],sd)
        for name in ('low_recall','high_recall','low_FP','high_FP'):
            r[name]=st.mean(float(x[name]) for x in aa);close(r[name],ref[name],name)
            if name.endswith('recall'):
                r[name+'_SD']=st.stdev(float(x[name]) for x in aa)
                close(r[name+'_SD'],ref[name+'_SD'],name+'_SD')
        for x in aa:
            original=next(v for v in rr if v['seed']==x['seed'])
            close(float(x['high_recall'])-float(x['low_recall']),original['recall_gain_pp'],'within-seed recall difference')
        core.append(r)
    if not all(float(x['recall_gain_pp'])>0 and float(x['mAP_gain_pp'])>0 for x in seed):
        raise ValueError('Core positive-direction claim no longer holds')
    write(output/'table2_strength.csv',core)

    dist_seed=read(d/'distribution_by_seed.csv');dist=[]
    for ref in read(d/'distribution_summary.csv'):
        rr=[x for x in dist_seed if all(x[k]==ref[k] for k in ('detector','dataset','candidate','condition')) and float(x['budget'])==float(ref['budget'])];seeds012(rr)
        r={k:ref[k] for k in ('detector','dataset','candidate','condition','budget')}
        for raw,mean,sd in [('gain','recall_gain','recall_SD'),('map_gain','mAP_gain','mAP_SD')]:
            r[mean]=st.mean(float(x[raw]) for x in rr);r[sd]=st.stdev(float(x[raw]) for x in rr)
            close(r[mean],ref[mean],mean);close(r[sd],ref[sd],sd)
        r['FP_change']=st.mean(float(x['fp_change']) for x in rr);close(r['FP_change'],ref['FP_change'],'FP change')
        dist.append(r)
    write(output/'table3_all_distribution_endpoints.csv',dist)

    paths=[]
    for detector,prefix in [('DM-YOLO','dm'),('YOLOv8n','yolov8n')]:
        rows=read(root/f'data/operating_points/{prefix}_paths.csv')
        for r in rows:
            if r['detector']!=detector:raise ValueError('Detector mismatch in path export')
            close(int(r['tp'])/int(r['gt'])*100,r['recall_pct'],'path recall')
            close(int(r['fp'])/int(r['views']),r['fppi'],'path FP/view')
            if (r['exceeds_target']=='True')!=(float(r['fppi'])>0.25):raise ValueError('Strict path exceedance mismatch')
            lock=json.loads((root/f"configs/{prefix}_{r['dataset']}_seed{r['seed']}_cutoffs.json").read_text(encoding='utf-8'))[r['arm']][str(float(r['budget']))]
            close(r['threshold'],lock['threshold'],'frozen cutoff')
            v=lock['validation_counts'];close(v['fp']/v['views'],v['fppi'],'validation FP/view')
            if v['fppi']>float(r['budget'])+1e-12:raise ValueError('Recorded calibration exceeds budget')
        paths.extend(rows)
    budgets=[];pooled=[]
    tail_refs=read(d/'path_tail_summary.csv')
    for ref in read(d/'budget_summary.csv'):
        r={k:ref[k] for k in ('detector','dataset','family','budget')};arms={}
        for arm in ('low','high'):
            arm_values=[];exceeds=0;all_rates=[]
            for seed_index in (0,1,2):
                rr=[x for x in paths if all(x[k]==ref[k] for k in ('detector','dataset','family')) and int(x['seed'])==seed_index and x['arm']==arm and float(x['budget'])==float(ref['budget'])]
                if len(rr)!=12:raise ValueError('Expected twelve paths per seed and arm')
                tp=sum(int(x['tp']) for x in rr);fp=sum(int(x['fp']) for x in rr)
                gt=sum(int(x['gt']) for x in rr);views=sum(int(x['views']) for x in rr)
                recall=100*tp/gt;rate=fp/views
                pooled.append(dict(**{k:ref[k] for k in ('detector','dataset','family','budget')},seed=seed_index,arm=arm,tp=tp,fp=fp,gt=gt,views=views,recall_pct=recall,fppi=rate))
                arm_values.append((recall,rate));all_rates.extend(float(x['fppi']) for x in rr)
                exceeds+=sum(float(x['fppi'])>0.25 for x in rr)
            arms[arm]=arm_values
            r[arm+'_recall']=st.mean(x[0] for x in arm_values);r[arm+'_FP']=st.mean(x[1] for x in arm_values)
            r[arm+'_pass']=sum(x[1]<=0.25 for x in arm_values);r[arm+'_exceed']=exceeds;r[arm+'_max_FP']=max(all_rates)
            for suffix in ('recall','FP','pass','exceed'):close(r[arm+'_'+suffix],ref[arm+'_'+suffix],arm+' '+suffix)
            tail=next(x for x in tail_refs if all(x[k]==ref[k] for k in ('detector','dataset','family')) and x['arm']==arm and float(x['budget'])==float(ref['budget']))
            close(r[arm+'_max_FP'],tail['max_FP_view'],'maximum path FP/view')
        gain=[h[0]-l[0] for l,h in zip(arms['low'],arms['high'])]
        r['gain']=st.mean(gain);r['gain_SD']=st.stdev(gain)
        close(r['gain'],ref['gain'],'budget gain');close(r['gain_SD'],ref['gain_SD'],'budget gain SD')
        budgets.append(r)
    write(output/'table4_all_budgets_and_maxima.csv',budgets);write(output/'pooled_counts_by_seed.csv',pooled)
    cross=[];cross_seed=[]
    for detector in ('DM-YOLO','YOLOv8n'):
        for family in ('measured','new'):
            low=next(x for x in budgets if x['detector']==detector and x['dataset']=='gc10' and x['family']==family and float(x['budget'])==0.2)
            high=next(x for x in budgets if x['detector']==detector and x['dataset']=='gc10' and x['family']==family and float(x['budget'])==0.15)
            differences=[]
            for seed_index in (0,1,2):
                common=dict(detector=detector,dataset='gc10',family=family,seed=seed_index)
                lo=next(x for x in pooled if all(x[k]==v for k,v in common.items()) and x['arm']=='low' and float(x['budget'])==0.2)
                hi=next(x for x in pooled if all(x[k]==v for k,v in common.items()) and x['arm']=='high' and float(x['budget'])==0.15)
                delta=hi['recall_pct']-lo['recall_pct'];differences.append(delta)
                cross_seed.append(dict(**common,recall_gain=delta,low_FP=lo['fppi'],high_FP=hi['fppi']))
            cross.append(dict(detector=detector,dataset='gc10',family=family,low_budget=0.2,high_budget=0.15,low_recall=low['low_recall'],high_recall=high['high_recall'],gain=st.mean(differences),gain_SD=st.stdev(differences),low_FP=low['low_FP'],high_FP=high['high_FP'],low_exceed=low['low_exceed'],high_exceed=high['high_exceed'],low_max_FP=low['low_max_FP'],high_max_FP=high['high_max_FP']))
    write(output/'v17_cross_budget.csv',cross);write(output/'v17_cross_budget_by_seed.csv',cross_seed)

    transfers=read(d/'S7_original_recipe_transfer.csv');transfer=[]
    for dataset,detector in sorted({(x['dataset'],x['detector']) for x in transfers}):
        rr=[x for x in transfers if x['dataset']==dataset and x['detector']==detector];seeds012(rr)
        for arm,prefix in [('9/15','baseline'),('9/15/23','candidate')]:
            recalls=[float(x[prefix+'_recall_pct']) for x in rr];fps=[float(x[prefix+'_fppi']) for x in rr]
            transfer.append(dict(dataset=dataset,detector=detector,recipe=arm,recall=st.mean(recalls),recall_SD=st.stdev(recalls),FP=st.mean(fps),FP_SD=st.stdev(fps)))
    write(output/'table5_transfer.csv',transfer)

    scan_lookup={}
    for dataset in ('neu','gc10'):
        for r in read(root/f'data/scans/{dataset}_by_seed.csv'):
            if float(r['validation_budget'])!=1 or not r['contrast'].endswith('_cal23'):continue
            key=(dataset,int(r['seed']),r['model'],r['condition'])
            if key in scan_lookup:close(r['recall_pct'],scan_lookup[key]['recall_pct'],'duplicate scan endpoint')
            scan_lookup[key]=r
    scans=[]
    for ref in read(d/'scan_mean_SD.csv'):
        rr=[scan_lookup[(ref['dataset'],i,ref['model'],ref['condition'])] for i in (0,1,2)]
        r={k:ref[k] for k in ('dataset','model','condition')}
        for raw,prefix in [('recall_pct','recall_pct'),('fppi','fppi'),('mean_view_mAP50_95_pct','mean_view_mAP50_95_pct')]:
            r[prefix+'_mean']=st.mean(float(x[raw]) for x in rr);r[prefix+'_sample_SD']=st.stdev(float(x[raw]) for x in rr)
            close(r[prefix+'_mean'],ref[prefix+'_mean'],prefix+' scan mean');close(r[prefix+'_sample_SD'],ref[prefix+'_sample_SD'],prefix+' scan SD')
        scans.append(r)
    write(output/'figure2_scan.csv',scans)
    monotone=0;order=('single_match_9','motion_low_mean_only','motion_mean_only','single_match_23')
    for dataset in ('neu','gc10'):
        for seed_index in (0,1,2):
            for condition in ('unseen_27','unseen_31','unseen_35'):
                values=[float(scan_lookup[(dataset,seed_index,model,condition)]['recall_pct']) for model in order]
                if not all(b>a for a,b in zip(values,values[1:])):raise ValueError('Monotone scan claim failed')
                monotone+=3

    design=[]
    for dataset in ('neu','gc10'):
        policies=json.loads((root/f'configs/yolo11_{dataset}_protocol.json').read_text(encoding='utf-8'))['policies']
        total=sum(sum(x['reference_exposures'].values()) for x in policies.values())
        design.append(dict(dataset=dataset,blurred_exposures=total,low_moment=sum(x['low_moment']*sum(x['reference_exposures'].values()) for x in policies.values())/total,high_moment=sum(x['high_moment']*sum(x['reference_exposures'].values()) for x in policies.values())/total,three_moment=sum(x['three_mean']*sum(x['reference_exposures'].values()) for x in policies.values())/total))
    write(output/'table1_protocol_moments.csv',design)
    report={'status':'PASS','scope':'Arithmetic on exported metrics, archived path counts and recorded protocol values. Not GPU training/inference or bootstrap rerun.','core_seed_family_endpoints':len(seed),'distribution_summary_cells':len(dist),'path_records':len(paths),'budget_settings':len(budgets),'transfer_rows':len(transfer),'scan_summary_rows':len(scans),'monotone_scan_comparisons':monotone,'cross_budget_gc10_settings':len(cross),'all_cross_budget_seed_recall_gains_positive':all(x['recall_gain']>0 for x in cross_seed),'all_cross_budget_seed_FP_rates_lower':all(x['high_FP']<x['low_FP'] for x in cross_seed)}
    (output/'CHECKS.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path);p.add_argument('--skip-integrity',action='store_true',help='For deliberate development changes only; not a release verification')
    a=p.parse_args();root=a.root.resolve();out=(a.output or root/'outputs/tables').resolve()
    try:
        checked=0 if a.skip_integrity else verify_manifest(root)
        report=rebuild(root,out);report['release_files_checked']=checked
        print(json.dumps(report,indent=2))
    except (ValueError,KeyError,StopIteration,OSError,ZeroDivisionError) as e:
        print('FAIL: '+str(e),file=sys.stderr);return 1
    return 0

if __name__=='__main__':raise SystemExit(main())

"""Audit recall/FP counts from archived post-NMS prediction JSON.

Input is a JSON list of image rows, each with image, targets [x1,y1,x2,y2,class]
and predictions [x1,y1,x2,y2,score,class]. No inference, NMS or AP is performed.
"""
import argparse, json, math
from pathlib import Path

def iou(left,right):
    intersection=max(0.,min(left[2],right[2])-max(left[0],right[0]))*max(0.,min(left[3],right[3])-max(left[1],right[1]))
    area=lambda b:max(0.,b[2]-b[0])*max(0.,b[3]-b[1])
    union=area(left)+area(right)-intersection
    return intersection/union if union>0 else 0.

def events(images):
    result=[]
    for row in sorted(images,key=lambda x:x['image']):
        targets=row['targets'];remaining=set(range(len(targets)))
        for pred in sorted(row['predictions'],key=lambda x:-x[4]):
            if len(pred)!=6 or not all(math.isfinite(v) for v in pred):raise ValueError('Invalid prediction')
            candidates=[j for j in sorted(remaining) if targets[j][4]==pred[5]];hit=None
            if candidates:
                best=max(candidates,key=lambda j:iou(pred,targets[j]))
                if iou(pred,targets[best])>=.5:remaining.remove(best);hit=(row['image'],best)
            result.append((float(pred[4]),hit))
    return sorted(result,key=lambda x:-x[0])

def counts(images,threshold):
    if not images:raise ValueError('Expected non-empty view list')
    ee=events(images);tp=sum(score>=threshold and hit is not None for score,hit in ee);fp=sum(score>=threshold and hit is None for score,hit in ee)
    gt=sum(len(x['targets']) for x in images)
    if not gt:raise ValueError('This study export requires annotated targets')
    return {'tp':tp,'fp':fp,'gt':gt,'views':len(images),'recall_pct':100*tp/gt,'fppi':fp/len(images)}

def fit(images,budget):
    if not math.isfinite(budget) or budget<0 or not images:raise ValueError('Invalid validation budget or view list')
    ee=events(images);cap=math.floor(budget*len(images)+1e-9);best=(0,0,1.01);tp=fp=i=0
    while i<len(ee):
        score=ee[i][0]
        while i<len(ee) and ee[i][0]==score:
            tp+=ee[i][1] is not None;fp+=ee[i][1] is None;i+=1
        if fp>cap:break
        if (tp,-fp,score)>(best[0],-best[1],best[2]):best=(tp,fp,score)
    return best[2]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('predictions',type=Path);p.add_argument('--condition',help='Select one key when input is a condition-to-rows mapping');p.add_argument('--threshold',type=float);p.add_argument('--validation',type=Path,help='Optional separate validation file for fitting, never test data');p.add_argument('--validation-condition');p.add_argument('--budget',type=float);p.add_argument('--output',type=Path)
    a=p.parse_args();rows=json.loads(a.predictions.read_text(encoding='utf-8'));rows=rows[a.condition] if a.condition else rows
    if not isinstance(rows,list):p.error('Select a condition to obtain an image-row list')
    threshold=a.threshold
    if a.validation:
        if a.validation.resolve()==a.predictions.resolve():p.error('Validation and test files must be different')
        if threshold is not None or a.budget is None:p.error('Fitting needs --budget and no --threshold')
        vv=json.loads(a.validation.read_text(encoding='utf-8'));vv=vv[a.validation_condition] if a.validation_condition else vv
        threshold=fit(vv,a.budget)
    if threshold is None or not math.isfinite(threshold):p.error('Provide a finite frozen threshold or separate validation input')
    result={'threshold':threshold,**counts(rows,threshold),'scope':'Archived post-NMS predictions, IoU 0.50 counts only. No AP, training, inference or bootstrap reexecution.'}
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

"""Generate and verify study kernels without importing a detector runtime."""
import argparse, hashlib, json, math, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reference'))
from kernel_reference import line_kernel, moment, raster, read_paths

TRAJECTORY_SHA256='6c8676e1fd63809d42df05b0ec8733d56180adb26795c96427802f6ea963cf14'

def solve(path,target):
    low,high=0.,1.
    while moment(np,raster(np,path,high))<target:high*=1.5
    for _ in range(42):
        middle=(low+high)/2
        if moment(np,raster(np,path,middle))<target:low=middle
        else:high=middle
    kernel=raster(np,path,(low+high)/2)
    if abs(moment(np,kernel)-target)>2e-5:raise ValueError('Rasterized target moment not matched')
    return kernel

def new_paths():
    rng=np.random.default_rng(20261122);result={};design={}
    target=moment(np,line_kernel(np,27,0));t=np.linspace(-1,1,65)
    for family in ('arc','s_curve','zigzag'):
        for index,angle in enumerate((22.5,67.5,112.5,157.5)):
            amplitude=float(rng.uniform(.3,.7))
            y=amplitude*(t*t-1/3) if family=='arc' else amplitude*np.sin(np.pi*t) if family=='s_curve' else amplitude*(2/np.pi)*np.arcsin(np.sin(2*np.pi*t))
            path=np.column_stack((t,y));path-=path.mean(axis=0);theta=np.deg2rad(angle)
            path=path@np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]).T
            key=f'new_{family}_{index}';kernel=solve(path,target);result[key]=kernel
            design[key]={'angle':angle,'amplitude':amplitude,'moment':moment(np,kernel),'sha256':hashlib.sha256(kernel.tobytes()).hexdigest()}
    return result,design

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'outputs/kernels');p.add_argument('--trajectories',type=Path,help='Optional official, checksum-verified 500Frames.zip; never downloaded implicitly')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);arrays={};checks=[]
    for dataset in ('neu','gc10'):
        protocol=json.loads((ROOT/f'configs/yolo11_{dataset}_protocol.json').read_text(encoding='utf-8'))
        for angle_key,policy in protocol['policies'].items():
            angle=float(angle_key);direction=np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))]);path=np.arange(-6,7)[:,None]*direction
            for arm,target in [('low',policy['low_moment']),('high',policy['high_moment'])]:
                kernel=solve(path,target);arrays[f'{dataset}_{arm}_{angle_key}']=kernel
                checks.append({'dataset':dataset,'arm':arm,'angle':angle,'target':target,'actual':moment(np,kernel)})
            weighted=0;total=sum(policy['reference_exposures'].values())
            for length in (9,15,23):
                target=moment(np,line_kernel(np,length,angle));kernel=solve(path,target)
                arrays[f'{dataset}_three_{length}_{angle_key}']=kernel
                weighted+=policy['reference_exposures'][str(length)]*moment(np,kernel)/total
            if abs(weighted-policy['high_moment'])>2e-5:raise ValueError('Mixture moment mismatch')
    kernels,new_design=new_paths();arrays.update(kernels)
    reference=json.loads((ROOT/'configs/headroom_protocol.json').read_text(encoding='utf-8'))['new_synthetic_paths']
    if isinstance(reference,dict):
        for key,d in new_design.items():
            ref=reference[key]
            if abs(d['amplitude']-ref['amplitude'])>1e-12 or abs(d['moment']-ref['moment'])>2e-5:raise ValueError('New-path design differs from archived reference')
            # Bitwise kernel hashes can depend on NumPy/platform math. Report them,
            # but moment checks are the declared cross-platform numerical gate.
            d['archived_hash_match']=d['sha256']==ref['sha256']
    measured=0
    if a.trajectories:
        if hashlib.sha256(a.trajectories.read_bytes()).hexdigest()!=TRAJECTORY_SHA256:raise ValueError('Unexpected official trajectory archive checksum')
        ref=json.loads((ROOT/'configs/yolo11_neu_protocol.json').read_text(encoding='utf-8'))['measured_paths']
        for index,path,rms in read_paths(np,a.trajectories):
            for length in (23,27):
                target=moment(np,line_kernel(np,length,0));low,high=0.,float(np.sqrt(target))+1e-5
                for _ in range(38):
                    middle=(low+high)/2
                    if moment(np,raster(np,path,middle))<target:low=middle
                    else:high=middle
                kernel=np.flip(raster(np,path,(low+high)/2)).copy();key=f'measured_{length}_walk{index:02d}'
                if abs(moment(np,kernel)-ref[key]['actual_second_moment_px2'])>1e-5:raise ValueError('Measured kernel moment mismatch')
                arrays[key]=kernel;measured+=1
    np.savez_compressed(a.output/'kernels.npz',**arrays)
    result={'status':'PASS','numpy':np.__version__,'within_detector_arm_checks':len(checks),'new_paths':len(new_design),'measured_kernels_checked':measured,'measured_archive_supplied':bool(a.trajectories),'moment_tolerance':2e-5,'new_path_design':new_design,'arm_moments':checks,'scope':'Discrete kernel construction checks, not detector training or inference.'}
    (a.output/'KERNEL_CHECKS.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('new_path_design','arm_moments')},indent=2))

if __name__=='__main__':main()

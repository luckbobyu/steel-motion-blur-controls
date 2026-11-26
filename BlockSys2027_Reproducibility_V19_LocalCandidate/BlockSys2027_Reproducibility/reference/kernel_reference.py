"""Study-kernel reference functions extracted unchanged from archived study scripts.
See docs/PROVENANCE.md. Upstream detector implementations are not vendored.
"""

import math, hashlib, io, zipfile

from pathlib import Path

from types import SimpleNamespace

import numpy as np

ANGLES=(0,45,90,135)

control=SimpleNamespace(LENGTHS={2:9,3:15})


# Archived source: run_neu_motion_control_v1.py; SHA256 56b64ae70c24a97d6c692205a049e3673749b89f844dff98a172a446680c5059

def line_kernel(np, length, angle):
    """Deposit uniformly weighted line samples with bilinear pixel weights."""
    radius = length // 2
    kernel = np.zeros((length, length), dtype=np.float32)
    radians = np.deg2rad(angle)
    positions = np.arange(-radius, radius + 1, dtype=np.float64)
    for x, y in zip(radius + positions * np.cos(radians), radius + positions * np.sin(radians)):
        x, y = np.clip((x, y), 0, length - 1)
        left, top = int(x), int(y)
        for col, wx in ((left, 1 - (x - left)), (left + 1, x - left)):
            for row, wy in ((top, 1 - (y - top)), (top + 1, y - top)):
                if row < length and col < length:
                    kernel[row, col] += wx * wy
    return kernel / kernel.sum()


# Archived source: run_neu_matched_strength_v1.py; SHA256 ff3ec752eaf6d31d6607e01cebe8188fcc2f6fcee73cff5ab005709b01a4634b

def moment(np, kernel):
    positions = np.arange(kernel.shape[0], dtype=float) - kernel.shape[0] // 2
    return float((kernel * (positions[:, None]**2 + positions[None, :]**2)).sum())

def sample_slot(path, epoch):
    key = hashlib.sha256(f'neu-blur-v1:{epoch}:{Path(path).name}'.encode()).digest()
    if int.from_bytes(key[:4], 'big') % 100 >= 30:
        return None
    severity = 2 + (key[4] & 1)
    return severity, ANGLES[key[5] % len(ANGLES)], 23 if key[6] < 85 else control.LENGTHS[severity]


# Archived source: run_blur_paper_supplement_v1.py; SHA256 1ac7969c24dbbadbd59677a76efd491fb1a8012ae04c4ee1034f2ead48a738e9

def raster(np, xy, scale):
    canvas = 81
    center = canvas//2
    kernel = np.zeros((canvas, canvas), dtype=np.float64)
    for x, y in center + xy*scale:
        left, top = math.floor(x), math.floor(y)
        fx, fy = x-left, y-top
        for col, wx in ((left, 1-fx), (left+1, fx)):
            for row, wy in ((top, 1-fy), (top+1, fy)):
                if not (0 <= row < canvas and 0 <= col < canvas):
                    raise ValueError('Motion kernel exceeds its canvas')
                kernel[row, col] += wx*wy
    support = np.argwhere(kernel > 1e-14)
    extent = int(np.abs(support-center).max())
    region = kernel[center-extent:center+extent+1, center-extent:center+extent+1]
    return (region/region.sum()).astype(np.float32)


# Archived source: run_blur_paper_measured_v1.py; SHA256 1d56cb643e4f48180c318b6ca15ea5db3e4c273e9cfb59af35608d10d1164cc6

def read_paths(np, source):
    # Project the central point of the authors' fronto-parallel reference plane.
    # Rz has no effect on that point. The camera convention follows their MATLAB script.
    with zipfile.ZipFile(source) as archive:
        for index in range(1, 13):
            name = f'500Frames/walk{index}.txt'
            samples = np.loadtxt(io.BytesIO(archive.read(name)), delimiter=',')
            if samples.shape != (167, 6) or not np.isfinite(samples).all():
                raise ValueError('Unexpected published trajectory: ' + name)
            rx, ry = np.deg2rad(-samples[:, 0]), np.deg2rad(samples[:, 2])
            direction = np.column_stack((
                np.sin(ry)*np.cos(rx), -np.sin(rx), np.cos(ry)*np.cos(rx)
            ))
            xyz = 620*direction + samples[:, [3, 5, 4]]*np.array([-1., 1., -1.])
            xy = 3900*xyz[:, :2]/xyz[:, 2:3]
            xy -= xy.mean(axis=0)
            rms = float(np.sqrt(np.square(xy).sum(axis=1).mean()))
            if not np.isfinite(rms) or rms <= 0:
                raise ValueError('Degenerate projected camera motion: ' + name)
            yield index, xy/rms, rms

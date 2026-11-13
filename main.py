import glob
import os
from os import path as opath

import cv2
import numpy as np

from mypylibs.calibration import select_calibration_frames, calibrate, draw_corners

ROOT_IN = 'in'
ROOT_OUT = 'out'


def save_corner_overlays(cam, selected):
    """Save a contact sheet of the selected frames with detected corners drawn
    on the grayscale image the detector used."""
    thumbs = []
    for d in selected:
        gray = cv2.cvtColor(d['frame'], cv2.COLOR_BGR2GRAY)
        vis = draw_corners(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), d['corners'])
        thumbs.append(cv2.resize(vis, (216, 384)))
    rows = [np.hstack(thumbs[i:i + 5]) for i in range(0, len(thumbs), 5)]
    out = opath.join(ROOT_OUT, f'{cam}_corners.jpg')
    cv2.imwrite(out, np.vstack(rows))
    return out


def save_calibration(cam, selected, calib):
    K, D = calib['K'], calib['D']
    np.savetxt(opath.join(ROOT_OUT, f'{cam}_K.csv'), K, delimiter=',', fmt='%.6f')
    np.savetxt(opath.join(ROOT_OUT, f'{cam}_D.csv'), D[None], delimiter=',', fmt='%.8f',
               header='k1,k2,p1,p2,k3')

    w, h = calib['image_size']
    lines = [
        f'Camera {cam} calibration ({len(selected)} frames, image {w}x{h})',
        f'fx = {K[0, 0]:.3f}  fy = {K[1, 1]:.3f}',
        f'cx = {K[0, 2]:.3f}  cy = {K[1, 2]:.3f}',
        f'skew = {K[0, 1]:.3f} (OpenCV assumes zero skew)',
        f'k1 = {D[0]:.6f}  k2 = {D[1]:.6f}  k3 = {D[4]:.6f}',
        f'p1 = {D[2]:.6f}  p2 = {D[3]:.6f}',
        f'RMS reprojection error  = {calib["rms"]:.4f} px',
        f'Mean reprojection error = {calib["mean_error"]:.4f} px',
        'Per-frame mean error (frame index: px):',
    ]
    lines += [f'  {d["index"]:4d}: {e:.4f}'
              for d, e in zip(selected, calib['per_frame_error'])]
    with open(opath.join(ROOT_OUT, f'{cam}_calibration.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return lines


def process_camera(cam):
    video = opath.join(ROOT_IN, f'{cam}_calibration.mp4')

    print(f'[{cam}] Step 2: extracting checkerboard corners from {video}')
    selected = select_calibration_frames(video)
    print(f'[{cam}] {len(selected)} frames selected: {[d["index"] for d in selected]}')
    print(f'[{cam}] corner overlays -> {save_corner_overlays(cam, selected)}')

    print(f'[{cam}] Step 3: calibrating camera')
    calib = calibrate(selected)
    for line in save_calibration(cam, selected, calib)[:8]:
        print(f'[{cam}]   {line}')
    return calib


def main():
    os.makedirs(ROOT_OUT, exist_ok=True)
    videos = sorted(glob.glob(opath.join(ROOT_IN, '*_calibration.mp4')))
    if not videos:
        raise SystemExit(f'No *_calibration.mp4 found in {ROOT_IN}/')
    for v in videos:
        process_camera(opath.basename(v).split('_')[0])


if __name__ == '__main__':
    main()

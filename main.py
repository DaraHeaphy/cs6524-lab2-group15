import glob
import os
from os import path as opath

import cv2
import numpy as np

from mypylibs.calibration import select_calibration_frames, draw_corners

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


def process_camera(cam):
    video = opath.join(ROOT_IN, f'{cam}_calibration.mp4')
    print(f'[{cam}] Step 2: extracting checkerboard corners from {video}')
    selected = select_calibration_frames(video)
    print(f'[{cam}] {len(selected)} frames selected: '
          f'{[d["index"] for d in selected]}')
    print(f'[{cam}] corner overlays -> {save_corner_overlays(cam, selected)}')


def main():
    os.makedirs(ROOT_OUT, exist_ok=True)
    videos = sorted(glob.glob(opath.join(ROOT_IN, '*_calibration.mp4')))
    if not videos:
        raise SystemExit(f'No *_calibration.mp4 found in {ROOT_IN}/')
    for v in videos:
        process_camera(opath.basename(v).split('_')[0])


if __name__ == '__main__':
    main()

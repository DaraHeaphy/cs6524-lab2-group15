import cv2
import numpy as np

NUM_DISPARITIES = 96   # multiple of 16; covers objects down to ~15 cm at B ~ 3.5 cm
BLOCK_SIZE = 7
# Depth error grows as dZ = Z^2 / (f B) * dd. With f ~ 430 px and B ~ 3 cm, a
# 1 px matching error at d = 3 px already changes depth by ~30%, so smaller
# disparities (beyond ~4.5 m here) are rejected instead of shown as noise.
MIN_DISPARITY_PX = 3.0


def make_matcher(num_disp=NUM_DISPARITIES, block=BLOCK_SIZE):
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=num_disp, blockSize=block,
        P1=8 * 3 * block * block, P2=32 * 3 * block * block,
        disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def disparity(left, right, matcher, num_disp=NUM_DISPARITIES):
    """SGBM disparity (px). Both images are padded on the left by num_disp so
    that SGBM does not blank out the first num_disp columns of the result."""
    pad = lambda im: cv2.copyMakeBorder(im, 0, 0, num_disp, 0, cv2.BORDER_REPLICATE)
    d = matcher.compute(pad(left), pad(right)).astype(np.float32) / 16.0
    return d[:, num_disp:]


def rectified_depth(frame_i, frame_j, K, D, R, t_unit, baseline_cm, matcher):
    """Rectify frames i and j with the relative pose (R, t), compute disparity
    and convert it to depth Z = f * B / d in cm.

    Depth is expressed in the rectified frame of camera i. If camera j ends up
    to the left of camera i, both images are mirrored before matching so that
    i is always the left image of the pair.
    """
    h, w = frame_i.shape[:2]
    T = np.asarray(t_unit, float).reshape(3, 1) * baseline_cm
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K, D, K, D, (w, h), R, T, alpha=0)
    map_i = cv2.initUndistortRectifyMap(K, D, R1, P1, (w, h), cv2.CV_32FC1)
    map_j = cv2.initUndistortRectifyMap(K, D, R2, P2, (w, h), cv2.CV_32FC1)
    rect_i = cv2.remap(frame_i, *map_i, cv2.INTER_LINEAR)
    rect_j = cv2.remap(frame_j, *map_j, cv2.INTER_LINEAR)
    gi = cv2.cvtColor(rect_i, cv2.COLOR_BGR2GRAY)
    gj = cv2.cvtColor(rect_j, cv2.COLOR_BGR2GRAY)

    mirrored = P2[0, 3] > 0
    if mirrored:
        gi, gj = gi[:, ::-1].copy(), gj[:, ::-1].copy()
    d = disparity(gi, gj, matcher)
    if mirrored:
        d = d[:, ::-1]

    f = P1[0, 0]
    B = abs(P2[0, 3] / P2[0, 0])
    Z = np.full(d.shape, np.nan, np.float32)
    valid = d >= MIN_DISPARITY_PX
    Z[valid] = f * B / d[valid]
    return {'Z': Z, 'disparity': d, 'rect': rect_i, 'R1': R1, 'P1': P1,
            'f': f, 'B': B, 'Q': Q}


def board_depth_error(result, board_cam_i, win=2):
    """Compare the depth map with the board's true depth.

    board_cam_i: board corners in camera-i coordinates (cm). Returns the
    per-corner errors (measured - true, cm) and the mean true depth.
    """
    R1, P1, Z = result['R1'], result['P1'], result['Z']
    h, w = Z.shape
    Xr = (R1 @ board_cam_i.T).T
    uv = (P1[:, :3] @ Xr.T).T
    uv = uv[:, :2] / uv[:, 2:]
    errs = []
    for (u, v), z in zip(uv, Xr[:, 2]):
        u, v = int(round(u)), int(round(v))
        if win <= u < w - win and win <= v < h - win:
            patch = Z[v - win:v + win + 1, u - win:u + win + 1]
            if np.isfinite(patch).sum() >= patch.size // 2:
                errs.append(float(np.nanmedian(patch) - z))
    return np.array(errs), float(Xr[:, 2].mean())

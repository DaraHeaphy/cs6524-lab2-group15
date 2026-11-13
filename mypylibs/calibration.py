import cv2
import numpy as np

from mypylibs.video import read_video_frames

# The printed board has 9x6 squares -> 8x5 inner corners, 25 mm squares.
PATTERN_SIZE = (8, 5)
SQUARE_SIZE_CM = 2.5

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


def board_sharpness(gray, corners):
    """Variance of the Laplacian inside the board's bounding box (higher = sharper)."""
    x, y, w, h = cv2.boundingRect(corners.astype(np.float32))
    roi = gray[y:y + h, x:x + w]
    return cv2.Laplacian(roi, cv2.CV_64F).var()


def subpix_window(corners, pattern_size=PATTERN_SIZE):
    """Half-size of the cornerSubPix search window, scaled to the board's
    smallest corner spacing so the window never reaches a neighbouring corner."""
    grid = corners.reshape(pattern_size[1], pattern_size[0], 2)
    spacing = min(np.linalg.norm(np.diff(grid, axis=1), axis=2).min(),
                  np.linalg.norm(np.diff(grid, axis=0), axis=2).min())
    half = int(np.clip(round(0.4 * spacing), 2, 5))
    return (half, half)


def select_calibration_frames(video_path, n_frames=25, step=2,
                              pattern_size=PATTERN_SIZE):
    """Pick `n_frames` sharp, temporally spread frames where the board is found.

    Each frame is converted to grayscale and searched for the inner corners.
    Detected frames are split into `n_frames` equal time bins and the sharpest
    frame of each bin is kept, so the selection covers the whole sweep of the
    board. Corners of the kept frames are refined to sub-pixel accuracy.

    Returns a list of dicts: {'index', 'frame', 'corners', 'sharpness'}.
    """
    frames, indices = read_video_frames(video_path, step)
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
             | cv2.CALIB_CB_FAST_CHECK)

    detections = []
    for frame, idx in zip(frames, indices):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        if found:
            detections.append({'index': idx, 'frame': frame, 'corners': corners,
                               'sharpness': board_sharpness(gray, corners)})

    if len(detections) < n_frames:
        raise RuntimeError(f'Board found in only {len(detections)} frames of '
                           f'{video_path}; need {n_frames}.')

    bins = np.array_split(np.arange(len(detections)), n_frames)
    selected = [max((detections[i] for i in b), key=lambda d: d['sharpness'])
                for b in bins]

    for d in selected:
        gray = cv2.cvtColor(d['frame'], cv2.COLOR_BGR2GRAY)
        d['corners'] = cv2.cornerSubPix(gray, d['corners'],
                                        subpix_window(d['corners'], pattern_size),
                                        (-1, -1), SUBPIX_CRITERIA)
    return selected


def board_object_points(pattern_size=PATTERN_SIZE, square=SQUARE_SIZE_CM):
    """3D corner coordinates (i*s, j*s, 0) in cm, in OpenCV's row-major order."""
    obj = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square
    return obj


def calibrate(selected, pattern_size=PATTERN_SIZE, square=SQUARE_SIZE_CM):
    """Run cv2.calibrateCamera on the selected frames.

    Returns a dict with K, D (k1, k2, p1, p2, k3), the RMS error returned by
    OpenCV, the mean per-corner reprojection error, and per-frame errors.
    """
    obj = board_object_points(pattern_size, square)
    img_pts = [d['corners'] for d in selected]
    h, w = selected[0]['frame'].shape[:2]

    rms, K, D, rvecs, tvecs = cv2.calibrateCamera([obj] * len(img_pts), img_pts,
                                                  (w, h), None, None)

    per_frame, all_err = [], []
    for pts, r, t in zip(img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, r, t, K, D)
        # Corner arrays are (N,1,2) or (N,2) depending on the OpenCV version.
        err = np.linalg.norm(proj.reshape(-1, 2) - pts.reshape(-1, 2), axis=1)
        all_err.append(err)
        per_frame.append(err.mean())
    all_err = np.concatenate(all_err)

    return {'K': K, 'D': D.ravel(), 'rms': rms, 'mean_error': all_err.mean(),
            'per_frame_error': np.array(per_frame), 'image_size': (w, h),
            'rvecs': rvecs, 'tvecs': tvecs}


def draw_corners(image, corners, pattern_size=PATTERN_SIZE):
    """Return a copy of the image with the detected corners drawn on it."""
    vis = image.copy()
    cv2.drawChessboardCorners(vis, pattern_size, corners, True)
    return vis

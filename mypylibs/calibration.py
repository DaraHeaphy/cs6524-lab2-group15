import cv2
import numpy as np

# The printed board has 9x6 squares -> 8x5 inner corners, 25 mm squares.
PATTERN_SIZE = (8, 5)
SQUARE_SIZE_CM = 2.5

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


def find_corners(image, pattern_size=PATTERN_SIZE, refine=True):
    """Detect the inner checkerboard corners in a BGR or grayscale image.

    Returns (found, corners) where corners has shape (N, 1, 2), float32,
    ordered row by row as OpenCV returns them, or None if not found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        return False, None

    if refine:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)

    return True, corners


def find_corners_in_frames(frames, pattern_size=PATTERN_SIZE, refine=True):
    """Run find_corners on a list of frames.

    Returns (image_points, used_indices): the corner arrays for frames where
    the board was found, and the indices of those frames in the input list.
    """
    image_points, used = [], []
    for i, frame in enumerate(frames):
        found, corners = find_corners(frame, pattern_size, refine)
        if found:
            image_points.append(corners)
            used.append(i)
    return image_points, used


def read_video_frames(path, step=1):
    """Read every `step`-th frame of a video. Returns (frames, frame_indices)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    frames, indices, i = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(frame)
            indices.append(i)
        i += 1
    cap.release()
    return frames, indices


def board_sharpness(gray, corners):
    """Variance of the Laplacian inside the board's bounding box (higher = sharper)."""
    x, y, w, h = cv2.boundingRect(corners.astype(np.float32))
    roi = gray[y:y + h, x:x + w]
    return cv2.Laplacian(roi, cv2.CV_64F).var()


def select_calibration_frames(video_path, n_frames=25, step=2,
                              pattern_size=PATTERN_SIZE):
    """Pick `n_frames` sharp, temporally spread frames where the board is found.

    Detected frames are split into `n_frames` equal time bins and the sharpest
    frame of each bin is kept, so the selection covers the whole sweep of the
    board instead of clustering in one pose.

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

    # Sub-pixel refinement only on the frames we keep.
    for d in selected:
        gray = cv2.cvtColor(d['frame'], cv2.COLOR_BGR2GRAY)
        d['corners'] = cv2.cornerSubPix(gray, d['corners'], (11, 11), (-1, -1),
                                        SUBPIX_CRITERIA)
    return selected


def draw_corners(image, corners, pattern_size=PATTERN_SIZE):
    """Return a copy of the image with the detected corners drawn on it."""
    vis = image.copy()
    cv2.drawChessboardCorners(vis, pattern_size, corners, True)
    return vis

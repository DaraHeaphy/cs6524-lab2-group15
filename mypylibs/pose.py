import cv2
import numpy as np

# Frame t is paired with frame t + FRAME_GAP. Between truly consecutive frames
# (gap 1) a hand-held slide at 30 fps moves the camera so little that most
# disparities are below 2 px and the pose is dominated by noise. Checked
# against board (solvePnP) ground truth on scene 01-01, gap 20 gave the most
# valid pairs and the smallest translation-direction error.
FRAME_GAP = 20

# 'E': cv2.findEssentialMat (5-point, uses K inside RANSAC).
# 'F': cv2.findFundamentalMat then E = K^T F K (8-point, K applied afterwards).
# 'E' was ~3-7 deg more accurate in translation direction on 01-01.
POSE_METHOD = 'E'
RATIO_TEST = 0.75          # Lowe's ratio test for SIFT matches
RANSAC_THRESH_PX = 1.0     # max distance to epipolar line for an F inlier
MIN_INLIERS = 30


def detect_features(frames):
    """SIFT keypoints and descriptors for every frame (computed on grayscale)."""
    sift = cv2.SIFT_create()
    feats = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        feats.append(sift.detectAndCompute(gray, None))
    return feats


def match_features(feat0, feat1, ratio=RATIO_TEST):
    """Match SIFT descriptors with brute force + Lowe's ratio test.

    Returns two (N, 2) float32 arrays of corresponding pixel coordinates.
    """
    (kp0, des0), (kp1, des1) = feat0, feat1
    if des0 is None or des1 is None or len(kp0) < 2 or len(kp1) < 2:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des0, des1, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < ratio * m[1].distance]
    p0 = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1, 2)
    p1 = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return p0, p1


def estimate_pose(p0, p1, K, D, method=POSE_METHOD):
    """Relative pose of frame 1 w.r.t. frame 0 from matched pixel coordinates.

    Points are undistorted first (both F and E assume a pinhole camera). Then
    either E = findEssentialMat (method 'E'), or F = findFundamentalMat and
    E = K^T F K projected onto the essential manifold (method 'F'). Finally
    recoverPose picks the (R, t) of the four decompositions that puts the
    points in front of both cameras. Its default distance threshold (50x the
    baseline) rejects points too far away to have measurable parallax, so a
    pair with too little parallax fails here instead of returning a guess.
    t is a unit vector: monocular pose has no metric scale.

    Returns a dict, or None if the pose could not be estimated reliably.
    """
    if len(p0) < MIN_INLIERS:
        return None
    u0 = cv2.undistortPoints(p0.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
    u1 = cv2.undistortPoints(p1.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)

    F = None
    if method == 'F':
        F, mask = cv2.findFundamentalMat(u0, u1, cv2.FM_RANSAC, RANSAC_THRESH_PX, 0.999)
        if F is None or F.shape != (3, 3):
            return None
        E = K.T @ F @ K
        U, _, Vt = np.linalg.svd(E)
        E = U @ np.diag([1.0, 1.0, 0.0]) @ Vt
    else:
        E, mask = cv2.findEssentialMat(u0, u1, K, cv2.RANSAC, 0.999, RANSAC_THRESH_PX)
        if E is None or E.shape != (3, 3):
            return None

    inl = mask.ravel().astype(bool)
    if inl.sum() < MIN_INLIERS:
        return None

    n_front, R, t, _ = cv2.recoverPose(E, u0[inl], u1[inl], K)
    if n_front < MIN_INLIERS:
        return None

    return {'F': F, 'E': E, 'R': R, 't': t.ravel(),
            'n_matches': len(p0), 'n_inliers': int(inl.sum()), 'n_front': int(n_front),
            'p0': p0[inl], 'p1': p1[inl]}


def rotation_deg(R):
    return float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])))


def estimate_scene_poses(frames, K, D, gap=FRAME_GAP):
    """Estimate (R, t) for every pair (t, t + gap) of a scene video.

    Returns a list of dicts with keys 'i', 'j' and the estimate_pose() fields;
    pairs that failed have 'ok' = False.
    """
    feats = detect_features(frames)
    poses = []
    for i in range(len(frames) - gap):
        j = i + gap
        p0, p1 = match_features(feats[i], feats[j])
        est = estimate_pose(p0, p1, K, D)
        poses.append({'i': i, 'j': j, 'ok': est is not None, 'n_matches': len(p0),
                      **(est or {})})
    return poses


def draw_flow(frame, p0, p1):
    """Draw inlier matches as arrows from frame-t to frame-(t+gap) positions.
    Longer arrows = more parallax = closer to the camera."""
    vis = frame.copy()
    lengths = np.linalg.norm(p1 - p0, axis=1)
    top = max(np.percentile(lengths, 95), 1e-6)
    for a, b, length in zip(p0, p1, lengths):
        c = cv2.applyColorMap(np.uint8([[min(length / top, 1) * 255]]), cv2.COLORMAP_TURBO)[0, 0]
        cv2.arrowedLine(vis, tuple(map(int, a)), tuple(map(int, b)),
                        tuple(int(x) for x in c), 1, cv2.LINE_AA, tipLength=0.3)
    return vis

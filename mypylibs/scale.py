import cv2
import numpy as np

from mypylibs.calibration import PATTERN_SIZE, board_object_points, subpix_window

# recoverPose only gives the direction of t. The checkerboard standing in each
# scene gives the camera's metric position in every frame (solvePnP), which
# provides the real baseline B in cm between any two frames.
SMOOTH_DEGREE = 5     # polynomial degree for smoothing the camera path


class BoardTrack:
    """Camera poses relative to a static checkerboard visible in the scene."""

    def __init__(self, frames, K, D):
        obj = board_object_points()
        self.R, self.t, centres = {}, {}, {}
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, flags)
            if not found:
                continue
            corners = cv2.cornerSubPix(gray, corners, subpix_window(corners), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
            _, rvec, tvec = cv2.solvePnP(obj, corners, K, D, flags=cv2.SOLVEPNP_IPPE)
            R = cv2.Rodrigues(rvec)[0]
            self.R[i], self.t[i] = R, tvec.ravel()
            centres[i] = (-R.T @ tvec).ravel()

        self.frames = np.array(sorted(centres))
        self.ok = len(self.frames) > SMOOTH_DEGREE + 5
        if self.ok:
            # Per-frame PnP positions jitter (small board, small rotation errors
            # become lateral position errors); the camera slides smoothly, so fit
            # a low-order polynomial in time to each coordinate.
            P = np.array([centres[k] for k in self.frames])
            self.coef = [np.polyfit(self.frames, P[:, a], SMOOTH_DEGREE) for a in range(3)]
            fit = np.stack([np.polyval(c, self.frames) for c in self.coef], 1)
            self.jitter_cm = np.std(P - fit, axis=0)

    def covers(self, i):
        return self.ok and self.frames[0] <= i <= self.frames[-1]

    def centre(self, i):
        """Smoothed camera centre (cm, board coordinates) at frame i."""
        return np.array([np.polyval(c, i) for c in self.coef])

    def baseline(self, i, j):
        return float(np.linalg.norm(self.centre(j) - self.centre(i)))

    def relative_pose(self, i, j):
        """(R, unit t) of frame j w.r.t. frame i from the board, same convention
        as cv2.recoverPose: X_j = R X_i + t."""
        R = self.R[j] @ self.R[i].T
        t = -self.R[j] @ (self.centre(j) - self.centre(i))
        return R, t / np.linalg.norm(t)

    def board_points_in_camera(self, i):
        """Board corners (cm) in camera-i coordinates, or None if not detected."""
        if i not in self.R:
            return None
        return (self.R[i] @ board_object_points().T).T + self.t[i]

import cv2
import numpy as np
from scipy.spatial import cKDTree

MAX_DEPTH_CM = 400        # beyond this the 3 px disparity floor makes depth unreliable
EDGE_RATIO = 0.04         # max relative depth jump to a 3x3 neighbour (flying pixels)
SOR_NEIGHBOURS = 16       # statistical outlier removal: neighbours per point
SOR_STD = 2.0             # drop points whose mean neighbour distance > mean + SOR_STD * std


def depth_to_cloud(result, max_depth=MAX_DEPTH_CM):
    """Back-project a rectified depth map to a coloured 3D point cloud (cm).

    Points are filtered so the cloud shows the scene rather than stereo noise:
      1. only depths up to max_depth (far pixels have disparity of a few px);
      2. flying pixels removed: at object boundaries SGBM blends foreground
         and background, creating points smeared between them. A pixel is
         dropped if its depth differs from any 3x3 neighbour by > EDGE_RATIO;
      3. statistical outlier removal on the remaining 3D points.
    Returns (xyz float32 (N,3), rgb uint8 (N,3)).
    """
    Z = result['Z'].astype(np.float32)
    P1, rect = result['P1'], result['rect']
    h, w = Z.shape

    valid = np.isfinite(Z) & (Z <= max_depth)
    Zf = np.where(valid, Z, 0).astype(np.float32)
    k = np.ones((3, 3), np.uint8)
    zmax = cv2.dilate(np.where(valid, Z, 0).astype(np.float32), k)
    zmin = -cv2.dilate(np.where(valid, -Z, -1e6).astype(np.float32), k)
    # Any invalid neighbour also marks a boundary.
    hole_near = cv2.dilate((~valid).astype(np.uint8), k).astype(bool)
    smooth = (zmax - zmin) <= EDGE_RATIO * np.maximum(Zf, 1e-6)
    keep = valid & smooth & ~hole_near

    v, u = np.nonzero(keep)
    z = Z[v, u]
    f, cx, cy = P1[0, 0], P1[0, 2], P1[1, 2]
    x = (u - cx) * z / f
    y = (v - cy) * z / f
    xyz = np.stack([x, y, z], 1).astype(np.float32)
    rgb = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)[v, u]

    if len(xyz) > SOR_NEIGHBOURS:
        d, _ = cKDTree(xyz).query(xyz, k=SOR_NEIGHBOURS + 1)
        mean_d = d[:, 1:].mean(1)
        inlier = mean_d <= mean_d.mean() + SOR_STD * mean_d.std()
        xyz, rgb = xyz[inlier], rgb[inlier]

    # Camera coordinates have y down and z forward; flip y and z so the cloud
    # is upright in CloudCompare and its default view looks from the camera.
    xyz = xyz * np.array([1, -1, -1], np.float32)
    return xyz, rgb


def write_ply(path, xyz, rgb):
    """Binary little-endian PLY with float xyz and uchar rgb."""
    n = len(xyz)
    header = ('ply\nformat binary_little_endian 1.0\n'
              'comment units: centimetres, origin at the camera\n'
              f'element vertex {n}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\n'
              'end_header\n')
    data = np.empty(n, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                              ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    data['x'], data['y'], data['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data['r'], data['g'], data['b'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(data.tobytes())

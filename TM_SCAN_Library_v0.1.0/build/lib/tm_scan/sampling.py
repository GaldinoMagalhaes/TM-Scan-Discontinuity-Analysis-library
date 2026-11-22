"""Occupied-voxel sampling of window centers, including the planar fallback."""
import logging
import numpy as np
from scipy.spatial import cKDTree

def _voxel_centers_from_scan(scan_df, smooth_k, use_all_points=False):
    """Generates voxel grid nodes from scan_df."""
    P = scan_df[["X","Y","Z"]].to_numpy(dtype=np.float64)
    N = len(P)
    if N == 0:
        return np.empty((0,3), dtype=float), np.nan
    if use_all_points:
        return P.copy(), np.nan
    xmin, ymin, zmin = P.min(axis=0)
    xmax, ymax, zmax = P.max(axis=0)
    extent = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
    zero_tol = 64 * np.finfo(float).eps * max(float(extent.max()), np.finfo(float).tiny)
    if np.any(extent <= zero_tol):
        unique_points = np.unique(P, axis=0)
        if len(unique_points) < 2:
            return unique_points, np.nan
        # Ylimäki, Kannala and Heikkilä (2015): k=2 gives twice the median d_10.
        neighbor_count = min(10, len(unique_points) - 1)
        distances = cKDTree(unique_points).query(unique_points, k=[neighbor_count + 1])[0][:, 0]
        mean_spacing = float(np.median(distances))
        logging.getLogger(__name__).info("Numerically null bounding-box extent: using nearest-neighbor voxel spacing.")
    else:
        dx, dy, dz = max(xmax - xmin, 1e-12), max(ymax - ymin, 1e-12), max(zmax - zmin, 1e-12)
        volume = dx * dy * dz
        mean_spacing = (volume / max(N, 1)) ** (1/3)
    voxel_size = float(smooth_k) * mean_spacing
    origin = P.min(axis=0)
    V = np.floor((P - origin) / voxel_size).astype(np.int64)
    V = np.ascontiguousarray(V)
    dtype = np.dtype([('i', np.int64), ('j', np.int64), ('k', np.int64)])
    V_struct = V.view(dtype).reshape(-1)
    uniq, inv = np.unique(V_struct, return_inverse=True)
    G = len(uniq)
    counts = np.bincount(inv)
    sumx = np.bincount(inv, weights=P[:,0])
    sumy = np.bincount(inv, weights=P[:,1])
    sumz = np.bincount(inv, weights=P[:,2])
    centroids = np.column_stack((sumx/counts, sumy/counts, sumz/counts))
    grid_nodes = np.empty_like(centroids)
    for g in range(G):
        idxs = np.flatnonzero(inv == g)
        if idxs.size == 1:
            grid_nodes[g] = P[idxs[0]]
        else:
            diffs = P[idxs] - centroids[g]
            d2 = np.einsum('ij,ij->i', diffs, diffs)
            grid_nodes[g] = P[idxs[np.argmin(d2)]]
    return grid_nodes, voxel_size



"""Geometric helpers: polyline segments, point-segment distances and angle conversions."""
import numpy as np

def _as_array(p):
    """Converts a tuple/list to a 1D float np.array."""
    a = np.asarray(p, dtype=float).ravel()
    if a.size < 2:
        raise ValueError(f"Invalid point: {p}")
    return a


def _dist_from_cols(p0, p1):
    """Euclidean distance between X0 and X1 (uses min 2D/3D dimension)."""
    a = _as_array(p0)
    b = _as_array(p1)
    dim = min(a.size, b.size)
    return float(np.linalg.norm(a[:dim] - b[:dim]))


def _extract_poly_segments(bounds_gdf):
    """Extracts 3D segments from the polyline/multiline of the GeoDataFrame."""
    A_list, B_list = [], []
    for geom in bounds_gdf.geometry:
        if geom is None: continue
        
        def process_linestring(ln):
            coords = np.asarray(ln.coords, float)
            if coords.shape[1] == 2:
                coords = np.column_stack([coords, np.zeros(len(coords))])
            if len(coords) > 1:
                A_list.extend(coords[:-1])
                B_list.extend(coords[1:])

        if geom.geom_type == "LineString":
            process_linestring(geom)
        elif geom.geom_type == "MultiLineString":
            for ln in geom.geoms:
                process_linestring(ln)

    if not A_list: return np.empty((0, 3)), np.empty((0, 3))
    return np.asarray(A_list, float), np.asarray(B_list, float)


def _point_to_segments_dist3D(p, A, B):
    """Distances from point p to each segment Ai->Bi."""
    AB, AP = B - A, p[None, :] - A
    denom = np.einsum('ij,ij->i', AB, AB)
    denom = np.where(denom > 0, denom, 1.0)
    t = np.einsum('ij,ij->i', AP, AB) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = A + t[:, None] * AB
    d = np.linalg.norm(p[None, :] - proj, axis=1)
    return d


def normal_to_dip_dipdir(normal):
    """
    Convert a 3D normal vector to geological dip and dip-direction (degrees).
    Dip is measured from horizontal (0° horizontal, 90° vertical).
    Dip-direction is the azimuth of the line of greatest dip (0–360°).
    """
    n = np.asarray(normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)

    # Dip = arccos(|nz|) in degrees
    dip = np.degrees(np.arccos(abs(n[2])))

    # Project normal onto horizontal plane to get azimuth
    proj = np.array([n[0], n[1], 0.0])
    if np.linalg.norm(proj) < 1e-12:
        dip_dir = 0.0
    else:
        # Geological convention: azimuth measured clockwise from north
        azim = np.degrees(np.arctan2(proj[0], proj[1]))  # atan2(x, y) => 0°=north, 90°=east
        dip_dir = (azim + 360.0) % 360.0

    return dip, dip_dir


def dip_dipdir_to_normal(dip, dip_dir):
    """
    Converts geological Dip / Dip-direction into a unit normal vector (Nz >= 0).
    Frame: X=east, Y=north, Z=up; dip direction measured clockwise from north.
    """
    a = np.deg2rad(float(dip_dir))
    d = np.deg2rad(float(dip))
    # upward-pointing plane pole (normal)
    return np.array([np.sin(a)*np.sin(d), np.cos(a)*np.sin(d), np.cos(d)], dtype=float)

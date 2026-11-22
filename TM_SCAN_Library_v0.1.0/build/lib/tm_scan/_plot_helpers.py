"""Low-level drawing helpers for traces, boundaries and window outlines."""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString

def _coords3d_from_linestring(geom, *, endpoints=False):
    coords = np.asarray(geom.coords, dtype=float)
    if endpoints:
        coords = coords[[0, -1]]
    if coords.shape[1] == 2:
        return coords[:, 0], coords[:, 1], np.zeros_like(coords[:, 0])
    return coords[:, 0], coords[:, 1], coords[:, 2]


def _iter_lines_from_traces(traces):
    if isinstance(traces, gpd.GeoDataFrame) and 'geometry' in traces.columns:
        for geom in traces.geometry:
            if isinstance(geom, LineString): yield _coords3d_from_linestring(geom, endpoints=True)
            elif isinstance(geom, MultiLineString):
                for ln in geom.geoms: yield _coords3d_from_linestring(ln, endpoints=True)
    elif isinstance(traces, pd.DataFrame) and {'X0', 'X1'}.issubset(traces.columns):
        for _, row in traces.iterrows():
            p0, p1 = row['X0'], row['X1']
            if len(p0) < 3: p0 = list(p0) + [0]
            if len(p1) < 3: p1 = list(p1) + [0]
            yield [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]]


def _plot_traces(ax, traces_data, lw=1.0, alpha=0.75, color='black', label='Traces'):
    containers = list(traces_data.values()) if isinstance(traces_data, dict) else [traces_data]
    first = True
    for t in containers:
        for xs, ys, zs in _iter_lines_from_traces(t):
            ax.plot(xs, ys, zs, lw=lw, color=color, alpha=alpha, zorder=15, label=label if first else None)
            first = False


def _plot_boundary(ax, bounds_gdf, color='red', lw=0.9, linestyle='--', label='Boundary'):
    first = True
    for geom in bounds_gdf.geometry:
        if isinstance(geom, (LineString, MultiLineString)):
            geoms = geom.geoms if isinstance(geom, MultiLineString) else [geom]
            for ln in geoms:
                xs, ys, zs = _coords3d_from_linestring(ln)
                ax.plot(xs, ys, zs, color=color, lw=lw, linestyle=linestyle, zorder=20, label=label if first else None)
                first = False


def _circle_segments_oriented(center, normal, radius, npts=128):
    c, n = np.asarray(center, float), np.asarray(normal, float)
    n_norm = np.linalg.norm(n)
    if not np.isfinite(n_norm) or n_norm < 1e-12: return None
    n /= n_norm

    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, a); un = np.linalg.norm(u)
    if un < 1e-12: return None
    u /= un; v = np.cross(n, u)

    theta = np.linspace(0.0, 2 * np.pi, int(npts), endpoint=False)
    pts = c[None, :] + radius * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)
    return np.stack([pts, np.roll(pts, -1, axis=0)], axis=1)



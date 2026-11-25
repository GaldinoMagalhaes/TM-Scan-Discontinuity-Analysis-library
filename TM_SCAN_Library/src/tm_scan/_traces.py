"""Trace endpoints as plain arrays (first/last vertex only, as in the analyses)."""
import numpy as np


def endpoints(frame):
    """(A, B): (n, 3) endpoint arrays of a trace table (X0/X1 columns or line geometries)."""
    if "X0" in frame.columns and "X1" in frame.columns:
        A = np.asarray(frame["X0"].tolist(), float)[:, :3]
        B = np.asarray(frame["X1"].tolist(), float)[:, :3]
        return A.reshape(-1, 3), B.reshape(-1, 3)
    if "geometry" not in frame.columns:
        raise ValueError("Traces need X0/X1 columns or line geometries.")
    starts, ends = [], []
    for geometry in frame["geometry"]:
        if geometry is None or geometry.is_empty:
            continue
        parts = list(geometry.geoms) if geometry.geom_type == "MultiLineString" else [geometry]
        for part in parts:
            coordinates = np.asarray(part.coords, float)
            starts.append(coordinates[0, :3])
            ends.append(coordinates[-1, :3])
    return np.asarray(starts, float).reshape(-1, 3), np.asarray(ends, float).reshape(-1, 3)

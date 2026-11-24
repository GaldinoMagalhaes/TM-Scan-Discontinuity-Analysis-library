"""Log of undefined (NaN) estimates and of the cause of each one.

The estimators never replace an indeterminate value by a number: when a
calculation is undefined (for example, a division by zero because no trace
end point lies inside the window) the value is stored as NaN. This module
records, for every NaN in the window tables, the window, the quantity and the
cause, so that the conditions behind undefined values can be distinguished.

The causes are derived from the counts already stored in the tables and
reproduce the conditions tested by the estimators (sphere: mu = (pi R / 2)
N_intersections / Terminations_inside; rectangle: Mauldon 1998, Eq. 24).
Nothing in the estimates is changed.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("tm_scan.undefined")

CAUSES = {
    "no_traces": "no trace intersects the window",
    "no_internal_endpoints": "no trace end point inside the window (division by zero)",
    "no_boundary_crossings": "every trace is contained in the window (Mauldon Eq. 24 numerator is zero)",
    "family_parallel_to_window": "family plane parallel to the window plane (intersection direction undefined)",
    "undefined_window_normal": "window normal undefined (fewer than 3 points or non-finite plane)",
    "zero_window_area": "window area is zero",
    "unclassified": "undefined value without a recognised cause",
}

COLUMNS = ["family", "window_kind", "window_id", "X", "Y", "Z", "quantity", "cause_code", "cause",
           "N_traces", "N_boundary", "N_internal"]

_NORMAL = ["Nx", "Ny", "Nz"]


def _is_nan(frame, column):
    if column not in frame:
        return np.zeros(len(frame), bool)
    return ~np.isfinite(pd.to_numeric(frame[column], errors="coerce").to_numpy(float))


def _sphere_mu_cause(frame):
    n = frame["N_traces"].to_numpy()
    inside = frame["Terminations_inside"].to_numpy()
    return np.select([n == 0, inside == 0], ["no_traces", "no_internal_endpoints"], "unclassified")


def _rectangle_mu_cause(frame, family_normal):
    n = frame["N_traces"].to_numpy()
    nt = frame["N_transecting"].to_numpy()
    nc = frame["N_contained"].to_numpy()
    normal = frame[_NORMAL].to_numpy(float)
    bad_normal = ~np.isfinite(normal).all(1)
    if family_normal is None:
        parallel = np.zeros(len(frame), bool)
    else:
        cross = np.cross(np.nan_to_num(normal), np.asarray(family_normal, float))
        parallel = ~bad_normal & (np.linalg.norm(cross, axis=1) < 1e-12)
    return np.select([bad_normal, parallel, n == 0, n - nt + nc <= 0, n + nt - nc <= 0],
                     ["undefined_window_normal", "family_parallel_to_window", "no_traces",
                      "no_internal_endpoints", "no_boundary_crossings"], "unclassified")


def undefined_log(tables, window_kind, orientations=None):
    """One row per undefined value: window, quantity (mu, P20, P21, normal) and cause.

    tables       : {family: window table} as in AnalysisResult.tables
    window_kind  : 'sphere' or 'rectangle'
    orientations : {family: (dip, dip_dir)} used by the rectangle (optional)
    """
    from .geometry import dip_dipdir_to_normal
    rows = []
    for family, frame in tables.items():
        if frame is None or not len(frame):
            continue
        frame = frame.reset_index(drop=True)
        ident = frame["Sphere_ID"] if window_kind == "sphere" else frame["Rectangle_ID"]
        if window_kind == "sphere":
            boundary, internal = frame["N_intersections"], frame["Terminations_inside"]
            mu_cause = _sphere_mu_cause(frame)
        else:
            boundary, internal = frame["N_transecting"], frame["N_contained"]
            normal = dip_dipdir_to_normal(*orientations[family]) if orientations and family in orientations else None
            mu_cause = _rectangle_mu_cause(frame, normal)
        checks = [("μ", _is_nan(frame, "μ"), mu_cause),
                  ("P21", _is_nan(frame, "P21"), np.full(len(frame), "zero_window_area")),
                  ("P20", _is_nan(frame, "P20"), np.full(len(frame), "zero_window_area")),
                  ("normal", np.column_stack([_is_nan(frame, c) for c in _NORMAL]).any(1),
                   np.full(len(frame), "undefined_window_normal"))]
        for quantity, mask, cause in checks:
            for i in np.flatnonzero(mask):
                rows.append([family, window_kind, int(ident.iloc[i]), frame.X.iloc[i], frame.Y.iloc[i],
                             frame.Z.iloc[i], quantity, cause[i], CAUSES[cause[i]],
                             int(frame.N_traces.iloc[i]), int(boundary.iloc[i]), int(internal.iloc[i])])
    log = pd.DataFrame(rows, columns=COLUMNS)
    _report(log, tables)
    return log


def summarize(log):
    """Counts of undefined values per family, quantity and cause."""
    if log is None or not len(log):
        return pd.DataFrame(columns=["family", "quantity", "cause_code", "cause", "windows"])
    return (log.groupby(["family", "quantity", "cause_code"], sort=False).size().rename("windows")
            .reset_index().assign(cause=lambda f: f.cause_code.map(CAUSES))
            [["family", "quantity", "cause_code", "cause", "windows"]])


def _report(log, tables):
    for family, frame in tables.items():
        part = log[log.family == family]
        if not len(part):
            logger.info("%s: no undefined values in %d windows.", family, len(frame))
            continue
        for quantity, group in part.groupby("quantity", sort=False):
            causes = ", ".join(f"{code}: {count}" for code, count in group.cause_code.value_counts().items())
            logger.info("%s: %d of %d windows with undefined %s (%s).", family, len(group), len(frame),
                        quantity, causes)

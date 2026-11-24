"""Window-scale roughness control, tied to the sampling windows (diagnostic only).

Detector 1 (cloud only), per window of radius R: iota = xi_R / R <= tau and
projected coverage >= min_cover, with xi_R the SD of the point distances to
the best-fit plane (sigma(D) of Lague et al. 2013, D = 2R). A rectangle reads
iota at min(size_u, size_v), the neighbourhood used to fit its plane, so the
smallest admissible size R_adm is both a sphere radius and a rectangle side.
Scaling law xi_r = A r^H after Sadeghi & Aubertin (2025).

Detector 2 (cloud + traces): predicted retention eta, the expected fraction
of trace length a window keeps, from the distance s between each trace chord
and the surface: sphere 1 - (s^2 + xi_R^2)/R^2; rectangle (side W = 2R)
fraction of chord length with |s| <= delta (slab half-thickness).

Nothing is corrected. Windows are only FLAGGED; rejection happens only with
analyze(..., roughness_filter=True).
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DEFAULT_TAU, DEFAULT_MIN_COVER = 0.08, 0.80
DEFAULT_SLAB_MODE = "max"          # same default as the rectangle (rectangle_options delta_mode)


# ------------------------------------------------------------ point cloud ---
def voxel_thin(points, edge):
    """One point per voxel of the given edge (the one closest to the voxel centroid)."""
    key = np.floor((points - points.min(0)) / edge).astype(np.int64)
    inverse = np.unique(key, axis=0, return_inverse=True)[1].ravel()
    count = np.bincount(inverse)
    centroid = np.column_stack([np.bincount(inverse, points[:, j]) / count for j in range(3)])
    order = np.lexsort((((points - centroid[inverse]) ** 2).sum(1), inverse))
    return points[order[np.r_[True, np.diff(inverse[order]) > 0]]]


def fine_normals(points, k=16):
    """KD-tree of the cloud and local normal of every point (PCA of k neighbours)."""
    tree = cKDTree(points)
    Q = points[tree.query(points, k=min(k + 1, len(points)))[1]]
    Q = Q - Q.mean(1, keepdims=True)
    return tree, np.linalg.eigh(np.einsum("nki,nkj->nij", Q, Q))[1][:, :, 0]


def _slab(abs_d, mode=DEFAULT_SLAB_MODE):
    """Slab half-thickness from |d|, as RectangleAnalyzer._delta_for_window."""
    if mode == "max":
        return float(abs_d.max())
    return float(np.percentile(abs_d, 99.0 if mode == "p99" else 95.0))


def window_roughness(points, tree, centers, radius, spacing, min_points=20, slab_mode=DEFAULT_SLAB_MODE):
    """Per spherical window: xi, iota = xi/R, delta (slab half-thickness of |d| to the
    plane through the centre, `slab_mode`) and cover (occupied fraction of the projected disc)."""
    out = np.full((len(centers), 4), np.nan)
    for i, index in enumerate(tree.query_ball_point(centers, radius)):
        if len(index) < min_points:
            continue
        selected = points[index]
        X = selected - selected.mean(0)
        w, V = np.linalg.eigh(X.T @ X / len(index))
        xi = np.sqrt(max(w[0], 0.0))
        delta = _slab(np.abs((selected - centers[i]) @ V[:, 0]), slab_mode)
        cells = np.unique(np.floor(X @ V[:, 1:] / (2 * spacing)), axis=0)
        out[i] = xi, xi / radius, delta, min(1.0, len(cells) * (2 * spacing) ** 2 / (np.pi * radius ** 2))
    return pd.DataFrame(out, columns=["xi", "iota", "delta", "cover"])


def roughness_scan(points, centers, radii, spacing, per_radius=16, slab_mode=DEFAULT_SLAB_MODE):
    """Multi-scale scan; at radius R the cloud is used at resolution max(spacing, R/16)."""
    out = {}
    for radius in radii:
        resolution = max(spacing, radius / per_radius)
        thinned = points if resolution <= spacing else voxel_thin(points, resolution)
        out[float(radius)] = window_roughness(thinned, cKDTree(thinned), centers, radius, resolution,
                                              slab_mode=slab_mode)
    return out


def admissible_radius(scan, tau=DEFAULT_TAU, min_cover=DEFAULT_MIN_COVER):
    """Smallest radius R with iota <= tau at R and at every larger tested radius, and
    coverage >= min_cover at R only (a larger radius leaving the cloud does not void it).

    Returns (R_adm per centre, NaN if none; boolean table centre x radius with
    iota <= tau and cover >= min_cover at each radius; reason per centre:
    'ok' | 'rough' | 'low_cover' | 'few_points')."""
    radii = np.asarray(sorted(scan))
    iota = np.column_stack([scan[R].iota.to_numpy(float) for R in radii])
    cover = np.column_stack([scan[R].cover.to_numpy(float) for R in radii])
    smooth = iota <= tau
    covered = cover >= min_cover
    ok = smooth & covered
    tail = np.cumprod(smooth[:, ::-1], axis=1)[:, ::-1].astype(bool) & covered
    r_adm = np.where(tail.any(1), radii[tail.argmax(1)], np.nan)
    reason = np.where(tail.any(1), "ok",
                      np.where(np.isnan(iota).all(1), "few_points",
                               np.where(smooth.any(1) & ~covered.any(1), "low_cover", "rough")))
    return r_adm, ok, reason


def power_law(radii, xi, fit_range=None):
    """xi_r = A r^H fitted on the radii inside fit_range = (lo, hi) (all if None); returns (A, H)."""
    radii, xi = np.asarray(radii, float), np.asarray(xi, float)
    valid = np.isfinite(xi) & (xi > 0)
    if fit_range is not None:
        valid &= (radii >= fit_range[0]) & (radii <= fit_range[1])
    if valid.sum() < 2:
        return np.nan, np.nan
    H, lnA = np.polyfit(np.log(radii[valid]), np.log(xi[valid]), 1)
    return float(np.exp(lnA)), float(H)


def r_min(A, H, tau):
    """Radius from which xi_R / R <= tau under xi = A r^H (requires H < 1)."""
    return (A / tau) ** (1.0 / (1.0 - H)) if np.isfinite(A) and H < 1 else np.nan


# ---------------------------------------------- predicted retention ---
def _trace_endpoints(dataset, families=None):
    from ._traces import endpoints
    names = list(dataset.traces) if families is None else list(families)
    parts = [endpoints(dataset.traces[name]) for name in names]
    return np.vstack([a for a, _ in parts]), np.vstack([b for _, b in parts])


def trace_samples(A, B, points, tree, normals, step):
    """Sample each trace chord every `step`; foot = nearest cloud point of each sample."""
    q, w = [], []
    for a, b in zip(A, B):
        length = np.linalg.norm(b - a)
        n = max(int(np.ceil(length / step)), 1)
        q.append(a + (b - a) * ((np.arange(n) + 0.5) / n)[:, None])
        w.append(np.full(n, length / n))
    q, w = np.vstack(q), np.concatenate(w)
    foot = tree.query(q)[1]
    s = np.einsum("ij,ij->i", q - points[foot], normals[foot])
    return dict(q=q, w=w, foot=foot, s=s, ftree=cKDTree(points[foot]))


def predicted_retention(centers, samples, radius, xi, delta):
    """Length-weighted predicted retention per window, from the chord samples whose foot
    lies within `radius` of the centre: sphere 1 - (s^2 + xi^2)/R^2 (section of the sphere
    at distance s); rectangle |s| <= delta (slab tolerance). NaN = no trace sample.

    Returns a DataFrame with eta_sphere and eta_rect (one row per centre)."""
    out = np.full((len(centers), 2), np.nan)
    for i, index in enumerate(samples["ftree"].query_ball_point(centers, radius)):
        if len(index):
            s, w = samples["s"][index], samples["w"][index]
            out[i, 0] = np.average(np.clip(1 - (s ** 2 + xi[i] ** 2) / radius ** 2, 0, None), weights=w)
            out[i, 1] = np.average(np.abs(s) <= delta[i], weights=w)
    return pd.DataFrame(out, columns=["eta_sphere", "eta_rect"])


# --------------------------------------------------------- high level ---
@dataclass
class RoughnessResult:
    """Roughness of sampling windows.

    table      : X, Y, Z, xi, iota, delta, cover, admissible (at `radius`), R_adm, reason
    by_radius  : per radius R -> median iota, admissible fraction of the sphere (iota at R)
                 and of the rectangle with side 2R (iota at 2R), and, with retention=True,
                 mean predicted retention eta_sphere and eta_rect
    scaling    : radius -> xi quantiles (25, 50, 75 %); the power law is fitted on fit_range
    """
    table: pd.DataFrame
    radius: float
    radii: tuple
    tau: float
    min_cover: float
    spacing: float
    by_radius: pd.DataFrame
    scaling: pd.DataFrame
    A: float
    H: float
    fit_range: tuple = None
    filtered: bool = False
    length_unit: str = "m"
    scan: dict = field(default_factory=dict, repr=False)

    @property
    def admissible(self):
        return self.table["admissible"].to_numpy(bool)

    @property
    def r_min(self):
        return r_min(self.A, self.H, self.tau)

    def summary(self):
        return pd.Series({"windows": len(self.table), "radius": self.radius, "tau": self.tau,
                          "min_cover": self.min_cover, "admissible_fraction": float(self.admissible.mean())
                          if len(self.table) else np.nan,
                          "median_iota": float(self.table.iota.median()) if len(self.table) else np.nan,
                          "A": self.A, "H": self.H, "R_min": self.r_min, "filter_applied": self.filtered})

    def plot(self, **kwargs):
        return plot_roughness(self, **kwargs)


def window_radius(window):
    """Roughness radius of a sampling window: R for spheres, min(size_u, size_v) for
    rectangles (the neighbourhood RectangleAnalyzer uses to fit the window plane)."""
    if window.kind == "sphere":
        return float(window.radius)
    return float(min(window.size_u, window.size_v))


def _assess(scan_xyz, centers, radius, *, radii=None, tau=DEFAULT_TAU, min_cover=DEFAULT_MIN_COVER,
            scaling=True, length_unit="m", traces=None, retention=False, fit_range=None,
            slab_mode=DEFAULT_SLAB_MODE, rectangle_radii=True):
    centers = np.asarray(centers, float).reshape(-1, 3)
    radius = float(radius)
    radii = sorted({float(r) for r in (radii if radii is not None else [radius])} | {radius})
    if min(radii) <= 0 or not 0 < tau or not 0 <= min_cover <= 1:
        raise ValueError("radii and tau must be positive; min_cover within [0, 1].")
    if slab_mode not in ("p95", "p99", "max"):
        raise ValueError("slab_mode must be 'p95', 'p99' or 'max'.")
    origin = scan_xyz.mean(0)
    working = min(radii) / 10                      # R >= 10 x spacing
    points = voxel_thin(scan_xyz - origin, working)
    tree = cKDTree(points)
    spacing = max(working, float(np.median(tree.query(points[::20], k=2)[0][:, 1])))
    local = centers - origin
    scan_radii = set(radii)
    if rectangle_radii or retention:               # rectangle of side W = 2R reads iota and delta at 2R
        scan_radii |= {2 * R for R in radii}
    if scaling:
        extra = np.round(np.geomspace(0.5 * min(radii), 2 * max(radii), 6), 3)
        scan_radii |= set(float(r) for r in extra)
    scan_radii = sorted(scan_radii)
    columns = ["xi", "iota", "delta", "cover"]
    scan = roughness_scan(points, local, scan_radii, spacing, slab_mode=slab_mode) if len(local) else {
        R: pd.DataFrame(columns=columns, dtype=float) for R in scan_radii}
    main = scan[radius]
    if len(local):
        r_adm, ok, reason = admissible_radius({R: scan[R] for R in radii}, tau, min_cover)
    else:
        r_adm, ok, reason = np.empty(0), np.empty((0, len(radii)), bool), np.empty(0, object)
    column = radii.index(radius)
    table = pd.DataFrame(centers, columns=["X", "Y", "Z"]).assign(
        xi=main.xi.to_numpy(), iota=main.iota.to_numpy(), delta=main.delta.to_numpy(),
        cover=main.cover.to_numpy(), admissible=ok[:, column], R_adm=r_adm, reason=reason)
    quantiles = np.array([scan[R].xi.quantile([0.25, 0.5, 0.75]).to_numpy() if len(local) else [np.nan] * 3
                          for R in scan_radii], dtype=float)
    scaling_table = pd.DataFrame(quantiles, columns=["xi_q25", "xi_median", "xi_q75"]).assign(radius=scan_radii)
    if fit_range is None:
        fit_range = (min(radii), max(radii))
    fit_range = (float(fit_range[0]), float(fit_range[1]))
    A, H = power_law(scan_radii, quantiles[:, 1], fit_range) if len(scan_radii) > 1 else (np.nan, np.nan)
    if not np.isfinite(A) and len(scan_radii) > 1:   # fewer than two radii inside fit_range
        A, H = power_law(scan_radii, quantiles[:, 1])
    samples = None
    if retention and traces is not None and len(local):
        TA, TB = traces[0] - origin, traces[1] - origin
        samples = trace_samples(TA, TB, points, tree, fine_normals(points)[1], min(radii) / 20)
    rows = []
    for j, R in enumerate(radii):
        row = dict(radius=R, iota_median=float(scan[R].iota.median()) if len(local) else np.nan,
                   admissible=float(ok[:, j].mean()) if len(local) else np.nan)
        if (rectangle_radii or retention) and len(local):
            wide = scan[2 * R]
            row["admissible_rect"] = float(((wide.iota <= tau) & (wide.cover >= min_cover)).mean())
        if samples is not None:
            eta = predicted_retention(local, samples, R, scan[R].xi.to_numpy(), scan[2 * R].delta.to_numpy())
            row.update(eta_sphere=float(eta.eta_sphere.mean()), eta_rect=float(eta.eta_rect.mean()))
        rows.append(row)
    return RoughnessResult(table, radius, tuple(radii), float(tau), float(min_cover), spacing,
                           pd.DataFrame(rows), scaling_table, A, H, fit_range=fit_range,
                           length_unit=length_unit, scan=scan)


def assess_roughness(dataset, result=None, *, centers=None, radius=None, radii=None,
                     tau=DEFAULT_TAU, min_cover=DEFAULT_MIN_COVER, scaling=True, annotate=True,
                     retention=False, fit_range=None, slab_mode=DEFAULT_SLAB_MODE):
    """Roughness of the sampling windows of `result` (flag only; nothing is rejected).

    Defaults follow the analysis: the accepted window centres and the window
    radius (min(size_u, size_v) for rectangles). Pass `centers`/`radius` to
    assess other locations or scales, and `radii` to also test other radii
    (smallest admissible size R_adm = sphere radius = rectangle side, and the
    admissible fractions per radius for the sphere R and the rectangle W = 2R).
    The table column `reason` tells why a centre has no admissible radius.
    fit_range = (lo, hi) limits the power-law fit to the roughness regime
    (default: the range of `radii`); the other scanned radii are only shown.
    With annotate=True the columns xi, iota, cover and admissible are added to
    every family table of `result`. retention=True also computes the predicted
    retention eta of trace length per radius (needs the traces; slower).
    slab_mode must match the rectangle delta_mode (default 'max').
    """
    scan_xyz = dataset.scan[["X", "Y", "Z"]].to_numpy(float)
    if centers is None:
        if result is None:
            raise ValueError("Provide an analysis result or explicit centers.")
        centers = result.centers
    if radius is None:
        if result is None:
            raise ValueError("Provide radius when no analysis result is given.")
        radius = window_radius(result.window)
    traces = _trace_endpoints(dataset) if retention else None
    out = _assess(scan_xyz, centers, radius, radii=radii, tau=tau, min_cover=min_cover, scaling=scaling,
                  length_unit=dataset.length_unit, traces=traces, retention=retention,
                  fit_range=fit_range, slab_mode=slab_mode)
    if annotate and result is not None and len(out.table) == len(result.centers):
        _annotate(result, out)
        result.roughness = out
    return out


def _annotate(result, roughness):
    for frame in result.tables.values():
        if len(frame) == len(roughness.table):
            for column in ("xi", "iota", "cover", "admissible"):
                frame[column] = roughness.table[column].to_numpy()


def filter_centers(scan_xyz, centers, window, options=None, length_unit="m"):
    """Used by analyze(roughness_filter=True): returns (kept centres, RoughnessResult of all)."""
    options = dict(options or {})
    allowed = {"tau", "min_cover", "radius", "scaling"}
    if set(options) - allowed:
        raise ValueError(f"Unknown roughness options: {sorted(set(options) - allowed)}.")
    radius = options.pop("radius", None) or window_radius(window)
    options.setdefault("scaling", False)
    out = _assess(scan_xyz, centers, radius, length_unit=length_unit, rectangle_radii=False, **options)
    out.filtered = True
    return np.asarray(centers, float).reshape(-1, 3)[out.admissible], out


# -------------------------------------------------------------- figure ---
def plot_roughness(roughness, *, dfn=None, families=None, case_id=0):
    """Roughness figure: (a) map, [(b) total P32 if `dfn`], (c) scaling law, (d) diagnostics.

    Panel (d) shows the admissible fractions of the sphere (R) and of the rectangle
    (W = 2R) and, when computed (assess_roughness(..., retention=True)), the
    predicted retention eta. Returns (figure, axes dict).
    """
    import matplotlib.pyplot as plt
    from .style import style_context, SEQUENTIAL, SEQUENTIAL_POINTS, ACCENT, MUTED, FAMILY_COLORS, ordinal_colors
    table, unit = roughness.table, roughness.length_unit
    C = table[["X", "Y", "Z"]].to_numpy(float)
    center = C.mean(0)
    basis = np.linalg.eigh(np.cov((C - center).T))[1][:, ::-1][:, :2]
    uv = (C - center) @ basis
    p32 = None
    if dfn is not None:
        summary = dfn.summary[dfn.summary.case_id == case_id]
        names = list(summary.family.unique()) if families is None else list(families)
        group = summary[summary.family.isin(names)].groupby("window_id")
        p32 = group[["X", "Y", "Z"]].first().assign(P32=group.P32_mean.sum(), n=group.family.nunique())
        p32 = p32[p32.n == len(names)]
    rows = 3 if p32 is not None else 2
    with style_context():
        figure = plt.figure(figsize=(12, 4.7 * rows))
        grid = figure.add_gridspec(rows, 2, height_ratios=[0.75] * (rows - 1) + [1], hspace=0.38, wspace=0.3)
        axes = {"map": figure.add_subplot(grid[0, :])}
        if p32 is not None:
            axes["p32"] = figure.add_subplot(grid[1, :])
        axes["scaling"], axes["admissibility"] = figure.add_subplot(grid[-1, 0]), figure.add_subplot(grid[-1, 1])
        ax = axes["map"]
        radii = roughness.radii
        if len(radii) > 1:
            for R, color in zip(radii, ordinal_colors(len(radii))):
                m = np.isclose(table.R_adm, R)
                ax.scatter(*uv[m].T, s=8, color=color, linewidths=0, label=fr"$R_{{adm}}$ = {R:g} {unit}")
            none = table.R_adm.isna().to_numpy()
            title = fr"Smallest admissible size ($\iota\leq{roughness.tau:g}$): sphere radius $R$ = rectangle side $W$"
        else:
            ok = roughness.admissible
            ax.scatter(*uv[ok].T, s=8, color=SEQUENTIAL(0.7), linewidths=0, label="Admissible")
            none = ~ok
            title = fr"Admissible windows at $R$ = {roughness.radius:g} {unit} ($\iota\leq{roughness.tau:g}$)"
        rough = none & (table.reason.to_numpy() == "rough") if len(radii) > 1 else none
        ax.scatter(*uv[rough].T, s=14, marker="x", color=ACCENT, lw=0.8,
                   label=r"None: $\iota > \tau$" if len(radii) > 1 else "Not admissible")
        if len(radii) > 1 and (none & ~rough).any():
            ax.scatter(*uv[none & ~rough].T, s=8, color=MUTED, alpha=0.4, linewidths=0,
                       label="None: low cover or few points")
        ax.set(aspect="equal", xlabel=f"Along slope ({unit})", ylabel=f"Across slope ({unit})",
               title=f"(a) {title}")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), markerscale=2)
        letters = iter("bcd" if p32 is not None else "bc")
        if p32 is not None:
            ax = axes["p32"]
            uv32 = (p32[["X", "Y", "Z"]].to_numpy(float) - center) @ basis
            points = ax.scatter(*uv32.T, s=8, c=p32.P32, cmap=SEQUENTIAL_POINTS, linewidths=0)
            bar = figure.colorbar(points, cax=ax.inset_axes([1.02, 0, 0.015, 1]),
                                  label=rf"Mean $P_{{32}}$ ({unit}$^{{-1}}$)")
            bar.outline.set_visible(False)
            ax.set(aspect="equal", xlim=axes["map"].get_xlim(), ylim=axes["map"].get_ylim(),
                   xlabel=f"Along slope ({unit})", ylabel=f"Across slope ({unit})",
                   title=rf"({next(letters)}) Total $P_{{32}}$ ({' + '.join(map(str, names))})")
        ax = axes["scaling"]
        s = roughness.scaling
        if len(s) > 1:
            r = s.radius.to_numpy(float)
            ax.fill_between(r, s.xi_q25, s.xi_q75, color=SEQUENTIAL(0.25), lw=0, label="25–75 % of windows")
            ax.plot(r, s.xi_median, "o", color=FAMILY_COLORS[0], ms=5)
            if np.isfinite(roughness.A):
                ax.plot(r, roughness.A * r ** roughness.H, color=FAMILY_COLORS[0],
                        label=fr"$\xi={roughness.A:.3f}\,r^{{{roughness.H:.2f}}}$")
                if roughness.fit_range is not None:
                    ax.axvspan(*roughness.fit_range, color=MUTED, alpha=0.08, lw=0, label="Fit range")
            ax.plot(r, roughness.tau * r, "--", color=MUTED, lw=1.3, label=fr"$\xi={roughness.tau:g}\,r$")
            ax.set(xscale="log", yscale="log")
            ticks = [v for v in roughness.radii] + [r.min(), r.max()]
            ax.set_xticks(sorted(set(ticks)), [f"{v:g}" for v in sorted(set(ticks))])
            ax.minorticks_off()
            ax.legend(loc="upper left")
        else:
            ax.text(0.5, 0.5, "Scaling law not computed\n(scaling=False)", ha="center", va="center",
                    transform=ax.transAxes, color=MUTED)
        ax.set(xlabel=f"Radius $r$ ({unit})", ylabel=rf"Roughness $\xi_r$ ({unit})",
               title=f"({next(letters)}) Roughness scaling")
        ax = axes["admissibility"]
        b = roughness.by_radius
        ax.plot(b.radius, b.admissible, "-o", color=FAMILY_COLORS[0], ms=5, label=r"Admissible ($\iota$), sphere")
        if "admissible_rect" in b:
            ax.plot(b.radius, b.admissible_rect, "-s", color=FAMILY_COLORS[1], ms=5,
                    label=r"Admissible ($\iota$), rectangle")
        if "eta_sphere" in b:
            ax.plot(b.radius, b.eta_sphere, "--o", color=FAMILY_COLORS[0], mfc="white", ms=5,
                    label=r"Retention $\eta$, sphere")
            ax.plot(b.radius, b.eta_rect, "--s", color=FAMILY_COLORS[1], mfc="white", ms=5,
                    label=r"Retention $\eta$, rectangle")
        if len(b) > 1:
            ax.set_xscale("log")
        ax.set_xticks(b.radius, [f"{v:g}" for v in b.radius])
        ax.minorticks_off()
        ax.set(ylim=(0, 1.05), xlabel=f"Window radius $R$ ({unit})", ylabel="Fraction",
               title=f"({next(letters)}) Diagnostics vs window size")
        ax.legend(loc="lower right")
        for ax in (axes["scaling"], axes["admissibility"]):
            ax.set_box_aspect(1)
            ax.grid(True)
    return figure, axes

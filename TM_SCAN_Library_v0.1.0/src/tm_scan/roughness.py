"""Window-scale roughness control, tied to the sampling windows.

Criterion (per window of radius R): rho = xi_R / R <= tau and projected
coverage >= min_cover, with xi_R the SD of the point distances to the
best-fit plane (sigma(D) of Lague et al. 2013, D = 2R). Scaling law
xi_r = A r^H after Sadeghi & Aubertin (2025). By default windows are only
FLAGGED; rejection happens only with analyze(..., roughness_filter=True).
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DEFAULT_TAU, DEFAULT_MIN_COVER = 0.08, 0.80


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


def window_roughness(points, tree, centers, radius, spacing, min_points=20):
    """Per spherical window: xi, rho = xi/R, delta (P95 |d| to the plane through
    the centre) and cover (occupied fraction of the projected disc)."""
    out = np.full((len(centers), 4), np.nan)
    for i, index in enumerate(tree.query_ball_point(centers, radius)):
        if len(index) < min_points:
            continue
        selected = points[index]
        X = selected - selected.mean(0)
        w, V = np.linalg.eigh(X.T @ X / len(index))
        xi = np.sqrt(max(w[0], 0.0))
        delta = np.percentile(np.abs((selected - centers[i]) @ V[:, 0]), 95)
        cells = np.unique(np.floor(X @ V[:, 1:] / (2 * spacing)), axis=0)
        out[i] = xi, xi / radius, delta, min(1.0, len(cells) * (2 * spacing) ** 2 / (np.pi * radius ** 2))
    return pd.DataFrame(out, columns=["xi", "rho", "delta", "cover"])


def roughness_scan(points, centers, radii, spacing, per_radius=16):
    """Multi-scale scan; at radius R the cloud is used at resolution max(spacing, R/16)."""
    out = {}
    for radius in radii:
        resolution = max(spacing, radius / per_radius)
        thinned = points if resolution <= spacing else voxel_thin(points, resolution)
        out[float(radius)] = window_roughness(thinned, cKDTree(thinned), centers, radius, resolution)
    return out


def admissible_radius(scan, tau=DEFAULT_TAU, min_cover=DEFAULT_MIN_COVER):
    """Smallest radius from which the window is planar and covered at every larger radius.

    Returns (radius per centre, NaN if none; boolean table centre x radius)."""
    radii = np.asarray(sorted(scan))
    ok = np.column_stack([(scan[R].rho <= tau) & (scan[R].cover >= min_cover) for R in radii])
    tail = np.cumprod(ok[:, ::-1], axis=1)[:, ::-1].astype(bool)
    return np.where(tail.any(1), radii[tail.argmax(1)], np.nan), ok


def power_law(radii, xi):
    """xi_r = A r^H; returns (A, H)."""
    radii, xi = np.asarray(radii, float), np.asarray(xi, float)
    valid = np.isfinite(xi) & (xi > 0)
    if valid.sum() < 2:
        return np.nan, np.nan
    H, lnA = np.polyfit(np.log(radii[valid]), np.log(xi[valid]), 1)
    return float(np.exp(lnA)), float(H)


def r_min(A, H, tau):
    """Radius from which xi_R / R <= tau under xi = A r^H (requires H < 1)."""
    return (A / tau) ** (1.0 / (1.0 - H)) if np.isfinite(A) and H < 1 else np.nan


# ------------------------------------------------------- trace capture ---
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


def _clip_len(a, d, lo, hi):
    t0, t1 = np.zeros(len(a)), np.ones(len(a))
    with np.errstate(divide="ignore", invalid="ignore"):
        for j in range(a.shape[1]):
            ta, tb = (lo[j] - a[:, j]) / d[:, j], (hi[j] - a[:, j]) / d[:, j]
            parallel = np.abs(d[:, j]) < 1e-12
            inside = (a[:, j] >= lo[j]) & (a[:, j] <= hi[j])
            t0 = np.maximum(t0, np.where(parallel, np.where(inside, -np.inf, np.inf), np.minimum(ta, tb)))
            t1 = np.minimum(t1, np.where(parallel, np.inf, np.maximum(ta, tb)))
    return np.clip(t1 - t0, 0, None) * np.linalg.norm(d, axis=1)


def _sphere_chord(c, A, B, R):
    d, m = B - A, A - c
    a, b, cc = (d * d).sum(1), 2 * (m * d).sum(1), (m * m).sum(1) - R * R
    disc = b * b - 4 * a * cc
    s = np.sqrt(np.clip(disc, 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t0, t1 = np.clip((-b - s) / (2 * a), 0, 1), np.clip((-b + s) / (2 * a), 0, 1)
    return np.nansum(np.where(disc > 0, t1 - t0, 0) * np.sqrt(a))


def _rect_frame(c, points, tree, su, sv):
    Q = points[tree.query_ball_point(c, min(su, sv))]
    n = np.linalg.eigh(np.cov(Q.T))[1][:, 0]
    n = n if n[2] >= 0 else -n
    u = np.array([1.0, 0, 0]) - n[0] * n
    u /= np.linalg.norm(u)
    return np.column_stack([u, np.cross(n, u), n]), np.percentile(np.abs((Q - c) @ n), 95)


def trace_capture(points, tree, centers, A, B, samples, radius):
    """Summed counted length per rule over the given windows.

    chord = current rule (straight chord tested against the window);
    foot  = trace evaluated on the surface (sample counted if its foot is in
    the window). Sphere of radius R and square 2R x 2R. P21 only; mu untested.
    """
    total = dict(sphere_chord=0.0, sphere_foot=0.0, rect_chord=0.0, rect_foot=0.0)
    side = 2 * radius
    for c in centers:
        total["sphere_chord"] += _sphere_chord(c, A, B, radius)
        total["sphere_foot"] += samples["w"][samples["ftree"].query_ball_point(c, radius)].sum()
        frame, delta = _rect_frame(c, points, tree, side, side)
        half = np.array([side / 2, side / 2, delta])
        total["rect_chord"] += _clip_len((A - c) @ frame, (B - A) @ frame, -half, half).sum()
        index = np.asarray(samples["ftree"].query_ball_point(c, side), int)
        if index.size:
            uv = (points[samples["foot"][index]] - c) @ frame[:, :2]
            inside = (np.abs(uv[:, 0]) <= side / 2) & (np.abs(uv[:, 1]) <= side / 2)
            total["rect_foot"] += samples["w"][index][inside].sum()
    return total


# --------------------------------------------------------- high level ---
@dataclass
class RoughnessResult:
    """Roughness of sampling windows.

    table      : X, Y, Z, xi, rho, delta, cover, admissible (at `radius`), R_adm
    by_radius  : per radius -> median rho and admissible fraction (+ capture ratios)
    scaling    : radius -> xi quantiles (25, 50, 75 %) for the scaling law
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
                          "median_rho": float(self.table.rho.median()) if len(self.table) else np.nan,
                          "A": self.A, "H": self.H, "R_min": self.r_min, "filter_applied": self.filtered})

    def plot(self, **kwargs):
        return plot_roughness(self, **kwargs)


def window_radius(window):
    """Roughness radius of a sampling window: R for spheres, equal-area radius for rectangles."""
    if window.kind == "sphere":
        return float(window.radius)
    return float(np.sqrt(window.size_u * window.size_v / np.pi))


def _assess(scan_xyz, centers, radius, *, radii=None, tau=DEFAULT_TAU, min_cover=DEFAULT_MIN_COVER,
            scaling=True, length_unit="m", traces=None, capture=False):
    centers = np.asarray(centers, float).reshape(-1, 3)
    radius = float(radius)
    radii = sorted({float(r) for r in (radii if radii is not None else [radius])} | {radius})
    if min(radii) <= 0 or not 0 < tau or not 0 <= min_cover <= 1:
        raise ValueError("radii and tau must be positive; min_cover within [0, 1].")
    origin = scan_xyz.mean(0)
    working = min(radii) / 10                      # R >= 10 x spacing
    points = voxel_thin(scan_xyz - origin, working)
    tree = cKDTree(points)
    spacing = max(working, float(np.median(tree.query(points[::20], k=2)[0][:, 1])))
    local = centers - origin
    scan_radii = list(radii)
    if scaling:
        extra = np.round(np.geomspace(0.5 * min(radii), 2 * max(radii), 6), 3)
        scan_radii = sorted(set(radii) | set(float(r) for r in extra))
    scan = roughness_scan(points, local, scan_radii, spacing) if len(local) else {
        R: pd.DataFrame(columns=["xi", "rho", "delta", "cover"], dtype=float) for R in scan_radii}
    main = scan[radius]
    r_adm, ok = admissible_radius({R: scan[R] for R in radii}, tau, min_cover)
    column = radii.index(radius)
    table = pd.DataFrame(centers, columns=["X", "Y", "Z"]).assign(
        xi=main.xi.to_numpy(), rho=main.rho.to_numpy(), delta=main.delta.to_numpy(),
        cover=main.cover.to_numpy(), admissible=ok[:, column] if len(local) else np.empty(0, bool),
        R_adm=r_adm)
    quantiles = np.array([scan[R].xi.quantile([0.25, 0.5, 0.75]).to_numpy() if len(local) else [np.nan] * 3
                          for R in scan_radii], dtype=float)
    scaling_table = pd.DataFrame(quantiles, columns=["xi_q25", "xi_median", "xi_q75"]).assign(radius=scan_radii)
    A, H = power_law(scan_radii, quantiles[:, 1]) if len(scan_radii) > 1 else (np.nan, np.nan)
    rows = []
    samples = None
    if capture and traces is not None and len(local):
        TA, TB = traces[0] - origin, traces[1] - origin
        samples = trace_samples(TA, TB, points, tree, fine_normals(points)[1], spacing / 2)
    for j, R in enumerate(radii):
        row = dict(radius=R, rho_median=float(scan[R].rho.median()) if len(local) else np.nan,
                   admissible=float(ok[:, j].mean()) if len(local) else np.nan)
        if samples is not None:
            sums = trace_capture(points, tree, local[ok[:, j]], TA, TB, samples, R)
            row.update(sphere_ratio=sums["sphere_chord"] / sums["sphere_foot"] if sums["sphere_foot"] else np.nan,
                       rect_ratio=sums["rect_chord"] / sums["rect_foot"] if sums["rect_foot"] else np.nan)
        rows.append(row)
    return RoughnessResult(table, radius, tuple(radii), float(tau), float(min_cover), spacing,
                           pd.DataFrame(rows), scaling_table, A, H, length_unit=length_unit, scan=scan)


def assess_roughness(dataset, result=None, *, centers=None, radius=None, radii=None,
                     tau=DEFAULT_TAU, min_cover=DEFAULT_MIN_COVER, scaling=True, annotate=True,
                     trace_capture=False):
    """Roughness of the sampling windows of `result` (flag only; nothing is rejected).

    Defaults follow the analysis: the accepted window centres and the window
    radius (equal-area radius for rectangles). Pass `centers`/`radius` to
    assess other locations or scales, and `radii` to also test other radii
    (smallest admissible radius R_adm, admissible fraction per radius).
    With annotate=True the columns xi, rho, cover and admissible are added to
    every family table of `result`. trace_capture=True also compares the P21
    numerator of the current (chord) and surface (foot) rules per radius; it
    is slow and meant for runs where rejection has been decided.
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
    traces = None
    if trace_capture:
        traces = _trace_endpoints(dataset)
    out = _assess(scan_xyz, centers, radius, radii=radii, tau=tau, min_cover=min_cover, scaling=scaling,
                  length_unit=dataset.length_unit, traces=traces, capture=trace_capture)
    if annotate and result is not None and len(out.table) == len(result.centers):
        _annotate(result, out)
        result.roughness = out
    return out


def _annotate(result, roughness):
    for frame in result.tables.values():
        if len(frame) == len(roughness.table):
            for column in ("xi", "rho", "cover", "admissible"):
                frame[column] = roughness.table[column].to_numpy()


def filter_centers(scan_xyz, centers, window, options=None, length_unit="m"):
    """Used by analyze(roughness_filter=True): returns (kept centres, RoughnessResult of all)."""
    options = dict(options or {})
    allowed = {"tau", "min_cover", "radius", "scaling"}
    if set(options) - allowed:
        raise ValueError(f"Unknown roughness options: {sorted(set(options) - allowed)}.")
    radius = options.pop("radius", None) or window_radius(window)
    options.setdefault("scaling", False)
    out = _assess(scan_xyz, centers, radius, length_unit=length_unit, **options)
    out.filtered = True
    return np.asarray(centers, float).reshape(-1, 3)[out.admissible], out


# -------------------------------------------------------------- figure ---
def plot_roughness(roughness, *, dfn=None, families=None, case_id=0):
    """Roughness figure: (a) map, [(b) total P32 if `dfn`], (c) scaling law, (d) admissibility.

    Panel (d) shows the P21 chord/foot ratios only when they were computed
    (assess_roughness(..., trace_capture=True)). Returns (figure, axes dict).
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
            title = fr"Smallest admissible radius ($\rho\leq{roughness.tau:g}$)"
        else:
            ok = roughness.admissible
            ax.scatter(*uv[ok].T, s=8, color=SEQUENTIAL(0.7), linewidths=0, label="Admissible")
            none = ~ok
            title = fr"Admissible windows at $R$ = {roughness.radius:g} {unit} ($\rho\leq{roughness.tau:g}$)"
        ax.scatter(*uv[none].T, s=14, marker="x", color=ACCENT, lw=0.8,
                   label="None" if len(radii) > 1 else "Not admissible")
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
        ax.plot(b.radius, b.admissible, "-o", color=MUTED, ms=5, label="Admissible windows")
        if "sphere_ratio" in b:
            ax.plot(b.radius, b.sphere_ratio, "-o", color=FAMILY_COLORS[0], ms=5,
                    label=r"$P_{21}$ current / surface rule, sphere")
            ax.plot(b.radius, b.rect_ratio, "-s", color=FAMILY_COLORS[1], ms=5,
                    label=r"$P_{21}$ current / surface rule, rectangle")
        if len(b) > 1:
            ax.set_xscale("log")
        ax.set_xticks(b.radius, [f"{v:g}" for v in b.radius])
        ax.minorticks_off()
        ax.set(ylim=(0, 1.05), xlabel=f"Window radius $R$ ({unit})", ylabel="Fraction",
               title=f"({next(letters)}) Admissibility" + (" and trace capture" if "sphere_ratio" in b else ""))
        ax.legend(loc="lower right")
        for ax in (axes["scaling"], axes["admissibility"]):
            ax.set_box_aspect(1)
            ax.grid(True)
    return figure, axes

"""Equal-area (Schmidt) lower-hemisphere stereonets and orientation-model comparison.

Geometry: x = East, y = North, z = up. XYZ poles are plotted as given (n and -n
are the same plane; everything is folded to the lower hemisphere). Angle input
is the geological dip / dip direction (clockwise from North); the lower pole is
(-sin d sin a, -sin d cos a, -cos d).
Projection (primitive radius 1): X = x / sqrt(1 - z), Y = y / sqrt(1 - z).
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from .style import (style_context, INK, MUTED, FAINT, SEQUENTIAL, METHOD_COLORS, METHOD_LABELS,
                    family_colors)


# ----------------------------------------------------------------- geometry ---
def pole_vectors(dip, dip_direction):
    """Lower-hemisphere unit poles of planes given by geological dip / dip direction."""
    dip = np.atleast_1d(np.asarray(dip, float))
    direction = np.atleast_1d(np.asarray(dip_direction, float))
    if dip.shape != direction.shape:
        raise ValueError("dip and dip_direction must have the same length.")
    if not np.isfinite(dip).all() or not np.isfinite(direction).all() or ((dip < 0) | (dip > 90)).any():
        raise ValueError("dip must be finite and within [0, 90] degrees.")
    d, a = np.radians(dip), np.radians(np.mod(direction, 360.0))
    return np.column_stack([-np.sin(d) * np.sin(a), -np.sin(d) * np.cos(a), -np.cos(d)])


def poles_to_angles(poles):
    """Geological (dip, dip_direction) in degrees of planes given by XYZ poles."""
    v = _lower(poles)
    dip = np.degrees(np.arccos(np.clip(-v[:, 2], -1, 1)))
    direction = np.degrees(np.arctan2(-v[:, 0], -v[:, 1])) % 360.0
    return dip, direction


def _lower(vectors):
    v = np.atleast_2d(np.asarray(vectors, float))
    if v.shape[1] != 3:
        raise ValueError("Poles must be an (n, 3) XYZ array.")
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    if not np.isfinite(v).all() or (norm == 0).any():
        raise ValueError("Poles must be finite nonzero vectors.")
    v = v / norm
    return np.where(v[:, 2:3] > 0, -v, v)


def as_poles(planes, dip_direction=None):
    """Accept (dip, dip_direction) arrays, an (n, 2) angle array, an (n, 3) pole
    array, or a DataFrame (Nx/Ny/Nz, else dip/dip_direction or dip/dip_dir)."""
    if dip_direction is not None:
        return pole_vectors(planes, dip_direction)
    if isinstance(planes, pd.DataFrame):
        if {"Nx", "Ny", "Nz"} <= set(planes):
            return _lower(planes[["Nx", "Ny", "Nz"]].to_numpy(float))
        column = "dip_direction" if "dip_direction" in planes else "dip_dir"
        if not {"dip", column} <= set(planes):
            raise ValueError("Provide Nx/Ny/Nz or dip/dip_direction columns.")
        return pole_vectors(planes["dip"], planes[column])
    raw = np.asarray(planes, float)
    if raw.ndim == 2 and raw.shape[1] == 2:
        return pole_vectors(raw[:, 0], raw[:, 1])
    return _lower(raw)


def project(vectors):
    """XYZ vectors -> (X, Y) on the equal-area lower-hemisphere net (radius 1)."""
    v = _lower(vectors)
    f = 1.0 / np.sqrt(1.0 - v[:, 2])
    return v[:, 0] * f, v[:, 1] * f


def _unproject(x, y):
    rho = np.hypot(x, y)
    theta = 2.0 * np.arcsin(np.clip(rho / np.sqrt(2.0), 0.0, 1.0))
    az = np.arctan2(x, y)
    return np.column_stack([np.sin(theta) * np.sin(az), np.sin(theta) * np.cos(az), -np.cos(theta)])


def great_circle(dip, dip_direction, n=181):
    """Unit vectors along the lower-hemisphere great circle of one plane."""
    d, a = np.radians(float(dip)), np.radians(float(dip_direction) % 360.0)
    strike = np.array([-np.cos(a), np.sin(a), 0.0])
    plunge = np.array([np.cos(d) * np.sin(a), np.cos(d) * np.cos(a), -np.sin(d)])
    t = np.linspace(0.0, np.pi, n)[:, None]
    return np.cos(t) * strike + np.sin(t) * plunge


def pole_density(poles, *, method="kamb", k=3.0, grid_size=201, margin=1.03):
    """Axial pole density on the net.

    'kamb'     : Kamb (1959), counting area k^2/(N+k^2); sigma above uniform.
    'kamb_exp' : exponentially smoothed Kamb; sigma above uniform.
    'schmidt'  : 1 % counting circle; % of poles per 1 % area.
    Returns a dict with X, Y, Z (NaN outside the net), N and label.
    """
    poles = _lower(poles)
    count = len(poles)
    if count < 3:
        raise ValueError("At least three poles are required for contouring.")
    g = np.linspace(-margin, margin, grid_size)
    X, Y = np.meshgrid(g, g)
    inside = np.hypot(X, Y) <= margin
    grid = _unproject(X[inside], Y[inside])
    if method == "schmidt":
        area = 0.01
    elif method == "kamb":
        area = k ** 2 / (count + k ** 2)
    elif method == "kamb_exp":
        f = 2.0 * (1.0 + count / k ** 2)
    else:
        raise ValueError("method must be 'kamb', 'kamb_exp' or 'schmidt'.")
    total = np.empty(len(grid))
    for i in range(0, len(grid), 5000):
        c = np.abs(grid[i:i + 5000] @ poles.T)
        total[i:i + 5000] = (np.exp(f * (c - 1.0)).sum(1) if method == "kamb_exp"
                             else (c >= 1.0 - area).sum(1))
    if method == "schmidt":
        z, label = 100.0 * total / count, "% of poles per 1% area"
    elif method == "kamb":
        z = (total - count * area) / np.sqrt(count * area * (1.0 - area))
        label = rf"$\sigma$ above uniform (Kamb, k={k:g})"
    else:
        m1, m2 = (1.0 - np.exp(-f)) / f, (1.0 - np.exp(-2.0 * f)) / (2.0 * f)
        z = (total - count * m1) / np.sqrt(count * (m2 - m1 ** 2))
        label = rf"$\sigma$ above uniform (exponential Kamb, k={k:g})"
    Z = np.full(X.shape, np.nan)
    Z[inside] = z
    return dict(X=X, Y=Y, Z=Z, N=count, method=method, label=label)


def _multiple(value, step):
    r = value % step
    return np.isclose(r, 0.0) or np.isclose(r, step)


# ---------------------------------------------------------------- the net ---
class Stereonet:
    """Schmidt net drawn on ordinary Cartesian axes.

    The curved meridians and parallels are drawn by default (`minor` and
    `major` spacing in degrees). The straight N-S / E-W axes are never drawn
    (nor curves within `straight_tol` degrees of them). grid=False leaves only the primitive circle, ticks and N/E/S/W.
    """

    def __init__(self, ax=None, title=None, *, grid=True, minor=10, major=30, straight_tol=0.5,
                 azimuth_labels=False, figsize=(5.5, 5.8)):
        if ax is None:
            with style_context():
                _, ax = plt.subplots(figsize=figsize)
        self.ax = ax
        self.straight_tol = float(straight_tol)
        self._draw(grid, minor, major, azimuth_labels)
        if title:
            ax.set_title(title, fontsize=11, pad=12, color=INK)

    def _draw(self, grid, minor, major, azimuth_labels):
        ax = self.ax
        if grid:
            fine, coarse = [], []
            for dip in np.arange(minor, 90, minor):
                if 90 - dip < self.straight_tol:        # almost the straight N-S line
                    continue
                for direction in (90.0, 270.0):
                    xy = np.column_stack(project(great_circle(dip, direction)))
                    (coarse if _multiple(dip, major) else fine).append(xy)
            phi = np.linspace(np.pi, 2 * np.pi, 181)
            for gamma in np.arange(minor, 180, minor):
                if abs(gamma - 90.0) < self.straight_tol:   # the straight E-W line and its neighbours
                    continue
                g = np.radians(gamma)
                # z is clamped at 0: a rounding-level positive z at the ends would be folded to
                # the antipode and draw a spurious straight line across the net.
                pts = np.column_stack([np.sin(g) * np.cos(phi), np.full_like(phi, np.cos(g)),
                                       np.minimum(np.sin(g) * np.sin(phi), 0.0)])
                (coarse if _multiple(gamma, major) else fine).append(np.column_stack(project(pts)))
            ax.add_collection(LineCollection(fine, colors="#DDDCD7", linewidths=0.45, zorder=1))
            ax.add_collection(LineCollection(coarse, colors="#BDBCB6", linewidths=0.65, zorder=1))
        self.primitive = Circle((0, 0), 1.0, fill=False, ec=INK, lw=1.1, zorder=6)
        ax.add_patch(self.primitive)
        ticks = []
        for az in range(0, 360, 10):
            a = np.radians(az)
            c = 0.035 if az % 90 == 0 else (0.025 if az % 30 == 0 else 0.014)
            ticks.append([(np.sin(a), np.cos(a)), ((1 + c) * np.sin(a), (1 + c) * np.cos(a))])
        ax.add_collection(LineCollection(ticks, colors=INK, linewidths=0.6, zorder=6))
        for az in range(0, 360, 30 if azimuth_labels else 90):
            a = np.radians(az)
            if az % 90 == 0:
                ax.text(1.10 * np.sin(a), 1.10 * np.cos(a), "NESW"[az // 90], ha="center", va="center",
                        fontsize=10, color=INK)
            else:
                ax.text(1.10 * np.sin(a), 1.10 * np.cos(a), f"{az}°", ha="center", va="center",
                        fontsize=7, color=MUTED)
        ax.set_xlim(-1.18, 1.18)
        ax.set_ylim(-1.18, 1.18)
        ax.set_aspect("equal")
        ax.axis("off")

    def poles(self, planes, dip_direction=None, *, color=INK, s=14, marker="o", alpha=0.75,
              label=None, edge=True, **kwargs):
        """Plot plane poles (angles or XYZ poles; see `as_poles`)."""
        x, y = project(as_poles(planes, dip_direction))
        edge_style = dict(edgecolors=SURFACE_EDGE, linewidths=0.3) if edge and s >= 10 else dict(linewidths=0)
        return self.ax.scatter(x, y, s=s, color=color, marker=marker, alpha=alpha, zorder=5,
                               label=label, **edge_style, **kwargs)

    def planes(self, planes, dip_direction=None, *, color=INK, lw=0.8, alpha=0.8, label=None):
        """Plot planes as great circles."""
        dip, direction = poles_to_angles(as_poles(planes, dip_direction))
        segments = [np.column_stack(project(great_circle(d, a))) for d, a in zip(dip, direction)]
        collection = LineCollection(segments, colors=color, linewidths=lw, alpha=alpha, zorder=3)
        self.ax.add_collection(collection)
        if label:
            self.ax.plot([], [], color=color, lw=1.8, label=label)
        return collection

    def contours(self, planes, dip_direction=None, *, method="kamb", k=3.0, levels=None, start=2.0,
                 step=2.0, cmap=None, alpha=0.9, lines=True, colorbar=True, grid_size=201):
        """Filled pole-density contours, clipped to the primitive. Returns the density dict."""
        density = pole_density(as_poles(planes, dip_direction), method=method, k=k, grid_size=grid_size)
        Z = np.ma.masked_invalid(density["Z"])
        zmax = float(Z.max())
        if levels is None:
            levels = np.arange(start, zmax + step, step)
            if len(levels) < 2:
                levels = np.linspace(float(Z.min()), zmax, 6)
        filled = self.ax.contourf(density["X"], density["Y"], Z, levels=levels, cmap=cmap or SEQUENTIAL,
                                  alpha=alpha, zorder=2)
        sets = [filled]
        if lines:
            sets.append(self.ax.contour(density["X"], density["Y"], Z, levels=levels, colors=MUTED,
                                        linewidths=0.4, zorder=3))
        for contour_set in sets:
            try:
                contour_set.set_clip_path(self.primitive)
            except AttributeError:      # matplotlib < 3.8
                for item in contour_set.collections:
                    item.set_clip_path(self.primitive)
        if colorbar:
            bar = self.ax.figure.colorbar(filled, ax=self.ax, orientation="horizontal", fraction=0.045,
                                          pad=0.03, shrink=0.75)
            bar.set_label(density["label"], fontsize=9)
            bar.ax.tick_params(labelsize=8)
            bar.outline.set_visible(False)
        density["levels"] = np.asarray(levels)
        return density

    def legend(self, **kwargs):
        options = dict(loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False, fontsize=9)
        options.update(kwargs)
        return self.ax.legend(**options)


SURFACE_EDGE = "#FFFFFF"


def _family_dict(planes):
    if isinstance(planes, dict):
        if not planes:
            raise ValueError("planes is empty.")
        empty = [name for name, values in planes.items() if len(values) == 0]
        if empty:
            raise ValueError(f"No planes for families {empty}; check the family/joint-set correspondence "
                             "(see split_planes).")
        return {name: as_poles(values) for name, values in planes.items()}
    return {"Planes": as_poles(planes)}


def plot_stereonet(planes, *, kind="poles", grid=True, title=None, colors=None, **kwargs):
    """One net with every family. kind: 'poles', 'planes' or 'contours'.

    planes: {family: angles or poles} or one array/DataFrame. Contours pool all
    families and overlay the poles. Returns (figure, axis).
    """
    families = _family_dict(planes)
    colors = colors or family_colors(families)
    with style_context():
        figure, axis = plt.subplots(figsize=(5.8, 6.2))
        net = Stereonet(axis, title, grid=grid)
        if kind == "contours":
            net.contours(np.vstack(list(families.values())), **kwargs)
            for name, poles in families.items():
                net.poles(poles, color=INK, s=5, alpha=0.6, edge=False)
        elif kind in ("poles", "planes"):
            for name, poles in families.items():
                label = f"{name} (n={len(poles)})"
                if kind == "poles":
                    net.poles(poles, color=colors[name], label=label, **kwargs)
                else:
                    net.planes(poles, color=colors[name], label=label, **kwargs)
            if len(families) > 1:
                net.legend()
        else:
            raise ValueError("kind must be 'poles', 'planes' or 'contours'.")
    return figure, axis


def split_planes(planes, families, *, column="Joint_set", mapping=None, verbose=True):
    """Split a mapped-planes table into {family: planes} for the given trace families.

    mapping={family: label} sets the correspondence by hand. Otherwise each
    family is matched to the label with the same name (case-insensitive) or,
    failing that, the same trailing number (e.g. F1 <-> j1 <-> 1). Labels left
    unmatched (e.g. foliation) are ignored. The correspondence is printed.
    """
    import re
    if column not in planes:
        raise ValueError(f"Column {column!r} not found; available: {list(planes.columns)}.")
    labels = list(dict.fromkeys(planes[column].dropna()))
    number = lambda name: (re.findall(r"\d+", str(name)) or [None])[-1]
    families = list(families)
    if mapping is None:
        mapping = {}
        for family in families:
            exact = [label for label in labels if str(label).lower() == str(family).lower()]
            same_number = [label for label in labels if number(family) and number(label) == number(family)]
            match = exact or (same_number if len(same_number) == 1 else [])
            mapping[family] = match[0] if match else None
    missing = [family for family in families if mapping.get(family) is None]
    if missing:
        raise ValueError(f"No {column} label matched families {missing}. Labels in the file: {labels}. "
                         f"Pass mapping={{family: label}}, e.g. mapping={{{families[0]!r}: {labels[0]!r}}}.")
    out = {family: planes[planes[column].astype(str) == str(mapping[family])] for family in families}
    empty = [family for family, frame in out.items() if frame.empty]
    if empty:
        raise ValueError(f"No planes for families {empty} with mapping {mapping}.")
    if verbose:
        ignored = [label for label in labels if label not in mapping.values()]
        print("Family <- " + column + ": " + ", ".join(f"{f} <- {mapping[f]} ({len(out[f])} planes)" for f in families)
              + (f" | ignored: {ignored}" if ignored else ""))
    return out


# --------------------------------------------- orientation-model comparison ---
@dataclass
class OrientationComparison:
    """Fits of several orientation models to the same measured families."""
    poles: dict                      # {family: measured poles (n, 3), lower hemisphere}
    fits: dict                       # {method: {family: OrientationDistribution}}
    summary: pd.DataFrame            # one row per method and family
    errors: dict = field(default_factory=dict)   # {(method, family): message} for failed fits

    def plot(self, **kwargs):
        return plot_orientation_comparison(self, **kwargs)


def compare_orientation_models(planes, *, methods=("fisher", "kent", "bingham", "spherical_kde", "empirical")):
    """Fit each model to each family, for a side-by-side visual comparison.

    planes: {family: (dip, dip_direction) pairs, XYZ poles or DataFrame}, with
    geological angles; or a single array for one family. Nothing is exported.
    A model that cannot be fitted to a family is reported in `.errors` and
    left out of the figure instead of stopping the comparison.
    """
    from .orientation import fit_from_planes, _canonical_method
    families = _family_dict(planes)
    methods = [_canonical_method(method) for method in methods]
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("methods must be distinct orientation model names.")
    fits, rows, errors = {method: {} for method in methods}, [], {}
    for method in methods:
        for family, poles in families.items():
            try:
                fit = fit_from_planes(poles, method=method)
            except (ValueError, RuntimeError) as error:
                errors[(method, family)] = str(error)
                continue
            fits[method][family] = fit
            dip, direction = poles_to_angles([fit.mean_normal])
            row = dict(method=method, family=family, measured_planes=len(poles),
                       representative_dip=float(dip[0]), representative_dip_direction=float(direction[0]),
                       kappa=fit.kappa if method in ("fisher", "kent", "spherical_kde") else np.nan,
                       beta=fit.beta if method == "kent" else np.nan,
                       concentrations=fit.concentrations if method == "bingham" else None)
            rows.append(row)
    return OrientationComparison(families, fits, pd.DataFrame(rows), errors)


def plot_orientation_comparison(comparison, *, n_samples=1000, seed=0, grid=True, labels=None,
                                panel_size=3.0):
    """Rows = families; columns = measured poles, then samples of each fitted model.

    Model samples are diagnostic draws from the fitted law (not DFN realizations).
    Returns (figure, axes).
    """
    families = list(comparison.poles)
    methods = list(comparison.fits)
    labels = {**METHOD_LABELS, **(labels or {})}
    columns = ["observed"] + methods
    with style_context():
        figure, axes = plt.subplots(len(families), len(columns), squeeze=False, layout="constrained",
                                    figsize=(panel_size * len(columns), (panel_size + 0.3) * len(families)))
        for i, family in enumerate(families):
            for j, method in enumerate(columns):
                color = METHOD_COLORS.get(method, MUTED)
                if method == "observed":
                    poles = comparison.poles[family]
                    title = f"{family} · measured (n={len(poles)})"
                    net = Stereonet(axes[i, j], title, grid=grid)
                    net.poles(poles, color=color, s=7, alpha=0.7, edge=False)
                    continue
                net = Stereonet(axes[i, j], f"{family} · {labels.get(method, method)}", grid=grid)
                fit = comparison.fits[method].get(family)
                if fit is None:
                    axes[i, j].text(0, 0, "fit not\navailable", ha="center", va="center", color=FAINT, fontsize=10)
                    continue
                sample = fit.rvs(n_samples, random_state=seed + 1000 * i + j)
                net.poles(sample, color=color, s=3, alpha=0.55, edge=False)
    return figure, axes

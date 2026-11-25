"""Matplotlib figures returned to the caller; saving is always explicit."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from .models import METRICS
from ._plot_helpers import _plot_traces, _plot_boundary, _circle_segments_oriented


from .style import (style_context, SEQUENTIAL, SEQUENTIAL_POINTS, FAMILY_COLORS, INK, MUTED, FAINT,
                    family_colors)


def _label(metric, unit):
    if metric == "mu":
        return rf"$\mu$ ({unit})"
    if metric in ("P20", "P21"):
        power = -2 if metric == "P20" else -1
        return rf"$P_{{{metric[1:]}}}$ ({unit}$^{{{power}}}$)"
    raise ValueError("metric must be 'mu', 'P20' or 'P21'.")


def _metrics(metrics):
    metrics = list(metrics)
    if not metrics or any(metric not in METRICS for metric in metrics):
        raise ValueError("Choose metrics from ('mu', 'P20', 'P21').")
    return metrics


def plot_study(study, *, metrics=("mu", "P20", "P21"), band="std"):
    """Means by family, with optional +/- one spatial sample SD.

    Bands describe between-window variability; they are not uncertainty
    intervals for a population mean. Missing means remain gaps in the curve.
    """
    metrics = _metrics(metrics)
    if band not in (None, "std"):
        raise ValueError("band must be None or 'std'.")
    summary = study.summary
    with style_context():
        figure, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.2), squeeze=False)
        for axis, metric in zip(axes[0], metrics):
            for family, group in summary.groupby("family", sort=False):
                if study.factor == "size":
                    group = group.sort_values("case_id")
                    x = np.arange(len(group), dtype=float)
                    axis.set_xticks(x, [f"{row.size_u:g} × {row.size_v:g}" for row in group.itertuples()])
                    xlabel = f"Full side lengths ({study.runs[0].length_unit})"
                else:
                    group = group.sort_values(study.factor)
                    x = group[study.factor].to_numpy(float)
                    xlabel = f"Radius ({study.runs[0].length_unit})" if study.factor == "radius" else r"$k_{smooth}$"
                y = group[f"{metric}_mean"].to_numpy(float)
                line, = axis.plot(x, y, "o-", label=family)
                if band == "std":
                    sd = group[f"{metric}_std"].to_numpy(float)
                    axis.fill_between(x, y - sd, y + sd, color=line.get_color(), alpha=.14)
            cases = summary.drop_duplicates("case_id")
            empty = cases[cases.final_windows == 0] if "final_windows" in cases else cases.iloc[:0]
            if study.factor != "size":
                values = np.sort(cases[study.factor].to_numpy(float))
                axis.set_xticks(values, [f"{v:g}" for v in values])
                axis.set_xlim(values.min() - .5 * np.diff(values).min() if len(values) > 1 else values.min() - 1,
                              values.max() + .5 * np.diff(values).min() if len(values) > 1 else values.max() + 1)
                for value in empty[study.factor].to_numpy(float):
                    axis.axvline(value, color=FAINT, ls=":", lw=1)
                if len(empty):
                    axis.plot([], [], color=FAINT, ls=":", label="No accepted windows")
            axis.set(xlabel=xlabel, ylabel=_label(metric, study.runs[0].length_unit))
            axis.grid(alpha=.2)
            axis.legend(frameon=False)
        if band == "std":
            figure.suptitle("Mean ± 1 spatial SD across windows")
        figure.tight_layout()
    return figure, axes[0]


def plot_sampling(study):
    """Window count and runtime for each parametric case."""
    summary = study.summary.drop_duplicates("case_id")
    with style_context():
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        if study.factor == "size":
            x = np.arange(len(summary))
            labels = [f"{row.size_u:g} × {row.size_v:g}" for row in summary.itertuples()]
            for axis in axes:
                axis.set_xticks(x, labels)
            xlabel = f"Full side lengths ({study.runs[0].length_unit})"
        else:
            summary = summary.sort_values(study.factor)
            x = summary[study.factor]
            xlabel = "Radius" if study.factor == "radius" else r"$k_{smooth}$"
            if study.factor == "radius":
                xlabel += f" ({study.runs[0].length_unit})"
        axes[0].plot(x, summary.initial_centers, "o--", label="Candidate centers")
        axes[0].plot(x, summary.final_windows, "o-", label="Accepted windows")
        axes[0].set_ylabel("Count")
        axes[0].legend(frameon=False)
        axes[1].plot(x, summary.elapsed_seconds, "o-")
        axes[1].set_ylabel("Analysis time (s)")
        for axis in axes:
            axis.set_xlabel(xlabel)
            axis.grid(alpha=.2)
        figure.tight_layout()
    return figure, axes


def plot_histograms(result, *, metrics=("mu", "P20", "P21"), bins=30):
    """Histogram of finite per-window values, separated by family."""
    metrics = _metrics(metrics)
    with style_context():
        figure, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
        for axis, metric in zip(axes[0], metrics):
            for family, frame in result.tables.items():
                values = frame[METRICS[metric]].to_numpy(float)
                values = values[np.isfinite(values)]
                if len(values):
                    axis.hist(values, bins=bins, histtype="step", linewidth=1.6, label=family)
            axis.set(xlabel=_label(metric, result.length_unit), ylabel="Window count")
            if axis.get_legend_handles_labels()[0]:
                axis.legend(frameon=False)
            axis.grid(alpha=.2)
        figure.tight_layout()
    return figure, axes[0]


def _limits(axis, points):
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = (lower + upper) / 2
    half = max(float((upper - lower).max()), 1e-6) * .55
    axis.set(xlim=(center[0] - half, center[0] + half),
             ylim=(center[1] - half, center[1] + half),
             zlim=(center[2] - half, center[2] + half))
    axis.set_box_aspect((1, 1, 1))


def _spread_indices(points, count, seed=0):
    """Greedy farthest-point selection: `count` points spread over the set.

    Starts at the point farthest from the centroid and adds, one at a time,
    the point farthest from those already chosen. This maximizes the minimum
    spacing, so the drawn windows overlap as little as possible.
    """
    points = np.asarray(points, float)
    if count >= len(points):
        return np.arange(len(points))
    first = int(np.argmax(np.linalg.norm(points - points.mean(axis=0), axis=1)))
    chosen = [first]
    distance = np.linalg.norm(points - points[first], axis=1)
    for _ in range(count - 1):
        nxt = int(np.argmax(distance))
        chosen.append(nxt)
        distance = np.minimum(distance, np.linalg.norm(points - points[nxt], axis=1))
    return np.sort(np.array(chosen))


def plot_windows(dataset, result, *, window_percent=None, sample_fraction=.03, max_windows=200,
                 selection="spread", seed=0, elev=None, azim=None):
    """Show traces, boundary and accepted windows without repeating analysis.

    window_percent (0-100) draws that percentage of the accepted windows;
    otherwise at most `max_windows` are drawn. selection='spread' (default)
    picks windows evenly spread along the face, with minimal overlap
    (farthest-point selection); selection='random' picks them at random. Display only: the analysis is unchanged. elev/azim
    set the camera (default: looking at the best-fit plane of the cloud).
    Spherical selections are represented by reference-plane circles. These
    circles are visual aids; the actual selector is a 3D sphere.
    """
    if window_percent is not None:
        if not np.isfinite(window_percent) or not 0 < window_percent <= 100:
            raise ValueError("window_percent must be in (0, 100].")
        max_windows = max(1, int(round(len(result.centers) * window_percent / 100)))
    if not np.isfinite(sample_fraction) or not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in (0, 1].")
    if isinstance(max_windows, bool) or int(max_windows) != max_windows or max_windows <= 0:
        raise ValueError("max_windows must be a positive integer.")
    if dataset.name != result.dataset_name:
        raise ValueError("Dataset and analysis result names differ.")
    sample = dataset.scan.sample(n=max(1, round(len(dataset.scan) * sample_fraction)), random_state=seed)
    count = min(int(max_windows), len(result.centers))
    if selection == "spread":
        indices = _spread_indices(result.centers, count, seed)
    elif selection == "random":
        indices = np.sort(np.random.default_rng(seed).choice(len(result.centers), count, replace=False))
    else:
        raise ValueError("selection must be 'spread' or 'random'.")
    with style_context():
        figure = plt.figure(figsize=(10, 7))
        axis = figure.add_subplot(111, projection="3d")
        axis.scatter(*sample[["X", "Y", "Z"]].to_numpy().T, s=1, color=FAINT, alpha=.25, linewidths=0)
        for family, color in family_colors(dataset.traces).items():
            _plot_traces(axis, {family: dataset.traces[family]}, lw=.8, color=color, label=family)
        _plot_boundary(axis, dataset.boundary, color=INK, lw=1.0)
        segments = []
        if result.window.kind == "rectangle":
            for index in indices:
                quad = result.quads[index]
                segments.extend([[quad[j], quad[(j + 1) % 4]] for j in range(4)])
        else:
            reference = next(iter(result.tables.values()))
            normals = reference[["Nx", "Ny", "Nz"]].to_numpy(float)
            for index in indices:
                circle = _circle_segments_oriented(result.centers[index], normals[index].copy(), result.window.radius)
                if circle is not None:
                    segments.extend(circle)
        if segments:
            label = "Rectangles" if result.window.kind == "rectangle" else "Reference circles (spherical selection)"
            axis.add_collection3d(Line3DCollection(segments, colors=MUTED, linewidths=.9, alpha=.8, label=label))
        if count:
            axis.scatter(*result.centers[indices].T, s=8, color=FAMILY_COLORS[1])
        axis.set(xlabel=f"X ({result.length_unit})", ylabel=f"Y ({result.length_unit})",
                 zlabel=f"Z ({result.length_unit})",
                 title=f"{dataset.name}: {len(result.centers):,} accepted windows; {count:,} displayed "
                       f"({100 * count / max(len(result.centers), 1):.0f} %)")
        default_elev, default_azim = _face_view(sample[["X", "Y", "Z"]].to_numpy(float))
        axis.view_init(elev=default_elev if elev is None else elev, azim=default_azim if azim is None else azim)
        _true_aspect(axis, dataset.scan[["X", "Y", "Z"]].to_numpy(float))
        axis.legend(loc="upper left", fontsize=8)
        figure.tight_layout()
    return figure, axis


def plot_metric_map(result, family, *, metric="P21", elev=28, azim=30):
    """Color accepted-window centers by a finite metric, without interpolation."""
    _metrics([metric])
    frame = result.tables[family]
    valid = np.isfinite(frame[METRICS[metric]].to_numpy(float))
    with style_context():
        figure = plt.figure(figsize=(8, 6))
        axis = figure.add_subplot(111, projection="3d")
        if valid.any():
            points = frame.loc[valid, ["X", "Y", "Z"]].to_numpy(float)
            image = axis.scatter(*points.T, c=frame.loc[valid, METRICS[metric]], cmap=SEQUENTIAL_POINTS, s=20)
            figure.colorbar(image, ax=axis, pad=.12, shrink=.7, label=_label(metric, result.length_unit))
            _limits(axis, points)
        else:
            axis.text2D(.25, .5, "No finite values for this metric", transform=axis.transAxes)
        axis.set(xlabel=f"X ({result.length_unit})", ylabel=f"Y ({result.length_unit})",
                 zlabel=f"Z ({result.length_unit})", title=f"{result.dataset_name} — {family}")
        axis.view_init(elev=elev, azim=azim)
        figure.tight_layout()
    return figure, axis


def save_figure(figure, filename, *, dpi=500):
    """Save an existing figure; do not close or show it implicitly."""
    path = Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def plot_joint_density(density, *, p21_fixed=None, families=None, labels=None,
                        grid_size=160, xlim=None, ylim=None, cmap=None):
    """One row per family: joint KDE left, normalized horizontal slice right.

    Axes and density scales are shared across families. There are exactly five
    colorbar ticks. Limits affect display only, never PDF normalization. Default
    X limits include at least 99.99% of each conditional's physical support.
    Returns (figure, axes), without saving/showing or refitting the KDE.
    """
    from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
    from .density import DensityFit
    if not isinstance(density, DensityFit):
        raise TypeError("density must be returned by fit_joint_density.")
    if not isinstance(grid_size, int) or grid_size < 20:
        raise ValueError("grid_size must be an integer >= 20.")
    names = list(density.models) if families is None else list(families)
    if not names or len(names) != len(set(names)) or not set(names) <= set(density.models):
        raise ValueError("families must select distinct fitted families.")
    # Restrict common conditioning support to the families actually displayed.
    selected = DensityFit({name: density.models[name] for name in names},
                          density.samples.loc[density.samples.family.isin(names)], density.metadata)
    conditionals = selected.conditionals(p21_fixed)
    fixed = next(iter(conditionals.values())).p21_fixed
    samples = selected.samples
    if xlim is None:
        upper = max(1.05*float(samples.P32.max()),
                    max(model.ppf(1.-1e-4) for model in conditionals.values()))
        xlim = (0., upper)
    if ylim is None:
        ylim = (0., 1.05*float(samples.P21.max()))
    for limits in (xlim, ylim):
        if len(limits) != 2 or not np.isfinite(limits).all() or limits[0] >= limits[1]:
            raise ValueError("Axis limits must be increasing finite pairs.")
    xx, yy = np.meshgrid(np.linspace(*xlim, grid_size), np.linspace(*ylim, grid_size))
    joint = {name: density.models[name].pdf(xx, yy) for name in names}
    zmax = max(float(values.max()) for values in joint.values())
    xgrid = np.linspace(*xlim, 800)
    pdf = {name: model.pdf(xgrid) for name, model in conditionals.items()}
    # Leave room for family labels above the highest conditional peak.
    ymax = 1.25*max(float(values.max()) for values in pdf.values())
    if zmax <= 0 or ymax <= 0:
        raise ValueError("No representable density inside the selected display limits.")
    unit = density.metadata.get("length_unit")
    suffix = rf" ({unit}$^{{-1}}$)" if unit else ""
    xname = r"Mean $P_{32}$" if density.metadata["aggregation"] == "mean" else r"$P_{32}$"
    labels = labels or {}
    with style_context():
        figure, axes = plt.subplots(len(names), 2, figsize=(12, 4*len(names)), squeeze=False)
        figure.subplots_adjust(hspace=.18, wspace=.30, bottom=.19)
        for i, name in enumerate(names):
            axis = axes[i, 0]
            contour = axis.contourf(xx, yy, joint[name], levels=np.linspace(0., zmax, 25), cmap=cmap or SEQUENTIAL)
            axis.axhline(fixed, color=INK, ls=":", lw=1.4)
            axis.text(.04, .94, labels.get(name, name), transform=axis.transAxes,
                      color=INK, fontsize=20, fontweight="bold", va="top")
            axis.set(xlim=xlim, ylim=ylim, xlabel=xname+suffix if i == len(names)-1 else "",
                     ylabel=r"$P_{21}$"+suffix)
            axis = axes[i, 1]
            axis.plot(xgrid, pdf[name], color=INK, lw=1.8)
            axis.fill_between(xgrid, pdf[name], color=".85")
            axis.text(.04, .94, labels.get(name, name), transform=axis.transAxes,
                      fontsize=20, fontweight="bold", va="top")
            axis.set(xlim=xlim, ylim=(0., ymax), xlabel=xname+suffix if i == len(names)-1 else "",
                     ylabel=rf"PDF at $P_{{21}}$ = {fixed:.3g}")
            for axis in axes[i]:
                axis.xaxis.set_minor_locator(AutoMinorLocator())
                axis.yaxis.set_minor_locator(AutoMinorLocator())
                axis.grid(True, which="both", ls="--", lw=.5, alpha=.35)
        position = axes[-1, 0].get_position()
        cax = figure.add_axes([position.x0, .07, position.width, .018])
        bar = figure.colorbar(contour, cax=cax, orientation="horizontal", ticks=np.linspace(0., zmax, 5))
        bar.set_label("Probability density", fontsize=16)
        bar.ax.tick_params(labelsize=12)
        bar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2g"))
    figure.tm_scan_metadata = {"P21_fixed": fixed, "aggregation": density.metadata["aggregation"],
                              "support": density.metadata["support"],
                              "visible_conditional_mass": {name: float(model.cdf(xlim[1])-model.cdf(xlim[0]))
                                                            for name, model in conditionals.items()}}
    return figure, axes


def _clip_polygon_to_box(polygon, half):
    """Sutherland-Hodgman clipping of a planar 3D polygon against the cube |x|,|y|,|z| <= half."""
    for axis in range(3):
        for sign in (1.0, -1.0):
            if len(polygon) == 0:
                return polygon
            inside = sign * polygon[:, axis] <= half
            if inside.all():
                continue
            following = np.roll(polygon, -1, axis=0)
            following_inside = np.roll(inside, -1)
            out = []
            for a, b, a_in, b_in in zip(polygon, following, inside, following_inside):
                if a_in:
                    out.append(a)
                if a_in != b_in:
                    t = (sign * half - a[axis]) / (b[axis] - a[axis])
                    out.append(a + t * (b - a))
            polygon = np.asarray(out).reshape(-1, 3)
    return polygon


def plot_dfn(simulation, window_id, *, families=None, realization=0, case_id=0, max_fractures=800,
             seed=0, elev=22, azim=35):
    """3D view of the fracture discs of one DFN realization, one colour per family.

    All families simulated for the window are drawn together (or those in
    `families`), clipped to the generation cube. Each family is regenerated
    from its stored seed, identical to the simulated realization.
    Coordinates are window-centred. Above `max_fractures` in total, discs are
    randomly subsampled in proportion within each family (the counts shown
    are the full numbers of fractures). Returns (figure, axis).
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    table = simulation.realizations
    here = table[(table.window_id == window_id) & (table.realization == realization) & (table.case_id == case_id)]
    names = list(dict.fromkeys(here.family)) if families is None else list(families)
    if not names:
        raise ValueError("No stored realization for this window_id/realization/case_id.")
    colors = family_colors(simulation.families)
    discs = {family: simulation.fractures(family, window_id, realization=realization, case_id=case_id)
             for family in names}
    total = sum(len(frame) for frame in discs.values())
    keep = min(1.0, max_fractures / total) if total else 1.0
    rng = np.random.default_rng(seed)
    angle = np.linspace(0, 2 * np.pi, 41)[:-1]
    half = float(simulation.config.half_size)
    polygons, faces = [], []
    for family, frame in discs.items():
        if not len(frame):
            continue
        rows = np.arange(len(frame))
        if keep < 1:
            rows = np.sort(rng.choice(rows, max(1, int(round(keep * len(rows)))), replace=False))
        centers = frame[["X", "Y", "Z"]].to_numpy(float)[rows]
        normals = frame[["Nx", "Ny", "Nz"]].to_numpy(float)[rows]
        radii = frame.diameter.to_numpy(float)[rows] / 2
        for c, n, r in zip(centers, normals, radii):
            u = np.cross(n, [0., 0., 1.] if abs(n[2]) < 0.9 else [1., 0., 0.])
            u /= np.linalg.norm(u)
            v = np.cross(n, u)
            polygon = _clip_polygon_to_box(c + r * (np.cos(angle)[:, None] * u + np.sin(angle)[:, None] * v), half)
            if len(polygon) >= 3:
                polygons.append(polygon)
                faces.append(colors[family])
    unit = here.length_unit.iloc[0] if len(here) else ""
    with style_context():
        figure = plt.figure(figsize=(8.5, 8.5))
        axis = figure.add_subplot(111, projection="3d")
        axis.set_axis_off()
        if polygons:
            axis.add_collection3d(Poly3DCollection(polygons, facecolors=faces, edgecolors=faces,
                                                   linewidths=.25, alpha=.5))
        axis.set(xlim=(-half, half), ylim=(-half, half), zlim=(-half, half))
        axis.set_box_aspect((1, 1, 1), zoom=1.2)
        axis.view_init(elev=elev, azim=azim)
        subsample = "" if keep >= 1 else f" ({len(polygons):,} drawn)"
        axis.set_title(f"Window {window_id} · realization {realization} · {total:,} fractures{subsample}\n"
                       f"cube edge {2 * half:g} {unit}", fontsize=12)
        for i, family in enumerate(names):
            axis.text2D(0.02, 0.96 - 0.045 * i, f"{family}: {len(discs[family]):,} fractures", color=colors[family],
                        transform=axis.transAxes, fontsize=12, fontweight="bold")
        figure.tight_layout()
    return figure, axis


def _plane_frame(points):
    """Centre and two in-plane axes (u, v) of the best-fit plane of `points`.

    The SVD gives the axes with arbitrary signs, which could flip the map.
    Here v is the upward direction within the plane (projection of +Z), so
    "Across face" grows upwards, and u = v x n completes a right-handed frame
    seen from the upper side of the plane (same side as the 3D view), so the
    map is never mirrored or upside down.
    """
    center = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points - center, full_matrices=False)
    normal = axes[2] if axes[2][2] >= 0 else -axes[2]
    up = np.array([0.0, 0.0, 1.0]) - normal[2] * normal
    if np.linalg.norm(up) < 1e-6:                     # near-horizontal plane: keep the SVD axis
        up = axes[1]
    v = up / np.linalg.norm(up)
    u = np.cross(v, normal)
    return center, np.vstack([u / np.linalg.norm(u), v])


def _face_view(points):
    """(elev, azim) looking at the best-fit plane of `points` from its upper side."""
    normal = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)[2][2]
    normal = normal if normal[2] >= 0 else -normal
    # slightly oblique (+20 deg elevation, -35 deg azimuth) so that the relief reads as 3D
    elev = float(np.clip(np.degrees(np.arcsin(np.clip(normal[2], -1, 1))) + 20, 15, 60))
    azim = float(np.degrees(np.arctan2(normal[1], normal[0]))) - 35
    return elev, azim


def _true_aspect(axis, points):
    """Data limits with true proportions (1 m = 1 m on every axis)."""
    lower, upper = points.min(axis=0), points.max(axis=0)
    span = np.maximum(upper - lower, 1e-6 * max(float((upper - lower).max()), 1e-6))
    axis.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]))
    axis.set_box_aspect(tuple(span), zoom=1.1)
    axis.tick_params(labelsize=8, colors=MUTED, pad=0)


def plot_dataset(dataset, *, view="3d", max_points=80_000, seed=0, elev=None, azim=None):
    """Point cloud, traces (one colour per family) and boundary.

    view='3d' (default) shows the real geometry with true proportions,
    looking at the best-fit plane of the cloud (change with elev/azim);
    view='plane' projects everything on that plane. Returns (figure, axis).
    """
    from ._traces import endpoints
    from matplotlib.collections import LineCollection
    if view not in ("3d", "plane"):
        raise ValueError("view must be '3d' or 'plane'.")
    xyz = dataset.scan[["X", "Y", "Z"]].to_numpy(float)
    pick = np.random.default_rng(seed).choice(len(xyz), min(len(xyz), max_points), replace=False)
    colors = family_colors(dataset.traces)
    unit = dataset.length_unit
    lines = []
    for geometry in dataset.boundary.geometry:
        for line in (geometry.geoms if geometry.geom_type == "MultiLineString" else [geometry]):
            lines.append(np.asarray(line.coords, float)[:, :3])
    with style_context():
        if view == "3d":
            figure = plt.figure(figsize=(12, 7.5))
            axis = figure.add_subplot(111, projection="3d")
            axis.scatter(*xyz[pick].T, s=.3, color=FAINT, alpha=.35, linewidths=0, rasterized=True)
            for family, frame in dataset.traces.items():
                A, B = endpoints(frame)
                axis.add_collection3d(Line3DCollection(np.stack([A, B], axis=1), colors=colors[family],
                                                       linewidths=1.2))
                axis.plot([], [], color=colors[family], lw=2, label=f"{family} (n={len(A)})")
            for i, b in enumerate(lines):
                axis.plot(*b.T, color=INK, lw=1.1, ls="--", label="Boundary" if i == 0 else None)
            _true_aspect(axis, xyz)
            default_elev, default_azim = _face_view(xyz[pick])
            axis.view_init(elev=default_elev if elev is None else elev, azim=default_azim if azim is None else azim)
            axis.set(xlabel=f"X ({unit})", ylabel=f"Y ({unit})", zlabel=f"Z ({unit})")
        else:
            center, axes2 = _plane_frame(xyz)
            project = lambda q: (np.asarray(q, float) - center) @ axes2.T
            figure, axis = plt.subplots(figsize=(13, 5.5))
            uv = project(xyz[pick])
            axis.scatter(uv[:, 0], uv[:, 1], s=.2, color=FAINT, alpha=.35, linewidths=0, rasterized=True)
            for family, frame in dataset.traces.items():
                A, B = endpoints(frame)
                axis.add_collection(LineCollection(np.stack([project(A), project(B)], axis=1),
                                                   colors=colors[family], linewidths=1.1))
                axis.plot([], [], color=colors[family], lw=2, label=f"{family} (n={len(A)})")
            for i, b in enumerate(lines):
                b = project(b)
                axis.plot(b[:, 0], b[:, 1], color=INK, lw=1.0, ls="--", label="Boundary" if i == 0 else None)
            axis.set(aspect="equal", xlabel=f"Along face ({unit})", ylabel=f"Across face ({unit})")
        axis.set_title(f"{dataset.name}: point cloud, traces and boundary")
        axis.legend(loc="center left", bbox_to_anchor=(1.02, .5))
        figure.tight_layout()
    return figure, axis


def plot_trace_lengths(dataset, *, bins=30):
    """Histogram of measured trace lengths per family with the best KS-fitted PDF.

    Returns (figure, axes).
    """
    from scipy import stats
    from .statistics import build_segment_lengths, ks_gof_by_family
    _, lengths = build_segment_lengths(dataset.traces)
    _, best = ks_gof_by_family(lengths)
    best = best.set_index("family")
    distributions = {"lognormal": stats.lognorm, "exponential": stats.expon,
                     "weibull": stats.weibull_min, "gamma": stats.gamma}
    colors = family_colors(lengths)
    unit = dataset.length_unit
    with style_context():
        figure, axes = plt.subplots(1, len(lengths), figsize=(5.5 * len(lengths), 4), squeeze=False)
        for axis, (family, values) in zip(axes[0], lengths.items()):
            axis.hist(values, bins=bins, density=True, color=colors[family], alpha=.35,
                      edgecolor=colors[family], linewidth=.5)
            name, params = best.loc[family, "dist"], best.loc[family, "params"]
            x = np.linspace(0, values.max(), 300)
            axis.plot(x, distributions[name].pdf(x, *params), color=INK, lw=1.6,
                      label=f"{name} (KS p = {best.loc[family, 'pvalue']:.2f})")
            cov = values.std(ddof=1) / values.mean()
            axis.set(xlabel=f"Measured trace length ({unit})", ylabel=f"Density ({unit}$^{{-1}}$)",
                     title=f"{family}: n = {len(values)}, mean = {values.mean():.2f} {unit}, CoV = {cov:.2f}")
            axis.legend()
            axis.grid(alpha=.6)
        figure.tight_layout()
    return figure, axes[0]


def _family_panels(simulation, families, total):
    names = list(simulation.families) if families is None else list(families)
    return names + (["Total"] if total and len(names) > 1 else [])


def plot_p32_map(simulation, *, families=None, total=True, case_id=0, view="3d", size=None,
                 elev=None, azim=None):
    """Mean P32 (over realizations) at each window centre.

    One panel per family and, with total=True, one for their sum, each with
    its own colour scale. view='3d' (default) shows the real geometry with
    true proportions; elev/azim set the camera (default: looking at the
    best-fit plane of the windows). view='plane' projects on that plane.
    Returns (figure, axes).
    """
    if view not in ("3d", "plane"):
        raise ValueError("view must be '3d' or 'plane'.")
    summary = simulation.summary[simulation.summary.case_id == case_id]
    panels = _family_panels(simulation, families, total)
    names = [p for p in panels if p != "Total"]
    wide = summary.pivot(index="window_id", columns="family", values="P32_mean")[names]
    xyz = summary.groupby("window_id")[["X", "Y", "Z"]].first().loc[wide.index].to_numpy(float)
    if "Total" in panels:
        wide["Total"] = wide.sum(axis=1, min_count=len(names))
    unit = summary.length_unit.iloc[0]
    with style_context():
        if view == "3d":
            size = size or 14
            default_elev, default_azim = _face_view(xyz)
            elev = default_elev if elev is None else elev
            azim = default_azim if azim is None else azim
            figure = plt.figure(figsize=(6.2 * len(panels), 6), layout="constrained")
            axes = []
            for i, panel in enumerate(panels):
                axis = figure.add_subplot(1, len(panels), i + 1, projection="3d")
                image = axis.scatter(*xyz.T, c=wide[panel].to_numpy(float), s=size, cmap=SEQUENTIAL_POINTS,
                                     linewidths=0, depthshade=False)
                _true_aspect(axis, xyz)
                axis.view_init(elev=elev, azim=azim)
                axis.set(xlabel=f"X ({unit})", ylabel=f"Y ({unit})", zlabel=f"Z ({unit})")
                title = " + ".join(names) if panel == "Total" else panel
                axis.set_title(rf"$P_{{32}}$ — {title}", fontsize=13)
                bar = figure.colorbar(image, ax=axis, orientation="horizontal", shrink=.6, pad=.02,
                                      label=rf"Mean $P_{{32}}$ ({unit}$^{{-1}}$)")
                bar.outline.set_visible(False)
                axes.append(axis)
            return figure, np.array(axes)
        size = size or 10
        center, axes2 = _plane_frame(xyz)
        uv = (xyz - center) @ axes2.T
        figure, axes = plt.subplots(len(panels), 1, figsize=(12, 3.6 * len(panels)), squeeze=False)
        for axis, panel in zip(axes[:, 0], panels):
            image = axis.scatter(uv[:, 0], uv[:, 1], c=wide[panel].to_numpy(float), s=size,
                                 cmap=SEQUENTIAL_POINTS, linewidths=0)
            bar = figure.colorbar(image, ax=axis, pad=.01, label=rf"Mean $P_{{32}}$ ({unit}$^{{-1}}$)")
            bar.outline.set_visible(False)
            title = " + ".join(names) if panel == "Total" else panel
            axis.set(aspect="equal", title=rf"$P_{{32}}$ — {title}", ylabel=f"Across face ({unit})")
        axes[-1, 0].set_xlabel(f"Along face ({unit})")
        figure.tight_layout()
    return figure, axes[:, 0]


def plot_p32_uncertainty(simulation, *, families=None, case_id=0, size=14, elev=None, azim=None):
    """How much P32 varies between DFN realizations of the same window.

    Top: 3D map (true proportions) of the coefficient of variation of P32
    across realizations at each window centre, one panel per family;
    elev/azim set the camera (default: looking at the best-fit plane of the
    windows). Bottom left: distribution of that CoV. Bottom right:
    convergence — median relative difference between the running mean after
    n realizations and the final mean. Windows with no observed trace of the
    family (P32 = 0 in every realization) are shown as grey rings on the maps
    and excluded from the statistics.
    Returns (figure, axes).
    """
    table = simulation.realizations
    table = table[table.case_id == case_id]
    # Windows with no observed trace of the family (P32 target = 0) generate no
    # fractures: P32 = 0 in every realization and the CoV (0/0) is undefined.
    # They are drawn in grey on the maps and left out of the statistics.
    empty_all = table[table.zero_target]
    table = table[~table.zero_target]
    names = list(simulation.families) if families is None else list(families)
    colors = family_colors(simulation.families)
    unit = table.length_unit.iloc[0]
    all_xyz = simulation.realizations[simulation.realizations.case_id == case_id]         .groupby("window_id")[["X", "Y", "Z"]].first().to_numpy(float)
    default_elev, default_azim = _face_view(all_xyz)
    elev = default_elev if elev is None else elev
    azim = default_azim if azim is None else azim
    with style_context():
        figure = plt.figure(figsize=(max(12, 6.2 * len(names)), 10.5), layout="constrained")
        grid = figure.add_gridspec(2, 2 * len(names), height_ratios=[1.35, 1])
        maps = []
        stats_axes = [figure.add_subplot(grid[1, :len(names)]), figure.add_subplot(grid[1, len(names):])]
        for i, family in enumerate(names):
            group = table[table.family == family]
            per_window = group.groupby("window_id").P32.agg(["mean", "std"])
            cov = per_window["std"] / per_window["mean"]
            xyz = group.groupby("window_id")[["X", "Y", "Z"]].first().loc[per_window.index].to_numpy(float)
            empty_xyz = empty_all[empty_all.family == family].groupby("window_id")[["X", "Y", "Z"]].first()                 .to_numpy(float)
            axis = figure.add_subplot(grid[0, 2 * i:2 * i + 2], projection="3d")
            maps.append(axis)
            if len(empty_xyz):
                axis.scatter(*empty_xyz.T, s=size, facecolors="none", edgecolors=FAINT, linewidths=.6,
                             depthshade=False,
                             label=f"No {family} traces, $P_{{32}}$ = 0 ({len(empty_xyz)} windows)")
                axis.legend(loc="upper left", frameon=False, fontsize=9)
            image = axis.scatter(*xyz.T, c=cov, s=size, cmap=SEQUENTIAL_POINTS, linewidths=0, depthshade=False)
            _true_aspect(axis, all_xyz)
            axis.view_init(elev=elev, azim=azim)
            axis.set(xlabel=f"X ({unit})", ylabel=f"Y ({unit})", zlabel=f"Z ({unit})")
            axis.set_title(f"{family}: variation of $P_{{32}}$ between realizations", fontsize=13)
            bar = figure.colorbar(image, ax=axis, orientation="horizontal", shrink=.6, pad=.02,
                                  label=r"CoV of $P_{32}$")
            bar.outline.set_visible(False)
            stats_axes[0].hist(cov.dropna(), bins=30, density=True, histtype="stepfilled", alpha=.35,
                               color=colors[family], edgecolor=colors[family],
                               label=f"{family} (median {cov.median():.2f})")
            wide = group.pivot(index="window_id", columns="realization", values="P32").sort_index(axis=1)
            running = wide.cumsum(axis=1) / np.arange(1, wide.shape[1] + 1)
            final = running.iloc[:, -1]
            relative = (running.sub(final, axis=0).abs().div(final, axis=0)).median(axis=0)
            stats_axes[1].plot(np.arange(1, len(relative) + 1), 100 * relative, "o-", ms=3,
                               color=colors[family], label=family)
        stats_axes[0].set(xlabel=r"CoV of $P_{32}$ across realizations", ylabel="Density",
                          title="Between-realization variability")
        stats_axes[1].set(xlabel="Number of realizations", ylabel="Median |running − final| (%)",
                          title="Convergence of the window mean", ylim=(0, None))
        for axis in stats_axes:
            axis.legend()
            axis.grid(alpha=.6)
    return figure, maps + stats_axes

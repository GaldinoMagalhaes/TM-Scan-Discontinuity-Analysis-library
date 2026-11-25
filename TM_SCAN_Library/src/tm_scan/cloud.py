"""Point-cloud description: size, potential occlusions and grid nodes vs k_smooth."""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree


def _xyz(cloud):
    scan = getattr(cloud, "scan", cloud)
    if isinstance(scan, pd.DataFrame):
        scan = scan[["X", "Y", "Z"]].to_numpy(float)
    points = np.asarray(scan, float)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("cloud must be a Dataset, a DataFrame with X/Y/Z or an (n, 3) array.")
    points = points[:, :3]
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 20:
        raise ValueError("At least 20 finite points are required.")
    return points


def _mean_spacing(points):
    """Mean spacing of the library's voxel rule (see sampling._voxel_centers_from_scan)."""
    extent = points.max(axis=0) - points.min(axis=0)
    zero_tol = 64 * np.finfo(float).eps * max(float(extent.max()), np.finfo(float).tiny)
    if np.any(extent <= zero_tol):
        unique = np.unique(points, axis=0)
        neighbors = min(10, len(unique) - 1)
        return float(np.median(cKDTree(unique).query(unique, k=[neighbors + 1])[0][:, 0]))
    return float((np.prod(np.maximum(extent, 1e-12)) / len(points)) ** (1 / 3))


def grid_node_counts(cloud, k_values=(1, 1.5, 2, 3, 4, 6, 8)):
    """Number of candidate grid nodes (occupied voxels) for each k_smooth.

    Uses the same spacing and voxel rule as `generate_centers`, without
    building the nodes, so it is fast even for large clouds.
    """
    points = _xyz(cloud)
    spacing = _mean_spacing(points)
    origin = points.min(axis=0)
    rows = []
    for k in k_values:
        k = float(k)
        if not np.isfinite(k) or k <= 0:
            raise ValueError("k_values must be positive.")
        voxels = np.floor((points - origin) / (k * spacing)).astype(np.int64)
        rows.append(dict(k_smooth=k, voxel_size=k * spacing, grid_nodes=len(np.unique(voxels, axis=0))))
    return pd.DataFrame(rows)



# ------------------------------------------------- angle-criterion occlusions ---
def _max_angular_gap(vectors, axes, direction=False):
    """vectors (n, k, 3); axes (n, 3, 2) in-plane basis -> largest angular gap (rad) per row.

    With direction=True also returns the unit 3D vector pointing to the middle of
    that gap (the side of the point where the surface is missing)."""
    uv = np.einsum("nki,nij->nkj", vectors, axes)
    angle = np.sort(np.arctan2(uv[..., 1], uv[..., 0]), axis=1)
    gaps = np.concatenate([np.diff(angle, axis=1), (angle[:, :1] + 2 * np.pi - angle[:, -1:])], axis=1)
    largest = gaps.argmax(axis=1)
    gap = gaps[np.arange(len(gaps)), largest]
    if not direction:
        return gap
    middle = angle[np.arange(len(angle)), largest] + gap / 2
    vector = np.cos(middle)[:, None] * axes[:, :, 0] + np.sin(middle)[:, None] * axes[:, :, 1]
    return gap, vector


def _boundary_lines(cloud, boundary):
    """Outer-edge polylines: explicit list of (m, 3) arrays, or the boundary of a Dataset."""
    if boundary is not None:
        return [np.asarray(line, float)[:, :3] for line in boundary]
    frame = getattr(cloud, "boundary", None)
    if frame is None:
        return []
    lines = []
    for geometry in frame.geometry:
        for line in (geometry.geoms if geometry.geom_type == "MultiLineString" else [geometry]):
            lines.append(np.asarray(line.coords, float)[:, :3])
    return lines


def _sample_polylines(lines, step):
    parts = []
    for line in lines:
        for a, b in zip(line[:-1], line[1:]):
            n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
            parts.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
    return np.vstack(parts) if parts else np.empty((0, 3))


def _empty_area(tree, points, centre, axes, uv, threshold, max_cells=1_000_000):
    """Area inside the rim (its own plane) with no projected point closer than `threshold`."""
    from scipy.spatial import Delaunay, ConvexHull
    hull_area = ConvexHull(uv).volume
    step = max(threshold / 2, np.sqrt(hull_area / max_cells))
    reach = np.linalg.norm(uv, axis=1).max() + 3 * threshold
    local_uv = (points[tree.query_ball_point(centre, reach)] - centre) @ axes[:2].T
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    gu, gv = np.meshgrid(np.arange(lo[0], hi[0], step), np.arange(lo[1], hi[1], step), indexing="ij")
    grid = np.column_stack([gu.ravel(), gv.ravel()])
    grid = grid[Delaunay(uv).find_simplex(grid) >= 0]
    if not len(grid):
        return 0.0
    return float((cKDTree(local_uv).query(grid)[0] > threshold).sum() * step ** 2)


def _angle_holes(points, lines, *, k=24, gap_deg=120.0, closure_deg=45.0, link=5.0, min_rim_points=8,
                 min_inward=0.75, empty_factor=0.66, boundary_tol=None, chunk=50_000):
    """Angle criterion (Bendels, Schnabel & Klein 2006) on the 3D cloud; see describe_cloud."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    tree = cKDTree(points)
    distance, index = tree.query(points, k=k + 1, workers=-1)
    area_per_point = np.pi * distance[:, -1] ** 2 / k
    # side of the area represented by one point: independent of how regular the sampling is
    spacing = float(np.sqrt(np.median(area_per_point)))
    gap, empty_side = np.empty(len(points)), np.empty((len(points), 3))
    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        neighbours = points[index[start:stop, 1:]] - points[start:stop, None, :]
        centred = neighbours - neighbours.mean(axis=1, keepdims=True)
        frames = np.linalg.eigh(np.einsum("nki,nkj->nij", centred, centred))[1]
        gap[start:stop], empty_side[start:stop] = _max_angular_gap(neighbours, frames[:, :, 1:], direction=True)
    candidate = gap > np.radians(gap_deg)
    near_edge = np.zeros(len(points), bool)
    if lines:
        tol = boundary_tol or 4 * spacing
        near_edge = cKDTree(_sample_polylines(lines, spacing / 2)).query(points, distance_upper_bound=tol)[0] < tol
    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    rim_index = np.flatnonzero(candidate & ~near_edge)
    rim_label = np.full(len(points), -1)
    holes = []
    if len(rim_index):
        pairs = cKDTree(points[rim_index]).query_pairs(link * spacing, output_type="ndarray")
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(rim_index),) * 2)
        count, labels = connected_components(graph, directed=False)
        for label in range(count):
            members = rim_index[labels == label]
            if len(members) < min_rim_points:
                continue
            rim = points[members]
            if np.linalg.norm(rim.max(axis=0) - rim.min(axis=0)) > 0.5 * diagonal:
                continue          # rim of the whole cloud (outer edge without a mapped boundary)
            centre = rim.mean(axis=0)
            axes = np.linalg.svd(rim - centre, full_matrices=False)[2]
            closure = np.degrees(_max_angular_gap((rim - centre)[None], axes[:2].T[None])[0])
            if closure >= closure_deg:
                continue          # open line, not a closed rim
            inward = np.einsum("ij,ij->i", empty_side[members], centre - rim)
            if np.mean(inward > 0) < min_inward:
                continue          # the empty side faces outwards: outer edge of the surface, not a hole
            around = np.unique(index[members, 1:])
            around = around[~candidate[around]]               # surface points next to the rim
            local = float(np.sqrt(np.median(area_per_point[around]))) if len(around) else spacing
            threshold = empty_factor * max(spacing, local)
            area = _empty_area(tree, points, centre, axes, (rim - centre) @ axes[:2].T, threshold)
            if area <= 0:
                continue
            rim_label[members] = len(holes)
            holes.append(dict(hole=len(holes), area=area, rim_points=len(members),
                              X=centre[0], Y=centre[1], Z=centre[2]))
    surface = float(len(points) * np.median(area_per_point))
    table = pd.DataFrame(holes, columns=["hole", "area", "rim_points", "X", "Y", "Z"])
    return table, surface, rim_label, candidate & near_edge


@dataclass
class CloudDescription:
    """Result of `describe_cloud`; print it, or use `.summary()` / `plot_cloud`."""
    n_points: int
    spacing: float                # median distance to the nearest neighbour
    cell: float                   # edge of the occlusion grid (projection method only)
    surface_area: float           # projected footprint (surface + inner gaps)
    gap_area: float
    n_gaps: int
    largest_gap: float
    median_density: float         # points per square unit
    low_density_fraction: float   # fraction of points below `low_density` x median
    low_density: float
    grid_nodes: pd.DataFrame      # k_smooth, voxel_size, grid_nodes
    length_unit: str = "m"
    method: str = "angle"
    holes: pd.DataFrame = field(default_factory=pd.DataFrame)   # one row per detected occlusion
    maps: dict = field(default_factory=dict, repr=False)

    @property
    def gap_fraction(self):
        if self.method == "angle":       # surface_area holds the sampled surface only
            total = self.surface_area + self.gap_area
            return self.gap_area / total if total > 0 else np.nan
        return self.gap_area / self.surface_area if self.surface_area > 0 else np.nan

    def summary(self):
        u = self.length_unit
        return pd.Series({
            "points": self.n_points,
            f"typical spacing ({u})": self.spacing,
            "gap method": self.method,
            (f"sampled surface ({u}2)" if self.method == "angle" else f"projected surface ({u}2)"): self.surface_area,
            "surface in gaps (%)": 100 * self.gap_fraction,
            "gaps": self.n_gaps,
            f"largest gap ({u}2)": self.largest_gap,
            f"median density (pts/{u}2)": self.median_density,
            f"points below {self.low_density:.0%} of median density (%)": 100 * self.low_density_fraction,
        })

    def __str__(self):
        u = self.length_unit
        lines = ["POINT CLOUD",
                 f"  points                         : {self.n_points:,}",
                 f"  typical spacing                : {self.spacing:.4g} {u}",
                 (f"  sampled surface (3D)           : {self.surface_area:.4g} {u}²" if self.method == "angle" else
                  f"  projected surface              : {self.surface_area:.4g} {u}²"),
                 "POTENTIAL OCCLUSIONS",
                 f"  method                         : {'angle criterion (3D)' if self.method == 'angle' else 'projection on one plane'}",
                 f"  surface in gaps                : {100 * self.gap_fraction:.2f} %  "
                 f"({self.n_gaps} gaps; largest {self.largest_gap:.3g} {u}²)",
                 f"  points in partial shadow       : {100 * self.low_density_fraction:.2f} %  "
                 f"(density < {self.low_density:.0%} of the median, {self.median_density:.4g} pts/{u}²)",
                 "GRID NODES BY k_smooth"]
        lines += [f"  k = {row.k_smooth:<4g} voxel {row.voxel_size:.4g} {u:<3}: {int(row.grid_nodes):,} nodes"
                  for row in self.grid_nodes.itertuples()]
        return "\n".join(lines)


def describe_cloud(cloud, *, method="angle", boundary=None, k_values=(1, 1.5, 2, 3, 4, 6, 8),
                   k_neighbors=10, low_density=0.25, max_points=2_000_000, seed=0, length_unit=None,
                   cell_factor=5.0, min_gap_cells=4):
    """Describe a point cloud: points, potential occlusions and grid nodes vs k_smooth.

    Gaps (unsampled areas), method='angle' (default): angle criterion of
    Bendels, Schnabel & Klein (2006), on the 3D cloud without any global
    projection. Each point's neighbours are projected on its LOCAL tangent
    plane; points whose largest angular gap between consecutive neighbours
    exceeds 120 deg (24 neighbours) lie on the rim of an unsampled area. Rims near the outer
    boundary (the Dataset boundary, or `boundary` = list of (m, 3) polylines)
    are discarded; only closed rims whose empty side faces inwards are kept,
    and each gap area is the empty area inside its rim, on the rim's own plane. Valid on curved surfaces
    (tunnels, benches).
    method='projection' keeps the former estimate on the best-fit plane of
    the whole cloud (grid of `cell_factor` x spacing; regions under
    `min_gap_cells` cells ignored); it is only meaningful for planar faces.

    Partial shadows: points whose local density (k nearest neighbours) is
    below `low_density` x the median density.
    Clouds above `max_points` are randomly subsampled for both estimates.
    """
    if method not in ("angle", "projection"):
        raise ValueError("method must be 'angle' or 'projection'.")
    points = _xyz(cloud)
    unit = length_unit or getattr(cloud, "length_unit", "m")
    nodes = grid_node_counts(points, k_values)
    total = len(points)
    work = points
    if total > max_points:
        work = points[np.random.default_rng(seed).choice(total, max_points, replace=False)]
    tree = cKDTree(work)
    distances = tree.query(work, k=k_neighbors + 1, workers=-1)[0]
    first = distances[:, 1]
    spacing = float(np.median(first[first > 0]))
    rk = distances[:, -1]
    density = np.where(rk > 0, (k_neighbors - 1) / (np.pi * rk ** 2), np.nan)
    median_density = float(np.nanmedian(density))
    low = density < low_density * median_density
    center = work.mean(axis=0)
    _, _, axes = np.linalg.svd(work - center, full_matrices=False)
    uv = (work - center) @ axes[:2].T
    scale = total / len(work)      # density of the full cloud
    maps = dict(relative_density=density / median_density, xyz=work, center=center, axes=axes,
                uv=uv - uv.min(axis=0), uv_origin=uv.min(axis=0))
    if method == "angle":
        holes, surface, rim_label, edge = _angle_holes(work, _boundary_lines(cloud, boundary))
        maps.update(rim_label=rim_label, outer_edge=edge)
        gap_area = float(holes.area.sum()) if len(holes) else 0.0
        return CloudDescription(
            n_points=total, spacing=spacing / np.sqrt(scale), cell=np.nan, surface_area=surface,
            gap_area=gap_area, n_gaps=len(holes), largest_gap=float(holes.area.max()) if len(holes) else 0.0,
            median_density=median_density * scale, low_density_fraction=float(np.nanmean(low)),
            low_density=low_density, grid_nodes=nodes, length_unit=unit, method="angle", holes=holes, maps=maps)
    cell = cell_factor * spacing
    ij = np.floor((uv - uv.min(axis=0)) / cell).astype(int)
    occupied = np.zeros(ij.max(axis=0) + 1, dtype=bool)
    occupied[ij[:, 0], ij[:, 1]] = True
    footprint = ndimage.binary_fill_holes(occupied)
    empty = footprint & ~occupied
    labels, count = ndimage.label(empty)
    sizes = np.asarray(ndimage.sum(empty, labels, index=np.arange(1, count + 1))) if count else np.empty(0)
    valid = np.flatnonzero(sizes >= min_gap_cells) + 1
    gaps = np.isin(labels, valid)
    cell_area = cell ** 2
    maps.update(footprint=footprint, gaps=gaps, cell=cell)
    return CloudDescription(
        n_points=total, spacing=spacing / np.sqrt(scale), cell=cell,
        surface_area=float(footprint.sum() * cell_area), gap_area=float(gaps.sum() * cell_area),
        n_gaps=int(len(valid)), largest_gap=float(sizes[valid - 1].max() * cell_area) if len(valid) else 0.0,
        median_density=median_density * scale, low_density_fraction=float(np.nanmean(low)),
        low_density=low_density, grid_nodes=nodes, length_unit=unit, method="projection", maps=maps)


def _gap_points(maps, k=8):
    """3D positions of gap cells: cell centre on the best-fit plane, lifted to the
    local surface height (mean offset of the nearest points along the plane normal)."""
    cells = np.argwhere(maps["gaps"])
    if not len(cells):
        return np.empty((0, 3))
    uv_cells = (cells + 0.5) * maps["cell"] + maps["uv_origin"]
    xyz, center, axes = maps["xyz"], maps["center"], maps["axes"]
    uv = (xyz - center) @ axes[:2].T
    height = (xyz - center) @ axes[2]
    index = cKDTree(uv).query(uv_cells, k=min(k, len(uv)))[1]
    w = height[index].mean(axis=1) if index.ndim > 1 else height[index]
    return center + uv_cells @ axes[:2] + w[:, None] * axes[2]


def plot_cloud(description, *, view="3d", panels=("a", "b", "c"), max_points=80_000, seed=0,
               elev=None, azim=None, figsize=None):
    """(a) Gaps, (b) relative point density and (c) grid nodes vs k_smooth.

    view='3d' (default) draws (a) and (b) on the real geometry, with true
    proportions; view='plane' draws them on the best-fit plane.
    panels chooses which panels to draw, e.g. panels="b" or ("a", "b").
    Returns (figure, axes), with one axis per drawn panel, in panel order.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from .style import style_context, SEQUENTIAL_POINTS as SEQUENTIAL, ACCENT, TOTAL, FAINT
    from .plotting import _face_view, _true_aspect
    if view not in ("3d", "plane"):
        raise ValueError("view must be '3d' or 'plane'.")
    panels = [p.lower() for p in ([panels] if isinstance(panels, str) else panels)]
    panels = [p for p in ("a", "b", "c") if p in panels]
    if not panels:
        raise ValueError("panels must contain at least one of 'a', 'b', 'c'.")
    maps, unit = description.maps, description.length_unit
    rng = np.random.default_rng(seed)
    ratios = {"a": 1.4, "b": 1.4, "c": 1.0}
    column = {p: i for i, p in enumerate(panels)}
    with style_context():
        width = 17 * sum(ratios[p] for p in panels) / 3.8 + (0.6 if "b" in panels else 0)
        figure = plt.figure(figsize=figsize or (max(width, 5.5), 5.4), layout="constrained")
        grid = figure.add_gridspec(1, len(panels), width_ratios=[ratios[p] for p in panels])
        hidden = plt.figure()          # unselected panels are drawn here and discarded

        def slot(name, **kwargs):
            if name in column:
                return figure.add_subplot(grid[0, column[name]], **kwargs)
            return hidden.add_subplot(**kwargs)
        title_a = f"(a) Gaps: {100 * description.gap_fraction:.2f} % of the surface ({description.n_gaps})"
        title_b = f"(b) Partial shadows: {100 * description.low_density_fraction:.2f} % of points"
        relative = maps["relative_density"]
        if view == "3d":
            xyz = maps["xyz"]
            pick = rng.choice(len(xyz), min(len(xyz), max_points), replace=False)
            default_elev, default_azim = _face_view(xyz[pick])
            elev = default_elev if elev is None else elev
            azim = default_azim if azim is None else azim
            ax_a = slot("a", projection="3d")
            ax_a.scatter(*xyz[pick].T, s=.3, color=FAINT, alpha=.35, linewidths=0, rasterized=True)
            if description.method == "angle":
                rims = maps["rim_label"] >= 0
                if rims.any():
                    ax_a.scatter(*xyz[rims].T, s=3, color=ACCENT, linewidths=0, label="Gap rims")
                    ax_a.legend(loc="upper left", fontsize=9, markerscale=3)
            else:
                gaps = _gap_points(maps)
                if len(gaps):
                    ax_a.scatter(*gaps.T, s=6, color=ACCENT, linewidths=0, label="Gap cells")
                    ax_a.legend(loc="upper left", fontsize=9)
            ax_b = slot("b", projection="3d")
            points = ax_b.scatter(*xyz[pick].T, c=relative[pick], s=.5, cmap=SEQUENTIAL, vmin=0, vmax=2,
                                  linewidths=0, rasterized=True)
            for axis, title in ((ax_a, title_a), (ax_b, title_b)):
                _true_aspect(axis, xyz)
                axis.view_init(elev=elev, azim=azim)
                axis.set(xlabel=f"X ({unit})", ylabel=f"Y ({unit})", zlabel=f"Z ({unit})")
                axis.set_title(title, fontsize=12)
        else:
            ax_a = slot("a")
            if description.method == "angle":
                uv_all = maps["uv"]
                pick_a = rng.choice(len(uv_all), min(len(uv_all), max_points), replace=False)
                ax_a.scatter(*uv_all[pick_a].T, s=.3, color=FAINT, alpha=.35, linewidths=0, rasterized=True)
                rims = maps["rim_label"] >= 0
                ax_a.scatter(*uv_all[rims].T, s=2, color=ACCENT, linewidths=0)
            else:
                image = np.full(maps["footprint"].shape, np.nan)
                image[maps["footprint"]] = 0
                image[maps["gaps"]] = 1
                extent = [0, image.shape[0] * maps["cell"], 0, image.shape[1] * maps["cell"]]
                ax_a.imshow(image.T, origin="lower", extent=extent, cmap=ListedColormap(["#DCDBD6", ACCENT]),
                            vmin=0, vmax=1, interpolation="nearest")
            ax_b = slot("b")
            uv = maps["uv"]
            pick = rng.choice(len(uv), min(len(uv), max_points), replace=False)
            points = ax_b.scatter(uv[pick, 0], uv[pick, 1], c=relative[pick], s=.4, cmap=SEQUENTIAL,
                                  vmin=0, vmax=2, rasterized=True, linewidths=0)
            for axis, title in ((ax_a, title_a), (ax_b, title_b)):
                axis.set(title=title, xlabel=f"u ({unit})", ylabel=f"v ({unit})", aspect="equal")
        if "b" in column:
            bar = figure.colorbar(points, ax=ax_b, label="Density / median", shrink=0.7)
            bar.outline.set_visible(False)
        ax_c = slot("c")
        nodes = description.grid_nodes
        ax_c.plot(nodes.k_smooth, nodes.grid_nodes, "o-", color=TOTAL, ms=5)
        ax_c.set(xscale="log", yscale="log", xlabel=r"$k_{smooth}$", ylabel="Grid nodes",
                 title=r"(c) Grid nodes vs $k_{smooth}$")
        ax_c.set_xticks(nodes.k_smooth, [f"{k:g}" for k in nodes.k_smooth])
        ax_c.minorticks_off()
        ax_c.grid(True, axis="y")
        ax_c.set_box_aspect(1)
        plt.close(hidden)
    drawn = {"a": ax_a, "b": ax_b, "c": ax_c}
    return figure, tuple(drawn[p] for p in panels)

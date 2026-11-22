"""A composable analysis API; scientific calculations remain in the cores."""
from time import perf_counter
import math
import numpy as np
from .models import AnalysisResult, Sampling, Window, positive
from .sampling import _voxel_centers_from_scan
from .sphere import SphereAnalyzer
from .rectangle import RectangleAnalyzer
from ._execution import validate_jobs


RECTANGLE_COLUMNS = ["Rectangle_ID", "X", "Y", "Z", "N_traces", "N_transecting",
                     "N_contained", "P21", "P20", "Nx", "Ny", "Nz", "μ",
                     "sum_L_real", "sum_L_proj"]


def generate_centers(dataset, sampling=None):
    """Return candidate XYZ centers and voxel edge, before window filtering.

    The edge is NaN when all points are used or spacing is undefined.
    """
    sampling = sampling or Sampling()
    return _voxel_centers_from_scan(dataset.scan, sampling.k_smooth,
                                    use_all_points=sampling.use_all_points)


def _orientations(dataset, orientations):
    if not isinstance(orientations, dict):
        raise ValueError("Rectangles require family_orientations={family: (dip, dip_dir)}.")
    normalized = {}
    for family in dataset.traces:
        if family not in orientations:
            raise ValueError(f"Missing mean orientation for {family!r}.")
        values = orientations[family]
        if isinstance(values, dict):
            values = (values.get("dip"), values.get("dip_dir"))
        try:
            dip, dip_dir = map(float, values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid dip/dip_dir pair for {family!r}.") from exc
        if not (math.isfinite(dip) and 0 <= dip <= 90 and
                math.isfinite(dip_dir) and 0 <= dip_dir <= 360):
            raise ValueError("dip must be in [0, 90] and dip_dir in [0, 360] degrees.")
        normalized[family] = (dip, dip_dir)
    return normalized


def _rectangle_analyzer(window, sampling, options, verbose):
    scale = min(window.size_u, window.size_v)
    parameters = dict(size_u=window.size_u, size_v=window.size_v,
                      inplane_rotation_cw=window.rotation_cw,
                      ref_dir=np.array([1., 0., 0.]), plane_tol=.03 * scale,
                      rect_buffer=.05 * scale, proj_band=.35 * scale,
                      inner_strip_2d=.5 * scale, min_neighbors=20,
                      up=np.array([0., 0., 1.]), smooth_k=sampling.k_smooth, eps=1e-9)
    options = dict(options or {})
    allowed = {"ref_dir", "plane_tol", "rect_buffer", "min_neighbors", "up", "eps"}
    if set(options) - allowed:
        raise ValueError(f"Unknown or inactive rectangle options: {sorted(set(options) - allowed)}.")
    parameters.update(options)
    for name in ("plane_tol", "rect_buffer"):
        value = float(parameters[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    positive(parameters["eps"], "eps")
    neighbors = positive(parameters["min_neighbors"], "min_neighbors")
    if neighbors < 3 or neighbors != int(neighbors):
        raise ValueError("min_neighbors must be an integer >= 3.")
    for name in ("ref_dir", "up"):
        vector = np.asarray(parameters[name], float)
        if vector.shape != (3,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            raise ValueError(f"{name} must be a finite nonzero XYZ vector.")
    return RectangleAnalyzer(paths=None, verbose=verbose, **parameters)


def _centers_array(centers):
    points = np.asarray(centers, dtype=float)
    if points.size == 0:
        return np.empty((0, 3), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("centers must be a finite (n, 3) array.")
    return points.copy()


def analyze(dataset, window, *, sampling=None, family_orientations=None,
            centers=None, rectangle_options=None, verbose=False, n_jobs=-1, engine="numba",
            roughness_filter=False, roughness_options=None):
    """Run one analysis in memory, without prompts, plots or file writes.

    roughness_filter=False (default) keeps every window. With True, windows
    that are not planar at the window scale (rho = xi_R/R > tau, or projected
    coverage < min_cover) are rejected before the trace analysis; the full
    assessment, including rejected centres, is kept in `result.roughness`.
    roughness_options may set tau (0.08), min_cover (0.80) and radius
    (default: sphere radius, or the equal-area radius of the rectangle).

    Explicit centers bypass k_smooth and voxel sampling, but still undergo
    the window/boundary filters. Family orientations are required
    only for the rectangular estimator.
    """
    return _analyze(dataset, window, sampling=sampling,
                    family_orientations=family_orientations, centers=centers,
                    rectangle_options=rectangle_options, verbose=verbose, n_jobs=n_jobs, engine=engine,
                    roughness_filter=roughness_filter, roughness_options=roughness_options)


def _roughness_keep(dataset, centers, window, options):
    from .roughness import filter_centers
    return filter_centers(dataset.scan[["X", "Y", "Z"]].to_numpy(float), centers, window, options,
                          dataset.length_unit)


def _analyze(dataset, window, *, sampling=None, family_orientations=None,
             centers=None, rectangle_options=None, verbose=False, candidate_grid=None, n_jobs=-1, engine="numba",
             roughness_filter=False, roughness_options=None):
    start = perf_counter()
    validate_jobs(n_jobs)
    if engine not in ("numba", "reference"):
        raise ValueError("engine must be numba or reference.")
    dataset.validate()
    if not isinstance(window, Window):
        raise TypeError("window must be a Window.")
    sampling = sampling or Sampling()
    if not isinstance(sampling, Sampling):
        raise TypeError("sampling must be a Sampling.")
    if centers is not None and candidate_grid is not None:
        raise ValueError("Provide either explicit centers or a cached grid.")
    if not isinstance(roughness_filter, (bool, np.bool_)):
        raise ValueError("roughness_filter must be boolean.")
    if roughness_options and not roughness_filter:
        raise ValueError("roughness_options apply only with roughness_filter=True.")
    roughness = None
    if window.kind == "sphere" and rectangle_options:
        raise ValueError("rectangle_options apply to rectangles only.")
    orientations = _orientations(dataset, family_orientations) if window.kind == "rectangle" else None
    if centers is not None:
        candidates, voxel_size = _centers_array(centers), np.nan
        center_source = "provided"
    else:
        candidates, voxel_size = candidate_grid if candidate_grid is not None else generate_centers(dataset, sampling)
        center_source = "all_points" if sampling.use_all_points else "voxel"
    quads = []
    if window.kind == "sphere":
        analyzer = SphereAnalyzer(window.radius, verbose=verbose)
        accepted, _, after_border, _ = analyzer.filter_centers_border_then_inside(
            candidates, dataset.boundary, print_summary=verbose)
        if roughness_filter:
            before_roughness = len(accepted)
            accepted, roughness = _roughness_keep(dataset, accepted, window, roughness_options)
        if engine == "numba":
            from ._accelerated import analyze_spheres_fast
            tables = analyze_spheres_fast(analyzer, accepted, dataset.traces, dataset.scan, n_jobs)
        else:
            tables = analyzer.analyze_spheres(accepted, dataset.traces, dataset.scan, dataset.name)
        diagnostics = {"initial_centers": len(candidates),
                       "after_boundary_filter": len(after_border), "final_windows": len(accepted)}
    else:
        analyzer = _rectangle_analyzer(window, sampling, rectangle_options, verbose)
        geometry = analyzer.generate_and_filter_windows(dataset.scan, dataset.boundary, centers=candidates)
        if roughness_filter:
            # Assess the windows that passed the geometric filters, then rebuild the
            # window geometry from the admissible centres only.
            before_roughness = len(geometry["centers_kept"])
            kept, roughness = _roughness_keep(dataset, geometry["centers_kept"], window, roughness_options)
            geometry = analyzer.generate_and_filter_windows(dataset.scan, dataset.boundary, centers=kept)
        accepted, quads = geometry["centers_kept"], geometry["quads_final"]
        if engine == "numba":
            from ._accelerated import analyze_rectangles_fast
            tables = analyze_rectangles_fast(analyzer, geometry, dataset.traces, orientations, n_jobs)
        else:
            tables = {family: analyzer._analyze_one_family(
                        quads, traces, geometry["scan_xyz"], family, orientations).reindex(columns=RECTANGLE_COLUMNS)
                      for family, traces in dataset.traces.items()}
        diagnostics = geometry["diag"].copy()
        diagnostics["family_orientations"] = repr(orientations)
        diagnostics["rectangle_options"] = repr(rectangle_options or {})
    diagnostics.update(center_source=center_source,
                       mean_spacing=voxel_size / sampling.k_smooth,
                       elapsed_seconds=perf_counter() - start,
                       point_count=len(dataset.scan), source=dataset.source,
                       engine=engine, n_jobs=n_jobs,
                       sampling_fraction=len(accepted) / len(candidates) if len(candidates) else np.nan)
    diagnostics["roughness_filter"] = bool(roughness_filter)
    if roughness is not None:
        diagnostics.update(roughness_tau=roughness.tau, roughness_min_cover=roughness.min_cover,
                           roughness_radius=roughness.radius, before_roughness_filter=before_roughness,
                           rejected_by_roughness=before_roughness - int(roughness.admissible.sum()),
                           final_windows=len(accepted))
    return AnalysisResult(dataset.name, dataset.length_unit, window, sampling,
                          tables, accepted, candidates, voxel_size, diagnostics, quads, roughness)

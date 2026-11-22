"""One-factor studies with compact progress instead of per-window logs."""
from dataclasses import replace
from .analysis import _analyze, generate_centers
from .models import Sampling, StudyResult, Window, positive
from ._execution import Progress, validate_jobs


def _values(values, name):
    numbers = [positive(value, name) for value in values]
    if not numbers or len(numbers) != len(set(numbers)):
        raise ValueError(f"{name} must contain distinct positive values.")
    return numbers


def scale_study(dataset, windows, *, sampling=None, family_orientations=None,
                centers=None, rectangle_options=None, verbose=False,
                n_jobs=-1, progress=True, engine="numba",
                roughness_filter=False, roughness_options=None):
    """Reuse the grid and parallelize window blocks within each case.

    verbose is retained for compatibility. Studies suppress family/window
    logs even with verbose=True. Set progress=False for silent execution.
    """
    validate_jobs(n_jobs)
    windows = list(windows)
    if not windows or any(not isinstance(window, Window) for window in windows):
        raise ValueError("windows must be a nonempty sequence of Window objects.")
    if len({window.kind for window in windows}) != 1:
        raise ValueError("Use one window type per scale study.")
    if len(set(windows)) != len(windows):
        raise ValueError("Duplicate windows in scale study.")
    if len({window.rotation_cw for window in windows}) != 1:
        raise ValueError("Keep rectangle rotation fixed in a scale study.")
    sampling = sampling or Sampling()
    grid = generate_centers(dataset, sampling) if centers is None else None
    meter = Progress(len(windows), "Scale study", progress)
    runs = []
    for i, window in enumerate(windows, 1):
        runs.append(_analyze(dataset, window, sampling=sampling,
                             family_orientations=family_orientations, centers=centers,
                             rectangle_options=rectangle_options, verbose=False,
                             candidate_grid=grid, n_jobs=n_jobs, engine=engine,
                             roughness_filter=roughness_filter, roughness_options=roughness_options))
        meter.update(i)
    return StudyResult("radius" if windows[0].kind == "sphere" else "size", runs)


def radius_study(dataset, radii, *, sampling=None, centers=None, verbose=False,
                 n_jobs=-1, progress=True, engine="numba",
                roughness_filter=False, roughness_options=None):
    windows = [Window.sphere(radius) for radius in _values(radii, "radii")]
    return scale_study(dataset, windows, sampling=sampling, centers=centers,
                       verbose=verbose, n_jobs=n_jobs, progress=progress, engine=engine,
                       roughness_filter=roughness_filter, roughness_options=roughness_options)


def k_study(dataset, window, k_values, *, sampling=None, family_orientations=None,
            rectangle_options=None, verbose=False, n_jobs=-1, progress=True, engine="numba",
                roughness_filter=False, roughness_options=None):
    validate_jobs(n_jobs)
    sampling = sampling or Sampling()
    if sampling.use_all_points:
        raise ValueError("k_smooth has no effect when use_all_points=True.")
    numbers = _values(k_values, "k_values")
    meter = Progress(len(numbers), "k study", progress)
    runs = []
    for i, value in enumerate(numbers, 1):
        runs.append(_analyze(dataset, window, sampling=replace(sampling, k_smooth=value),
                             family_orientations=family_orientations, rectangle_options=rectangle_options,
                             verbose=False, n_jobs=n_jobs, engine=engine,
                             roughness_filter=roughness_filter, roughness_options=roughness_options))
        meter.update(i)
    return StudyResult("k_smooth", runs)

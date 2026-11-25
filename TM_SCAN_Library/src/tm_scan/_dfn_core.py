"""DFN geometry kernels and stochastic fracture generator.

Internal module: every parameter reaches it through ``SimulationConfig``, which
``dfn.simulate_dfn`` builds from the user's ``FractureFamily`` and ``DFNConfig``.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from numba import njit
from ._dfn_geometry import _rectangle_trace_lengths_batch


_CUBE_VERTICES = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)


_CUBE_EDGES = np.array(
    [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [0, 2],
        [1, 3],
        [4, 6],
        [5, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ],
    dtype=np.int64,
)


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters shared by every sampling window (no defaults: always explicit)."""

    half_size: float
    window_radius: float
    diameter_mean: float
    diameter_std: float
    diameter_min: float
    diameter_max: float
    fisher_dip: float
    fisher_dip_direction: float
    fisher_kappa: float
    batch_size: int
    max_fractures: int
    p32_boundary: str
    geometry_tolerance: float


def dip_direction_to_normal(dip: float, dip_direction: float) -> np.ndarray:
    """Convert modeling-convention dip/dip-direction angles to an upper-hemisphere normal.

    Dip direction is clockwise from +Y, matching the numerical-modeling convention and the
    ``Nx, Ny, Nz`` columns of the TM-Scan window tables.
    """

    dip_rad = math.radians(dip)
    direction_rad = math.radians(dip_direction)
    sin_dip = math.sin(dip_rad)
    normal = np.array(
        [
            -sin_dip * math.sin(direction_rad),
            -sin_dip * math.cos(direction_rad),
            math.cos(dip_rad),
        ],
        dtype=np.float64,
    )
    return normal / np.linalg.norm(normal)


@njit(cache=True, nogil=True)
def _trace_length_single(
    center: np.ndarray,
    normal: np.ndarray,
    fracture_radius: float,
    window_normal: np.ndarray,
    window_radius: float,
    tolerance: float,
) -> float:
    """Return the exact common-segment length of two finite planar disks.

    The sampling-window centre is the local origin. For nonparallel planes,
    their common line cuts an interval from each disk; the trace is the overlap
    between those intervals. Exactly coplanar disks have zero probability under
    the continuous stochastic model and are conservatively ignored.
    """

    # Direction of the line shared by both planes: window_normal x normal.
    ux = window_normal[1] * normal[2] - window_normal[2] * normal[1]
    uy = window_normal[2] * normal[0] - window_normal[0] * normal[2]
    uz = window_normal[0] * normal[1] - window_normal[1] * normal[0]
    sin_angle_sq = ux * ux + uy * uy + uz * uz

    if sin_angle_sq <= tolerance * tolerance:
        return 0.0

    sin_angle = math.sqrt(sin_angle_sq)
    ux /= sin_angle
    uy /= sin_angle
    uz /= sin_angle

    center_dot_fracture_normal = (
        center[0] * normal[0]
        + center[1] * normal[1]
        + center[2] * normal[2]
    )
    center_dot_window_normal = (
        center[0] * window_normal[0]
        + center[1] * window_normal[1]
        + center[2] * window_normal[2]
    )

    distance_window_to_line = abs(center_dot_fracture_normal) / sin_angle
    distance_fracture_to_line = abs(center_dot_window_normal) / sin_angle

    if (
        distance_window_to_line > window_radius + tolerance
        or distance_fracture_to_line > fracture_radius + tolerance
    ):
        return 0.0

    window_chord_sq = (
        window_radius * window_radius
        - distance_window_to_line * distance_window_to_line
    )
    fracture_chord_sq = (
        fracture_radius * fracture_radius
        - distance_fracture_to_line * distance_fracture_to_line
    )
    if window_chord_sq < 0.0:
        window_chord_sq = 0.0
    if fracture_chord_sq < 0.0:
        fracture_chord_sq = 0.0

    window_half_chord = math.sqrt(window_chord_sq)
    fracture_half_chord = math.sqrt(fracture_chord_sq)
    fracture_interval_center = center[0] * ux + center[1] * uy + center[2] * uz

    lower = max(-window_half_chord, fracture_interval_center - fracture_half_chord)
    upper = min(window_half_chord, fracture_interval_center + fracture_half_chord)
    length = upper - lower
    return length if length > tolerance else 0.0


@njit(cache=True, nogil=True)
def _trace_lengths_batch(
    centers: np.ndarray,
    normals: np.ndarray,
    radii: np.ndarray,
    window_normal: np.ndarray,
    window_radius: float,
    tolerance: float,
) -> np.ndarray:
    lengths = np.empty(radii.size, dtype=np.float64)
    for index in range(radii.size):
        lengths[index] = _trace_length_single(
            centers[index],
            normals[index],
            radii[index],
            window_normal,
            window_radius,
            tolerance,
        )
    return lengths


@njit(cache=True, nogil=True)
def _edge_circle_intersection_area(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    radius: float,
    tolerance: float,
) -> float:
    """Signed area of circle intersection with triangle O-A-B."""

    dx = bx - ax
    dy = by - ay
    segment_sq = dx * dx + dy * dy
    if segment_sq <= tolerance * tolerance:
        return 0.0

    radius_sq = radius * radius
    linear = 2.0 * (ax * dx + ay * dy)
    constant = ax * ax + ay * ay - radius_sq
    discriminant = linear * linear - 4.0 * segment_sq * constant

    parameters = np.empty(4, dtype=np.float64)
    parameters[0] = 0.0
    parameters[1] = 1.0
    parameter_count = 2

    if discriminant > tolerance:
        root = math.sqrt(discriminant)
        first = (-linear - root) / (2.0 * segment_sq)
        second = (-linear + root) / (2.0 * segment_sq)
        if tolerance < first < 1.0 - tolerance:
            parameters[parameter_count] = first
            parameter_count += 1
        if tolerance < second < 1.0 - tolerance:
            parameters[parameter_count] = second
            parameter_count += 1

    # Insertion sort; there are at most four values.
    for i in range(1, parameter_count):
        value = parameters[i]
        j = i - 1
        while j >= 0 and parameters[j] > value:
            parameters[j + 1] = parameters[j]
            j -= 1
        parameters[j + 1] = value

    area = 0.0
    for i in range(parameter_count - 1):
        first = parameters[i]
        second = parameters[i + 1]
        if second - first <= tolerance:
            continue

        px = ax + first * dx
        py = ay + first * dy
        qx = ax + second * dx
        qy = ay + second * dy
        midpoint_x = 0.5 * (px + qx)
        midpoint_y = 0.5 * (py + qy)
        cross = px * qy - py * qx

        if midpoint_x * midpoint_x + midpoint_y * midpoint_y <= radius_sq:
            area += 0.5 * cross
        else:
            dot = px * qx + py * qy
            area += 0.5 * radius_sq * math.atan2(cross, dot)

    return area


@njit(cache=True, nogil=True)
def _disk_area_in_box(
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    half_size: float,
    tolerance: float,
) -> float:
    """Area of a planar disk contained inside an axis-aligned cube.

    A fast projection test handles complete disks. Boundary disks are treated
    exactly by intersecting their plane with the cube and then calculating the
    circle-convex-polygon intersection in local 2-D coordinates.
    """

    full_area = math.pi * radius * radius
    fully_inside = True
    for axis in range(3):
        projected_extent_sq = 1.0 - normal[axis] * normal[axis]
        if projected_extent_sq < 0.0:
            projected_extent_sq = 0.0
        projected_extent = radius * math.sqrt(projected_extent_sq)
        if abs(center[axis]) + projected_extent > half_size + tolerance:
            fully_inside = False
            break
    if fully_inside:
        return full_area

    points = np.empty((12, 3), dtype=np.float64)
    point_count = 0
    duplicate_tolerance_sq = (100.0 * tolerance) ** 2

    for edge_index in range(_CUBE_EDGES.shape[0]):
        first_index = _CUBE_EDGES[edge_index, 0]
        second_index = _CUBE_EDGES[edge_index, 1]

        p0x = half_size * _CUBE_VERTICES[first_index, 0]
        p0y = half_size * _CUBE_VERTICES[first_index, 1]
        p0z = half_size * _CUBE_VERTICES[first_index, 2]
        p1x = half_size * _CUBE_VERTICES[second_index, 0]
        p1y = half_size * _CUBE_VERTICES[second_index, 1]
        p1z = half_size * _CUBE_VERTICES[second_index, 2]

        distance0 = (
            normal[0] * (p0x - center[0])
            + normal[1] * (p0y - center[1])
            + normal[2] * (p0z - center[2])
        )
        distance1 = (
            normal[0] * (p1x - center[0])
            + normal[1] * (p1y - center[1])
            + normal[2] * (p1z - center[2])
        )

        candidates = np.empty((2, 3), dtype=np.float64)
        candidate_count = 0
        if abs(distance0) <= tolerance and abs(distance1) <= tolerance:
            candidates[0, 0] = p0x
            candidates[0, 1] = p0y
            candidates[0, 2] = p0z
            candidates[1, 0] = p1x
            candidates[1, 1] = p1y
            candidates[1, 2] = p1z
            candidate_count = 2
        elif distance0 * distance1 <= 0.0:
            denominator = distance0 - distance1
            if abs(denominator) > tolerance:
                fraction = distance0 / denominator
                if -tolerance <= fraction <= 1.0 + tolerance:
                    candidates[0, 0] = p0x + fraction * (p1x - p0x)
                    candidates[0, 1] = p0y + fraction * (p1y - p0y)
                    candidates[0, 2] = p0z + fraction * (p1z - p0z)
                    candidate_count = 1

        for candidate_index in range(candidate_count):
            is_duplicate = False
            for existing_index in range(point_count):
                delta_x = candidates[candidate_index, 0] - points[existing_index, 0]
                delta_y = candidates[candidate_index, 1] - points[existing_index, 1]
                delta_z = candidates[candidate_index, 2] - points[existing_index, 2]
                if (
                    delta_x * delta_x + delta_y * delta_y + delta_z * delta_z
                    <= duplicate_tolerance_sq
                ):
                    is_duplicate = True
                    break
            if not is_duplicate and point_count < points.shape[0]:
                points[point_count, 0] = candidates[candidate_index, 0]
                points[point_count, 1] = candidates[candidate_index, 1]
                points[point_count, 2] = candidates[candidate_index, 2]
                point_count += 1

    if point_count < 3:
        return 0.0

    # Orthonormal basis spanning the fracture plane.
    if abs(normal[2]) < 0.9:
        reference_x, reference_y, reference_z = 0.0, 0.0, 1.0
    else:
        reference_x, reference_y, reference_z = 1.0, 0.0, 0.0

    basis1_x = reference_y * normal[2] - reference_z * normal[1]
    basis1_y = reference_z * normal[0] - reference_x * normal[2]
    basis1_z = reference_x * normal[1] - reference_y * normal[0]
    basis1_norm = math.sqrt(
        basis1_x * basis1_x + basis1_y * basis1_y + basis1_z * basis1_z
    )
    basis1_x /= basis1_norm
    basis1_y /= basis1_norm
    basis1_z /= basis1_norm

    basis2_x = normal[1] * basis1_z - normal[2] * basis1_y
    basis2_y = normal[2] * basis1_x - normal[0] * basis1_z
    basis2_z = normal[0] * basis1_y - normal[1] * basis1_x

    centroid_x = 0.0
    centroid_y = 0.0
    centroid_z = 0.0
    for index in range(point_count):
        centroid_x += points[index, 0]
        centroid_y += points[index, 1]
        centroid_z += points[index, 2]
    centroid_x /= point_count
    centroid_y /= point_count
    centroid_z /= point_count

    angles = np.empty(12, dtype=np.float64)
    order = np.empty(12, dtype=np.int64)
    for index in range(point_count):
        relative_x = points[index, 0] - centroid_x
        relative_y = points[index, 1] - centroid_y
        relative_z = points[index, 2] - centroid_z
        local_x = (
            relative_x * basis1_x
            + relative_y * basis1_y
            + relative_z * basis1_z
        )
        local_y = (
            relative_x * basis2_x
            + relative_y * basis2_y
            + relative_z * basis2_z
        )
        angles[index] = math.atan2(local_y, local_x)
        order[index] = index

    # Selection sort of the small plane/cube polygon (at most six vertices in
    # the nondegenerate case).
    for i in range(point_count - 1):
        minimum = i
        for j in range(i + 1, point_count):
            if angles[order[j]] < angles[order[minimum]]:
                minimum = j
        temporary = order[i]
        order[i] = order[minimum]
        order[minimum] = temporary

    local_points = np.empty((12, 2), dtype=np.float64)
    for sorted_index in range(point_count):
        source_index = order[sorted_index]
        relative_x = points[source_index, 0] - center[0]
        relative_y = points[source_index, 1] - center[1]
        relative_z = points[source_index, 2] - center[2]
        local_points[sorted_index, 0] = (
            relative_x * basis1_x
            + relative_y * basis1_y
            + relative_z * basis1_z
        )
        local_points[sorted_index, 1] = (
            relative_x * basis2_x
            + relative_y * basis2_y
            + relative_z * basis2_z
        )

    area = 0.0
    for index in range(point_count):
        next_index = (index + 1) % point_count
        area += _edge_circle_intersection_area(
            local_points[index, 0],
            local_points[index, 1],
            local_points[next_index, 0],
            local_points[next_index, 1],
            radius,
            tolerance,
        )

    area = abs(area)
    if area < 0.0:
        return 0.0
    if area > full_area:
        return full_area
    return area


@njit(cache=True, nogil=True)
def _disk_areas_in_box_batch(
    centers: np.ndarray,
    normals: np.ndarray,
    radii: np.ndarray,
    half_size: float,
    tolerance: float,
) -> np.ndarray:
    areas = np.empty(radii.size, dtype=np.float64)
    for index in range(radii.size):
        areas[index] = _disk_area_in_box(
            centers[index],
            normals[index],
            radii[index],
            half_size,
            tolerance,
        )
    return areas


def _sample_truncated_lognormal(
    random_generator: np.random.Generator,
    count: int,
    log_mean: float,
    log_std: float,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    values = random_generator.lognormal(log_mean, log_std, count)
    invalid = (values < minimum) | (values > maximum)
    while np.any(invalid):
        values[invalid] = random_generator.lognormal(log_mean, log_std, int(invalid.sum()))
        invalid = (values < minimum) | (values > maximum)
    return values


def _sample_fisher_normals(
    random_generator: np.random.Generator,
    count: int,
    mean_normal: np.ndarray,
    kappa: float,
) -> np.ndarray:
    """Sample axial plane normals from a 3-D von Mises-Fisher law."""

    uniform = random_generator.random(count)
    if kappa <= 1.0e-8:
        axial_component = 2.0 * uniform - 1.0
    else:
        # Stable inverse CDF for exp(kappa * cos(theta)) on the unit sphere.
        exponential_tail = math.exp(-2.0 * kappa)
        axial_component = 1.0 + np.log(
            uniform + (1.0 - uniform) * exponential_tail
        ) / kappa

    azimuth = 2.0 * math.pi * random_generator.random(count)
    radial_component = np.sqrt(np.maximum(0.0, 1.0 - axial_component**2))

    reference = (
        np.array([0.0, 0.0, 1.0])
        if abs(mean_normal[2]) < 0.9
        else np.array([1.0, 0.0, 0.0])
    )
    basis1 = np.cross(reference, mean_normal)
    basis1 /= np.linalg.norm(basis1)
    basis2 = np.cross(mean_normal, basis1)

    normals = (
        axial_component[:, None] * mean_normal[None, :]
        + radial_component[:, None]
        * (
            np.cos(azimuth)[:, None] * basis1[None, :]
            + np.sin(azimuth)[:, None] * basis2[None, :]
        )
    )

    # A fracture plane is axial: n and -n describe the same plane. The convention uses
    # the upper-hemisphere representation (0 <= dip <= 90 degrees).
    normals[normals[:, 2] < 0.0] *= -1.0
    return normals


def _lognormal_parameters(arithmetic_mean: float, arithmetic_std: float) -> tuple[float, float]:
    log_variance = math.log1p((arithmetic_std / arithmetic_mean) ** 2)
    return math.log(arithmetic_mean) - 0.5 * log_variance, math.sqrt(log_variance)


def simulate_realization(
    n_traces: int,
    window_normal: np.ndarray,
    config: SimulationConfig,
    seed: int,
    rectangle_frame=None,
    orientation=None,
) -> tuple[float, float, int]:
    """Run one realization and return P32, P21 and generated-fracture count."""

    if n_traces == 0:
        return 0.0, 0.0, 0

    random_generator = np.random.default_rng(seed)
    fracture_mean_normal = dip_direction_to_normal(
        config.fisher_dip, config.fisher_dip_direction
    )
    log_mean, log_std = _lognormal_parameters(
        config.diameter_mean, config.diameter_std
    )

    intersections = 0
    generated = 0
    accumulated_area = 0.0
    accumulated_trace_length = 0.0

    while intersections < n_traces:
        if generated >= config.max_fractures:
            raise RuntimeError(
                f"Stopping criterion was not reached after {config.max_fractures:,} fractures."
            )

        current_batch_size = min(config.batch_size, config.max_fractures - generated)
        centers = random_generator.uniform(
            -config.half_size,
            config.half_size,
            size=(current_batch_size, 3),
        )
        diameters = _sample_truncated_lognormal(
            random_generator,
            current_batch_size,
            log_mean,
            log_std,
            config.diameter_min,
            config.diameter_max,
        )
        radii = 0.5 * diameters
        if orientation is None:
            normals = _sample_fisher_normals(
                random_generator, current_batch_size,
                fracture_mean_normal, config.fisher_kappa)
        else:
            normals = orientation.rvs(current_batch_size, random_state=random_generator)

        if rectangle_frame is None:
            trace_lengths = _trace_lengths_batch(
                centers, normals, radii, window_normal,
                config.window_radius, config.geometry_tolerance)
        else:
            trace_lengths = _rectangle_trace_lengths_batch(
                centers, normals, radii, rectangle_frame,
                config.geometry_tolerance)
        hit_indices = np.flatnonzero(trace_lengths > 0.0)
        remaining = n_traces - intersections

        if hit_indices.size >= remaining:
            selected_count = int(hit_indices[remaining - 1]) + 1
        else:
            selected_count = current_batch_size

        selected_centers = centers[:selected_count]
        selected_normals = normals[:selected_count]
        selected_radii = radii[:selected_count]

        if config.p32_boundary == "clip":
            areas = _disk_areas_in_box_batch(
                selected_centers,
                selected_normals,
                selected_radii,
                config.half_size,
                config.geometry_tolerance,
            )
            accumulated_area += float(areas.sum())
        else:
            accumulated_area += math.pi * float(np.dot(selected_radii, selected_radii))

        accumulated_trace_length += float(trace_lengths[:selected_count].sum())
        intersections += int(np.count_nonzero(trace_lengths[:selected_count] > 0.0))
        generated += selected_count

    generation_volume = (2.0 * config.half_size) ** 3
    window_area = (math.pi * config.window_radius**2 if rectangle_frame is None
                   else 4.0 * rectangle_frame[2, 0] * rectangle_frame[2, 1])
    return (
        accumulated_area / generation_volume,
        accumulated_trace_length / window_area,
        generated,
    )


def _warm_up_compiled_geometry(config: SimulationConfig) -> None:
    """Compile Numba kernels before process-level parallel execution."""

    centers = np.zeros((1, 3), dtype=np.float64)
    normals = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    radii = np.ones(1, dtype=np.float64)
    window_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    _trace_lengths_batch(
        centers,
        normals,
        radii,
        window_normal,
        config.window_radius,
        config.geometry_tolerance,
    )
    if config.p32_boundary == "clip":
        _disk_areas_in_box_batch(
            centers,
            normals,
            radii,
            config.half_size,
            config.geometry_tolerance,
        )

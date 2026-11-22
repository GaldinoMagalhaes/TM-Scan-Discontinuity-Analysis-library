"""Numba-accelerated window geometry (same tolerances and counts as the pure-Python path; no fastmath)."""
import numpy as np
import pandas as pd
from numba import njit
from scipy.spatial import cKDTree
from ._execution import run_tasks
from .geometry import dip_dipdir_to_normal


def pack_traces(analyzer, traces, sphere=False):
    trace_type = analyzer._validate_traces(traces)
    iterator = (analyzer._get_coords_iterator(traces, trace_type) if sphere
                else analyzer._iter_trace_coords(traces, trace_type))
    points, offsets, group_ids, groups = [], [0], [], {}
    for identity, coordinates in iterator:
        coordinates = np.asarray(coordinates, dtype=np.float64)
        groups.setdefault(identity, len(groups))
        group_ids.append(groups[identity])
        points.extend(coordinates)
        offsets.append(len(points))
    return (np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3),
            np.array(offsets, dtype=np.int64), np.array(group_ids, dtype=np.int64), len(groups))


@njit(cache=True, nogil=True)
def _classify(point, center, radius, eps):
    delta = point-center
    distance = np.dot(delta, delta)
    if distance <= (radius-eps)**2:
        return 1
    if distance >= (radius+eps)**2:
        return -1
    return 0


@njit(cache=True, nogil=True)
def _sphere_segment(p0, p1, center, radius, eps):
    direction, relative = p1-p0, p0-center
    length = np.linalg.norm(direction)
    roots = np.empty(2)
    count = 0
    aa = np.dot(direction, direction)
    if aa > eps**2:
        bb = 2.0*np.dot(relative, direction)
        cc = np.dot(relative, relative)-radius*radius
        discriminant = bb*bb-4*aa*cc
        if discriminant > (eps*radius*np.sqrt(aa))**2:
            square = np.sqrt(discriminant)
            for root in ((-bb-square)/(2*aa), (-bb+square)/(2*aa)):
                if -eps <= root <= 1+eps:
                    roots[count] = min(1., max(0., root))
                    count += 1
    if length == 0:
        return 0., roots, count
    first, last = _classify(p0, center, radius, eps), _classify(p1, center, radius, eps)
    lower, upper = 0., 0.
    if first == 1 and last == 1:
        lower, upper = 0., 1.
    elif first == 1 and last != 1 and count:
        lower, upper = 0., roots[count-1]
    elif first != 1 and last == 1 and count:
        lower, upper = roots[0], 1.
    elif count >= 2 and roots[1]-roots[0] > eps:
        lower, upper = roots[0], roots[1]
    lower, upper = lower+eps/length, upper-eps/length
    return ((upper-lower)*length if upper > lower else 0.), roots, count


@njit(cache=True, nogil=True)
def sphere_metrics(centers, points, offsets, groups, n_groups, radius, eps):
    output = np.zeros((len(centers), 5))
    for wi in range(len(centers)):
        center = centers[wi]
        present = np.zeros(n_groups, np.bool_)
        hits = np.empty((2*len(points), 3))
        n_hits, term_in, term_out, total_length = 0, 0, 0, 0.
        for trace in range(len(groups)):
            start, stop = offsets[trace], offsets[trace+1]
            trace_length = 0.
            for segment in range(start, stop-1):
                p0, p1 = points[segment], points[segment+1]
                length, roots, count = _sphere_segment(p0, p1, center, radius, eps)
                trace_length += length
                total_length += length
                for ri in range(count):
                    hit = p0+roots[ri]*(p1-p0)
                    unique = True
                    for previous in range(n_hits):
                        if np.linalg.norm(hit-hits[previous]) <= 5*eps:
                            unique = False
                            break
                    if unique:
                        hits[n_hits] = hit
                        n_hits += 1
            if trace_length > 0:
                present[groups[trace]] = True
                for endpoint in (start, stop-1):
                    classification = _classify(points[endpoint], center, radius, eps)
                    term_in += int(classification == 1)
                    term_out += int(classification == -1)
        output[wi] = np.array([present.sum(), n_hits, term_in, term_out, total_length])
    return output


@njit(cache=True, nogil=True)
def _clip_box(a, b, limits, eps):
    low, high = 0., 1.
    for axis in range(3):
        direction = b[axis]-a[axis]
        for p, q in ((direction, limits[axis]-a[axis]), (-direction, limits[axis]+a[axis])):
            if abs(p) < eps:
                if q < -eps:
                    return False, 0., 0.
            else:
                value = q/p
                if p < 0:
                    low = max(low, value)
                else:
                    high = min(high, value)
                if low-high > eps:
                    return False, 0., 0.
    return high-low > 0., low, high


@njit(cache=True, nogil=True)
def rectangle_metrics(frames, deltas, points, offsets, groups, n_groups, eps):
    output = np.zeros((len(frames), 5))
    for wi in range(len(frames)):
        center, u, v, normal = frames[wi, 0], frames[wi, 1], frames[wi, 2], frames[wi, 3]
        hu, hv = frames[wi, 4, 0], frames[wi, 4, 1]
        limits = np.array([hu, hv, deltas[wi]])
        present, transecting, contained = np.zeros(n_groups, np.bool_), np.zeros(n_groups, np.bool_), np.zeros(n_groups, np.bool_)
        total_real, total_proj = 0., 0.
        for ti in range(len(groups)):
            real, projected, crossings, inside = 0., 0., 0, False
            for segment in range(offsets[ti], offsets[ti+1]-1):
                aa, bb = points[segment], points[segment+1]
                relative_a, relative_b = aa-center, bb-center
                a = np.array([np.dot(relative_a, u), np.dot(relative_a, v), np.dot(relative_a, normal)])
                b = np.array([np.dot(relative_b, u), np.dot(relative_b, v), np.dot(relative_b, normal)])
                inside_a = abs(a[0]) <= hu+eps and abs(a[1]) <= hv+eps and abs(a[2]) <= deltas[wi]+eps
                inside_b = abs(b[0]) <= hu+eps and abs(b[1]) <= hv+eps and abs(b[2]) <= deltas[wi]+eps
                hit, t0, t1 = _clip_box(a, b, limits, eps)
                if not hit or t1-t0 <= eps:
                    continue
                # Projected-length column keeps the B+t1*(B-A) endpoint convention of
                # RectangleAnalyzer (reproducibility). P21 uses real lengths.
                p0, p1 = a+t0*(b-a), b+t1*(b-a)
                length_proj = np.hypot(p1[0]-p0[0], p1[1]-p0[1])
                length_real = np.linalg.norm(bb-aa)*(t1-t0)
                if length_proj > eps and length_real > eps:
                    projected += length_proj
                    real += length_real
                crossings += int(eps < t0 < 1-eps)+int(eps < t1 < 1-eps)
                inside = inside or (inside_a and inside_b)
            if real > 0:
                identity = groups[ti]
                present[identity] = True
                if crossings >= 2:
                    transecting[identity] = True
                elif crossings == 0 and inside:
                    contained[identity] = True
                total_real += real
                total_proj += projected
        output[wi] = np.array([present.sum(), transecting.sum(), contained.sum(), total_real, total_proj])
    return output


def _chunks(count, size=64):
    return [slice(start, min(start+size, count)) for start in range(0, count, size)]


def _sphere_chunk(centers, packed, radius, eps):
    return sphere_metrics(centers, *packed, radius, eps)


def _rectangle_chunk(frames, deltas, packed, eps):
    return rectangle_metrics(frames, deltas, *packed, eps)


def analyze_spheres_fast(analyzer, centers, traces, scan_df, n_jobs):
    from .sphere import SphereAnalyzer
    xyz = scan_df[["X", "Y", "Z"]].to_numpy(float)
    tree = cKDTree(xyz)
    normals = np.full((len(centers), 3), np.nan)
    # Keep original point order and exact final distance test for PCA.
    for i, center in enumerate(centers):
        search_radius = analyzer.sphere_radius*(1.+1e-12)+1e-12
        indices = np.sort(tree.query_ball_point(center, search_radius))
        selected = xyz[indices]
        selected = selected[np.linalg.norm(selected-center, axis=1) <= analyzer.sphere_radius]
        normals[i] = SphereAnalyzer.fit_plane(selected)
    normals[normals[:, 2] < 0] *= -1
    tables = {}
    for family, frame in traces.items():
        packed = pack_traces(analyzer, frame, sphere=True)
        # Warm the kernel before starting Joblib threads.
        sphere_metrics(centers[:0], *packed, analyzer.sphere_radius, analyzer.eps)
        blocks = run_tasks(_sphere_chunk,
                           [(centers[sl], packed, analyzer.sphere_radius, analyzer.eps) for sl in _chunks(len(centers))],
                           n_jobs=n_jobs)
        metrics = np.vstack(blocks) if blocks else np.empty((0, 5))
        out = pd.DataFrame({"Sphere_ID": np.arange(len(centers)),
                            "X": centers[:, 0], "Y": centers[:, 1], "Z": centers[:, 2],
                            "N_traces": metrics[:, 0].astype(int), "N_intersections": metrics[:, 1].astype(int),
                            "Terminations_inside": metrics[:, 2].astype(int), "Terminations_outside": metrics[:, 3].astype(int),
                            "P21": metrics[:, 4]/analyzer.area_circle, "P20": metrics[:, 0]/analyzer.area_circle,
                            "Nx": normals[:, 0], "Ny": normals[:, 1], "Nz": normals[:, 2]})
        out["μ"] = (np.pi*analyzer.sphere_radius/2)*(out.N_intersections/out.Terminations_inside.replace(0, np.nan))
        tables[family] = out
    return tables


def analyze_rectangles_fast(analyzer, geometry, traces, orientations, n_jobs):
    from .analysis import RECTANGLE_COLUMNS
    quads, xyz = geometry["quads_final"], geometry["scan_xyz"]
    frames = np.empty((len(quads), 5, 3))
    deltas = np.empty(len(quads))
    tree = cKDTree(xyz)
    for wi, quad in enumerate(quads):
        center, u, v, hu, hv, normal = analyzer._rect_frame_from_quad(quad)
        if normal[2] < 0:
            normal = -normal
        frames[wi] = np.array([center, u, v, normal, [hu, hv, 0.]])
        indices = tree.query_ball_point(center, min(analyzer.size_u, analyzer.size_v))
        deltas[wi] = analyzer._delta_for_window(center, normal, neigh_points=xyz[indices])
    tables = {}
    for family, traces_frame in traces.items():
        packed = pack_traces(analyzer, traces_frame)
        rectangle_metrics(frames[:0], deltas[:0], *packed, analyzer.eps)
        blocks = run_tasks(_rectangle_chunk,
                           [(frames[sl], deltas[sl], packed, analyzer.eps) for sl in _chunks(len(quads))], n_jobs=n_jobs)
        metrics = np.vstack(blocks) if blocks else np.empty((0, 5))
        mu = np.full(len(quads), np.nan)
        family_normal = dip_dipdir_to_normal(*orientations[family])
        for wi in range(len(quads)):
            direction = np.cross(frames[wi, 3], family_normal)
            norm = np.linalg.norm(direction)
            if not np.isfinite(norm) or norm < 1e-12:
                continue
            direction /= norm
            u = frames[wi, 1]
            cosine = np.clip(abs(float(np.dot(direction, u))/(norm*np.linalg.norm(u))), 0., 1.)
            sine = np.sqrt(max(0., 1.-cosine**2))
            nn, nt, nc = metrics[wi, :3]
            denominator = analyzer.size_u*sine+analyzer.size_v*cosine
            if denominator > 0 and nn-nt+nc > 0 and nn+nt-nc > 0:
                mu[wi] = ((analyzer.size_u*analyzer.size_v)/denominator)*((nn+nt-nc)/(nn-nt+nc))
        area = analyzer.size_u*analyzer.size_v
        out = pd.DataFrame({"Rectangle_ID": np.arange(len(quads)), "X": frames[:, 0, 0],
                            "Y": frames[:, 0, 1], "Z": frames[:, 0, 2],
                            "N_traces": metrics[:, 0].astype(int), "N_transecting": metrics[:, 1].astype(int),
                            "N_contained": metrics[:, 2].astype(int), "P21": metrics[:, 3]/area,
                            "P20": metrics[:, 0]/area, "Nx": frames[:, 3, 0], "Ny": frames[:, 3, 1],
                            "Nz": frames[:, 3, 2], "μ": mu, "sum_L_real": metrics[:, 3], "sum_L_proj": metrics[:, 4]})
        tables[family] = out.reindex(columns=RECTANGLE_COLUMNS)
    return tables

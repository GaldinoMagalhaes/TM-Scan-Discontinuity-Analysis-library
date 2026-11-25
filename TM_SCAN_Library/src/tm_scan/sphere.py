"""Spherical-window analyzer (P20, P21 and mean trace length per window)."""
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from sklearn.decomposition import PCA
from shapely.geometry import Polygon, Point, LineString, MultiLineString
from shapely.ops import polygonize, unary_union
from .geometry import _extract_poly_segments, _point_to_segments_dist3D

class SphereAnalyzer:
    def __init__(self, sphere_radius, paths=None, verbose=False, eps=None):
        self.sphere_radius = float(sphere_radius)
        self.area_circle = np.pi * self.sphere_radius**2
        self.paths = paths
        self.verbose = verbose
        self.eps = float(eps) if eps is not None else max(1e-9, 1e-9*self.sphere_radius)

    def _validate_traces(self, traces):
        """Validates the format of the input data for traces."""
        if isinstance(traces, gpd.GeoDataFrame) and 'geometry' in traces.columns:
            return 'geodataframe'
        elif isinstance(traces, pd.DataFrame) and {'X0', 'X1'}.issubset(traces.columns):
            return 'dataframe'
        raise ValueError("Unrecognized 'traces' format.")
    
    def _classify_point(self, p, c):
        """Classifies p as IN/ON/OUT with tolerance."""
        R, eps = self.sphere_radius, self.eps
        d2 = np.dot(p-c, p-c); Rm, Rp = (R-eps)**2, (R+eps)**2
        if d2 <= Rm:  return "IN"
        if d2 >= Rp:  return "OUT"
        return "ON"

    def _segment_sphere_roots(self, p0, p1, c):
        R, eps = self.sphere_radius, self.eps
        d, m = p1 - p0, p0 - c
        A = np.dot(d, d)
        if A <= eps**2:
            return []
        B = 2.0*np.dot(m, d)
        C = np.dot(m, m) - R*R
        disc = B*B - 4*A*C
        tang_tol = (eps * R * np.sqrt(A))**2
        if disc <= tang_tol:
            return []
        s = np.sqrt(disc)
        t1, t2 = (-B - s)/(2*A), (-B + s)/(2*A)
        ts = [t for t in (t1, t2) if -eps <= t <= 1+eps]
        ts = [min(1.0, max(0.0, t)) for t in ts if 0.0 - eps <= t <= 1.0 + eps]
        return ts

    def _clip_length_inside_segment(self, p0, p1, c):
        """Length of segment [p0,p1] strictly inside the sphere."""
        d = p1 - p0
        L = np.linalg.norm(d)
        eps = self.eps
        if L == 0: return 0.0
        s0, s1 = self._classify_point(p0, c), self._classify_point(p1, c)
        ts = sorted(self._segment_sphere_roots(p0, p1, c))
        inside_intervals = []
        if s0 == "IN" and s1 == "IN":
            inside_intervals = [(0.0, 1.0)]
        elif s0 == "IN" and s1 != "IN":
            if ts: inside_intervals = [(0.0, ts[-1])]
        elif s0 != "IN" and s1 == "IN":
            if ts: inside_intervals = [(ts[0], 1.0)]
        else:
            if len(ts) >= 2:
                t_in, t_out = ts[0], ts[1]
                if t_out - t_in > eps:
                    inside_intervals = [(t_in, t_out)]
        Lin = 0.0
        for a, b in inside_intervals:
            a2, b2 = a + eps/L, b - eps/L
            if b2 > a2:
                Lin += (b2 - a2) * L
        return Lin

    def _extract_bounds_coords(self, bounds_gdf):
        """Extracts all coordinates from a GeoDataFrame with LineString/MultiLineString geometries."""
        pts = []
        for geom in bounds_gdf.geometry:
            if isinstance(geom, LineString):
                arr = np.asarray(geom.coords, dtype=float)
                if arr.shape[1] == 2:
                    arr = np.column_stack([arr, np.zeros(len(arr))])
                pts.append(arr)
            elif isinstance(geom, MultiLineString):
                for ln in geom.geoms:
                    arr = np.asarray(ln.coords, dtype=float)
                    if arr.shape[1] == 2:
                        arr = np.column_stack([arr, np.zeros(len(arr))])
                    pts.append(arr)
        if not pts:
            return np.empty((0,3), dtype=float)
        return np.vstack(pts)

    def _plane_frame_from_bounds(self, bounds_gdf):
        """
        Fits a plane to the boundary polyline points using PCA
        and returns the origin and two basis vectors (e1, e2) for the plane.
        """
        P = self._extract_bounds_coords(bounds_gdf)
        if P.shape[0] < 3:
            origin = P.mean(axis=0) if len(P) else np.zeros(3)
            e1, e2 = np.array([1.0,0.0,0.0]), np.array([0.0,1.0,0.0])
            return origin, e1, e2
        C = P.mean(axis=0)
        X = P - C
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        e1, e2 = Vt[0], Vt[1]
        return C, e1/np.linalg.norm(e1), e2/np.linalg.norm(e2)

    def _project_to_plane_uv(self, Q, origin, e1, e2):
        """Projects 3D points Q onto a 2D plane defined by origin, e1, and e2."""
        Q = np.asarray(Q, dtype=float)
        V = Q - origin
        return np.column_stack([V @ e1, V @ e2])

    def _polygon_from_bounds_in_uv(self, bounds_gdf, origin, e1, e2):
        """
        Takes the boundary polyline, projects it onto a 2D plane, and
        attempts to form a Polygon from the projected lines.
        """
        lines_uv = []
        for geom in bounds_gdf.geometry:
            if isinstance(geom, LineString):
                arr = np.asarray(geom.coords, dtype=float)
                if arr.shape[1] == 2:
                    arr = np.column_stack([arr, np.zeros(len(arr))])
                lines_uv.append(LineString(self._project_to_plane_uv(arr, origin, e1, e2)))
            elif isinstance(geom, MultiLineString):
                for ln in geom.geoms:
                    arr = np.asarray(ln.coords, dtype=float)
                    if arr.shape[1] == 2:
                        arr = np.column_stack([arr, np.zeros(len(arr))])
                    lines_uv.append(LineString(self._project_to_plane_uv(arr, origin, e1, e2)))
        if not lines_uv:
            return None
        merged = unary_union(lines_uv)
        polys = list(polygonize(merged))
        if polys:
            return max(polys, key=lambda p: p.area)
        if hasattr(merged, "is_ring") and merged.is_ring:
            return Polygon(list(merged.coords))
        return None

    def filter_centers_border_then_inside(self, centers, bounds_gdf, print_summary=True):
        """Filters centers, removing those close to a border or outside ALL boundaries."""
        centers = np.asarray(centers, dtype=float)
        total0 = len(centers)
        if total0 == 0:
            return centers, np.zeros(0, bool), centers, np.zeros(0, bool)

        boundaries = []
        for geom in bounds_gdf.geometry:
            if geom is None:
                continue
            if geom.geom_type == "LineString":
                gdf_tmp = gpd.GeoDataFrame(geometry=[geom], crs=bounds_gdf.crs)
            elif geom.geom_type == "MultiLineString":
                gdf_tmp = gpd.GeoDataFrame(geometry=list(geom.geoms), crs=bounds_gdf.crs)
            else:
                continue

            # Extract segments for the 3D border
            A, B = _extract_poly_segments(gdf_tmp)
            # Fit an individual PCA plane
            origin, e1, e2 = self._plane_frame_from_bounds(gdf_tmp)
            # Projected 2D polygon
            poly_uv = self._polygon_from_bounds_in_uv(gdf_tmp, origin, e1, e2)
            if A.shape[0] > 0 and poly_uv is not None:
                boundaries.append({"A": A, "B": B, "origin": origin, "e1": e1, "e2": e2, "poly_uv": poly_uv})

        if len(boundaries) == 0:
            # fallback: keep everything if there are no boundaries
            mask_final = np.ones(total0, dtype=bool)
            return centers, mask_final, centers, mask_final

        keep_border = np.ones(total0, dtype=bool)
        for i, p in enumerate(centers):
            for bd in boundaries:
                dmin = _point_to_segments_dist3D(p, bd["A"], bd["B"]).min()
                if dmin <= self.sphere_radius:
                    keep_border[i] = False
                    break  # already intersects a border

        centers_nb = centers[keep_border]
        if print_summary:
            print(f"Total number of centers on the grid: {total0}")
            print(f"After removing boundary intersections: {len(centers_nb)}")

        if len(centers_nb) == 0:
            mask_final = np.zeros(total0, dtype=bool)
            return centers_nb, mask_final, centers_nb, keep_border

        keep_inside_any = np.zeros(len(centers_nb), dtype=bool)
        for bd in boundaries:
            uv = self._project_to_plane_uv(centers_nb, bd["origin"], bd["e1"], bd["e2"])
            keep_inside_any |= np.fromiter(
                (bd["poly_uv"].contains(Point(u, v)) for u, v in uv),
                count=len(uv),
                dtype=bool
            )

        mask_final = np.zeros(total0, dtype=bool)
        idx_nb = np.flatnonzero(keep_border)
        mask_final[idx_nb[keep_inside_any]] = True

        centers_final = centers[mask_final]
        if print_summary:
            print(f"Total number of centers considered for analysis: {len(centers_final)}")
        return centers_final, mask_final, centers_nb, keep_border

    def _get_coords_iterator(self, traces, trace_type):
        """Iterates coordinates for traces."""
        if trace_type == 'geodataframe':
            for idx, geom in zip(traces.index, traces.geometry):
                if isinstance(geom, LineString):
                    yield idx, np.asarray(geom.coords, dtype=float)[[0, -1]]
                elif isinstance(geom, MultiLineString):
                    for ln in geom.geoms:
                        yield idx, np.asarray(ln.coords, dtype=float)[[0, -1]]
            return
        for idx, row in traces.iterrows():
            yield idx, np.asarray([row['X0'], row['X1']], dtype=float)

    def count_intersections(self, center, traces):
        """Returns: (N_traces, N_intersections, Terminations_inside, Terminations_outside)."""
        C, trace_type = np.asarray(center, dtype=float), self._validate_traces(traces)
        def coords_by_trace():
            if trace_type == 'geodataframe':
                for idx, geom in zip(traces.index, traces.geometry):
                    if isinstance(geom, LineString):
                        yield idx, np.asarray(geom.coords, dtype=float)[[0, -1]]
                    elif isinstance(geom, MultiLineString):
                        for ln in geom.geoms:
                            yield idx, np.asarray(ln.coords, dtype=float)[[0, -1]]
            else:
                for idx, row in traces.iterrows():
                    yield idx, np.vstack([row.X0, row.X1]).astype(float)
        
        trace_has_length, term_in, term_out, hit_points = {}, 0, 0, []
        for idx, coords in coords_by_trace():
            if coords.shape[1] == 2:
                coords = np.column_stack([coords, np.zeros(len(coords))])
            Lin, local_hits = 0.0, []
            for i in range(len(coords)-1):
                p0, p1 = coords[i], coords[i+1]
                Lin += self._clip_length_inside_segment(p0, p1, C)
                for t in self._segment_sphere_roots(p0, p1, C):
                    x = p0 + t*(p1 - p0); local_hits.append(x)
            
            trace_has_length[idx] = trace_has_length.get(idx, False) or (Lin > 0.0)
            if Lin > 0.0:
                s0, sm = self._classify_point(coords[0], C), self._classify_point(coords[-1], C)
                if s0 == "IN":  term_in  += 1
                if sm == "IN":  term_in  += 1
                if s0 == "OUT": term_out += 1
                if sm == "OUT": term_out += 1
            hit_points.extend(local_hits)
            
        n_traces = sum(1 for v in trace_has_length.values() if v)
        def dedupe_points(points, tol):
            if not points: return 0
            kept = [points[0]]
            for p in points[1:]:
                if min(np.linalg.norm(p - q) for q in kept) > tol:
                    kept.append(p)
            return len(kept)
            
        n_intersections = dedupe_points(hit_points, tol=5*self.eps)
        return n_traces, n_intersections, term_in, term_out

    def compute_length_inside_sphere(self, center, traces):
        C, trace_type = np.asarray(center, dtype=float), self._validate_traces(traces)
        def iter_traces_coords():
            if trace_type == 'geodataframe':
                for geom in traces.geometry:
                    if isinstance(geom, LineString): yield np.asarray(geom.coords, dtype=float)[[0, -1]]
                    elif isinstance(geom, MultiLineString):
                        for ln in geom.geoms: yield np.asarray(ln.coords, dtype=float)[[0, -1]]
            else:
                for _, row in traces.iterrows(): yield np.vstack([row.X0, row.X1]).astype(float)
        
        total = 0.0
        for coords in iter_traces_coords():
            if coords.shape[1] == 2:
                coords = np.column_stack([coords, np.zeros(len(coords))])
            for i in range(len(coords)-1):
                p0, p1 = coords[i], coords[i+1]
                total += self._clip_length_inside_segment(p0, p1, C)
        return total

    @staticmethod
    def fit_plane(points):
        """Fits a plane to 3D points using PCA."""
        if points.shape[0] < 3:
            return np.array([np.nan, np.nan, np.nan])
        pca = PCA(n_components=2)
        pca.fit(points)
        normal = np.cross(pca.components_[0], pca.components_[1])
        return normal / np.linalg.norm(normal)

    def analyze_spheres(self, sphere_centers, traces, scan_df, input_filename):
        """Performs the full analysis for all spheres."""
        if isinstance(traces, dict):
            family_results = {}
            for family_name, family_traces in traces.items():
                if self.verbose:
                    print(f"[info] Analyzing family: {family_name}")
                family_filename = f"{family_name}_{Path(input_filename).name}"
                family_results[family_name] = self.analyze_spheres(
                    sphere_centers, family_traces, scan_df, input_filename=family_filename)
            return family_results
        
        results = []
        for i, center in enumerate(sphere_centers):
            n_traces, n_intersections, term_inside, term_outside = self.count_intersections(center, traces)
            length_inside = self.compute_length_inside_sphere(center, traces)
            P21 = length_inside / self.area_circle if self.area_circle > 0 else 0
            distances = np.linalg.norm(scan_df[['X', 'Y', 'Z']].values - np.array(center), axis=1)
            inside_sphere = scan_df[distances <= self.sphere_radius]
            normal = self.fit_plane(inside_sphere[['X', 'Y', 'Z']].values)
            results.append([i, center[0], center[1], center[2], n_traces, n_intersections, term_inside, term_outside, P21, n_traces / self.area_circle, *normal])
        
        columns = ['Sphere_ID', 'X', 'Y', 'Z', 'N_traces', 'N_intersections', 'Terminations_inside', 'Terminations_outside', 'P21', 'P20', 'Nx', 'Ny', 'Nz']
        output_df = pd.DataFrame(results, columns=columns)
        mask = output_df['Nz'] < 0
        output_df.loc[mask, ['Nx', 'Ny', 'Nz']] *= -1
        
        sphere_radius = self.sphere_radius
        output_df['μ'] = ((np.pi * sphere_radius) / 2) * (output_df['N_intersections'] / output_df['Terminations_inside'].replace(0, np.nan))
        
        if self.paths is not None:
            output_filename = f"output_{Path(input_filename).stem}.txt"
            output_file_path = self.paths['output_db'] / output_filename
            output_df.to_csv(output_file_path, sep=' ', index=False)
            if self.verbose:
                print(f"[info] File export concluded: {output_file_path}")
        return output_df



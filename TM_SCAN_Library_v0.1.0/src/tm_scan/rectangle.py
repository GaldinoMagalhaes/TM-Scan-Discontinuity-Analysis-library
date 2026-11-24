"""Rectangular-window analyzer (P20, P21 and mean trace length per window)."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, Point, LineString, MultiLineString
from shapely.ops import polygonize, unary_union
from .sampling import _voxel_centers_from_scan
from .geometry import (_extract_poly_segments,
                       normal_to_dip_dipdir, dip_dipdir_to_normal)

class RectangleAnalyzer:
    """
    Complete class for analysis with rectangular windows.
    """

    def __init__(self,
                 paths,
                 size_u,
                 size_v,
                 inplane_rotation_cw,
                 ref_dir,
                 plane_tol,
                 rect_buffer,
                 proj_band,
                 inner_strip_2d,
                 min_neighbors,
                 up,
                 smooth_k,
                 eps,
                 delta_mode="max",
                 verbose=False):
        self.paths = paths
        self.size_u = float(size_u)
        self.size_v = float(size_v)
        self.inplane_rotation_cw = float(inplane_rotation_cw)
        self.ref_dir = np.asarray(ref_dir, float)
        self.plane_tol = float(plane_tol)
        self.rect_buffer = float(rect_buffer)
        self.proj_band = float(proj_band)
        self.inner_strip_2d = float(inner_strip_2d)
        self.min_neighbors = int(min_neighbors)
        self.up = np.asarray(up, float) / (np.linalg.norm(up) + 1e-12)
        if delta_mode not in ("p95", "p99", "max"):
            raise ValueError("delta_mode must be 'p95', 'p99' or 'max'.")
        self.delta_mode = delta_mode          # slab half-thickness used to clip the traces
        self.smooth_k = float(smooth_k)
        self.eps = float(eps)
        self.verbose = bool(verbose)

    @staticmethod
    def _to_xyz(a, colnames=('X', 'Y', 'Z')):
        if isinstance(a, pd.DataFrame):
            arr = a.loc[:, list(colnames)].to_numpy(dtype=float) if all(c in a.columns for c in colnames) else a.iloc[:, :3].to_numpy(dtype=float)
        else:
            arr = np.asarray(a, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError("Input needs to have at least 3 columns.")
        return arr[:, :3]

    @staticmethod
    def _pca_normal(points):
        X = np.asarray(points, float)
        Xc = X - X.mean(axis=0, keepdims=True)
        pca = PCA(n_components=3).fit(Xc)
        n = pca.components_[-1]
        return n / (np.linalg.norm(n) + 1e-12), pca.explained_variance_

    @staticmethod
    def _project_unit_onto_plane(vec, n, eps=1e-12):
        v = np.asarray(vec, float)
        n = np.asarray(n, float) / (np.linalg.norm(n) + eps)
        v_proj = v - np.dot(v, n) * n
        nv = np.linalg.norm(v_proj)
        return v_proj / nv if nv >= eps else None

    def _frame_from_normal_and_reference(self, n, ref_dir=None):
        n = np.asarray(n, float) / (np.linalg.norm(n) + 1e-12)
        if ref_dir is not None:
            u = self._project_unit_onto_plane(ref_dir, n)
            if u is None:
                helper = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
                u = np.cross(n, helper); u /= (np.linalg.norm(u) + 1e-12)
        else:
            helper = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
            u = np.cross(n, helper); u /= (np.linalg.norm(u) + 1e-12)
        v = np.cross(n, u); v /= (np.linalg.norm(v) + 1e-12)
        return u, v

    def _rotate_uv_inplane_clockwise(self, u, v, angle_cw_deg):
        theta = -np.deg2rad(float(angle_cw_deg))
        c, s = np.cos(theta), np.sin(theta)
        u2 = c * u + s * v
        v2 = -s * u + c * v
        u2 /= (np.linalg.norm(u2) + 1e-12)
        v2 -= np.dot(v2, u2) * u2
        v2 /= (np.linalg.norm(v2) + 1e-12)
        return u2, v2

    @staticmethod
    def _window_corners_from_uv(center, u, v, size_u, size_v):
        c = np.asarray(center, float)
        hu, hv = size_u / 2.0, size_v / 2.0
        return np.array([
            c + (-hu) * u + (-hv) * v, c + (hu) * u + (-hv) * v,
            c + (hu) * u + (hv) * v, c + (-hu) * u + (hv) * v,
        ], float)

    def _ensure_centers(self, scan_df, centers=None):
        if centers is not None:
            return self._to_xyz(centers), np.nan
        return _voxel_centers_from_scan(scan_df, self.smooth_k)

    def _make_oriented_windows_from_cloud(self, scan_points, centers):
        tree = cKDTree(scan_points)
        radius = float(min(self.size_u, self.size_v))
        quads, diag = [], {}
        for i, c in enumerate(centers):
            idx = tree.query_ball_point(c, r=radius)
            neigh = scan_points[idx] if len(idx) > 0 else np.empty((0, 3))
            info = {"ok": False, "n_neighbors": len(neigh), "evals": None}
            if len(neigh) < self.min_neighbors:
                quads.append(None); diag[i] = info; continue
            n, evals = self._pca_normal(neigh)
            if np.dot(n, self.up) < 0: n = -n
            u, v = self._frame_from_normal_and_reference(n, ref_dir=self.ref_dir)
            if self.inplane_rotation_cw and abs(self.inplane_rotation_cw) > 1e-12:
                u, v = self._rotate_uv_inplane_clockwise(u, v, self.inplane_rotation_cw)
            quad = self._window_corners_from_uv(c, u, v, self.size_u, self.size_v)
            quads.append(quad); info.update({"ok": True, "evals": evals}); diag[i] = info
        return quads, diag

    @staticmethod
    def _bounds_coords(bounds_gdf):
        A, B = _extract_poly_segments(bounds_gdf)
        if A.size == 0: return np.empty((0, 3), float)
        return np.vstack([A, B[-1:]])

    @staticmethod
    def _fit_plane_frame_from_bounds(bounds_gdf):
        P = RectangleAnalyzer._bounds_coords(bounds_gdf)
        if P.shape[0] < 3:
            C = P.mean(axis=0) if P.size else np.zeros(3)
            e1, e2, n = np.array([1.0,0,0]), np.array([0,1.0,0]), np.array([0,0,1.0])
            return C, e1, e2, n
        C = P.mean(axis=0); X = P - C
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        e1, e2 = Vt[0], Vt[1]
        e1 /= (np.linalg.norm(e1) + 1e-12); e2 /= (np.linalg.norm(e2) + 1e-12)
        n = np.cross(e1, e2); n /= (np.linalg.norm(n) + 1e-12)
        return C, e1, e2, n

    @staticmethod
    def _project_to_uv(Q, C, e1, e2):
        Q = np.asarray(Q, float); V = Q - C
        return np.column_stack([V @ e1, V @ e2])

    @staticmethod
    def _polygon_from_bounds_uv(bounds_gdf, C, e1, e2):
        lines_uv = []
        for geom in bounds_gdf.geometry:
            if geom is None: continue
            if geom.geom_type == "LineString":
                arr = np.asarray(geom.coords, float)
                if arr.shape[1] == 2: arr = np.column_stack([arr, np.zeros(len(arr))])
                lines_uv.append(LineString(RectangleAnalyzer._project_to_uv(arr, C, e1, e2)))
            elif geom.geom_type == "MultiLineString":
                for ln in geom.geoms:
                    arr = np.asarray(ln.coords, float)
                    if arr.shape[1] == 2: arr = np.column_stack([arr, np.zeros(len(arr))])
                    lines_uv.append(LineString(RectangleAnalyzer._project_to_uv(arr, C, e1, e2)))
        if not lines_uv: return None
        merged = unary_union(lines_uv)
        polys = list(polygonize(merged))
        if polys: return max(polys, key=lambda p: p.area)
        if hasattr(merged, "is_ring") and merged.is_ring:
            return Polygon(list(merged.coords))
        return None

    @staticmethod
    def _seg_intersects_aabb2d(p0, p1, hu, hv):
        u0, v0 = p0; u1, v1 = p1
        if (max(u0, u1) < -hu) or (min(u0, u1) > hu) or (max(v0, v1) < -hv) or (min(v0, v1) > hv):
            return False
        if (-hu <= u0 <= hu and -hv <= v0 <= hv) or (-hu <= u1 <= hu and -hv <= v1 <= hv):
            return True
        du, dv = (u1 - u0), (v1 - v0)
        if abs(du) > 1e-15:
            for uc in (-hu, hu):
                t = (uc - u0) / du
                if 0.0 <= t <= 1.0:
                    v = v0 + t * dv
                    if -hv <= v <= hv: return True
        if abs(dv) > 1e-15:
            for vc in (-hv, hv):
                t = (vc - v0) / dv
                if 0.0 <= t <= 1.0:
                    u = u0 + t * du
                    if -hu <= u <= hu: return True
        return False

    @staticmethod
    def _rect_frame_from_quad(q):
        q = np.asarray(q, float)
        c = q.mean(axis=0)
        u_vec, v_vec = q[1] - q[0], q[3] - q[0]
        hu, hv = np.linalg.norm(u_vec) / 2.0, np.linalg.norm(v_vec) / 2.0
        u, v = u_vec / (np.linalg.norm(u_vec) + 1e-12), v_vec / (np.linalg.norm(v_vec) + 1e-12)
        n = np.cross(u, v); n /= (np.linalg.norm(n) + 1e-12)
        return c, u, v, hu, hv, n
    
    @staticmethod
    def _uvd(P, c, u, v, n):
        P, c = np.asarray(P, float), np.asarray(c, float)
        W = P - c
        return float(W @ u), float(W @ v), float(W @ n)
    
    @staticmethod
    def _inside_interval(x: float, a: float, eps: float) -> bool:
        return (-a - eps) <= x <= (a + eps)

    @classmethod
    def _inside_rect(cls, U: float, V: float, hu: float, hv: float, eps: float) -> bool:
        return cls._inside_interval(U, hu, eps) and cls._inside_interval(V, hv, eps)

    def _segment_intersects_rect3d(self, A, B, q):
        c, u, v, hu, hv, n = self._rect_frame_from_quad(q)
        A, B = np.asarray(A, float), np.asarray(B, float)
        n /= (np.linalg.norm(n) + 1e-12)
        dA, dB = float(np.dot(n, A - c)), float(np.dot(n, B - c))
        coplanar = (abs(dA) <= self.plane_tol and abs(dB) <= self.plane_tol)
        crosses = (dA * dB <= 0.0) or (min(abs(dA), abs(dB)) <= self.plane_tol)
        if coplanar:
            Au, Av = np.dot(A - c, u), np.dot(A - c, v)
            Bu, Bv = np.dot(B - c, u), np.dot(B - c, v)
            return self._seg_intersects_aabb2d((Au, Av), (Bu, Bv), hu + self.rect_buffer, hv + self.rect_buffer)
        if crosses:
            denom = (dA - dB)
            P = 0.5 * (A + B) if abs(denom) < 1e-15 else A + (dA / denom) * (B - A)
            if abs(denom) >= 1e-15:
                t = dA / denom
                if not (0.0 <= t <= 1.0): return False
            Pu, Pv = np.dot(P - c, u), np.dot(P - c, v)
            return (abs(Pu) <= hu + self.rect_buffer) and (abs(Pv) <= hv + self.rect_buffer)
        return False

    @staticmethod
    def _segment_min_plane_distance(A, B, c, n):
        A, B, n = np.asarray(A, float), np.asarray(B, float), np.asarray(n, float)
        n /= (np.linalg.norm(n) + 1e-12)
        dA, dB = float(np.dot(n, A - c)), float(np.dot(n, B - c))
        denom = (dB - dA)
        if abs(denom) < 1e-15:
            t_star = 0.0 if abs(dA) <= abs(dB) else 1.0
            return abs(dA), t_star, dA, dB
        t_star = max(0.0, min(1.0, -dA / denom))
        dmin = abs(dA + t_star * denom)
        return dmin, t_star, dA, dB

    def _filter_windows_two_stage(self, quads, bounds_gdf, scan_points, *,
                                  band_mode='delta', band_factor=1.0):
        quads = [q for q in quads if q is not None]
        A, B = _extract_poly_segments(bounds_gdf)
        if A.shape[0] == 0:
            return quads, np.ones(len(quads), bool)

        tree = cKDTree(np.asarray(scan_points, float))
        radius = float(min(self.size_u, self.size_v))
        keep = np.ones(len(quads), bool)

        for i, q in enumerate(quads):
            c, u, v, hu, hv, n = self._rect_frame_from_quad(q)
            n /= (np.linalg.norm(n) + 1e-12)
            H_u, H_v = (hu + self.rect_buffer), (hv + self.rect_buffer)

            if any(self._segment_intersects_rect3d(a, b, q) for a, b in zip(A, B)):
                keep[i] = False
                continue

            idx = tree.query_ball_point(c, r=radius)
            neigh = scan_points[idx] if len(idx) > 0 else np.empty((0, 3), float)
            delta_i = self._delta_for_window(c, n, neigh_points=neigh)
            proj_band_i = float(band_factor) * float(delta_i) if band_mode == 'delta' else float(self.proj_band)

            for a, b in zip(A, B):
                dmin, _, dA, dB = self._segment_min_plane_distance(a, b, c, n)
                if dmin > proj_band_i: continue
                
                a_p, b_p = a - dA * n, b - dB * n
                Au, Av = np.dot(a_p - c, u), np.dot(a_p - c, v)
                Bu, Bv = np.dot(b_p - c, u), np.dot(b_p - c, v)
                
                if self._seg_intersects_aabb2d((Au, Av), (Bu, Bv), H_u, H_v):
                    keep[i] = False
                    break
        
        quads_kept = [q for q, k in zip(quads, keep) if k]
        return quads_kept, keep

    def _filter_inside_inset_polygon(self, quads_in, centers_in, bounds_gdf, *, strict=False):
        # Build projected lines and polygonize -> possibly multiple polygons
        quads_in = [q for q in quads_in if q is not None]
        centers_in = np.asarray(centers_in, float)
        if len(quads_in) == 0 or centers_in.shape[0] == 0:
            return [], np.zeros(0, bool), None, None

        C, e1, e2, n = self._fit_plane_frame_from_bounds(bounds_gdf)

        lines_uv = []
        for geom in bounds_gdf.geometry:
            if geom is None:
                continue
            if geom.geom_type == "LineString":
                arr = np.asarray(geom.coords, float)
                if arr.shape[1] == 2:
                    arr = np.column_stack([arr, np.zeros(len(arr))])
                lines_uv.append(LineString(self._project_to_uv(arr, C, e1, e2)))
            elif geom.geom_type == "MultiLineString":
                for ln in geom.geoms:
                    arr = np.asarray(ln.coords, float)
                    if arr.shape[1] == 2:
                        arr = np.column_stack([arr, np.zeros(len(arr))])
                    lines_uv.append(LineString(self._project_to_uv(arr, C, e1, e2)))

        if not lines_uv:
            return quads_in, np.ones(len(quads_in), bool), None, (C, e1, e2, n)

        merged = unary_union(lines_uv)
        polys = list(polygonize(merged))
        if not polys:
            # If no polygonization, accept all
            return quads_in, np.ones(len(quads_in), bool), None, (C, e1, e2, n)

        # Optional inset on each polygon
        used_polys = []

        # Comment following line to account for numerical tolerance
        self.inner_strip_2d = 0

        if self.inner_strip_2d and self.inner_strip_2d > 0:
            inset_val = float(self.inner_strip_2d)
            for p in polys:
                buf = p.buffer(-inset_val)
                if buf and not buf.is_empty:
                    parts = getattr(buf, "geoms", [buf])
                    # Keep largest positive-area part from buffer
                    parts = [g for g in parts if isinstance(g, Polygon) and g.area > 0]
                    if parts:
                        used_polys.append(max(parts, key=lambda g: g.area))
        else:
            used_polys = polys

        if not used_polys:
            # If everything vanished after inset, reject all
            return [], np.zeros(len(quads_in), bool), None, (C, e1, e2, n)

        uv_centers = self._project_to_uv(centers_in, C, e1, e2)

        def inside_any(u, v):
            pt = Point(u, v)
            if strict:
                return any(poly.contains(pt) for poly in used_polys)
            return any(poly.covers(pt) for poly in used_polys)

        mask_inside = np.fromiter((inside_any(u, v) for u, v in uv_centers),
                                count=len(uv_centers), dtype=bool)

        quads_inside = [q for q, ok in zip(quads_in, mask_inside) if ok]

        # Return list of used polygons (kept for compatibility, caller ignores it)
        return quads_inside, mask_inside, used_polys, (C, e1, e2, n)


    def _clip_segment_box(self, Ua, Va, Da, Ub, Vb, Db, hu, hv, delta):
        du, dv, dd = (Ub - Ua), (Vb - Va), (Db - Da)
        t0, t1 = 0.0, 1.0

        def update(p, q, t0, t1):
            if abs(p) < self.eps:
                return (False, None, None) if q < -self.eps else (True, t0, t1)
            t = q / p
            if p < 0: t0 = max(t0, t)
            else: t1 = min(t1, t)
            return (False, None, None) if t0 - t1 > self.eps else (True, t0, t1)

        ok, t0, t1 = update(du, hu - Ua, t0, t1);  
        if not ok: return False, None, None, None, None
        ok, t0, t1 = update(-du, hu + Ua, t0, t1); 
        if not ok: return False, None, None, None, None
        ok, t0, t1 = update(dv, hv - Va, t0, t1);  
        if not ok: return False, None, None, None, None
        ok, t0, t1 = update(-dv, hv + Va, t0, t1); 
        if not ok: return False, None, None, None, None
        ok, t0, t1 = update(dd, delta - Da, t0, t1);  
        if not ok: return False, None, None, None, None
        ok, t0, t1 = update(-dd, delta + Da, t0, t1); 
        if not ok: return False, None, None, None, None

        if t1 - t0 <= 0.0: return False, None, None, None, None
        U0, V0, D0 = (Ua + t0 * du), (Va + t0 * dv), (Da + t0 * dd)
        U1, V1, D1 = (Ub + t1 * du), (Vb + t1 * dv), (Db + t1 * dd)
        return True, t0, t1, (U0, V0, D0), (U1, V1, D1)

    def _delta_for_window(self, c, n, neigh_points=None, mode='p95', pad=0.0, delta_min=1e-4):
        n = np.asarray(n, float); n /= (np.linalg.norm(n) + 1e-12)
        if neigh_points is None or len(neigh_points) == 0:
            return max(0.6, float(delta_min))
        
        D = (np.asarray(neigh_points, float) - np.asarray(c, float)) @ n
        absD = np.abs(D)
        
        if mode == 'p95':   core = np.percentile(absD, 95.0)
        elif mode == 'p99': core = np.percentile(absD, 99.0)
        elif mode == 'max': core = absD.max()
        else:           core = np.percentile(absD, 95.0)
            
        return max(float(core) + float(pad), float(delta_min))

    def _validate_traces(self, traces):
        if isinstance(traces, pd.DataFrame) and {'X0', 'X1'}.issubset(traces.columns):
            return 'dataframe'
        if hasattr(traces, "geometry") and 'geometry' in traces.columns:
            return 'geodataframe'
        raise ValueError("Traces must be a DataFrame or GeoDataFrame.")

    def _iter_trace_coords(self, traces, trace_type):
        if trace_type == 'geodataframe':
            for idx, geom in zip(traces.index, traces.geometry):
                if isinstance(geom, LineString):
                    coords = np.asarray(geom.coords, float)[[0, -1]]
                    if coords.shape[1] == 2: coords = np.column_stack([coords, np.zeros(len(coords))])
                    yield idx, coords
                elif isinstance(geom, MultiLineString):
                    for j, ln in enumerate(geom.geoms):
                        coords = np.asarray(ln.coords, float)[[0, -1]]
                        if coords.shape[1] == 2: coords = np.column_stack([coords, np.zeros(len(coords))])
                        yield (idx, j), coords
            return
        for idx, (x0, x1) in enumerate(traces[['X0', 'X1']].itertuples(index=False)):
            yield idx, np.vstack([np.asarray(x0, float).ravel()[:3], np.asarray(x1, float).ravel()[:3]])

    def generate_and_filter_windows(self, scan_df, bounds_gdf, centers=None):
        scan_xyz = self._to_xyz(scan_df)
        centers_xyz, voxel_size = self._ensure_centers(scan_df, centers)
        
        initial_quads, diag = self._make_oriented_windows_from_cloud(scan_xyz, centers_xyz)
        valid_initial_quads = [q for q in initial_quads if q is not None]
        
        quads_kept, keep_mask = self._filter_windows_two_stage(
            initial_quads, bounds_gdf, scan_xyz, band_mode='delta', band_factor=1.0
        )
        
        centers_kept = np.vstack([q.mean(axis=0, keepdims=True) for q in quads_kept]) if quads_kept else np.empty((0, 3), float)

        quads_final, inside_mask, _, _ = self._filter_inside_inset_polygon(
            quads_kept, centers_kept, bounds_gdf, strict=False
        )

        diag_info = {
            "initial_centers": len(centers_xyz),
            "valid_pca_windows": len(valid_initial_quads),
            "after_boundary_filter": len(quads_kept),
            "final_windows": len(quads_final)
        }
        
        if self.verbose:
            print(f"[diag] Initial grid centers: {diag_info['initial_centers']}")
            print(f"[diag] Valid windows (PCA/neighbors): {diag_info['valid_pca_windows']}")
            print(f"[diag] After boundary intersection filter: {diag_info['after_boundary_filter']}")
            print(f"[diag] Final windows (inside outcrop + inset): {diag_info['final_windows']}")

        # ✅ STEP 2: Convert window normals to dip / dip-direction
        dip_list = []
        dip_dir_list = []
        for q in quads_final:
            # Extract plane normal for each quad
            c, u, v, hu, hv, n = self._rect_frame_from_quad(q)
            if n[2] < 0: 
                n = -n  # Ensure upward convention
            dip, dip_dir = normal_to_dip_dipdir(n)
            dip_list.append(dip)
            dip_dir_list.append(dip_dir)

        # Store orientation info inside the returned dict
        orientation_info = pd.DataFrame({
            "Dip": dip_list,
            "DipDir": dip_dir_list
        })
        
        return {
            "quads_final": quads_final,
            "centers_kept": centers_kept[inside_mask] if len(centers_kept) > 0 else centers_kept,
            "scan_xyz": scan_xyz,
            "diag": diag_info,
            "voxel_size": voxel_size,
            "orientation_info": orientation_info
        }

    def _analyze_one_family(self, quads_final, traces, scan_xyz, family_name, family_orientations):
        """
        Analyze one fracture family over rectangular windows.

        Core ideas:
        - Work in each window's local (u, v, d) frame.
        - Clip every trace segment against the 3D box: [-hu, hu] x [-hv, hv] x [-delta, delta]
            using Liang–Barsky (via self._clip_segment_box).
        - Classify each *trace* (not segment) in the window:
            * transecting (NT): the clipped path has TWO interior boundary hits (two crossings)
            * contained  (NC): both endpoints inside and no boundary crossings
            * intersecting (Nint): one interior boundary hit (not needed in μ, but useful for diagnostics)
        - Sum real and projected lengths (P21) and counts (P20).
        - Compute Mauldon (1998) Eq. 24 per window:
                μ = [a*b / (a*E|sinθ| + b*E|cosθ|)] * [(N + NT - NC) / (N - NT + NC)]
            where θ is the angle between intersection-line (family plane ∩ window plane) and window u-axis.
        """

        # --- Validate traces container type ---
        trace_type = self._validate_traces(traces)

        # --- Collect valid windows ---
        Q = [q for q in quads_final if q is not None]
        nW = len(Q)
        if nW == 0:
            return pd.DataFrame()

        # --- Allocate per-window accumulators ---
        sum_L_proj = np.zeros(nW, dtype=float)   # sum of projected (u,v) lengths of clipped segments
        sum_L_real = np.zeros(nW, dtype=float)   # sum of 3D lengths of clipped segments
        n_traces   = np.zeros(nW, dtype=int)     # number of distinct traces with L_real > 0 in the window
        NT         = np.zeros(nW, dtype=int)     # number of transecting traces (2 crossings)
        NC         = np.zeros(nW, dtype=int)     # number of fully contained traces (0 crossings, both endpoints inside)
        normals    = np.zeros((nW, 3), dtype=float)
        centers    = np.zeros((nW, 3), dtype=float)

        # --- Spatial index for neighborhood-based delta estimation ---
        tree   = cKDTree(np.asarray(scan_xyz, float))
        radius = float(min(self.size_u, self.size_v))

        # --- Family plane normal from (dip, dip_dir) ---
        fam_dip, fam_dd = family_orientations[family_name]
        n_fam = dip_dipdir_to_normal(fam_dip, fam_dd)  # unit, Nz >= 0 by construction

        eps = self.eps
        a, b = float(self.size_u), float(self.size_v)  # window side lengths

        # --- Iterate windows ---
        for w_idx, q in enumerate(Q):
            # Local frame and half-sizes
            c, u, v, hu, hv, n_win = self._rect_frame_from_quad(q)
            if n_win[2] < 0:  # ensure upward convention for the window normal
                n_win = -n_win
            centers[w_idx] = c
            normals[w_idx] = n_win

            # Neighborhood-based thickness (delta) for clipping in the local d-direction
            idx = tree.query_ball_point(c, r=radius)
            neigh = scan_xyz[idx] if len(idx) > 0 else np.empty((0, 3), float)
            delta_i = self._delta_for_window(c, n_win, neigh_points=neigh, mode=self.delta_mode)

            # Per-window sets to ensure each trace is counted only once
            traces_with_length = set()
            transecting_traces = set()
            contained_traces   = set()

            # Iterate all traces (lines or multilines)
            for trace_id, coords in self._iter_trace_coords(traces, trace_type):
                # Ensure 3D array
                if coords.shape[1] == 2:
                    coords = np.column_stack([coords, np.zeros(len(coords))])

                # Accumulators per *trace* within this window
                L_real_trace = 0.0
                L_proj_trace = 0.0

                # To classify a *trace* (not segment), we will count crossings across all its segments
                # and also check if both endpoints of the WHOLE trace lie inside the box.
                # Because traces may be polylines in general, we derive classification from clipping results.
                total_crossings = 0       # sum of interior boundary hits across segments
                has_inside_endpoints = False  # true if *both* endpoints of the entire trace are inside the box

                # If traces are strictly simple lines (2 points), this loop runs once; for polylines, multiple times.
                for i in range(len(coords) - 1):
                    A = coords[i]
                    B = coords[i + 1]

                    # Endpoints in local coords
                    Ua, Va, Da = self._uvd(A, c, u, v, n_win)
                    Ub, Vb, Db = self._uvd(B, c, u, v, n_win)

                    # Endpoint "inside" flags for this segment
                    inA = (abs(Ua) <= hu + eps) and (abs(Va) <= hv + eps) and (abs(Da) <= delta_i + eps)
                    inB = (abs(Ub) <= hu + eps) and (abs(Vb) <= hv + eps) and (abs(Db) <= delta_i + eps)

                    # Clip segment to the 3D box using Liang–Barsky variant implemented in _clip_segment_box
                    hit, t0, t1, P0, P1 = self._clip_segment_box(Ua, Va, Da, Ub, Vb, Db, hu, hv, delta_i)
                    if not hit or (t1 - t0) <= eps:
                        # No effective overlap with the box (or tangential contact) → ignore this segment
                        continue

                    # Clipped endpoints in local coords (U,V,D) for length computations
                    (U0, V0, D0), (U1, V1, D1) = P0, P1

                    # Projected 2D length (in the window plane)
                    L_proj = float(np.hypot(U1 - U0, V1 - V0))
                    # Real 3D length fraction of the original segment (scaled by t1 - t0)
                    seg_len_3d = float(np.linalg.norm(B - A)) * float(t1 - t0)

                    # Accumulate only if the clipped portion is non-trivial
                    if L_proj > eps and seg_len_3d > eps:
                        L_proj_trace += L_proj
                        L_real_trace += seg_len_3d

                    # Count interior boundary crossings for this *segment*:
                    # Liang–Barsky returns t0/t1 within [0,1] after clipping.
                    # If t0 (or t1) lies strictly inside (0,1), it corresponds to a crossing of the box boundary.
                    cross_this_seg = 0
                    if (t0 > eps) and (t0 < 1.0 - eps):
                        cross_this_seg += 1
                    if (t1 > eps) and (t1 < 1.0 - eps):
                        cross_this_seg += 1
                    total_crossings += cross_this_seg

                    # If the segment's endpoints are inside, keep that info to help classify "contained"
                    if inA and inB:
                        # Segment endpoints are inside for this piece; for a strictly 2-point trace,
                        # this is equivalent to the whole trace endpoints being inside.
                        # For polylines, we mark that at least one segment has both inside;
                        # final "contained" decision will still be based on global crossings==0 and having inside endpoints.
                        has_inside_endpoints = True

                # After processing all segments of this trace in this window, classify the trace:
                if L_real_trace > 0.0:
                    traces_with_length.add(trace_id)

                    if total_crossings >= 2:
                        transecting_traces.add(trace_id)
                    elif total_crossings == 0 and has_inside_endpoints:
                        contained_traces.add(trace_id)
                    # total_crossings == 1 → "intersecting" (we don't need to store it explicitly for Eq. 24)

                    # Accumulate lengths to the window
                    sum_L_real[w_idx] += L_real_trace
                    sum_L_proj[w_idx] += L_proj_trace

            # Finalize window-level counts
            n_traces[w_idx] = len(traces_with_length)
            NT[w_idx]       = len(transecting_traces)
            NC[w_idx]       = len(contained_traces)

        # --- Intensities over each rectangle ---
        area_rect = a * b
        P21 = sum_L_real / area_rect if area_rect > 0 else np.nan
        P20 = n_traces   / area_rect if area_rect > 0 else np.nan

        # --- Mauldon Eq. 24 per window ---
        mu_values = np.full(nW, np.nan, dtype=float)
        for w_idx in range(nW):
            n_win = normals[w_idx]
            # Intersection line direction between family plane and window plane
            l_vec = np.cross(n_win, n_fam)
            nl = np.linalg.norm(l_vec)
            if not np.isfinite(nl) or nl < 1e-12:
                # Parallel planes (no defined intersection direction) → μ undefined
                continue
            l_vec /= nl

            # Angle θ between intersection-line and window's u-axis
            c, u_ref, _, _, _, _ = self._rect_frame_from_quad(Q[w_idx])
            cu = np.linalg.norm(u_ref)
            if not np.isfinite(cu) or cu < 1e-12:
                continue
            cos_theta = abs(float(np.dot(l_vec, u_ref)) / (nl * cu))  # both normalized, but safe
            cos_theta = np.clip(cos_theta, 0.0, 1.0)
            sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta**2))

            # Expected absolute values for a single family are just |sinθ| and |cosθ|
            E_sin, E_cos = sin_theta, cos_theta

            N  = int(n_traces[w_idx])
            NTi = int(NT[w_idx])
            NCi = int(NC[w_idx])

            # Geometric factor
            denom_geom = a * E_sin + b * E_cos
            if denom_geom <= 0:
                mu_values[w_idx] = np.nan
                continue
            geom_term = (a * b) / denom_geom

            # Endpoint term (Mauldon Eq. 24)
            num_end = (N + NTi - NCi)
            den_end = (N - NTi + NCi)
            if den_end <= 0 or num_end <= 0:
                mu_values[w_idx] = np.nan
                continue

            mu_values[w_idx] = geom_term * (num_end / den_end)

        # --- Build output DataFrame ---
        out = pd.DataFrame({
            'Rectangle_ID'   : np.arange(nW, dtype=int),
            'X'              : centers[:, 0],
            'Y'              : centers[:, 1],
            'Z'              : centers[:, 2],
            'N_traces'       : n_traces.astype(int),
            'N_transecting'  : NT.astype(int),
            'N_contained'    : NC.astype(int),
            'P21'            : P21,
            'P20'            : P20,
            'Nx'             : normals[:, 0],
            'Ny'             : normals[:, 1],
            'Nz'             : normals[:, 2],
            'μ'              : mu_values,
            'sum_L_real'     : sum_L_real,
            'sum_L_proj'     : sum_L_proj,
        })

        return out

    def analyze_rectangles(self,
                        scan_df,
                        bounds_gdf,
                        traces,
                        input_filename,
                        family_orientations,
                        centers=None):
        """
        Runs the rectangular window analysis:
        - Generates and filters windows with generate_and_filter_windows(...)
        - Computes metrics for each family and exports one file per family
        - Returns (None, window_data) for multiple families
            or (output_df, window_data) for a single dataset
        """

        if self.verbose:
            print(f"[info] Starting rectangular window analysis for: {input_filename}")

        # 1) Window generation and filtering
        window_data = self.generate_and_filter_windows(scan_df, bounds_gdf, centers=centers)
        quads_final = window_data["quads_final"]
        scan_xyz    = window_data["scan_xyz"]

        # 2) Normalize family_orientations -> {fam: (dip, dip_dir)}
        ori_map = {}
        if isinstance(family_orientations, dict):
            for k, v in family_orientations.items():
                if isinstance(v, dict):
                    ori_map[k] = (v.get("dip"), v.get("dip_dir"))
                elif isinstance(v, (list, tuple)) and len(v) >= 2:
                    ori_map[k] = (v[0], v[1])
                else:
                    raise ValueError(f"Invalid family_orientations['{k}']: {v}")
        else:
            raise ValueError("family_orientations must be a dict.")

        # 3) Multiple families
        if isinstance(traces, dict):
            for family_name, fam_traces in traces.items():
                if self.verbose:
                    print(f"[info] Analyzing family: {family_name}")
                if family_name not in ori_map:
                    raise ValueError(f"Missing mean orientation for family '{family_name}'.")

                out_df = self._analyze_one_family(
                    quads_final=quads_final,
                    traces=fam_traces,
                    scan_xyz=scan_xyz,
                    family_name=family_name,
                    family_orientations=ori_map
                )

                # Export per family
                family_filename = f"{family_name}_{Path(input_filename).stem}"
                out_path = self.paths['output_db'] / f"output_{family_filename}.txt"
                out_df.to_csv(out_path, sep=' ', index=False, na_rep='NaN', encoding='utf-8')
                if self.verbose:
                    print(f"[info] File export concluded: {out_path}")

            return None, window_data

        # 4) Single dataset (no dictionary)
        else:
            fam_name = "ALL"
            if fam_name not in ori_map:
                raise ValueError("For a single dataset, include {'ALL': (dip, dip_dir)} in family_orientations.")
            out_df = self._analyze_one_family(
                quads_final=quads_final,
                traces=traces,
                scan_xyz=scan_xyz,
                family_name=fam_name,
                family_orientations=ori_map
            )
            out_path = self.paths['output_db'] / f"output_{Path(input_filename).stem}.txt"
            out_df.to_csv(out_path, sep=' ', index=False, na_rep='NaN', encoding='utf-8')
            if self.verbose:
                print(f"[info] File export concluded: {out_path}")
            return out_df, window_data



"""Explicit inputs and inspectable outputs for library workflows."""
from dataclasses import dataclass, field
from typing import Optional
import math
import numpy as np
import pandas as pd
import geopandas as gpd


def positive(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite positive number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number.") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return number


@dataclass(frozen=True)
class Window:
    """Sphere radius or rectangle FULL side lengths, in coordinate units."""
    kind: str
    radius: Optional[float] = None
    size_u: Optional[float] = None
    size_v: Optional[float] = None
    rotation_cw: float = 0.0

    def __post_init__(self):
        if self.kind == "sphere":
            object.__setattr__(self, "radius", positive(self.radius, "radius"))
            if self.size_u is not None or self.size_v is not None:
                raise ValueError("A sphere takes radius only.")
        elif self.kind == "rectangle":
            object.__setattr__(self, "size_u", positive(self.size_u, "size_u"))
            object.__setattr__(self, "size_v", positive(self.size_v, "size_v"))
            if self.radius is not None:
                raise ValueError("A rectangle takes full side lengths only.")
        else:
            raise ValueError("kind must be 'sphere' or 'rectangle'.")
        rotation = float(self.rotation_cw)
        if not math.isfinite(rotation):
            raise ValueError("rotation_cw must be finite.")
        if self.kind == "sphere" and rotation != 0:
            raise ValueError("rotation_cw applies to rectangles only.")
        object.__setattr__(self, "rotation_cw", rotation)

    @classmethod
    def sphere(cls, radius):
        return cls("sphere", radius=radius)

    @classmethod
    def rectangle(cls, size_u, size_v, rotation_cw=0.0):
        return cls("rectangle", size_u=size_u, size_v=size_v, rotation_cw=rotation_cw)


@dataclass(frozen=True)
class Sampling:
    """k_smooth scales the voxel edge. It does not smooth the point cloud."""
    k_smooth: float = 2.0
    use_all_points: bool = False

    def __post_init__(self):
        object.__setattr__(self, "k_smooth", positive(self.k_smooth, "k_smooth"))
        if not isinstance(self.use_all_points, (bool, np.bool_)):
            raise ValueError("use_all_points must be boolean.")


@dataclass
class Dataset:
    """In-memory inputs; coordinates must share a Cartesian frame and unit.

    Trace DataFrames contain X0/X1 3D endpoints. Trace GeoDataFrames retain
    original polyline identity, but analysis uses only the first and last
    points of each LineString. No automatic coordinate reprojection occurs.
    """
    scan: pd.DataFrame
    traces: dict
    boundary: gpd.GeoDataFrame
    name: str = "dataset"
    length_unit: str = "m"
    source: Optional[str] = None

    def rename_families(self, mapping):
        """Rename trace families in place, e.g. {"j1_nodist": "F1"}; returns the dataset.

        Families not in `mapping` keep their names. Run before any analysis."""
        unknown = set(mapping) - set(self.traces)
        if unknown:
            raise ValueError(f"Unknown families {sorted(unknown)}; available: {list(self.traces)}.")
        renamed = {mapping.get(name, name): frame for name, frame in self.traces.items()}
        if len(renamed) != len(self.traces):
            raise ValueError("Renaming would merge two families.")
        self.traces = renamed
        return self

    def __post_init__(self):
        self.validate()

    def validate(self):
        if not isinstance(self.scan, pd.DataFrame) or not {"X", "Y", "Z"} <= set(self.scan):
            raise ValueError("scan must be a DataFrame with X, Y, Z columns.")
        if any(not pd.api.types.is_numeric_dtype(self.scan[column]) for column in ("X", "Y", "Z")):
            raise ValueError("scan coordinates must have numeric dtypes.")
        xyz = self.scan[["X", "Y", "Z"]].to_numpy(dtype=float)
        if len(xyz) < 3 or not np.isfinite(xyz).all():
            raise ValueError("scan needs at least three finite 3D points.")
        if not isinstance(self.traces, dict) or not self.traces:
            raise ValueError("traces must be a nonempty family-to-table dictionary.")
        for family, frame in self.traces.items():
            if not isinstance(family, str) or not family:
                raise ValueError("Family names must be nonempty strings.")
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                raise ValueError(f"Family {family!r} has no traces.")
            if not frame.index.is_unique:
                raise ValueError(f"Family {family!r} needs unique trace indices.")
            if isinstance(frame, gpd.GeoDataFrame):
                if frame.geometry.isna().any() or frame.geometry.is_empty.any():
                    raise ValueError(f"Family {family!r} contains empty geometries.")
                if not frame.geom_type.isin(["LineString", "MultiLineString"]).all():
                    raise ValueError("Trace geometries must be lines.")
                for geometry in frame.geometry:
                    lines = geometry.geoms if geometry.geom_type == "MultiLineString" else [geometry]
                    for line in lines:
                        coordinates = np.asarray(line.coords, float)
                        if coordinates.shape[1] != 3 or not np.isfinite(coordinates).all():
                            raise ValueError("Trace geometries must have finite XYZ coordinates.")
            else:
                if not {"X0", "X1"} <= set(frame):
                    raise ValueError(f"Family {family!r} needs X0 and X1 columns.")
                for column in ("X0", "X1"):
                    points = np.asarray(frame[column].tolist(), dtype=float)
                    if points.shape != (len(frame), 3) or not np.isfinite(points).all():
                        raise ValueError(f"{family}.{column} must contain finite XYZ endpoints.")
        if not isinstance(self.boundary, gpd.GeoDataFrame) or self.boundary.empty:
            raise ValueError("boundary must be a nonempty GeoDataFrame of boundary lines.")
        for geometry in self.boundary.geometry:
            if geometry is None or geometry.is_empty:
                raise ValueError("Boundary contains empty geometries.")
            if geometry.geom_type not in ("LineString", "MultiLineString"):
                raise ValueError("Boundary geometries must be closed lines, not polygons.")
            lines = list(geometry.geoms) if geometry.geom_type == "MultiLineString" else [geometry]
            for line in lines:
                coordinates = np.asarray(line.coords, float)
                if coordinates.shape[1] != 3 or not np.isfinite(coordinates).all():
                    raise ValueError("Boundary lines must have finite XYZ coordinates.")
            # Preserve the original multi-line representation while rejecting
            # unclosed linework that would silently disable containment filters.
            from shapely.geometry import LineString
            from shapely.ops import polygonize, unary_union
            points = np.vstack([np.asarray(line.coords, float) for line in lines])
            origin = points.mean(axis=0)
            _, _, basis = np.linalg.svd(points - origin, full_matrices=False)
            projected = [LineString((np.asarray(line.coords, float) - origin) @ basis[:2].T)
                         for line in lines]
            if not list(polygonize(unary_union(projected))):
                raise ValueError("Each boundary feature must form a closed polygon.")
        if not isinstance(self.length_unit, str) or not self.length_unit.strip():
            raise ValueError("length_unit must name the coordinate unit.")
        return self


METRICS = {"mu": "μ", "P20": "P20", "P21": "P21"}


@dataclass
class AnalysisResult:
    dataset_name: str
    length_unit: str
    window: Window
    sampling: Sampling
    tables: dict
    centers: np.ndarray
    candidate_centers: np.ndarray
    voxel_size: float
    diagnostics: dict
    quads: list = field(default_factory=list)
    roughness: object = None      # RoughnessResult, set by assess_roughness / roughness_filter
    undefined: object = None      # DataFrame: one row per NaN estimate and its cause (tm_scan.undefined)

    def table(self):
        """Stack families without changing original per-window column names."""
        frames = [frame.assign(family=family) for family, frame in self.tables.items()]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def summary(self):
        """Descriptive spatial statistics; quantiles are not confidence bounds."""
        rows = []
        for family, frame in self.tables.items():
            row = {"dataset": self.dataset_name, "family": family,
                   "window_kind": self.window.kind, "radius": self.window.radius,
                   "size_u": self.window.size_u, "size_v": self.window.size_v,
                   "rotation_cw": self.window.rotation_cw,
                   "k_smooth": self.sampling.k_smooth,
                   "use_all_points": self.sampling.use_all_points,
                   "voxel_size": self.voxel_size, "length_unit": self.length_unit,
                   **self.diagnostics}
            for name, column in METRICS.items():
                values = pd.to_numeric(frame.get(column, pd.Series(dtype=float)), errors="coerce")
                finite = values[np.isfinite(values)]
                row.update({f"{name}_n_valid": len(finite),
                            f"{name}_n_invalid": len(frame) - len(finite),
                            f"{name}_mean": finite.mean(), f"{name}_std": finite.std(ddof=1),
                            f"{name}_median": finite.median(),
                            f"{name}_q05": finite.quantile(.05),
                            f"{name}_q95": finite.quantile(.95)})
            row["n_zero_trace_windows"] = int((frame["N_traces"] == 0).sum()) if len(frame) else 0
            rows.append(row)
        return pd.DataFrame(rows)

    def undefined_summary(self):
        """Counts of undefined (NaN) estimates per family, quantity and cause."""
        from .undefined import summarize
        return summarize(self.undefined)

    def export(self, directory):
        from .io import export_result
        return export_result(self, directory)


@dataclass
class StudyResult:
    factor: str
    runs: list

    @property
    def summary(self):
        return pd.concat([run.summary().assign(case_id=i) for i, run in enumerate(self.runs)],
                         ignore_index=True)

    def table(self):
        return pd.concat([run.table().assign(case_id=i) for i, run in enumerate(self.runs)],
                         ignore_index=True)

    def export(self, directory):
        from .io import export_study
        return export_study(self, directory)

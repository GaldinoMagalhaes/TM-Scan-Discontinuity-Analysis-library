"""Load without modifying inputs; export only when explicitly requested."""
from dataclasses import asdict
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import platform
import re
import pandas as pd
import geopandas as gpd
from ._readers import get_valid_trace_files, parse_tuple_data, process_text_file
from .models import Dataset


def list_point_clouds(directory):
    root = _database_root(directory)
    files = sorted(p for p in (root / "surface_scan").rglob("*")
                   if p.is_file() and p.suffix.lower() in {".txt", ".xyz", ".csv", ".dat"})
    if not files:
        raise FileNotFoundError(f"No point cloud found in {root / 'surface_scan'}.")
    return files


def _database_root(directory):
    root = Path(directory).expanduser().resolve()
    if (root / "surface_scan").is_dir():
        return root
    matches = [root / name for name in ("01_database", "database", "Database")
               if (root / name / "surface_scan").is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError("Provide a database folder containing surface_scan, traces and slope_boundary.")
    return matches[0]


def load_dataset(directory, *, point_cloud=None, length_unit="m"):
    """Read the existing database layout. No folders are renamed or created.

    If several clouds exist, select point_cloud explicitly or iterate over
    list_point_clouds. A relative cloud path is relative to surface_scan.
    Each SHP LineString is one straight trace defined by its first and last
    XYZ points. Intermediate vertices are ignored; multipart lines are read
    one component at a time. Text rows likewise define endpoint pairs.
    """
    root = _database_root(directory)
    clouds = list_point_clouds(root)
    if point_cloud is None:
        if len(clouds) != 1:
            raise ValueError("Several point clouds found; select point_cloud explicitly.")
        cloud = clouds[0]
    else:
        cloud = Path(point_cloud).expanduser()
        if not cloud.is_absolute():
            cloud = root / "surface_scan" / cloud
        cloud = cloud.resolve()
    if not cloud.is_file():
        raise FileNotFoundError(cloud)
    scan = pd.read_csv(cloud, sep=r"\s+|,", engine="python", usecols=[0, 1, 2],
                       names=["X", "Y", "Z"], header=None)
    boundary_files = sorted((root / "slope_boundary").rglob("*.shp"))
    if not boundary_files:
        raise FileNotFoundError(f"No boundary shapefile in {root / 'slope_boundary'}.")
    boundaries = [gpd.read_file(path) for path in boundary_files]
    if any(frame.crs != boundaries[0].crs for frame in boundaries):
        raise ValueError("Boundary files have different coordinate systems.")
    boundary = gpd.GeoDataFrame(pd.concat(boundaries, ignore_index=True), crs=boundaries[0].crs)
    traces = {}
    trace_files = get_valid_trace_files(root / "traces")
    for path in trace_files:
        if path.stem in traces:
            raise ValueError(f"Duplicate family {path.stem!r}; select one trace file per family.")
        if path.suffix.lower() == ".shp":
            frame = gpd.read_file(path)
            if frame.crs != boundary.crs:
                raise ValueError(f"Trace CRS differs from boundary CRS: {path.name}.")
            rows = []
            for geometry in frame.geometry:
                if geometry is None or geometry.is_empty:
                    raise ValueError(f"Empty trace geometry in {path.name}.")
                if geometry.geom_type not in ("LineString", "MultiLineString"):
                    raise ValueError(f"Expected line traces in {path.name}.")
                lines = geometry.geoms if geometry.geom_type == "MultiLineString" else [geometry]
                for line in lines:
                    points = list(line.coords)
                    if any(len(p) != 3 for p in points):
                        raise ValueError(f"Trace shapefile must contain XYZ coordinates: {path.name}.")
                    rows.append((points[0], points[-1]))
            frame = pd.DataFrame(rows, columns=["X0", "X1"])
        else:
            frame = process_text_file(path, parse_tuple_data())
        if frame.empty:
            raise ValueError(f"No readable traces in {path.name}.")
        traces[path.stem] = frame.reset_index(drop=True)
    return Dataset(scan, traces, boundary, name=cloud.stem,
                   length_unit=length_unit, source=str(cloud))


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value).strip(".") or "family"


def export_result(result, directory):
    """Write separate family CSVs, an aggregate summary and a text run record."""
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    targets = [root / f"output_{_safe_name(family)}.csv" for family in result.tables]
    if len(set(targets)) != len(targets):
        raise ValueError("Family names produce duplicate export filenames.")
    paths = []
    for frame, path in zip(result.tables.values(), targets):
        frame.to_csv(path, index=False, na_rep="NaN", encoding="utf-8")
        paths.append(path)
    summary_path = root / "summary.csv"
    result.summary().to_csv(summary_path, index=False, na_rep="NaN")
    paths.append(summary_path)
    centers_path = root / "candidate_centers.csv"
    pd.DataFrame(result.candidate_centers, columns=["X", "Y", "Z"]).to_csv(centers_path, index=False)
    paths.append(centers_path)
    record = [f"dataset = {result.dataset_name}", f"length_unit = {result.length_unit}",
              f"window = {asdict(result.window)}", f"sampling = {asdict(result.sampling)}",
              f"voxel_size = {result.voxel_size}", f"diagnostics = {result.diagnostics}",
              "summary_std = sample SD across windows; not a confidence interval",
              "P21 = clipped 3D trace length / planar window area",
              f"python = {platform.python_version()}"]
    for package in ("tm-scan", "numpy", "pandas", "scipy", "scikit-learn", "shapely", "geopandas", "matplotlib"):
        try:
            record.append(f"{package} = {version(package)}")
        except PackageNotFoundError:
            record.append(f"{package} = not installed as distribution")
    record_path = root / "run.txt"
    record_path.write_text("\n".join(record) + "\n", encoding="utf-8")
    paths.append(record_path)
    return paths


def export_study(study, directory):
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, run in enumerate(study.runs):
        paths.extend(export_result(run, root / f"case_{index:03d}"))
    path = root / "study_summary.csv"
    study.summary.to_csv(path, index=False, na_rep="NaN")
    paths.append(path)
    return paths

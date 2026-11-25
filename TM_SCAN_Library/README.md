# tm-scan

Composable TM-Scan analyses of mapped fracture traces on 3D surface scans:
window sampling (P20, P21, mean trace length), scale and sampling studies,
orientation and size fitting, window-conditioned DFN simulation (P32),
spatial dependence and plotting.

## Installation

From this folder (always installs the current version):

```
pip install .
```

or from one of the wheels in `dist/`, which keeps every released version:

```
pip install dist/tm_scan-0.1.1-py3-none-any.whl
```

Requires Python >= 3.9 with numpy, pandas, scipy, scikit-learn, shapely,
geopandas, matplotlib, joblib and numba (installed automatically).

## Database layout

`load_dataset(directory)` reads an existing folder; nothing is created or renamed:

Minimum content:
```
database/
├── surface_scan/     point cloud(s): X Y Z per row (space- or comma-separated)
├── slope_boundary/   boundary shapefile(s) (*.shp)
└── traces/           one file per family (*.txt, *.csv, *.dat, *.xyz or XYZ *.shp)
```

Each trace is the straight segment between its first and last XYZ point.
The file name (without extension) is the family name; use
`Dataset.rename_families({...})` to rename.

## Main components

| Area | Functions / classes |
|---|---|
| Data | `load_dataset`, `list_point_clouds`, `Dataset`, `describe_cloud`, `plot_cloud` |
| Windows | `Window.sphere`, `Window.rectangle`, `Sampling`, `analyze`, `generate_centers` |
| Studies | `radius_study`, `scale_study`, `k_study`, `assess_roughness` |
| Orientation | `fit_from_planes`, `OrientationDistribution`, `compare_orientation_models`, `plot_stereonet` |
| Sizes | `fit_from_mu`, `SizeDistribution` |
| DFN | `FractureFamily`, `DFNConfig`, `simulate_dfn`, `DFNResult` |
| Spatial / density | `spatial_correlation`, `spatial_bootstrap`, `fit_joint_density` |
| Plots | `plot_dataset`, `plot_windows`, `plot_study`, `plot_dfn`, `plot_p32_map`, `plot_p32_uncertainty`, ... |

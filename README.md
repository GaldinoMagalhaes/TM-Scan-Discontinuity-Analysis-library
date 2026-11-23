[![DOI](https://zenodo.org/badge/1467938610.svg)](https://doi.org/10.5281/zenodo.24451837)

# What is this repository for?
TM-Scan is an open-source Python library for estimating discontinuity intensity (P20, P21) and mean discontinuity trace length (μ) from traces mapped on 3D point clouds, and for using these results to parameterize and simulate conditioned Discrete Fracture Networks (DFN). Besides the window analyses, the library provides point-cloud quality checks (occlusions and roughness), orientation models and stereonets, spatial-dependence diagnostics with a spatial block bootstrap, and P32 maps and uncertainty plots.

The repository includes a complete case study (`case_study.ipynb`) that can be used as a template for new analyses.

# Where do I run it?
In any Jupyter environment (Jupyter Lab/Notebook, or VS Code with the Jupyter extension), with Python 3.9 or later.

# How do I install it?
The first cell of `case_study.ipynb` installs the library and all its dependencies from the wheel included in the repository. After running it, restart the kernel. To install manually, run in the repository folder:

```text
pip install ./TM_SCAN_Library_v0.1.4/dist/tm_scan-0.1.4-py3-none-any.whl
```

The following libraries are installed automatically: NumPy, Pandas, SciPy, Scikit-learn, Shapely, GeoPandas, Matplotlib, Joblib and Numba.

# How is the repository organized?
The notebook uses paths relative to its own folder, so keep the structure below:

```text
Main_folder/
├── TM_SCAN_Library_v0.1.4/          library source code and installable wheel (dist/)
├── case_studies/
│   ├── slope_01/
│   │   ├── case_study.ipynb         complete workflow
│   │   └── database/
│   │       ├── mapped_planes/
│   │       ├── slope_boundary/
│   │       ├── surface_scan/
│   │       └── traces/
│   └── slope_02/                    same structure as slope_01
└── tests_and_validation/            tests and validation of the library
```

# How do I prepare a database?
Each database is one folder with the four subfolders below. All files must share the same Cartesian reference frame and length unit (metres); no reprojection or unit conversion is applied.

- **`surface_scan/`**: the point cloud, as a `.txt`, `.xyz`, `.csv` or `.dat` file without header, with X, Y and Z in the first three columns (space- or comma-separated; extra columns such as RGB are ignored). If the folder holds more than one cloud, select it with `tm.load_dataset(folder, point_cloud="name.txt")`.
- **`slope_boundary/`**: one or more shapefiles with the outcrop boundary as closed 3D lines (LineString or MultiLineString). All boundary files are combined and must share the same coordinate system.
- **`traces/`**: one file per discontinuity family; the file name is the family name (families can be renamed in the notebook with `FAMILY_NAMES`). Each trace is the straight segment between its two end points:
  - `.shp`: 3D LineString or MultiLineString geometries; only the first and last vertices of each line are used;
  - `.txt`: one trace per row, written as two coordinate triplets:

| First_point  | End_point    |
|--------------|--------------|
| (X0, Y0, Z0) | (X1, Y1, Z1) |

- **`mapped_planes/`**: one `.csv` file with the discontinuity planes measured on the point cloud, used for the orientation analysis. The required columns are `Nx`, `Ny`, `Nz` (plane normal), `Dip`, `Dip_Dir` and `Joint_set`. Each trace family is linked to the `Joint_set` label with the same name or the same number (e.g. F1 ↔ j1). Planes of sets without traces (e.g. foliation) appear in the stereonets but are not used in the DFN.

# What does the case study cover?
`case_study.ipynb` runs the complete workflow on `database01`, step by step:

1. Installation;
2. Reading the database;
3. Point-cloud quality (occlusions and grid nodes vs `k_smooth`);
4. 3D view of the point cloud, traces and boundary;
5. General data and measured trace lengths;
6. Orientations: stereonets and comparison of orientation models;
7. TM-Scan with circular (spherical) sampling windows, and roughness of the windows;
8. Analysis summary and scale effect;
9. Trace-length and size parameters (μ̄, σ̄, μ_D, σ_D, dip and dip direction);
10. Conditioned DFN simulations;
11. DFN summary and an example realization;
12. Spatial dependence and spatial block bootstrap;
13. Empirical joint probability of P32 and P21;
14. P32 map;
15. P32 uncertainty across DFN realizations.

All analysis parameters (window radius, `k_smooth`, radii of the scale-effect study, DFN settings, number of bootstrap replicates) are set in a single configuration cell at the top of the notebook.

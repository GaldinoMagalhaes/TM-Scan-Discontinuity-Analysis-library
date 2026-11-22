"""Window-conditioned DFN realizations with explicit assumptions and seeds."""
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import numpy as np
import pandas as pd
from .models import AnalysisResult, StudyResult, positive
from ._execution import run_tasks, validate_jobs
from . import _dfn_core as core
from .orientation import OrientationDistribution
from .sizes import SizeDistribution
from .io import _safe_name


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


@dataclass(frozen=True)
class FractureFamily:
    """Arithmetic diameter moments BEFORE truncation; Fisher angles use the modeling convention.

    Manual angles use the normal convention (-sin(dip)*sin(azimuth),
    -sin(dip)*cos(azimuth), cos(dip)). Window normals are used directly.
    """
    diameter_mean: float
    diameter_std: float
    fisher_dip: float = None
    fisher_dip_direction: float = None
    fisher_kappa: float = None
    diameter_min: float = 1.0
    diameter_max: float = math.inf
    orientation: OrientationDistribution = None
    size_fit: SizeDistribution = None

    def __post_init__(self):
        positive(self.diameter_mean, "diameter_mean")
        if not np.isfinite(self.diameter_min) or self.diameter_min < 0:
            raise ValueError("diameter_min must be finite and nonnegative.")
        if not np.isfinite(self.diameter_std) or self.diameter_std < 0:
            raise ValueError("diameter_std must be finite and nonnegative.")
        if self.orientation is None:
            if any(value is None for value in (self.fisher_dip, self.fisher_dip_direction, self.fisher_kappa)):
                raise ValueError("Supply manual Fisher parameters or an OrientationDistribution.")
            if not np.isfinite(self.fisher_kappa) or self.fisher_kappa < 0:
                raise ValueError("fisher_kappa must be finite and nonnegative.")
            if not np.isfinite(self.fisher_dip) or not 0 <= self.fisher_dip <= 90:
                raise ValueError("fisher_dip must be in [0,90] degrees.")
            if not np.isfinite(self.fisher_dip_direction) or not 0 <= self.fisher_dip_direction <= 360:
                raise ValueError("fisher_dip_direction must be in [0,360] degrees.")
        else:
            if not isinstance(self.orientation, OrientationDistribution):
                raise TypeError("orientation must be an OrientationDistribution.")
            if any(value is not None for value in (self.fisher_dip, self.fisher_dip_direction, self.fisher_kappa)):
                raise ValueError("Do not mix manual Fisher parameters with an orientation object.")
        if math.isnan(self.diameter_max) or self.diameter_max <= self.diameter_min:
            raise ValueError("diameter_max must exceed diameter_min.")
        SizeDistribution(self.diameter_mean, self.diameter_std, self.diameter_min, self.diameter_max)
        if self.size_fit is not None:
            if not isinstance(self.size_fit, SizeDistribution):
                raise TypeError("size_fit must be a SizeDistribution.")
            expected = (self.size_fit.diameter_mean, self.size_fit.diameter_std,
                        self.size_fit.diameter_min, self.size_fit.diameter_max)
            actual = (self.diameter_mean, self.diameter_std, self.diameter_min, self.diameter_max)
            if expected != actual:
                raise ValueError("Stored size fit and active diameter parameters disagree.")

    @classmethod
    def from_distributions(cls, *, orientation, size):
        if not isinstance(orientation, OrientationDistribution) or not isinstance(size, SizeDistribution):
            raise TypeError("Supply OrientationDistribution and SizeDistribution objects.")
        return cls(size.diameter_mean, size.diameter_std,
                   diameter_min=size.diameter_min, diameter_max=size.diameter_max,
                   orientation=orientation, size_fit=size)

    def core_parameters(self):
        return {"diameter_mean": self.diameter_mean, "diameter_std": self.diameter_std,
                "diameter_min": self.diameter_min, "diameter_max": self.diameter_max,
                "fisher_dip": self.fisher_dip if self.fisher_dip is not None else 0.,
                "fisher_dip_direction": self.fisher_dip_direction if self.fisher_dip_direction is not None else 0.,
                "fisher_kappa": self.fisher_kappa if self.fisher_kappa is not None else 0.}

    def summary(self):
        return {"diameter_parameters": {key: getattr(self, key) for key in
                                        ("diameter_mean", "diameter_std", "diameter_min", "diameter_max")},
                "orientation": self.orientation.summary() if self.orientation else
                    {"method": "manual Fisher", "dip": self.fisher_dip,
                     "dip_direction": self.fisher_dip_direction, "kappa": self.fisher_kappa,
                     "angle_convention": "modeling"},
                "size_fit": self.size_fit.summary() if self.size_fit else "manual"}



@dataclass(frozen=True)
class DFNConfig:
    """Cube edges are 2*half_size; all lengths use the input coordinate unit."""
    half_size: float = 36.0
    realizations: int = 10
    seed: int = 10000
    batch_size: int = 4096
    max_fractures: int = 100000000
    p32_boundary: str = "clip"
    geometry_tolerance: float = 1e-12

    def __post_init__(self):
        positive(self.half_size, "half_size")
        positive(self.geometry_tolerance, "geometry_tolerance")
        for name in ("realizations", "batch_size", "max_fractures"):
            _integer(getattr(self, name), name)
        _integer(self.seed, "seed", minimum=0)
        if self.p32_boundary not in ("clip", "full"):
            raise ValueError("p32_boundary must be clip or full.")


@dataclass
class DFNResult:
    realizations: pd.DataFrame
    summary: pd.DataFrame
    config: DFNConfig
    families: dict

    def export(self, directory):
        """Save each realization, grouped summaries and reproducibility metadata."""
        from importlib.metadata import version
        from . import __version__
        root = Path(directory).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        paths = [root/"dfn_realizations.csv", root/"dfn_summary.csv", root/"dfn_run.txt"]
        self.realizations.to_csv(paths[0], index=False, na_rep="NaN")
        self.summary.to_csv(paths[1], index=False, na_rep="NaN")
        lines = [f"tm-scan = {__version__}", f"config = {asdict(self.config)}",
                 "fracture_orientation = direct poles for fitted objects; modeling-convention angles for manual input",
                 "window_normal = directly copied from analyzed window",
                 "stopping_rule = exact N_traces = P20 * window_area",
                 "P32 = area of ALL generated fractures / generation volume",
                 "P32_q05/q95 = empirical realization quantiles, not posterior bounds",
                 "zero_target = windows with N_traces = 0 get P32 = 0"]
        lines += [f"{name} = {parameters.summary()}" for name, parameters in self.families.items()]
        names = [_safe_name(name) for name in self.families]
        if len(names) != len(set(names)):
            raise ValueError("Family names collide in parameter export paths.")
        for name, parameters in self.families.items():
            destination = root/"parameters"/_safe_name(name)
            if parameters.orientation:
                paths.extend(parameters.orientation.export(destination))
            if parameters.size_fit:
                paths.extend(parameters.size_fit.export(destination))
        lines += [f"{name} = {version(name)}" for name in ("numpy", "numba", "joblib")]
        paths[2].write_text("\n".join(lines)+"\n", encoding="utf-8")
        return paths


    def describe(self):
        """Diagnostic table, one row per family (print it or read the columns).

        simulations = family x window x realization runs; cube_edge and
        generation_volume describe each generated DFN; boundary is the P32
        area rule at the cube faces; diameter_min/max are the truncation.
        """
        table = self.realizations
        unit = table.length_unit.iloc[0] if "length_unit" in table and len(table) else ""
        rows = []
        for family, group in table.groupby("family", sort=False):
            parameters = self.families[family]
            generated = group.generated_fractures
            rows.append({
                "family": family,
                "cases": group.case_id.nunique(),
                "windows": len(group.drop_duplicates(["case_id", "window_id"])),
                "realizations_per_window": self.config.realizations,
                "simulations": len(group),
                "cube_edge": 2 * self.config.half_size,
                "generation_volume": (2 * self.config.half_size) ** 3,
                "length_unit": unit,
                "boundary_treatment": ("clip: fracture area inside the cube" if self.config.p32_boundary == "clip"
                                       else "full: whole area of fractures centred in the cube"),
                "diameter_min": parameters.diameter_min,
                "diameter_max": parameters.diameter_max,
                "diameter_mean": parameters.diameter_mean,
                "diameter_std": parameters.diameter_std,
                "orientation_model": parameters.orientation.method if parameters.orientation else "fisher (manual)",
                "fractures_per_dfn_mean": generated.mean(),
                "fractures_per_dfn_min": generated.min(),
                "fractures_per_dfn_max": generated.max(),
                "zero_target_simulations": int(group.zero_target.sum()),
                "seed": self.config.seed,
            })
        out = pd.DataFrame(rows).set_index("family")
        out.attrs["total_simulations"] = len(table)
        return out

    def fractures(self, family, window_id, *, realization=0, case_id=0):
        """Regenerate the discs of one stored realization from its seed.

        Returns a DataFrame (X, Y, Z, Nx, Ny, Nz, diameter, trace_length,
        intersects) in window-centred coordinates (the window centre is the
        origin; add the row's X, Y, Z to return to map coordinates). The
        regenerated fracture count is checked against the stored one.
        """
        import ast
        table = self.realizations
        match = table[(table.family == family) & (table.window_id == window_id)
                      & (table.realization == realization) & (table.case_id == case_id)]
        if len(match) != 1:
            raise ValueError("No single stored realization matches family/window_id/realization/case_id.")
        row = match.iloc[0]
        parameters = self.families[family]
        low_config = core.SimulationConfig(
            half_size=self.config.half_size, window_radius=row.radius if pd.notna(row.radius) else 1.,
            **parameters.core_parameters(), batch_size=self.config.batch_size,
            max_fractures=self.config.max_fractures, p32_boundary=self.config.p32_boundary,
            geometry_tolerance=self.config.geometry_tolerance)
        normal = np.array([row.Nx, row.Ny, row.Nz], dtype=float)
        frame = None if row.window_basis == "circle" else np.array(ast.literal_eval(row.window_basis), float)
        centers, normals, radii, traces = _generate_fractures(int(row.N_traces), normal, low_config,
                                                             int(row.seed), frame, parameters.orientation)
        if len(centers) != int(row.generated_fractures):
            raise RuntimeError("Regenerated realization differs from the stored one (version or inputs changed).")
        out = pd.DataFrame(np.column_stack([centers, normals]), columns=["X", "Y", "Z", "Nx", "Ny", "Nz"])
        out["diameter"], out["trace_length"], out["intersects"] = 2 * radii, traces, traces > 0
        out.attrs.update(window_normal=normal, window_center=np.array([row.X, row.Y, row.Z], float),
                         rectangle_frame=frame, radius=row.radius, half_size=self.config.half_size,
                         family=family, length_unit=row.length_unit, P32=row.P32)
        return out


def _generate_fractures(n_traces, window_normal, config, seed, rectangle_frame=None, orientation=None):
    """Same random sequence and stopping rule as core.simulate_realization, keeping the geometry."""
    empty = np.empty((0, 3))
    if n_traces == 0:
        return empty, empty, np.empty(0), np.empty(0)
    rng = np.random.default_rng(seed)
    mean_normal = core.dip_direction_to_normal(config.fisher_dip, config.fisher_dip_direction)
    log_mean, log_std = core._lognormal_parameters(config.diameter_mean, config.diameter_std)
    kept, intersections, generated = [], 0, 0
    while intersections < n_traces:
        if generated >= config.max_fractures:
            raise RuntimeError("Stopping criterion was not reached.")
        size = min(config.batch_size, config.max_fractures - generated)
        centers = rng.uniform(-config.half_size, config.half_size, size=(size, 3))
        radii = 0.5 * core._sample_truncated_lognormal(rng, size, log_mean, log_std,
                                                       config.diameter_min, config.diameter_max)
        normals = (core._sample_fisher_normals(rng, size, mean_normal, config.fisher_kappa)
                   if orientation is None else orientation.rvs(size, random_state=rng))
        if rectangle_frame is None:
            lengths = core._trace_lengths_batch(centers, normals, radii, window_normal,
                                                config.window_radius, config.geometry_tolerance)
        else:
            from ._dfn_geometry import _rectangle_trace_lengths_batch
            lengths = _rectangle_trace_lengths_batch(centers, normals, radii, rectangle_frame,
                                                     config.geometry_tolerance)
        hits = np.flatnonzero(lengths > 0.0)
        remaining = n_traces - intersections
        selected = int(hits[remaining - 1]) + 1 if hits.size >= remaining else size
        kept.append((centers[:selected], normals[:selected], radii[:selected], lengths[:selected]))
        intersections += int(np.count_nonzero(lengths[:selected] > 0.0))
        generated += selected
    return tuple(np.concatenate([part[i] for part in kept]) for i in range(4))


def _one_realization(record, normal, rectangle_frame, config, seed, orientation=None):
    p32, p21, generated = core.simulate_realization(record["N_traces"], normal, config,
                                                   int(seed), rectangle_frame=rectangle_frame, orientation=orientation)
    return {**record, "seed": int(seed), "P32": p32, "P21_sim": p21,
            "generated_fractures": generated, "zero_target": record["N_traces"] == 0}


def simulate_dfn(analysis, family_parameters, *, config=None, n_jobs=-1,
                 progress=True, window_ids=None, families=None):
    """Match observed trace count in each exact planar window geometry.

    analysis may be an AnalysisResult or StudyResult. Fracture parameters
    must be supplied for every analyzed family. window_ids optionally selects
    local zero-based window IDs in EACH case; missing IDs raise an error.
    Each window/family/realization is simulated independently. No continuous
    regional DFN is constructed, and P21 is not a conditioning target.
    """
    validate_jobs(n_jobs)
    config = config or DFNConfig()
    if not isinstance(config, DFNConfig):
        raise TypeError("config must be DFNConfig.")
    if isinstance(analysis, AnalysisResult):
        runs = [analysis]
    elif isinstance(analysis, StudyResult):
        runs = analysis.runs
    else:
        raise TypeError("analysis must be AnalysisResult or StudyResult.")
    if not isinstance(family_parameters, dict):
        raise TypeError("family_parameters must map family names to FractureFamily.")
    selected_families = None if families is None else set(families)
    available = set().union(*(run.tables.keys() for run in runs))
    if selected_families is not None and (not selected_families or not selected_families <= available):
        raise ValueError("families must select existing analyzed family names.")
    requested = None if window_ids is None else set(_integer(i, "window_id", 0) for i in window_ids)
    records = []
    for case_id, run in enumerate(runs):
        if requested is not None and not requested <= set(range(len(run.centers))):
            raise ValueError(f"Some requested window_ids are unavailable in case {case_id}.")
        for family, frame in run.tables.items():
            if selected_families is not None and family not in selected_families:
                continue
            parameters = family_parameters.get(family)
            if not isinstance(parameters, FractureFamily):
                raise ValueError(f"Provide FractureFamily parameters for {family!r}.")
            area = (math.pi*run.window.radius**2 if run.window.kind == "sphere"
                    else run.window.size_u*run.window.size_v)
            low_config = core.SimulationConfig(
                half_size=config.half_size, window_radius=run.window.radius or 1.,
                **parameters.core_parameters(), batch_size=config.batch_size,
                max_fractures=config.max_fractures, p32_boundary=config.p32_boundary,
                geometry_tolerance=config.geometry_tolerance)
            for wi, row in enumerate(frame.to_dict("records")):
                if requested is not None and wi not in requested:
                    continue
                normal = np.array([row["Nx"], row["Ny"], row["Nz"]], dtype=float)
                if not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-12:
                    raise ValueError(f"Invalid window normal: case {case_id}, {family}, window {wi}.")
                normal /= np.linalg.norm(normal)
                raw_count = float(row["N_traces"])
                if not math.isfinite(raw_count) or raw_count < 0 or raw_count != int(raw_count):
                    raise ValueError("N_traces must be a nonnegative integer.")
                count = int(raw_count)
                if not math.isclose(float(row["P20"])*area, count, rel_tol=1e-10, abs_tol=1e-9):
                    raise ValueError("Observed P20 and N_traces are inconsistent with window area.")
                rectangle_frame = None
                if run.window.kind == "rectangle":
                    quad = np.asarray(run.quads[wi], float)
                    u, v = quad[1]-quad[0], quad[3]-quad[0]
                    u, v = u/np.linalg.norm(u), v/np.linalg.norm(v)
                    rectangle_frame = np.array([u, v, [run.window.size_u/2, run.window.size_v/2, 0.]])
                    extent = np.abs(quad-quad.mean(axis=0)).max(axis=0)
                else:
                    extent = run.window.radius*np.sqrt(np.maximum(0., 1.-normal*normal))
                if np.max(extent) > config.half_size+config.geometry_tolerance:
                    raise ValueError("The sampling window must fit inside the generation cube; increase half_size.")
                base = dict(case_id=case_id, dataset=run.dataset_name, family=family, window_id=wi,
                            window_kind=run.window.kind,
                            dfn_geometry="planar_disk" if run.window.kind == "sphere" else "planar_rectangle",
                            radius=run.window.radius,
                            size_u=run.window.size_u, size_v=run.window.size_v,
                            rotation_cw=run.window.rotation_cw, k_smooth=run.sampling.k_smooth,
                            length_unit=run.length_unit, X=row["X"], Y=row["Y"], Z=row["Z"],
                            Nx=normal[0], Ny=normal[1], Nz=normal[2], N_traces=count,
                            window_area=area, P20=count/area, P21_observed=row["P21"],
                            generation_volume=(2*config.half_size)**3,
                            window_basis=repr(rectangle_frame.tolist()) if rectangle_frame is not None else "circle")
                records.append((base, normal, rectangle_frame, low_config, parameters.orientation))
    if not records:
        raise ValueError("No windows selected for DFN simulation.")
    # Seed assignment follows deterministic case/family/window/realization order.
    seeds = np.random.SeedSequence(config.seed).spawn(len(records)*config.realizations)
    arguments = []
    for index, (base, normal, rectangle_frame, low_config, orientation) in enumerate(records):
        for realization in range(config.realizations):
            seed = seeds[index*config.realizations+realization].generate_state(1, dtype=np.uint64)[0]
            arguments.append(({**base, "realization": realization}, normal, rectangle_frame, low_config, seed, orientation))
    # Compile geometry once before process workers load the cached kernels.
    core._warm_up_compiled_geometry(records[0][3])
    if any(record[2] is not None for record in records):
        from ._dfn_geometry import _rectangle_trace_lengths_batch
        frame = next(record[2] for record in records if record[2] is not None)
        _rectangle_trace_lengths_batch(np.zeros((1, 3)), np.array([[1., 0., 0.]]),
                                        np.ones(1), frame, config.geometry_tolerance)
    rows = run_tasks(_one_realization, arguments, n_jobs=n_jobs, backend="loky",
                     progress=progress, label="DFN realizations")
    table = pd.DataFrame(rows)
    summary_rows = []
    for _, group in table.groupby(["case_id", "family", "window_id"], sort=False):
        summary = {key: group.iloc[0][key] for key in records[0][0]}
        values = group.P32
        summary.update(realizations=len(group), P32_mean=values.mean(), P32_std=values.std(ddof=1),
                       P32_q05=values.quantile(.05), P32_q95=values.quantile(.95),
                       P21_sim_mean=group.P21_sim.mean(),
                       generated_fractures_mean=group.generated_fractures.mean())
        summary_rows.append(summary)
    used_families = {name: family_parameters[name] for name in table.family.unique()}
    return DFNResult(table, pd.DataFrame(summary_rows), config, used_families)

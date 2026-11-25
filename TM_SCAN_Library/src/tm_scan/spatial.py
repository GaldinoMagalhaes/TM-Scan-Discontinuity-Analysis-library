"""Spatial dependence between windows and spatial block bootstrap.

Uncertainty here concerns spatially averaged estimates of the OBSERVED map:
blocks of neighbouring windows (and traces) are resampled together, so the
dependence created by overlapping windows and by geology is preserved.
No mapping-error scenario is simulated.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from .models import AnalysisResult

METRIC_COLUMNS = {"P20": "P20", "P21": "P21", "mu": "μ"}
_SHIFTS = {2: [(0., 0.), (.5, 0.), (0., .5), (.5, .5)],
           3: [(0., 0., 0.), (.5, .5, 0.), (.5, 0., .5), (0., .5, .5)]}


def _window_scale(window):
    """Half-width used for lags and blocks: R for spheres, half the longest side for rectangles."""
    return float(window.radius) if window.kind == "sphere" else 0.5 * max(window.size_u, window.size_v)


def window_values(result, dfn=None, *, families=None, case_id=0):
    """Wide per-window table: X, Y, Z, P20_/P21_/mu_<family> and, with `dfn`,
    P32_<family> (mean over realizations) and P32_Total."""
    if not isinstance(result, AnalysisResult):
        raise TypeError("result must be an AnalysisResult (one case of a study: study.runs[i]).")
    names = list(result.tables) if families is None else list(families)
    if not names or not set(names) <= set(result.tables):
        raise ValueError("families must select analyzed families.")
    wide = pd.DataFrame(np.asarray(result.centers, float).reshape(-1, 3), columns=["X", "Y", "Z"])
    wide.index.name = "window_id"
    for family in names:
        frame = result.tables[family]
        if len(frame) != len(wide):
            raise ValueError(f"Family {family!r} table does not match the accepted windows.")
        for metric, column in METRIC_COLUMNS.items():
            wide[f"{metric}_{family}"] = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    if dfn is not None:
        summary = dfn.summary[dfn.summary.case_id == case_id]
        if summary.duplicated(["window_id", "family"]).any():
            raise ValueError("Duplicate window/family records in the DFN summary.")
        pivot = summary.pivot(index="window_id", columns="family", values="P32_mean")
        available = [family for family in names if family in pivot]
        for family in available:
            wide[f"P32_{family}"] = pivot[family].reindex(wide.index).to_numpy(float)
        if len(available) > 1:
            wide["P32_Total"] = wide[[f"P32_{family}" for family in available]].sum(axis=1, min_count=len(available))
    return wide


# ------------------------------------------------------------ correlogram ---
def spatial_correlation(result, dfn=None, *, metrics=None, families=None, lag_width=None,
                        max_lag=None, min_pairs=30, case_id=0):
    """Empirical pair correlation of window values against 3D centre distance.

    Pairs are binned by lag (default width R/2, up to 6R). Each bin uses a
    common mean and variance for both members of the pair (classical
    correlogram estimator). Bins with fewer than `min_pairs` are dropped.
    Default metrics: P21 of each family and, with `dfn`, P32_Total.
    Returns a DataFrame (metric, lag, pair_count, correlation); the window
    half-width is stored in `.attrs['radius']`.
    """
    wide = window_values(result, dfn, families=families, case_id=case_id)
    scale = _window_scale(result.window)
    if metrics is None:
        metrics = [column for column in wide if column.startswith("P21_")]
        if "P32_Total" in wide:
            metrics.append("P32_Total")
        elif dfn is not None:
            metrics += [column for column in wide if column.startswith("P32_")]
    missing = [metric for metric in metrics if metric not in wide]
    if missing:
        raise ValueError(f"Unknown metrics {missing}; available: {list(wide.columns[3:])}.")
    lag_width = float(lag_width or scale / 2)
    max_lag = float(max_lag or 6 * scale)
    edges = np.arange(0, max_lag + lag_width, lag_width)
    xyz = wide[["X", "Y", "Z"]].to_numpy(float)
    pairs = cKDTree(xyz).query_pairs(edges[-1], output_type="ndarray")
    left, right = pairs[:, 0], pairs[:, 1]
    distance = np.linalg.norm(xyz[left] - xyz[right], axis=1)
    lag_bin = np.digitize(distance, edges) - 1
    rows = []
    for metric in metrics:
        values = wide[metric].to_numpy(float)
        finite = np.isfinite(values[left]) & np.isfinite(values[right])
        for b in range(len(edges) - 1):
            selected = finite & (lag_bin == b)
            count = int(selected.sum())
            if count < min_pairs:
                continue
            a, c = values[left[selected]], values[right[selected]]
            mean = 0.5 * (a.mean() + c.mean())
            variance = 0.5 * ((a * a).mean() + (c * c).mean()) - mean ** 2
            rows.append(dict(metric=metric, lag=0.5 * (edges[b] + edges[b + 1]), pair_count=count,
                             correlation=((a * c).mean() - mean ** 2) / variance if variance > 1e-15 else np.nan))
    table = pd.DataFrame(rows, columns=["metric", "lag", "pair_count", "correlation"])
    table.attrs.update(radius=scale, length_unit=result.length_unit,
                       families=list(result.tables) if families is None else list(families))
    return table


def overlap_correlation(distance, radius):
    """Correlation expected only from the area shared by two discs of radius R."""
    x = np.clip(np.asarray(distance, float) / (2 * radius), 0, 1)
    return (2 / np.pi) * (np.arccos(x) - x * np.sqrt(1 - x ** 2))


# -------------------------------------------------------- block bootstrap ---
def _block_frame(xyz, blocks):
    """Coordinates used to build blocks: 2D best-fit plane or 3D, and the planar variance share."""
    centered = xyz - xyz.mean(axis=0)
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    planar = float(np.sum(singular[:2] ** 2) / np.sum(singular ** 2))
    if blocks == "auto":
        blocks = "2d" if planar >= 0.90 else "3d"
    if blocks not in ("2d", "3d"):
        raise ValueError("blocks must be 'auto', '2d' or '3d'.")
    transform = (lambda p: (p - xyz.mean(axis=0)) @ axes[:2].T) if blocks == "2d" else (lambda p: p - xyz.mean(axis=0))
    return blocks, planar, transform


def _draw_multiplicities(item_coordinates, edge, n_boot, seed, min_blocks):
    """Multiplicity of every item in each replicate, from resampled fixed blocks.

    Four shifted grid origins share the replicates; grids with fewer than
    `min_blocks` occupied blocks are skipped. Blocks are occupied when they
    hold at least one item of the FIRST group (the windows)."""
    rng = np.random.default_rng(seed)
    dimension = item_coordinates[0].shape[1]
    low = np.min([c.min(axis=0) for c in item_coordinates if len(c)], axis=0)
    shifts = _SHIFTS[dimension]
    per_shift = int(np.ceil(n_boot / len(shifts)))
    parts, block_counts = [[] for _ in item_coordinates], []
    for shift in shifts:
        grids = [np.floor((c - low) / edge + np.asarray(shift)).astype(np.int64) for c in item_coordinates]
        unique, inverse = np.unique(np.vstack(grids), axis=0, return_inverse=True)
        inverse = inverse.ravel()
        splits = np.cumsum([len(g) for g in grids])[:-1]
        membership = np.split(inverse, splits)
        occupied = np.unique(membership[0])
        if len(occupied) < min_blocks:
            continue
        block_counts.append(len(occupied))
        chosen = occupied[rng.integers(0, len(occupied), size=(per_shift, len(occupied)))]
        counts = np.zeros((per_shift, len(unique)), dtype=np.int32)
        rows = np.repeat(np.arange(per_shift), chosen.shape[1])
        np.add.at(counts, (rows, chosen.ravel()), 1)
        for group, member in enumerate(membership):
            parts[group].append(counts[:, member])
    if not block_counts:
        return None, np.nan
    return [np.vstack(p)[:n_boot] for p in parts], float(np.mean(block_counts))


def _weighted_means(values, multiplicities):
    """Bootstrap means of every column (NaN ignored per column). values: (n, m)."""
    finite = np.isfinite(values)
    clean = np.where(finite, values, 0.0)
    numerator = multiplicities @ clean
    denominator = multiplicities @ finite.astype(float)
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator > 0)


def _kde(values, grid, bandwidth, weights=None):
    """Gaussian KDE on `grid`; `weights` may be (replicates, n) -> (replicates, grid)."""
    kernel = np.exp(-0.5 * ((grid[None, :] - values[:, None]) / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))
    if weights is None:
        return kernel.mean(axis=0)
    total = weights.sum(axis=1, keepdims=True).astype(float)
    return np.divide(weights @ kernel, total, out=np.full((len(weights), len(grid)), np.nan), where=total > 0)


@dataclass
class SpatialBootstrap:
    """Result of `spatial_bootstrap`.

    values          : per-window table (see `window_values`)
    means           : bootstrap means, one row per replicate, one column per metric
    iid_means       : same, resampling windows independently (no blocks)
    multiplicities  : (replicates, windows) block-bootstrap counts
    trace_lengths   : {family: lengths}; trace_multiplicities likewise (if a dataset was given)
    """
    values: pd.DataFrame
    means: pd.DataFrame
    iid_means: pd.DataFrame
    multiplicities: np.ndarray
    block_edge: float
    n_blocks: float
    blocks: str
    planar_fraction: float
    radius: float
    families: list
    length_unit: str = "m"
    seed: int = 0
    min_blocks: int = 8
    trace_lengths: dict = field(default_factory=dict)
    trace_multiplicities: dict = field(default_factory=dict, repr=False)
    _coordinates: np.ndarray = field(default=None, repr=False)

    @property
    def metrics(self):
        return list(self.means.columns)

    def _observed(self, metric):
        if metric.startswith("trace_length_"):
            return np.asarray(self.trace_lengths[metric[len("trace_length_"):]], float)
        return self.values[metric].to_numpy(float)

    def summary(self, level=0.95):
        """Per metric: observed mean, SE without and with spatial blocks, and interval of the MEAN.

        n_eff is SE-equivalent (not a count of independent fractures)."""
        tail = 100 * (1 - level) / 2
        rows = []
        for metric in self.metrics:
            observed = self._observed(metric)
            n_valid = int(np.isfinite(observed).sum())
            mean = float(np.nanmean(observed)) if n_valid else np.nan
            block = self.means[metric].to_numpy(float)
            block = block[np.isfinite(block)]
            iid = self.iid_means[metric].to_numpy(float)
            iid = iid[np.isfinite(iid)]
            spatial_se = float(block.std(ddof=1)) if len(block) > 1 else np.nan
            iid_se = float(iid.std(ddof=1)) if len(iid) > 1 else np.nan
            lower, upper = np.percentile(block, [tail, 100 - tail]) if len(block) else (np.nan, np.nan)
            rows.append(dict(metric=metric, n_valid=n_valid, mean=mean, iid_se=iid_se, spatial_se=spatial_se,
                             spatial_relative_se_pct=100 * spatial_se / abs(mean) if mean else np.nan,
                             se_inflation=spatial_se / iid_se if iid_se and iid_se > 0 else np.nan,
                             n_eff=n_valid * (iid_se / spatial_se) ** 2 if spatial_se and spatial_se > 0 else np.nan,
                             lower=lower, upper=upper, valid_replicates=len(block)))
        return pd.DataFrame(rows).set_index("metric")

    def density(self, metric, *, level=0.95, grid_size=256):
        """KDE of the observed values with a pointwise bootstrap envelope.

        Returns a DataFrame: value, density, lower, upper. The bandwidth is
        fixed from the observed values (Scott-type), so the envelope shows
        the sensitivity to the spatial composition of the sample.
        """
        observed = self._observed(metric)
        weights = (self.trace_multiplicities[metric[len("trace_length_"):]]
                   if metric.startswith("trace_length_") else self.multiplicities)
        finite = np.isfinite(observed)
        values, weights = observed[finite], weights[:, finite]
        if len(values) < 3 or values.std(ddof=1) == 0:
            raise ValueError(f"Not enough distinct finite values of {metric!r} for a density.")
        bandwidth = values.std(ddof=1) * len(values) ** (-0.2)
        grid = np.linspace(0.0 if values.min() >= 0 else values.min() - 3 * bandwidth,
                           values.max() + 3 * bandwidth, grid_size)
        base = _kde(values, grid, bandwidth)
        draws = np.vstack([_kde(values, grid, bandwidth, weights[i:i + 100]) for i in range(0, len(weights), 100)])
        draws = draws[np.isfinite(draws).all(axis=1)]
        tail = 100 * (1 - level) / 2
        lower, upper = np.percentile(draws, [tail, 100 - tail], axis=0)
        return pd.DataFrame(dict(value=grid, density=base, lower=lower, upper=upper))

    def block_sensitivity(self, edges=None, *, n_boot=None):
        """Relative SE of every mean for several block edges (default 2R..6R)."""
        if edges is None:
            edges = np.arange(2, 7) * self.radius
        n_boot = n_boot or len(self.multiplicities)
        columns = [m for m in self.metrics if not m.startswith("trace_length_")]
        values = self.values[columns].to_numpy(float)
        observed = np.nanmean(values, axis=0)
        rows = []
        for edge in edges:
            drawn, n_blocks = _draw_multiplicities([self._coordinates], float(edge), n_boot,
                                                   self.seed + 520 + int(float(edge) * 100), self.min_blocks)
            if drawn is None:
                continue
            se = np.nanstd(_weighted_means(values, drawn[0].astype(float)), axis=0, ddof=1)
            rows += [dict(block_edge=float(edge), metric=metric, mean_blocks=n_blocks, spatial_se=se[j],
                          spatial_relative_se_pct=100 * se[j] / abs(observed[j]) if observed[j] else np.nan)
                     for j, metric in enumerate(columns)]
        return pd.DataFrame(rows)

    def size_parameters(self, family, *, diameter_min=0.0, diameter_max=np.inf, distribution="lognormal"):
        """Per replicate: mean mu, trace CV and the lognormal diameter fit (`fit_from_mu`).

        Requires trace lengths (run `spatial_bootstrap` with `dataset=`).
        Inadmissible fits are kept as NaN with the reason."""
        from .sizes import fit_from_mu
        if family not in self.trace_lengths:
            raise ValueError("Trace lengths unavailable: pass dataset= to spatial_bootstrap.")
        lengths = np.asarray(self.trace_lengths[family], float)
        w = self.trace_multiplicities[family].astype(float)
        total = w.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = (w @ lengths) / total
            second = (w @ lengths ** 2) / total
            variance = (second - mean ** 2) * total / (total - 1)
            cv = np.sqrt(np.clip(variance, 0, None)) / mean
        mu = self.means[f"mu_{family}"].to_numpy(float)
        rows = []
        for replicate, (m, c) in enumerate(zip(mu, cv)):
            row = dict(replicate=replicate, mu=m, trace_cv=c, diameter_mean=np.nan, diameter_std=np.nan,
                       invalid_reason="")
            try:
                fit = fit_from_mu(m, trace_cv=c, distribution=distribution,
                                  diameter_min=diameter_min, diameter_max=diameter_max)
                row.update(diameter_mean=fit.diameter_mean, diameter_std=fit.diameter_std)
            except (ValueError, RuntimeError) as error:
                row["invalid_reason"] = str(error)
            rows.append(row)
        return pd.DataFrame(rows)

    def plot(self, **kwargs):
        return plot_bootstrap(self, **kwargs)


def spatial_bootstrap(result, dfn=None, *, dataset=None, families=None, block_edge=None, n_boot=1000,
                      seed=0, blocks="auto", min_blocks=8, case_id=0):
    """Spatial block bootstrap of the window estimates (P20, P21, mu and, with `dfn`, P32).

    Windows are grouped in fixed blocks of edge `block_edge` (default 2R, the
    distance beyond which windows no longer overlap) on four shifted grids;
    whole blocks are resampled with replacement. With `dataset`, mapped
    traces are resampled in the same blocks (by midpoint), giving the trace
    length distribution and its uncertainty.
    blocks: 'auto' uses the 2D best-fit plane of the window centres when it
    holds >= 90 % of their variance (slopes), otherwise 3D cubes (e.g. tunnels).
    For a study, pass one case: `spatial_bootstrap(study.runs[i], ...)`.
    """
    if isinstance(n_boot, bool) or int(n_boot) != n_boot or n_boot < 20:
        raise ValueError("n_boot must be an integer >= 20.")
    n_boot = int(n_boot)
    wide = window_values(result, dfn, families=families, case_id=case_id)
    names = list(result.tables) if families is None else list(families)
    if len(wide) < 30:
        raise ValueError("At least 30 windows are required for a spatial bootstrap.")
    scale = _window_scale(result.window)
    block_edge = float(block_edge or 2 * scale)
    xyz = wide[["X", "Y", "Z"]].to_numpy(float)
    blocks, planar, transform = _block_frame(xyz, blocks)
    coordinates = [transform(xyz)]
    lengths = {}
    if dataset is not None:
        from ._traces import endpoints
        for family in names:
            A, B = endpoints(dataset.traces[family])
            lengths[family] = np.linalg.norm(B - A, axis=1)
            coordinates.append(transform(0.5 * (A + B)))
    drawn, n_blocks = _draw_multiplicities(coordinates, block_edge, n_boot, seed, min_blocks)
    if drawn is None:
        raise ValueError(f"Fewer than {min_blocks} occupied blocks at edge {block_edge:g}; "
                         "reduce block_edge or min_blocks.")
    multiplicities = drawn[0]
    columns = [column for column in wide.columns if column not in ("X", "Y", "Z")]
    values = wide[columns].to_numpy(float)
    means = pd.DataFrame(_weighted_means(values, multiplicities.astype(float)), columns=columns)
    rng = np.random.default_rng(seed + 410)
    iid = np.empty((n_boot, len(columns)))
    for start in range(0, n_boot, 100):
        stop = min(start + 100, n_boot)
        picks = rng.integers(0, len(values), size=(stop - start, len(values)))
        counts = np.zeros((stop - start, len(values)))
        np.add.at(counts, (np.repeat(np.arange(stop - start), len(values)), picks.ravel()), 1)
        iid[start:stop] = _weighted_means(values, counts)
    iid_means = pd.DataFrame(iid, columns=columns)
    trace_multiplicities = {}
    for family, weights in zip(lengths, drawn[1:]):
        trace_multiplicities[family] = weights
        name = f"trace_length_{family}"
        means[name] = _weighted_means(lengths[family][:, None], weights.astype(float))[:, 0]
        picks = rng.integers(0, len(lengths[family]), size=(n_boot, len(lengths[family])))
        iid_means[name] = lengths[family][picks].mean(axis=1)
    return SpatialBootstrap(wide, means, iid_means, multiplicities, block_edge, n_blocks, blocks, planar,
                            scale, names, result.length_unit, seed, min_blocks, lengths,
                            trace_multiplicities, coordinates[0])


# ---------------------------------------------------------------- figures ---
def _metric_label(metric, unit):
    kind, _, family = metric.partition("_")
    if metric.startswith("trace_length_"):
        return f"Biased trace length ({unit})", rf"Density ({unit}$^{{-1}}$)"
    return {"P20": (rf"$P_{{20}}$ ({unit}$^{{-2}}$)", rf"Density ({unit}$^2$)"),
            "P21": (rf"$P_{{21}}$ ({unit}$^{{-1}}$)", f"Density ({unit})"),
            "P32": (rf"$P_{{32}}$ ({unit}$^{{-1}}$)", f"Density ({unit})"),
            "mu": (rf"Window estimate of $\mu$ ({unit})", rf"Density ({unit}$^{{-1}}$)")}[kind]


def plot_bootstrap(bootstrap, *, metrics=None, families=None, level=0.95):
    """Panels of per-window PDFs with bootstrap envelopes; inset = bootstrap means.

    One panel per metric (default trace length if available, P20, P21, mu),
    one colour per family. Curves: KDE of the observed windows; bands:
    pointwise envelope from the spatial block bootstrap; inset histograms:
    bootstrap distribution of the spatial mean, dashed line = observed mean.
    Returns (figure, axes).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator
    from .style import style_context, family_colors
    families = list(bootstrap.families) if families is None else list(families)
    colors = family_colors(bootstrap.families)
    if metrics is None:
        metrics = (["trace_length"] if bootstrap.trace_lengths else []) + ["P20", "P21", "mu"]
        if any(m.startswith("P32_") for m in bootstrap.metrics) and not bootstrap.trace_lengths:
            metrics.append("P32")
    metrics = list(metrics)
    rows = int(np.ceil(len(metrics) / 2))
    columns = 1 if len(metrics) == 1 else 2
    with style_context({"axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
                         "legend.fontsize": 14}):
        figure, axes = plt.subplots(rows, columns, figsize=(5.5 * columns, 3.75 * rows), squeeze=False)
        figure.subplots_adjust(left=0.09, right=0.98, top=0.97, bottom=0.6 / (3.75 * rows) + 0.07,
                               wspace=0.28, hspace=0.38)
        for metric, ax in zip(metrics, axes.flat):
            samples, estimates = {}, {}
            for family in families:
                name = f"{metric}_{family}"
                if name not in bootstrap.metrics:
                    continue
                curve = bootstrap.density(name, level=level)
                ax.fill_between(curve.value, curve.lower, curve.upper, color=colors[family], alpha=0.20, linewidth=0)
                ax.plot(curve.value, curve.density, color=colors[family], lw=2)
                draws = bootstrap.means[name].to_numpy(float)
                samples[family] = draws[np.isfinite(draws)]
                estimates[family] = float(np.nanmean(bootstrap._observed(name)))
            xlabel, ylabel = _metric_label(f"{metric}_", bootstrap.length_unit)
            if samples:
                inset = ax.inset_axes([0.53, 0.57, 0.44, 0.38], zorder=5)
                inset.set_facecolor("white")
                bins = np.histogram_bin_edges(np.concatenate(list(samples.values())), bins=25)
                for family, draws in samples.items():
                    inset.hist(draws, bins=bins, density=True, histtype="bar", color=colors[family], alpha=0.40,
                               edgecolor=colors[family], linewidth=0.4)
                    inset.axvline(estimates[family], color=colors[family], lw=1.2, ls="--")
                inset.set_title("Bootstrap estimates", fontsize=10, pad=3)
                inset.xaxis.set_major_locator(MaxNLocator(3))
                inset.tick_params(axis="x", labelsize=10, length=2)
                inset.tick_params(axis="y", left=False, labelleft=False)
                inset.spines[["top", "right", "left"]].set_visible(False)
            ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, None), ylim=(0, None))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(which="minor", length=2)
            ax.set_axisbelow(True)
            ax.grid(alpha=0.6)
        for ax in axes.flat[len(metrics):]:
            ax.set_visible(False)
        handles = [Line2D([0], [0], color=colors[family], lw=2.5, label=family) for family in families]
        figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=len(families),
                      frameon=False, columnspacing=2.5)
    return figure, axes


def plot_spatial_correlation(correlation, *, radius=None, labels=None, block_edge_label=True):
    """Diagnostic correlogram: observed pair correlation vs the part explained by window overlap.

    `correlation` comes from `spatial_correlation`. The shaded zone marks
    overlapping windows (centres closer than 2R); the dotted curve is the
    correlation expected from the shared window area alone.
    Returns (figure, axis).
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
    from .style import style_context, TOTAL, MUTED, family_colors
    if correlation.empty:
        raise ValueError("The correlogram is empty (too few window pairs per lag).")
    R = float(radius or correlation.attrs.get("radius"))
    unit = correlation.attrs.get("length_unit", "m")
    colors = family_colors(correlation.attrs.get("families", []))
    labels = labels or {}
    x_max = correlation["lag"].max() + R / 4
    y_min = np.floor((np.nanmin(correlation["correlation"]) - 0.05) * 10) / 10
    with style_context({"font.size": 13, "axes.labelsize": 14, "legend.fontsize": 12,
                         "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.spines.top": True,
                         "axes.spines.right": True, "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True, "xtick.major.size": 5, "ytick.major.size": 5,
                         "xtick.minor.size": 2.5, "ytick.minor.size": 2.5}):
        figure, ax = plt.subplots(figsize=(6.5, 4.3), dpi=150)
        ax.axvspan(0, 2 * R, color="0.92", zorder=0)
        ax.text(R, 0.97, "Overlapping\nwindows", transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=12, color="0.35")
        ax.axvline(2 * R, color="0.35", ls="--", lw=1.1, zorder=1)
        ax.text(2 * R + 0.012 * x_max, 0.97, "2R = block edge" if block_edge_label else "2R", ha="left",
                va="top", rotation=90, transform=ax.get_xaxis_transform(), fontsize=12, color="0.35")
        lags = np.linspace(0, x_max, 400)
        ax.plot(lags, overlap_correlation(lags, R), ls=":", lw=1.6, color="0.25", zorder=2,
                label="Window overlap only")
        for metric in dict.fromkeys(correlation["metric"]):
            kind, _, family = metric.partition("_")
            color = TOTAL if family == "Total" else colors.get(family, MUTED)
            names = correlation.attrs.get("families", [])
            default = rf"$P_{{{kind[1:]}}}$ " + ("+".join(map(str, names)) if family == "Total" else family) \
                if kind in ("P20", "P21", "P32") else (rf"$\mu$ {family}" if kind == "mu" else metric)
            s = correlation[correlation["metric"] == metric].sort_values("lag")
            ax.plot(s["lag"], s["correlation"], "o-", color=color, lw=1.6, ms=6, zorder=3,
                    label=labels.get(metric, default))
        ax.axhline(0, color="0.6", lw=0.8, zorder=1)
        ax.set(xlim=(0, x_max), ylim=(y_min, 1.05), xlabel=f"3D distance between window centres ({unit})",
               ylabel="Empirical pair correlation")
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(axis="y", which="major", alpha=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper right")
        figure.tight_layout()
    return figure, ax

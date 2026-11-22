"""Joint P32/P21 KDEs and analytically normalized horizontal conditionals."""
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr, log_ndtr, logsumexp
from scipy.stats import gaussian_kde, norm


@dataclass
class ConditionalDensity:
    p21_fixed: float
    means: np.ndarray
    std: float
    log_weights: np.ndarray
    support: str
    log_normalizer: float

    def pdf(self, p32):
        raw = np.asarray(p32, float)
        x = np.atleast_1d(raw).ravel()
        logs = norm.logpdf(x[:, None], loc=self.means, scale=self.std)+self.log_weights
        values = np.exp(logsumexp(logs, axis=1)-self.log_normalizer)
        if self.support == "positive":
            values[x < 0] = 0.
        return values.reshape(raw.shape)

    def cdf(self, p32):
        raw = np.asarray(p32, float)
        x = np.atleast_1d(raw).ravel()
        weights = np.exp(self.log_weights)
        z = (x[:, None]-self.means)/self.std
        cdf = ndtr(z)
        if self.support == "positive":
            lower = -self.means/self.std
            # Use survival probabilities in the upper tail to avoid subtracting
            # two numbers rounded to one when the conditional mean is negative.
            large = np.where(lower > 0, log_ndtr(-lower), log_ndtr(z))
            small = np.where(lower > 0, log_ndtr(-z), log_ndtr(lower))
            with np.errstate(divide="ignore", invalid="ignore"):
                log_mass = large+np.log(-np.expm1(small-large))
            values = np.exp(logsumexp(log_mass+self.log_weights, axis=1)-self.log_normalizer)
            values[x <= 0] = 0.
        else:
            values = cdf@weights
        return np.clip(values, 0., 1.).reshape(raw.shape)

    def ppf(self, probability):
        probability = float(probability)
        if not 0 < probability < 1:
            raise ValueError("probability must be strictly between zero and one.")
        lower = 0. if self.support == "positive" else float(self.means.min()-10*self.std)
        upper = float(max(0., self.means.max())+10*self.std)
        while float(self.cdf(upper)) < probability:
            upper += max(abs(upper), self.std)
        return brentq(lambda value: float(self.cdf(value))-probability, lower, upper, xtol=1e-12)

    def grid(self, *, size=500, upper=None):
        if not isinstance(size, int) or size < 2:
            raise ValueError("size must be an integer >= 2.")
        lower = 0. if self.support == "positive" else self.ppf(1e-6)
        upper = self.ppf(1.-1e-6) if upper is None else float(upper)
        if upper <= lower:
            raise ValueError("Invalid conditional grid limits.")
        x = np.linspace(lower, upper, size)
        return x, self.pdf(x)


@dataclass
class JointDensity:
    family: str
    kde: object
    support: str
    positive_mass: float

    def pdf(self, p32, p21):
        x, y = np.broadcast_arrays(np.asarray(p32, float), np.asarray(p21, float))
        values = self.kde(np.vstack([x.ravel(), y.ravel()]))
        if self.support == "positive":
            values /= self.positive_mass
            values[(x.ravel() < 0) | (y.ravel() < 0)] = 0.
        return values.reshape(x.shape)

    def conditional(self, p21_fixed):
        value = float(p21_fixed)
        if not math.isfinite(value) or (self.support == "positive" and value < 0):
            raise ValueError("p21_fixed must be finite and within the physical support.")
        covariance = self.kde.covariance
        sy = math.sqrt(covariance[1, 1])
        slope = covariance[0, 1]/covariance[1, 1]
        variance = covariance[0, 0]-covariance[0, 1]**2/covariance[1, 1]
        if variance <= 0:
            raise ValueError("Conditional covariance is singular.")
        means = self.kde.dataset[0]+slope*(value-self.kde.dataset[1])
        log_weights = np.log(self.kde.weights)+norm.logpdf(value, self.kde.dataset[1], sy)
        log_weights -= logsumexp(log_weights)
        sigma = math.sqrt(variance)
        normalizer = (float(logsumexp(log_weights+log_ndtr(means/sigma)))
                      if self.support == "positive" else 0.)
        return ConditionalDensity(value, means, sigma, log_weights, self.support, normalizer)


def _quadrant_mass(kde):
    # Deterministic 1-D integrations per Gaussian kernel, in standardized Y.
    # +/-10 SD tails are below 1e-22. No random bivariate CDF integration.
    covariance = kde.covariance
    sx, sy = math.sqrt(covariance[0, 0]), math.sqrt(covariance[1, 1])
    correlation = covariance[0, 1]/(sx*sy)
    denominator = math.sqrt(max(np.finfo(float).eps, 1.-correlation*correlation))
    masses = []
    for mx, my in kde.dataset.T:
        lower = max(-10., -my/sy)
        value = quad(lambda t: math.exp(-.5*t*t)/math.sqrt(2*math.pi)
                     *ndtr((mx/sx+correlation*t)/denominator),
                     lower, 10., epsabs=1e-11, epsrel=1e-10, limit=200)[0]
        masses.append(value)
    return float(np.dot(kde.weights, masses))


@dataclass
class DensityFit:
    models: dict
    samples: pd.DataFrame
    metadata: dict

    def conditionals(self, p21_fixed=None):
        ranges = self.samples.groupby("family").P21.agg(["min", "max"])
        lower, upper = float(ranges["min"].max()), float(ranges["max"].min())
        if lower >= upper:
            raise ValueError("The observed P21 ranges of the families do not overlap.")
        if p21_fixed is None:
            subset = self.samples.loc[self.samples.P21.between(lower, upper)].sort_values("P21")
            weights = subset.weight.to_numpy(float)
            cumulative = np.cumsum(weights)
            midpoint = .5*weights.sum()
            index = min(int(np.searchsorted(cumulative, midpoint)), len(subset)-1)
            p21_fixed = float(subset.P21.iloc[index])
            if index+1 < len(subset) and np.isclose(cumulative[index], midpoint, rtol=1e-12, atol=1e-14):
                p21_fixed = .5*(p21_fixed+float(subset.P21.iloc[index+1]))
        p21_fixed = float(p21_fixed)
        if not lower <= p21_fixed <= upper:
            raise ValueError("p21_fixed must lie within every family's observed P21 range.")
        return {family: model.conditional(p21_fixed) for family, model in self.models.items()}

    def summary(self):
        rows = []
        for family, model in self.models.items():
            samples = self.samples.loc[self.samples.family == family]
            rows.append({"family": family, "rows": len(samples),
                         "windows": samples._window_key.nunique(),
                         "aggregation": self.metadata["aggregation"],
                         "weighting": self.metadata["weighting"], "support": model.support,
                         "positive_mass_before_normalizing": model.positive_mass,
                         "kde_factor": model.kde.factor,
                         "P32_min": samples.P32.min(), "P32_max": samples.P32.max(),
                         "P21_min": samples.P21.min(), "P21_max": samples.P21.max()})
        return pd.DataFrame(rows)

    def export(self, directory, *, p21_fixed=None):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        paths = [root/"density_samples.csv", root/"density_summary.csv", root/"density_fit.txt"]
        self.samples.to_csv(paths[0], index=False)
        self.summary().to_csv(paths[1], index=False)
        paths[2].write_text(repr(self.metadata)+"\n", encoding="utf-8")
        if p21_fixed is not None:
            rows = []
            for family, model in self.conditionals(p21_fixed).items():
                grid, pdf = model.grid()
                rows.append(pd.DataFrame({"family": family, "P21_fixed": p21_fixed,
                                           "P32": grid, "conditional_pdf": pdf}))
            path = root/"conditional_pdfs.csv"
            pd.concat(rows, ignore_index=True).to_csv(path, index=False)
            paths.append(path)
        return paths


def _resolve_column(frame, requested, alternatives):
    if requested is not None:
        if requested not in frame:
            raise ValueError(f"Missing column: {requested}.")
        return requested
    candidates = [name for name in alternatives if name in frame]
    if not candidates:
        raise ValueError(f"Specify a column; expected one of {alternatives}.")
    return candidates[0]


def fit_joint_density(results, *, aggregation="mean", family_col=None, p32_col=None,
                      p21_col=None, window_cols=None, case_id=None, families=None,
                      bw_method="scott", support="positive", weighting="window"):
    """Fit one joint KDE per family, at ONE selected analysis scale.

    Accepts DFNResult or a DataFrame. mean uses one mean P32 per window;
    all retains every realization. The default weighting gives each window
    equal total weight even when realization counts differ. P21_observed is
    preferred over P21_sim. Repeated-window columns must be available when a
    realization column is present. No automatic scale pooling or random jitter.

    positive truncates and normalizes Gaussian KDEs on the positive quadrant;
    this is support normalization, not a claim of boundary-bias correction.
    unbounded exposes the original Gaussian KDE over R2.
    """
    from .dfn import DFNResult
    if aggregation not in ("mean", "all"):
        raise ValueError("aggregation must be mean or all.")
    if support not in ("positive", "unbounded"):
        raise ValueError("support must be positive or unbounded.")
    if weighting not in ("window", "realization"):
        raise ValueError("weighting must be window or realization.")
    if isinstance(results, DFNResult):
        raw = results.realizations.copy()
    elif isinstance(results, pd.DataFrame):
        raw = results.copy()
    else:
        raise TypeError("results must be DFNResult or DataFrame.")
    if raw.empty:
        raise ValueError("No samples for density fitting.")
    if "case_id" in raw:
        if case_id is not None:
            raw = raw.loc[raw.case_id == case_id].copy()
        if raw.empty:
            raise ValueError("Requested case_id has no samples.")
        if raw.case_id.nunique(dropna=False) > 1:
            raise ValueError("Select case_id explicitly; mixing different window sizes is not automatic.")
    elif case_id is not None:
        raise ValueError("case_id was supplied but is absent from the input.")
    family_col = _resolve_column(raw, family_col, ("family", "Family"))
    p32_col = _resolve_column(raw, p32_col, ("P32", "P32_mean", "P32m"))
    p21_col = _resolve_column(raw, p21_col, ("P21_observed", "P21"))
    if aggregation == "all" and p32_col in ("P32_mean", "P32m"):
        raise ValueError("Individual realizations cannot be recovered from a column of P32 means.")
    if raw[family_col].isna().any():
        raise ValueError("Family identifiers cannot be missing.")
    if families is not None:
        selected = list(families)
        if not selected or not set(selected) <= set(raw[family_col]):
            raise ValueError("families must select existing names.")
        raw = raw.loc[raw[family_col].isin(selected)].copy()
    if window_cols is None:
        window_cols = [key for key in ("dataset", "case_id", "window_id") if key in raw]
        if "window_id" not in raw:
            if "realization" in raw:
                raise ValueError("Realization-level inputs need window_cols or window_id.")
            window_cols = []
    else:
        window_cols = list(window_cols)
        if any(key not in raw for key in window_cols):
            raise ValueError("window_cols contains missing input columns.")
        window_cols = [key for key in window_cols if key != family_col]
    if "realization" in raw:
        keys = [family_col]+window_cols+["realization"]
        if raw.realization.isna().any() or raw.duplicated(keys).any():
            raise ValueError("Missing or duplicate realization identifiers within a window.")
    clean = pd.DataFrame({"family": raw[family_col].astype(str),
                          "P32": pd.to_numeric(raw[p32_col], errors="raise"),
                          "P21": pd.to_numeric(raw[p21_col], errors="raise")})
    for key in window_cols:
        clean[key] = raw[key]
    if not np.isfinite(clean[["P32", "P21"]]).all().all() or (clean[["P32", "P21"]] < 0).any().any():
        raise ValueError("Intensity samples must be finite and nonnegative; no silent dropping.")
    if window_cols and clean[window_cols].isna().any().any():
        raise ValueError("Window identifiers cannot be missing.")
    if window_cols:
        clean["_window_key"] = clean.groupby(["family"]+window_cols, sort=False).ngroup()
    else:
        clean["_window_key"] = np.arange(len(clean))
    sizes = clean.groupby("_window_key", sort=False).P32.transform("size")
    if p21_col != "P21_sim":
        spread = clean.groupby("_window_key").P21.agg(["min", "max"])
        if ((spread["max"]-spread["min"]) > 1e-10*np.maximum(1., spread["max"])).any():
            raise ValueError("Observed P21 is inconsistent within repeated-window rows.")
    if aggregation == "mean":
        # Geometry/group identifiers remain inspectable alongside each mean.
        operations = {"P32": "mean", "P21": "mean", "family": "first"}
        operations.update({key: "first" for key in window_cols})
        samples = clean.groupby("_window_key", sort=False).agg(operations).reset_index()
        samples["weight"] = 1.
        samples["n_realizations"] = clean.groupby("_window_key", sort=False).size().to_numpy()
    else:
        samples = clean.copy()
        if "realization" in raw:
            samples["realization"] = raw["realization"]
        samples["weight"] = 1./sizes if weighting == "window" else 1.
        samples["n_realizations"] = sizes
    models = {}
    for family, frame in samples.groupby("family", sort=False):
        data = frame[["P32", "P21"]].to_numpy(float).T
        if len(frame) < 3 or np.linalg.matrix_rank(data-data.mean(axis=1, keepdims=True)) < 2:
            raise ValueError(f"Family {family!r} needs at least three non-collinear samples; no artificial jitter is added.")
        try:
            kde = gaussian_kde(data, bw_method=bw_method, weights=frame.weight.to_numpy(float))
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"Singular joint KDE for {family!r}; review sample variability.") from exc
        mass = _quadrant_mass(kde) if support == "positive" else 1.
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError(f"Cannot normalize positive-support KDE for {family!r}.")
        models[family] = JointDensity(family, kde, support, mass)
    units = list(raw.length_unit.unique()) if "length_unit" in raw else []
    if len(units) > 1:
        raise ValueError("Do not pool different coordinate units.")
    metadata = {"aggregation": aggregation,
                "weighting": "window" if aggregation == "mean" else weighting, "support": support,
                "p32_source": p32_col, "p21_source": p21_col,
                "case_id": raw.case_id.iloc[0] if "case_id" in raw else None,
                "window_cols": window_cols, "bw_method": str(bw_method),
                "length_unit": units[0] if units else None,
                "normalization": "positive quadrant" if support == "positive" else "whole real plane",
                "interpretation": "descriptive KDE; repetitions are not independent spatial observations",
                "p32_grain": "window mean" if aggregation == "mean" else "individual realization"}
    return DensityFit(models, samples.reset_index(drop=True), metadata)

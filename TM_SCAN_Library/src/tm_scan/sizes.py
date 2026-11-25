"""Lognormal diameters from explicit moments or trace stereological moments."""
from dataclasses import dataclass, field
from pathlib import Path
import math
import numpy as np
from scipy.optimize import least_squares
from scipy.special import log_ndtr
from scipy.stats import lognorm
from .models import positive


def _log_interval(lower, upper):
    if lower >= upper:
        return -np.inf
    if lower > 0:
        large, small = log_ndtr(-lower), log_ndtr(-upper)
    else:
        large, small = log_ndtr(upper), log_ndtr(lower)
    return float(large+np.log(-np.expm1(small-large)))


def _log_moment(order, log_mean, log_std, lower, upper):
    if log_std == 0:
        diameter = math.exp(log_mean)
        if not lower <= diameter <= upper:
            raise ValueError("Constant diameter lies outside truncation limits.")
        return order*log_mean
    low = -np.inf if lower == 0 else (math.log(lower)-log_mean)/log_std
    high = np.inf if np.isinf(upper) else (math.log(upper)-log_mean)/log_std
    mass = _log_interval(low, high)
    if not np.isfinite(mass):
        raise ValueError("Truncation probability is numerically zero.")
    shifted = _log_interval(low-order*log_std, high-order*log_std)
    return order*log_mean+.5*order*order*log_std*log_std+shifted-mass


@dataclass(frozen=True)
class SizeDistribution:
    """Arithmetic diameter moments of the PARENT lognormal before truncation.

    The realized diameter moments can be queried with moment(). Bounds are
    diameter bounds, not radius bounds. The default has no finite truncation.
    """
    diameter_mean: float
    diameter_std: float
    diameter_min: float = 0.0
    diameter_max: float = math.inf
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        positive(self.diameter_mean, "diameter_mean")
        if not np.isfinite(self.diameter_std) or self.diameter_std < 0:
            raise ValueError("diameter_std must be finite and nonnegative.")
        if not np.isfinite(self.diameter_min) or self.diameter_min < 0:
            raise ValueError("diameter_min must be finite and nonnegative.")
        if math.isnan(self.diameter_max) or self.diameter_max <= self.diameter_min:
            raise ValueError("diameter_max must exceed diameter_min.")
        # Validate the actual truncated support before a rejection sampler runs.
        self.log_moment(0)
        if self.log_std > 0:
            low, high = self._bounds_z()
            if _log_interval(low, high) < math.log(1e-12):
                raise ValueError("Truncation probability is below 1e-12; revise bounds or parameters.")

    @property
    def log_std(self):
        return math.sqrt(math.log1p((self.diameter_std/self.diameter_mean)**2))

    @property
    def log_mean(self):
        return math.log(self.diameter_mean)-.5*self.log_std**2

    def _bounds_z(self):
        low = -np.inf if self.diameter_min == 0 else (math.log(self.diameter_min)-self.log_mean)/self.log_std
        high = np.inf if np.isinf(self.diameter_max) else (math.log(self.diameter_max)-self.log_mean)/self.log_std
        return low, high

    def log_moment(self, order):
        return _log_moment(order, self.log_mean, self.log_std, self.diameter_min, self.diameter_max)

    def moment(self, order):
        if not np.isfinite(order) or order < 0:
            raise ValueError("Moment order must be finite and nonnegative.")
        return math.exp(self.log_moment(order))

    @property
    def realized_mean(self):
        return self.moment(1)

    @property
    def realized_std(self):
        return math.sqrt(max(0., self.moment(2)-self.moment(1)**2))

    def implied_trace_moments(self):
        mean = math.pi/4*math.exp(self.log_moment(2)-self.log_moment(1))
        second = 2/3*math.exp(self.log_moment(3)-self.log_moment(1))
        return mean, math.sqrt(max(0., second-mean*mean))

    def pdf(self, diameters):
        if self.log_std == 0:
            raise ValueError("A constant diameter is a point mass, without a continuous PDF.")
        x = np.asarray(diameters, float)
        low, high = self._bounds_z()
        normalizer = _log_interval(low, high)
        values = np.exp(lognorm.logpdf(x, self.log_std, scale=math.exp(self.log_mean))-normalizer)
        return np.where((x >= self.diameter_min) & (x <= self.diameter_max), values, 0.)

    def rvs(self, size, random_state=None):
        from ._dfn_core import _sample_truncated_lognormal
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 0:
            raise ValueError("size must be a nonnegative integer.")
        rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
        return _sample_truncated_lognormal(rng, size, self.log_mean, self.log_std,
                                            self.diameter_min, self.diameter_max)

    def summary(self):
        mu, sigma = self.implied_trace_moments()
        return {"distribution": "lognormal", "diameter_mean": self.diameter_mean,
                "diameter_std": self.diameter_std, "diameter_min": self.diameter_min,
                "diameter_max": self.diameter_max, "log_mean": self.log_mean, "log_std": self.log_std,
                "realized_diameter_mean": self.realized_mean, "realized_diameter_std": self.realized_std,
                "implied_trace_mean": mu, "implied_trace_std": sigma, **self.diagnostics}

    def export(self, directory):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        path = root/"size_fit.txt"
        path.write_text(repr(self.summary())+"\n", encoding="utf-8")
        return [path]


def fit_from_mu(mu, *, trace_lengths=None, trace_cv=None, trace_std=None,
                distribution="lognormal", diameter_min=0., diameter_max=math.inf, ddof=1):
    """Infer diameters for ONE family using a user-selected representative mu.

    Exactly one of trace_lengths, trace_cv or trace_std supplies the second
    trace moment. With measured lengths, CV = sample SD / measured mean;
    corrected trace SD = CV * mu. Transferring this CV is an explicit assumption.

    Uses E[l] = pi E[D^2]/(4 E[D]) and E[l^2] = 2 E[D^3]/(3 E[D]).
    With finite bounds, solves these relations using TRUNCATED moments rather
    than inferring untruncated moments and silently clipping the samples.
    """
    mu = positive(mu, "mu")
    if distribution != "lognormal":
        raise ValueError("This version implements the approved lognormal conversion only.")
    if sum(value is not None for value in (trace_lengths, trace_cv, trace_std)) != 1:
        raise ValueError("Provide exactly one of trace_lengths, trace_cv or trace_std.")
    if isinstance(ddof, bool) or not isinstance(ddof, (int, np.integer)) or ddof < 0:
        raise ValueError("ddof must be a nonnegative integer.")
    if not np.isfinite(diameter_min) or diameter_min < 0 or math.isnan(diameter_max) or diameter_max <= diameter_min:
        raise ValueError("Invalid diameter truncation interval.")
    diagnostics = {"source": "trace stereology", "trace_mean": mu,
                   "representative_mu_selection": "provided by user; no automatic scale selection"}
    if trace_lengths is not None:
        values = np.asarray(trace_lengths, float)
        if values.ndim != 1 or len(values) <= max(1, ddof) or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Measured trace lengths must be finite, positive and sufficiently numerous.")
        measured_mean, measured_std = float(values.mean()), float(values.std(ddof=ddof))
        cv = measured_std/measured_mean
        diagnostics.update(n_measured_traces=len(values), measured_trace_mean=measured_mean,
                           measured_trace_std=measured_std, ddof=int(ddof),
                           cv_assumption="CV of corrected traces equals CV of measured biased traces")
    elif trace_cv is not None:
        cv = float(trace_cv)
        diagnostics["cv_assumption"] = "trace CV supplied by user"
    else:
        cv = float(trace_std)/mu
        diagnostics["cv_assumption"] = "corrected trace SD supplied by user"
    if not np.isfinite(cv) or cv < 0:
        raise ValueError("Trace coefficient of variation must be finite and nonnegative.")
    sigma = cv*mu
    q = 3*math.pi**2/32*(1+cv*cv)
    if q < 1.-1e-12:
        raise ValueError("Inadmissible trace moments: q < 1; no lognormal diameter solution. Do not clip q silently.")
    q = max(q, 1.)
    mean_d = 4*mu/(math.pi*q)
    std_d = mean_d*math.sqrt(q-1.)
    diagnostics.update(trace_cv=cv, corrected_trace_std=sigma, q=q,
                       moment_relation="area sampling of circular fractures; size-independent orientation")
    if diameter_min == 0 and math.isinf(diameter_max):
        return SizeDistribution(mean_d, std_d, diagnostics=diagnostics)
    if std_d == 0:
        return SizeDistribution(mean_d, 0., diameter_min, diameter_max, diagnostics)
    initial_std = math.sqrt(math.log1p((std_d/mean_d)**2))
    initial_mean = math.log(mean_d)-.5*initial_std**2
    target_second = mu*mu+sigma*sigma
    def residual(parameters):
        log_mean, log_std = parameters[0], math.exp(parameters[1])
        try:
            moments = [_log_moment(i, log_mean, log_std, diameter_min, diameter_max) for i in (1, 2, 3)]
            if not np.isfinite(moments).all():
                return np.array([1e6, 1e6])
            return np.array([math.log(math.pi/4)+moments[1]-moments[0]-math.log(mu),
                             math.log(2/3)+moments[2]-moments[0]-math.log(target_second)])
        except (ValueError, OverflowError):
            return np.array([1e6, 1e6])
    fit = least_squares(residual, [initial_mean, math.log(initial_std)],
                         bounds=([initial_mean-20, -13.8], [initial_mean+20, math.log(5.)]),
                         ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=2000)
    if not fit.success or np.max(np.abs(residual(fit.x))) > 1e-7:
        raise ValueError("Trace moments cannot be reproduced by a lognormal with these truncation limits.")
    log_mean, log_std = fit.x[0], math.exp(fit.x[1])
    parent_mean = math.exp(log_mean+.5*log_std*log_std)
    parent_std = parent_mean*math.sqrt(math.expm1(log_std*log_std))
    diagnostics.update(truncation_fit="two truncated stereological moments",
                       max_log_moment_residual=float(np.max(np.abs(residual(fit.x)))))
    return SizeDistribution(parent_mean, parent_std, diameter_min, diameter_max, diagnostics)

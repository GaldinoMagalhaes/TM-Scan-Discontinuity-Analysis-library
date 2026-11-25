"""Axial orientation models on S2 and empirical resampling of plane axes."""
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy.integrate import quad, quad_vec
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.special import i0e, i1e, logsumexp
from scipy.spatial.transform import Rotation


def _unit_vectors(values):
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("Expected a nonempty (n, 3) array of poles.")
    norms = np.linalg.norm(points, axis=1)
    if not np.isfinite(points).all() or (norms <= 0).any():
        raise ValueError("Poles must be finite, nonzero vectors.")
    return np.ascontiguousarray(points/norms[:, None])


def _log_constant(kappa):
    if kappa < 1e-4:
        return -math.log(4*math.pi)-kappa*kappa/6+kappa**4/180
    return math.log(kappa)-math.log(2*math.pi)-kappa-math.log(-math.expm1(-2*kappa))


def _log_cosh(values):
    return np.logaddexp(values, -values)-math.log(2.)


def _angles_to_poles(dips, directions, convention):
    if convention not in ("modeling", "geological"):
        raise ValueError("convention must be 'modeling' or 'geological'.")
    dip, direction = np.asarray(dips, float), np.asarray(directions, float)
    if not np.isfinite(dip).all() or not np.isfinite(direction).all():
        raise ValueError("Orientation angles must be finite.")
    if ((dip < 0) | (dip > 90)).any() or ((direction < 0) | (direction > 360)).any():
        raise ValueError("dip must be in [0,90], dip_direction in [0,360] degrees.")
    dip, direction = np.deg2rad(dip), np.deg2rad(direction)
    sign = -1. if convention == "modeling" else 1.
    return np.column_stack([sign*np.sin(dip)*np.sin(direction),
                            sign*np.sin(dip)*np.cos(direction), np.cos(dip)])


def _principal_axis(points):
    values, vectors = np.linalg.eigh(points.T@points/len(points))
    axis = vectors[:, -1]
    if axis[2] < 0 or (abs(axis[2]) < 1e-12 and axis[np.argmax(abs(axis))] < 0):
        axis = -axis
    return axis, values


def _fit_axial_fisher(points):
    axis, eigenvalues = _principal_axis(points)
    if eigenvalues[-1] > 1.-1e-12:
        raise ValueError("Identical plane axes have unbounded Fisher concentration; define kappa manually.")
    aligned = points*np.where(points@axis < 0, -1., 1.)[:, None]
    resultant = np.linalg.norm(aligned.mean(axis=0))
    initial = min(1000., max(1., resultant*(3-resultant*resultant)/max(1e-6, 1-resultant*resultant)))
    def objective(vector):
        kappa = float(np.linalg.norm(vector))
        dots = points@vector
        value = -(_log_constant(kappa)+_log_cosh(dots).mean())
        if kappa < 1e-4:
            radial = -1/3+kappa*kappa/45
        else:
            radial = (1/kappa-1/math.tanh(kappa))/kappa
        gradient = -(radial*vector+(np.tanh(dots)[:, None]*points).mean(axis=0))
        return value, gradient
    result = minimize(objective, initial*axis, jac=True, method="L-BFGS-B",
                      options={"maxiter": 1000, "ftol": 1e-13, "gtol": 1e-9})
    if not result.success and np.linalg.norm(result.jac) > 1e-6:
        raise ValueError(f"Axial Fisher fit failed: {result.message}")
    kappa = float(np.linalg.norm(result.x))
    if -result.fun <= -math.log(4*math.pi)+1e-9:
        return axis, 0., eigenvalues
    fitted = result.x/kappa
    if fitted@axis < 0:
        fitted = -fitted
    return fitted, kappa, eigenvalues


def _fit_kernel_kappa(points):
    # Leave-one-out likelihood on plane axes; n and -n are equivalent.
    count = len(points)
    def loss(log_kappa):
        kappa = float(np.exp(log_kappa))
        total = 0.
        for start in range(0, count, 128):
            stop = min(start+128, count)
            logs = _log_cosh(kappa*(points[start:stop]@points.T))
            logs[np.arange(stop-start), np.arange(start, stop)] = -np.inf
            total += np.sum(logsumexp(logs, axis=1)-math.log(count-1))
        return -(_log_constant(kappa)+total/count)
    bounds = (math.log(.01), math.log(1e4))
    result = minimize_scalar(loss, bounds=bounds, method="bounded", options={"xatol": 1e-5})
    if not result.success:
        raise ValueError("Spherical KDE bandwidth selection failed.")
    kappa = math.exp(result.x)
    if -result.fun <= -math.log(4*math.pi):
        kappa = 0.
    at_limit = kappa > .99e4
    return kappa, at_limit


def _canonical_method(method):
    if not isinstance(method, str):
        raise ValueError("method must be a string.")
    aliases = {"axial_kde": "spherical_kde", "kde": "spherical_kde",
               "bootstrap": "empirical", "axial_kent": "kent"}
    method = aliases.get(method, method)
    if method not in ("fisher", "spherical_kde", "kent", "bingham", "empirical"):
        raise ValueError("method must be fisher, kent, bingham, spherical_kde, or empirical.")
    return method


def _frame(axes):
    axes = np.asarray(axes, dtype=float)
    if axes.shape != (3, 3) or not np.isfinite(axes).all():
        raise ValueError("axes must be a finite (3,3) matrix with orthonormal columns.")
    if not np.allclose(axes.T@axes, np.eye(3), rtol=0, atol=1e-8):
        raise ValueError("axes must have orthonormal columns.")
    return axes


def _integration_points(scale):
    # Resolve the endpoint layer even for strongly concentrated distributions.
    return sorted({1.-step/max(1., scale) for step in (.1, 1., 10., 100.)
                   if 0. < 1.-step/max(1., scale) < 1.})


@lru_cache(maxsize=128)
def _kent_log_partition_shifted(kappa, beta):
    """log integral[cosh(k*z)*exp(beta*(x*x-y*y))] - k on S2."""
    def integrand(z):
        argument = beta*(1.-z*z)
        return math.exp(float(_log_cosh(kappa*z))-kappa+argument)*i0e(argument)
    integral, error = quad(integrand, 0., 1., epsabs=1e-13, epsrel=2e-10,
                           points=_integration_points(kappa), limit=250)
    if integral <= 0 or error > max(1e-12, 1e-7*integral):
        raise ValueError("Kent normalization quadrature did not converge.")
    return math.log(4*math.pi*integral)


@lru_cache(maxsize=128)
def _bingham_partition(penalty_y, penalty_z):
    """Return log normalizer and second moments in the model's eigenframe."""
    def integrand(z):
        radial = 1.-z*z
        argument = .5*abs(penalty_y-penalty_z)*radial
        bessel = i0e(argument)
        weight = math.exp(-min(penalty_y, penalty_z)*radial)*bessel
        difference = (i1e(argument)/bessel)*np.sign(penalty_z-penalty_y)
        return weight*np.array([1., z*z, .5*radial*(1.+difference),
                                .5*radial*(1.-difference)])
    integrals, error = quad_vec(integrand, 0., 1., epsabs=1e-13, epsrel=2e-10,
                                points=_integration_points(max(penalty_y, penalty_z)), limit=250)
    if integrals[0] <= 0 or error > max(1e-12, 1e-7*integrals[0]):
        raise ValueError("Bingham normalization quadrature did not converge.")
    return math.log(4*math.pi*integrals[0]), tuple(integrals[1:]/integrals[0])


def _sample_bingham(rng, size, penalties):
    """Independent BACG rejection samples (Kent, Ganeiber & Mardia, 2018)."""
    penalties = np.asarray(penalties, dtype=float)
    b = brentq(lambda value: np.sum(1./(value+2*penalties))-1., 1e-12, 3.)
    scales = 1./np.sqrt(1.+2*penalties/b)
    log_bound = -.5*(3.-b)+1.5*math.log(3./b)
    output = np.empty((size, 3))
    filled = 0
    while filled < size:
        count = min(262144, max(32, 2*(size-filled)))
        candidates = rng.normal(size=(count, 3))*scales
        candidates /= np.linalg.norm(candidates, axis=1)[:, None]
        energy = (candidates*candidates)@penalties
        log_accept = -energy+1.5*np.log1p(2*energy/b)-log_bound
        accepted = candidates[np.log(np.maximum(rng.random(count), np.finfo(float).tiny)) < log_accept]
        take = min(len(accepted), size-filled)
        output[filled:filled+take] = accepted[:take]
        filled += take
    return output


def _fit_bingham(points):
    eigenvalues, axes = np.linalg.eigh(points.T@points/len(points))
    eigenvalues, axes = eigenvalues[::-1], axes[:, ::-1]
    if eigenvalues[-1] <= 1e-12:
        raise ValueError("Rank-deficient plane axes have unbounded Bingham concentration; define parameters manually.")
    initial = np.maximum(0., .5/eigenvalues[1:]-.5/eigenvalues[0])
    def objective(penalties):
        log_partition, moments = _bingham_partition(*map(float, penalties))
        return (log_partition+penalties@eigenvalues[1:],
                eigenvalues[1:]-np.asarray(moments[1:]))
    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      bounds=[(0., 1e6)]*2,
                      options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-9})
    if (not result.success and np.linalg.norm(result.jac) > 1e-6) or max(result.x) >= .999e6:
        raise ValueError(f"Bingham fit did not converge to finite interior parameters: {result.message}")
    concentrations = np.r_[0., -result.x]
    order = np.argsort(-concentrations, kind="stable")
    return axes[:, order], concentrations[order], float(-result.fun)


def _fit_kent(points):
    eigenvalues, initial_axes = np.linalg.eigh(points.T@points/len(points))
    eigenvalues, initial_axes = eigenvalues[::-1], initial_axes[:, ::-1]
    if eigenvalues[-1] <= 1e-12:
        raise ValueError("Rank-deficient plane axes do not admit a regular Kent fit; define parameters manually.")
    # A proper frame lets rotations parameterize all three orientation angles.
    if np.linalg.det(initial_axes) < 0:
        initial_axes[:, -1] *= -1
    initial_kappa = min(1e5, .5/eigenvalues[1]+.5/eigenvalues[2])
    initial_ratio = min(.95, max(0., (eigenvalues[1]-eigenvalues[2])/
                                (eigenvalues[1]+eigenvalues[2])))
    def objective(parameters):
        axes = Rotation.from_rotvec(parameters[:3]).as_matrix()@initial_axes
        kappa = math.exp(parameters[3])
        beta = .5*kappa*parameters[4]
        local = points@axes
        log_kernel = _log_cosh(kappa*local[:, 0])-kappa+beta*(local[:, 1]**2-local[:, 2]**2)
        return _kent_log_partition_shifted(kappa, beta)-float(log_kernel.mean())
    bounds = [(-math.pi, math.pi)]*3+[(math.log(1e-5), math.log(1e6)), (0., 1.-1e-6)]
    starts = [np.r_[np.zeros(3), math.log(initial_kappa), initial_ratio]]
    fisher_axis, fisher_kappa, _ = _fit_axial_fisher(points)
    if fisher_kappa > 1e-4:
        aligned_axis = fisher_axis*np.sign(fisher_axis@initial_axes[:, 0])
        rotation, _ = Rotation.align_vectors([aligned_axis], [initial_axes[:, 0]])
        starts.append(np.r_[rotation.as_rotvec(), math.log(fisher_kappa), .01])
    results = [minimize(objective, start, method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": 500, "ftol": 1e-12, "gtol": 2e-6, "maxls": 40})
               for start in starts]
    result = min(results, key=lambda item: item.fun)
    if not result.success and np.linalg.norm(result.jac) > 1e-4:
        raise ValueError(f"Axial Kent fit failed: {result.message}")
    kappa = math.exp(result.x[3])
    beta = .5*kappa*result.x[4]
    if kappa >= .999e6:
        raise ValueError("Kent concentration reached the numerical fitting limit; define parameters manually.")
    axes = Rotation.from_rotvec(result.x[:3]).as_matrix()@initial_axes
    if -result.fun <= -math.log(4*math.pi)+1e-9:
        kappa, beta = 0., 0.
    return axes, kappa, beta, {"fit_method": "axial maximum likelihood (numerical multistart)",
        "log_likelihood": -len(points)*result.fun,
        "ovalness_at_upper_limit": bool(result.x[4] > 1.-2e-6),
        "optimizer_success": bool(result.success)}


@dataclass(frozen=True)
class OrientationDistribution:
    """Axial model on S2; continuous pdf integrates to one over the FULL sphere.

    rvs returns the equivalent upper-hemisphere plane representation. Its
    hemisphere density is twice pdf. No projection-plane Jacobian is needed.
    Empirical models are discrete: use pmf, not pdf. Kent is antipodally
    symmetrized, not a directional Kent law fitted after a global sign choice.
    Frame columns are [mean axis, major dispersion axis, minor dispersion axis].
    """
    method: str
    mean_normal: tuple
    kappa: float
    poles: tuple = ()
    diagnostics: dict = field(default_factory=dict)
    axes: tuple = ()
    concentrations: tuple = ()
    beta: float = 0.
    _log_partition: float = field(default=0., init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "method", _canonical_method(self.method))
        mean = _unit_vectors([self.mean_normal])[0]
        object.__setattr__(self, "mean_normal", tuple(float(x) for x in mean))
        if not np.isfinite(self.kappa) or self.kappa < 0:
            raise ValueError("kappa must be finite and nonnegative.")
        if self.method in ("spherical_kde", "empirical"):
            poles = _unit_vectors(self.poles)
            object.__setattr__(self, "poles", tuple(map(tuple, poles.tolist())))
        if self.method in ("kent", "bingham"):
            axes = _frame(self.axes)
            if not np.isclose(abs(mean@axes[:, 0]), 1., atol=1e-8, rtol=0):
                raise ValueError("mean_normal must coincide with the first frame axis.")
            object.__setattr__(self, "axes", tuple(map(tuple, axes.tolist())))
        if not np.isfinite(self.beta) or self.beta < 0:
            raise ValueError("beta must be finite and nonnegative.")
        object.__setattr__(self, "beta", float(self.beta))
        if self.method == "kent":
            if not ((self.kappa == 0 and self.beta == 0) or 2*self.beta < self.kappa):
                raise ValueError("Kent requires 0 <= 2*beta < kappa, or kappa=beta=0 (uniform).")
            object.__setattr__(self, "_log_partition", _kent_log_partition_shifted(float(self.kappa), float(self.beta)))
        elif self.beta != 0:
            raise ValueError("beta applies only to Kent.")
        if self.method == "bingham":
            concentrations = np.asarray(self.concentrations, dtype=float)
            if (concentrations.shape != (3,) or not np.isfinite(concentrations).all()
                    or concentrations[0] != 0 or (np.diff(concentrations) > 0).any()):
                raise ValueError("Bingham concentrations must be ordered: 0 = z1 >= z2 >= z3.")
            object.__setattr__(self, "concentrations", tuple(map(float, concentrations)))
            object.__setattr__(self, "_log_partition", _bingham_partition(*map(float, -concentrations[1:]))[0])

    @classmethod
    def fisher(cls, dip, dip_direction, kappa, *, convention="modeling"):
        normal = _angles_to_poles([dip], [dip_direction], convention)[0]
        return cls("fisher", tuple(normal), float(kappa),
                   diagnostics={"source": "manual", "angle_convention": convention})

    @classmethod
    def kent(cls, dip, dip_direction, kappa, beta, *, roll_deg=0., convention="modeling"):
        """Manual axial Kent; roll rotates the major axis about the mean pole.

        At zero roll the major axis is cross(global Z, mean), using global X
        instead of Z for |mean.Z| >= 0.9. Positive roll follows the right hand rule.
        """
        if not np.isfinite(roll_deg):
            raise ValueError("roll_deg must be finite.")
        mean = _angles_to_poles([dip], [dip_direction], convention)[0]
        reference = np.array([0., 0., 1.]) if abs(mean[2]) < .9 else np.array([1., 0., 0.])
        major = np.cross(reference, mean)
        major /= np.linalg.norm(major)
        minor = np.cross(mean, major)
        angle = math.radians(roll_deg)
        axes = np.column_stack([mean, math.cos(angle)*major+math.sin(angle)*minor,
                                -math.sin(angle)*major+math.cos(angle)*minor])
        return cls("kent", tuple(mean), float(kappa), axes=tuple(map(tuple, axes)), beta=float(beta),
                   diagnostics={"source": "manual", "angle_convention": convention, "axial": True})

    @classmethod
    def bingham(cls, axes, concentrations):
        """Manual axial Bingham; axes are columns, concentrations ordered 0>=z2>=z3."""
        axes = _frame(axes)
        return cls("bingham", tuple(axes[:, 0]), 0., axes=tuple(map(tuple, axes)),
                   concentrations=tuple(concentrations), diagnostics={"source": "manual", "axial": True})

    def logpdf(self, poles):
        """Log density per unit solid angle over the full sphere."""
        points = _unit_vectors(poles)
        if self.method == "empirical":
            raise ValueError("Empirical orientations are discrete point masses; use pmf or rvs, not pdf.")
        if self.method in ("kent", "bingham"):
            local = points@np.asarray(self.axes)
            if self.method == "kent":
                return (_log_cosh(self.kappa*local[:, 0])-self.kappa
                        +self.beta*(local[:, 1]**2-local[:, 2]**2)-self._log_partition)
            return (local*local)@np.asarray(self.concentrations)-self._log_partition
        kernels = np.asarray(self.poles if self.method == "spherical_kde" else [self.mean_normal])
        output = np.empty(len(points))
        for start in range(0, len(points), 256):
            dots = points[start:start+256]@kernels.T
            output[start:start+256] = (_log_constant(self.kappa)
                +logsumexp(_log_cosh(self.kappa*dots), axis=1)-math.log(len(kernels)))
        return output

    def pdf(self, poles):
        return np.exp(self.logpdf(poles))

    def pmf(self, poles):
        """Empirical mass per PLANE AXIS, combining repeated and antipodal poles."""
        if self.method != "empirical":
            raise ValueError("pmf applies only to empirical orientations; use pdf for continuous models.")
        points, support = _unit_vectors(poles), np.asarray(self.poles)
        output = np.empty(len(points))
        for start in range(0, len(points), 128):
            block = points[start:start+128, None, :]
            distance = np.minimum(np.linalg.norm(block-support, axis=2), np.linalg.norm(block+support, axis=2))
            output[start:start+128] = (distance <= 1e-10).mean(axis=1)
        return output

    def rvs(self, size, random_state=None):
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 0:
            raise ValueError("size must be a nonnegative integer.")
        rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
        if self.method == "fisher":
            from ._dfn_core import _sample_fisher_normals
            return _sample_fisher_normals(rng, size, np.array(self.mean_normal), self.kappa)
        if size == 0:
            return np.empty((0, 3))
        if self.method == "empirical":
            normals = np.asarray(self.poles)[rng.integers(0, len(self.poles), size=size)].copy()
            normals[normals[:, 2] < 0] *= -1
            return normals
        if self.method in ("kent", "bingham"):
            if self.method == "bingham":
                local = _sample_bingham(rng, size, -np.asarray(self.concentrations))
            else:
                # cosh(k*z)*exp(beta*d-k) <= exp[-k*(1-z*z)/2+beta*d].
                # A Bingham envelope gives independent exact rejection samples.
                penalties = np.array([0., .5*self.kappa-self.beta, .5*self.kappa+self.beta])
                local = np.empty((size, 3))
                filled = 0
                while filled < size:
                    candidates = _sample_bingham(rng, min(262144, max(32, 3*(size-filled))), penalties)
                    z = candidates[:, 0]
                    log_accept = _log_cosh(self.kappa*z)-self.kappa+.5*self.kappa*(1.-z*z)
                    accepted = candidates[np.log(np.maximum(rng.random(len(z)), np.finfo(float).tiny)) < log_accept]
                    take = min(len(accepted), size-filled)
                    local[filled:filled+take] = accepted[:take]
                    filled += take
            normals = local@np.asarray(self.axes).T
            normals[normals[:, 2] < 0] *= -1
            return normals
        means = np.asarray(self.poles)[rng.integers(0, len(self.poles), size=size)]
        uniform = rng.random(size)
        if self.kappa <= 1e-8:
            axial = 2*uniform-1
        else:
            axial = 1+np.log(uniform+(1-uniform)*math.exp(-2*self.kappa))/self.kappa
        azimuth = 2*math.pi*rng.random(size)
        references = np.tile([0., 0., 1.], (size, 1))
        references[np.abs(means[:, 2]) >= .9] = [1., 0., 0.]
        first = np.cross(references, means)
        first /= np.linalg.norm(first, axis=1)[:, None]
        second = np.cross(means, first)
        radial = np.sqrt(np.maximum(0., 1-axial*axial))
        normals = axial[:, None]*means+radial[:, None]*(np.cos(azimuth)[:, None]*first+np.sin(azimuth)[:, None]*second)
        normals[normals[:, 2] < 0] *= -1
        return normals

    def summary(self):
        result = {"method": self.method, "mean_normal": self.mean_normal,
                "mean_angles_modeling": self.mean_angles(convention="modeling"),
                "kappa": self.kappa, "kernel_count": len(self.poles), **self.diagnostics}
        if self.method in ("kent", "bingham"):
            result["axes_columns"] = self.axes
        if self.method == "kent":
            result.update(beta=self.beta, axial=True)
        if self.method == "bingham":
            result.update(concentrations=self.concentrations, axial=True,
                          representative_axis_note="Not unique for uniform or exact girdle distributions.")
        if self.method == "empirical":
            result.update(n_observations=len(self.poles), measure="discrete plane-axis probability mass")
        return result

    def mean_angles(self, *, convention="modeling"):
        """Representative axis as dip/dip-direction in an explicit convention.

        For a KDE this axis is a summary, not a replacement for its multimodal PDF.
        Use convention='geological' for analyze(..., family_orientations=...).
        """
        if convention not in ("modeling", "geological"):
            raise ValueError("convention must be modeling or geological.")
        normal = np.array(self.mean_normal)
        if normal[2] < 0:
            normal = -normal
        sign = -1. if convention == "modeling" else 1.
        direction = (math.degrees(math.atan2(sign*normal[0], sign*normal[1])) % 360
                     if np.linalg.norm(normal[:2]) > 1e-12 else 0.)
        return {"dip": math.degrees(math.acos(np.clip(normal[2], -1., 1.))),
                "dip_dir": direction}

    def export(self, directory):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        paths = [root/"orientation_fit.txt"]
        paths[0].write_text(repr(self.summary())+"\n", encoding="utf-8")
        if self.poles:
            path = root/"orientation_poles.csv"
            pd.DataFrame(self.poles, columns=["Nx", "Ny", "Nz"]).to_csv(path, index=False)
            paths.append(path)
        return paths


def fit_from_planes(planes, *, method="fisher", dip_col="dip", direction_col="dip_direction",
                    convention="modeling", bandwidth_deg=None):
    """Fit ONE already-identified family from dip/dip-direction or XYZ poles.

    DataFrames use Nx/Ny/Nz when all three exist, otherwise the named angle
    columns. Arrays with two columns hold angles; three columns hold poles.
    Axial Fisher uses maximum likelihood of C(k)*cosh(k*mu.n), without
    averaging antipodal directions. KDE bandwidth is selected by LOO likelihood
    unless bandwidth_deg is explicit (kappa = 1 / radians(bandwidth_deg)**2).
    method selects fisher (default), kent (axial), bingham, spherical_kde
    (alias axial_kde), or empirical (alias bootstrap).
    Empirical resampling is discrete, with no continuous pdf or smoothing.
    """
    method = _canonical_method(method)
    if bandwidth_deg is not None and method != "spherical_kde":
        raise ValueError("bandwidth_deg applies only to spherical_kde.")
    if isinstance(planes, pd.DataFrame):
        if {"Nx", "Ny", "Nz"} <= set(planes):
            points = planes[["Nx", "Ny", "Nz"]].to_numpy(float)
            representation = "XYZ poles"
        else:
            if not {dip_col, direction_col} <= set(planes):
                raise ValueError("Provide Nx/Ny/Nz or dip/dip_direction columns.")
            points = _angles_to_poles(planes[dip_col], planes[direction_col], convention)
            representation = "dip/dip_direction"
    else:
        raw = np.asarray(planes, dtype=float)
        if raw.ndim != 2 or raw.shape[1] not in (2, 3):
            raise ValueError("planes must contain (dip, dip_direction) pairs or XYZ poles.")
        points = _angles_to_poles(raw[:, 0], raw[:, 1], convention) if raw.shape[1] == 2 else raw
        representation = "dip/dip_direction" if raw.shape[1] == 2 else "XYZ poles"
    points = _unit_vectors(points)
    if len(points) < 3 and method != "empirical":
        raise ValueError("At least three planes are required; define parameters manually otherwise.")
    axis, eigenvalues = _principal_axis(points)
    diagnostics = {"source": "planes", "n_planes": len(points), "input_representation": representation,
                   "angle_convention": convention if representation == "dip/dip_direction" else "not applied (XYZ poles)",
                   "tensor_eigenvalues": eigenvalues.tolist(), "axial": True}
    if method == "empirical":
        diagnostics.update(fit_method="empirical resampling with replacement", smoothing=False)
        return OrientationDistribution("empirical", tuple(axis), 0.,
                                       tuple(map(tuple, points.tolist())), diagnostics)
    if method == "bingham":
        axes, concentrations, log_likelihood = _fit_bingham(points)
        diagnostics.update(fit_method="axial maximum likelihood", log_likelihood=len(points)*log_likelihood)
        return OrientationDistribution("bingham", tuple(axes[:, 0]), 0., diagnostics=diagnostics,
                                       axes=tuple(map(tuple, axes)), concentrations=tuple(concentrations))
    if method == "kent":
        axes, kappa, beta, fit_diagnostics = _fit_kent(points)
        diagnostics.update(fit_diagnostics)
        return OrientationDistribution("kent", tuple(axes[:, 0]), kappa, diagnostics=diagnostics,
                                       axes=tuple(map(tuple, axes)), beta=beta)
    if method == "fisher":
        axis, kappa, _ = _fit_axial_fisher(points)
        diagnostics["fit_method"] = "axial maximum likelihood"
        return OrientationDistribution("fisher", tuple(axis), kappa, diagnostics=diagnostics)
    if bandwidth_deg is None:
        kappa, at_limit = _fit_kernel_kappa(points)
        diagnostics.update(bandwidth_method="leave-one-out likelihood", kernel_kappa_at_upper_limit=at_limit)
    else:
        bandwidth = float(bandwidth_deg)
        if not np.isfinite(bandwidth) or not 0 < bandwidth <= 180:
            raise ValueError("bandwidth_deg must lie in (0,180].")
        kappa = 1/math.radians(bandwidth)**2
        diagnostics.update(bandwidth_method="manual angular scale", bandwidth_deg=bandwidth)
    return OrientationDistribution("spherical_kde", tuple(axis), kappa,
                                   tuple(map(tuple, points.tolist())), diagnostics)

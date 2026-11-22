"""Segment-length diagnostics (descriptive KS ranking)."""
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import kstest, lognorm, expon, weibull_min, gamma
from .geometry import _dist_from_cols

def build_segment_lengths(traces_data, *, inplace=False):
    """
    Return 3D endpoint distances, ignoring intermediate trace vertices.
    """
    out_frames = []
    dist_by_family = {}
    for fam, df in traces_data.items():
        if isinstance(df, gpd.GeoDataFrame):
            dists = [_dist_from_cols(line.coords[0], line.coords[-1])
                     for geometry in df.geometry
                     for line in (geometry.geoms if geometry.geom_type == "MultiLineString" else [geometry])]
        else:
            dists = [_dist_from_cols(p0, p1) for p0, p1 in zip(df['X0'], df['X1'])]
        dist_by_family[fam] = np.asarray(dists, dtype=float)
        if inplace and len(dists) == len(df):
            traces_data[fam] = df.assign(dist=dists)
        elif inplace:
            raise ValueError("inplace lengths require one LineString per trace row.")
        out_frames.append(pd.DataFrame({
            'family': fam,
            'seg_idx': np.arange(len(dists), dtype=int),
            'dist': dists
        }))
    lengths_df = pd.concat(out_frames, ignore_index=True)
    return lengths_df, dist_by_family


def ks_gof_by_family(dist_by_family, *, fix_loc0=True, min_positive=1e-12):
    """Performs a Kolmogorov-Smirnov goodness-of-fit test."""
    rows = []
    dist_map = {"lognormal": lognorm, "exponential": expon, "weibull": weibull_min, "gamma": gamma}
    for fam, arr in dist_by_family.items():
        x = np.asarray(arr, float); x = x[np.isfinite(x) & (x > min_positive)]; n = x.size
        if n == 0:
            for name in dist_map:
                rows.append({"family": fam, "dist": name, "D": np.nan, "pvalue": np.nan, "params": None, "n": 0, "note": "no data > 0"})
            continue
        fitkw = {"floc": 0} if fix_loc0 else {}
        for name, dist in dist_map.items():
            try:
                params = dist.fit(x, **fitkw)
                D, p = kstest(x, dist.cdf, args=params)
                rows.append({"family": fam, "dist": name, "D": float(D), "pvalue": float(p), "params": tuple(map(float, params)), "n": int(n)})
            except Exception as e:
                rows.append({"family": fam, "dist": name, "D": np.nan, "pvalue": np.nan, "params": None, "n": int(n), "note": f"error: {e}"})
    results_df = pd.DataFrame(rows)
    tmp = results_df.copy()
    tmp["_p"] = tmp["pvalue"].astype(float).fillna(-np.inf)
    tmp["_D"] = tmp["D"].astype(float).fillna(np.inf)
    best_df = (tmp.sort_values(["family", "_p", "_D"], ascending=[True, False, True]).groupby("family", as_index=False).head(1).drop(columns=["_p", "_D"]))
    return results_df, best_df




def trace_length_summary(dataset):
    """Per family: trace count, mean, unbiased SD (ddof=1), CoV and best KS distribution.

    Lengths are 3D endpoint distances of the MEASURED (biased) traces.
    """
    _, lengths = build_segment_lengths(dataset.traces)
    _, best = ks_gof_by_family(lengths)
    best = best.set_index("family")
    rows = []
    for family, values in lengths.items():
        values = np.asarray(values, float)
        rows.append({"family": family, "n_traces": len(values), "mean": values.mean(),
                     "std": values.std(ddof=1), "cov": values.std(ddof=1) / values.mean(),
                     "min": values.min(), "max": values.max(),
                     "best_fit": best.loc[family, "dist"], "ks_pvalue": best.loc[family, "pvalue"]})
    return pd.DataFrame(rows).set_index("family")

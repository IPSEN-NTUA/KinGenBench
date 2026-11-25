"""Training data and benchmark scenarios.

The ORACLE training set is shipped as two compact tables:
    data/oracle/oracle_km.csv.gz    models x 259 Km values
    data/oracle/oracle_kpi.csv.gz   models x 9 zeta-score KPIs (NaN = perturbation
                                    simulation failed for that model)

`build_scenario(name)` turns them into the arrays used to train a conditional
generator: log-standardized Km inputs, the binary (joint high-performance)
label, and the continuous standardized log-KPI conditioning vector.

Example
-------
>>> from benchmark.data import build_scenario
>>> s = build_scenario("all_mutants")
>>> s["X"].shape, s["y_binary"].mean(), s["Y_kpi"].shape
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from benchmark import paths
from benchmark.config import SCENARIOS, cutoffs as _cutoffs, scenario_tag
from benchmark.ml.utils.preprocessing import preprocess_multilabel_dfs

KPI_EPS = 1e-8


def load_oracle_km() -> pd.DataFrame:
    return pd.read_csv(paths.ORACLE_KM, index_col=0)


def load_oracle_kpi() -> pd.DataFrame:
    return pd.read_csv(paths.ORACLE_KPI, index_col=0)


def training_frames(perts, km: pd.DataFrame | None = None,
                    kpi: pd.DataFrame | None = None):
    """One DataFrame per perturbation: Km columns + 'Error' (the zeta KPI),
    restricted to models with a valid KPI for that perturbation."""
    km = load_oracle_km() if km is None else km
    kpi = load_oracle_kpi() if kpi is None else kpi
    out = []
    for p in perts:
        err = kpi[p].dropna()
        df = km.loc[err.index].copy()
        df["Error"] = err.values
        out.append(df)
    return out


def build_kpi_labels(df_aligned, log=True, eps=KPI_EPS):
    """Continuous KPI matrix (one column per perturbation) + fitted scaler."""
    Y_raw = np.column_stack([df["Error"].values for df in df_aligned]).astype(float)
    Y_proc = np.log(Y_raw + eps) if log else Y_raw.copy()
    sc = StandardScaler().fit(Y_proc)
    return sc.transform(Y_proc).astype("float32"), Y_raw, sc


def build_scenario(name: str, pgi_threshold: float | None = None, verbose: bool = True):
    """Arrays for training a conditional generator on one benchmark scenario.

    Returns a dict with
        X            (n, 259) float32, log-transformed + standardized Km
        y_multilabel (n, k)   per-perturbation labels (1 = KPI <= threshold)
        y_binary     (n,)     joint label (1 = good for all k perturbations)
        Y_kpi        (n, k)   float32 standardized log(KPI + 1e-8)
        Y_raw        (n, k)   raw zeta KPIs
        x_scaler, kpi_scaler  fitted scalers (inverse-transform generated data)
        columns, index        Km column names and ORACLE model ids
        perts, cutoffs, thresholds, tag
    """
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; choose from {list(SCENARIOS)}")
    perts = SCENARIOS[name]["perts"]
    cut = _cutoffs(name, pgi_threshold)
    frames = training_frames(perts)
    prep = preprocess_multilabel_dfs(df_list=frames, percentile_cutoffs=tuple(cut))
    df_aligned = prep["df_aligned_list"]
    y_multi = prep["y_multilabel"]
    y_binary = np.all(y_multi == 1, axis=1).astype(int)
    Y_kpi, Y_raw, kpi_scaler = build_kpi_labels(df_aligned)
    out = {
        "X": prep["X_log_standard"].astype("float32"),
        "y_multilabel": y_multi, "y_binary": y_binary,
        "Y_kpi": Y_kpi, "Y_raw": Y_raw,
        "x_scaler": prep["scalers"]["log_standard"], "kpi_scaler": kpi_scaler,
        "columns": list(df_aligned[0].columns.drop("Error")),
        "index": list(df_aligned[0].index),
        "perts": perts, "cutoffs": cut,
        "thresholds": [float(t) for t in prep["thresholds"]],
        "tag": scenario_tag(name, pgi_threshold),
    }
    if verbose:
        print(f"[scenario] {name}: {len(y_binary)} models, "
              f"{int(y_binary.sum())} positive ({100 * y_binary.mean():.2f}%)")
    return out


def postprocess(x_gen, scaler, cols, eps=KPI_EPS):
    """Generated (scaled) samples -> Km DataFrame; drops non-finite / non-positive rows."""
    if hasattr(x_gen, "numpy"):
        x_gen = x_gen.numpy()
    x = np.exp(scaler.inverse_transform(x_gen)) - eps
    df = pd.DataFrame(x, columns=cols)
    n0 = len(df)
    valid = np.isfinite(df.values).all(axis=1) & (df.values > 0).all(axis=1)
    df = df[valid].reset_index(drop=True)
    return df, n0 - len(df)

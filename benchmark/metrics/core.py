#!/usr/bin/env python3
"""
Shared metric and figure helpers used by the paper scripts (paper/) and by
scripts/evaluate.py.

Metrics
    prdc            k-NN precision / recall (Kynkaenniemi et al., k=5) in the
                    PCA-50 embedding of log-standardized Km space
    flux_r2         R^2 / Spearman of ensemble-mean simulated vs measured fluxes
    extrapolation_rho  normalised k-NN distance of generated models to a reference
Figures
    plot_kpi_dist_core, km_embedding_grid, correlation_diff_heatmaps,
    per_parameter_mean_shift_heatmap, nnd_violin

Pure numpy / pandas / matplotlib (umap-learn optional for the Km embedding).
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
METHOD_COLORS = {
    "CVAE": "#1f77b4",   # blue
    "GAN":  "#2ca02c",   # green
    "NF":   "#9467bd",   # purple
    "ORACLE": "#ff7f0e", # orange
}
N_GENERATED = 2000            # models drawn from each generator before filtering
CONDITIONAL_PCT = 10.0        # PGI conditioning threshold (lowest 10% of ORACLE KPI)
TOP_PCT = 1.0                 # "top 1%" class
KNN_K = 5                     # neighbourhood size for precision/recall/density/coverage
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Small numpy helpers
# ---------------------------------------------------------------------------
def _pairwise_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix between rows of A (n,d) and B (m,d)."""
    a2 = (A * A).sum(1)[:, None]
    b2 = (B * B).sum(1)[None, :]
    d2 = a2 + b2 - 2.0 * A @ B.T
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def _kth_nn_radius(X: np.ndarray, k: int) -> np.ndarray:
    """Distance from each row of X to its k-th nearest neighbour within X."""
    d = _pairwise_dist(X, X)
    np.fill_diagonal(d, np.inf)
    kth = min(k, d.shape[1] - 1)
    part = np.partition(d, kth - 1, axis=1)[:, :kth]
    return part.max(axis=1)


def prdc(real: np.ndarray, fake: np.ndarray, k: int = KNN_K) -> dict:
    """Precision, Recall, Density, Coverage (Kynkaanniemi 2019 / Naeem 2020)."""
    r_radius = _kth_nn_radius(real, k)
    f_radius = _kth_nn_radius(fake, k)
    d_rf = _pairwise_dist(real, fake)          # (n_real, n_fake)

    # precision: fake points inside any real hypersphere
    precision = (d_rf <= r_radius[:, None]).any(axis=0).mean()
    # recall: real points inside any fake hypersphere
    recall = (d_rf <= f_radius[None, :]).any(axis=1).mean()
    # density: avg number of real hyperspheres each fake falls into / k
    density = (1.0 / k) * (d_rf <= r_radius[:, None]).sum(axis=0).mean()
    # coverage: fraction of real points with a fake inside their own radius
    coverage = (d_rf <= r_radius[:, None]).any(axis=1).mean()
    return {"precision": float(precision), "recall": float(recall),
            "density": float(density), "coverage": float(coverage)}


def _pca_fit(real: np.ndarray, n_components: int):
    """Fit PCA on the reference set (SVD). k-NN diversity metrics are unreliable
    in the raw 259-dim Km space (distance concentration), so PRDC/Vendi are
    computed in this reduced embedding fit on the ORACLE high-performers."""
    mean = real.mean(axis=0)
    U, S, Vt = np.linalg.svd(real - mean, full_matrices=False)
    return mean, Vt[:n_components]


def _pca_transform(X: np.ndarray, mean, comps) -> np.ndarray:
    return (X - mean) @ comps.T


def _signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _r2(pred: np.ndarray, obs: np.ndarray) -> float:
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _kde(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Gaussian KDE (Scott's rule) — avoids the scipy dependency."""
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return np.zeros_like(grid)
    bw = x.std(ddof=1) * n ** (-1.0 / 5.0)
    if bw <= 0:
        bw = 1e-3
    u = (grid[:, None] - x[None, :]) / bw
    k = np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)
    return k.sum(axis=1) / (n * bw)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _one(colname_glob: Path) -> Path:
    # case-insensitive glob so PGI / pgi filename conventions both resolve
    pat = colname_glob.name
    hits = sorted(set(colname_glob.parent.glob(pat))
                  | set(colname_glob.parent.glob(pat.lower()))
                  | set(colname_glob.parent.glob(pat.upper())))
    if "*" in pat:   # a wildcard never refers to the ORACLE re-scored tables
        hits = [h for h in hits if not h.name.lower().startswith("kpi_oracle_")]
    if not hits:
        raise FileNotFoundError(f"No file matching {colname_glob}")
    return hits[0]


# ---------------------------------------------------------------------------
# Metric assembly
# ---------------------------------------------------------------------------
def flux_r2(enzyme: pd.DataFrame, exp_vec: pd.Series):
    model_cols = [c for c in enzyme.columns if c != "Steady"]
    mean_sim = enzyme[model_cols].mean(axis=1)
    common = mean_sim.index.intersection(exp_vec.dropna().index)
    pred = mean_sim.loc[common].values.astype(float)
    obs = exp_vec.loc[common].values.astype(float)
    return {
        "R2_raw": _r2(pred, obs),
        "R2_signed_log": _r2(_signed_log(pred), _signed_log(obs)),
        "spearman": _spearman(pred, obs),
        "n_reactions": len(common),
    }


def log_standard(df: pd.DataFrame, ref_mean, ref_std):
    return (np.log(df.values) - ref_mean) / ref_std


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_kpi_dist_core(series, order, p_cond, best_or, title, out_png,
                       xmin=None, xmax=None, cond_pct=None):
    """Reusable KPI-distribution plot (KDE + means + ORACLE-best inset).

    series : dict label -> KPI array; order : list of labels ending in 'ORACLE'.
    """
    method_labels = [l for l in order if l != "ORACLE"]
    allvals = np.concatenate([series[l] for l in order])
    lo = float(np.min(allvals)) if xmin is None else xmin
    hi = float(np.percentile(allvals, 99.5)) if xmax is None else xmax
    span = hi - lo
    grid = np.linspace(max(0.0, lo - 0.02 * span), hi + 0.02 * span, 500)
    dens_map = {lbl: _kde(series[lbl], grid) for lbl in order}

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for lbl in order:
        v = series[lbl]
        color = METHOD_COLORS.get(lbl, None)
        ax.fill_between(grid, dens_map[lbl], color=color, alpha=0.18)
        ax.plot(grid, dens_map[lbl], color=color, lw=2,
                label=f"{lbl} (mean={np.mean(v):.2f})")
        ax.axvline(np.mean(v), color=color, ls="--", lw=1.4, alpha=0.9)

    plbl = f"ORACLE p{cond_pct:g} = {p_cond:.2f}" if cond_pct is not None \
        else f"ORACLE threshold = {p_cond:.2f}"
    ax.axvline(p_cond, color="black", ls=":", lw=1.6, label=plbl)
    ax.axvline(best_or, color="#d62728", ls="-.", lw=1.8,
               label=f"ORACLE best = {best_or:.2f}")
    ax.axvspan(grid[0], best_or, color="#d62728", alpha=0.07)

    ax.set_xlabel("KPI (ζ-score)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9, loc="upper right")
    ax.grid(alpha=0.25, ls="--")
    ax.set_xlim(grid[0], grid[-1])

    # highlight the individual generated models that beat the best ORACLE model,
    # as a per-method rug at the bottom of the outperform region (KPI < best_or)
    ymax_plot = max(float(dens_map[lbl].max()) for lbl in order)
    for i, lbl in enumerate(method_labels):
        xb = series[lbl][series[lbl] < best_or]
        if len(xb):
            y0 = 0.012 * ymax_plot + i * 0.022 * ymax_plot
            ax.vlines(xb, y0, y0 + 0.017 * ymax_plot,
                      color=METHOD_COLORS[lbl], lw=1.7, alpha=0.95)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# ===========================================================================
# Reusable parameter-space figures (shared by the dual / capstone pipelines)
# km_by_method: {method_label -> DataFrame of Km values (rows=models, cols=params)}
# oracle_km:    DataFrame of ORACLE Km values (same columns)
# ===========================================================================
def _w1(a, b, nq=100):
    """Quantile-based 1-Wasserstein distance between two 1-D samples (numpy-only)."""
    q = np.linspace(0.01, 0.99, nq)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def _embed2d(X, method="umap", seed=0):
    """2-D embedding: UMAP if available, else PCA fallback (numpy SVD)."""
    if method == "umap":
        try:
            import umap
            emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed).fit_transform(X)
            return np.asarray(emb), "UMAP"
        except Exception as exc:
            print(f"[embed] UMAP unavailable ({type(exc).__name__}); using PCA. "
                  f"pip install umap-learn for UMAP.")
    mean = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
    return (X - mean) @ Vt[:2].T, "PCA"


def km_embedding_grid(km_by_method, oracle_km, order, colors, out_png,
                      method="umap", cap=1500, oracle_cap=3000, seed=0):
    cols = oracle_km.columns
    o_log = np.log(oracle_km[cols].values)
    mean = o_log.mean(0); std = o_log.std(0); std[std == 0] = 1
    rng = np.random.default_rng(seed)

    def feat(df, n):
        X = (np.log(df[cols].values) - mean) / std
        return X[rng.choice(len(X), n, replace=False)] if len(X) > n else X

    groups = [("ORACLE", feat(oracle_km, oracle_cap))]
    for l in order:
        if l in km_by_method:
            groups.append((l, feat(km_by_method[l], cap)))
    sizes = [g[1].shape[0] for g in groups]
    emb, name = _embed2d(np.vstack([g[1] for g in groups]), method, seed)
    coords, i = {}, 0
    for (lbl, _), sz in zip(groups, sizes):
        coords[lbl] = emb[i:i + sz]; i += sz

    panels = [l for l, _ in groups]
    ncol = 4; nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.9 * nrow))
    axes = np.atleast_1d(axes).ravel()
    o_xy = coords["ORACLE"]
    xlim = (np.percentile(emb[:, 0], 0.5), np.percentile(emb[:, 0], 99.5))
    ylim = (np.percentile(emb[:, 1], 0.5), np.percentile(emb[:, 1], 99.5))
    for ax, lbl in zip(axes, panels):
        if lbl != "ORACLE":
            ax.scatter(o_xy[:, 0], o_xy[:, 1], s=5, color="0.8", alpha=0.5, edgecolors="none")
        xy = coords[lbl]
        ax.scatter(xy[:, 0], xy[:, 1], s=6, color=colors.get(lbl), alpha=0.55, edgecolors="none")
        ax.set_title(lbl, fontsize=11, fontweight="bold")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel(f"{name}-1"); ax.set_ylabel(f"{name}-2"); ax.grid(alpha=0.2, ls="--")
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"Km-space {name} embedding (grey = ORACLE reference)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out_png}")


def _cmd(C1, C2):
    """Correlation-matrix distance in [0, 1] (0 = identical structure)."""
    num = np.trace(C1 @ C2)
    den = np.linalg.norm(C1) * np.linalg.norm(C2)
    return float(1.0 - num / den) if den > 0 else float("nan")


def correlation_diff_heatmaps(km_by_method, oracle_km, order, colors_unused, out_png):
    cols = oracle_km.columns
    Corr_o = np.corrcoef(np.log(oracle_km[cols].values), rowvar=False)
    methods = [l for l in order if l in km_by_method]
    ncol = 3; nrow = int(np.ceil(len(methods) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    im = None
    for ax, lbl in zip(axes, methods):
        Cm = np.corrcoef(np.log(km_by_method[lbl][cols].values), rowvar=False)
        diff = Cm - Corr_o
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_title(f"{lbl}  (CMD={_cmd(Cm, Corr_o):.3f})", fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"{len(cols)} $K_m$ params"); ax.set_ylabel(f"{len(cols)} $K_m$ params")
    for ax in axes[len(methods):]:
        ax.axis("off")
    if im is not None:
        cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02)
        cb.set_label("corr(method) − corr(ORACLE)", fontweight="bold")
    fig.suptitle("Km–Km correlation structure: generated − ORACLE",
                 fontsize=13, fontweight="bold")
    fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out_png}")


def per_parameter_mean_shift_heatmap(km_by_method, oracle_km, order, out_png, top_n=30):
    """Signed shift of each method's MEAN log(Km) from the ORACLE baseline.
    Diverging colours -> a row of the same colour = methods agree on the
    direction (and magnitude) of the shift for that parameter."""
    cols = oracle_km.columns
    o_lmean = np.log(oracle_km[cols].values).mean(0)
    methods = [l for l in order if l in km_by_method]
    S = np.zeros((len(cols), len(methods)))
    for j, lbl in enumerate(methods):
        S[:, j] = np.log(km_by_method[lbl][cols].values).mean(0) - o_lmean
    top = np.argsort(np.abs(S).max(1))[::-1][:top_n]
    St = S[top]; names = [cols[i] for i in top]
    vmax = float(np.abs(St).max())
    fig, ax = plt.subplots(figsize=(1.2 * len(methods) + 3, 0.32 * top_n + 1.5))
    im = ax.imshow(St, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_yticks(range(top_n)); ax.set_yticklabels(names, fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(r"mean $\log(K_m)$ shift vs ORACLE  (red = up, blue = down)", fontweight="bold")
    ax.set_title(f"Top {top_n} most-shifted $K_m$ parameters (mean shift vs ORACLE)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out_png}")


# ---------------------------------------------------------------------------
# Extrapolation / novelty: normalized nearest-neighbour distance to a reference
# ---------------------------------------------------------------------------
def _knn_mean_dist(query, ref, k=5):
    """For each query row, mean Euclidean distance to its k nearest ref rows."""
    D = _pairwise_dist(query, ref)
    k = min(k, ref.shape[0])
    part = np.partition(D, k - 1, axis=1)[:, :k]
    return part.mean(axis=1)


def _within_ref_scale(ref, k=5, cap=3000, seed=0):
    """Median within-reference k-NN mean distance (the 'ρ=1' scale)."""
    R = ref
    if len(R) > cap:
        R = R[np.random.default_rng(seed).choice(len(R), cap, replace=False)]
    D = _pairwise_dist(R, R)
    np.fill_diagonal(D, np.inf)
    k = min(k, R.shape[0] - 1)
    part = np.partition(D, k - 1, axis=1)[:, :k]
    return float(np.median(part.mean(axis=1)))


def project_km(km_df, ref):
    """Project raw Km into a ref's log-standardized PCA space (ref from *ref_projection)."""
    return _pca_transform((np.log(km_df[ref["cols"]].values) - ref["rmean"]) / ref["rstd"],
                          ref["pmean"], ref["pcomps"])


def extrapolation_rho(fake_proj, ref_proj, scale, k=5):
    """Per-model normalized NN distance ρ_j = knn_mean_dist(f_j, ref) / scale.
    ρ≈1 interpolates within the reference; ρ≫1 extrapolates; ρ≈0 ~ memorization."""
    return _knn_mean_dist(fake_proj, ref_proj, k) / (scale + 1e-12)


def nnd_violin(rho_by_method, colors, order, out_png, ref_name="reference", clip_pct=99.0):
    """Violin of the per-model normalized NN distance ρ, one violin per method."""
    methods = [l for l in order if l in rho_by_method and len(rho_by_method[l])]
    data = [np.clip(rho_by_method[l], None, np.percentile(rho_by_method[l], clip_pct))
            for l in methods]
    fig, ax = plt.subplots(figsize=(1.3 * len(methods) + 2, 5.5))
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for b, lbl in zip(parts["bodies"], methods):
        b.set_facecolor(colors.get(lbl, "#888")); b.set_alpha(0.7); b.set_edgecolor("black")
    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
    ax.axhline(1.0, color="grey", ls="--", lw=1.3, label="reference scale (ρ=1)")
    ax.set_xticks(range(1, len(methods) + 1)); ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel(f"normalized NN distance ρ to {ref_name}", fontsize=11, fontweight="bold")
    ax.set_title(f"Extrapolation: distance of generated models from {ref_name}\n"
                 f"(ρ≈1 interpolate · ρ≫1 extrapolate · ρ≈0 memorize)",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25, ls="--")
    fig.tight_layout(); fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out_png}")



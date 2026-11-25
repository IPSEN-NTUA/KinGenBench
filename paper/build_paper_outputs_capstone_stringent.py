#!/usr/bin/env python3
"""
Stringent 9-mutant joint-robustness capstone figures + metrics for the six
architecture/labelling variants (discrete + continuous CVAE/GAN/NF), using the
difficulty-calibrated composite per-mutant thresholds, aggregated over seeds.

Seed handling:
  * Summary-metric figures (ladder, enrichment, degree histogram, per-mutant
    satisfaction) show the MEAN across seeds with +/- std error bars / bands.
  * Distribution figures (KPI grid, per-model R2, KPI-PCA scatter) POOL the
    seeds' samples into one distribution per method.
  * metrics_per_seed.csv / metrics_mean_std.csv report the main metrics per
    seed and as mean +/- std.

Each variant automatically uses every seed it has under
  outputs/runs/benchmark_<prefix>_all_mutants_seed<seed>/
(restrict with --seeds). Output -> results/paper_figures/capstone_stringent/.

Run (from the repository root):  python paper/build_paper_outputs_capstone_stringent.py
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # repo root
from benchmark import paths
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from benchmark.metrics import core as B
from benchmark.scoring.predicted_vs_observed import plot_per_model_r2_kde


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MUTANTS = ["PGI", "PGL", "RPI", "TALA", "PYK", "G6PDH2r", "PPS", "TKT2", "GND"]
COMPOSITE_PCT = {"PGI": 25, "PGL": 40, "RPI": 60, "TALA": 60, "PYK": 60,
                 "G6PDH2r": 25, "PPS": 40, "TKT2": 60, "GND": 40}
NM = len(MUTANTS)
AGG_REF_PCT = 10.0
KPI_NONPHYSICAL = 1e3     # zeta above this = simulation blow-up; excluded from mean KPI

VARIANT_PREFIX = {
    "CVAE": "cvae", "GAN": "gan", "NF": "nf",
    "CVAE-cont": "cvae_continuous", "GAN-cont": "gan_continuous",
    "NF-cont": "nf_continuous",
}
# Consistent scheme: discrete = strong hue, continuous = light hue of same family.
COLORS = {"CVAE": "#1f77b4", "CVAE-cont": "#9ecae1",   # blue / light blue
          "GAN": "#2ca02c",  "GAN-cont": "#98df8a",    # green / light green
          "NF": "#9467bd",   "NF-cont": "#c5b0d5",      # purple / light purple
          "ORACLE": "#ff7f0e"}

ORDER = []   # filled in main()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _aligned_kpi(tables: Path, pattern):
    kpi = {m: pd.read_csv(B._one(tables / pattern(m)), index_col=0).iloc[:, 0]
           for m in MUTANTS}
    idx = kpi[MUTANTS[0]].index
    for m in MUTANTS[1:]:
        idx = idx.intersection(kpi[m].index)
    return {m: kpi[m].loc[idx] for m in MUTANTS}, idx


def discover_seeds(pref, allow=None):
    seeds = set()
    for d in paths.RUNS_DIR.glob(f"benchmark_{pref}_all_mutants_seed*"):
        m = re.search(r"seed(\d+)$", d.name)
        if m and (d / "tables").is_dir():
            seeds.add(int(m.group(1)))
    if allow:
        seeds &= set(allow)
    return sorted(seeds)


def load_seed(folder, km_file):
    run = Path(folder)
    kpi, idx = _aligned_kpi(run / "tables", lambda m: f"kpi_*_{m}.csv")
    enzyme = {m: str(B._one(run / "enzyme_perturbations" / f"*_{m}.csv")) for m in MUTANTS}
    if km_file and Path(km_file).exists():
        gen = pd.read_csv(km_file, index_col=0)
        ks = [int(str(i).split(",")[-1]) for i in idx]
        km = gen.loc[ks]; km.index = idx; n_gen = len(gen)
    else:
        km, n_gen = None, 2000
    return {"kpi": kpi, "enzyme": enzyme, "km": km, "n_gen": n_gen, "n_scored": len(idx)}


def load_variant_ms(lbl, seeds):
    pref = VARIANT_PREFIX[lbl]
    per = []
    for s in seeds:
        folder = f"{paths.RUNS_DIR}/benchmark_{pref}_all_mutants_seed{s}"
        km = f"{paths.SYNTH_DIR}/benchmark_{pref}_all_mutants_seed{s}.csv"
        per.append(load_seed(folder, km))
    return {"seeds": seeds, "per_seed": per}


def load_oracle(any_folder):
    tables = Path(any_folder) / "tables"
    kpi, _ = _aligned_kpi(tables, lambda m: f"kpi_oracle_{m}.csv")
    kms = pd.read_csv(paths.ORACLE_KM, index_col=0)
    enzyme = {m: str(paths.oracle_flux(m)) for m in MUTANTS}
    return {"kpi": kpi, "kms": kms, "enzyme": enzyme, "idx": kpi[MUTANTS[0]].index}


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------
def thresholds(oracle):
    return {m: {"cond": float(np.percentile(oracle["kpi"][m].values, COMPOSITE_PCT[m])),
                "p1": float(np.percentile(oracle["kpi"][m].values, 1.0)),
                "best": float(oracle["kpi"][m].values.min())} for m in MUTANTS}


def degrees(kpi, thr):
    """Number of satisfied mutant constraints per model. `kpi` maps mutant ->
    Series or ndarray of KPI values."""
    return np.column_stack([(np.asarray(kpi[m]) <= thr[m]["cond"]) for m in MUTANTS]).sum(1)


def pooled_kpi(v_ms):
    return {m: np.concatenate([ps["kpi"][m].values for ps in v_ms["per_seed"]]) for m in MUTANTS}


def pooled_enzyme_df(v_ms, m):
    dfs = []
    for i, ps in enumerate(v_ms["per_seed"]):
        e = pd.read_csv(ps["enzyme"][m], index_col=0)
        e = e.drop(columns=[c for c in ["Steady"] if c in e.columns])
        dfs.append(e.add_prefix(f"s{i}_"))
    return pd.concat(dfs, axis=1)


def diversity_reference(oracle, thr):
    idx = oracle["kms"].index.intersection(oracle["idx"])
    meanK = pd.DataFrame({m: oracle["kpi"][m].loc[idx] for m in MUTANTS}).mean(axis=1)
    return oracle["kms"].loc[idx][meanK.values <= np.percentile(meanK.values, AGG_REF_PCT)]


def ref_projection(oracle, thr):
    """PCA embedding of the ORACLE broadly-good reference for diversity metrics."""
    ref_km = diversity_reference(oracle, thr); cols = ref_km.columns
    rmean = np.log(ref_km.values).mean(0); rstd = np.log(ref_km.values).std(0); rstd[rstd == 0] = 1
    real = (np.log(ref_km.values) - rmean) / rstd
    ncomp = int(min(50, real.shape[1], real.shape[0] - 1))
    pmean, pcomps = B._pca_fit(real, ncomp)
    return {"cols": cols, "rmean": rmean, "rstd": rstd, "pmean": pmean,
            "pcomps": pcomps, "real_proj": B._pca_transform(real, pmean, pcomps)}


def km_prdc(km_df, ref):
    fp = B._pca_transform((np.log(km_df[ref["cols"]].values) - ref["rmean"]) / ref["rstd"],
                          ref["pmean"], ref["pcomps"])
    return B.prdc(ref["real_proj"], fp)


def _survivor_km(km_file, idx):
    gen = pd.read_csv(km_file, index_col=0)
    ks = [int(str(i).split(",")[-1]) for i in idx]
    return gen.loc[ks], len(gen)


_ORACLE_EXP = {}
def oracle_exp():
    if not _ORACLE_EXP:
        _ORACLE_EXP["df"] = pd.read_csv(paths.EXP_FLUX, index_col=0)
    return _ORACLE_EXP["df"]


# ---------------------------------------------------------------------------
# Metrics: per-seed matrix + mean/std, and pooled full (supplementary) table
# ---------------------------------------------------------------------------
def metric_context(oracle, thr, verbose=True):
    """Reference sets shared by every run (joint base rate, diversity, extrapolation)."""
    or_joint = 100.0 * np.mean(degrees(oracle["kpi"], thr) == NM)
    ref = ref_projection(oracle, thr)
    ref_top = ref["real_proj"]; scale_top = B._within_ref_scale(ref_top)
    o_full = B.project_km(oracle["kms"], ref)
    rng = np.random.default_rng(0)
    if len(o_full) > 4000:
        o_full = o_full[rng.choice(len(o_full), 4000, replace=False)]
    scale_full = B._within_ref_scale(o_full)
    if verbose:
        print(f"[ref] ORACLE joint base rate = {or_joint:.3f}%")
    return {"or_joint": or_joint, "ref": ref, "ref_top": ref_top, "scale_top": scale_top,
            "o_full": o_full, "scale_full": scale_full}


def run_metrics(folder, km_file, thr, ctx):
    """Paper metrics for one scored run folder (None if the run is incomplete)."""
    folder, km = Path(folder), Path(km_file)
    try:
        kpi, idx = _aligned_kpi(folder / "tables", lambda m: f"kpi_*_{m}.csv")
    except FileNotFoundError:
        return None
    ref, or_joint = ctx["ref"], ctx["or_joint"]
    deg = degrees(kpi, thr)
    joint = 100.0 * np.mean(deg == NM)
    # number of perturbations (0..NM) for which the method produced at
    # least one model beating the single best ORACLE model for that mutant
    n_new_best = int(sum(np.any(kpi[m].values < thr[m]["best"]) for m in MUTANTS))
    # diversity (Km space) needs the km file for this run
    prec = rec = rho_top = rho_full = np.nan; n_gen = 2000
    if km.exists():
        km_df, n_gen = _survivor_km(km, idx)
        dc = km_prdc(km_df, ref); prec, rec = dc["precision"], dc["recall"]
        fp = B.project_km(km_df, ref)
        rho_top = float(B._knn_mean_dist(fp, ctx["ref_top"]).mean() / ctx["scale_top"])
        rho_full = float(B._knn_mean_dist(fp, ctx["o_full"]).mean() / ctx["scale_full"])
    return {"pct_survived": 100.0 * len(idx) / n_gen,
            "joint_pct_all9": joint,
            "enrichment_vs_oracle": (joint / or_joint) if or_joint > 0 else np.nan,
            "pct_ge7": 100.0 * np.mean(deg >= NM - 2),
            "n_new_best_pert": n_new_best,
            "precision": prec, "recall": rec,
            "extrap_vs_top": rho_top, "extrap_vs_full": rho_full}


def seed_metric_matrices(oracle, thr, out, allow_seeds=None):
    ctx = metric_context(oracle, thr)
    rows = []
    for lbl, pref in VARIANT_PREFIX.items():
        for s in discover_seeds(pref, allow_seeds):
            folder = Path(f"{paths.RUNS_DIR}/benchmark_{pref}_all_mutants_seed{s}")
            km = Path(f"{paths.SYNTH_DIR}/benchmark_{pref}_all_mutants_seed{s}.csv")
            row = run_metrics(folder, km, thr, ctx)
            if row is not None:
                rows.append({"variant": lbl, "seed": s, **row})
    per = pd.DataFrame(rows)
    if per.empty:
        print("[warn] no seed runs found"); return per, None
    order = [l for l in VARIANT_PREFIX if l in set(per["variant"])]
    per["variant"] = pd.Categorical(per["variant"], categories=order, ordered=True)
    per = per.sort_values(["variant", "seed"])
    metrics = ["pct_survived", "joint_pct_all9", "enrichment_vs_oracle", "pct_ge7",
               "n_new_best_pert", "precision", "recall", "extrap_vs_top", "extrap_vs_full"]
    agg = per.groupby("variant", observed=True)[metrics].agg(["mean", "std", "count"])
    agg.columns = [f"{m}__{st}" for m, st in agg.columns]
    pretty = pd.DataFrame(index=agg.index)
    for m in metrics:
        pretty[m] = agg.apply(
            lambda r: f"{r[f'{m}__mean']:.2f} ± "
                      f"{0.0 if pd.isna(r[f'{m}__std']) else r[f'{m}__std']:.2f}", axis=1)
    pretty["n_seeds"] = agg[f"{metrics[0]}__count"].astype(int)
    per.to_csv(out / "metrics_per_seed.csv", index=False)
    agg.round(4).to_csv(out / "metrics_mean_std.csv")
    pretty.to_csv(out / "metrics_mean_std_pretty.csv")
    print("\n=== per-seed metrics ===\n" + per.round(3).to_string(index=False))
    print("\n=== mean ± std across seeds ===\n" + pretty.to_string())
    return per, agg


def build_full_table(variants_ms, oracle, thr, out):
    """Pooled per-mutant + diversity table (supplementary)."""
    ref_km = diversity_reference(oracle, thr); cols = ref_km.columns
    rmean = np.log(ref_km.values).mean(0); rstd = np.log(ref_km.values).std(0); rstd[rstd == 0] = 1
    real = (np.log(ref_km.values) - rmean) / rstd
    ncomp = int(min(50, real.shape[1], real.shape[0] - 1))
    pmean, pcomps = B._pca_fit(real, ncomp); real_proj = B._pca_transform(real, pmean, pcomps)

    rows = []
    for lbl in ORDER:
        vms = variants_ms[lbl]; pk = pooled_kpi(vms); deg = degrees(pk, thr)
        row = {"variant": lbl, "n_seeds": len(vms["seeds"]),
               "joint_pct_all9": 100.0 * np.mean(deg == NM)}
        for m in MUTANTS:
            row[f"{m}_mean_KPI"] = float(pk[m][pk[m] < KPI_NONPHYSICAL].mean())
            row[f"{m}_pct_below_p{COMPOSITE_PCT[m]:g}"] = 100.0 * np.mean(pk[m] <= thr[m]["cond"])
            row[f"{m}_R2_raw"] = B.flux_r2(pooled_enzyme_df(vms, m), oracle_exp()[m.lower()])["R2_raw"]
        row["mean_KPI_9"] = float(np.mean([row[f"{m}_mean_KPI"] for m in MUTANTS]))
        row["n_nonphysical_KPI"] = int(sum((pk[m] >= KPI_NONPHYSICAL).sum() for m in MUTANTS))
        kms = [ps["km"] for ps in vms["per_seed"] if ps["km"] is not None]
        if kms:
            km = pd.concat(kms, ignore_index=True)
            fp = B._pca_transform((np.log(km[cols].values) - rmean) / rstd, pmean, pcomps)
            dc = B.prdc(real_proj, fp)
            row.update({"precision": dc["precision"], "recall": dc["recall"]})
        else:
            row.update({"precision": np.nan, "recall": np.nan})
        rows.append(row)
    df = pd.DataFrame(rows).set_index("variant")
    df.round(6).to_csv(out / "metrics_full.csv")
    print(f"[saved] {out / 'metrics_full.csv'}")
    return df


# ---------------------------------------------------------------------------
# Summary-metric figures (mean +/- std across seeds)
# ---------------------------------------------------------------------------
def _per_seed_degrees(v_ms, thr):
    return [degrees(ps["kpi"], thr) for ps in v_ms["per_seed"]]


def plot_headline(variants_ms, oracle, thr, out_png):
    ks = np.arange(NM + 1)
    or_joint = 100.0 * np.mean(degrees(oracle["kpi"], thr) == NM)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 5.8))

    enr_mean, enr_std = {}, {}
    for lbl in ORDER:
        degs = _per_seed_degrees(variants_ms[lbl], thr)
        F = np.array([[100.0 * np.mean(d >= k) for k in ks] for d in degs])  # (nseed, 10)
        mean, std = F.mean(0), F.std(0)
        a1.plot(ks, mean, "-o", color=COLORS[lbl], lw=2.2, ms=5, label=lbl, zorder=3)
        a1.fill_between(ks, np.clip(mean - std, 0, None), mean + std,
                        color=COLORS[lbl], alpha=0.16)
        e = np.array([100.0 * np.mean(d == NM) / or_joint for d in degs]) if or_joint > 0 else np.array([np.nan])
        enr_mean[lbl], enr_std[lbl] = float(np.nanmean(e)), float(np.nanstd(e))
    odeg = degrees(oracle["kpi"], thr)
    a1.plot(ks, [100.0 * np.mean(odeg >= k) for k in ks], "-o", color=COLORS["ORACLE"],
            lw=2, ms=4, label="ORACLE", zorder=2)
    a1.set_yscale("symlog", linthresh=0.05); a1.set_ylim(0, 160)
    from matplotlib.ticker import FuncFormatter, FixedLocator
    a1.yaxis.set_major_locator(FixedLocator([0, 0.1, 1, 10, 100]))
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a1.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    a1.set_xticks(ks); a1.axvline(NM, color="#d62728", ls="--", lw=1.3, alpha=0.6)
    a1.set_xlabel("k = mutant constraints satisfied (≥ k)", fontsize=12, fontweight="bold")
    a1.set_ylabel("% of models", fontsize=12, fontweight="bold")
    a1.set_title("(A) Joint-robustness ladder  (mean ± std over seeds)",
                 fontsize=12, fontweight="bold")
    a1.legend(fontsize=9.5, framealpha=0.9); a1.grid(alpha=0.25, ls="--")

    means = np.array([enr_mean[l] for l in ORDER])
    stds = np.array([enr_std[l] for l in ORDER])
    ypos = np.arange(len(ORDER))[::-1]
    lower = np.clip(np.minimum(stds, means - 0.03), 0, None)
    a2.barh(ypos, np.clip(means, 0.03, None), xerr=np.vstack([lower, stds]),
            color=[COLORS[l] for l in ORDER], alpha=0.85, edgecolor="black",
            error_kw=dict(lw=1.1, capsize=3))
    for y, l in zip(ypos, ORDER):
        # place the label just past the right whisker so it never overlaps the error bar
        xr = (enr_mean[l] + max(enr_std[l], 0.0)) * 1.15
        a2.text(xr, y, f"{enr_mean[l]:.0f}× ± {enr_std[l]:.0f}",
                va="center", ha="left", fontsize=8.5)
    a2.set_xscale("log"); a2.axvline(1.0, color="black", ls="--", lw=1.4, label="ORACLE (1×)")
    a2.set_yticks(ypos); a2.set_yticklabels(ORDER)
    a2.set_xlabel("rare-class enrichment vs ORACLE (mean ± std)", fontsize=11, fontweight="bold")
    a2.set_title("(B) Rare-class enrichment", fontsize=12, fontweight="bold")
    a2.set_xlim(0.02, max((means + stds).max() * 6, 30))
    a2.legend(fontsize=9); a2.grid(axis="x", alpha=0.25, ls="--")
    fig.suptitle("Stringent 9-mutant joint class", fontsize=13.5, fontweight="bold")
    fig.tight_layout(); fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out_png}")


def plot_degree_hist(variants_ms, oracle, thr, out_png):
    ks = np.arange(NM + 1); order = ORDER + ["ORACLE"]
    w = 0.8 / len(order)
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    for gi, lbl in enumerate(order):
        if lbl == "ORACLE":
            d = degrees(oracle["kpi"], thr)
            mean = np.array([100.0 * np.mean(d == k) for k in ks]); std = np.zeros_like(mean)
        else:
            F = np.array([[100.0 * np.mean(dd == k) for k in ks]
                          for dd in _per_seed_degrees(variants_ms[lbl], thr)])
            mean, std = F.mean(0), F.std(0)
        ax.bar(ks + gi * w - 0.4 + w / 2, mean, w, yerr=std, color=COLORS[lbl],
               alpha=0.85, edgecolor="white", label=lbl, error_kw=dict(lw=0.8, capsize=1.5))
    ax.set_xticks(ks)
    ax.set_xlabel("number of mutant constraints satisfied (exactly k)", fontsize=12, fontweight="bold")
    ax.set_ylabel("% of models", fontsize=12, fontweight="bold")
    ax.set_title("Degree distribution (mean ± std over seeds)", fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9, ncol=2, framealpha=0.9); ax.grid(axis="y", alpha=0.25, ls="--")
    fig.tight_layout(); fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out_png}")


def plot_per_mutant(variants_ms, oracle, thr, out_png):
    order = ORDER + ["ORACLE"]
    x = np.arange(NM); w = 0.8 / len(order)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for gi, lbl in enumerate(order):
        if lbl == "ORACLE":
            mean = np.array([100.0 * np.mean(oracle["kpi"][m].values <= thr[m]["cond"]) for m in MUTANTS])
            std = np.zeros_like(mean)
        else:
            F = np.array([[100.0 * np.mean(ps["kpi"][m].values <= thr[m]["cond"]) for m in MUTANTS]
                          for ps in variants_ms[lbl]["per_seed"]])
            mean, std = F.mean(0), F.std(0)
        ax.bar(x + gi * w - 0.4 + w / 2, mean, w, yerr=std, color=COLORS[lbl],
               alpha=0.85, edgecolor="white", label=lbl, error_kw=dict(lw=0.8, capsize=1.5))
    ax.set_xticks(x); ax.set_xticklabels([f"{m}\n(p{COMPOSITE_PCT[m]:g})" for m in MUTANTS], fontsize=8.5)
    ax.set_ylabel("% of models below mutant threshold", fontsize=12, fontweight="bold")
    ax.set_title("Per-mutant satisfaction (mean ± std over seeds)", fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9, ncol=3, framealpha=0.9); ax.grid(axis="y", alpha=0.25, ls="--")
    fig.tight_layout(); fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out_png}")


# ---------------------------------------------------------------------------
# Distribution figures (pooled across seeds)
# ---------------------------------------------------------------------------
def plot_kpi_grid(variants_ms, oracle, thr, out_png, clip_pct=98.0):
    order = ORDER + ["ORACLE"]
    kpis = {**{l: pooled_kpi(variants_ms[l]) for l in ORDER},
            "ORACLE": {m: oracle["kpi"][m].values for m in MUTANTS}}
    n_cols = 3; n_rows = int(np.ceil(NM / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.2 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, m in zip(axes, MUTANTS):
        allv = np.concatenate([kpis[l][m] for l in order])
        # trim extreme KPI values so a few outliers don't stretch the panel
        lo, hi = float(np.min(allv)), float(np.percentile(allv, clip_pct))
        grid = np.linspace(max(0.0, lo - 0.02 * (hi - lo)), hi, 400)
        for lbl in order:
            v = kpis[lbl][m]
            v = v[v <= hi]                     # discard extreme values for the plot
            dens = B._kde(v, grid)
            ax.fill_between(grid, dens, color=COLORS[lbl], alpha=0.12)
            ax.plot(grid, dens, color=COLORS[lbl], lw=1.8, label=lbl)
        ax.axvline(thr[m]["cond"], color="black", ls=":", lw=1.5)
        ax.set_title(f"{m}  (p{COMPOSITE_PCT[m]:g} = {thr[m]['cond']:.2f})", fontsize=12, fontweight="bold")
        ax.set_xlabel("KPI (ζ-score)", fontsize=10); ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(grid[0], grid[-1]); ax.grid(alpha=0.25, ls="--")
    for ax in axes[NM:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(order), fontsize=11,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Per-mutant KPI distributions — all methods vs ORACLE "
                 f"(pooled over seeds, >{clip_pct:g}th-pct outliers trimmed; "
                 f"dotted line = good-fit threshold)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out_png}")


def plot_kpi_pca(variants_ms, oracle, thr, out_png, oracle_sample=4000):
    def mat(kpi_arrays): return np.column_stack([kpi_arrays[m] for m in MUTANTS])
    pooled = {l: pooled_kpi(variants_ms[l]) for l in ORDER}
    o_arr = {m: oracle["kpi"][m].values for m in MUTANTS}
    pool = [mat(pooled[l]) for l in ORDER]
    o_all = mat(o_arr)
    oi = np.random.default_rng(0).choice(len(o_all), min(oracle_sample, len(o_all)), replace=False)
    P = np.vstack(pool + [o_all[oi]])
    mu, sd = P.mean(0), P.std(0); sd[sd == 0] = 1
    pmean, pcomps = B._pca_fit((P - mu) / sd, 2)
    proj = lambda km: B._pca_transform((mat(km) - mu) / sd, pmean, pcomps)

    panels = [(l, pooled[l]) for l in ORDER] + [("ORACLE", o_arr)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9)); axes = axes.flatten()
    norm = Normalize(0, NM)
    allp = np.vstack([proj(k) for _, k in panels])
    xlim = (np.percentile(allp[:, 0], 0.5), np.percentile(allp[:, 0], 99.5))
    ylim = (np.percentile(allp[:, 1], 0.5), np.percentile(allp[:, 1], 99.5))
    for ax, (lbl, km) in zip(axes, panels):
        pr = proj(km); deg = degrees(km, thr); j = deg == NM
        ax.scatter(pr[~j, 0], pr[~j, 1], c=deg[~j], cmap="viridis", norm=norm, s=7, alpha=0.5, edgecolors="none")
        ax.scatter(pr[j, 0], pr[j, 1], color="#d62728", s=22, edgecolors="black", lw=0.4, zorder=3)
        ax.set_title(f"{lbl} — joint all-9: {100*j.mean():.2f}%", fontsize=11, fontweight="bold")
        ax.set_xlabel("KPI-PC1"); ax.set_ylabel("KPI-PC2"); ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.grid(alpha=0.2, ls="--")
    sm = ScalarMappable(norm=norm, cmap="viridis"); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02); cb.set_label("# constraints satisfied", fontweight="bold")
    fig.suptitle("9-mutant KPI space (PCA, pooled over seeds) — joint class in red",
                 fontsize=13, fontweight="bold")
    fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out_png}")


def plot_per_model_r2(variants_ms, oracle, out, zoom=0.02):
    tmp = out / "_pooled_enzyme"; tmp.mkdir(parents=True, exist_ok=True)
    for lbl in ORDER:
        vdir = tmp / lbl.replace("-", "_"); vdir.mkdir(exist_ok=True)
        perts = []
        for m in MUTANTS:
            p = vdir / f"{m}.csv"
            pooled_enzyme_df(variants_ms[lbl], m).to_csv(p)
            perts.append((m, oracle["enzyme"][m], str(p)))
        safe = lbl.replace("-", "_")
        fig, _ = plot_per_model_r2_kde(
            perts, str(paths.EXP_FLUX),
            save_path=str(out / f"per_model_r2_{safe}.png"),
            show=False, zoom_quantile=zoom, dpi=300)
        plt.close(fig)
        print(f"[saved] {out / f'per_model_r2_{safe}.png'}  (pooled over {len(variants_ms[lbl]['seeds'])} seeds)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.PAPER_FIG_DIR / "capstone_stringent"))
    ap.add_argument("--continuous-target", choices=["cls", "p5"], default="cls",
                    help="Which continuous-variant runs to use: 'cls' = class-matched targets "
                         "(main, fair comparison; *_continuous_cls_* files) or 'p5' = fixed "
                         "5th-percentile KPI target (*_continuous_p5_* files; "
                         "targeted-generation comparison).")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="Restrict to these seeds (default: all discovered per variant).")
    ap.add_argument("--kpi-clip-pct", type=float, default=98.0,
                    help="Trim KPI values above this percentile in the KPI grid (zoom).")
    ap.add_argument("--no-r2", action="store_true", help="skip per-model R2 (slow) figures")
    args = ap.parse_args()
    if args.continuous_target == "cls":
        for lbl in ("CVAE-cont", "GAN-cont", "NF-cont"):
            VARIANT_PREFIX[lbl] = VARIANT_PREFIX[lbl] + "_cls"
        print("[target] continuous variants: class-matched (_cls) runs")
    else:
        for lbl in ("CVAE-cont", "GAN-cont", "NF-cont"):
            VARIANT_PREFIX[lbl] = VARIANT_PREFIX[lbl] + "_p5"
        print("[target] continuous variants: 5th-percentile target runs (_p5)")
        if args.out.rstrip("/").endswith("capstone_stringent"):
            args.out = args.out.rstrip("/") + "_p5"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    global ORDER
    variant_seeds = {}
    for lbl, pref in VARIANT_PREFIX.items():
        s = discover_seeds(pref, args.seeds)
        if s:
            variant_seeds[lbl] = s
        else:
            print(f"[skip] {lbl}: no seed folders found")
    ORDER = [l for l in VARIANT_PREFIX if l in variant_seeds]
    if not ORDER:
        raise SystemExit("No variant result folders found.")
    for l in ORDER:
        print(f"[variant] {l}: seeds {variant_seeds[l]}")

    variants_ms = {l: load_variant_ms(l, variant_seeds[l]) for l in ORDER}
    oracle = load_oracle(f"{paths.RUNS_DIR}/benchmark_{VARIANT_PREFIX[ORDER[0]]}_all_mutants_seed{variant_seeds[ORDER[0]][0]}")
    thr = thresholds(oracle)

    pd.set_option("display.width", 220, "display.max_columns", 20)
    per, _ = seed_metric_matrices(oracle, thr, out, allow_seeds=args.seeds)
    build_full_table(variants_ms, oracle, thr, out)

    plot_headline(variants_ms, oracle, thr, out / "headline_ladder_enrichment.png")
    plot_degree_hist(variants_ms, oracle, thr, out / "degree_histogram.png")
    plot_per_mutant(variants_ms, oracle, thr, out / "per_mutant_satisfaction.png")
    plot_kpi_grid(variants_ms, oracle, thr, out / "kpi_distribution_grid.png",
                  clip_pct=args.kpi_clip_pct)
    plot_kpi_pca(variants_ms, oracle, thr, out / "kpi_pca_scatter.png")

    # --- parameter-space figures ---
    km_by_method, _ = parameter_space_inputs(variants_ms, per)
    B.km_embedding_grid(km_by_method, oracle["kms"], ORDER, COLORS, out / "km_embedding.png")
    B.correlation_diff_heatmaps(km_by_method, oracle["kms"], ORDER, COLORS, out / "km_correlation_diff.png")
    B.per_parameter_mean_shift_heatmap(km_by_method, oracle["kms"], ORDER, out / "km_param_shift_heatmap.png")

    ref = ref_projection(oracle, thr)
    ref_top = ref["real_proj"]; scale_top = B._within_ref_scale(ref_top)
    rho_by_method = {l: B.extrapolation_rho(B.project_km(km_by_method[l], ref), ref_top, scale_top)
                     for l in ORDER if l in km_by_method}
    B.nnd_violin(rho_by_method, COLORS, ORDER, out / "extrapolation_violin.png",
                 ref_name="9-mutant broadly-good set")

    if not args.no_r2:
        plot_per_model_r2(variants_ms, oracle, out)
    print("[done]")


def parameter_space_inputs(variants_ms, per):
    """Pooled Km per variant + precision/recall mean/std for the parameter figures."""
    km_by_method = {}
    for l in ORDER:
        kms = [ps["km"] for ps in variants_ms[l]["per_seed"] if ps["km"] is not None]
        if kms:
            km_by_method[l] = pd.concat(kms, ignore_index=True)
    pr = {}
    if per is not None and not per.empty:
        for l in ORDER:
            sub = per[per["variant"] == l]
            if len(sub) and "precision" in sub:
                pr[l] = (sub["precision"].mean(), sub["precision"].std(),
                         sub["recall"].mean(), sub["recall"].std())
    return km_by_method, pr


if __name__ == "__main__":
    main()

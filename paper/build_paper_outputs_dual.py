#!/usr/bin/env python3
"""
Paper figures + metrics for the TWO-perturbation scenario (PGI @ 10% + PGL @ 30%),
for the six architecture/labelling variants, aggregated over seeds.

Seed handling (matches build_paper_outputs_capstone_stringent.py):
  * metric matrices  -> per seed + mean/std across seeds
  * distribution figures (per-perturbation KPI, joint scatter, Km boxplot) POOL seeds
Each variant auto-uses every seed it has under
  outputs/runs/benchmark_<prefix>_pgi_pgl_seed<seed>/  (restrict with --seeds).

Outputs -> results/paper_figures/pgi_pgl/
  metrics_per_seed.csv / metrics_mean_std(.csv/_pretty.csv)
  joint_kpi_scatter_pgi_pgl.png   - PGI vs PGL, good-fit box, joint models highlighted
  kpi_distribution_pgi.png / _pgl.png
  parameter_boxplot_pgi_pgl.png

Run (from the repository root):  python paper/build_paper_outputs_dual.py
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

from benchmark.metrics import core as B


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PERTS = ["PGI", "PGL"]
PCT = {"PGI": 10.0, "PGL": 30.0}            # conditional percentiles
NP_ = len(PERTS)

VARIANT_PREFIX = {
    "CVAE": "cvae", "GAN": "gan", "NF": "nf",
    "CVAE-cont": "cvae_continuous", "GAN-cont": "gan_continuous",
    "NF-cont": "nf_continuous",
}
COLORS = {"CVAE": "#1f77b4", "CVAE-cont": "#9ecae1",
          "GAN": "#2ca02c",  "GAN-cont": "#98df8a",
          "NF": "#9467bd",   "NF-cont": "#c5b0d5",
          "ORACLE": "#ff7f0e"}
ORDER = []


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _aligned_kpi(tables: Path, pattern):
    kpi = {p: pd.read_csv(B._one(tables / pattern(p)), index_col=0).iloc[:, 0] for p in PERTS}
    idx = kpi[PERTS[0]].index
    for p in PERTS[1:]:
        idx = idx.intersection(kpi[p].index)
    return {p: kpi[p].loc[idx] for p in PERTS}, idx


def discover_seeds(pref, allow=None):
    seeds = set()
    for d in paths.RUNS_DIR.glob(f"benchmark_{pref}_pgi_pgl_seed*"):
        m = re.search(r"seed(\d+)$", d.name)
        if m and (d / "tables").is_dir():
            seeds.add(int(m.group(1)))
    if allow:
        seeds &= set(allow)
    return sorted(seeds)


def load_seed(folder, km_file):
    run = Path(folder)
    kpi, idx = _aligned_kpi(run / "tables", lambda p: f"kpi_*_{p}.csv")
    enzyme = {p: str(B._one(run / "enzyme_perturbations" / f"*_{p}.csv")) for p in PERTS}
    if km_file and Path(km_file).exists():
        gen = pd.read_csv(km_file, index_col=0)
        ks = [int(str(i).split(",")[-1]) for i in idx]
        km = gen.loc[ks]; km.index = idx; n_gen = len(gen)
    else:
        km, n_gen = None, 2000
    return {"kpi": kpi, "enzyme": enzyme, "km": km, "n_gen": n_gen, "n_scored": len(idx)}


def load_variant_ms(lbl, seeds):
    pref = VARIANT_PREFIX[lbl]
    per = [load_seed(f"{paths.RUNS_DIR}/benchmark_{pref}_pgi_pgl_seed{s}",
                     f"{paths.SYNTH_DIR}/benchmark_{pref}_pgi_pgl_seed{s}.csv")
           for s in seeds]
    return {"seeds": seeds, "per_seed": per}


def load_oracle(any_folder):
    tables = Path(any_folder) / "tables"
    kpi, _ = _aligned_kpi(tables, lambda p: f"kpi_oracle_{p}.csv")
    kms = pd.read_csv(paths.ORACLE_KM, index_col=0)
    enzyme = {p: str(paths.oracle_flux(p)) for p in PERTS}
    exp = pd.read_csv(paths.EXP_FLUX, index_col=0)
    return {"kpi": kpi, "kms": kms, "enzyme": enzyme, "exp": exp, "idx": kpi[PERTS[0]].index}


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def thresholds(oracle):
    return {p: {"cond": float(np.percentile(oracle["kpi"][p].values, PCT[p])),
                "p1": float(np.percentile(oracle["kpi"][p].values, 1.0)),
                "best": float(oracle["kpi"][p].values.min())} for p in PERTS}


def joint_mask(kpi, thr):
    return np.logical_and.reduce([np.asarray(kpi[p]) <= thr[p]["cond"] for p in PERTS])


def pooled_kpi(v_ms):
    return {p: np.concatenate([ps["kpi"][p].values for ps in v_ms["per_seed"]]) for p in PERTS}


def diversity_reference(oracle, thr):
    """ORACLE models jointly satisfying both thresholds (the good region we emulate)."""
    idx = oracle["kms"].index.intersection(oracle["idx"])
    km = oracle["kms"].loc[idx]
    kpi = {p: oracle["kpi"][p].loc[idx] for p in PERTS}
    return km[joint_mask(kpi, thr)]


def ref_projection(oracle, thr):
    ref = diversity_reference(oracle, thr); cols = ref.columns
    rmean = np.log(ref.values).mean(0); rstd = np.log(ref.values).std(0); rstd[rstd == 0] = 1
    real = (np.log(ref.values) - rmean) / rstd
    ncomp = int(min(50, real.shape[1], real.shape[0] - 1))
    pmean, pcomps = B._pca_fit(real, ncomp)
    return {"cols": cols, "rmean": rmean, "rstd": rstd, "pmean": pmean, "pcomps": pcomps,
            "real_proj": B._pca_transform(real, pmean, pcomps), "n": len(ref)}


def km_prdc(km_df, ref):
    fp = B._pca_transform((np.log(km_df[ref["cols"]].values) - ref["rmean"]) / ref["rstd"],
                          ref["pmean"], ref["pcomps"])
    return B.prdc(ref["real_proj"], fp)


def pooled_enzyme_df(v_ms, p):
    dfs = []
    for i, ps in enumerate(v_ms["per_seed"]):
        e = pd.read_csv(ps["enzyme"][p], index_col=0)
        e = e.drop(columns=[c for c in ["Steady"] if c in e.columns])
        dfs.append(e.add_prefix(f"s{i}_"))
    return pd.concat(dfs, axis=1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metric_context(oracle, thr, verbose=True):
    """Reference sets shared by every run (joint base rate, diversity, extrapolation)."""
    o_kpi = {p: oracle["kpi"][p].loc[oracle["idx"]] for p in PERTS}
    or_joint = 100.0 * np.mean(joint_mask(o_kpi, thr))
    ref = ref_projection(oracle, thr)
    if verbose:
        print(f"[ref] ORACLE joint base rate = {or_joint:.3f}%  | diversity reference n = {ref['n']}")
    ref_top = ref["real_proj"]; scale_top = B._within_ref_scale(ref_top)
    o_full = B.project_km(oracle["kms"], ref)
    rng = np.random.default_rng(0)
    if len(o_full) > 4000:
        o_full = o_full[rng.choice(len(o_full), 4000, replace=False)]
    scale_full = B._within_ref_scale(o_full)
    return {"or_joint": or_joint, "ref": ref, "ref_top": ref_top, "scale_top": scale_top,
            "o_full": o_full, "scale_full": scale_full}


def run_metrics(folder, km_file, thr, ctx):
    """Paper metrics for one scored run folder (None if the run is incomplete)."""
    folder, km = Path(folder), Path(km_file)
    try:
        kpi, idx = _aligned_kpi(folder / "tables", lambda p: f"kpi_*_{p}.csv")
    except FileNotFoundError:
        return None
    ref, or_joint = ctx["ref"], ctx["or_joint"]
    joint = 100.0 * np.mean(joint_mask(kpi, thr))
    prec = rec = rho_top = rho_full = np.nan; n_gen = 2000
    if km.exists():
        gen = pd.read_csv(km, index_col=0); n_gen = len(gen)
        km_df = gen.loc[[int(str(i).split(",")[-1]) for i in idx]]
        dc = km_prdc(km_df, ref); prec, rec = dc["precision"], dc["recall"]
        fp = B.project_km(km_df, ref)
        rho_top = float(B._knn_mean_dist(fp, ctx["ref_top"]).mean() / ctx["scale_top"])
        rho_full = float(B._knn_mean_dist(fp, ctx["o_full"]).mean() / ctx["scale_full"])
    row = {"pct_survived": 100.0 * len(idx) / n_gen,
           "joint_pct": joint,
           "enrichment_vs_oracle": (joint / or_joint) if or_joint > 0 else np.nan}
    for p in PERTS:
        v = kpi[p].values
        row[f"{p}_mean_KPI"] = float(v.mean())
        row[f"{p}_pct_below_p{PCT[p]:g}"] = 100.0 * np.mean(v <= thr[p]["cond"])
        row[f"{p}_pct_outperform_best"] = 100.0 * np.mean(v < thr[p]["best"])
    row["precision"] = prec; row["recall"] = rec
    row["extrap_vs_top"] = rho_top; row["extrap_vs_full"] = rho_full
    return row


def seed_metric_matrices(oracle, thr, out, allow_seeds=None):
    ctx = metric_context(oracle, thr)
    rows = []
    for lbl, pref in VARIANT_PREFIX.items():
        for s in discover_seeds(pref, allow_seeds):
            folder = Path(f"{paths.RUNS_DIR}/benchmark_{pref}_pgi_pgl_seed{s}")
            km = Path(f"{paths.SYNTH_DIR}/benchmark_{pref}_pgi_pgl_seed{s}.csv")
            row = run_metrics(folder, km, thr, ctx)
            if row is not None:
                rows.append({"variant": lbl, "seed": s, **row})

    per = pd.DataFrame(rows)
    if per.empty:
        print("[warn] no seed runs found"); return per
    order = [l for l in VARIANT_PREFIX if l in set(per["variant"])]
    per["variant"] = pd.Categorical(per["variant"], categories=order, ordered=True)
    per = per.sort_values(["variant", "seed"])
    metrics = [c for c in per.columns if c not in ("variant", "seed")]
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
    pd.set_option("display.width", 240, "display.max_columns", 30)
    print("\n=== mean ± std across seeds ===\n" + pretty.to_string())
    return per


# ---------------------------------------------------------------------------
# Figures (distributions pooled over seeds)
# ---------------------------------------------------------------------------
def plot_joint_scatter(variants_ms, oracle, thr, out_png):
    tA, tB = thr["PGI"]["cond"], thr["PGL"]["cond"]
    panels = ([(l, pooled_kpi(variants_ms[l])) for l in ORDER]
              + [("ORACLE", {p: oracle["kpi"][p].loc[oracle["idx"]].values for p in PERTS})])
    ncol = 4; nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    xall = np.concatenate([k["PGI"] for _, k in panels])
    yall = np.concatenate([k["PGL"] for _, k in panels])
    xmax = np.percentile(xall, 98); ymax = np.percentile(yall, 98)
    for ax, (lbl, k) in zip(axes, panels):
        x, y = k["PGI"], k["PGL"]; j = (x <= tA) & (y <= tB)
        ax.scatter(x[~j], y[~j], s=6, color=COLORS[lbl], alpha=0.28, edgecolors="none")
        ax.scatter(x[j], y[j], s=11, color="#d62728", alpha=0.85, edgecolors="none")
        ax.axvline(tA, color="black", ls=":", lw=1.2); ax.axhline(tB, color="black", ls=":", lw=1.2)
        ax.axvspan(0, tA, ymin=0, ymax=tB / (ymax * 1.05), color="#d62728", alpha=0.05)
        ax.set_xlim(0, xmax * 1.05); ax.set_ylim(0, ymax * 1.05)
        ax.set_xlabel("PGI KPI"); ax.set_ylabel("PGL KPI")
        ax.set_title(f"{lbl} — joint good-fit: {100 * j.mean():.1f}%", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.2, ls="--")
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"PGI+PGL joint KPI (pooled over seeds) — good-fit box "
                 f"(PGI≤p{PCT['PGI']:g}, PGL≤p{PCT['PGL']:g})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out_png}")


def plot_kpi_dist(variants_ms, oracle, thr, pert, out_png, clip_pct=98.0):
    order = ORDER + ["ORACLE"]
    series = {**{l: pooled_kpi(variants_ms[l])[pert] for l in ORDER},
              "ORACLE": oracle["kpi"][pert].loc[oracle["idx"]].values}
    allv = np.concatenate([series[l] for l in order])
    lo, hi = float(np.min(allv)), float(np.percentile(allv, clip_pct))
    grid = np.linspace(max(0.0, lo - 0.02 * (hi - lo)), hi, 400)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for lbl in order:
        v = series[lbl]; v = v[v <= hi]
        dens = B._kde(v, grid)
        ax.fill_between(grid, dens, color=COLORS[lbl], alpha=0.12)
        ax.plot(grid, dens, color=COLORS[lbl], lw=2, label=f"{lbl} (mean={series[lbl].mean():.2f})")
    ax.axvline(thr[pert]["cond"], color="black", ls=":", lw=1.6,
               label=f"good-fit p{PCT[pert]:g} = {thr[pert]['cond']:.2f}")
    ax.set_xlim(grid[0], grid[-1]); ax.grid(alpha=0.25, ls="--")
    ax.set_xlabel("KPI (ζ-score)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax.set_title(f"{pert} (PGI+PGL scenario) — KPI distribution, pooled over seeds",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9, loc="upper left", bbox_to_anchor=(1.005, 1.0))
    fig.tight_layout(); fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out_png}")


def plot_parameter_boxplot(variants_ms, oracle, out_png, top_n=10):
    km_cols = oracle["kms"].columns
    logs = {}
    for l in ORDER:
        kms = [ps["km"] for ps in variants_ms[l]["per_seed"] if ps["km"] is not None]
        logs[l] = np.log(pd.concat(kms, ignore_index=True)[km_cols]) if kms else None
    log_or = np.log(oracle["kms"][km_cols]); or_mean = log_or.mean(axis=0)
    shift = pd.Series(0.0, index=km_cols)
    for l in ORDER:
        if logs[l] is not None:
            shift = np.maximum(shift, (logs[l].mean(axis=0) - or_mean).abs())
    top = shift.sort_values(ascending=False).head(top_n).index.tolist()

    groups = [l for l in ORDER if logs[l] is not None] + ["ORACLE"]
    ng = len(groups); width = 0.8 / ng
    fig, ax = plt.subplots(figsize=(max(11, top_n * 1.5), 6))
    for gi, lbl in enumerate(groups):
        data = log_or[top] if lbl == "ORACLE" else logs[lbl][top]
        pos = np.arange(top_n) + gi * width - 0.4 + width / 2
        bp = ax.boxplot([data[p].dropna().values for p in top], positions=pos,
                        widths=width * 0.9, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.1))
        for box in bp["boxes"]:
            box.set(facecolor=COLORS.get(lbl), alpha=0.8, edgecolor="black", linewidth=0.4)
    ax.set_xticks(np.arange(top_n)); ax.set_xticklabels(top, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(r"$\log(K_m)$", fontsize=12, fontweight="bold")
    ax.set_title(f"PGI+PGL — top {top_n} most-shifted $K_m$ parameters (pooled over seeds)",
                 fontsize=12.5, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS.get(g), alpha=0.8, edgecolor="black")
               for g in groups]
    ax.legend(handles, groups, fontsize=9, ncol=2, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, ls="--")
    fig.tight_layout(); fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.PAPER_FIG_DIR / "pgi_pgl"))
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--kpi-clip-pct", type=float, default=98.0)
    args = ap.parse_args()
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
        raise SystemExit("No pgi_pgl result folders found.")
    for l in ORDER:
        print(f"[variant] {l}: seeds {variant_seeds[l]}")

    variants_ms = {l: load_variant_ms(l, variant_seeds[l]) for l in ORDER}
    oracle = load_oracle(f"{paths.RUNS_DIR}/benchmark_{VARIANT_PREFIX[ORDER[0]]}_pgi_pgl_seed{variant_seeds[ORDER[0]][0]}")
    thr = thresholds(oracle)

    per = seed_metric_matrices(oracle, thr, out, allow_seeds=args.seeds)
    plot_joint_scatter(variants_ms, oracle, thr, out / "joint_kpi_scatter_pgi_pgl.png")
    for p in PERTS:
        plot_kpi_dist(variants_ms, oracle, thr, p, out / f"kpi_distribution_{p.lower()}.png",
                      clip_pct=args.kpi_clip_pct)
    plot_parameter_boxplot(variants_ms, oracle, out / "parameter_boxplot_pgi_pgl.png", top_n=args.top_n)

    # --- parameter-space figures ---
    km_by_method = {}
    for l in ORDER:
        kms = [ps["km"] for ps in variants_ms[l]["per_seed"] if ps["km"] is not None]
        if kms:
            km_by_method[l] = pd.concat(kms, ignore_index=True)
    B.km_embedding_grid(km_by_method, oracle["kms"], ORDER, COLORS, out / "km_embedding.png")
    B.correlation_diff_heatmaps(km_by_method, oracle["kms"], ORDER, COLORS, out / "km_correlation_diff.png")
    B.per_parameter_mean_shift_heatmap(km_by_method, oracle["kms"], ORDER, out / "km_param_shift_heatmap.png")

    ref = ref_projection(oracle, thr)
    ref_top = ref["real_proj"]; scale_top = B._within_ref_scale(ref_top)
    rho_by_method = {l: B.extrapolation_rho(B.project_km(km_by_method[l], ref), ref_top, scale_top)
                     for l in ORDER if l in km_by_method}
    B.nnd_violin(rho_by_method, COLORS, ORDER, out / "extrapolation_violin.png",
                 ref_name="PGI+PGL joint good-fit")
    print("[done]")


if __name__ == "__main__":
    main()

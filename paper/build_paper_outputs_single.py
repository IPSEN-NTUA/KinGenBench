#!/usr/bin/env python3
"""
Paper figures + metrics for the SINGLE-PGI scenario (PGI @ 10%), for the six
architecture/labelling variants, aggregated over seeds. Mirrors the dual /
capstone pipelines (seed-aware, mean±std metrics, pooled distribution figures,
consistent colours).

Each variant auto-uses every seed it has under
  outputs/runs/benchmark_<prefix>_pgi_thr<THR>_seed<seed>/   (restrict with --seeds).

Outputs -> results/paper_figures/pgi/
  metrics_per_seed.csv / metrics_mean_std(.csv/_pretty.csv)
  kpi_distribution_pgi.png     - all methods + ORACLE, pooled, outperform-best inset
  parameter_boxplot_pgi.png
  km_embedding.png / km_correlation_diff.png / km_param_shift_heatmap.png

Run (from the repository root):  python paper/build_paper_outputs_single.py
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
PERT = "PGI"
THR_PCT = 10.0          # conditioning percentile (folder tag pgi_thr10)
TOP_PCT = 1.0

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
TAG = f"pgi_thr{int(THR_PCT)}"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def discover_seeds(pref, allow=None):
    seeds = set()
    for d in paths.RUNS_DIR.glob(f"benchmark_{pref}_{TAG}_seed*"):
        m = re.search(r"seed(\d+)$", d.name)
        if m and (d / "tables").is_dir():
            seeds.add(int(m.group(1)))
    if allow:
        seeds &= set(allow)
    return sorted(seeds)


def load_seed(folder, km_file):
    run = Path(folder)
    kpi = pd.read_csv(B._one(run / "tables" / "kpi_*_pgi.csv"), index_col=0).iloc[:, 0]
    km = pd.read_csv(B._one(run / "tables" / "parameter_set_*_pgi.csv"), index_col=0)
    km = km.loc[kpi.index]
    enzyme = str(B._one(run / "enzyme_perturbations" / "enzyme_perturbation_*_pgi.csv"))
    n_gen = (sum(1 for _ in open(km_file)) - 1) if Path(km_file).exists() else 2000
    return {"kpi": kpi, "km": km, "enzyme": enzyme, "n_gen": n_gen, "n_scored": len(kpi)}


def load_variant_ms(lbl, seeds):
    pref = VARIANT_PREFIX[lbl]
    per = [load_seed(f"{paths.RUNS_DIR}/benchmark_{pref}_{TAG}_seed{s}",
                     f"{paths.SYNTH_DIR}/benchmark_{pref}_{TAG}_seed{s}.csv")
           for s in seeds]
    return {"seeds": seeds, "per_seed": per}


def load_oracle(any_folder):
    kpi = pd.read_csv(B._one(Path(any_folder) / "tables" / "kpi_oracle_pgi.csv"),
                      index_col=0).iloc[:, 0]
    kms = pd.read_csv(paths.ORACLE_KM, index_col=0)
    enzyme = str(paths.oracle_flux(PERT))
    exp = pd.read_csv(paths.EXP_FLUX, index_col=0)
    return {"kpi": kpi, "kms": kms, "enzyme": enzyme, "exp": exp}


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def thresholds(oracle):
    v = oracle["kpi"].values
    return {"cond": float(np.percentile(v, THR_PCT)),
            "p1": float(np.percentile(v, TOP_PCT)),
            "best": float(v.min())}


def pooled_kpi(v_ms):
    return np.concatenate([ps["kpi"].values for ps in v_ms["per_seed"]])


def diversity_reference(oracle, thr):
    idx = oracle["kms"].index.intersection(oracle["kpi"].index)
    hp = oracle["kpi"].loc[idx].values <= thr["cond"]
    return oracle["kms"].loc[idx][hp]


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


def pooled_enzyme_df(v_ms):
    dfs = []
    for i, ps in enumerate(v_ms["per_seed"]):
        e = pd.read_csv(ps["enzyme"], index_col=0)
        e = e.drop(columns=[c for c in ["Steady"] if c in e.columns])
        dfs.append(e.add_prefix(f"s{i}_"))
    return pd.concat(dfs, axis=1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metric_context(oracle, thr, verbose=True):
    """Reference sets shared by every run (diversity / extrapolation)."""
    ref = ref_projection(oracle, thr)
    if verbose:
        print(f"[ref] ORACLE p{THR_PCT:g}={thr['cond']:.3f}  p{TOP_PCT:g}={thr['p1']:.3f}  "
              f"best={thr['best']:.3f}  | diversity reference n={ref['n']}")
    # extrapolation references (top-performer good set + full ORACLE), same PCA space
    ref_top = ref["real_proj"]; scale_top = B._within_ref_scale(ref_top)
    o_full = B.project_km(oracle["kms"], ref)
    rng = np.random.default_rng(0)
    if len(o_full) > 4000:
        o_full = o_full[rng.choice(len(o_full), 4000, replace=False)]
    scale_full = B._within_ref_scale(o_full)
    return {"ref": ref, "ref_top": ref_top, "scale_top": scale_top,
            "o_full": o_full, "scale_full": scale_full}


def run_metrics(folder, km_file, thr, ctx):
    """Paper metrics for one scored run folder (None if the run is incomplete)."""
    folder, km_file = Path(folder), Path(km_file)
    try:
        kpi = pd.read_csv(B._one(folder / "tables" / "kpi_*_pgi.csv"), index_col=0).iloc[:, 0]
    except FileNotFoundError:
        return None
    ref = ctx["ref"]
    v = kpi.values
    n_gen = (sum(1 for _ in open(km_file)) - 1) if km_file.exists() else 2000
    km = pd.read_csv(B._one(folder / "tables" / "parameter_set_*_pgi.csv"), index_col=0).loc[kpi.index]
    dc = km_prdc(km, ref)
    fp = B.project_km(km, ref)
    return {"pct_survived": 100.0 * len(v) / n_gen,
            "mean_KPI": float(v.mean()),
            f"pct_below_p{THR_PCT:g}": 100.0 * np.mean(v <= thr["cond"]),
            f"pct_top{TOP_PCT:g}": 100.0 * np.mean(v <= thr["p1"]),
            "pct_outperform_best": 100.0 * np.mean(v < thr["best"]),
            "precision": dc["precision"], "recall": dc["recall"],
            "extrap_vs_top": float(B._knn_mean_dist(fp, ctx["ref_top"]).mean() / ctx["scale_top"]),
            "extrap_vs_full": float(B._knn_mean_dist(fp, ctx["o_full"]).mean() / ctx["scale_full"])}


def seed_metric_matrices(oracle, thr, out, allow_seeds=None):
    ctx = metric_context(oracle, thr)
    rows = []
    for lbl, pref in VARIANT_PREFIX.items():
        for s in discover_seeds(pref, allow_seeds):
            folder = Path(f"{paths.RUNS_DIR}/benchmark_{pref}_{TAG}_seed{s}")
            km_file = Path(f"{paths.SYNTH_DIR}/benchmark_{pref}_{TAG}_seed{s}.csv")
            row = run_metrics(folder, km_file, thr, ctx)
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
# Figures
# ---------------------------------------------------------------------------
def plot_kpi(variants_ms, oracle, thr, out_png):
    series = {l: pooled_kpi(variants_ms[l]) for l in ORDER}
    series["ORACLE"] = oracle["kpi"].values
    order = ORDER + ["ORACLE"]
    # temporarily point the shared plotter's colour map at ours
    saved = B.METHOD_COLORS
    B.METHOD_COLORS = COLORS
    try:
        B.CONDITIONAL_PCT = THR_PCT
        B.plot_kpi_dist_core(series, order, thr["cond"], thr["best"],
                             f"{PERT} — KPI distribution (pooled over seeds) vs ORACLE",
                             out_png, cond_pct=THR_PCT)
    finally:
        B.METHOD_COLORS = saved


def plot_parameter_boxplot(variants_ms, oracle, out_png, top_n=10):
    cols = oracle["kms"].columns
    logs = {}
    for l in ORDER:
        kms = [ps["km"] for ps in variants_ms[l]["per_seed"]]
        logs[l] = np.log(pd.concat(kms, ignore_index=True)[cols])
    log_or = np.log(oracle["kms"][cols]); or_mean = log_or.mean(axis=0)
    shift = pd.Series(0.0, index=cols)
    for l in ORDER:
        shift = np.maximum(shift, (logs[l].mean(axis=0) - or_mean).abs())
    top = shift.sort_values(ascending=False).head(top_n).index.tolist()
    groups = ORDER + ["ORACLE"]; ng = len(groups); w = 0.8 / ng
    fig, ax = plt.subplots(figsize=(max(11, top_n * 1.5), 6))
    for gi, lbl in enumerate(groups):
        data = log_or[top] if lbl == "ORACLE" else logs[lbl][top]
        pos = np.arange(top_n) + gi * w - 0.4 + w / 2
        bp = ax.boxplot([data[p].dropna().values for p in top], positions=pos, widths=w * 0.9,
                        patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=1.1))
        for box in bp["boxes"]:
            box.set(facecolor=COLORS.get(lbl), alpha=0.8, edgecolor="black", linewidth=0.4)
    ax.set_xticks(np.arange(top_n)); ax.set_xticklabels(top, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(r"$\log(K_m)$", fontsize=12, fontweight="bold")
    ax.set_title(f"{PERT} — top {top_n} most-shifted $K_m$ parameters (pooled over seeds)",
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
    ap.add_argument("--out", default=str(paths.PAPER_FIG_DIR / "pgi"))
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--top-n", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    global ORDER
    variant_seeds = {}
    for lbl, pref in VARIANT_PREFIX.items():
        s = discover_seeds(pref, args.seeds)
        if s:
            variant_seeds[lbl] = s
        else:
            print(f"[skip] {lbl}: no {TAG} seed folders found")
    ORDER = [l for l in VARIANT_PREFIX if l in variant_seeds]
    if not ORDER:
        raise SystemExit(f"No {TAG} result folders found.")
    for l in ORDER:
        print(f"[variant] {l}: seeds {variant_seeds[l]}")

    variants_ms = {l: load_variant_ms(l, variant_seeds[l]) for l in ORDER}
    oracle = load_oracle(f"{paths.RUNS_DIR}/benchmark_{VARIANT_PREFIX[ORDER[0]]}_{TAG}_seed{variant_seeds[ORDER[0]][0]}")
    thr = thresholds(oracle)

    seed_metric_matrices(oracle, thr, out, allow_seeds=args.seeds)
    plot_kpi(variants_ms, oracle, thr, out / "kpi_distribution_pgi.png")
    plot_parameter_boxplot(variants_ms, oracle, out / "parameter_boxplot_pgi.png", top_n=args.top_n)

    km_by_method = {l: pd.concat([ps["km"] for ps in variants_ms[l]["per_seed"]], ignore_index=True)
                    for l in ORDER}
    B.km_embedding_grid(km_by_method, oracle["kms"], ORDER, COLORS, out / "km_embedding.png")
    B.correlation_diff_heatmaps(km_by_method, oracle["kms"], ORDER, COLORS, out / "km_correlation_diff.png")
    B.per_parameter_mean_shift_heatmap(km_by_method, oracle["kms"], ORDER, out / "km_param_shift_heatmap.png")

    # extrapolation violin (per-model ρ vs the scenario top-performers, pooled seeds)
    ref = ref_projection(oracle, thr)
    ref_top = ref["real_proj"]; scale_top = B._within_ref_scale(ref_top)
    rho_by_method = {l: B.extrapolation_rho(B.project_km(km_by_method[l], ref), ref_top, scale_top)
                     for l in ORDER}
    B.nnd_violin(rho_by_method, COLORS, ORDER, out / "extrapolation_violin.png",
                 ref_name=f"{PERT} top-performers")
    print("[done]")


if __name__ == "__main__":
    main()

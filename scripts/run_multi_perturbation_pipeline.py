#!/usr/bin/env python3
"""
Run multiple enzyme-perturbation experiments end-to-end and produce all
the plots needed for a multi-mutant comparison against the initial
oracle ensemble.

Pipeline stages
---------------
1. Stability + pruning (once per Km file) — same logic as
   `run_all_mutant_pipeline.py`.
2. Parallel simulation of every requested perturbation — also reused
   from `run_all_mutant_pipeline.py`.
3. Per-perturbation scoring with the ζ-score (default), referenced
   against the initial oracle ensemble for that mutant. SMAPE and MAPE
   are also supported via --scoring.
4. Per-perturbation plots (per mutant):
      - KPI histogram (candidate vs oracle)
      - oracle-threshold barplot
5. One Km-comparison plot for the run (candidate Km set vs oracle Km set).
6. One UpSet plot showing how the candidate ensemble's "best
   performers" overlap across the chosen mutants, with the membership
   threshold per mutant taken from the oracle KPI distribution. An
   oracle-only UpSet plot is also produced for comparison.

ζ-scoring requires the initial oracle ensemble's enzyme-perturbation
CSVs (one per mutant). Either pass them via the directory shortcut
`--oracle-enzyme-dir` (files are auto-resolved as
``<dir>/<MUTANT>.csv``) or supply per-mutant overrides via
``--oracle-enzyme-files PGI:./path/PGI.csv ...``.

Example
-------
    python run_multi_perturbation_pipeline.py \\
        --km-file outputs/synthetic_parameters/my_model.csv \\
        --perturbations PGI:0.06 PGL:0.10 RPI:0.50 TALA:0.33 PYK:0.34 G6PDH2r:0.013 \\
        --oracle-enzyme-dir data/oracle/perturbed_fluxes \\
        --scoring zscore \\
        --threshold-pct 20 \\
        --n-jobs 5
"""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # repo root
from benchmark import paths

import argparse
import builtins
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def slugify(value: str) -> str:
    value = Path(value).stem if value else "model"
    value = re.sub(r"[^A-Za-z0-9_\-]+", "_", value).strip("_")
    return value or "model"


def _parse_perturbation(pert_str: str) -> Tuple[str, float]:
    if ":" not in pert_str:
        raise argparse.ArgumentTypeError(
            f"Invalid --perturbations entry '{pert_str}'. Use ENZYME:CHANGE, e.g. PGI:0.06"
        )
    enzyme, change = pert_str.split(":", 1)
    return enzyme.strip(), float(change)


def _parse_kv_path(s: str) -> Tuple[str, str]:
    if ":" not in s:
        raise argparse.ArgumentTypeError(
            f"Invalid mutant:path entry '{s}'. Use MUTANT:./path/to.csv"
        )
    k, v = s.split(":", 1)
    return k.strip(), v.strip()


def _parse_kv_float(s: str) -> Tuple[str, float]:
    if ":" not in s:
        raise argparse.ArgumentTypeError(
            f"Invalid mutant:percentile entry '{s}'. Use MUTANT:PCT, e.g. PGI:10"
        )
    k, v = s.split(":", 1)
    return k.strip(), float(v)


def _parse_sigma_floor(s: str):
    """Accept 'auto' or a non-negative float for --sigma-floor."""
    if s.lower() == "auto":
        return "auto"
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--sigma-floor must be 'auto' or a non-negative float (got {s!r})."
        )
    if v < 0:
        raise argparse.ArgumentTypeError("--sigma-floor must be non-negative.")
    return v


def as_dataframe(obj, name: str) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_frame(name=name)
    arr = np.asarray(obj)
    if arr.ndim == 1:
        return pd.DataFrame({name: arr})
    return pd.DataFrame(arr)


def save_table(obj, path: Path, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    as_dataframe(obj, name=name).to_csv(path)
    print(f"[saved] {path}")


def save_current_figure(path: Path, dpi: int = 300) -> None:
    """Save the current Matplotlib figure and close it."""
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    if fig is None or not fig.axes:
        print(f"[warning] No active figure to save for {path.name}")
        return
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(
        description="Multi-perturbation end-to-end pipeline (stability -> "
                    "perturbations -> ζ-scoring -> plots -> UpSet).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Inputs
    p.add_argument("--km-file", required=True,
                   help="Candidate Km parameter-set CSV (rows=models, cols=Kms).")
    p.add_argument("--model-name", default=None,
                   help="Run name. Defaults to the --km-file stem.")
    p.add_argument(
        "--perturbations", nargs="+", type=_parse_perturbation,
        default=[
            ("PGI",     0.06),
            ("PGL",     0.10),
            ("RPI",     0.50),
            ("TALA",    0.33),
            ("PYK",     0.34),
            # G6PDH2r perturbation from GenAI.ipynb cell 17: a near-total
            # knockdown (vmax * 0.013, i.e. 1.3% of baseline). The oracle
            # baseline lives at data/oracle/perturbed_fluxes/G6PDH2r.csv.gz and
            # benchmark.scoring.flux_utils remaps G6PDH2r->PGL for the
            # key-reaction set in compute_KPI.
            ("G6PDH2r", 0.013),
        ],
        help="ENZYME:CHANGE entries.",
    )

    # Output layout
    p.add_argument("--results-root", default=str(paths.RUNS_DIR),
                   help="Top-level output folder.")
    p.add_argument("--output-dir", default=None,
                   help="Exact output directory. Defaults to <results-root>/<model-name>.")

    # Heavy model files (same defaults as run_all_mutant_pipeline.py)
    p.add_argument("--tmodel", default=str(paths.TMODEL))
    p.add_argument("--kmodel", default=str(paths.KMODEL))
    p.add_argument("--samples", default=str(paths.TFA_SAMPLE))

    # Stability / pruning / perturbation knobs
    p.add_argument("--params-index-col", type=int, default=0)
    p.add_argument("--stability-sample-row", type=int, default=0)
    p.add_argument("--perturbation-sample-row", type=int, default=144)
    p.add_argument("--max-eigenvalue", type=float, default=-10.0)
    p.add_argument("--ncpu-compile", type=int, default=4)
    p.add_argument("--n-jobs", type=int, default=5,
                   help="Parallel workers (one per perturbation).")
    p.add_argument("--flux-scaling", type=float, default=456300.0)

    # Skip flags
    p.add_argument("--skip-perturbation", action="store_true",
                   help="Skip stability+perturbation, reuse existing per-mutant CSVs "
                        "under <output-dir>/enzyme_perturbations/.")
    p.add_argument("--skip-scoring", action="store_true",
                   help="Stop after the perturbation stage.")
    p.add_argument("--skip-upset", action="store_true",
                   help="Skip the UpSet plot stage.")
    p.add_argument("--skip-pred-vs-obs", action="store_true",
                   help="Skip the per-perturbation predicted-vs-observed scatter "
                        "+ ζ-score bar plot.")
    p.add_argument("--skip-per-model-r2", action="store_true",
                   help="Skip the per-model R² KDE figure (one panel per mutant) "
                        "that compares the distribution of per-model fits between "
                        "the candidate and oracle ensembles.")
    p.add_argument("--per-model-r2-clip", nargs=2, type=float,
                   metavar=("LO", "HI"), default=[-1.0, 1.0],
                   help="Visualisation clip range for per-model R². Per-model R² "
                        "can be very negative for poor models; clipping keeps the "
                        "KDE readable.")
    p.add_argument("--per-model-r2-threshold", type=float, default=0.5,
                   help="Threshold used in the per-panel annotation "
                        "(fraction of models with R² >= threshold).")
    p.add_argument("--keep-pruned-hdf5", action="store_true",
                   help="Keep the intermediate pruned-parameter HDF5 in the output dir.")

    # Scoring inputs
    p.add_argument("--scoring", choices=["mape", "smape", "zscore"], default="zscore",
                   help="Per-reaction error function used to build the KPI.")
    p.add_argument("--oracle-km-file", default=str(paths.ORACLE_KM),
                   help="Oracle Km CSV (used as the Km feature source for all KPIs and "
                        "as the comparison baseline in the Km-means plot).")
    p.add_argument("--flux-exp-file", default=str(paths.EXP_FLUX),
                   help="Experimental flux CSV with one column per perturbation (lowercase).")
    p.add_argument("--oracle-enzyme-dir", default=str(paths.ORACLE_FLUX_DIR),
                   help="Directory containing initial oracle enzyme CSVs named <MUTANT>.csv(.gz). "
                        "Used as the ζ-score reference per mutant and as the oracle baseline "
                        "in every per-mutant plot.")
    p.add_argument("--oracle-enzyme-files", nargs="*", type=_parse_kv_path, default=[],
                   help="Per-mutant overrides for oracle enzyme CSVs "
                        "(e.g. PGI:./path/PGI.csv).")

    # Plot knobs
    p.add_argument("--thresholds", default="0.2,0.1,0.01,0.00001",
                   help="Comma-separated oracle-percentile thresholds for the per-mutant "
                        "threshold bar plot.")
    p.add_argument("--threshold-pct-per-mutant", nargs="*", type=_parse_kv_float, default=[],
                   help="Per-mutant oracle KPI percentiles for the UpSet 'good-fit' "
                        "membership, e.g. PGI:10 PGL:30. Any mutant not listed falls "
                        "back to the scalar --threshold-pct.")
    p.add_argument("--threshold-abs-per-mutant", nargs="*", type=_parse_kv_float, default=[],
                   help="Per-mutant ABSOLUTE KPI threshold values (composite thresholds), "
                        "e.g. PGI:8.13 PGL:4.64. Takes precedence over the percentile "
                        "options for the listed mutants; a model is 'good-fit' for that "
                        "mutant if its KPI <= the given value.")
    p.add_argument("--upset-by-degree", action="store_true",
                   help="Also produce a degree-distribution plot: number of models "
                        "satisfying exactly k experiments (one bar per class, 1..N). "
                        "Recommended when the full UpSet has too many conditions.")
    p.add_argument("--upset-degree-include-zero", action="store_true",
                   help="Include the k=0 class (models satisfying no experiment) in "
                        "the --upset-by-degree plot.")
    p.add_argument("--threshold-pct", type=float, default=20.0,
                   help="Oracle KPI percentile used to define the UpSet 'good-fit' "
                        "set per mutant.")
    p.add_argument("--hist-xmin", type=float, default=0.0)
    p.add_argument("--hist-xmax", type=float, default=20.0)
    p.add_argument("--hist-width", type=float, default=12.9)
    p.add_argument("--hist-height", type=float, default=7.52)
    p.add_argument("--compare-km-name", default="ORACLE",
                   help="Legend label for the comparison Km set.")
    p.add_argument("--dpi", type=int, default=300)

    # Predicted-vs-observed plot knobs
    p.add_argument("--exclude-from-scatter", nargs="*", default=["PGM", "CO2t"],
                   help="Reactions kept in the bar plot but skipped in the scatter "
                        "panel (used to prevent a few extreme points from "
                        "dominating the axes).")
    p.add_argument("--use-sigma-floor", action="store_true",
                   help="Apply a lower bound to sigma_r in the ζ-score bar plot. "
                        "Mitigates blow-ups on near-constant-residual reactions.")
    p.add_argument("--sigma-floor", type=_parse_sigma_floor, default="auto",
                   help="Floor for sigma_r when --use-sigma-floor is set. "
                        "Either 'auto' (0.05 × median(sigma_r)) or a non-negative float. "
                        "Ignored unless --use-sigma-floor is also passed.")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Resolve which oracle enzyme CSV to use per mutant
# ---------------------------------------------------------------------------
def _oracle_file(d: Path, mutant: str) -> Path:
    """<d>/<MUTANT>.csv.gz (shipped) or <d>/<MUTANT>.csv (uncompressed)."""
    gz = d / f"{mutant}.csv.gz"
    return gz if gz.exists() else d / f"{mutant}.csv"


def resolve_oracle_enzyme_files(args, mutants: List[str]) -> Dict[str, str]:
    overrides = dict(args.oracle_enzyme_files)
    out = {}
    for m in mutants:
        if m in overrides:
            out[m] = overrides[m]
        else:
            out[m] = str(_oracle_file(Path(args.oracle_enzyme_dir), m))
        if not Path(out[m]).exists():
            raise FileNotFoundError(
                f"Oracle enzyme CSV for mutant {m} not found at {out[m]}. "
                "Provide --oracle-enzyme-dir pointing to a folder with <MUTANT>.csv "
                "files, or pass --oracle-enzyme-files {m}:./path/to/csv."
            )
    return out


# ---------------------------------------------------------------------------
# Stage 1+2: stability + parallel perturbations (reuse run_all_mutant_pipeline)
# ---------------------------------------------------------------------------
def run_perturbations(args, model_name: str, output_dir: Path) -> Dict[str, Path]:
    """Run stability+pruning and then every requested perturbation in parallel.
    Returns a dict mapping MUTANT -> output CSV path."""
    enzyme_dir = output_dir / "enzyme_perturbations"
    enzyme_dir.mkdir(parents=True, exist_ok=True)

    enzyme_csvs = {
        mutant: enzyme_dir / f"{model_name}_{mutant}.csv"
        for mutant, _ in args.perturbations
    }

    if args.skip_perturbation:
        for mutant, p in enzyme_csvs.items():
            if not p.exists():
                raise FileNotFoundError(
                    f"--skip-perturbation set but expected enzyme CSV is missing: {p}"
                )
        print(f"[stage] Reusing existing per-mutant enzyme CSVs in {enzyme_dir}")
        return enzyme_csvs

    # Reuse the heavy lifting that already lives in run_all_mutant_pipeline.py
    from benchmark.sim.all_mutant_pipeline import (
        run_stability_and_prune,
        simulate_one_perturbation,
    )
    from pytfa.io.json import load_json_model
    from skimpy.io.yaml import load_yaml_model

    index_col = None if args.params_index_col < 0 else args.params_index_col
    print(f"[load] Reading initial Km parameter set from {args.km_file}")
    params_df = pd.read_csv(args.km_file, index_col=index_col)

    print(f"[load] Reading TFA samples from {args.samples}")
    samples = pd.read_csv(args.samples, header=0, index_col=0)

    print("[load] Loading tmodel/kmodel...")
    tmodel = load_json_model(args.tmodel)
    kmodel = load_yaml_model(args.kmodel)

    pruned = run_stability_and_prune(
        params_df=params_df,
        kmodel=kmodel,
        tmodel=tmodel,
        samples=samples,
        stability_sample_row=args.stability_sample_row,
        max_eigenvalue=args.max_eigenvalue,
        ncpu_compile=args.ncpu_compile,
    )

    # Stash pruned parameters so the loky workers can reload them.
    if args.keep_pruned_hdf5:
        pruned_path = str(output_dir / "tables" / "pruned_parameters.hdf5")
        Path(pruned_path).parent.mkdir(parents=True, exist_ok=True)
        cleanup_pruned = False
    else:
        tmp_dir = tempfile.mkdtemp(prefix="multi_perturbation_")
        pruned_path = os.path.join(tmp_dir, "pruned_parameters.hdf5")
        cleanup_pruned = True
    print(f"[prune] Saving pruned parameters to {pruned_path}")
    pruned.save(pruned_path)

    # Parallel perturbations
    from joblib import Parallel, delayed
    print(f"[perturb] Launching {len(args.perturbations)} perturbations on "
          f"{args.n_jobs} workers...")
    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
        delayed(simulate_one_perturbation)(
            perturbation,
            pruned_path,
            samples,
            args.tmodel,
            args.kmodel,
            args.perturbation_sample_row,
            args.flux_scaling,
        )
        for perturbation in args.perturbations
    )

    # Save one CSV per perturbation under enzyme_perturbations/.
    for enzyme_id, df in results:
        out_path = enzyme_csvs[enzyme_id]
        df.to_csv(out_path)
        print(f"[save] {enzyme_id}: wrote {df.shape[0]} reactions x "
              f"{df.shape[1]} models -> {out_path}")

    if cleanup_pruned:
        try:
            shutil.rmtree(os.path.dirname(pruned_path))
        except OSError:
            pass

    return enzyme_csvs


# ---------------------------------------------------------------------------
# Stage 3+4: scoring + per-mutant plots
# ---------------------------------------------------------------------------
def import_scoring_dependencies():
    """Import the same scoring/plot helpers used by run_pgi_pipeline.py."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401

    builtins.display = print  # for libraries that expect IPython's display

    from benchmark.scoring.flux_utils import (
        build_dataset,
        build_dataset_smape,
        build_dataset_zscore,
    )
    from benchmark.scoring.plots import (
        compare_km_means,
        plot_error_histograms_vibrant,
        plot_thresholds_per_reaction,
    )
    from benchmark.scoring.predicted_vs_observed import (
        plot_predicted_vs_observed,
        plot_per_model_r2_kde,
    )
    return {
        "build_dataset": build_dataset,
        "build_dataset_smape": build_dataset_smape,
        "build_dataset_zscore": build_dataset_zscore,
        "compare_km_means": compare_km_means,
        "plot_error_histograms_vibrant": plot_error_histograms_vibrant,
        "plot_thresholds_per_reaction": plot_thresholds_per_reaction,
        "plot_predicted_vs_observed": plot_predicted_vs_observed,
        "plot_per_model_r2_kde": plot_per_model_r2_kde,
    }


def _score_pair(args, scoring_deps, mutant: str, enzyme_csv: Path, oracle_enzyme_file: str):
    """Return (df_candidate, kpi_candidate, df_oracle, kpi_oracle) under args.scoring."""
    build_dataset = scoring_deps["build_dataset"]
    build_dataset_smape = scoring_deps["build_dataset_smape"]
    build_dataset_zscore = scoring_deps["build_dataset_zscore"]

    def _go(enzyme_file: str):
        if args.scoring == "mape":
            return build_dataset(
                km_file=args.oracle_km_file,
                flux_exp_file=args.flux_exp_file,
                enzyme_file=enzyme_file,
                cut_reaction=mutant,
            )
        if args.scoring == "smape":
            return build_dataset_smape(
                km_file=args.oracle_km_file,
                flux_exp_file=args.flux_exp_file,
                enzyme_file=enzyme_file,
                cut_reaction=mutant,
            )
        if args.scoring == "zscore":
            return build_dataset_zscore(
                km_file=args.oracle_km_file,
                flux_exp_file=args.flux_exp_file,
                enzyme_file=enzyme_file,
                oracle_enzyme_file=oracle_enzyme_file,
                cut_reaction=mutant,
            )
        raise ValueError(f"Unknown --scoring value: {args.scoring}")

    df_cand, _, kpi_cand = _go(str(enzyme_csv))
    df_or, _, kpi_or = _go(oracle_enzyme_file)
    return df_cand, kpi_cand, df_or, kpi_or


def run_per_mutant_scoring_and_plots(
    args, scoring_deps, model_name: str, output_dir: Path,
    enzyme_csvs: Dict[str, Path], oracle_files: Dict[str, str],
) -> Dict[str, dict]:
    """For each mutant: score candidate+oracle, save KPIs, draw the histogram
    and the oracle-threshold bar plot. Returns per-mutant payloads for the
    later Km/UpSet stages."""
    import matplotlib.pyplot as plt

    plot_error_histograms_vibrant = scoring_deps["plot_error_histograms_vibrant"]
    plot_thresholds_per_reaction = scoring_deps["plot_thresholds_per_reaction"]
    plot_predicted_vs_observed = scoring_deps["plot_predicted_vs_observed"]

    tables_dir = output_dir / "tables"
    images_dir = output_dir / "images"
    tables_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    threshold_values = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    per_mutant = {}
    for mutant, _change in args.perturbations:
        print(f"[score] {mutant}: scoring with {args.scoring} "
              f"(oracle ref = {oracle_files[mutant]})")
        df_cand, kpi_cand, df_or, kpi_or = _score_pair(
            args, scoring_deps, mutant, enzyme_csvs[mutant], oracle_files[mutant],
        )

        save_table(kpi_cand, tables_dir / f"kpi_{model_name}_{mutant}.csv",
                   f"kpi_{model_name}_{mutant}")
        save_table(kpi_or, tables_dir / f"kpi_oracle_{mutant}.csv",
                   f"kpi_oracle_{mutant}")

        # ----- KPI histogram -----
        plt.figure(figsize=(args.hist_width, args.hist_height))
        plot_error_histograms_vibrant(
            {model_name: kpi_cand, "ORACLE": kpi_or},
            title=mutant,
            xlim=(args.hist_xmin, args.hist_xmax),
        )
        save_current_figure(images_dir / f"histogram_{model_name}_{mutant}.png", dpi=args.dpi)

        # ----- oracle-threshold barplot -----
        threshold_data = {
            "km":  {model_name: df_cand, "ORACLE": df_or},
            "kpi": {mutant: {model_name: kpi_cand, "ORACLE": kpi_or}},
        }
        thresholds_config = {"plot": {mutant: threshold_values}}
        plot_thresholds_per_reaction(
            data=threshold_data,
            thresholds=thresholds_config,
            output_dir=str(images_dir),
        )

        # ----- predicted-vs-observed scatter + ζ-score barplot -----
        if not args.skip_pred_vs_obs:
            pvo_path = images_dir / f"pred_vs_obs_{model_name}_{mutant}.png"
            plot_predicted_vs_observed(
                oracle_file=oracle_files[mutant],
                genki_file=str(enzyme_csvs[mutant]),
                exp_file=args.flux_exp_file,
                perturbed_reaction=mutant,
                scoring="zeta",
                exclude_from_scatter=tuple(args.exclude_from_scatter),
                save_path=str(pvo_path),
                dpi=args.dpi,
                use_sigma_floor=args.use_sigma_floor,
                sigma_floor=args.sigma_floor,
                show=False,
            )
            print(f"[saved] {pvo_path}")

        per_mutant[mutant] = {
            "df_cand": df_cand,
            "df_or": df_or,
            "kpi_cand": kpi_cand,
            "kpi_or": kpi_or,
        }

    return per_mutant


# ---------------------------------------------------------------------------
# Stage 5: single Km-comparison plot
# ---------------------------------------------------------------------------
def run_km_comparison(args, scoring_deps, model_name: str, output_dir: Path) -> None:
    """One Km-means comparison: candidate Km file vs oracle Km file."""
    import matplotlib.pyplot as plt  # noqa: F401

    images_dir = output_dir / "images"
    compare_km_means = scoring_deps["compare_km_means"]

    graph_kms_df = pd.read_csv(args.km_file, index_col=0)
    comparison_kms_df = pd.read_csv(args.oracle_km_file, index_col=0)

    compare_km_means(graph_kms_df, comparison_kms_df,
                     name1=model_name, name2=args.compare_km_name)
    save_current_figure(images_dir / f"compare_km_{model_name}.png", dpi=args.dpi)


# ---------------------------------------------------------------------------
# Stage 6: UpSet plot — thresholds derived from the initial oracle ensemble
# ---------------------------------------------------------------------------
# Colours by number of mutant constraints satisfied (1..5). Mirrors the
# AGREEMENT_COLORS palette used in plots_for_publication.ipynb.
AGREEMENT_COLORS = {
    1: "#6223C8",
    2: "#881990",
    3: "#D06A34",
    4: "#E68883",
    5: "#F3D856",
    6: "#FFD27F",  # extra slots in case future runs use >5 mutants
    7: "#FFE9A8",
    8: "#B5D99C",
    9: "#7FB77E",
}


def _labels_below_oracle_threshold(
    kpi_by_mutant: Dict[str, pd.Series],
    oracle_thresholds: Dict[str, float],
) -> pd.DataFrame:
    """Align KPI series across mutants on the inner join of their indices and
    return a bool DataFrame: row = model, col = mutant, value = KPI <= threshold."""
    aligned = pd.concat(
        {m: pd.Series(s, dtype=float) for m, s in kpi_by_mutant.items()},
        axis=1, join="inner",
    )
    return pd.DataFrame(
        {m: aligned[m] <= oracle_thresholds[m] for m in aligned.columns},
        index=aligned.index,
    )


def _plot_upset_one(labels: pd.DataFrame, title: str, savepath: Path, dpi: int):
    from itertools import combinations
    from upsetplot import UpSet, from_indicators
    import matplotlib.pyplot as plt

    mutants = list(labels.columns)
    upset_data = from_indicators(indicators=mutants, data=labels)
    up = UpSet(
        upset_data,
        subset_size="count",
        show_percentages="{:.1%}",
        sort_by="cardinality",
        element_size=50,
    )
    for k in range(1, len(mutants) + 1):
        for combo in combinations(mutants, k):
            up.style_subsets(
                present=list(combo),
                facecolor=AGREEMENT_COLORS.get(k, "#888888"),
                edgecolor="black",
                linewidth=0.6 if k < len(mutants) else 1.2,
            )

    fig = plt.figure(figsize=(15.09, 4.61))
    up.plot()
    plt.suptitle(title, fontweight="bold", fontsize=20)
    plt.tight_layout()
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savepath, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {savepath}")
    plt.close(fig)


def _plot_degree_distribution(labels: pd.DataFrame, title: str, savepath: Path,
                              dpi: int, include_zero: bool = False) -> pd.Series:
    """Bar plot of how many models satisfy exactly k experiments (k = 1..N).

    Collapses the full UpSet's 2^N intersections into N (or N+1) 'agreement'
    classes, coloured with the same AGREEMENT_COLORS palette used by the UpSet.
    """
    import matplotlib.pyplot as plt

    n = labels.shape[1]
    degree = labels.sum(axis=1).astype(int)
    total = len(degree)

    start = 0 if include_zero else 1
    classes = list(range(start, n + 1))
    counts = [int((degree == k).sum()) for k in classes]
    colors = [AGREEMENT_COLORS.get(k, "#888888") for k in classes]

    fig, ax = plt.subplots(figsize=(max(6.0, 1.1 * len(classes)), 5.0))
    bars = ax.bar([str(k) for k in classes], counts,
                  color=colors, edgecolor="black", linewidth=0.8)
    for b, c in zip(bars, counts):
        pct = (100.0 * c / total) if total else 0.0
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{c}\n{pct:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Number of experiments satisfied", fontweight="bold")
    ax.set_ylabel("Number of models", fontweight="bold")
    ax.set_title(title, fontweight="bold", fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if counts and max(counts) > 0:
        ax.set_ylim(0, max(counts) * 1.15)

    fig.tight_layout()
    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {savepath}")
    plt.close(fig)

    return pd.Series(counts, index=classes, name="n_models")


def run_upset_stage(
    args, model_name: str, output_dir: Path,
    per_mutant: Dict[str, dict],
):
    """Build the candidate and oracle UpSet plots from per-mutant KPI series."""
    images_dir = output_dir / "images"
    tables_dir = output_dir / "tables"

    mutants = [m for m, _ in args.perturbations]
    kpi_cand = {m: per_mutant[m]["kpi_cand"] for m in mutants}
    kpi_or = {m: per_mutant[m]["kpi_or"] for m in mutants}

    # Per-mutant good-fit thresholds. Precedence per mutant:
    #   1. --threshold-abs-per-mutant  (absolute KPI value / composite threshold)
    #   2. --threshold-pct-per-mutant  (oracle percentile)
    #   3. --threshold-pct             (scalar oracle percentile fallback)
    pct_overrides = dict(args.threshold_pct_per_mutant)
    abs_overrides = dict(getattr(args, "threshold_abs_per_mutant", []) or [])

    oracle_thresholds, source, pct_per_mutant = {}, {}, {}
    for m in mutants:
        if m in abs_overrides:
            oracle_thresholds[m] = float(abs_overrides[m])
            source[m] = f"abs {abs_overrides[m]:g}"
            pct_per_mutant[m] = float("nan")
        else:
            pct = float(pct_overrides.get(m, args.threshold_pct))
            pct_per_mutant[m] = pct
            oracle_thresholds[m] = float(np.percentile(kpi_or[m], pct))
            source[m] = f"p{pct:g}"

    print("[upset] per-mutant good-fit thresholds:")
    for m in mutants:
        print(f"        {m}: {source[m]} -> KPI <= {oracle_thresholds[m]:.4f}")

    save_table(
        pd.DataFrame({
            "source": pd.Series(source),
            "percentile": pd.Series(pct_per_mutant),
            "threshold": pd.Series(oracle_thresholds),
        }),
        tables_dir / f"upset_thresholds_{model_name}.csv",
        "upset_thresholds",
    )

    # Candidate + oracle membership labels.
    labels_cand = _labels_below_oracle_threshold(kpi_cand, oracle_thresholds)
    save_table(
        labels_cand.astype(int),
        tables_dir / f"upset_labels_{model_name}.csv",
        f"upset_labels_{model_name}",
    )
    labels_or = _labels_below_oracle_threshold(kpi_or, oracle_thresholds)
    thr_title = ", ".join(f"{m} {source[m]}" for m in mutants)

    # ---- rare-class metric: models in the JOINT class (all mutants satisfied) ----
    rare_cand = labels_cand.all(axis=1)
    rare_or = labels_or.all(axis=1)
    n_cand, n_or = len(labels_cand), len(labels_or)
    pct_c = 100.0 * float(rare_cand.mean()) if n_cand else 0.0
    pct_o = 100.0 * float(rare_or.mean()) if n_or else 0.0
    enrich = (pct_c / pct_o) if pct_o > 0 else float("nan")
    rare_tbl = pd.DataFrame(
        {"n_total": [n_cand, n_or],
         "n_rare_class": [int(rare_cand.sum()), int(rare_or.sum())],
         "pct_rare_class": [pct_c, pct_o],
         "enrichment_vs_oracle": [enrich, 1.0]},
        index=[model_name, "ORACLE"],
    )
    rare_tbl.index.name = "source"
    print(f"[rare-class] joint class (all {len(mutants)} mutants satisfied): "
          f"{model_name} {int(rare_cand.sum())}/{n_cand} ({pct_c:.3f}%) | "
          f"ORACLE {int(rare_or.sum())}/{n_or} ({pct_o:.3f}%) | "
          f"enrichment {enrich:.2f}x")
    save_table(rare_tbl, tables_dir / f"rare_class_metric_{model_name}.csv",
               "rare_class_metric")

    # Optional degree-distribution ("agreement") plot: number of models
    # satisfying exactly k experiments. One bar per class instead of the
    # full UpSet's 2^N intersections — the readable option for many mutants.
    if getattr(args, "upset_by_degree", False):
        inc0 = getattr(args, "upset_degree_include_zero", False)
        counts_cand = _plot_degree_distribution(
            labels_cand,
            title=f"{model_name} — models by number of experiments satisfied",
            savepath=images_dir / f"upset_degree_{model_name}.png",
            dpi=args.dpi, include_zero=inc0,
        )
        save_table(counts_cand,
                   tables_dir / f"upset_degree_counts_{model_name}.csv",
                   "n_models_by_degree")
        counts_or = _plot_degree_distribution(
            labels_or,
            title="ORACLE — models by number of experiments satisfied",
            savepath=images_dir / "upset_degree_oracle.png",
            dpi=args.dpi, include_zero=inc0,
        )
        save_table(counts_or,
                   tables_dir / "upset_degree_counts_oracle.csv",
                   "n_models_by_degree")

    # Full UpSet plots (unchanged). Kept non-fatal: with many conditions the
    # combinatorial width can exceed matplotlib's pixel limit, so a failure
    # here should not abort the run or discard the degree plots above.
    try:
        _plot_upset_one(
            labels_cand,
            title=f"{model_name} — models below threshold ({thr_title})",
            savepath=images_dir / f"upset_{model_name}.png",
            dpi=args.dpi,
        )
        _plot_upset_one(
            labels_or,
            title=f"ORACLE — models below threshold ({thr_title})",
            savepath=images_dir / "upset_oracle.png",
            dpi=args.dpi,
        )
    except Exception as exc:
        print(f"[warning] Full UpSet plot skipped ({type(exc).__name__}: {exc}).")
        print("          With many conditions the full UpSet can exceed the "
              "image-size limit; use --upset-by-degree for the agreement summary.")


# ---------------------------------------------------------------------------
# Stage 7: per-model R² KDE (one panel per perturbation)
# ---------------------------------------------------------------------------
def run_per_model_r2_stage(
    args, scoring_deps, model_name: str, output_dir: Path,
    enzyme_csvs: Dict[str, Path], oracle_files: Dict[str, str],
):
    """Generate the per-model R² KDE figure across all requested perturbations.

    The view is invariant to ensemble-mean cancellation and is the strongest
    visual response to a reviewer who reads the ensemble-mean scatter as
    'no improvement': it shows how concentrated the per-model fits are
    around the experimental truth, which the scatter cannot show.
    """
    plot_per_model_r2_kde = scoring_deps["plot_per_model_r2_kde"]
    images_dir = output_dir / "images"
    tables_dir = output_dir / "tables"
    images_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    perturbations_list = [
        (mutant, oracle_files[mutant], str(enzyme_csvs[mutant]))
        for mutant, _change in args.perturbations
    ]
    save_path = images_dir / f"per_model_r2_{model_name}.png"

    fig, summary = plot_per_model_r2_kde(
        perturbations=perturbations_list,
        exp_file=args.flux_exp_file,
        exclude_reactions=tuple(args.exclude_from_scatter),
        save_path=str(save_path),
        dpi=args.dpi,
        r2_clip=tuple(args.per_model_r2_clip),
        r2_threshold=args.per_model_r2_threshold,
        show=False,
    )
    print(f"[saved] {save_path}")

    if summary is not None and len(summary):
        summary_path = tables_dir / f"per_model_r2_summary_{model_name}.csv"
        summary.to_csv(summary_path)
        print(f"[saved] {summary_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    km_file = Path(args.km_file)
    if not km_file.exists():
        raise FileNotFoundError(f"--km-file not found: {km_file}")
    model_name = slugify(args.model_name or km_file.stem)

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.results_root) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] model_name={model_name}  output_dir={output_dir}")

    # Resolve per-mutant oracle enzyme CSVs eagerly so we fail fast on typos.
    mutants = [m for m, _ in args.perturbations]
    oracle_files = resolve_oracle_enzyme_files(args, mutants)
    print("[run] oracle enzyme files:")
    for m, p in oracle_files.items():
        print(f"      {m}: {p}")

    # Stage 1+2: perturbations
    enzyme_csvs = run_perturbations(args, model_name, output_dir)

    if args.skip_scoring:
        print("[done] --skip-scoring set; stopping after the perturbation stage.")
        return 0

    # Stage 3+4: scoring + per-mutant plots
    scoring_deps = import_scoring_dependencies()
    per_mutant = run_per_mutant_scoring_and_plots(
        args, scoring_deps, model_name, output_dir, enzyme_csvs, oracle_files,
    )

    # Stage 5: single Km-comparison plot
    run_km_comparison(args, scoring_deps, model_name, output_dir)

    # Stage 6: UpSet plots
    if not args.skip_upset:
        run_upset_stage(args, model_name, output_dir, per_mutant)

    # Stage 7: per-model R² KDE
    if not args.skip_per_model_r2:
        run_per_model_r2_stage(
            args, scoring_deps, model_name, output_dir, enzyme_csvs, oracle_files,
        )

    print("[done] Multi-perturbation pipeline completed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise

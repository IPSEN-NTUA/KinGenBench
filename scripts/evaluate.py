#!/usr/bin/env python3
"""
Evaluate an external generative model on the benchmark.

Input: a CSV of generated K_m parameter sets (rows = models, columns = the 259
K_m parameters of data/oracle/oracle_km.csv.gz; an index column is allowed) and
the scenario it was trained for. The script then runs exactly the pipeline used
for the publication:

  1. stability screening + pruning (SKiMpy, max eigenvalue < -10)
  2. in-silico enzyme perturbations of the scenario
  3. zeta-score KPI per perturbation, referenced to the ORACLE ensemble
  4. the paper metrics (same functions as paper/build_paper_outputs_*.py)

and writes one metrics row comparable to the published tables.

    python scripts/evaluate.py --km-file my_generator.csv --scenario all_mutants \
        --name my_generator --n-jobs 9

Outputs
    outputs/submissions/<name>_<tag>.csv          validated / re-ordered Km input
    outputs/runs/<name>_<tag>/                    simulation + scoring results
    outputs/evaluations/<name>_<tag>_metrics.csv  the metrics row
Use --skip-simulation to recompute metrics from an existing run folder.

Generate N = 2000 models (the benchmark default) so that pct_survived and the
class percentages are on the same footing as the published results.
Steps 1-3 need the simulation environment (Python 3.8 + SKiMpy, see README).
"""
from __future__ import annotations
import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from benchmark import paths
from benchmark.config import SCENARIOS, PERTURBATION_FACTORS, N_GENERATED

PAPER_SCRIPT = {"pgi": "build_paper_outputs_single.py",
                "pgi_pgl": "build_paper_outputs_dual.py",
                "all_mutants": "build_paper_outputs_capstone_stringent.py"}


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def load_and_validate(km_file: Path) -> pd.DataFrame:
    cols = list(pd.read_csv(paths.ORACLE_KM, index_col=0, nrows=1).columns)
    df = pd.read_csv(km_file)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[error] {len(missing)} K_m columns missing from {km_file}, "
                         f"e.g. {missing[:5]}. Column names must match "
                         f"{paths.ORACLE_KM.relative_to(ROOT)}.")
    extra = [c for c in df.columns if c not in cols]
    if extra:
        print(f"[input] ignoring {len(extra)} non-K_m column(s): {extra[:5]}")
    df = df[cols].astype(float)
    ok = np.isfinite(df.values).all(1) & (df.values > 0).all(1)
    if (~ok).any():
        print(f"[input] dropping {int((~ok).sum())} rows with non-finite or non-positive K_m")
    n_raw = len(df)
    df = df[ok].reset_index(drop=True)        # pipelines label models by row position
    print(f"[input] {len(df)} valid models (of {n_raw}) x {df.shape[1]} K_m")
    if n_raw != N_GENERATED:
        print(f"[note] the benchmark uses N = {N_GENERATED} generated models; "
              f"you supplied {n_raw}. Percentages remain comparable, counts do not.")
    return df, n_raw


# ---------------------------------------------------------------------------
# Simulation + scoring (calls the publication pipelines)
# ---------------------------------------------------------------------------
def run_pipeline(scenario: str, km_csv: Path, run_name: str, n_jobs: int, extra: list):
    sc = SCENARIOS[scenario]
    py = sys.executable
    if sc["pipeline"] == "single":
        m = sc["perts"][0]
        cmd = [py, str(ROOT / "scripts" / "run_pgi_pipeline.py"),
               "--km-file", str(km_csv), "--model-name", run_name,
               "--enzymes", m, "--changes", str(PERTURBATION_FACTORS[m]),
               "--cut-reaction", m, "--oracle-enzyme-file", str(paths.oracle_flux(m)),
               "--parallel-stability-workers", str(n_jobs),
               "--parallel-perturbation-workers", str(n_jobs)]
    else:
        pert_args = [f"{m}:{PERTURBATION_FACTORS[m]}" for m in sc["perts"]]
        thr_args = [f"{m}:{c:g}" for m, c in zip(sc["perts"], sc["cutoffs"])]
        cmd = [py, str(ROOT / "scripts" / "run_multi_perturbation_pipeline.py"),
               "--km-file", str(km_csv), "--model-name", run_name,
               "--perturbations", *pert_args,
               "--threshold-pct-per-mutant", *thr_args,
               "--n-jobs", str(n_jobs)]
        if len(sc["perts"]) > 4:
            cmd.append("--upset-by-degree")
    cmd += extra
    print("[run] " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


# ---------------------------------------------------------------------------
# Metrics (identical code path to the paper scripts)
# ---------------------------------------------------------------------------
def load_paper_module(scenario: str):
    f = ROOT / "paper" / PAPER_SCRIPT[scenario]
    spec = importlib.util.spec_from_file_location(f.stem, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_metrics(scenario: str, run_dir: Path, km_csv: Path) -> dict:
    pm = load_paper_module(scenario)
    oracle = pm.load_oracle(str(run_dir))          # ORACLE KPIs re-scored inside the run
    thr = pm.thresholds(oracle)
    ctx = pm.metric_context(oracle, thr)
    row = pm.run_metrics(run_dir, km_csv, thr, ctx)
    if row is None:
        raise SystemExit(f"[error] no scored KPI tables found in {run_dir / 'tables'}")
    return row


def published_table(scenario: str):
    f = paths.METRICS_DIR / SCENARIOS[scenario]["tag"] / "metrics_mean_std.csv"
    return pd.read_csv(f, index_col=0) if f.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--km-file", required=True, help="CSV of generated K_m parameter sets.")
    ap.add_argument("--scenario", required=True, choices=list(SCENARIOS))
    ap.add_argument("--name", default=None, help="Method name (default: CSV file stem).")
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="Parallel workers for the simulations.")
    ap.add_argument("--skip-simulation", action="store_true",
                    help="Reuse an existing run folder; only recompute the metrics.")
    ap.add_argument("pipeline_args", nargs=argparse.REMAINDER,
                    help="Extra arguments after '--' are passed to the simulation pipeline.")
    args = ap.parse_args()

    tag = SCENARIOS[args.scenario]["tag"]
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", args.name or Path(args.km_file).stem).strip("_")
    run_name = f"{name}_{tag}"          # same slug rule as the pipelines
    sub_dir = paths.OUTPUTS / "submissions"; sub_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = paths.OUTPUTS / "evaluations"; eval_dir.mkdir(parents=True, exist_ok=True)
    km_csv = sub_dir / f"{run_name}.csv"
    run_dir = paths.RUNS_DIR / run_name

    km, n_raw = load_and_validate(Path(args.km_file))
    km.to_csv(km_csv)

    if not args.skip_simulation:
        extra = [a for a in args.pipeline_args if a != "--"]
        run_pipeline(args.scenario, km_csv, run_name, args.n_jobs, extra)

    row = compute_metrics(args.scenario, run_dir, km_csv)
    if n_raw != len(km):   # rows dropped at validation count as not surviving
        row["pct_survived"] *= len(km) / n_raw
    res = pd.DataFrame([{"method": name, "scenario": args.scenario, **row}])
    out = eval_dir / f"{run_name}_metrics.csv"
    res.to_csv(out, index=False)
    pd.set_option("display.width", 240, "display.max_columns", 40)
    print("\n=== metrics ===\n" + res.round(4).to_string(index=False))
    print(f"[saved] {out}")

    pub = published_table(args.scenario)
    if pub is not None:
        means = pub[[c for c in pub.columns if c.endswith("__mean")]]
        means.columns = [c[:-6] for c in means.columns]
        cmp = pd.concat([means, res.set_index("method")[[c for c in means.columns
                                                          if c in res.columns]]])
        print("\n=== compared with the published results (mean over seeds) ===\n"
              + cmp.round(3).to_string())
    else:
        print(f"[note] published metric table not found under {paths.METRICS_DIR}")


if __name__ == "__main__":
    main()

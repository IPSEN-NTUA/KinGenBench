#!/usr/bin/env python3
"""
Build the training arrays of the benchmark scenarios from the shipped ORACLE
tables, so a new generative model can be trained without any other code.

Writes, per scenario, data/scenarios/<tag>.npz containing
    X            (n, 259)  log-transformed, standardized Km values (model input)
    y_binary     (n,)      joint high-performance label (discrete conditioning)
    y_multilabel (n, k)    per-perturbation labels
    Y_kpi        (n, k)    standardized log(zeta + 1e-8) (continuous conditioning)
    Y_raw        (n, k)    raw zeta KPIs
    x_mean, x_scale        log-Km standardization (to invert generated samples:
                           Km = exp(x * x_scale + x_mean) - 1e-8)
    kpi_mean, kpi_scale    log-KPI standardization
    columns, index         Km parameter names, ORACLE model ids
    perts, cutoffs, thresholds
plus data/scenarios/<tag>.json with the scenario summary.

Scenarios: pgi (I), pgi_pgl (II), all_mutants (III).

    python scripts/build_scenarios.py                # all three
    python scripts/build_scenarios.py --scenario all_mutants
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmark import paths
from benchmark.config import SCENARIOS
from benchmark.data import build_scenario


def save(s, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    xs, ks = s["x_scaler"], s["kpi_scaler"]
    np.savez_compressed(
        out_dir / f"{s['tag']}.npz",
        X=s["X"], y_binary=s["y_binary"], y_multilabel=s["y_multilabel"],
        Y_kpi=s["Y_kpi"], Y_raw=s["Y_raw"],
        x_mean=xs.mean_, x_scale=xs.scale_, kpi_mean=ks.mean_, kpi_scale=ks.scale_,
        columns=np.array(s["columns"]), index=np.array(s["index"]),
        perts=np.array(s["perts"]), cutoffs=np.array(s["cutoffs"], dtype=float),
        thresholds=np.array(s["thresholds"]),
    )
    summary = {"tag": s["tag"], "perturbations": s["perts"],
               "labelling_percentiles": s["cutoffs"],
               "labelling_thresholds_zeta": s["thresholds"],
               "n_models": int(len(s["y_binary"])),
               "n_positive": int(s["y_binary"].sum()),
               "n_km_parameters": int(s["X"].shape[1])}
    with open(out_dir / f"{s['tag']}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[saved] {out_dir / (s['tag'] + '.npz')}  "
          f"({summary['n_models']} models, {summary['n_positive']} positive)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=list(SCENARIOS), default=None,
                    help="Build one scenario (default: all).")
    ap.add_argument("--out-dir", default=str(paths.SCENARIO_DIR))
    args = ap.parse_args()
    names = [args.scenario] if args.scenario else list(SCENARIOS)
    for name in names:
        save(build_scenario(name), Path(args.out_dir))


if __name__ == "__main__":
    main()

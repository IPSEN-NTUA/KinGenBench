#!/usr/bin/env python3
"""
Recompute the ORACLE zeta-score KPI table (data/oracle/oracle_kpi.csv.gz)
from the shipped ORACLE perturbed fluxes and the experimental fluxes.

For each perturbation m and model j:
    sigma_r   = std over ORACLE models of |v_r^ORACLE - v_r^exp|
    zeta_r,j  = |v_r,j - v_r^exp| / sigma_r
    KPI_j     = sum_{r in key(m)} zeta_r,j + mean_{r not in key(m)} zeta_r,j
with key(m) = {m, biomass, GLCpts, SUCCt2_2, SUCCt3, CO2t} (G6PDH2r uses PGL).

The ORACLE ensemble is its own reference (self-referential sigma_r). Running
this script reproduces the shipped table to floating-point precision; use
--check to compare instead of overwriting.

    python scripts/build_oracle_kpi.py --check
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from benchmark import paths
from benchmark.config import ALL_MUTANTS
from benchmark.scoring.flux_utils import (
    _load_fluxes_exp, compute_zscore_vs_oracle, compute_KPI, get_key_reactions,
)


def oracle_kpi(mutant: str, fluxes_exp: pd.DataFrame) -> pd.Series:
    enz = pd.read_csv(paths.oracle_flux(mutant), index_col=0)
    enz = enz.drop(columns=[c for c in ["Steady"] if c in enz.columns])
    enz = enz[enz.index.isin(fluxes_exp.index)]
    exp_vec = fluxes_exp[mutant.lower()].reindex(enz.index)
    z = compute_zscore_vs_oracle(enz, exp_vec, enz)
    key = get_key_reactions("PGL" if mutant == "G6PDH2r" else mutant)
    return compute_KPI(z, key)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="Compare against the shipped table instead of writing it.")
    ap.add_argument("--out", default=str(paths.ORACLE_KPI))
    args = ap.parse_args()

    km_index = pd.read_csv(paths.ORACLE_KM, index_col=0, usecols=[0]).index
    fluxes_exp = _load_fluxes_exp(str(paths.EXP_FLUX))
    table = pd.DataFrame(index=km_index)
    for m in ALL_MUTANTS:
        table[m] = oracle_kpi(m, fluxes_exp).reindex(km_index)
        print(f"[{m}] {int(table[m].notna().sum())} models, "
              f"median KPI {table[m].median():.3f}")

    if args.check:
        ref = pd.read_csv(paths.ORACLE_KPI, index_col=0).reindex(km_index)[table.columns]
        diff = np.nanmax(np.abs(ref.values - table.values))
        same_nan = (ref.isna().values == table.isna().values).all()
        print(f"[check] max |diff| = {diff:.3e}, identical missing entries: {same_nan}")
        sys.exit(0 if (diff < 1e-9 and same_nan) else 1)
    table.to_csv(args.out, compression="gzip")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()

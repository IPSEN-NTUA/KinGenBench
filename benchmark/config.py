"""Benchmark definition: perturbations, enzyme-abundance factors and the
labelling thresholds of the three scenarios (single source of truth)."""

from __future__ import annotations
# Enzyme abundance factor applied to vmax in each in-silico perturbation.
PERTURBATION_FACTORS = {
    "PGI": 0.06, "PGL": 0.10, "RPI": 0.50, "TALA": 0.33, "PYK": 0.34,
    "G6PDH2r": 0.013, "PPS": 0.014, "TKT2": 0.23, "GND": 0.001,
}

ALL_MUTANTS = ["PGI", "PGL", "TALA", "G6PDH2r", "RPI", "PYK", "PPS", "TKT2", "GND"]

# Scenario -> ordered perturbations + labelling percentiles of the ORACLE KPI.
# A model is in the positive (high-performance) class when its KPI is at or
# below the given percentile for EVERY perturbation of the scenario.
SCENARIOS = {
    "pgi": {                                   # Scenario I
        "perts": ["PGI"], "cutoffs": [10],
        "tag": "pgi_thr10", "pipeline": "single",
    },
    "pgi_pgl": {                               # Scenario II
        "perts": ["PGI", "PGL"], "cutoffs": [10, 30],
        "tag": "pgi_pgl", "pipeline": "multi",
    },
    "all_mutants": {                           # Scenario III
        "perts": ["PGI", "PGL", "TALA", "G6PDH2r", "RPI", "PYK", "PPS", "TKT2", "GND"],
        "cutoffs": [25, 40, 60, 25, 60, 60, 40, 60, 40],
        "tag": "all_mutants", "pipeline": "multi",
    },
}

N_GENERATED = 2000            # models drawn from each generator before screening
SEEDS = [65, 66, 67]          # seeds used in the publication


def scenario_tag(name: str, pgi_threshold: float | None = None) -> str:
    if name == "pgi" and pgi_threshold is not None:
        return f"pgi_thr{int(pgi_threshold)}"
    return SCENARIOS[name]["tag"]


def cutoffs(name: str, pgi_threshold: float | None = None):
    if name == "pgi" and pgi_threshold is not None:
        return [float(pgi_threshold)]
    return list(SCENARIOS[name]["cutoffs"])

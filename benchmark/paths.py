"""Central file locations. Every script resolves paths through this module, so
the repository can be run from any working directory."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# ---- shipped with the repository -------------------------------------------
DATA = REPO / "data"
ORACLE_DIR = DATA / "oracle"
ORACLE_KM = ORACLE_DIR / "oracle_km.csv.gz"            # 12,524 ORACLE models x 259 Km
ORACLE_KPI = ORACLE_DIR / "oracle_kpi.csv.gz"          # same models x 9 zeta-score KPIs
ORACLE_FLUX_DIR = ORACLE_DIR / "perturbed_fluxes"      # <MUTANT>.csv.gz, reactions x models
EXP_FLUX = DATA / "experimental" / "fluxes_exp.csv"    # measured fluxes, one column per mutant
KM_MAPPING = DATA / "experimental" / "km_mapping.csv"  # Km parameter -> readable label
TFA_SAMPLE = DATA / "reference_state" / "tfa_sample1.csv"
SCENARIO_DIR = DATA / "scenarios"                      # written by scripts/build_scenarios.py

MODELS = REPO / "models"
KMODEL = MODELS / "kmodel3_last.yml"                   # SKiMpy kinetic model
TMODEL = MODELS / "tmodel_last.json"                   # pyTFA thermodynamic model

METRICS_DIR = REPO / "results" / "metrics"             # published metric tables
PAPER_FIG_DIR = REPO / "results" / "paper_figures"     # regenerated figures (gitignored)

# ---- produced locally or downloaded from Zenodo (gitignored) ---------------
OUTPUTS = REPO / "outputs"
SYNTH_DIR = OUTPUTS / "synthetic_parameters"           # generated Km CSVs
GEN_DIR = OUTPUTS / "generators"                       # trained-network checkpoints
RUNS_DIR = OUTPUTS / "runs"                            # simulation + scoring results per Km file


def oracle_flux(mutant: str) -> Path:
    """ORACLE perturbed-flux reference for one mutant (zeta-score scale)."""
    return ORACLE_FLUX_DIR / f"{mutant}.csv.gz"

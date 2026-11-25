# A Benchmark for Generative Modeling of Enzyme Kinetic Parameters

Code and data for:

> **Xenios, S., Doulaveris, A., Trokanas, N., Mexis, K., Kokossis, A. (2026).**
> *A Benchmark for Generative Modeling of High-Dimensional Enzyme Kinetic Parameters of Metabolic Networks.* DOI: TODO

The benchmark asks a conditional generative model to produce **new kinetic parameter
sets (259 Michaelis constants, K<sub>m</sub>) for the *E. coli* core kinetic model** that
reproduce measured fluxes under in-silico enzyme knockdowns better than the ORACLE
ensemble they were trained on. Every generated model is screened for dynamic stability
and simulated with SKiMpy, and its fit to experiment is scored with an ORACLE-referenced
ζ-score KPI.

Three scenarios form a difficulty ladder:

| Scenario | Perturbations | Labelling percentiles (ORACLE KPI) | Positive class |
|---|---|---|---|
| I `pgi` | PGI | 10 | 1,251 / 12,503 (10.0%) |
| II `pgi_pgl` | PGI, PGL | 10, 30 | 388 / 12,475 (3.1%) |
| III `all_mutants` | PGI, PGL, TALA, G6PDH2r, RPI, PYK, PPS, TKT2, GND | 25, 40, 60, 25, 60, 60, 40, 60, 40 | 16 / 12,429 (0.13%) |

Reference implementations of three architectures are included: a conditional β-VAE,
a conditional WGAN-GP, and a conditional RealNVP normalizing flow. Each is trained
with **discrete** conditioning (binary joint high-performance label) and with
**continuous** conditioning (standardized log-KPI vector), giving six variants.

---

## What is in this repository

Everything needed to **train a generator on the benchmark and evaluate it** is included.
No download is needed for that.

```
data/
├── oracle/
│   ├── oracle_km.csv.gz           # 12,524 ORACLE models x 259 Km        (training input)
│   ├── oracle_kpi.csv.gz          # same models x 9 zeta-score KPIs     (training labels)
│   └── perturbed_fluxes/<M>.csv.gz  # ORACLE fluxes under each perturbation (KPI reference)
├── experimental/
│   ├── fluxes_exp.csv             # measured fluxes, reference + 13 mutants
│   └── km_mapping.csv             # Km parameter -> reaction / metabolite labels
└── reference_state/tfa_sample1.csv  # thermodynamically feasible reference steady state
models/                            # SKiMpy kinetic model + pyTFA thermodynamic model
benchmark/                         # python package
├── config.py                      # scenarios, thresholds, perturbation factors
├── data.py                        # load the ORACLE tables, build scenario arrays
├── ml/                            # reference CVAE / WGAN-GP / RealNVP implementations
├── sim/                           # stability screening + perturbation simulations
├── scoring/                       # zeta-score KPI and per-run plots
└── metrics/                       # benchmark metrics (precision/recall, extrapolation, ...)
scripts/
├── build_scenarios.py             # scenario arrays -> data/scenarios/<tag>.npz
├── build_oracle_kpi.py            # recompute the ORACLE KPI table from the fluxes
├── generate_scenarios.py          # train the 6 reference variants and generate Km sets
├── run_pgi_pipeline.py            # simulate + score, Scenario I
├── run_multi_perturbation_pipeline.py  # simulate + score, Scenarios II and III
└── evaluate.py                    # one command to benchmark an external model
paper/                             # scripts that build the paper figures and tables
results/metrics/                   # published metric tables (per seed and mean +/- std)
docs/EXTERNAL_MODELS.md            # how to benchmark your own generator
```

**Hosted on Zenodo** (DOI: TODO): the generated K<sub>m</sub> sets of all reference runs in
the paper, the trained Scenario III networks, and the complete simulation and scoring outputs.
These are needed only to **regenerate the paper figures** or to compare model-level
distributions (KPI distributions, K<sub>m</sub> embeddings, per-model R²) with ours. Unpack the
three archives at the repository root so they create:

```
outputs/
├── synthetic_parameters/   benchmark_<variant>_<scenario>_seed<seed>.csv
├── generators/             <variant>_all_mutants_seed<seed>/  (trained networks, Scenario III)
└── runs/                   benchmark_<variant>_<scenario>_seed<seed>/  (simulation + scoring)
```

Continuous-variant file names carry the generation target: `*_continuous_*` = class threshold
(Scenarios I–II), `*_continuous_cls_*` = positive-class targets (Scenario III, main results),
`*_continuous_p5_*` = 5th-percentile target (Scenario III, targeted-generation supplement).

---

## Setup

Two Python environments are used: one for training, one for simulation. They are the same
as for [GENKI](https://github.com/stefanosxenios/GENKI).

**Training (`env_benchmark_ml`, Python 3.10).** No SKiMpy needed.

```bash
conda create -n env_benchmark_ml python=3.10
conda activate env_benchmark_ml
pip install -r requirements_ml.txt
pip install -e .
```

**Simulation, scoring and figures (`env_benchmark_sim`, Python 3.8).** This needs SKiMpy and pyTFA,
which must be installed first. The exact steps, including system libraries and solvers, are at the
top of [`requirements_sim.txt`](requirements_sim.txt).

```bash
conda create -n env_benchmark_sim python=3.8
conda activate env_benchmark_sim
conda install weilandtd::skimpy          # see requirements_sim.txt for alternatives
pip install -r requirements_sim.txt
pip install -e .
```

Run all commands from the repository root.

---

## Quick start: train your own generator

```python
from benchmark.data import build_scenario

s = build_scenario("all_mutants")     # "pgi", "pgi_pgl" or "all_mutants"
X        = s["X"]          # (12429, 259) log-standardized Km  -> model input
y        = s["y_binary"]   # (12429,)     1 = good for all 9 perturbations
Y_kpi    = s["Y_kpi"]      # (12429, 9)   standardized log-KPI -> continuous condition
x_scaler = s["x_scaler"]   # Km = exp(x_scaler.inverse_transform(x)) - 1e-8
```

Or write framework-independent `.npz` files with `python scripts/build_scenarios.py`.

Generate **2,000** K<sub>m</sub> sets from the high-performance condition and save them as a CSV
with the 259 K<sub>m</sub> columns. Then evaluate (simulation environment):

```bash
python scripts/evaluate.py \
    --km-file my_generator_all_mutants.csv \
    --scenario all_mutants \
    --name my_generator \
    --n-jobs 9
```

This runs stability screening, the perturbation simulations, ζ-scoring and the paper metrics. It
prints your metrics next to the published ones. See [docs/EXTERNAL_MODELS.md](docs/EXTERNAL_MODELS.md)
for the rules that keep a comparison fair.

---

## Reproducing the paper

### 1. Train the reference generators (`env_benchmark_ml`)

One run trains all six variants for one scenario and seed, saves the networks to
`outputs/generators/`, and writes the generated K<sub>m</sub> sets to
`outputs/synthetic_parameters/`. Seeds used in the paper:

| Scenario | Seeds | Continuous target (`--kpi-target-mode`) |
|---|---|---|
| I `pgi` | 45, 46, 66, 67 (cGAN-cont: 65, 66, 67) | `threshold` |
| II `pgi_pgl` | 45, 66, 67 (cGAN-cont: 65, 66, 67) | `threshold` |
| III `all_mutants` | 65, 66, 67 | `class` (main), `percentile` 5% (supplement; cVAE-cont / NF-cont seeds 45, 66, 67) |

Every target lies inside the discrete positive class, so the discrete and continuous variants
aim at the same performance class: `threshold` sets each KPI to its labelling threshold, and
`class` conditions each sample on the KPI vector of a randomly drawn positive-class training
model (files suffixed `_cls`).

```bash
# Scenario I (one command per seed)
python scripts/generate_scenarios.py --scenario pgi --pgi-threshold 10 --seed 45 --kpi-target-mode threshold

# Scenario II
python scripts/generate_scenarios.py --scenario pgi_pgl --seed 45 --kpi-target-mode threshold

# Scenario III (main)
python scripts/generate_scenarios.py --scenario all_mutants --seed 65 --kpi-target-mode class
```

The targeted-generation comparison (5th percentile of every KPI) reuses the trained networks
(files suffixed `_p5`):

```bash
python scripts/generate_scenarios.py \
    --scenario all_mutants \
    --seed 65 \
    --from-checkpoint \
    --kpi-target-mode percentile \
    --kpi-target-pct 5 \
    --variants cvae_continuous,gan_continuous,nf_continuous
```

### 2. Simulate and score (`env_benchmark_sim`)

Results go to `outputs/runs/<km-file-stem>/`. Per generated file:

```bash
# Scenario I
python scripts/run_pgi_pipeline.py \
    --km-file outputs/synthetic_parameters/benchmark_cvae_pgi_thr10_seed65.csv \
    --parallel-stability-workers 4 \
    --parallel-perturbation-workers 4

# Scenario II
python scripts/run_multi_perturbation_pipeline.py \
    --km-file outputs/synthetic_parameters/benchmark_cvae_pgi_pgl_seed65.csv \
    --perturbations PGI:0.06 PGL:0.10 \
    --threshold-pct-per-mutant PGI:10 PGL:30 \
    --n-jobs 2

# Scenario III
python scripts/run_multi_perturbation_pipeline.py \
    --km-file outputs/synthetic_parameters/benchmark_cvae_all_mutants_seed65.csv \
    --perturbations PGI:0.06 PGL:0.10 RPI:0.50 TALA:0.33 PYK:0.34 G6PDH2r:0.013 PPS:0.014 TKT2:0.23 GND:0.001 \
    --threshold-pct-per-mutant PGI:25 PGL:40 RPI:60 TALA:60 PYK:60 G6PDH2r:25 PPS:40 TKT2:60 GND:40 \
    --upset-by-degree \
    --n-jobs 9
```

### 3. Figures and metric tables

Steps 1 and 2 can be skipped by downloading `outputs/` from Zenodo. Then:

```bash
python paper/build_paper_outputs_single.py              # Scenario I   -> results/paper_figures/pgi/
python paper/build_paper_outputs_dual.py                # Scenario II  -> results/paper_figures/pgi_pgl/
python paper/build_paper_outputs_capstone_stringent.py  # Scenario III -> results/paper_figures/capstone_stringent/
python paper/build_paper_outputs_capstone_stringent.py --continuous-target p5   # -> capstone_stringent_p5/
```

Each script writes `metrics_per_seed.csv`, `metrics_mean_std.csv` and
`metrics_mean_std_pretty.csv` (Scenario III also `metrics_full.csv`), identical to the published
tables in `results/metrics/`, together with the figures.

---

## Benchmark metrics

All metrics are computed on the models that pass stability screening. Thresholds come from the
ORACLE KPI distribution of the scenario.

| Metric | Meaning |
|---|---|
| `pct_survived` | % of the 2,000 generated models that are stable and simulate successfully |
| `joint_pct` / `joint_pct_all9` | % of models in the positive class (KPI ≤ threshold for every perturbation) |
| `enrichment_vs_oracle` | joint % divided by the ORACLE base rate |
| `pct_ge7` (III) | % of models satisfying at least 7 of the 9 perturbations |
| `n_new_best_pert` (III) | perturbations (0–9) for which at least one model beats the best ORACLE model |
| `mean_KPI`, `pct_below_p*`, `pct_top1`, `pct_outperform_best` | per-perturbation KPI statistics |
| `precision`, `recall` | k-NN (k = 5) precision / recall against the ORACLE high-performance reference set, in the PCA-50 embedding of log-standardized K<sub>m</sub> |
| `extrap_vs_top`, `extrap_vs_full` | mean k-NN distance to the high-performance reference / to the full ORACLE ensemble, normalized by the within-reference distance (ρ > 1: generation beyond the training support) |

The high-performance reference set is: Scenario I, ORACLE models below the 10th KPI percentile;
Scenario II, ORACLE models in the joint positive class; Scenario III, ORACLE models whose mean
KPI over the 9 perturbations is below its 10th percentile. (The joint class has only 16 models,
too few for a k-NN reference.)

---

## Runtime

Training takes minutes per variant (GPU supported for all three frameworks). Simulation time
grows with the number of surviving models and perturbations. Scenario III (9 perturbations) is
the most expensive and benefits from one worker per perturbation (`--n-jobs 9`).

## License and citation

Code: MIT ([LICENSE](LICENSE)). Please cite the paper ([CITATION.cff](CITATION.cff)). The ORACLE
ensemble and the kinetic model follow [GENKI](https://github.com/stefanosxenios/GENKI).

## Contact

Stefanos Xenios, National Technical University of Athens

# Benchmarking your own generative model

This guide is for groups who want to evaluate a new generator (any framework) on the
benchmark and compare it with the published results. Only the data in this repository
is needed. The Zenodo archive is needed only to compare model-level distributions with
the reference generators.

## 1. Get the training data

```python
from benchmark.data import build_scenario
s = build_scenario("all_mutants")      # or "pgi", "pgi_pgl"
```

| key | shape | content |
|---|---|---|
| `X` | (n, 259) | log(K<sub>m</sub> + 1e-8), standardized per parameter (training statistics) |
| `y_binary` | (n,) | 1 if the model is in the positive class for **all** perturbations of the scenario |
| `y_multilabel` | (n, k) | the per-perturbation labels |
| `Y_kpi` | (n, k) | standardized log(ζ + 1e-8): the continuous conditioning signal |
| `Y_raw` | (n, k) | raw ζ-score KPIs (lower = better fit to experiment) |
| `x_scaler`, `kpi_scaler` | | fitted scalers. To recover K<sub>m</sub>: `np.exp(x_scaler.inverse_transform(x)) - 1e-8` |
| `columns`, `index` | | K<sub>m</sub> names (column order) and ORACLE model ids |

Without our package, `python scripts/build_scenarios.py` writes the same arrays to
`data/scenarios/<tag>.npz`, together with the scaler means and scales, for use in any framework.

The raw tables are `data/oracle/oracle_km.csv.gz` and `data/oracle/oracle_kpi.csv.gz`
(missing KPI = that perturbation's simulation failed for the model). You may use them directly,
e.g. to design your own preprocessing.

## 2. Rules for a comparable result

1. **Train only on the ORACLE training set of the scenario.** Do not use the experimental
   fluxes or the perturbation simulations as a training signal beyond the provided KPIs.
2. **Generate N = 2,000 models** per run, at the high-performance condition:
   - for a label-conditioned model: the positive label;
   - for a KPI-conditioned model: the **class-matched** target, i.e. draw each sample's
     condition from the KPI vectors of the positive-class training models
     (`Y_kpi[y_binary == 1]`). This is the main comparison. Lower targets (e.g. the 5th
     percentile) are a separate "targeted generation" comparison and should be reported as such.
3. **Use the model's default sampling** (e.g. temperature 1). Do not filter generated samples
   with the simulator. Screening is part of the evaluation.
4. **Report three independent training seeds** (mean ± std), as in the paper.
5. Keep the parameter budget in the range of the reference models (2.2–2.8 × 10<sup>5</sup>
   trainable parameters), or report it alongside the results.

## 3. Save the generated models

A CSV with one row per generated model and the 259 K<sub>m</sub> columns of
`data/oracle/oracle_km.csv.gz` (the column order does not matter; an index column is allowed).
Values must be in raw K<sub>m</sub> units: apply the inverse transform. Rows with non-finite or
non-positive values are dropped and count as non-surviving.

## 4. Evaluate

In the simulation environment (Python 3.8 + SKiMpy, see the README):

```bash
python scripts/evaluate.py \
    --km-file my_generator_seed1.csv \
    --scenario all_mutants \
    --name my_generator_seed1 \
    --n-jobs 9
```

| output | content |
|---|---|
| `outputs/runs/<name>_<scenario>/` | surviving parameter sets, simulated fluxes, KPI tables, diagnostic plots |
| `outputs/evaluations/<name>_<scenario>_metrics.csv` | the metric row |

The metric row is computed by the same functions as the published tables
(`paper/build_paper_outputs_*.py`) and printed next to `results/metrics/`.
`--skip-simulation` recomputes the metrics of an existing run. Any arguments after `--`
are passed on to the simulation pipeline.

## 5. Share your result

Open a pull request that adds your mean ± std row to
`results/metrics/<scenario>/external_results.csv`, with the per-seed metric files and a link to
your code. We will add it to the comparison table.

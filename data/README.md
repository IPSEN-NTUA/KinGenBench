# Data

All files are derived from the ORACLE ensemble of the *E. coli* core kinetic model used in
[GENKI](https://github.com/stefanosxenios/GENKI). Model ids (`"i,k"`) are shared by every table.

| file | rows × cols | content |
|---|---|---|
| `oracle/oracle_km.csv.gz` | 12,524 × 259 | Michaelis constants of every ORACLE model (training input) |
| `oracle/oracle_kpi.csv.gz` | 12,524 × 9 | ζ-score KPI of every model under each perturbation (PGI, PGL, TALA, G6PDH2r, RPI, PYK, PPS, TKT2, GND); empty = simulation failed |
| `oracle/perturbed_fluxes/<M>.csv.gz` | 31 × (1 + models) | simulated fluxes of the ORACLE models under perturbation `<M>`, restricted to the 31 measured reactions; first column `Steady` = reference steady state. Defines the ζ-score scale σ<sub>r</sub> and is the ORACLE baseline in every evaluation |
| `experimental/fluxes_exp.csv` | 31 × 14 | measured fluxes for the reference strain and 13 mutants (columns = lower-case mutant names) |
| `experimental/km_mapping.csv` | | K<sub>m</sub> parameter → reaction / metabolite, used for plot labels |
| `reference_state/tfa_sample1.csv` | | thermodynamically feasible steady state used for stability screening and simulation |

## The ζ-score KPI

For perturbation *m* and model *j*, with measured fluxes *v*<sup>exp</sup>:

- σ<sub>r</sub> = std over ORACLE models of |*v*<sub>r</sub><sup>ORACLE</sup> − *v*<sub>r</sub><sup>exp</sup>|
- ζ<sub>r,j</sub> = |*v*<sub>r,j</sub> − *v*<sub>r</sub><sup>exp</sup>| / σ<sub>r</sub>
- KPI<sub>j</sub> = Σ<sub>r ∈ key(m)</sub> ζ<sub>r,j</sub> + mean<sub>r ∉ key(m)</sub> ζ<sub>r,j</sub>

The key reactions are key(m) = {m, biomass, GLCpts, SUCCt2_2, SUCCt3, CO2t} (PGL for G6PDH2r).
Lower is better. Biomass and succinate transport measurements receive fixed corrections
(`benchmark/scoring/flux_utils.py::_load_fluxes_exp`).

`python scripts/build_oracle_kpi.py --check` recomputes `oracle_kpi.csv.gz` from the
perturbed fluxes and confirms that it is reproduced.

## Scenarios

`python scripts/build_scenarios.py` writes `scenarios/<tag>.npz` + `.json` (not tracked by git).
The scenario definitions are in `benchmark/config.py`.

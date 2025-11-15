# Bayesian Regression DSC Framework

This repository collects Python and R components for exploring a family of
Bayesian linear regression models under the [Dynamic Statistical Comparisons
(DSC)](https://stephenslab.github.io/dsc-wiki/) workflow.  A DSC specification
(`main.dsc`) orchestrates synthetic data generation, fits multiple analysis
modules, and scores their predictions so you can compare modelling
assumptions across controlled noise and sparsity sweeps.【F:DSC_framework/main.dsc†L1-L94】

## Repository layout

- `DSC_framework/main.dsc` – DSC pipeline definition that wires the simulation,
  analysis, and scoring stages together while enumerating the noise and
  sparsity grids to explore.【F:DSC_framework/main.dsc†L21-L94】
- `DSC_framework/PY/model3_simulate.py` – simulation module that synthesises
  design matrices and responses for the learned group sparsity generative
  model, supporting configurable noise levels and sparsity probabilities at run
  time.【F:DSC_framework/PY/model3_simulate.py†L1-L100】
- `DSC_framework/models/` – collection of Python analysis modules implementing
  different Bayesian linear regression flavours (ridge, ARD, group switches,
  reparameterised priors, and spike-and-slab variants).【F:DSC_framework/models/model_1a_bayesian_regression_shared_prior.py†L1-L14】【F:DSC_framework/models/model_1b_bayesian_regression_with_ard.py†L1-L14】【F:DSC_framework/models/model_2_bayesian_regression_group_switch.py†L1-L104】【F:DSC_framework/models/model_3_bayesian_regression_learned_group_sparsity.py†L1-L132】【F:DSC_framework/models/model_4_reparameterized_regression.py†L1-L89】【F:DSC_framework/models/model_6_spike_and_slab_shared_precision.py†L1-L181】【F:DSC_framework/models/model_7_spike_and_slab_ard_precision.py†L1-L162】
- `DSC_framework/PY/{rmse.py, mae.py}` – scoring modules that compute percentage
  RMSE and MAE from the simulated truth and fitted predictions for each
  replicate.【F:DSC_framework/PY/rmse.py†L1-L13】【F:DSC_framework/PY/mae.py†L1-L14】
- `DSC_framework/R/noise_summary.R` – tidyverse helpers that query DSC outputs,
  summarise noise/sparsity sweeps, and render comparative plots across
  models.【F:DSC_framework/R/noise_summary.R†L1-L200】【F:DSC_framework/R/noise_summary.R†L320-L624】【F:DSC_framework/R/noise_summary.R†L640-L745】
- `DSC_framework/R/prediction_curves.R` – visualisation utilities that read the
  pickle artefacts emitted by DSC and overlay fitted curves with the simulated
  observations for each replicate using the sinusoidal design matrix and weight
  vectors saved by the pipeline.【F:DSC_framework/R/prediction_curves.R†L1-L432】
- `SyntheticData.py` – stand-alone class for constructing oscillatory synthetic
  datasets with optional spline waveforms, additive noise, and plotting
  helpers used by the exploratory notebooks/scripts outside the DSC
  workflow.【F:SyntheticData.py†L6-L224】
- `Model 1a*.py`, `Model 1b*.py`, `model 1b random.py` – exploratory notebooks
  (in script form) demonstrating variational Bayes derivations for specific
  model variants; they are independent of DSC but illustrate the underlying
  maths and visual diagnostics.【F:Model 1a.py†L1-L120】【F:Model 1b.py†L1-L120】

## Prerequisites

### Python environment

Create a Python environment (3.9+ recommended) and install the numerical stack
required by the simulation, modelling, and scoring modules:

```bash
pip install numpy scipy scikit-learn matplotlib
```

`model3_simulate.py` and the variational inference models rely on NumPy, SciPy,
and scikit-learn APIs, while the stand-alone demos also use Matplotlib for
plots.【F:DSC_framework/PY/model3_simulate.py†L33-L98】【F:DSC_framework/models/model_1a_bayesian_regression_shared_prior.py†L1-L14】【F:Model 1a.py†L1-L120】【F:SyntheticData.py†L1-L224】

You will also need the DSC command-line tool (available via `pip install dsc`
or the `dscrutils` R package) to execute the workflow described below.

### R environment

The R visualisation helpers depend on tidyverse packages (`dplyr`, `tidyr`,
`ggplot2`), `readr`, and the DSC utilities (`dscrutils`).  Install them via:

```r
install.packages(c("dplyr", "tidyr", "ggplot2", "readr"))
# dscrutils lives on GitHub:
# remotes::install_github("stephenslab/dsc")
```

`prediction_curves.R` additionally requires `reticulate` to load the pickle
files written by Python modules.【F:DSC_framework/R/prediction_curves.R†L36-L141】

## Running the DSC pipeline

1. Change into `DSC_framework/`.
2. Execute the pipeline with:
   ```bash
   dsc main.dsc
   ```

The DSC file enumerates a grid of noise standard deviations and sparsity
probabilities, fans out analysis modules, and evaluates each combination with
both RMSE and MAE scorers across five replicates.【F:DSC_framework/main.dsc†L21-L94】

Successful runs create a `dsc_result/` directory containing per-module pickle
artefacts, summary tables, and HTML reports (if `dsc` is configured to emit
them).  The folder structure mirrors the module names defined in `main.dsc`.

## Inspecting results

- Use the R helpers to summarise performance across sweeps:
  ```r
  source("R/noise_summary.R")
  results <- run_noise_sparsity_analysis(dsc_path = "dsc_result")
  ```
  The functions compute grouped statistics, reshape tidy summaries, and create
  standardised plots for noise-only, sparsity-only, and joint sweeps, writing
  PNGs into `plot_outputs/` by default.【F:DSC_framework/R/noise_summary.R†L1-L624】【F:DSC_framework/R/noise_summary.R†L640-L745】
- To visualise fitted curves for individual replicates, run:
  ```r
  source("R/prediction_curves.R")
  plots <- plot_model3_signal_predictions(dsc_path = "dsc_result")
  ```
  Each plot overlays the simulated observations with model-specific prediction
  lines, with filenames encoding the noise and sparsity settings for easy
  navigation.【F:DSC_framework/R/prediction_curves.R†L242-L431】 The helper
  reconstructs sampling times directly from the sinusoidal design matrix
  (`simulate.x`) and uses the simulated `w` coefficients (sin, cos, intercept)
  to rebuild the true signal for reference.【F:DSC_framework/R/prediction_curves.R†L142-L241】

## Extending the workflow

- Modify `main.dsc` to adjust the simulation grid, add/remove analysis modules,
  or introduce new scoring functions.  DSC will automatically expand the cartesian
  product of stages you define.【F:DSC_framework/main.dsc†L21-L94】
- Create new analysis modules by adding Python scripts under
  `DSC_framework/models/` that accept `x` and `y`, populate `fit`, and emit
  predictions `y_hat`, mirroring the existing implementations.【F:DSC_framework/models/model_4_reparameterized_regression.py†L11-L89】【F:DSC_framework/models/model_6_spike_and_slab_shared_precision.py†L23-L181】
- Update or add scoring modules under `DSC_framework/PY/` to capture alternative
  metrics (e.g., calibration or coverage) using the same signature as the
  percentage RMSE/MAE helpers.【F:DSC_framework/PY/rmse.py†L1-L13】【F:DSC_framework/PY/mae.py†L1-L14】

## Stand-alone experimentation

Outside the DSC workflow, the scripts `Model 1a.py`, `Model 1b.py`, and related
variants provide detailed variational Bayes derivations, ELBO tracking, and
plotting for sinusoidal toy problems.  They serve as references when extending
or debugging the production modules.【F:Model 1a.py†L1-L120】【F:Model 1b.py†L1-L120】

The `SyntheticData` class offers additional flexibility for constructing custom
oscillatory datasets with harmonic groupings, optional spline waveforms, and
noise corruption, which can be useful for bespoke experiments before encoding
them in DSC.【F:SyntheticData.py†L6-L224】


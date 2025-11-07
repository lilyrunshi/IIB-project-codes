#!/usr/bin/env dsc

# SUMMARY
# =======
#
# PIPELINE VARIABLES
# $X        simulated features (matrix)
# $y        simulated responses (vector)
# $fit      fitted model object
# $y_hat    model predictions (vector)
# $error    error score (scalar)
#
# MODULE TYPES
# name     inputs                outputs
# ----     ------                -------
# simulate none                  $x, $y
# analyze  $x, $y                $fit, $y_hat
# score    $y, $y_hat            $error
#

# --- simulate ---
# Simulate data from Model 3 (learned group sparsity)
model3_simulate: model3_simulate.py
  n: 200         # samples
  d: 20          # features
  seed: 8675309  # reproducibility for the latent draws
  $x: x
  $y: y

# --- analyze (complex: separate Python files) ---
model1a: model_1a_bayesian_regression_shared_prior.py
  x: $x
  y: $y
  $fit: fit
  $y_hat: y_hat

model1b: model_1b_bayesian_regression_with_ard.py
  x: $x
  y: $y
  $fit: fit
  $y_hat: y_hat

model2: model_2_bayesian_regression_group_switch.py
  x: $x
  y: $y
  $fit: fit
  $y_hat: y_hat

model3: model_3_bayesian_regression_learned_group_sparsity.py
  x: $x
  y: $y
  $fit: fit
  $y_hat: y_hat

model4: model_4_reparameterized_regression.py
  x: $x
  y: $y
  $fit: fit
  $y_hat: y_hat

model6: model_6_spike_and_slab_shared_precision.py
  x: $x
  y: $y
  $fit: fit
  $y_hat: y_hat

model7: model_7_spike_and_slab_ard_precision.py
  x: $x
  y: $y
  $fit: fit
  $y_hat: y_hat

# --- score (no access to true underlying coefs; fit quality only) ---
rmse: rmse.py
  y_true: $y
  y_pred: $y_hat
  $error: e

mae: mae.py
  y_true: $y
  y_pred: $y_hat
  $error: e

DSC:
  define:
    simulate: model3_simulate
    analyze: model1a, model1b, model2, model3, model4, model6, model7
    score: rmse, mae
  run: simulate * analyze * score
  output: dsc_result
  exec_path: PY, models
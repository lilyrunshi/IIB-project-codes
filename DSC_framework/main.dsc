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
# Simulate data with Additive White Gaussian Noise
awgn: awgn.py
  n: 200         # samples
  d: 20          # features
  snr_db: 10     # signal-to-noise ratio (in dB)
  $x: x
  $y: y

# --- analyze (complex: separate Python files) ---
model_1a: model_1a.py
  x: $x
  y: $y
  $fit: fit
  $y_hat: y_hat

model_1b: model_1b.py
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
    simulate: awgn
    analyze: model_1a, model_1b
    score: rmse, mae
  run: simulate * analyze * score
  output: dsc_result
  exec_path: PY, models
# --- simulate ---
simulate_awgn: awgn.py
  n: 200          # samples
  d: 20           # features
  snr_db: 10      # signal-to-noise ratio (in dB)
  $X: X
  $y: y

# --- analyze (complex: separate Python files) ---
model_1a, model_1b: (model_1a.py, model_1b.py)
  X: $X
  y: $y
  $fit: fit
  $y_hat: y_hat

# --- score (no access to true underlying coefs; fit quality only) ---
rmse, mae: (rmse.py, mae.py)
  y_true: $y
  y_pred: $y_hat
  $error: score

DSC:
  define:
    simulate: simulate_awgn
    analyze: model_1a, model_1b
    score: rmse, mae
  run: simulate * analyze * score
  replicate: 10

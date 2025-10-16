import numpy as np
from scipy.special import digamma 
from scipy.special import gammaln
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# --- VBLR CORE FUNCTIONS (Model 1b ARD) ---

def calculate_elbo_model1b(N, M, X, y, mN, SN, aN, bN, cN, dN, a0, b0, c0, d0):
    """Calculates the ELBO for Model 1b (ARD)."""
    
    E_log_alpha = digamma(aN) - np.log(bN)
    E_log_beta = digamma(cN) - np.log(dN)
    E_alpha_vector = aN / bN
    E_beta = cN / dN
    E_omega_sq_vector = mN.flatten()**2 + np.diag(SN)
    y_residual = y - X @ mN
    E_residual_sq = y_residual.T @ y_residual + np.trace(X @ SN @ X.T)
    (sign, logdet_SN) = np.linalg.slogdet(SN)
    
    # E[log p] terms (Implementation truncated for brevity)
    E_log_lik = N/2 * E_log_beta - N/2 * np.log(2 * np.pi) - E_beta/2 * E_residual_sq.item()
    E_log_prior_omega = -M/2 * np.log(2 * np.pi) + 0.5 * np.sum(E_log_alpha) - 0.5 * np.sum(E_alpha_vector * E_omega_sq_vector)
    E_log_prior_alpha = M * a0 * np.log(b0) - M * gammaln(a0) + (a0 - 1) * np.sum(E_log_alpha) - b0 * np.sum(E_alpha_vector)
    E_log_prior_beta = c0 * np.log(d0) + (c0 - 1) * E_log_beta - d0 * E_beta - gammaln(c0)
    E_log_joint = E_log_lik + E_log_prior_omega + E_log_prior_alpha + E_log_prior_beta
    
    # -E[log q] terms (Implementation truncated for brevity)
    neg_E_log_q_omega = M/2 * (1 + np.log(2 * np.pi)) + 0.5 * logdet_SN
    neg_E_log_q_alpha = np.sum(gammaln(aN) - (aN - 1) * E_log_alpha + aN - aN * np.log(bN))
    neg_E_log_q_beta = gammaln(cN) - (cN - 1) * E_log_beta + cN - cN * np.log(dN)
    neg_E_log_q = neg_E_log_q_omega + neg_E_log_q_alpha + neg_E_log_q_beta

    ELBO = E_log_joint + neg_E_log_q
    return ELBO.item()

def variational_bayes_linear_regression_ard(X, y, a0, b0, c0, d0, max_iter=200, tol=1e-6):
    """Implements Variational Bayes Linear Regression with ARD (Model 1b)."""
    N, M = X.shape 

    E_alpha_vector = np.ones(M) * (a0 / b0)
    E_beta = c0 / d0

    aN = a0 + 0.5
    cN = c0 + N / 2
    bN = np.ones(M) * b0
    dN = d0

    E_A = np.diag(E_alpha_vector) 
    SN_inv = E_beta * X.T @ X + E_A
    SN = np.linalg.inv(SN_inv)
    mN = SN @ (E_beta * X.T @ y)

    elbo_history = []
    i = 0 
    for i in range(max_iter):
        mN_old = mN.copy()

        mN_sq_vector = mN.flatten()**2
        SN_diag_vector = np.diag(SN)
        bN = b0 + 0.5 * (mN_sq_vector + SN_diag_vector)
        
        E_alpha_vector = aN / bN
        E_A = np.diag(E_alpha_vector)

        y_residual = y - X @ mN
        E_residual_sq = y_residual.T @ y_residual + np.trace(X @ SN @ X.T)
        dN = d0 + 0.5 * E_residual_sq.item()
        E_beta = cN / dN

        SN_inv = E_beta * X.T @ X + E_A
        SN = np.linalg.inv(SN_inv)
        mN = SN @ (E_beta * X.T @ y)

        ELBO = calculate_elbo_model1b(N, M, X, y, mN, SN, aN, bN, cN, dN, a0, b0, c0, d0)
        elbo_history.append(ELBO)
        
        if i > 0 and np.linalg.norm(mN - mN_old) / np.linalg.norm(mN_old) < tol:
            break

    E_alpha_mean = np.mean(E_alpha_vector)
    
    return X @ mN, elbo_history, i + 1, mN, E_alpha_mean, E_beta, E_alpha_vector


# --- ARD MODEL FORMULA GENERATOR ---

def get_inferred_model_formula(mN, basis_names, thresh=0.05):
    """
    Constructs the inferred model formula by filtering out components
    whose estimated coefficients E[omega] are below the threshold.
    """
    formula_parts = []
    mN_flat = mN.flatten()
    
    # Check DC offset first (Feature 0)
    if np.abs(mN_flat[0]) >= thresh:
        formula_parts.append(f"{mN_flat[0]:.3f}")
    else:
        # Include DC offset even if small, as it's typically required
        formula_parts.append(f"{mN_flat[0]:.3f}") 

    # Check all other features (Sin/Cos components)
    for i in range(1, len(mN_flat)):
        coeff = mN_flat[i]
        name = basis_names[i]
        
        if np.abs(coeff) >= thresh:
            sign = " + " if coeff > 0 else " - "
            term = f"{np.abs(coeff):.3f} * {name}"
            formula_parts.append(sign.strip() + term)

    if not formula_parts:
        return "y ≈ 0 (All coefficients shrunk to zero)"

    return "y ≈ " + "".join(formula_parts)


# --- EXPERIMENT SETUP ---
N_SAMPLES = 400
T = np.linspace(0, 5, N_SAMPLES)
np.random.seed(42) # For reproducibility of the 'random' signal

# --- 1. RANDOM RHYTHMIC SIGNAL GENERATION ---
N_COMPONENTS = 3 # Number of random sin/cos pairs
TRUE_FREQS = np.sort(np.round(np.random.uniform(1.0, 8.0, N_COMPONENTS), 1))
# Random coefficients between -2.0 and 2.0
TRUE_COEFFS = np.random.uniform(-2.0, 2.0, N_COMPONENTS * 2) 
DC_OFFSET = np.random.uniform(-1.0, 1.0) # Random DC offset

True_Signal = np.full(N_SAMPLES, DC_OFFSET)
for i in range(N_COMPONENTS):
    f = TRUE_FREQS[i]
    c_sin = TRUE_COEFFS[2*i]
    c_cos = TRUE_COEFFS[2*i + 1]
    True_Signal += c_sin * np.sin(2 * np.pi * f * T) + c_cos * np.cos(2 * np.pi * f * T)

# 2. Construct the Overcomplete Dictionary X_dict (Shared)
CANDIDATE_FREQS = np.linspace(1.0, 10.0, 20)
# Add the true random frequencies to ensure they are in the dictionary
CANDIDATE_FREQS = np.unique(np.sort(np.append(CANDIDATE_FREQS, TRUE_FREQS)))

BASIS_FUNCTIONS = []
BASIS_NAMES = []
FREQS_MAP = [] 

BASIS_FUNCTIONS.append(np.ones(N_SAMPLES))
BASIS_NAMES.append("DC_Offset")
FREQS_MAP.append(0.0)

for f in CANDIDATE_FREQS:
    BASIS_FUNCTIONS.append(np.sin(2 * np.pi * f * T))
    BASIS_FUNCTIONS.append(np.cos(2 * np.pi * f * T))
    BASIS_NAMES.extend([f"sin(2π*{f:.1f}t)", f"cos(2π*{f:.1f}t)"])
    FREQS_MAP.extend([f, f])

X_DICT = np.vstack(BASIS_FUNCTIONS).T
M_FEATURES = X_DICT.shape[1]

# 3. VBLR Priors
a0 = 1e-4; b0 = 1e-4 
c0 = 1.0; d0 = 1.0


# --- 4. Scenario 1: LOW NOISE (Variance = 0.1) ---
NOISE_VAR_LOW = 0.1
TRUE_BETA_LOW = 1.0 / NOISE_VAR_LOW
y_noisy_low = (True_Signal + np.sqrt(NOISE_VAR_LOW) * np.random.randn(N_SAMPLES)).reshape(-1, 1)

Est_Sig_Low, ELBO_Hist_Low, Iters_Low, mN_Low, E_alpha_mean_Low, E_beta_Low, E_alpha_vector_Low = variational_bayes_linear_regression_ard(
    X_DICT, y_noisy_low, a0, b0, c0, d0
)

# --- 5. Scenario 2: HIGH NOISE (Variance = 0.5) ---
NOISE_VAR_HIGH = 0.5
TRUE_BETA_HIGH = 1.0 / NOISE_VAR_HIGH
y_noisy_high = (True_Signal + np.sqrt(NOISE_VAR_HIGH) * np.random.randn(N_SAMPLES)).reshape(-1, 1)

Est_Sig_High, ELBO_Hist_High, Iters_High, mN_High, E_alpha_mean_High, E_beta_High, E_alpha_vector_High = variational_bayes_linear_regression_ard(
    X_DICT, y_noisy_high, a0, b0, c0, d0
)


# --- 6. ANALYSIS AND PRINTING WITH FORMULA ---

def print_summary_and_formula(scenario, noise_var, true_beta, mN, E_beta, E_alpha_vector, Iters, ELBO):
    print(f"\n--- {scenario.upper()} SCENARIO (Variance={noise_var}) ---")
    print(f"Convergence Iterations: {Iters}")
    print(f"Estimated Noise Precision (E[β]): {E_beta:.4f} (True: {true_beta:.4f})")
    print(f"Final ELBO: {ELBO[-1]:.4f}")
    
    # Generate and print the inferred formula
    inferred_formula = get_inferred_model_formula(mN, BASIS_NAMES, thresh=0.08)
    print("\nInferred Model Formula (Coeffs > 0.08):")
    print(inferred_formula)
    
    # Print Top Relevances
    results = []
    for i in range(M_FEATURES):
        results.append({
            'name': BASIS_NAMES[i],
            'freq': FREQS_MAP[i],
            'coeff': mN.flatten()[i],
            'precision': E_alpha_vector[i],
        })
    # ARD selects the most *relevant* features by *shrinking* their precision E[alpha]
    results_sorted_precision = sorted(results, key=lambda x: x['precision'])

    print("\nTop 8 Selected Bases (Lowest E[α]):")
    print("{:<15} {:>8} {:>10} {:>15}".format("Basis Name", "Frequency", "Est. Coeff.", "Est. Precision (E[α])"))
    print("-" * 50)
    for r in results_sorted_precision[:8]:
        marker = " (TRUE)" if r['freq'] in TRUE_FREQS or r['freq'] == 0.0 else ""
        print("{:<15} {:>8.1f} {:>10.4f} {:>15.2f}{}".format(
            r['name'][:15], r['freq'], r['coeff'], r['precision'], marker))
    print("-" * 50)


print("\n" + "="*80)
print("ARD Basis Selection (Model 1b) Results Comparison: RANDOM SIGNAL")
print("==================================================================")
print(f"Total Features in Dictionary (M): {M_FEATURES}")
print(f"True Rhythmic Frequencies: {TRUE_FREQS}")
print(f"DC Offset: {DC_OFFSET:.4f}")

print_summary_and_formula("Random Low Noise", NOISE_VAR_LOW, TRUE_BETA_LOW, mN_Low, E_beta_Low, E_alpha_vector_Low, Iters_Low, ELBO_Hist_Low)
print_summary_and_formula("Random High Noise", NOISE_VAR_HIGH, TRUE_BETA_HIGH, mN_High, E_beta_High, E_alpha_vector_High, Iters_High, ELBO_Hist_High)

print("="*80)


# --- 7. VISUALIZATION ---

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Row 1: Low Noise
axes[0, 0].plot(T, y_noisy_low, 'b.', alpha=0.3, label=f'Noisy Target (Var={NOISE_VAR_LOW})')
axes[0, 0].plot(T, True_Signal, 'g-', linewidth=2, label='True Target Signal')
axes[0, 0].plot(T, Est_Sig_Low, 'r--', linewidth=2, label='ARD Estimated Signal')
axes[0, 0].set_title(f'Random Low Noise Fit (Var={NOISE_VAR_LOW})')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Target Amplitude')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(ELBO_Hist_Low, 'm-', linewidth=2)
axes[0, 1].set_title(f'Random Low Noise ARD ELBO (Converged in {Iters_Low} Iters)')
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].set_ylabel('ELBO (Evidence Lower Bound)')
axes[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[0, 1].grid(True)

# Row 2: High Noise
axes[1, 0].plot(T, y_noisy_high, 'b.', alpha=0.3, label=f'Noisy Target (Var={NOISE_VAR_HIGH})')
axes[1, 0].plot(T, True_Signal, 'g-', linewidth=2, label='True Target Signal')
axes[1, 0].plot(T, Est_Sig_High, 'r--', linewidth=2, label='ARD Estimated Signal')
axes[1, 0].set_title(f'Random High Noise Fit (Var={NOISE_VAR_HIGH})')
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Target Amplitude')
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].plot(ELBO_Hist_High, 'm-', linewidth=2)
axes[1, 1].set_title(f'Random High Noise ARD ELBO (Converged in {Iters_High} Iters)')
axes[1, 1].set_xlabel('Iteration')
axes[1, 1].set_ylabel('ELBO (Evidence Lower Bound)')
axes[1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()
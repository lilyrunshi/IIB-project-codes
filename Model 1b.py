import numpy as np
from scipy.special import digamma 
from scipy.special import gammaln
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# NOTE: The SyntheticData class is not needed here as we revert to the simple sinusoidal model (X_rhythm).

# --- ELBO Calculation Function for Model 1b (ARD) ---

def calculate_elbo_model1b(N, M, X, y, mN, SN, aN, bN, cN, dN, a0, b0, c0, d0):
    """Calculates the ELBO for Model 1b (ARD)."""
    
    # E[log alpha_i] is now a vector
    E_log_alpha = digamma(aN) - np.log(bN) # aN is scalar, bN is vector
    E_log_beta = digamma(cN) - np.log(dN)
    
    # E[alpha_i] is now a vector
    E_alpha_vector = aN / bN
    E_beta = cN / dN
    
    # E[omega_i^2] = mN_i^2 + S_N_ii (component-wise squared magnitude)
    E_omega_sq_vector = mN.flatten()**2 + np.diag(SN)
    
    y_residual = y - X @ mN
    E_residual_sq = y_residual.T @ y_residual + np.trace(X @ SN @ X.T)
    (sign, logdet_SN) = np.linalg.slogdet(SN)
    
    # E[log p] terms
    E_log_lik = N/2 * E_log_beta - N/2 * np.log(2 * np.pi) - E_beta/2 * E_residual_sq.item()
    E_log_prior_omega = -M/2 * np.log(2 * np.pi) + 0.5 * np.sum(E_log_alpha) - 0.5 * np.sum(E_alpha_vector * E_omega_sq_vector)
    E_log_prior_alpha = M * a0 * np.log(b0) - M * gammaln(a0) + (a0 - 1) * np.sum(E_log_alpha) - b0 * np.sum(E_alpha_vector)
    E_log_prior_beta = c0 * np.log(d0) + (c0 - 1) * E_log_beta - d0 * E_beta - gammaln(c0)
    E_log_joint = E_log_lik + E_log_prior_omega + E_log_prior_alpha + E_log_prior_beta
    
    # -E[log q] terms
    neg_E_log_q_omega = M/2 * (1 + np.log(2 * np.pi)) + 0.5 * logdet_SN
    neg_E_log_q_alpha = np.sum(gammaln(aN) - (aN - 1) * E_log_alpha + aN - aN * np.log(bN))
    neg_E_log_q_beta = gammaln(cN) - (cN - 1) * E_log_beta + cN - cN * np.log(dN)
    neg_E_log_q = neg_E_log_q_omega + neg_E_log_q_alpha + neg_E_log_q_beta

    ELBO = E_log_joint + neg_E_log_q
    return ELBO.item()

def variational_bayes_linear_regression_ard(X, y, a0, b0, c0, d0, max_iter=100, tol=1e-5):
    """
    Implements Variational Bayes Linear Regression with ARD (Model 1b).
    """
    N, M = X.shape 

    # Initial estimates (E[alpha] is a vector of size M)
    E_alpha_vector = np.ones(M) * (a0 / b0)
    E_beta = c0 / d0

    # Initialize N-parameters
    aN = a0 + 0.5 # Scalar (aN is a single value for all alpha_i, see eq (7))
    cN = c0 + N / 2 # Scalar
    bN = np.ones(M) * b0 # Vector of size M (bNi, see eq (8))
    dN = d0 # Scalar

    # E[A] is diagonal matrix of E[alpha_i]
    E_A = np.diag(E_alpha_vector) 

    # Initialize mN and SN (using initial E_A)
    SN_inv = E_beta * X.T @ X + E_A
    SN = np.linalg.inv(SN_inv)
    mN = SN @ (E_beta * X.T @ y)

    elbo_history = []
    i = 0 
    for i in range(max_iter):
        mN_old = mN.copy()

        # 1. Update bN (Vector of size M) - Eq (8)
        mN_sq_vector = mN.flatten()**2
        SN_diag_vector = np.diag(SN)
        bN = b0 + 0.5 * (mN_sq_vector + SN_diag_vector)
        
        E_alpha_vector = aN / bN
        E_A = np.diag(E_alpha_vector)

        # 2. Update dN (Scalar) - Eq (10)
        y_residual = y - X @ mN
        E_residual_sq = y_residual.T @ y_residual + np.trace(X @ SN @ X.T)
        dN = d0 + 0.5 * E_residual_sq.item()
        E_beta = cN / dN

        # 3. Update SN_inv and mN - Eq (11) and (12)
        SN_inv = E_beta * X.T @ X + E_A # E[A] is diagonal matrix
        SN = np.linalg.inv(SN_inv)
        mN = SN @ (E_beta * X.T @ y)

        # 4. ELBO Calculation 
        ELBO = calculate_elbo_model1b(N, M, X, y, mN, SN, aN, bN, cN, dN, a0, b0, c0, d0)
        elbo_history.append(ELBO)
        
        # 5. Convergence Check 
        if i > 0 and np.linalg.norm(mN - mN_old) / np.linalg.norm(mN_old) < tol:
            break

    Estimated_Signal = X @ mN
    
    # We return the mean vector of all alpha precisions for summary printing
    E_alpha_mean = np.mean(E_alpha_vector)
    
    # Return all final parameters
    return Estimated_Signal, elbo_history, i + 1, mN, E_alpha_mean, E_beta, E_alpha_vector


# --- EXPERIMENT SETUP: Sinusoidal Data ---

# 1. Generate Rhythmic Features (Basis Functions)
N_SAMPLES = 100
T = np.linspace(0, 4 * np.pi, N_SAMPLES) 
Rhythm_Freq = 1.0 
Coefficients = np.array([1.5, -2.0, 0.5]) # True coefficients: [DC, sin, cos]
M_FEATURES = len(Coefficients)

# Design matrix: (N x M)
X_rhythm = np.vstack([
    np.ones(N_SAMPLES),        # Feature 0: DC offset (Intercept)
    np.sin(Rhythm_Freq * T),   # Feature 1: Sine component
    np.cos(Rhythm_Freq * T)    # Feature 2: Cosine component
]).T
True_Target = X_rhythm @ Coefficients

# 2. Priors (Shared for both scenarios)
a0 = 1.0; b0 = 0.1   # Weak Gamma prior on alpha
c0 = 2.0; d0 = 0.5   # Prior E[beta]=4.0 (same as before)


# --- 3. Scenario 1: Low Noise (Variance = 0.25) ---
LOW_NOISE_VAR = 0.25
TRUE_BETA_LOW = 1.0 / LOW_NOISE_VAR
y_low_noise = (True_Target + np.sqrt(LOW_NOISE_VAR) * np.random.randn(N_SAMPLES)).reshape(-1, 1)

Est_Sig_Low, ELBO_Hist_Low, Iters_Low, mN_Low, E_alpha_mean_Low, E_beta_Low, E_alpha_vector_Low = variational_bayes_linear_regression_ard(
    X_rhythm, y_low_noise, a0, b0, c0, d0
)

# --- 4. Scenario 2: High Noise (Variance = 1.0) ---
HIGH_NOISE_VAR = 1.0
TRUE_BETA_HIGH = 1.0 / HIGH_NOISE_VAR
y_high_noise = (True_Target + np.sqrt(HIGH_NOISE_VAR) * np.random.randn(N_SAMPLES)).reshape(-1, 1)

Est_Sig_High, ELBO_Hist_High, Iters_High, mN_High, E_alpha_mean_High, E_beta_High, E_alpha_vector_High = variational_bayes_linear_regression_ard(
    X_rhythm, y_high_noise, a0, b0, c0, d0
)


# --- 5. ANALYSIS AND PRINTING ---

print("\n" + "="*80)
print("VBLR with ARD (Model 1b) Results on Sinusoidal Data")
print("="*80)

def print_results(scenario, noise_var, true_beta, mN, E_beta, E_alpha_vector, Iters, ELBO):
    print(f"\n--- {scenario.upper()} SCENARIO (Variance={noise_var}) ---")
    print(f"Convergence Iterations: {Iters}")
    print(f"True Noise Precision (β): {true_beta:.4f}")
    print(f"Estimated Noise Precision (E[β]): {E_beta:.4f}")
    print(f"Final ELBO: {ELBO[-1]:.4f}")
    print("-" * 50)
    print("{:<10} {:>8} {:>10} {:>10}".format("Feature", "True ω", "Est. E[ω]", "Est. E[α]"))
    print("-" * 50)
    
    feature_names = ["DC Offset", "Sine (ω=1)", "Cosine (ω=1)"]
    for i in range(M_FEATURES):
        print("{:<10} {:>8.4f} {:>10.4f} {:>10.4f}".format(
            feature_names[i], 
            Coefficients[i], 
            mN.flatten()[i], 
            E_alpha_vector[i]))
    print("-" * 50)

# Print Low Noise Results
print_results("Low Noise", LOW_NOISE_VAR, TRUE_BETA_LOW, mN_Low, E_beta_Low, E_alpha_vector_Low, Iters_Low, ELBO_Hist_Low)

# Print High Noise Results
print_results("High Noise", HIGH_NOISE_VAR, TRUE_BETA_HIGH, mN_High, E_beta_High, E_alpha_vector_High, Iters_High, ELBO_Hist_High)

print("="*80)


# --- 6. VISUALIZATION ---

# Figure 1: Low Noise (Signal and ELBO)
fig1, (ax1, ax1_elbo) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(T, y_low_noise, 'b.', alpha=0.5, label=f'Noisy Obs (Var={LOW_NOISE_VAR})')
ax1.plot(T, True_Target, 'g-', linewidth=2, label='True Signal')
ax1.plot(T, Est_Sig_Low, 'r--', linewidth=2, label='VBLR (ARD) Estimated Signal')
ax1.set_title(f'ARD Model on Low Noise (Var={LOW_NOISE_VAR}) Signal')
ax1.set_xlabel('Time (T)')
ax1.set_ylabel('Signal Amplitude')
ax1.legend()
ax1.grid(True)

ax1_elbo.plot(ELBO_Hist_Low, 'm-', linewidth=2)
ax1_elbo.set_title(f'Low Noise ARD ELBO (Converged in {Iters_Low} Iters)')
ax1_elbo.set_xlabel('Iteration')
ax1_elbo.set_ylabel('ELBO (Evidence Lower Bound)')
ax1_elbo.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1_elbo.grid(True)

plt.tight_layout()
plt.show()

# Figure 2: High Noise (Signal and ELBO)
fig2, (ax2, ax2_elbo) = plt.subplots(1, 2, figsize=(16, 6))

ax2.plot(T, y_high_noise, 'b.', alpha=0.5, label=f'Noisy Obs (Var={HIGH_NOISE_VAR})')
ax2.plot(T, True_Target, 'g-', linewidth=2, label='True Signal')
ax2.plot(T, Est_Sig_High, 'r--', linewidth=2, label='VBLR (ARD) Estimated Signal')
ax2.set_title(f'ARD Model on High Noise (Var={HIGH_NOISE_VAR}) Signal')
ax2.set_xlabel('Time (T)')
ax2.set_ylabel('Signal Amplitude')
ax2.legend()
ax2.grid(True)

ax2_elbo.plot(ELBO_Hist_High, 'm-', linewidth=2)
ax2_elbo.set_title(f'High Noise ARD ELBO (Converged in {Iters_High} Iters)')
ax2_elbo.set_xlabel('Iteration')
ax2_elbo.set_ylabel('ELBO (Evidence Lower Bound)')
ax2_elbo.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2_elbo.grid(True)

plt.tight_layout()
plt.show()
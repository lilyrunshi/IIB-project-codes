import numpy as np
from scipy.special import digamma 
from scipy.special import gammaln
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# --- ELBO Calculation Function ---

def calculate_elbo(N, M, X, y, mN, SN, aN, bN, cN, dN, a0, b0, c0, d0):
    """
    Calculates the Evidence Lower Bound (ELBO) for Model 1a.
    """
    
    # 1. Required Expectations/Quantities
    E_log_alpha = digamma(aN) - np.log(bN)
    E_log_beta = digamma(cN) - np.log(dN)
    E_omega_sq = mN.T @ mN + np.trace(SN)
    y_residual = y - X @ mN
    E_residual_sq = y_residual.T @ y_residual + np.trace(X @ SN @ X.T)
    E_alpha = aN / bN
    E_beta = cN / dN
    (sign, logdet_SN) = np.linalg.slogdet(SN)
    
    # 2. E[log p(y, omega, alpha, beta)] terms (Expected Log-Joint Probability)
    E_log_lik = N/2 * E_log_beta - N/2 * np.log(2 * np.pi) - E_beta/2 * E_residual_sq.item()
    E_log_prior_omega = -M/2 * np.log(2 * np.pi) + M/2 * E_log_alpha - E_alpha/2 * E_omega_sq.item()
    E_log_prior_alpha = a0 * np.log(b0) + (a0 - 1) * E_log_alpha - b0 * E_alpha - gammaln(a0)
    E_log_prior_beta = c0 * np.log(d0) + (c0 - 1) * E_log_beta - d0 * E_beta - gammaln(c0)
    E_log_joint = E_log_lik + E_log_prior_omega + E_log_prior_alpha + E_log_prior_beta

    # 3. -E[log q(omega, alpha, beta)] terms (Negative Entropy)
    neg_E_log_q_omega = M/2 * (1 + np.log(2 * np.pi)) + 0.5 * logdet_SN
    neg_E_log_q_alpha = gammaln(aN) - (aN - 1) * E_log_alpha + aN - aN * np.log(bN)
    neg_E_log_q_beta = gammaln(cN) - (cN - 1) * E_log_beta + cN - cN * np.log(dN)
    neg_E_log_q = neg_E_log_q_omega + neg_E_log_q_alpha + neg_E_log_q_beta

    # 4. Total ELBO
    ELBO = E_log_joint + neg_E_log_q
    return ELBO.item()

    
# --- VBLR Algorithm Implementation ---

def variational_bayes_linear_regression(X, y, a0, b0, c0, d0, max_iter=100, tol=1e-5):
    """
    Implements Variational Bayes Linear Regression (Model 1a)
    with iterative updates and ELBO tracking.
    """
    N, M = X.shape 

    # Initial estimates
    E_alpha = a0 / b0
    E_beta = c0 / d0

    # Initialize N-parameters
    aN = a0 + M / 2
    cN = c0 + N / 2

    # Initial values for bN, dN
    bN = d0
    dN = d0

    # Initialize mN and SN
    SN_inv = E_beta * X.T @ X + E_alpha * np.eye(M)
    SN = np.linalg.inv(SN_inv)
    mN = SN @ (E_beta * X.T @ y)

    elbo_history = []
    i = 0 
    for i in range(max_iter):
        mN_old = mN.copy()

        # 2. Update bN (for q*(alpha))
        E_omega_sq = mN.T @ mN + np.trace(SN)
        bN = b0 + 0.5 * E_omega_sq.item()
        E_alpha = aN / bN

        # 3. Update dN (for q*(beta))
        y_residual = y - X @ mN
        E_residual_sq = y_residual.T @ y_residual + np.trace(X @ SN @ X.T)
        dN = d0 + 0.5 * E_residual_sq.item()
        E_beta = cN / dN

        # 4. Update SN_inv and mN (for q*(omega))
        SN_inv = E_beta * X.T @ X + E_alpha * np.eye(M) 
        SN = np.linalg.inv(SN_inv)
        mN = E_beta * SN @ X.T @ y 

        # 5. ELBO Calculation 
        ELBO = calculate_elbo(N, M, X, y, mN, SN, aN, bN, cN, dN, a0, b0, c0, d0)
        elbo_history.append(ELBO)
        
        # 6. Convergence Check 
        if i > 0 and np.linalg.norm(mN - mN_old) / np.linalg.norm(mN_old) < tol:
            break

    # Calculate final estimated signal
    Estimated_Signal = X @ mN
    
    # Return all final parameters
    return Estimated_Signal, elbo_history, i + 1, mN, aN, bN, cN, dN

# --- 2. Synthetic Data Generation Setup ---

N_samples = 100
T = np.linspace(0, 4 * np.pi, N_samples) 
Rhythm_Freq = 1.0 
Coefficients = np.array([1.5, -2.0, 0.5]) 
X_rhythm = np.vstack([np.ones(N_samples), np.sin(Rhythm_Freq * T), np.cos(Rhythm_Freq * T)]).T
True_Signal = X_rhythm @ Coefficients
M_features = X_rhythm.shape[1]
True_Noise_Precision_Low = 1.0 / 0.25 
True_Noise_Precision_High = 1.0 / 1.0  

# --- 3. Noise and Priors ---

# Priors (Shared for both scenarios)
a0 = 1.0
b0 = 0.1
c0 = 2.0
d0 = 0.5 

# --- 4. Running VBLR for TWO Noise Scenarios ---

# Scenario 1: Low Noise (Variance = 0.25)
LOW_NOISE_VAR = 0.25
y_low_noise = True_Signal + np.sqrt(LOW_NOISE_VAR) * np.random.randn(N_samples)
y_low_noise = y_low_noise.reshape(-1, 1)
Est_Sig_Low, ELBO_Hist_Low, Iters_Low, mN_Low, aN_Low, bN_Low, cN_Low, dN_Low = variational_bayes_linear_regression(
    X_rhythm, y_low_noise, a0, b0, c0, d0
)

# Scenario 2: High Noise (Variance = 1.0)
HIGH_NOISE_VAR = 1.0
y_high_noise = True_Signal + np.sqrt(HIGH_NOISE_VAR) * np.random.randn(N_samples)
y_high_noise = y_high_noise.reshape(-1, 1)
Est_Sig_High, ELBO_Hist_High, Iters_High, mN_High, aN_High, bN_High, cN_High, dN_High = variational_bayes_linear_regression(
    X_rhythm, y_high_noise, a0, b0, c0, d0
)

# --- 5. Analysis: Print Estimated Parameters ---

print("\n" + "="*60)
print("VBLR Analysis Results for Low and High Noise Datasets")
print("="*60)

# Low Noise Results
E_omega_Low = mN_Low.flatten()
E_alpha_Low = aN_Low / bN_Low
E_beta_Low = cN_Low / dN_Low

print(f"--- LOW NOISE SCENARIO (Variance={LOW_NOISE_VAR}) ---")
print(f"Convergence Iterations: {Iters_Low}")
print(f"True Coefficients (ω): {Coefficients}")
print(f"Estimated Coefficients E[omega]: {E_omega_Low}")
print(f"Estimated Coeff. Precision E[alpha]: {E_alpha_Low:.4f}")
print(f"Estimated Noise Precision E[beta]: {E_beta_Low:.4f}")
print(f"Final ELBO: {ELBO_Hist_Low[-1]:.4f}")
print("-" * 60)

# High Noise Results
E_omega_High = mN_High.flatten()
E_alpha_High = aN_High / bN_High
E_beta_High = cN_High / dN_High

print(f"--- HIGH NOISE SCENARIO (Variance={HIGH_NOISE_VAR}) ---")
print(f"Convergence Iterations: {Iters_High}")
print(f"True Coefficients (ω): {Coefficients}")
print(f"Estimated Coefficients E[omega]: {E_omega_High}")
print(f"Estimated Coeff. Precision E[alpha]: {E_alpha_High:.4f}")
print(f"Estimated Noise Precision E[beta]: {E_beta_High:.4f}")
print(f"Final ELBO: {ELBO_Hist_High[-1]:.4f}")
print("="*60)

# --- 6. Visualization: Plotting ---

# Figure 1: Low Noise (Signal and ELBO)
fig1, (ax1, ax1_elbo) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(T, y_low_noise, 'b.', alpha=0.5, label=f'Noisy Obs (Var={LOW_NOISE_VAR})')
ax1.plot(T, True_Signal, 'g-', linewidth=2, label='True Signal')
ax1.plot(T, Est_Sig_Low, 'r--', linewidth=2, label='VBLR Estimated Signal')
ax1.set_title(f'Low Noise (Var={LOW_NOISE_VAR}) Signal Approximation')
ax1.set_xlabel('Time (T)')
ax1.set_ylabel('Signal Amplitude')
ax1.legend()
ax1.grid(True)

ax1_elbo.plot(ELBO_Hist_Low, 'm-', linewidth=2)
ax1_elbo.set_title(f'Low Noise ELBO Progression (Converged in {Iters_Low} Iters)')
ax1_elbo.set_xlabel('Iteration')
ax1_elbo.set_ylabel('ELBO (Evidence Lower Bound)')
ax1_elbo.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1_elbo.grid(True)

plt.tight_layout()
plt.show()

# Figure 2: High Noise (Signal and ELBO)
fig2, (ax2, ax2_elbo) = plt.subplots(1, 2, figsize=(16, 6))

ax2.plot(T, y_high_noise, 'b.', alpha=0.5, label=f'Noisy Obs (Var={HIGH_NOISE_VAR})')
ax2.plot(T, True_Signal, 'g-', linewidth=2, label='True Signal')
ax2.plot(T, Est_Sig_High, 'r--', linewidth=2, label='VBLR Estimated Signal')
ax2.set_title(f'High Noise (Var={HIGH_NOISE_VAR}) Signal Approximation')
ax2.set_xlabel('Time (T)')
ax2.set_ylabel('Signal Amplitude')
ax2.legend()
ax2.grid(True)

ax2_elbo.plot(ELBO_Hist_High, 'm-', linewidth=2)
ax2_elbo.set_title(f'High Noise ELBO Progression (Converged in {Iters_High} Iters)')
ax2_elbo.set_xlabel('Iteration')
ax2_elbo.set_ylabel('ELBO (Evidence Lower Bound)')
ax2_elbo.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2_elbo.grid(True)

plt.tight_layout()
plt.show()

# Figure 3: Global ELBO Comparison
fig_global, ax3 = plt.subplots(1, 1, figsize=(10, 6))
ax3.plot(ELBO_Hist_Low, 'r-', linewidth=2, label=f'Low Noise (Var={LOW_NOISE_VAR})')
ax3.plot(ELBO_Hist_High, 'b--', linewidth=2, label=f'High Noise (Var={HIGH_NOISE_VAR})')
ax3.set_title('Global ELBO Progression Comparison')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('ELBO (Evidence Lower Bound)')
ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
ax3.legend()
ax3.grid(True)
plt.tight_layout()
plt.show()
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr

def evaluate_model(beta_true, Y, beta_hat, predictions, selection):
    """
    Evaluate model performance with metrics specially designed for oscillatory data analysis
    
    Parameters:
    beta_true (array): True coefficients
    Y (array): True target values
    beta_hat (array): Estimated coefficients
    predictions (array): Model predictions
    selection (array): Boolean array indicating selected features
    
    Returns:
    mse_beta (float): Mean squared error of coefficient estimates
    R2 (float): R-squared score for prediction accuracy
    FDR (float): False Discovery Rate in feature selection
    Power (float): Statistical power (sensitivity) in feature selection
    """
    # 1. Coefficient Recovery Metrics
    mse_beta = mean_squared_error(beta_true, beta_hat)
    
    # Calculate relative error for non-zero coefficients
    true_nonzero = np.abs(beta_true) > 1e-10
    if np.any(true_nonzero):
        rel_error = np.mean(np.abs(beta_true[true_nonzero] - beta_hat[true_nonzero]) / 
                           np.abs(beta_true[true_nonzero]))
    else:
        rel_error = 0.0
    
    # 2. Prediction Quality Metrics
    R2 = r2_score(Y, predictions)
    diff = Y - predictions
    scale = (np.abs(Y) + np.abs(predictions)) / 2.0
    valid = scale > np.finfo(float).eps

    relative_squared_error = np.zeros_like(diff, dtype=float)
    relative_squared_error[valid] = (diff[valid] / scale[valid]) ** 2

    rmse = np.sqrt(np.mean(relative_squared_error)) * 100.0
    # Calculate rank correlation to assess pattern matching
    rho, _ = spearmanr(Y, predictions)
    
    # 3. Feature Selection Quality Metrics
    # Standard metrics
    n_selected = np.sum(selection)
    n_true_nonzero = np.sum(true_nonzero)
    
    # True/False Positives/Negatives
    TP = np.sum(selection & true_nonzero)
    FP = np.sum(selection & ~true_nonzero)
    FN = np.sum(~selection & true_nonzero)
    TN = np.sum(~selection & ~true_nonzero)
    
    # Calculate rates
    FDR = FP / max(n_selected, 1)  # False Discovery Rate
    Power = TP / max(n_true_nonzero, 1)  # Power/Recall/Sensitivity
    Precision = TP / max(n_selected, 1)
    Specificity = TN / max(np.sum(~true_nonzero), 1)
    
    # F1 score for balance between precision and recall
    F1 = 2 * (Precision * Power) / max(Precision + Power, 1e-10)
    
    # 4. Combine metrics
    # We return the original 4 metrics for DSC compatibility
    # but we store additional metrics in global variables for detailed analysis
    global additional_metrics
    additional_metrics = {
        'relative_error': rel_error,
        'rmse': rmse,
        'rank_correlation': rho,
        'precision': Precision,
        'specificity': Specificity,
        'f1_score': F1,
        'n_selected': n_selected,
        'n_true_nonzero': n_true_nonzero
    }
    
    return mse_beta, R2, FDR, Power
import numpy as np

def interquartile_range(x, keepdims=False):
    """Compute interquartile range along 1D array x"""
    q25, q75 = np.percentile(x, [25, 75], axis=0, keepdims=keepdims, method="interpolated_inverted_cdf")
    return q25, q75

def interquartile_mean(x):
    """Compute interquartile mean along 1D array x."""
    q25, q75 = interquartile_range(x, keepdims=True)
    mask = (x >= q25) & (x <= q75)
    return np.sum(x * mask, axis=0) / mask.sum(axis=0)

def get_bootstrapped_iqm_ci(
    data: np.ndarray,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the Interquartile Mean (IQM) and its Bootstrapped 95% CI.
    
    Args:
        data: Array of shape (n_seeds, n_points)
        
    Returns:
        iqm_mean: The IQM of the original data.
        ci_lower: The lower bound (alpha/2 percentile of bootstrapped IQMs).
        ci_upper: The upper bound (1 - alpha/2 percentile of bootstrapped IQMs).
    """
    rng = np.random.default_rng(seed)
    n_seeds, n_points = data.shape
    
    # 1. Compute Point Estimate (IQM on raw data)
    # -------------------------------------------
    q25, q75 = np.percentile(data, [25, 75], axis=0, method="interpolated_inverted_cdf")
    mask = (data >= q25[None, :]) & (data <= q75[None, :])
    iqm_point = np.sum(data * mask, axis=0) / np.maximum(mask.sum(axis=0), 1.0)

    # 2. Vectorized Bootstrap
    # -----------------------
    boot_idx = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
    
    # Resample data: (n_boot, n_seeds, n_points)
    boot_samples = data[boot_idx] 

    # Compute IQM across axis 1 (seeds) for all bootstraps at once
    # Quantiles shape: (2, n_boot, n_points)
    b_q25, b_q75 = np.percentile(
        boot_samples, [25, 75], axis=1, method="interpolated_inverted_cdf"
    )
    
    # Mask shape: (n_boot, n_seeds, n_points)
    # We expand dims of quantiles to broadcast over the seeds axis (axis 1)
    b_mask = (boot_samples >= b_q25[:, None, :]) & (boot_samples <= b_q75[:, None, :])
    
    # Compute IQM for each bootstrap sample
    boot_iqms = np.sum(boot_samples * b_mask, axis=1) / np.maximum(b_mask.sum(axis=1), 1.0)
    
    # 3. Compute Confidence Intervals
    # -------------------------------
    ci_lower = np.percentile(boot_iqms, 100 * (alpha / 2), axis=0)
    ci_upper = np.percentile(boot_iqms, 100 * (1 - alpha / 2), axis=0)
    
    return iqm_point, ci_lower, ci_upper
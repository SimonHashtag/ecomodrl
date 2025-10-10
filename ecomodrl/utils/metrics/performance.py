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
import numpy as np
from dwt_mra import dwt_mra  # Make sure this is in the same folder

def wasp(Y, X, method='dwtmra', wavelet_name='db1', level=3, flag_sign=False):
    """
    WASP: Wavelet-based Signal Processing using DWT multiresolution analysis (dwt_mra)

    Parameters:
        Y : 1D array (length N), response variable
        X : 2D array (N+N_fc x n_var), predictors
        method : str, should be 'dwtmra'
        wavelet_name : str, e.g., 'db1'
        level : int, decomposition levels
        flag_sign : bool, optional sign adjustment

    Returns:
        X_WaSP : refined predictors (same shape as X)
        C : covariance weights per level per variable (shape: level+1 x n_var)
    """
    N = len(Y)
    num_obs, num_var = X.shape
    X_WaSP = np.full((num_obs, num_var), np.nan)
    C = np.full((level + 1, num_var), np.nan)

    for i in range(num_var):
        x_raw = X[:, i]
        x_centered = x_raw - np.mean(x_raw)

        # === Use custom DWT MRA ===
        X_DWT = dwt_mra(x_centered, wavelet_name, level=level)  # shape: (num_obs, level+1)

        # === Standardization ===
        X_DWT_cal = X_DWT[:N, :]
        mean_cal = np.mean(X_DWT_cal, axis=0)
        std_cal = np.std(X_DWT_cal, axis=0, ddof=0)

        # Set a small threshold to avoid division by zero
        epsilon = 1e-8
        std_cal_safe = np.where(std_cal < epsilon, 1.0, std_cal)
        # print(std_cal_safe)

        X_DWT_centered = X_DWT - mean_cal
        X_DWT_norm = X_DWT_centered / std_cal_safe

        # === Covariance (Eq. 10 WRR2020) ===
        C[:, i] = (1 / (N - 1)) * (Y @ X_DWT_norm[:N, :])

        # === Normalize to unit norm ===
        C_norm = C[:, i] / np.sqrt(np.sum(C[:, i] ** 2))

        # === Variance Transformation (Eq. 9 WRR2020) ===
        std_orig = np.std(x_raw[:N], ddof=0)
        x_ref = X_DWT_norm @ (std_orig * C_norm)

        # === Sign check (optional) ===
        if flag_sign:
            rho = np.corrcoef(x_ref, x_centered)[0, 1]
            if rho < 0:
                C_norm = -C_norm
                x_ref = X_DWT_norm @ (std_orig * C_norm)

        # === Add mean back ===
        X_WaSP[:, i] = x_ref + np.mean(x_raw)

    return X_WaSP, C

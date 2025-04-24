import numpy as np
from dwt_mra import dwt_mra 

def wasp_val(X, C, method='dwtmra', wavelet_name='db1', flag_sign=False):
    """
    Python version of WaSP_val.m using custom dwt_mra for validation period.

    Parameters:
        X : 2D array (n_obs x n_var), predictors (validation or forecast)
        C : 2D array (level+1 x n_var), covariance weights from calibration
        method : str, must be 'dwtmra'
        wavelet_name : str, e.g., 'db1'
        flag_sign : bool, if sign correction is applied

    Returns:
        X_WaSP : variance-transformed predictors (same shape as X)
    """
    num_obs, num_var = X.shape
    lev = C.shape[0] - 1
    X_WaSP = np.full((num_obs, num_var), np.nan)

    for i in range(num_var):
        x_raw = X[:, i]
        x_centered = x_raw - np.mean(x_raw)

        # Padding 
        min_length = 2 ** (lev + 1)
        if len(x_centered) < min_length:
            n_rep = int(np.ceil(min_length / len(x_centered)))
            x_centered_padded = np.tile(x_centered, n_rep + 1)[:min_length]
        else:
            x_centered_padded = x_centered

        # Apply DWT MRA
        X_DWT_full = dwt_mra(x_centered_padded, wavelet_name, level=lev)
        X_DWT = X_DWT_full[:num_obs, :]  # crop back to original size

        # Standardization
        mean_dwt = np.mean(X_DWT, axis=0)
        std_dwt = np.std(X_DWT, axis=0, ddof=0)

        # Set a small threshold to avoid division by zero
        epsilon = 1e-8
        std_dwt_safe = np.where(std_dwt < epsilon, 1.0, std_dwt)
        # print(std_dwt_safe)

        X_DWT_norm = (X_DWT - mean_dwt) / std_dwt_safe

        # Normalize C
        C_norm = C[:, i] / np.sqrt(np.sum(C[:, i] ** 2))

        # Variance transformation
        std_orig = np.std(x_raw, ddof=0)
        x_trans = X_DWT_norm @ (std_orig * C_norm)

        # Optional sign flip
        if flag_sign:
            rho = np.corrcoef(x_trans, x_raw)[0, 1]
            if rho < 0:
                C_norm = -C_norm
                x_trans = X_DWT_norm @ (std_orig * C_norm)

        # Add mean back
        X_WaSP[:, i] = x_trans + np.mean(x_raw)

    return X_WaSP

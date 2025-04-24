import numpy as np
import pywt

def dwt_mra(X, wavelet_name, level):
    
    coeffs = pywt.wavedec(X, wavelet_name, level=level)
    N = len(X)
    X_DWT_MRA = []

    for i in range(1, level + 1):
        
        cA = np.zeros_like(coeffs[0])
        cDs = [np.zeros_like(c) for c in coeffs[1:]]
        cDs[i - 1] = coeffs[i]
        x_rec = pywt.waverec([cA] + cDs, wavelet_name)
        X_DWT_MRA.append(x_rec[:N])

    # Final approximation
    x_approx = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], wavelet_name)
    X_DWT_MRA.append(x_approx[:N])

    return np.stack(X_DWT_MRA, axis=1)

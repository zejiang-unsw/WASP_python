import numpy as np
import matplotlib.pyplot as plt
from dwt_mra import dwt_mra

fs = 50
dt = 1 / fs
N = 500
t = np.arange(0, N) * dt

# Sine signal with random phase + noise
phase = np.random.randn(1).item()
X = np.sin(2 * np.pi * t + phase) + 0.1 * np.random.randn(N)

n_vanish = 1
wavelet_name = f'db{n_vanish}'
level = int(np.floor(np.log2(N))) - 1

X_DWT_MRA = dwt_mra(X, wavelet_name, level)

plt.figure(figsize=(10, 8))
plt.subplot(level + 2, 1, 1)
plt.plot(t, X)
plt.title('Original Signal')
plt.ylabel('Amplitude')
plt.grid(True)

for i in range(level):
    plt.subplot(level + 2, 1, i + 2)
    plt.plot(t, X_DWT_MRA[:, i])
    plt.ylabel(f'Detail L{i+1}')
    plt.grid(True)

plt.subplot(level + 2, 1, level + 2)
plt.plot(t, X_DWT_MRA[:, -1])
plt.ylabel(f'Approx L{level}')
plt.xlabel('Time (s)')
plt.grid(True)

plt.tight_layout()
plt.show()
# Check additivity and variance conservation
X_tmp = X[:X_DWT_MRA.shape[0]]  # ensure same length

additive_error = np.sum(np.abs(np.sum(X_DWT_MRA, axis=1) - X_tmp))
variance_error = np.sum(np.var(X_DWT_MRA, axis=0)) - np.var(X_tmp)

print(f'Additive: {additive_error:.6f}')
print(f'Variance: {variance_error:.6f}')

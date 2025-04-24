import numpy as np
import matplotlib.pyplot as plt
from wasp import wasp  # Make sure this is in the same folder

# === Parameters ===
N = 200          # Number of observations
N_fc = 50        # Forecast length
n_var = 4        # Number of predictor variables
fs = 50
dt = 1 / fs
t = np.arange(0, N + N_fc) * dt

# === Generate synthetic Y and X ===
Y_ALL = (np.sin(2 * np.pi * t + np.random.randn(1)) + 0.1 * np.random.randn(len(t))).flatten()
X = np.random.randn(N + N_fc, n_var)

Y = Y_ALL[:N]
Y_val = Y_ALL[N:]

# === Wavelet parameters ===
n_vanish = 2
wname = f'db{n_vanish}'
flag_sign = True
method = 'dwtmra'  # Only dwtmra is implemented in our wasp()

# === Decomposition level ===
lev = int(np.floor(np.log2(X.shape[0])) - 1)

# === Apply WASP ===
X_WaSP, C = wasp(Y, X, method=method, wavelet_name=wname, level=lev, flag_sign=flag_sign)

# === RMSE Calculation ===
RMSE = np.full(n_var, np.nan)
RMSE_WaSP = np.full(n_var, np.nan)
RMSE_opti = np.full(n_var, np.nan)

for i in range(n_var):
    x_raw = X[:N, i]
    x_wasp = X_WaSP[:N, i]

    # Optimal RMSE (WRR2020 Eq. 12)
    ratio = np.var(x_raw) / np.var(x_wasp)
    # RMSE_opti[i] = np.sqrt((N - 1) / N * (np.var(Y) - (np.linalg.norm(C[:, i]) ** 2) * ratio))
    RMSE_opti[i] = np.sqrt((np.var(Y) - (np.linalg.norm(C[:, i]) ** 2) * ratio)) # different variance calculation in python

    # Standard linear regression
    p1 = np.polyfit(x_raw, Y, 1)
    Y_fit_std = np.polyval(p1, x_raw)
    RMSE[i] = np.sqrt(np.mean((Y - Y_fit_std) ** 2))

    # WaSP regression
    p2 = np.polyfit(x_wasp, Y, 1)
    Y_fit_wasp = np.polyval(p2, x_wasp)
    RMSE_WaSP[i] = np.sqrt(np.mean((Y - Y_fit_wasp) ** 2))

# === Print results ===
print("RMSE_WaSP:", RMSE_WaSP)
print("RMSE_opt:", RMSE_opti)

# === Plot RMSE Comparison ===
plt.figure(figsize=(8, 5))
bar_width = 0.25
x = np.arange(n_var)

plt.bar(x, RMSE, width=bar_width, label='Std')
plt.bar(x + bar_width, RMSE_WaSP, width=bar_width, label='VT')
plt.bar(x + 2 * bar_width, RMSE_opti, width=bar_width, label='Optimal')

plt.xticks(x + bar_width, [f'{i+1}' for i in x])  # Show 1, 2, 3, 4

plt.xlabel('Variable Number')
plt.ylabel('RMSE')
plt.title(f'Variance transformation based on {method.upper()} using {wname}')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('RMSE.png', dpi=300)
plt.show()


# === Plot Y, X, and X_WaSP ===
plt.figure(figsize=(10, 8))
plt.suptitle(f'Calibration: {method.upper()} using {wname}')
for i in range(n_var):
    plt.subplot(n_var, 1, i + 1)
    plt.plot(Y, 'k', label='Y')
    plt.plot(X[:, i], 'b', label='X')
    plt.plot(X_WaSP[:, i], 'r', label='Xnew')
    plt.xlim([0, N + N_fc])
    if i == 0:
        plt.legend(loc='upper right', ncol=1)
    plt.ylabel(f'Var {i+1}')
plt.xlabel('Time Step')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('comparison.png')
plt.show()


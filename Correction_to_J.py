import numpy as np
import matplotlib.pyplot as plt

# Parameters
beta = 2.0

# Grid for Δ and Δ'
Delta = np.linspace(0.0, 10.0, 400)
Delta1 = np.linspace(0.0, 10.0, 400)  # Δ' axis
D, D1 = np.meshgrid(Delta, Delta1)

# s = sqrt(Δ^2 + (Δ')^2)
s = np.sqrt(D**2 + D1**2)

# Partition functions
Z  = 2.0 * np.cosh(beta * D / 2.0)
Zp = 2.0 * np.cosh(beta * D1 / 2.0)

# Exponential average <e^{-βW}>_Ω
# Handle the D = D1 = 0 point safely to avoid 0/0 in D/s
ratio = np.divide(D, s, out=np.zeros_like(D), where=(s != 0))

expavg = (np.cosh(beta * s / 2.0) - ratio * np.tanh(beta * D / 2.0) * np.sinh(beta * s / 2.0))

# Xi_Omega
Xi = np.log(expavg/(Zp/Z))

# Plot
fig, ax = plt.subplots()
im = ax.imshow(
    Xi,
    origin="lower",
    extent=[Delta.min(), Delta.max(), Delta1.min(), Delta1.max()],
    aspect="auto",
)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label(r"$\Xi_\Omega$", fontsize=18)
cbar.ax.tick_params(labelsize=15)

ax.set_xlabel(r"$\Delta$", fontsize=18)
ax.set_ylabel(r"$\Delta'$", fontsize=18)
ax.tick_params(axis="both", labelsize=18)

plt.tight_layout()
fig.savefig("Xi_Omega.pdf")
plt.show()
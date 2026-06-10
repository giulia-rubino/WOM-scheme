import numpy as np
import matplotlib.pyplot as plt

# \eta range
eta = np.linspace(0.0, 1.0, 500)

# \Delta'/\Delta ratios
r_list = [0.5, 1.0, 2.0, 3.0]

def R_Omega(eta, r):
    return (eta**2 * r**2) / (1 + r**2 * (1 - eta**2))

fig, ax = plt.subplots(figsize=(8, 5))

# HEX colours for HOW curves
colors = ["#574964", "#9F8383", "#C8AAAA", "#FFDAB3"]

for r, c in zip(r_list, colors):
    ax.plot(
        eta, R_Omega(eta, r),
        color=c, linewidth=3,
        label=rf"$\Delta'/\Delta = {r}$"
    )

# TPM baseline (zero)
ax.axhline(0.0, color="#000000", linestyle="--", linewidth=3, label=r"$\mathrm{SNR}_\mathrm{TPM}$")

ax.set_xlabel(r"$\eta$", fontsize=24)
ax.set_ylabel(r"$\mathrm{SNR}_\Omega$", fontsize=24)
ax.set_xlim(0, 1)
ax.tick_params(axis="both", labelsize=24)

ax.legend(fontsize=24)  # 32 is usually huge; adjust as you like
plt.tight_layout()

plt.show()
fig.savefig("SNR_example2.pdf")

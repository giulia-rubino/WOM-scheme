import numpy as np
import matplotlib.pyplot as plt

beta = 2.0

# Sweep Δ
Delta = np.linspace(1e-6, 5.0, 800)  # start slightly above 0 to avoid 0/0 at Δ=Δ'=0
# Choose several Δ' curves to compare on the same axes
Delta1_list = [0.5, 1.0, 2.0, 3.0]

def Z_of(Delta, beta):
    return 2.0 * np.cosh(0.5 * beta * Delta)

def kl_binary(pF_plus, pB_plus, eps=1e-15):
    # D_KL( p_F || p_B ) for binary distributions
    pF_plus = np.clip(pF_plus, eps, 1 - eps)
    pB_plus = np.clip(pB_plus, eps, 1 - eps)
    pF_minus = 1.0 - pF_plus
    pB_minus = 1.0 - pB_plus
    return pF_plus * np.log(pF_plus / pB_plus) + pF_minus * np.log(pF_minus / pB_minus)

def pF_pB_qubit(Delta, Delta1, beta):
    s = np.sqrt(Delta**2 + Delta1**2)

    ratio_F = np.divide(Delta, s, out=np.zeros_like(Delta), where=(s != 0))
    ratio_B = np.divide(Delta1, s, out=np.zeros_like(Delta), where=(s != 0))

    tF = np.tanh(0.5 * beta * Delta)
    tB = np.tanh(0.5 * beta * Delta1)

    pF_plus  = 0.5 * (1.0 + ratio_F * tF)
    pF_minus = 0.5 * (1.0 - ratio_F * tF)

    pB_plus  = 0.5 * (1.0 - ratio_B * tB)
    pB_minus = 0.5 * (1.0 + ratio_B * tB)

    return pF_plus, pF_minus, pB_plus, pB_minus

def sigmaOmega_mean(Delta, Delta1, beta):
    pF_plus, pF_minus, pB_plus, pB_minus = pF_pB_qubit(Delta, Delta1, beta)
    return kl_binary(pF_plus, pB_plus)

def sigmaTPM_mean(Delta, Delta1, beta):
    Z  = Z_of(Delta, beta)
    Zp = Z_of(Delta1, beta)

    W_TPM = 0.5 * Delta * np.tanh(0.5 * beta * Delta)
    DeltaF = -(1.0 / beta) * np.log(Zp / Z)

    return beta * (W_TPM - DeltaF)

fig, ax = plt.subplots(figsize=(9, 6))

# HEX colours: one per Δ'
colors_HOW = ["#005461", "#0C7779", "#249E94", "#3BC1A8"]   # solid lines
colors_TPM = ["#E17564", "#BE3144", "#872341", "#09122C"]   # dashed lines

for Delta1, c_how, c_tpm in zip(Delta1_list, colors_HOW, colors_TPM):
    sO = sigmaOmega_mean(Delta, Delta1, beta)
    sT = sigmaTPM_mean(Delta, Delta1, beta)

    # WOM (solid), TPM (dashed)
    ax.plot(Delta, sO, color=c_how, linewidth=3, label=rf"$\langle\sigma_\Omega\rangle$ ($\Delta'={Delta1}$)")
    ax.plot(Delta, sT, color=c_tpm, linestyle="--", linewidth=3, label=rf"$\langle\sigma_{{\rm TPM}}\rangle$ ($\Delta'={Delta1}$)")

ax.set_xlabel(r"$\Delta$", fontsize=20)
ax.set_ylabel(r"$\langle\sigma\rangle$", fontsize=20)
ax.tick_params(axis="both", labelsize=20)
ax.axhline(0.0, linewidth=1, color="black")

ax.legend(fontsize=20, ncol=2, loc="upper right")
plt.tight_layout()
fig.savefig("sigma_compare.pdf")
plt.show()

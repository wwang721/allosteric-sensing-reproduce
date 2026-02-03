import numpy as np
import matplotlib.pyplot as plt


FI = np.load("FI.npy")

g = 0.05
KD = 100
alpha = 10
n = 400000  # number of receptors

C_listA = np.logspace(0, 1.5, 8)
C_listB = np.logspace(1.5, 2.5, 20)
C_list = []
for C in C_listA:
    C_list.append(C)
for C in C_listB:
    C_list.append(C)
C_list = np.array(C_list)


fig, ax = plt.subplots(1, 1, figsize=(4.3, 3))

C_list1 = np.logspace(0, 3., 100)
FI_perfect = np.zeros_like(C_list1)
for i in range(len(FI_perfect)):
    if C_list1[i] <= KD/alpha:
        FI_perfect[i] = n*g**2 * C_list1[i] * \
            (KD/alpha) / (8 * (C_list1[i] + (KD/alpha))**2)
    elif C_list1[i] > KD/alpha and C_list1[i] < KD:
        FI_perfect[i] = n * g**2 / 32
    elif C_list1[i] >= KD:
        FI_perfect[i] = n*g**2 * C_list1[i] * KD / (8 * (C_list1[i] + KD)**2)


ax.set_xscale('log')

ax.plot(C_list1, FI_perfect, '-', color="C2", label=r"Optimal", lw=2, zorder=1)

plt.plot(C_list, FI[:, 0], 's--', label=r"$K_M = 10^{-4}\,$nM", color="C0",
         lw=2, markerfacecolor="None", markersize=6, markeredgewidth=1.5, zorder=3,)
plt.plot(C_list, FI[:, 1], 'o-.', label=r"$K_M = 10^{-3}\,$nM", color="C1",
         lw=2, markersize=5.75, markerfacecolor="None", markeredgewidth=1.5, zorder=3,)
plt.plot(C_list, FI[:, 2], '^:', label=r"$K_M = 10^{-2}\,$nM", color="C4", lw=2,
         markerfacecolor="None", markersize=5.5, markeredgewidth=1.5, zorder=3, clip_on=False)

ax.axvspan(KD/alpha, KD, facecolor='C2', alpha=0.1, edgecolor='none', zorder=0)

ax.set_xlabel(r"Mean ligand concentration $C_0$ [nM]")
ax.set_ylabel(r"Fisher information $I_{\phi\phi}$")
ax.set_xlim(1, 400)
ax.set_ylim(5.1, 34)
plt.legend(frameon=False, fontsize=12)
plt.savefig('FI_vary_KM.png', dpi=150, bbox_inches='tight')

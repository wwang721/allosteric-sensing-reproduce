import numpy as np
import matplotlib.pyplot as plt

max_steps = 600000
time = [0]

# reaction terms
C_i = 110
C_f = 111  # initial and final (mean) ligand concentrations
C0 = np.zeros(max_steps + 1)
C0[0:int(max_steps/3)] = C_i
C0[int(max_steps/3):] = C_f

G_i = 1  # initial G-protein concentration

g = 0.05
phi = np.pi/6

kc = 1  # time in units of 1/kc
KD = 100
KG = 1
alpha = 10
KM = 0.00001  # Michaelis-Menten constant


# Initializing array
l = 40  # number of modeled G-protein sites
n = 400000  # number of receptors
theta = np.linspace(-np.pi, np.pi, l+1)
theta = theta[0:l]

# Diffusion terms
r = 1  # position in units of r
dt = 0.00001
dx = 2*np.pi*r/l
D = 100  # units of x^2/t
# D*dt/dx**2  # checking stability - this has to be < 0.5


C_list1 = np.logspace(0, 3, 100)

D_h = 50
D_l = 0.1
FI_high_pert = n * g**2 / 32 * \
    (1 - (alpha*C_list1 - KD)**2 / (2 * D_h * (alpha-1) * C_list1 * KD * KG))**2

delta = (alpha*C_list1 - KD)
eps = 1e-16
FI_low_pert = D_l**2 * g**2 * n * \
    (alpha-1)**2 * C_list1**2 * KD**2 * KG**2 / (8 * np.maximum(delta, eps)**4)


FI_perfect = np.zeros_like(C_list1)
for i in range(len(FI_perfect)):
    if C_list1[i] <= KD/alpha:
        FI_perfect[i] = n*g**2 * C_list1[i] * \
            (KD/alpha) / (8 * (C_list1[i] + (KD/alpha))**2)
    elif C_list1[i] > KD/alpha and C_list1[i] < KD:
        FI_perfect[i] = n * g**2 / 32
    elif C_list1[i] >= KD:
        FI_perfect[i] = n*g**2 * C_list1[i] * KD / (8 * (C_list1[i] + KD)**2)

FI_single_rec = n * g**2 * C_list1 * KD / (C_list1 + KD)**2 / 8
FI_two_rec = n * g**2 * C_list1 * KD / \
    (C_list1 + KD)**2 / 16 + n * g**2 * C_list1 * \
    KD/alpha / (C_list1 + KD/alpha)**2 / 16


fig, ax = plt.subplots(1, 1, dpi=100, figsize=(4, 3))

ax.set_xscale('log')

ax.plot(C_list1, FI_perfect, '-', color="C2", label=r"Adjust $K_\mathrm{eff}$ to $C_0$", lw=2, zorder=1)
ax.plot(C_list1, FI_single_rec, '--', color='C3', label="Single receptor type", lw=2, zorder=2)
ax.plot(C_list1, FI_two_rec, '-.', color="C0", label="Two receptor types", lw=2, zorder=1)
ax.axvspan(KD/alpha, KD, facecolor='C2', alpha=0.1, edgecolor='none', zorder=0)

ax.set_xlabel(r"Mean ligand concentration $C_0$ [nM]")
ax.set_ylabel(r"Optimal Fisher information $I_{\phi\phi}$")
ax.set_xlim(1, 1000)
ax.set_ylim(0, 34)
ax.set_yticks([5, 15, 25], minor=True)
plt.legend(frameon=False, fontsize=11)
plt.savefig('FI_comparing.png', dpi=150, bbox_inches='tight')

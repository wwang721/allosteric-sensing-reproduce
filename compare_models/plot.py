import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import AutoMinorLocator


data = np.load('data.npy')
num_rows = data.shape[0]
performance_measure, performance_measure1 = data[:int(num_rows/2)], data[int(num_rows/2):]

max_steps = 400000
g = 0.05
phi = np.pi/6

KD = 100
alpha = 10

# Initializing array
l = 40  # number of modeled G-protein sites
n = 400000  # number of receptors
theta = np.linspace(-np.pi, np.pi, l+1)
theta = theta[0:l]

D_list = np.logspace(-1, 2, 15)
kc = 1
kc_list = np.logspace(-2, 1, 10)
D = 50
KM = 0.0001
C_f = KD/alpha - 1
C_i = KD / np.sqrt(alpha)
C0 = np.zeros(max_steps + 1)
C_profile = C_f * np.exp(g/2 * np.cos(theta - phi))
C_list = np.logspace(0.5, 2.5, 20)

D_mesh, C_mesh = np.meshgrid(D_list, C_list)


#=============================================================
# vs single-receptor-type
#=============================================================

fig, ax = plt.subplots(1, 1, dpi=100, figsize=(3.5, 3))
background = ax.pcolormesh(C_mesh, D_mesh, performance_measure,
                           shading='nearest', vmax=24, vmin=-24, cmap=cm.RdBu_r, alpha=0.8)

ax.set_title("vs single-receptor-type", pad=10)

cbar = fig.colorbar(background)
cbar.ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.axvline(x=KD/alpha, linestyle='--', lw=1.5, color="black", alpha=0.6)
ax.axvline(x=KD, linestyle='--', lw=1.5, color="black", alpha=0.6)


ax.set_ylabel(r"$\tilde{\,D}$")

ax.set_xlabel(r"$C_0$ [nM]")

ax.set_xscale('log')
ax.set_yscale('log')

cbar.ax.set_title(r"$~~\Delta I_{\phi\phi}$", pad=5)
plt.savefig("vs1type.png", dpi=150, bbox_inches='tight')


#=============================================================
# vs two-receptor-type
#=============================================================

fig, ax = plt.subplots(1, 1, dpi=100, figsize=(3.5, 3))
background = ax.pcolormesh(C_mesh, D_mesh, performance_measure1,
                           shading='nearest', vmax=24, vmin=-24, cmap=cm.RdBu_r, alpha=0.8)

ax.set_title("vs two-receptor-type", pad=10)

cbar = fig.colorbar(background)
cbar.ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.axvline(x=KD/alpha, linestyle='--', lw=1.5, color="black", alpha=0.6)
ax.axvline(x=KD, linestyle='--', lw=1.5, color="black", alpha=0.6)


ax.set_ylabel(r"$\tilde{\,D}$")

ax.set_xlabel(r"$C_0$ [nM]")

ax.set_xscale('log')
ax.set_yscale('log')

cbar.ax.set_title(r"$~~\Delta I_{\phi\phi}$", pad=5)
plt.savefig("vs2type.png", dpi=150, bbox_inches='tight')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


data = np.load('FI.npy')
num_rows = data.shape[0]
FI_10, FI_100 = data[:int(num_rows/2)], data[int(num_rows/2):]


KD = 100
alpha = 10

alpha_list = np.logspace(0, 3, 15)
C_list = np.logspace(-1, 3, 15)

alpha_mesh, C_mesh = np.meshgrid(alpha_list, C_list)

#=============================================================
# D = 10
#=============================================================

fig, ax = plt.subplots(1, 1, dpi=100, figsize=(3.5, 3))
background = ax.pcolormesh(C_mesh, alpha_mesh, FI_10, shading='nearest', cmap="YlGn_r", vmin=0, alpha=0.7, vmax=32, zorder=1)

ax.plot([KD, KD], [0, 1e3], '--', lw=1.5, color="gray", zorder=2)
ax.plot([KD/1e3, KD], [1e3, 1], '--', lw=1.5, color="gray", zorder=2)

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlim(1e-1, 1e3)
ax.set_ylim(1, 1e3)

ax.set_title(r"$\tilde{\,D}=%g$" % 10, pad=10)

cbar = fig.colorbar(background, ax=ax, ticks=np.linspace(0, 35, 8))
cbar.ax.set_title(r"$~I_{\phi\phi}$", pad=10)  # Add title with padding
cbar.ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_ylabel(r"$\alpha$")
ax.set_xlabel(r"$C_0$ [nM]")
plt.savefig("D10.png", dpi=150, bbox_inches='tight')


#=============================================================
# D = 100
#=============================================================

fig, ax = plt.subplots(1, 1, dpi=100, figsize=(3.5, 3))
background = ax.pcolormesh(C_mesh, alpha_mesh, FI_100, shading='nearest', cmap="YlGn_r", vmin=0, alpha=0.7, vmax=32, zorder=1)

ax.plot([KD, KD], [0, 1e3], '--', lw=1.5, color="gray", zorder=2)
ax.plot([KD/1e3, KD], [1e3, 1], '--', lw=1.5, color="gray", zorder=2)

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlim(1e-1, 1e3)
ax.set_ylim(1, 1e3)

ax.set_title(r"$\tilde{\,D}=%g$" % 100, pad=10)

cbar = fig.colorbar(background, ax=ax, ticks=np.linspace(0, 35, 8))
cbar.ax.set_title(r"$~I_{\phi\phi}$", pad=10)  # Add title with padding
cbar.ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_ylabel(r"$\alpha$")
ax.set_xlabel(r"$C_0$ [nM]")

plt.savefig("D100.png", dpi=150, bbox_inches='tight')

import numpy as np
import matplotlib.pyplot as plt


C_i = 80
C_f = 10  # initial and final (mean) ligand concentrations
l = 300000
C = np.zeros(l)
C[0:int(l/3)] = C_i
C[int(l/3):] = C_f

kc = .01
dt = .01
KD = 100
KG = 1
alpha = 100
KM = .001  # Michaelis-Menten constant
G_tot = 10
f_bound = []

# initial G-protein concentration
G_active_i = KG * (KD-C_i) / (alpha*C_i - KD)

G = [G_active_i]
t = [0]
times = np.linspace(0, dt*l, l)

for i in range(l):
    t.append(t[i]+dt)
    K_eff = KD * (G[i]+KG)/(alpha*G[i]+KG)

    f_b = C[i] / (C[i] + K_eff)

    dGdt = kc*(1-f_b)*(G_tot - G[i]) / \
        (KM + G_tot - G[i]) - kc*f_b*G[i]/(KM + G[i])

    G.append(G[i] + dt*dGdt)
    f_bound.append(f_b)
G.pop(0)


plt.rcParams['xtick.top'] = False

fig, ax = plt.subplots(3, 1, figsize=(3.5, 3), dpi=100)
plt.subplots_adjust(hspace=0)
ax[0].plot(times, C, color="C0", linewidth=2)
ax[0].set_ylabel("$C$ [nM]", labelpad=15)

ax[1].plot(times, G, color="C3", linewidth=2)
ax[1].set_ylabel("$G$ [nM]")

ax[2].plot(times, f_bound, color="C7", linewidth=2)
ax[2].set_ylabel("$f_b$")
ax[2].set_xlabel(r"Time [$K_G/V_\mathrm{max}$]")


ax[0].xaxis.set_tick_params(labelsize=0)
ax[1].xaxis.set_tick_params(labelsize=0)


ax[0].set_xticks(range(800, 1201, 50))
ax[1].set_xticks(range(800, 1201, 50))
ax[2].set_xticks(range(800, 1201, 50))
ax[0].set_xticks(range(925, 1151, 50), minor=True)
ax[1].set_xticks(range(925, 1151, 50), minor=True)
ax[2].set_xticks(range(925, 1151, 50), minor=True)

ax[0].set_xlim(900, 1100)
ax[1].set_xlim(900, 1100)
ax[2].set_xlim(900, 1100)

ax[0].set_ylim(0, 95)
ax[1].set_ylim(0.0, 0.12)
ax[2].set_ylim(0., 0.65)

ax[0].set_yticks([0, 40, 80])
ax[1].set_yticks([0.0, 0.05, 0.1])
ax[2].set_yticks([0., 0.25, 0.5])

plt.savefig("time_series.png", dpi=150, bbox_inches='tight')

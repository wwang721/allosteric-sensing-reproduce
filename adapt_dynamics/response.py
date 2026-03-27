import numpy as np
import matplotlib.pyplot as plt


C_i = 80
C_f = 10  # initial and final (mean) ligand concentrations
l = 300000

kc = .01
dt = .01
KD = 100
KG = 1
alpha = 100
KM = .001  # Michaelis-Menten constant
G_tot = 10

C_list = np.logspace(-2, 4, 50)
FI = []
fb_list = []
G_list = []
G_pa = []
FI_pa = np.zeros_like(C_list)  # in case of perfect adaptation
for i in range(len(FI_pa)):
    if C_list[i] < KD/alpha:
        FI_pa[i] = C_list[i]*KD/alpha / (C_list[i] + KD/alpha)**2
    elif C_list[i] > KD:
        FI_pa[i] = C_list[i]*KD / (C_list[i] + KD)**2
    else:
        FI_pa[i] = 1/4


for C_f in C_list:
    C = np.zeros(l)
    C[0:int(l/3)] = C_i
    C[int(l/3):] = C_f

    G_i = 1  # initial G-protein concentration

    G = [G_i]
    t = [0]

    for i in range(l):
        t.append(t[i]+dt)
        K_eff = KD * (G[i]+KG)/(alpha*G[i]+KG)
        f_b = C[i] / (C[i] + K_eff)

        dGdt = kc*(1-f_b)*(G_tot - G[i])/(KM +
                                          G_tot - G[i]) - kc*f_b*G[i]/(KM + G[i])

        G.append(G[i] + dt*dGdt)
    FI.append(C[-1]*K_eff / (C[-1] + K_eff)**2)
    fb_list.append(f_b)
    G_list.append(G[-1])
    G_pa.append(KG*(C_f-KD)/(KD-alpha*C_f))

G_pa = np.array(G_pa)


C_list = np.logspace(-2, 4, 50)
FI1 = []
fb1_list = []
G1_list = []
G1_pa = []
FI_pa = np.zeros_like(C_list)  # in case of perfect adaptation
KM1 = 0.1

for C_f in C_list:
    C = np.zeros(l)
    C[0:int(l/3)] = C_i
    C[int(l/3):] = C_f

    G_i = 1  # initial G-protein concentration

    G = [G_i]
    t = [0]

    for i in range(l):
        t.append(t[i]+dt)
        K_eff = KD * (G[i]+KG)/(alpha*G[i]+KG)
        f_b = C[i] / (C[i] + K_eff)

        dGdt = kc*(1-f_b)*(G_tot - G[i])/(KM1 +
                                          G_tot - G[i]) - kc*f_b*G[i]/(KM1 + G[i])

        G.append(G[i] + dt*dGdt)
    FI1.append(C[-1]*K_eff / (C[-1] + K_eff)**2)
    fb1_list.append(f_b)
    G1_list.append(G[-1])
    G1_pa.append(KG*(C_f-KD)/(KD-alpha*C_f))

G1_pa = np.array(G1_pa)

C_list = np.logspace(-2, 4, 50)
FI2 = []
fb2_list = []
G2_list = []
G2_pa = []
FI_pa = np.zeros_like(C_list)  # in case of perfect adaptation
KM2 = 0.01

for C_f in C_list:
    C = np.zeros(l)
    C[0:int(l/3)] = C_i
    C[int(l/3):] = C_f

    G_i = 1  # initial G-protein concentration

    G = [G_i]
    t = [0]

    for i in range(l):
        t.append(t[i]+dt)
        K_eff = KD * (G[i]+KG)/(alpha*G[i]+KG)
        f_b = C[i] / (C[i] + K_eff)

        dGdt = kc*(1-f_b)*(G_tot - G[i])/(KM2 +
                                          G_tot - G[i]) - kc*f_b*G[i]/(KM2 + G[i])

        G.append(G[i] + dt*dGdt)
    FI2.append(C[-1]*K_eff / (C[-1] + K_eff)**2)
    fb2_list.append(f_b)
    G2_list.append(G[-1])
    G2_pa.append(KG*(C_f-KD)/(KD-alpha*C_f))

G2_pa = np.array(G2_pa)

start, end = 0.2, 0.85
cmap = plt.cm.plasma
colors = cmap(np.linspace(start, end, 3))


# =======================================================================

plt.figure(figsize=(3.5, 3))

ax = plt.subplot(1, 1, 1)
ax.plot(C_list, fb1_list, '--', color=colors[2], lw=2, label=r"$10^{-1}$", zorder=2)
ax.plot(C_list, fb2_list, '--', color=colors[1], lw=2, label=r"$10^{-2}$", zorder=2)
ax.plot(C_list, fb_list, '--', color=colors[0], lw=2, label=r"$10^{-3}$", zorder=2)

ax.set_xscale("log")
ax.set_ylabel(r"Fraction bound $f_b$")
ax.set_xlabel(r"Ligand concentration $C$ [nM]")

ax.axhline(y=0.5, color="gray", linestyle='-', zorder=1, alpha=0.2, lw=1.5)

ax.legend(frameon=False, title=r"$K_M$ [nM]", title_fontsize=14)

ax.axvspan(KD/alpha, KD, facecolor='C2',  alpha=0.1, edgecolor='none', zorder=0)

ax.set_yticks(ticks = [0, 0.25, 0.5, 0.75, 1])

ax.set_xticks(ticks=[1e-2, 1, 1e2, 1e4])
ax.set_xticklabels([r"$10^{-2}$", r"$K_D/\alpha$", r"$K_D$", r"$10^4$"])
ax.set_ylim(0, 1)
ax.set_xlim(1e-2, 1e4)

plt.savefig("fb_C.png", dpi=150, bbox_inches='tight')


#=======================================================================

alpha = 100.

KG = 1
G_opt = np.logspace(-4, 4, 100)

KD = 100
Keff = KD * (G_opt+KG)/(alpha*G_opt+KG)

fig, ax = plt.subplots(figsize=(3.5, 3))

ax.set_yscale("log")
ax.set_xscale("log")
ax.axvspan(KD/alpha, KD, facecolor='C2', alpha=0.1, edgecolor='none', zorder=0)


ax.plot(C_list, G1_list, '--', lw=2, color=colors[2], label=r"$K_M = 10^{-1}\,\mathrm{nM}$", zorder=2)
ax.plot(C_list, G2_list, '--', lw=2, color=colors[1], label=r"$K_M = 10^{-2}\,\mathrm{nM}$", zorder=2)
ax.plot(C_list, G_list, '--', lw=2, color=colors[0], label=r"$K_M = 10^{-3}\,\mathrm{nM}$", zorder=2)


ax.plot(Keff, G_opt, '-', lw=2, label=r"$G_\mathrm{opt}$", color='C2', zorder=0)

ax.set_xlim(1e-2, 1e4)
ax.set_xticks(ticks=[1e-2, 1, 1e2, 1e4])
ax.set_xticklabels([r"$10^{-2}$", r"$K_D/\alpha$", r"$K_D$", r"$10^4$"])
ax.set_ylim(10**-3.5, 10**3.5)

ax.legend(frameon=False)

ax.set_xlabel(r"Ligand concentration $C$ [nM]")
ax.set_ylabel(r"Allosteric protein G [nM]")
plt.savefig("G_C.png", dpi=150, bbox_inches='tight')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import (AutoMinorLocator, LogLocator)


#===============================================
# panel a
#===============================================

data = np.load('raw_data1.npz')
D_list = data['dlist']
FI_time = data['fitime']
FI_end = data['fiend']


# Define the colormap and colors
start_c, end_c = 0.1, 0.9
cmap = plt.cm.viridis  # viridis  # or 'plasma'
colors = cmap(np.linspace(start_c, end_c, len(D_list)))
cmap_new = mcolors.LinearSegmentedColormap.from_list("truncated_greens", cmap(np.linspace(start_c, end_c, 256)))


max_steps = 1000 * (10**5)
dt = 10**-5
time = []  # np.arange(0, max_steps*dt, dt)
for i in range(max_steps):
    if i % 10000 == 0:
        time.append(i*dt)

time = np.array(time)

start = int(len(time) / 3)
vmax_list = 0.1 / 5**2 / D_list # actually V_max / K_G


alpha = 10
g = 0.05
KD = 100
n = 400000  # number of receptors

adap_time_fisher_alt = np.zeros_like(D_list)
for idx in range(len(D_list)):
    FIs = FI_time[idx]
    FIend = FI_end[idx]
    diff = np.abs(FIs[start:] - FIend)
    adap_time_fisher_alt[idx] = np.sum((time[start:] - time[start]) * diff)/np.sum(diff)

fig, ax = plt.subplots(1, 1, dpi=100, figsize=(3, 3))

C_f = 1.1 * KD/alpha
ax.plot(np.array(adap_time_fisher_alt/vmax_list)[:9], FI_end[:9], '-', color="C2", markerfacecolor="None",
        lw=2, markeredgewidth=2, zorder=2, clip_on=False)
for i in range(9):
    ax.plot(np.array(adap_time_fisher_alt/vmax_list)[i], FI_end[i], 'o', color=colors[i], markerfacecolor="None", markeredgewidth=2, zorder=2, clip_on=False)

ax.axhline(y=n * g**2 * C_f * KD / (C_f + KD)**2 / 8, linestyle='--',
           color='C3', lw=2, label="Single receptor type", zorder=1)
ax.axhline(y=n * g**2 * C_f * KD / (C_f + KD)**2 / 16 + n * g**2 * C_f * KD/alpha / (C_f +
           KD/alpha) ** 2 / 16, linestyle='-.', color='C0', lw=2, label="Two receptor types", zorder=1)
ax.set_xlabel(r"Adaptation time $\tau_A$ [s]")
ax.set_ylabel(r"Fisher Information $I_{\phi\phi}$")
ax.set_xscale('log')
ax.set_ylim(0, 34)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.set_title(r"$1.2K_D/\alpha\;{\to}\:1.1K_D/\alpha$", fontsize=14)

ax.xaxis.set_minor_locator(LogLocator(numticks=10, subs=[.2, .4, .6, .8]))
ax.set_xticks([1, 10, 1e2, 1e3, 1e4])
ax.set_xlim(2, 5e4)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(top=False, right=False, which='both')

plt.savefig('panel_a.png', dpi=150, bbox_inches='tight')


#===============================================
# panel b
#===============================================

data = np.load('raw_data2.npz')
D_list = data['dlist']
FI_time1 = data['fitime1']
FI_end1 = data['fiend1']

max_steps = 100 * 2 * (10**5) 
dt = 10**-5
time = [] # np.arange(0, max_steps*dt, dt)
for i in range(max_steps):
    if i % 2000 == 0:
        time.append(i*dt)

time = np.array(time)

start = int(len(time) / 3)
vmax_list = 0.1 / 5**2/ D_list # actually V_max / K_G


alpha = 10
g = 0.05
KD = 100
n = 400000  # number of receptors

adap_time_fisher1_alt = np.zeros_like(D_list)
for idx in range(len(D_list)):
    FIs = FI_time1[idx]
    FIend = FI_end1[idx]
    diff = np.abs(FIs[start:] - FIend)
    adap_time_fisher1_alt[idx] = np.sum((time[start:] - time[start]) * diff)/np.sum(diff)


fig, ax = plt.subplots(1, 1, dpi=100, figsize=(3, 3))

C_f1 = KD / np.sqrt(alpha)

ax.plot(adap_time_fisher1_alt/vmax_list, FI_end1, '-', color="C2", markerfacecolor="None",
        lw=2, markeredgewidth=2, label="Simulations", zorder=2, clip_on=False)

for idx in range(len(D_list)):
    ax.plot((adap_time_fisher1_alt/vmax_list)[idx], FI_end1[idx], 'o', color=colors[idx], markerfacecolor="None", markeredgewidth=2, zorder=2, clip_on=False)

ax.axhline(y=n * g**2 * C_f1 * KD / (C_f1 + KD)**2 / 8, linestyle='--',
           color='C3', lw=2, label="Single receptor type", zorder=1)
ax.axhline(y=n * g**2 * C_f1 * KD / (C_f1 + KD)**2 / 16 + n * g**2 * C_f1 * KD/alpha /
           (C_f1 + KD/alpha)**2 / 16, linestyle='-.', lw=2, color='C0', label="Two receptor types", zorder=1)
ax.set_xlabel(r"Adaptation time $\tau_A$ [s]")
ax.set_ylabel(r"Fisher Information $I_{\phi\phi}$")
ax.set_xscale('log')
ax.set_ylim(0, 34)
ax.set_xlim(right=5800)
ax.set_title(r"$0.8K_D/\sqrt{\alpha}\:{\to}\:K_D/\sqrt{\alpha}$", fontsize=14)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.xaxis.set_minor_locator(LogLocator(numticks=10, subs=[.2, .4, .6, .8]))
ax.set_xticks([1, 10, 100, 1e3, 1e4])
plt.legend(frameon=False, fontsize=14)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(top=False, right=False, which='both')
ax.set_xlim(2, 5e4)
plt.savefig('panel_b.png', dpi=150, bbox_inches='tight')


#===============================================
# panel c
#===============================================

data = np.load('raw_data3.npz')
D_list = data['dlist']
FI_time2 = data['fitime2']
FI_end2 = data['fiend2']

max_steps = 100 * 2 * (10**5)
dt = 10**-6
time = []  # np.arange(0, max_steps*dt, dt)
for i in range(max_steps):
    if i % 2000 == 0:
        time.append(i*dt)

time = np.array(time)

start = int(len(time) / 3)
vmax_list = 0.1 / 5**2 / D_list  # actually V_max / K_G


alpha = 10
g = 0.05
KD = 100
n = 400000  # number of receptors

adap_time_fisher2_alt = np.zeros_like(D_list)
for idx in range(len(D_list)):
    FIs = FI_time2[idx]
    FIend = FI_end2[idx]
    diff = np.abs(FIs[start:] - FIend)
    adap_time_fisher2_alt[idx] = np.sum((time[start:] - time[start]) * diff)/np.sum(diff)

fig, ax = plt.subplots(1, 1, dpi=100, figsize=(3, 3))

C_f = 0.8*KD

ax.plot(np.array(adap_time_fisher2_alt/vmax_list)[1:], FI_end2[1:], '-', color="C2", markerfacecolor="None",
        lw=2, markeredgewidth=2, zorder=2, clip_on=False)

for idx in range(1, len(D_list)):
    ax.plot(np.array(adap_time_fisher2_alt/vmax_list)[idx], FI_end2[idx], 'o', color=colors[idx], markerfacecolor="None",
            markeredgewidth=2, zorder=2, clip_on=False)

ax.axhline(y=n * g**2 * C_f * KD / (C_f + KD)**2 / 8, linestyle='--',
           color='C3', lw=2, label="Single receptor type", zorder=1)
ax.axhline(y=n * g**2 * C_f * KD / (C_f + KD)**2 / 16 + n * g**2 * C_f * KD/alpha / (C_f +
           KD/alpha)**2 / 16, linestyle='-.',  lw=2, color='C0', label="Two receptor types", zorder=1)
ax.set_xlabel(r"Adaptation time $\tau_A$ [s]")
ax.set_ylabel(r"Fisher Information $I_{\phi\phi}$")
ax.set_xscale('log')
ax.set_ylim(0, 34)
ax.set_title(r"$0.7K_D\:{\to}\;0.8K_D$", fontsize=14)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.xaxis.set_minor_locator(LogLocator(numticks=10, subs=[.2, .4, .6, .8]))
ax.set_xticks([0.1, 1, 10, 100, 1e3, 1e4])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(top=False, right=False, which='both')

ax.plot((adap_time_fisher2_alt/vmax_list)[8], FI_end2[8], 'o', color="C1", markerfacecolor="None", 
        markeredgewidth=2.5, zorder=3, clip_on=False)
ax.plot((adap_time_fisher2_alt/vmax_list)[9], FI_end2[9], 'o', color="C4", markerfacecolor="None",
        markeredgewidth=2.5, zorder=3, clip_on=False)

ax.text((adap_time_fisher2_alt/vmax_list)[8]+85, FI_end2[8]-0.5, "A", fontsize=14, color="C1", weight='bold')
ax.text((adap_time_fisher2_alt/vmax_list)[9]*0.4, FI_end2[9], "B", fontsize=14, color="C4", weight='bold')

ax.set_xlim(left=0.1)
ax.set_xlim(2, 5e4)
plt.savefig('panel_c.png', dpi=150, bbox_inches='tight')


#===============================================
# panel d
#===============================================

fig, ax = plt.subplots(figsize=(3, 3), dpi=100)

indices = [8, 9]    # point index
labels = ["A", "B"]
colors = ["C1", "C4"]
for idx in indices:
    FIs = FI_time2[idx]
    FIend = FI_end2[idx]

    diff = np.abs(FIs[start:] - FIend)
    tau_a = np.sum((time[start:] - time[start]) * diff)/np.sum(diff)
    tau_a /= vmax_list[idx]
    if idx == 8:
        ax.vlines(tau_a, 3.7, 5.5, color="C1", linestyle='--', zorder=3)
    else:
        ax.vlines(tau_a, 8.3, 10.5, color="C4", linestyle='--', zorder=3)

    ax.plot((time-time[start])/vmax_list[idx], FIs, zorder=1, label=labels[indices.index(idx)],
            color=colors[indices.index(idx)], lw=2)
    ax.axhline(y=FIend, color='C7', linestyle='--', zorder=2)

ax.axvline(x=0, color="C7", linestyle=':', zorder=0)
ax.set_ylim(3.5, 10.5)
ax.set_xlim(-100, 690)

ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_xlabel(r'Time $t$ [s]')
ax.set_ylabel(r'$I_{\phi\phi}$')
ax.legend(frameon=False, labelcolor=["C1", "C4"], prop={'weight': 'bold'})

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(top=False, right=False, which='both')

plt.savefig('panel_d.png', dpi=150, bbox_inches='tight')

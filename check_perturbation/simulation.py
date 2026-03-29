import numpy as np
import scipy.optimize as opt


max_steps = 400000
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
dt = 0.00005
dx = 2*np.pi*r/l
D = 100  # units of x^2/t
# D*dt/dx**2  # checking stability - this has to be < 0.5


#===========================================================
# High Diffusion
#===========================================================

print('High diffusion:')

D_list = np.logspace(-4, 2, 10)
kc = 1
kc_list = np.logspace(-2, 1, 7)
D = 50
alpha = 10
C_f = KD/alpha - 1
C_i = KD / np.sqrt(alpha)
C0 = np.zeros(max_steps + 1)

C_profile = C_f * np.exp(g/2 * np.cos(theta - phi))
C_list = np.logspace(0, 2.5, 12)
first_order_corr = []
first_order_disc = []
G_end = []
first_order_prob = []
p_bound = []
G_pred_list = []
K_eff_list = []
FI_calc = []
FI_high_pert = []
f_bound_list = []
f_bound_fit = []


def linear_prob_bound(theta, A, B):
    return A + B * np.cos(theta-phi)


def first_order_high_diff(theta, D):
    G_opt = KG * (KD - C_f) / (alpha*C_f - KD)
    G_pert = G_opt - g * np.cos(theta-phi) / (4 * D)
    return G_pert


for j, C_f in enumerate(C_list):
    C_profile = C_f * np.exp(g/2 * np.cos(theta - phi))
    G_inact = []
    G_inact_init = np.zeros(l)
    for i in range(l):
        G_inact_init[i] = 50
    G_inact.append(G_inact_init)
    G_active_init = KG * (KD - C_i) / (alpha*C_i - KD)

    G_act = G_active_init
    G_next_act = np.copy(G_active_init)
    G_inact = G_inact_init
    G_next_inact = np.copy(G_inact_init)

    for i in range(max_steps):
        K_eff = KD * (G_next_act + KG)/(alpha*G_next_act + KG)

        # adjusting to the local fraction bound (approx.)
        f_b = C_profile / (C_profile + K_eff)

        G_next_act = G_act + (dt*D/dx**2)*(np.roll(G_act, 1) - 2*G_act + np.roll(G_act, -1)) + \
            dt*(kc*(1-f_b)*(G_inact)/(KM + G_inact) -
                kc*f_b*G_next_act/(KM + G_act))
        G_act = G_next_act

        G_next_inact = G_inact + (dt*D/dx**2)*(np.roll(G_inact, 1) - 2*G_inact + np.roll(
            G_inact, -1)) + dt*(-kc*(1-f_b) * (G_inact)/(KM + G_inact) + kc*f_b*G_act/(KM + G_act))
        G_inact = G_next_inact

    # Calculating Fisher information from probability bound in first order
    p, cov = opt.curve_fit(linear_prob_bound, theta, f_b)
    A = p[0]
    B = p[1]
    FI_calc.append(n * B**2 / (2 * (A - A**2)))

    G_half = KG * (KD - C_profile) / (alpha*C_profile - KD)
    G_opt = KG * (KD - C_f) / (alpha*C_f - KD)
    G_end.append(G_next_act)

    f_bound_list.append(f_b)
    f_bound_fit.append(linear_prob_bound(theta, A, B))
    print(f'{j+1}/{len(C_list)}, C0 = {C_f:.4f}')


# ===========================================================
# Low Diffusion
# ===========================================================

print('\nLow diffusion:')

D = 0.1

first_order_corr = []
first_order_disc = []
G_end = []
first_order_prob = []
p_bound = []
FI_calc1 = []
FI_low_pert = []
FI_time = []

for j, C_f in enumerate(C_list):
    C_profile = C_f * np.exp(g/2 * np.cos(theta - phi))
    G_inact = []
    G_inact_init = np.zeros(l)
    for i in range(l):
        G_inact_init[i] = 50
    G_inact.append(G_inact_init)
    G_active_init = KG * (KD - C_i) / (alpha*C_i - KD)

    G_act = G_active_init
    G_next_act = np.copy(G_active_init)
    G_inact = G_inact_init
    G_next_inact = np.copy(G_inact_init)
    FI_time_j = []

    for i in range(max_steps):
        K_eff = KD * (G_next_act + KG)/(alpha*G_next_act + KG)

        # adjusting to the local fraction bound (approx.)
        f_b = C_profile / (C_profile + K_eff)

        G_next_act = G_act + (dt*D/dx**2)*(np.roll(G_act, 1) - 2*G_act + np.roll(G_act, -1)) + \
            dt*(kc*(1-f_b)*(G_inact)/(KM + G_inact) -
                kc*f_b*G_next_act/(KM + G_act))
        G_act = G_next_act

        G_next_inact = G_inact + (dt*D/dx**2)*(np.roll(G_inact, 1) - 2*G_inact + np.roll(
            G_inact, -1)) + dt*(-kc*(1-f_b) * (G_inact)/(KM + G_inact) + kc*f_b*G_act/(KM + G_act))
        G_inact = G_next_inact

        p, cov = opt.curve_fit(linear_prob_bound, theta, f_b)
        E = p[0]
        F = p[1]
        FI_time_j.append(n * F**2 / (2 * (E - E**2)))

    # Calculating Fisher information from probability bound in first order
    p, cov = opt.curve_fit(linear_prob_bound, theta, f_b)
    A = p[0]
    B = p[1]
    FI_calc1.append(n * B**2 / (2 * (A - A**2)))

    G_half = KG * (KD - C_profile) / (alpha*C_profile - KD)
    G_opt = KG * (KD - C_f) / (alpha*C_f - KD)
    G_end.append(G_next_act)

    FI_time.append(FI_time_j)

    print(f'{j+1}/{len(C_list)}, C0 = {C_f:.4f}')


np.save("FI_calc", np.vstack([FI_calc, FI_calc1]))
print("\nData saved.")

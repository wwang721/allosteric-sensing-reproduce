import numpy as np
import scipy.optimize as opt


def linear_prob_bound(theta, A, B):
    return A + B * np.cos(theta-phi)


def first_order_high_diff(theta, D):
    G_opt = KG * (KD - C_f) / (alpha*C_f - KD)
    G_pert = G_opt - g * np.cos(theta-phi) / (4 * D)
    return G_pert


max_steps = 400000

g = 0.05
phi = np.pi/6

kc = 1  # time in units of 1/kc
KD = 100
KG = 1
alpha = 10
KM = 0.0001  # Michaelis-Menten constant

# Initializing array
l = 40  # number of modeled G-protein sites
n = 400000  # number of receptors
theta = np.linspace(-np.pi, np.pi, l+1)
theta = theta[0:l]

r = 1  # position in units of r
dt = 0.00005
dx = 2*np.pi*r/l
D = 50

D_list = np.logspace(-1, 2,15)
kc_list = np.logspace(-2, 1, 10)
C_f = KD/alpha - 1
C_i = KD / np.sqrt(alpha)
C0 = np.zeros(max_steps + 1)
C_profile = C_f * np.exp(g/2 * np.cos(theta - phi))
C_list = np.logspace(0.5,2.5,20)

D_mesh, C_mesh = np.meshgrid(D_list, C_list)


performance_measure = np.zeros_like(C_mesh)
performance_measure1 = np.zeros_like(C_mesh)

FI = np.zeros_like(C_mesh)

FI_exp = []
FI_pred = []

for i in range(len(D_list)):
    D = D_list[i]
    for j,C_f in enumerate(C_list):
        C_profile = C_f * np.exp(g/2 * np.cos(theta - phi))
        G_end = []
        G_pred_list = []
        K_eff_list= []
        FI_calc = []

        C_profile = C_f * np.exp(g/2 * np.cos(theta - phi))
        G_inact = 50 * np.ones(l)
        G_act = np.abs(KG * (KD - C_i) / (alpha*C_i - KD)) * np.ones(l)

        for k in range(max_steps):
            K_eff = KD * (G_act + KG)/(alpha*G_act + KG)
            
            # adjusting to the local fraction bound (approx.)
            f_b = C_profile / (C_profile + K_eff) 
            f_u = 1 - f_b
            
            G_next_act = G_act + (dt*D/dx**2)*(np.roll(G_act,1) - 2*G_act + np.roll(G_act,-1)) + dt*(kc*f_u*(G_inact)/(KM + G_inact) - kc*f_b*G_act/(KM +G_act))
            G_act = G_next_act

            G_next_inact = G_inact + (dt*D/dx**2)*(np.roll(G_inact,1) - 2*G_inact + np.roll(G_inact,-1)) - dt*(kc*f_u*(G_inact)/(KM + G_inact) - kc*f_b*G_act/(KM +G_act))
            G_inact = G_next_inact
        
        # Calculating Fisher information from probability bound in first order
        p, cov = opt.curve_fit(linear_prob_bound, theta, f_b)
        A = p[0]; B = p[1]
        FI_calc = n * B**2 / (2 * (A - A**2))
        FI[j,i] = FI_calc
        FI_regular = n * g**2 * C_f * KD / (C_f + KD)**2 / 8
        FI_tworec = n * g**2 * C_f * KD / (C_f + KD)**2 / 16 + n * g**2 * C_f * KD/alpha / (C_f + KD/alpha)**2 / 16
        performance_measure[j,i] = FI_calc - FI_regular
        performance_measure1[j,i] = FI_calc - FI_tworec

        G_opt = KG * (KD - C_f) / (alpha*C_f - KD)
        G_end.append(G_next_act)

    print(f'{i+1}/{len(D_list)}, D = {D:.4f}')


np.save("data", np.vstack([performance_measure, performance_measure1]))
print("\nData saved.")

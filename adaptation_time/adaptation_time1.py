import numpy as np
import scipy.optimize as opt


def linear_prob_bound(theta, A, B):
    return A + B * np.cos(theta-phi)


max_steps = 1000 * (10**5) 
time = [0]

kc = 1 # time in units of 1/kc
KD = 100
KG = 1
alpha = 10
KM = 0.0001 # Michaelis-Menten constant

G_i = 1 # initial G-protein concentration

g = 0.05
phi = np.pi/6

# Initializing array
l = 40 # number of modeled G-protein sites
n = 400000 # number of receptors
theta = np.linspace(-np.pi, np.pi, l+1)
theta = theta[0:l]

# Diffusion terms
r = 1 # position in units of r
dt = 10**-5
dx = 2*np.pi*r/l
D = 100 # units of x^2/t
print(D*dt/dx**2) # checking stability - this has to be < 0.5

# Jump near KD/alpha
D_list = np.logspace(-2, 2,15)
C_i = 1.2*KD/alpha
C_f = 1.1 * KD/alpha #initial and final (mean) ligand concentrations

G_end = []
first_order_prob = []
p_bound = []
G_pred_list = []
K_eff_list= []
FI_calc = []
FI_end = []
FI_low_pert = []
FI_high_pert = []
f_bound_list = []
f_bound_fit = []
FI_time = []


for j,D in enumerate(D_list):
    C_profile = C_i * np.exp(g/2 * np.cos(theta - phi))
    G_inact = []
    G_inact = 50 * np.ones(l)
    G_act = np.abs(KG * (KD - C_i) / (alpha*C_i - KD)) * np.ones(l)
    FI_time_j = []

    for i in range(max_steps):
        if i >= int(max_steps/3):
            C_profile = C_f * np.exp(g/2 * np.cos(theta - phi))

        K_eff = KD * (G_act + KG)/(alpha*G_act + KG)
        
        # adjusting to the local fraction bound (approx.)
        f_b = C_profile / (C_profile + K_eff) 
        f_u = 1 - f_b
        
        G_next_act = G_act + (dt*D/dx**2)*(np.roll(G_act,1) - 2*G_act + np.roll(G_act,-1)) + dt*(kc*f_u*(G_inact)/(KM + G_inact) - kc*f_b*G_act/(KM +G_act))
        G_act = G_next_act

        G_next_inact = G_inact + (dt*D/dx**2)*(np.roll(G_inact,1) - 2*G_inact + np.roll(G_inact,-1)) - dt*(kc*f_u*(G_inact)/(KM + G_inact) - kc*f_b*G_act/(KM +G_act))
        G_inact = G_next_inact
        

        if i % 10000 == 0:
            p, cov = opt.curve_fit(linear_prob_bound, theta, f_b)
            E = p[0]; F = p[1]
            FI_time_j.append(n * F**2 / (2 * (E - E**2)))
        

    # Calculating Fisher information from probability bound in first order
    p, cov = opt.curve_fit(linear_prob_bound, theta, f_b)
    A = p[0]; B = p[1]
    FI_end.append(n * B**2 / (2 * (A - A**2)))
    
    G_end.append(G_next_act)
    FI_time.append(FI_time_j)
    f_bound_list.append(f_b)
    f_bound_fit.append(linear_prob_bound(theta, A, B))
    print(f'{j+1}/{len(D_list)}, D = {D:.4f}. Fisher Information at end = {FI_end[j]}')


np.savez("raw_data1.npz", dlist=D_list, fitime=FI_time, fiend=FI_end)

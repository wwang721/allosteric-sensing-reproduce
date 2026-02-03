import numpy as np
import scipy.optimize as opt


def linear_prob_bound(theta, A, B):
    return A + B * np.cos(theta-phi)


def first_order_high_diff(theta, D):
    G_opt = KG * (KD - C_f) / (alpha*C_f - KD)
    G_pert = G_opt - g * np.cos(theta-phi) / (4 * D)
    return G_pert


max_steps = 400000
time = [0]

# reaction terms
C_i = 110
C_f = 111 #initial and final (mean) ligand concentrations
C0 = np.zeros(max_steps + 1)
C0[0:int(max_steps/3)] = C_i
C0[int(max_steps/3):] = C_f

G_i = 1 # initial G-protein concentration

g = 0.05
phi = np.pi/6

kc = 1 # time in units of 1/kc
KD = 100
KG = 1
alpha = 10
KM = 0.0001 # Michaelis-Menten constant


# Initializing array
l = 40 # number of modeled G-protein sites
n = 400000 # number of receptors
theta = np.linspace(-np.pi, np.pi, l+1)
theta = theta[0:l]

# Diffusion terms
r = 1 # position in units of r
dt = 0.00005
dx = 2*np.pi*r/l


#====================================
# D = 100
#====================================

print('D=100:')

D = 100 # units of x^2/t
# D*dt/dx**2 # checking stability - this has to be < 0.5
kc = 1
alpha_list = np.logspace(0, 3, 15)
C_list = np.logspace(-1,3,15)

alpha_mesh, C_mesh = np.meshgrid(alpha_list, C_list)
FI = np.zeros_like(alpha_mesh)


for i in range(len(alpha_list)):
    alpha = alpha_list[i]
    for j,C_f in enumerate(C_list):
        C_profile = C_f * np.exp(g/2 * np.cos(theta - phi))
        G_end = []
        G_pred_list = []
        K_eff_list= []
        FI_calc = []
        G_inact = []
        G_inact_init = 50 * np.ones(l)
        G_inact.append(G_inact_init)
        G_act = np.abs(KG * (KD - C_i) / (alpha*C_i - KD)) * np.ones(l)
        G_next_act= np.copy(G_act)
        G_inact = G_inact_init
        G_next_inact = np.copy(G_inact_init)

        for k in range(max_steps):
            K_eff = KD * (G_next_act + KG)/(alpha*G_next_act + KG)
            
            # adjusting to the local fraction bound (approx.)
            f_b = C_profile / (C_profile + K_eff) 
            
            G_next_act = G_act + (dt*D/dx**2)*(np.roll(G_act,1) - 2*G_act + np.roll(G_act,-1)) + dt*(kc*(1-f_b)*(G_inact)/(KM + G_inact) - kc*f_b*G_act/(KM +G_act))
            G_act = G_next_act

            G_next_inact = G_inact + (dt*D/dx**2)*(np.roll(G_inact,1) - 2*G_inact + np.roll(G_inact,-1)) + dt*(-kc*(1-f_b)* (G_inact)/(KM + G_inact) + kc*f_b*G_act/(KM +G_act))
            G_inact = G_next_inact
        
        # Calculating Fisher information from probability bound in first order
        p, cov = opt.curve_fit(linear_prob_bound, theta, f_b)
        A = p[0]; B = p[1]
        FI_calc = n * B**2 / (2 * (A - A**2))
        FI[j,i] = FI_calc 


        G_half = KG * (KD - C_profile) / (alpha*C_profile - KD)
        G_opt = KG * (KD - C_f) / np.maximum((alpha*C_f - KD), 1e-16)
        G_end.append(G_next_act)

    print(f'{i+1}/{len(alpha_list)}, alpha = {alpha:.4f}')


#====================================
# D = 10
#====================================

print('\nD=10:')

FI1 = np.zeros_like(alpha_mesh)
D1 = 10

for i in range(len(alpha_list)):
    alpha = alpha_list[i]
    for j,C_f in enumerate(C_list):
        C_profile = C_f * np.exp(g/2 * np.cos(theta - phi))
        G_end = []
        G_pred_list = []
        K_eff_list= []
        FI_calc = []
        G_inact = 50 * np.ones(l)
        G_act = np.abs(KG * (KD - C_i) / (alpha*C_i - KD) * np.ones(l))
        
        G_next_act= np.copy(G_act)
        G_next_inact = np.copy(G_inact_init)

        for k in range(max_steps):
            K_eff = KD * (G_next_act + KG)/(alpha*G_next_act + KG)
            
            # adjusting to the local fraction bound (approx.)
            f_b = C_profile / (C_profile + K_eff) 
            
            G_next_act = G_act + (dt*D1/dx**2)*(np.roll(G_act,1) - 2*G_act + np.roll(G_act,-1)) + dt*(kc*(1-f_b)*(G_inact)/(KM + G_inact) - kc*f_b*G_act/(KM +G_act))
            G_act = G_next_act

            G_next_inact = G_inact + (dt*D1/dx**2)*(np.roll(G_inact,1) - 2*G_inact + np.roll(G_inact,-1)) + dt*(-kc*(1-f_b)* (G_inact)/(KM + G_inact) + kc*f_b*G_act/(KM +G_act))
            G_inact = G_next_inact
        
        # Calculating Fisher information from probability bound in first order
        p, cov = opt.curve_fit(linear_prob_bound, theta, f_b)
        A = p[0]; B = p[1]
        FI_calc = n * B**2 / (2 * (A - A**2))
        FI1[j,i] = FI_calc 


        G_half = KG * (KD - C_profile) / (alpha*C_profile - KD)
        G_opt = KG * (KD - C_f) / np.maximum((alpha*C_f - KD), 1e-16)
        G_end.append(G_next_act)

    print(f'{i+1}/{len(alpha_list)}, alpha = {alpha:.4f}')


np.save("FI", np.vstack([FI, FI1]))
print("\nData saved.")

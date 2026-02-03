import numpy as np
import scipy.optimize as opt


def linear_prob_bound(theta, A, B):
    return A + B * np.cos(theta-phi)


max_steps = 400000
time = [0]

g = 0.05
phi = np.pi/6
kc = 1  # time in units of 1/kc
KD = 100
KG = 1
alpha = 10
# Initializing array
l = 25  # number of modeled G-protein sites
n = 400000  # number of receptors
theta = np.linspace(-np.pi, np.pi, l+1)
theta = theta[0:l]
# Diffusion terms
r = 1  # position in units of r
dt = 0.00001
dx = 2*np.pi*r/l
D = 10  # unitless
# D*dt/dx**2  # checking stability - this has to be < 0.5

C_listA = np.logspace(0,1.5,8)
C_listB = np.logspace(1.5, 2.5, 20)
C_list = []
for C in C_listA:
    C_list.append(C)
for C in C_listB:
    C_list.append(C)
C_list = np.array(C_list)
KM_list = np.logspace(-4,-2, 3)
KM_mesh, C_mesh = np.meshgrid(KM_list, C_list)
FI = np.zeros_like(KM_mesh)


FI_exp = []
FI_pred = []
G_end = []

for i in range(len(KM_list)):
    KM = KM_list[i]
    for j,C_f in enumerate(C_list):
        C_profile = C_f * np.exp(g/2 * np.cos(theta - phi))
        G_pred_list = []
        K_eff_list= []
        FI_calc = []
        G_inact = []
        G_inact = 10 * np.ones(l)
        G_act = np.abs(KG * (KD - C_f) / (alpha*C_f - KD)) * np.ones(l)

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

        G_half = KG * (KD - C_profile) / (alpha*C_profile - KD)
        G_opt = KG * (KD - C_f) / (alpha*C_f - KD)

        if i == 0:
            G_end.append(G_act)
        
    print(f'{i+1}/{len(KM_list)}, KM = {KM:.4f}')


np.save("FI", FI)
print("\nData saved.")

import time

from util import diag_rescale_generalized

from plot import *
from calc import *
from sampling import *

BO = True
M = 1836.153

# Basis set maximum powers (rAB^h * r12^k * s^n * t^m * (mu1^i*mu2^j + mu1^j*mu2^i) * exp( - alpha*s - beta*rAB - gamma*r12 )
h_max = 5
k_max = 8
n_max = 8
m_max = 8
ij_max = 8
total_max = 5
delta = 0.1

# Todo: allow cusp-enabling functions - maybe s^-1 would help? Go back to s1, s2? (^-1, or more ideally ln)?

# delta = 0.1: E[0] := -1.174474883468479:
# E[0] := -1.1744788198234721 at delta=0.1, alpha=0.695

plot_phi_target = np.pi/2*0.1
plot_rAB_target = 1.4
plot_s2_target = 8

nMu = 24
nS = 50
sMax = 50

if BO:
    M1M = 1
    M_inv = 0
else:
    M1M = (M + 1) / M
    M_inv = 1 / M

if BO:  # Reset if not used
    h_max = 0  # avoid rAB dependence
    beta = 0

rows = []
for h in frange(0, h_max, 1):
    for k in frange(0, k_max, 1):
        for n in frange(0, n_max, 1):
            for m in range(m_max + 1):  # careful: Whenever using negative indices, adjust power_table call accordingly.
                for i in range(ij_max + 1):
                    for j in range(i + 1):

                        # constraints
                        if (i + j) % 2 != 0: continue
                        if m % 2 != 0: continue
                        t = h + k + n + m + i + j
                        if t > total_max: continue
                        rows.append((h, k, n, m, i, j))  #-k-m

basis_idx = np.array(rows, dtype=np.int16)

# Shorthands for later use (no copies)
h_idx = basis_idx[:, 0]
k_idx = basis_idx[:, 1]
n_idx = basis_idx[:, 2]
m_idx = basis_idx[:, 3]
i_idx = basis_idx[:, 4]
j_idx = basis_idx[:, 5]

Fij, Fji = calc_Fij_stmu(h_idx, k_idx, n_idx, m_idx, i_idx, j_idx)

matSize = h_idx.size
print(f"MatSize = {matSize}")

start = time.time()

nrP = 0
abRange = frange(1.4, 1.4, 0.2)

S_layers = {}
H_1_layers = {}
H_alpha_layers = {}
H_alpha_2_layers = {}
H_beta_layers = {}
H_beta_2_layers = {}
H_alpha_beta_layers = {}

plot_chunks = init_plot_chunks()

for rAB in abRange:
    s_shells, sW = build_s_shells(rAB, Ks=nS, s_max=sMax, gamma = 3.0)
    idx = np.abs(s_shells - plot_s2_target - plot_rAB_target).argmin()
    plot_s2_target = s_shells[idx] - plot_rAB_target - 0.0001  # Ensure small s1 values are included in the plot-sampling
    print(f"s2 = {plot_s2_target}")

    for ks, s in enumerate(s_shells):

        s1_vals, s2_vals, splitW = split_s(s, rAB, Ku=10)

        Sl = np.zeros((matSize, matSize), dtype=np.float64)
        Hl_1 = np.zeros((matSize, matSize), dtype=np.float64)
        Hl_alpha = np.zeros((matSize, matSize), dtype=np.float64)
        Hl_alpha2 = np.zeros((matSize, matSize), dtype=np.float64)
        Hl_alphabeta = np.zeros((matSize, matSize), dtype=np.float64)
        Hl_beta = np.zeros((matSize, matSize), dtype=np.float64)
        Hl_beta2 = np.zeros((matSize, matSize), dtype=np.float64)

        if s-rAB > plot_s2_target:
            s2_vals = np.append(s2_vals, plot_s2_target)
            s1_vals = np.append(s1_vals, s - plot_s2_target)
            splitW = np.append(splitW, 0.0)

        for j, (s1, s2) in enumerate(zip(s1_vals, s2_vals)):

            # sample electron 1 on its s1-shell
            x1, y1, _, mu1, _, w1 = sample_s_shell(rAB, s1, Nphi=2, nMu=nMu)  # x-y plane only
            rA1 = np.sqrt((x1 + rAB/2) ** 2 + y1 ** 2)
            rB1 = np.sqrt((x1 - rAB/2) ** 2 + y1 ** 2)  # vectorized distances
            mu1_p = power_table(mu1, ij_max)

            # sample electron 2 on its s2-shell
            x2, y2, z2, mu2, phi2, w2 = sample_s_shell(rAB, s2, octant=True, nMu=nMu)
            rA2 = np.sqrt((x2 + rAB/2)**2 + y2**2 + z2**2)
            rB2 = np.sqrt((x2 - rAB/2)**2 + y2**2 + z2**2)
            mu2_p = power_table(mu2, ij_max)

            # r12-dependent quantities
            dx = x1[:, None] - x2[None, :]
            dy = y1[:, None] - y2[None, :]
            dz = 0.0 - z2[None, :]
            r12 = np.sqrt(dx * dx + dy * dy + dz * dz).ravel()  # vector of r12 values for all e1, e2 positions
            r12 = np.maximum(r12, 10 ** (-8))
            r12_p = power_table(r12, k_max)

            P1 = rA1.size
            P2 = rA2.size
            P = P1 * P2

            H_1_ij, H_1_ji, H_alpha_ij, H_alpha_ji, H_alpha2, H_alphabeta, H_beta_ij, H_beta_ji, H_beta2 = calc_H_alphabeta_stmu(Fij, Fji, rAB, rA1, rB1, rA2, rB2, r12, M_inv, M1M, s1, s2, s, mu1, mu2, delta)

            # Tile/repeat single-electron arrays to match the entire sample point array
            mu1_p = np.repeat(mu1_p, P2, axis=0)  # shape (P, Npow)
            mu2_p = np.tile(mu2_p, (P1, 1))
            # scalar
            t = (s1 - s2) / rAB * 0.5

            # Assemble basis functions from power matrices
            B = np.ones((r12_p.shape[0], matSize), dtype=np.float64) * (rAB**h_idx*s**n_idx*t**m_idx)[None, :]
            B *= r12_p[:, k_idx]
            B *= np.exp(-delta*r12)[:, None]

            mu_part_ij = mu1_p[:, i_idx] * mu2_p[:, j_idx]
            mu_part_ji = mu1_p[:, j_idx] * mu2_p[:, i_idx]

            if abs(s2-plot_s2_target) < 1e-8:
                plot_chunks = update_plot_chunks(plot_chunks, mu1, mu2, B, mu_part_ij, mu_part_ji, P1, P2, x1, y1, phi2, plot_phi_target, H_1_ij, H_1_ji, H_alpha_ij, H_alpha_ji, H_alpha2, H_alphabeta, H_beta_ij, H_beta_ji, H_beta2)

            # weights
            sqrtW = np.sqrt((sW[ks] * splitW[j]) * (w1[:, None] * w2[None, :]).ravel())   # (P,)
            B *= sqrtW[:, None]

            B_ij = B * mu_part_ij
            B_ji = B * mu_part_ji

            B = B_ij + B_ji

            Sl += B.T @ B
            Hl_1 += B.T @ (H_1_ij * B_ij + H_1_ji * B_ji)
            Hl_alpha += B.T @ (H_alpha_ij * B_ij + H_alpha_ji * B_ji)
            Hl_beta += B.T @ (H_beta_ij * B_ij + H_beta_ji * B_ji)
            Hl_alpha2 += B.T @ (H_alpha2 * (B_ij + B_ji))
            Hl_alphabeta += B.T @ (H_alphabeta * (B_ij + B_ji))
            Hl_beta2 += B.T @ (H_beta2 * (B_ij + B_ji))

            nrP += P

        S_layers[rAB, s] = Sl
        H_1_layers[rAB, s] = Hl_1
        H_alpha_layers[rAB, s] = Hl_alpha
        H_alpha_2_layers[rAB, s] = Hl_alpha2
        H_alpha_beta_layers[rAB, s] = Hl_alphabeta
        H_beta_layers[rAB, s] = Hl_beta
        H_beta_2_layers[rAB, s] = Hl_beta2

        print(ks, rAB, s, nrP, time.time() - start)


plot_mu2_with_alpha_beta(
    plot_chunks = plot_chunks,

    S_layers=S_layers,
    H_1_layers=H_1_layers,
    H_alpha_layers=H_alpha_layers,
    H_beta_layers=H_beta_layers,
    H_alpha_2_layers=H_alpha_2_layers,
    H_beta_2_layers=H_beta_2_layers,
    H_alpha_beta_layers=H_alpha_beta_layers,

    matSize=matSize,
    diag_rescale_generalized=diag_rescale_generalized,

    plot_rAB_target=plot_rAB_target,
    plot_s2_target=plot_s2_target,
    plot_phi_target=plot_phi_target,

    alpha_values=np.arange(0.6, 1.3, 0.005),
    beta_values=np.array([0.0]),

    zlim_eloc=(-3, 0),
    subsample=4000,
)
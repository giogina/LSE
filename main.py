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

plot_mu2_chunks = []
plot_x1_chunks = []
plot_y1_chunks = []
plot_B_chunks  = []
plot_chunks_A_1 = []
plot_chunks_A_alpha = []
plot_chunks_A_beta = []
plot_chunks_A_alpha2 = []
plot_chunks_A_alphabeta = []
plot_chunks_A_beta2 = []

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

            P1 = rA1.size
            P2 = rA2.size
            P = P1 * P2

            # r12-dependent quantities
            dx = x1[:, None] - x2[None, :]
            dy = y1[:, None] - y2[None, :]
            dz = 0.0 - z2[None, :]
            r12 = np.sqrt(dx * dx + dy * dy + dz * dz).ravel()  # vector of r12 values for all e1, e2 positions
            r12 = np.maximum(r12, 10 ** (-8))
            r12_p = power_table(r12, k_max)

            H_1_ij, H_1_ji, H_alpha_ij, H_alpha_ji, H_alpha2, H_alphabeta, H_beta_ij, H_beta_ji, H_beta2 = calc_H_alphabeta_stmu(Fij, Fji, rAB, rA1, rB1, rA2, rB2, r12, M_inv, M1M, s1, s2, s, mu1, mu2, delta)

            mu1_p = np.repeat(mu1_p, P2, axis=0)  # shape (P, Npow)
            mu2_p = np.tile(mu2_p, (P1, 1))

            # weights
            W0 = sW[ks] * splitW[j]  # scalar
            Wpair = W0 * (w1[:, None] * w2[None, :]).ravel()  # (P,)
            sqrtW = np.sqrt(Wpair)  # (P,)

            # Assemble basis functions from power matrices
            B = np.ones((r12_p.shape[0], matSize), dtype=np.float64) * (rAB**h_idx*s**n_idx*t**m_idx)[None, :]
            B *= r12_p[:, k_idx]
            B *= np.exp(-delta*r12)[:, None]

            mu_part_ij = mu1_p[:, i_idx] * mu2_p[:, j_idx]
            mu_part_ji = mu1_p[:, j_idx] * mu2_p[:, i_idx]

            if abs(s2-plot_s2_target) < 1e-8:
                mask_e2, phi_used = mask_closest_phi(phi2, plot_phi_target)
                mask_pair = np.tile(mask_e2, P1)
                x1_sel = np.repeat(x1, P2)[mask_pair]
                y1_sel = np.repeat(y1, P2)[mask_pair]
                mu1 = np.repeat(mu1, P2)
                mu2 = np.tile(mu2, P1)
                mu2_sel = mu2[mask_pair]
                B_ij = (B * mu_part_ij)
                B_ji = (B * mu_part_ji)
                B_plot = (B_ij + B_ji)[mask_pair, :]
                plot_mu2_chunks.append(mu2_sel)
                plot_x1_chunks.append(x1_sel)
                plot_y1_chunks.append(y1_sel)
                plot_B_chunks.append(B_plot)

                A_1_full = H_1_ij * B_ij + H_1_ji * B_ji
                A_alpha_full = H_alpha_ij * B_ij + H_alpha_ji * B_ji
                A_beta_full = H_beta_ij * B_ij + H_beta_ji * B_ji
                Bs = (B_ij + B_ji)
                A_alpha2_full = H_alpha2 * Bs
                A_beta2_full = H_beta2 * Bs
                A_ab_full = H_alphabeta * Bs
                A_1_sel       = A_1_full[mask_pair, :]
                A_alpha_sel   = A_alpha_full[mask_pair, :]
                A_beta_sel    = A_beta_full[mask_pair, :]
                A_alpha2_sel  = A_alpha2_full[mask_pair, :]
                A_beta2_sel   = A_beta2_full[mask_pair, :]
                A_ab_sel      = A_ab_full[mask_pair, :]
                assert B_plot.shape == A_alpha_sel.shape
                plot_chunks_A_1.append(A_1_sel)
                plot_chunks_A_alpha.append(A_alpha_sel)
                plot_chunks_A_beta.append(A_beta_sel)
                plot_chunks_A_alpha2.append(A_alpha2_sel)
                plot_chunks_A_alphabeta.append(A_ab_sel)
                plot_chunks_A_beta2.append(A_beta2_sel)

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

mu2_all = np.concatenate(plot_mu2_chunks)
x1_all = np.concatenate(plot_x1_chunks)
y1_all = np.concatenate(plot_y1_chunks)
B_plot_all = np.vstack(plot_B_chunks)
A_1_all = np.vstack(plot_chunks_A_1)
A_alpha_all = np.vstack(plot_chunks_A_alpha)
A_beta_all = np.vstack(plot_chunks_A_beta)
A_alpha2_all = np.vstack(plot_chunks_A_alpha2)
A_alphabeta_all = np.vstack(plot_chunks_A_alphabeta)
A_beta2_all = np.vstack(plot_chunks_A_beta2)

plot_mu2_with_alpha_beta(
    x1_all=x1_all, y1_all=y1_all, mu2_all=mu2_all,
    B_plot_all=B_plot_all,
    A_1_all=A_1_all, A_alpha_all=A_alpha_all, A_beta_all=A_beta_all,
    A_alpha2_all=A_alpha2_all, A_alphabeta_all=A_alphabeta_all, A_beta2_all=A_beta2_all,

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
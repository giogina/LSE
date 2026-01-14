import time
from scipy.sparse import csr_matrix

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
m_max = 8 # stmu only
ij_max = 8 # stmu only
ab_max = 1 # rij only
total_max = 4
delta = 0.1
nm_min = -2 # s12mu only

# delta = 0.1: E[0] := -1.174474883468479:
# E[0] := -1.1744788198234721 at delta=0.1, alpha=0.695

plot_phi_target = np.pi/2*0.1
plot_rAB_target = 1.4
plot_s2_target = 2

nMu = 24
nS = 40
sMax = 50

# coords = "rij"
# coords = "stmu"
coords = "s12mu"

if BO:
    M1M = 1
    M_inv = 0
else:
    M1M = (M + 1) / M
    M_inv = 1 / M

if BO:  # Reset if not used
    h_max = 0  # avoid rAB dependence
    beta = 0

X = None
amin = 0  # for power_table
amax = 0
# (rA1^n*rB1^m*rA2^i*rB2^j*r12^k*rAB^h)/sqrt(rA1^2+rA2^2)^a/sqrt(rB1^2+rB2^2)^b*exp(-alpha*(rA1+rA2+rB1+rB2)-beta*rAB-delta*r12);
# (1-2 swap: n-i, m-j. A-B swap: n-m, i-j, a-b)
# (n, m, i, j, a, b)
# (i, j, n, m, a, b)
# (m, n, j, i, b, a)
# (j, i, m, n, b, a)
# Unique: n>m, n>=i, n>=j
# or n=m, n>=i, i>j
# or n=m=i=j, a>=b

attime = 0
htime = 0
btime = 0
atime = 0
wtime = 0
ltime = 0

rows = []
if coords == "rij":
    sym_b = {}
    for h in frange(0, h_max, 1):
        for k in frange(0, k_max, 1):
            for n in frange(0, n_max, 1):
                for m in frange(0, n_max, 1):  # careful: Whenever using negative indices, adjust power_table call accordingly.
                    for i in frange(0, n_max, 1):
                        for j in frange(0, n_max, 1):
                            for ah in frange(0, ab_max, 1):
                                for bh in frange(0, ab_max, 1):

                                    a = ah
                                    b = bh
                                    # constraints
                                    t = h + k + n + m + i + j
                                    if t > total_max: continue
                                    if a>1 and abs(n-i)>1: continue # prevent repeating basis functions by cancellation of rA, rB terms
                                    if b>1 and abs(m-j)>1: continue
                                    if a>n+i: continue
                                    if b>m+j: continue

                                    # a += np.floor(k/2)
                                    # b += np.floor(k/2)

                                    indexTuples = [(h, k, n, m, i, j, a, b), (h, k, i, j, n, m, a, b), (h, k, m, n, j, i, b, a), (h, k, j, i, m, n, b, a)]
                                    indexTuples = tuple(sorted(set(indexTuples)))
                                    rep = indexTuples[0]  # representative for all symmetry-equivalent tuples
                                    if rep in sym_b:
                                        sym_b[rep].append(len(rows))
                                    else:
                                        sym_b[rep] = [len(rows)]
                                    rows.append((h, k, n, m, i, j, a, b))
                                    amin = min(a, amin)
                                    amax = max(a, amax)
    # X = np.zeros((len(rows), len(sym_b.keys())), dtype=int)
    # for col, l in enumerate(sym_b.values()):
    #     for idx in l:
    #         X[idx, col] = 1

    row_idx = []
    col_idx = []
    for col, l in enumerate(sym_b.values()):
        for idx in l:
            row_idx.append(idx)
            col_idx.append(col)
    data = np.ones(len(row_idx), dtype=np.int8)
    X = csr_matrix((data, (row_idx, col_idx)), shape=(len(rows), len(sym_b)))
    # groups = [np.asarray(l, dtype=np.int32) for l in sym_b.values()]
    amin = int(amin)
    amax = int(amax)
    print(amin, amax)

elif coords == "stmu" or coords == "s12mu":
    for h in frange(0, h_max, 1):
        for k in frange(0, k_max, 1):
            for n in frange(nm_min, n_max, 1):
                for m in frange(nm_min, n_max, 1):  # careful: Whenever using negative indices, adjust power_table call accordingly.
                    for i in range(ij_max + 1):
                        for j in range(i + 1):
                            # constraints
                            if (i + j) % 2 != 0: continue  # A<->B symmetry
                            if coords == "stmu":
                                if m % 2 != 0: continue
                            elif coords == "s12mu":
                                if i == j and m>n: continue # avoid duplication of (n, m, i, j=i) and (m, n, j=i, i)
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
if coords == "rij":
    a_idx = basis_idx[:, 6]
    b_idx = basis_idx[:, 7]

matSize = h_idx.size
bSize = len(sym_b) if coords == "rij" else matSize
print(f"MatSize = {bSize}, Coords: {coords}")

if coords == "stmu":
    Fij, Fji = calc_Fij_stmu(h_idx, k_idx, n_idx, m_idx, i_idx, j_idx)
elif coords == "s12mu":
    Fij, Fji = calc_Fij_s12mu(h_idx, k_idx, n_idx, m_idx, i_idx, j_idx)
elif coords == "rij":
    F = calc_F_rij(h_idx, k_idx, n_idx, m_idx, i_idx, j_idx, a_idx, b_idx)

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

        Sl = np.zeros((bSize, bSize), dtype=np.float64)
        Hl_1 = np.zeros((bSize, bSize), dtype=np.float64)
        Hl_alpha = np.zeros((bSize, bSize), dtype=np.float64)
        Hl_alpha2 = np.zeros((bSize, bSize), dtype=np.float64)
        Hl_alphabeta = np.zeros((bSize, bSize), dtype=np.float64)
        Hl_beta = np.zeros((bSize, bSize), dtype=np.float64)
        Hl_beta2 = np.zeros((bSize, bSize), dtype=np.float64)

        if s-rAB > plot_s2_target:
            s2_vals = np.append(s2_vals, plot_s2_target)
            s1_vals = np.append(s1_vals, s - plot_s2_target)
            splitW = np.append(splitW, 0.0)

        for j, (s1, s2) in enumerate(zip(s1_vals, s2_vals)):

            # sample electron 1 on its s1-shell
            x1, y1, _, mu1, _, w1 = sample_s_shell(rAB, s1, Nphi=2, nMu=nMu)  # x-y plane only
            rA1 = np.sqrt((x1 + rAB/2) ** 2 + y1 ** 2)
            rB1 = np.sqrt((x1 - rAB/2) ** 2 + y1 ** 2)  # vectorized distances

            # sample electron 2 on its s2-shell
            x2, y2, z2, mu2, phi2, w2 = sample_s_shell(rAB, s2, octant=True, nMu=nMu)
            rA2 = np.sqrt((x2 + rAB/2)**2 + y2**2 + z2**2)
            rB2 = np.sqrt((x2 - rAB/2)**2 + y2**2 + z2**2)

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

            if coords == "stmu":
                H_1_ij, H_1_ji, H_alpha_ij, H_alpha_ji, H_alpha2, H_alphabeta, H_beta_ij, H_beta_ji, H_beta2 = calc_H_alphabeta_stmu(Fij, Fji, rAB, rA1, rB1, rA2, rB2, r12, M_inv, M1M, s1, s2, s, mu1, mu2, delta)
                mu1_p = power_table(mu1, ij_max)
                mu2_p = power_table(mu2, ij_max)
                mu1_p = np.repeat(mu1_p, P2, axis=0)  # shape (P, Npow)  # Tile/repeat single-electron arrays to match the entire sample point array
                mu2_p = np.tile(mu2_p, (P1, 1))
                t = (s1 - s2) / rAB * 0.5

                # Assemble basis functions from power matrices
                B = np.ones((r12_p.shape[0], matSize), dtype=np.float64) * (rAB ** h_idx * s ** n_idx * t ** m_idx)[None, :]
                B *= r12_p[:, k_idx]
                B *= np.exp(-delta*r12)[:, None]

                mu_part_ij = mu1_p[:, i_idx] * mu2_p[:, j_idx]
                mu_part_ji = mu1_p[:, j_idx] * mu2_p[:, i_idx]

                B_ij = B * mu_part_ij
                B_ji = B * mu_part_ji

                B = B_ij + B_ji

                A_1 = H_1_ij * B_ij + H_1_ji * B_ji
                A_alpha = H_alpha_ij * B_ij + H_alpha_ji * B_ji
                A_beta = H_beta_ij * B_ij + H_beta_ji * B_ji
                A_alpha2 = H_alpha2 * (B_ij + B_ji)
                A_alphabeta = H_alphabeta * (B_ij + B_ji)
                A_beta2 = H_beta2 * (B_ij + B_ji)

            elif coords == "s12mu":
                now = time.time()
                H_1_12, H_1_21, H_alpha_12, H_alpha_21, H_alpha2, H_alphabeta, H_beta_12, H_beta_21, H_beta2 = calc_H_alphabeta_s12mu(Fij, Fji, rAB, rA1, rB1, rA2, rB2, r12, M_inv, M1M, s1, s2, mu1, mu2, delta)
                htime += (time.time() - now)

                now = time.time()
                mu1_p = power_table(mu1, ij_max)
                mu2_p = power_table(mu2, ij_max)
                mu1_p = np.repeat(mu1_p, P2, axis=0)  # shape (P, Npow)  # Tile/repeat single-electron arrays to match the entire sample point array
                mu2_p = np.tile(mu2_p, (P1, 1))

                # Assemble basis functions from power matrices
                B = np.broadcast_to(rAB**h_idx, (P, matSize)).copy()
                B *= r12_p[:, k_idx]
                B *= np.exp(-delta*r12)[:, None]

                part_12 = mu1_p[:, i_idx] * mu2_p[:, j_idx] * s1 ** n_idx * s2 ** m_idx
                part_21 = mu1_p[:, j_idx] * mu2_p[:, i_idx] * s1 ** m_idx * s2 ** n_idx

                B_12 = B * part_12
                B_21 = B * part_21

                B = B_12 + B_21
                btime += (time.time() - now)

                now = time.time()
                A_1 = H_1_12 * B_12 + H_1_21 * B_21
                A_alpha = H_alpha_12 * B_12 + H_alpha_21 * B_21
                A_beta = H_beta_12 * B_12 + H_beta_21 * B_21
                A_alpha2 = H_alpha2 * B
                A_alphabeta = H_alphabeta * B
                A_beta2 = H_beta2 * B
                atime += (time.time() - now)

            elif coords == "rij":
                now = time.time()
                H_1, H_alpha, H_alpha2, H_alphabeta, H_beta, H_beta2, inv_rA, inv_rB = calc_H_alphabeta_rij(F, rAB, rA1, rB1, rA2, rB2, r12, M_inv, M1M, delta)
                htime += (time.time() - now)

                now = time.time()
                rA1_p = power_table(rA1, n_max)
                rB1_p = power_table(rB1, n_max)
                rA2_p = power_table(rA2, n_max)
                rB2_p = power_table(rB2, n_max)
                inv_rA_p  = power_table(inv_rA, int(amax), int(amin))  # already full size P
                inv_rB_p  = power_table(inv_rB, int(amax), int(amin))

                rA1_p = np.repeat(rA1_p, P2, axis=0)  # shape (P, Npow)
                rB1_p = np.repeat(rB1_p, P2, axis=0)  # shape (P, Npow)
                rA2_p = np.tile(rA2_p, (P1, 1))
                rB2_p = np.tile(rB2_p, (P1, 1))

                # Assemble basis functions from power matrices
                B = np.broadcast_to(rAB**h_idx, (P, matSize)).copy() # rA1^n*rB1^m*rA2^i*rB2^j*r12^k*rAB^h/rA^a/rB^b
                B *= rA1_p[:, n_idx]
                B *= rA2_p[:, i_idx]
                B *= rB1_p[:, m_idx]
                B *= rB2_p[:, j_idx]
                B *= r12_p[:, k_idx]
                B *= inv_rA_p[:, a_idx]
                B *= inv_rB_p[:, b_idx]
                B *= np.exp(-delta*r12)[:, None]
                btime += (time.time() - now)

                # Apply Hamiltonian prefactors
                now = time.time()
                A_1 = H_1 * B
                A_alpha = H_alpha * B
                A_alpha2 = H_alpha2 * B
                A_alphabeta = H_alphabeta * B
                A_beta = H_beta * B
                A_beta2 = H_beta2 * B
                atime += (time.time() - now)

                # Apply linear combinations to assemble symmetric basis
                # B, A_1, A_alpha, A_alpha2, A_alphabeta, A_beta, A_beta2 = combine_fcts(B, A_1, A_alpha, A_alpha2, A_alphabeta, A_beta, A_beta2, groups)
                now = time.time()
                B = B @ X
                A_1 = A_1 @ X
                A_alpha = A_alpha @ X
                A_alpha2 = A_alpha2 @ X
                A_alphabeta = A_alphabeta @ X
                A_beta = A_beta @ X
                A_beta2 = A_beta2 @ X
                attime += (time.time() - now)

            if abs(s2-plot_s2_target) < 1e-8:
                plot_chunks = update_plot_chunks(plot_chunks, mu2, B, P1, P2, x1, y1, phi2, plot_phi_target, A_1, A_alpha, A_alpha2, A_alphabeta, A_beta, A_beta2)

            # weights
            now = time.time()
            sqrtW = np.sqrt((sW[ks] * splitW[j]) * (w1[:, None] * w2[None, :]).ravel())  # (P,)
            B *= sqrtW[:, None]
            A_1 *= sqrtW[:, None]
            A_alpha *= sqrtW[:, None]
            A_alpha2 *= sqrtW[:, None]
            A_alphabeta *= sqrtW[:, None]
            A_beta *= sqrtW[:, None]
            A_beta2 *= sqrtW[:, None]  # todo: separate plot logic (re-compute quantities there); then this can happen in B directly.
            wtime += (time.time() - now)

            now = time.time()
            B = np.asfortranarray(B)
            BT = np.asfortranarray(B.T)
            A_1 = np.asfortranarray(A_1)
            A_alpha = np.asfortranarray(A_alpha)
            A_beta = np.asfortranarray(A_beta)
            A_alpha2 = np.asfortranarray(A_alpha2)
            A_alphabeta = np.asfortranarray(A_alphabeta)

            Sp = BT @ B
            Sl += Sp
            Hl_1 += BT @ A_1
            Hl_alpha += BT @ A_alpha
            Hl_beta += BT @ A_beta
            Hl_alpha2 += BT @ A_alpha2
            Hl_alphabeta += BT @ A_alphabeta
            if coords == "s12mu":
                Hl_beta2 += Sp * H_beta2  # H_beta2 is just a scalar -Minv -> avoid redoing the @
            else:
                A_beta2 = np.asfortranarray(A_beta2)
                Hl_beta2 += BT @ A_beta2
            ltime += (time.time() - now)

            nrP += P

        S_layers[rAB, s] = Sl
        H_1_layers[rAB, s] = Hl_1
        H_alpha_layers[rAB, s] = Hl_alpha
        H_alpha_2_layers[rAB, s] = Hl_alpha2
        H_alpha_beta_layers[rAB, s] = Hl_alphabeta
        H_beta_layers[rAB, s] = Hl_beta
        H_beta_2_layers[rAB, s] = Hl_beta2

        print(ks, rAB, s, nrP, time.time() - start, attime, htime, btime, atime, wtime, ltime)


plot_mu2_with_alpha_beta(
    plot_chunks = plot_chunks,

    S_layers=S_layers,
    H_1_layers=H_1_layers,
    H_alpha_layers=H_alpha_layers,
    H_beta_layers=H_beta_layers,
    H_alpha_2_layers=H_alpha_2_layers,
    H_beta_2_layers=H_beta_2_layers,
    H_alpha_beta_layers=H_alpha_beta_layers,

    matSize=bSize,
    diag_rescale_generalized=diag_rescale_generalized,

    plot_rAB_target=plot_rAB_target,
    plot_s2_target=plot_s2_target,
    plot_phi_target=plot_phi_target,

    alpha_values=np.arange(0.4, 1.3, 0.005),
    beta_values=np.array([0.0]),

    zlim_eloc=(-3, 0),
    subsample=4000,
)
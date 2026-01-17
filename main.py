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
ab_max = 0 # rij only
total_max = 5
nm_min = -0 # TODO: why doesn't that improve things?
delta = 0

# todo: alpha = 0.8 actually better, again? (at least at total_max = 5)

# delta = 0.1: E[0] := -1.174474883468479:
# E[0] := -1.1744788198234721 at delta=0.1, alpha=0.695

plot_rAB_target = 1.4

nMu = 32  # todo: test effect of these values on solution quality
nS = 30
sMax = 50

# coords = "rij"
# coords = "stmu"
coords = "s12mu" # todo: H slightly non-hermitian? How to fix that?
# TODO: check worst_pair function - for some monomials, H.T@H - H remains large. Why those? Is it an error?
# TODO: octant = False gives almost exactly the same plots, but slightly less-negative energy, and less asym.
# todo: at high s, the H entries become MASSIVE (lack of exp). Numerical problem?
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
                                    if a>0 and abs(n-i)>1: continue # prevent repeating basis functions by cancellation of rA, rB terms
                                    if b>0 and abs(m-j)>1: continue

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
                                if i == j and m > n: continue # avoid duplication of (n, m, i, j=i) and (m, n, j=i, i)
                            t = h + k + n + m + i + j
                            if t > total_max: continue
                            # rows.append((h, k, n - np.floor(i/2), m - np.floor(j/2), i, j))  #-k-m
                            rows.append((h, k, n, m, i, j))  #-k-m


basis_idx = np.array(rows, dtype=np.int16)

matSize = basis_idx[:, 0].size
bSize = len(sym_b) if coords == "rij" else matSize
print(f"MatSize = {bSize}, Coords: {coords}")

if coords == "stmu":
    Fij, Fji = calc_Fij_stmu(basis_idx)
elif coords == "s12mu":
    Fij, Fji = calc_Fij_s12mu(basis_idx)
elif coords == "rij":
    Fij = calc_F_rij(basis_idx)
    Fji = None

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

for rAB in abRange:
    s_shells, sW = build_s_shells(rAB, Ks=nS, s_max=sMax, gamma = 3.0)

    for ks, s in enumerate(s_shells):

        s1_vals, s2_vals, splitW = split_s(s, rAB, Ku=10)

        Sl = np.zeros((bSize, bSize), dtype=np.float64)
        Hl_1 = np.zeros((bSize, bSize), dtype=np.float64)
        Hl_alpha = np.zeros((bSize, bSize), dtype=np.float64)
        Hl_alpha2 = np.zeros((bSize, bSize), dtype=np.float64)
        Hl_alphabeta = np.zeros((bSize, bSize), dtype=np.float64)
        Hl_beta = np.zeros((bSize, bSize), dtype=np.float64)
        Hl_beta2 = np.zeros((bSize, bSize), dtype=np.float64)

        for j, (s1, s2) in enumerate(zip(s1_vals, s2_vals)):

            # sample electron 1 on its s1-shell
            x1, y1, _, mu1, _, w1 = sample_s_shell(rAB, s1, Nphi=2, nMu=nMu)  # x-y plane only

            # sample electron 2 on its s2-shell
            x2, y2, z2, mu2, _, w2 = sample_s_shell(rAB, s2, octant=True, nMu=nMu)
            # x2, y2, z2, mu2, _, w2 = sample_s_shell(rAB, s2, nMu=4*nMu)

            B, A_1, A_alpha, A_beta, A_alphabeta, A_alpha2, P = calc_AB(x1, y1, x2, y2, z2, rAB, s, s1, s2, mu1, mu2, w1, w2, sW[ks] * splitW[j], coords, basis_idx, delta, M1M, M_inv, Fij, Fji, X)

            def worst_pair(B, A, str):
                BW = B
                H = BW.T @ A  # This is the assembled block (for diagnosis only)
                As = H - H.T
                i, j = np.unravel_index(np.argmax(np.abs(As)), As.shape)
                # if np.abs(As[i, j] / (H[i, j] + H[j, i] + np.abs(As[i, j]))) > 1e-8:
                print(str, i, j, basis_idx[i], basis_idx[j], H[i, j], H[j, i], As[i, j], np.linalg.norm(As) / max(1e-300, np.linalg.norm(H)))
            print(s, j, s1, s2)
            (worst_pair(B, A_1, "A1"))
            (worst_pair(B, A_alpha, "Aalpha"))
            # (worst_pair(B, A_beta, "Abeta"))

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
            # if coords == "s12mu":
            Hl_beta2 += Sp * (-M_inv)  # H_beta2 is just a scalar -Minv -> avoid redoing the @
            # else:
            #     A_beta2 = np.asfortranarray(A_beta2)
            #     Hl_beta2 += BT @ A_beta2

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
    # evaluator
    calc_AB=calc_AB,

    # plot grid (electron 1)
    x1_min=-9.0,
    x1_max= 9.0,
    y1_min=-9.0,
    y1_max= 9.0,
    nx1=50,
    ny1=50,

    # basis / physics
    coords=coords,
    basis_idx=basis_idx,
    delta=delta,
    M1M=M1M,
    M_inv=M_inv,
    Fij=Fij,
    Fji=Fji,
    X=X,

    # matrices from main computation
    S_layers=S_layers,
    H_1_layers=H_1_layers,
    H_alpha_layers=H_alpha_layers,
    H_beta_layers=H_beta_layers,
    H_alpha_2_layers=H_alpha_2_layers,
    H_beta_2_layers=H_beta_2_layers,
    H_alpha_beta_layers=H_alpha_beta_layers,

    # generalized eig helper
    matSize=bSize,
    diag_rescale_generalized=diag_rescale_generalized,

    # geometry labels / exp factor
    plot_rAB_target=plot_rAB_target,

    # alpha/beta sliders
    alpha_values=np.arange(0.4, 1.3, 0.005),
    beta_values=[0],

    # numerics / plot behavior
    only_negative_E=True,
    eps=1e-14,
    zlim_eloc=(-1.5, -0.9)
)

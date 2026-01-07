import time
from distutils.command.sdist import sdist

import numpy as np
from scipy.linalg import eig
from scipy.special import gammaln

from plot import plot_wavefn_and_local_energy
from util import *
import math

# from r1_matrix import *
from r1_matrix_LP import *
# from r1_matrix_Legendre_26 import *
# from r1_matrix_Legendre_52 import *
# from r1_matrix_Legendre_126 import *
# from r1_matrix_monomial_127 import *
# from r1_matrix_Laguerre_127 import *

# Helper to mimic Maple's floating for-loops
def frange(start, stop, step):
    x = start
    if step > 0:
        while x <= stop + 1e-12:
            yield x
            x += step
    else:
        while x >= stop - 1e-12:
            yield x
            x += step


# def power_table(x, p_max):
#     # Returns matrix with x^n for columns n=0..p_max
#     x = np.asarray(x, dtype=np.float64)
#     out = np.empty((x.shape[0], p_max+1), dtype=np.float64) if x.ndim else np.empty((p_max+1,), dtype=np.float64)
#     out[..., 0] = 1.0
#     for e in range(1, p_max+1):
#         out[..., e] = out[..., e-1] * x
#     return out


def power_table(x, p_max, p_min=0):
    x = np.asarray(x, dtype=np.float64)

    n_pos = p_max + 1              # includes 0
    n_neg = -p_min if p_min < 0 else 0
    n_cols = n_pos + n_neg

    out = np.empty((x.shape[0], n_cols) if x.ndim else (n_cols,), dtype=np.float64)

    # s^0
    out[..., 0] = 1.0

    # s^1 ... s^p_max
    for e in range(1, p_max + 1):
        out[..., e] = out[..., e - 1] * x

    # s^p_min ... s^-1
    if p_min < 0:
        col = p_max + 1
        out[..., col] = x ** p_min
        for e in range(p_min + 1, 0):
            out[..., col + 1] = out[..., col] * x
            col += 1

    return out



# Todo: Ideas:
#  * treat rAB as the scaling length (exponent saved separately) - then use Kronecker products to quickly assemble enhanced S&H
#  * Also save HP components of alpha^2, alpha, 1 separately, AND then - assemble H with different alpha values directly to test for optimal alpha!
#  * Make grid tighter near nuclei (maybe just one of them) for better sampling?

# Todo: Summarize current workings in a PDF (so I don't forget all the stuff that's already implemented)

# maybe it optimizes outside? Try reducing R1max, or the minimum values.
def build_SH_xyz_separate_V_fast(
    # alpha = 0.95/1.4,
    # alpha = 0.75,
    # beta = 0.0,  # 0.1: quite alright (epsilon[0] := 0.191, no longer duplicated); 0.2 (really well behaving functions; epsilon[0] := 0.1443, all solution functions have about the same shape)
    # delta = 0.0,  # 0.5 worse than 0.1  # Careful - not currently implemented in maple
    Rmax=6,
    rStep=0.11,  # todo: resulting wave function shape is *incredibly* dependent on these values. Why? Can I make them denser at closer r still?
    R1max=12,  # Maximal radius for radial scanning of rA1
    sigma=3  # Exponent of the rA1 sampling distribution: u1 in [0.1, sqrt(R1max)], rA1=u1^sigma (higher sigma => more points near 0)
):
    # E[0] := -1.174814627652896: epsilon[0] := 328.86396264265125:

    # Coefficients not really getting smaller to the end, again.

    BO = True

    # No giant peak at the electron: E[0] := -1.1745053541808201; BUT not very smooth at large r
    #     h_max = 6
    #     k_max = 5
    #     n_max = 6
    #     m_max = 6
    #     ij_max = 6
    #     n_min = 0  # for power table
    #     total_max = 6
    # alpha_grid = np.array([0.74, 1.2, 2.0, 2.5])
    # alpha_thrs = np.array([2, 4, 6])               with ai = np.searchsorted(alpha_thrs, n, side="left")


    # Basis set maximum powers (rAB^h * r12^k * s^n * t^m * (mu1^i*mu2^j + mu1^j*mu2^i) * exp( - alpha*s - beta*rAB - gamma*r12 )
    h_max = 6
    k_max = 6
    n_max = 5
    m_max = 6
    ij_max = 6
    n_min = 0  # for power table
    # n_min = -k_max-m_max  # for power table
    total_max = 8

    # alpha_grid = np.array([0.75, 2, 5])
    # alpha_thrs = np.array([2, 4, 6])  #E[0] := -1.1742607464862156:
    # alpha_grid = np.array([0.75])
    # alpha_thrs = np.array([])  # E[0] := -1.1742766567556162: but way worse local energy, cond 10^13
    # alpha_grid = np.array([0.75, 2.0])
    # alpha_thrs = np.array([2])  # E[0] := -1.174512485344357:, pretty good local energy! cond 10^14
    # alpha_grid = np.array([0.74, 1.2, 2])
    # alpha_thrs = np.array([2, 4])  # E[0] := -1.1740258261028165: Worse energy but smooooth LE! 10^13
    # alpha_grid = np.array([0.74, 1.2, 2, 2.5])
    # alpha_thrs = np.array([2, 4, 5])  # E[0] := -1.1744567099820111: Better energy, okayish LE
    # alpha_grid = np.array([0.74, 1.2, 2.0, 2.5])
    # alpha_thrs = np.array([2, 4, 5])  # E[0] := -1.174466890469607
    # alpha_grid = np.array([0.55, 0.74, 1.2, 2.0, 2.5]) # E[0] := -1.174473374118624: (delta = 0.7)
    alpha_grid = np.array([0.55, 0.74, 1.2, 2.0, 3.0, 3.6])
    alpha_thrs = np.array([0, 2, 4, 5, 6])  # degrees up to which each alpha applies (always one shorter than _grid)
    beta_grid = np.array([0])
    beta_thrs = np.array([])
    delta_grid = np.array([0.7])  # TODO: this gives a reasonable-looking (but wide) shape; just crossing 0 instead of approaching it.
    delta_thrs = np.array([])     #    But: function is not smooth at x=0?! Should I adjust that in the ansatz? Or is it due to the electron position and okay?
                                  #      Plot both parts (from different electron positions?)
# 0.74: -1.1744386278080048, cond 9

    # TODO: try negative s powers again - did I fully test that after fixing?

# TODO: Recover logic used by the maple exports previously.
#  All inv_s etc are multiplied by n etc such that there are never negative powers => May as well expand them and return to r1's, r2's, r12-dependent logic.


    # TODO: See nice_at_e1_sol !
    #  Just fixed i-j exchange bug - try again to use gamma = 1.8 for k=1

    # (Speedup factor 1.5?)

    if BO:  # Reset if not used
        h_max = 0  # avoid rAB dependence
        beta_grid = np.array([0])
        beta_thrs = np.array([])

    use_beta = not (beta_thrs.size == 0 and beta_grid[0] == 0)
    use_delta = not (delta_thrs.size == 0 and delta_grid[0] == 0)

    print(use_beta, use_delta)


    # E[0] := -1.1627930436517466: epsilon[0] := 2.9937579205389784e-05
    # alpha = 0.75, beta=0, gamma=0
    # h_max = 2
    # k_max = 2
    # n_max = 2
    # m_max = 4
    # ij_max = 4
    # total_max = 4
# n-k, -1.1743159541558044 (try different alpha dependence too)
    rows = []
    for h in frange(0, h_max, 1):
        for k in frange(0, k_max, 1):
            for n in frange(0, n_max, 1):  # careful: Whenever using negative indices, adjust power_table call accordingly.
                for m in range(m_max + 1):
                    for i in range(ij_max + 1):
                        for j in range(i + 1):

                            # constraints
                            if (i + j) % 2 != 0: continue
                            if m % 2 != 0: continue
                            t = h + k + n + m + i + j
                            if t > total_max: continue
                            ai = np.searchsorted(alpha_thrs, n, side="left")  # Smallest index for which t<=alpha_thrs[ai]
                            bi = np.searchsorted(beta_thrs, h, side="left")
                            di = np.searchsorted(delta_thrs, k, side="left")
                            rows.append((h, k, n, m, i, j, ai, bi, di))  #-k-m

    basis_idx = np.array(rows, dtype=np.int16)

    # Shorthands for later use (no copies)
    h_idx = basis_idx[:, 0]
    k_idx = basis_idx[:, 1]
    n_idx = basis_idx[:, 2]
    m_idx = basis_idx[:, 3]
    i_idx = basis_idx[:, 4]
    j_idx = basis_idx[:, 5]
    alpha_idx = basis_idx[:, 6]
    beta_idx = basis_idx[:, 7]
    delta_idx = basis_idx[:, 8]

    h = h_idx.astype(np.float64)
    k = k_idx.astype(np.float64)
    n = n_idx.astype(np.float64)
    m = m_idx.astype(np.float64)
    i = i_idx.astype(np.float64)
    j = j_idx.astype(np.float64)
    alpha_b = alpha_grid[alpha_idx].astype(np.float64)
    beta_b = beta_grid[beta_idx].astype(np.float64)
    delta_b = delta_grid[delta_idx].astype(np.float64)

    Fij = np.stack([n * n - n, n, k * n, n * m, n * (m - h), n * i, n * j,
                     m * m, m, m * k, m * i, m * j, m * (i + j - h),
                     i * i - i, i, i * j, i * k, i * h,
                     j * j - j, j, j * k, j * h,
                     h * h + h, h, k * k + k, k,
                     np.ones_like(n)], axis=0)
    Fji = np.stack([n * n - n, n, k * n, n * m, n * (m - h), n * j, n * i,
                    m * m, m, m * k, m * j, m * i, m * (j + i - h),
                    j * j - j, j, j * i, j * k, j * h,
                    i * i - i, i, i * k, i * h,
                    h * h + h, h, k * k + k, k,
                    np.ones_like(n)], axis=0)

    matSize = h_idx.size
    print(f"MatSize = {matSize}")

    nrP = 0

    # Non-zero start values necessary to avoid division by 0 (if mu2=0 or rB2=0)
    x2_vals = np.linspace(0.00001, np.floor(Rmax / rStep) * rStep, int(np.floor(Rmax / rStep)) + 1)
    y2_vals = np.linspace(0, np.floor(Rmax / rStep) * rStep, int(np.floor(Rmax / rStep)) + 1)
    z2_vals = np.linspace(0.001, np.floor(Rmax / rStep) * rStep, int(np.floor(Rmax / rStep)) + 1)  # slight offset prevents rX2=0 (all other points are in (x,y)-plane)  # todo: temp
    X2s, Y2s, Z2s = np.meshgrid(x2_vals, y2_vals, z2_vals, indexing="ij")
    x2s = X2s.ravel()
    y2s = Y2s.ravel()
    z2s = Z2s.ravel()

    if BO:
        M1M = 1
        M_inv = 0
    else:
        M = 1836.153
        M1M = (M+1) / M
        M_inv = 1 / M

    start = time.time()

    S = np.zeros((matSize, matSize), dtype=np.float64)
    H = np.zeros((matSize, matSize), dtype=np.float64)

    test_A = None
    test_B = None

    # 1.3: -1.172233875122012
    # 1.35 -1.1719523463883157
    # 1.38 -1.1746696369875254
    # 1.4: -1.1747651120383216
    # 1.41 -1.1747227721042222
    # 1.43 -1.1745027866756477
    # 1.45: -1.1741347875229644:
    # 1.5: -1.172763095219384

    abRange = frange(1.4, 1.4, 0.2) if BO else frange(0.4, 3.0, 0.2)

    for rAB in abRange:
        xB = rAB/2
        rA2 = np.sqrt((x2s + rAB / 2) ** 2 + y2s ** 2 + z2s ** 2)
        rB2 = np.sqrt((x2s - rAB / 2) ** 2 + y2s ** 2 + z2s ** 2)

        # r2_dependencies = np.asarray(r2deps)
        # R2 = build_R2(rA2_vals, rB2_vals, r2_dependencies)  # requires: MInfTest2.mw codeGeneration (bottom of file)


        # rAB, e2-only primitives
        inv_rAB = 1.0 / rAB
        inv_rA2 = 1.0 / rA2
        inv_rB2 = 1.0 / rB2
        rA2_2 = rA2 * rA2
        rB2_2 = rB2 * rB2
        rAB_2 = rAB * rAB
        s2 = rA2+rB2
        mu2 = (rA2-rB2) * inv_rAB
        inv_mu2 = 1.0 / mu2
        inv_rAB_2 = inv_rAB * inv_rAB
        inv_mu2_2 = inv_mu2 * inv_mu2
        v2 = inv_rA2 + inv_rB2
        cos2AB = (rA2_2 + rAB_2 - rB2_2) * inv_rA2 * inv_rAB * 0.5
        cos2BA = (rAB_2 + rB2_2 - rA2_2) * inv_rAB * inv_rB2 * 0.5
        cosA2B = (rA2_2 - rAB_2 + rB2_2) * inv_rA2 * inv_rB2 * 0.5
        c12 = M_inv*(cos2AB-cos2BA)

        # Power tables
        rAB_p = power_table(rAB, h_max)
        mu2_p = power_table(mu2, ij_max)

        for u1 in frange(rStep/2, np.sqrt(R1max), rStep):
            rA1 = u1**sigma
            weight=rA1**(2-1/sigma)
            print("rAB =", rAB, "rA1 =", rA1, nrP, f"{int(10*(time.time() - start))/10}s")
            for theta in np.linspace(0, 2*np.pi, 16, endpoint=False): # Slight offset to avoid hitting nucleus B
                x1 = rA1*np.cos(theta)-xB
                y1 = rA1*np.sin(theta)
                rB1 = np.sqrt(((x1-xB)**2+y1**2))
                if abs(rA1)<0.001 or abs(rB1)<0.001: continue  # todo: temp 01

                # Compute primitives
                r12 = np.sqrt((x1 - x2s) ** 2 + (y1 - y2s) ** 2 + z2s ** 2)  # vector of r12 values for all e2 positions

                inv_r12 = 1.0 / r12
                inv_rA1 = 1.0 / rA1
                inv_rB1 = 1.0 / rB1

                r12_2 = r12 * r12
                rA1_2 = rA1 * rA1
                rB1_2 = rB1 * rB1

                s1 = rA1+rB1
                s = s1 + s2
                t = (s1-s2) * inv_rAB * 0.5
                mu1 = (rA1-rB1) * inv_rAB

                if abs(mu1) < 0.0001: continue

                inv_s = 1/s
                inv_t = 1/t
                inv_mu1 = 1/mu1

                inv_r12_2 = inv_r12 * inv_r12
                inv_s_2 = inv_s * inv_s
                inv_t_2 = inv_t * inv_t
                inv_mu1_2 = inv_mu1 * inv_mu1

                v1 = inv_rA1 + inv_rB1
                v12 = v1 + v2

                cos12A = (r12_2 - rA1_2 + rA2_2) * inv_r12 * inv_rA2 * 0.5
                cos12B = (r12_2 - rB1_2 + rB2_2) * inv_r12 * inv_rB2 * 0.5
                cos1A2 = (rA1_2 + rA2_2 - r12_2) * inv_rA1 * inv_rA2 * 0.5
                cos1AB = (rA1_2 + rAB_2 - rB1_2) * inv_rA1 * inv_rAB * 0.5
                cos1B2 = (rB1_2 + rB2_2 - r12_2) * inv_rB1 * inv_rB2 * 0.5
                cos1BA = (rAB_2 + rB1_2 - rA1_2) * inv_rAB * inv_rB1 * 0.5
                cos21A = (r12_2 + rA1_2 - rA2_2) * inv_r12 * inv_rA1 * 0.5
                cos21B = (r12_2 + rB1_2 - rB2_2) * inv_r12 * inv_rB1 * 0.5
                cosA1B = (rA1_2 - rAB_2 + rB1_2) * inv_rA1 * inv_rB1 * 0.5

                # Recurring combinations
                c1 = cos12A + cos12B + cos21A + cos21B
                c2 = cosA1B + cosA2B
                c3 = (cos12A - cos21B + cos12B - cos21A)
                c4 = cos21A - cos21B
                c5 = cos12A - cos12B
                c6 = cosA2B - cosA1B
                c7 = M_inv*(cos1A2 + cos1B2)
                c8 = M_inv*(cos1AB + cos1BA + cos2AB + cos2BA)
                c9 = M_inv*(cos1AB + cos1BA - cos2AB - cos2BA)
                c10 = M_inv*(cos1A2-cos1B2)
                c11 = M_inv*(cos1AB-cos1BA)

                inv_rAB_s = inv_rAB * inv_s
                inv_rAB_t = inv_rAB * inv_t

                # See precalc.mw for derivation

                # Coefficients of: [n^2, n, k*n, n*m, m^2, m, m*k, i^2-i, i, i*k, j^2-j, j, j*k, k^2 + k, k, 1]
                c_n2 = -( 2.0*M1M + c2 + c7 ) * inv_s_2  # n^2 - n
                c_n  = - M1M*v12 * inv_s  # n
                c_n_alpha = ( 2.0*(M1M*2.0 + c2 + c7) ) * inv_s  # n
                c_kn = -c1 * inv_r12 * inv_s  # k*n
                c_nm = c6 * inv_t  * inv_rAB_s  # n*m
                c_nmh= c8 * inv_rAB_s  # n*(m-h)
                c_ni = (c8 - c10 * inv_mu1) * inv_rAB_s  # n*i
                c_nj = (c8 - c10 * inv_mu2) * inv_rAB_s # n*j
                c_m2 = -0.25*( 2*M1M + c2 - c7)*inv_t_2*inv_rAB_2 - M_inv*inv_rAB_2 + 0.5*c9*inv_rAB_2*inv_t  # m^2
                c_m  = 0.5* M1M*(v2-v1) * inv_rAB_t + 0.25*( M1M*2.0 + c2 - c7 )*inv_t_2*inv_rAB_2 + M_inv*inv_rAB_2  # m
                c_m_alpha = - c6 * inv_rAB_t - c8 * inv_rAB
                c_mk = 0.5 * c3 *inv_r12*inv_rAB_t  # m*k
                c_mi = (c11 + 0.5*c10*inv_t) * inv_mu1 * inv_rAB_2  # m*i
                c_mj = (c12 - 0.5*c10*inv_t) * inv_mu2 * inv_rAB_2  # m*j
                c_mijh = ( 0.5 * c9 * inv_t - 2.0*M_inv ) * inv_rAB_2  # m*(i+j-h)
                c_i2 =  (( -M1M  + cosA1B )*inv_mu1_2*inv_rAB_2  + c11*inv_mu1*inv_rAB_2- M_inv*inv_rAB_2) * np.ones_like(c_n)  # i^2-i
                c_i  = ( -M1M*(inv_rA1-inv_rB1)  + c11*inv_rAB ) *inv_mu1*inv_rAB  * np.ones_like(c_n) #i
                c_i_alpha  = ( c10*inv_mu1 - c8 ) *inv_rAB * np.ones_like(c_n)  #i
                c_ij = ( -c7*inv_mu1*inv_mu2 + c11*inv_mu1 + c12*inv_mu2 - 2.0*M_inv ) * inv_rAB_2  # i*j
                c_ik = -c4 * inv_r12 * inv_mu1 * inv_rAB  # i*k
                c_ih = ( -c11*inv_mu1 + 2.0*M_inv ) * inv_rAB_2 * np.ones_like(c_n)  # i*h
                c_j2 = ( -M1M + cosA2B )*inv_mu2_2*inv_rAB_2  + c12*inv_mu2*inv_rAB_2 - M_inv*inv_rAB_2  # j^2-j
                c_j  = ( -M1M*(inv_rA2-inv_rB2) + c12*inv_rAB)*inv_mu2*inv_rAB # j
                c_j_alpha  = ( c10*inv_mu2 - c8)*inv_rAB  # j
                c_jk = -c5*inv_r12*inv_mu2*inv_rAB  # j*k
                c_jh = ( -c12*inv_mu2 + 2.0*M_inv ) * inv_rAB_2  # j*h
                c_h2 = -M_inv*inv_rAB_2 * np.ones_like(c_n)  # h^2 + h
                c_h_alpha = c8 * inv_rAB  * np.ones_like(c_n) # h
                c_k2 = - inv_r12_2  # k^2 + k
                c_k_alpha  = c1 * inv_r12  # k
                c_1_alpha  = M1M*v12
                c_1_alpha2  = -( 2*M1M + c2 + c7 )

                if use_beta:
                    c_n_beta = c8 * inv_s  # n
                    c_m_beta = 0.5*c9*inv_rAB_t - 2.*M_inv*inv_rAB
                    c_i_beta  = (c11*inv_mu1 - 2.*M_inv) * inv_rAB * np.ones_like(c_n)  #i
                    c_j_beta  = (c12*inv_mu2 - 2.*M_inv) * inv_rAB  # j
                    c_h_beta  = 2.0 * M_inv * inv_rAB  * np.ones_like(c_n) # h
                    c_1_alphabeta = -c8
                    c_1_beta =   M_inv*2.*inv_rAB * np.ones_like(c_n)
                    c_1_beta2 =  -M_inv * np.ones_like(c_n)

                if use_delta:
                    c_m_delta = - 0.5*c3*inv_rAB_t
                    c_n_delta = c1 * inv_s  # n
                    c_i_delta = c4 * inv_mu1*inv_rAB * np.ones_like(c_n) # i
                    c_j_delta = c5 * inv_mu2*inv_rAB  # j
                    c_1_alphadelta = -c1
                    c_1_delta =  2.0*inv_r12
                    c_1_delta2 =  -1.0 * np.ones_like(c_n)
                    c_k_delta  = 2.0 * inv_r12  # k

                zero = np.zeros_like(r12)
                coef_vector_1 = np.stack([c_n2, c_n, c_kn, c_nm, c_nmh, c_ni, c_nj, c_m2, c_m, c_mk, c_mi, c_mj, c_mijh, c_i2, c_i, c_ij, c_ik, c_ih, c_j2, c_j, c_jk, c_jh, c_h2, zero, c_k2, zero, zero], axis=1)
                coef_vector_alpha = np.stack([zero, c_n_alpha, zero, zero, zero, zero, zero, zero, c_m_alpha, zero, zero, zero, zero, zero, c_i_alpha, zero, zero, zero, zero, c_j_alpha, zero, zero, zero, c_h_alpha, zero, c_k_alpha, c_1_alpha], axis=1)
                coef_vector_alpha2 = np.stack([zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, c_1_alpha2], axis=1)
                if use_beta:
                    coef_vector_beta = np.stack([zero, c_n_beta, zero, zero, zero, zero, zero, zero, c_m_beta, zero, zero, zero, zero, zero, c_i_beta, zero, zero, zero, zero, c_j_beta, zero, zero, zero, c_h_beta, zero, zero, c_1_beta], axis=1)
                    coef_vector_alphabeta = np.stack([zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, c_1_alphabeta], axis=1)
                    coef_vector_beta2 = np.stack([zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, c_1_beta2], axis=1)
                if use_delta:
                    coef_vector_delta = np.stack([zero, c_n_delta, zero, zero, zero, zero, zero, zero, c_m_delta, zero, zero, zero, zero, zero, c_i_delta, zero, zero, zero, zero, c_j_delta, zero, zero, zero, zero, zero, c_k_delta, c_1_delta], axis=1)
                    coef_vector_alphadelta = np.stack([zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, c_1_alphadelta], axis=1)
                    coef_vector_delta2 = np.stack([zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, c_1_delta2], axis=1)


                # Base (no parameters)
                Hpoly_ij = coef_vector_1 @ Fij
                Hpoly_ji = coef_vector_1 @ Fji

                Hpoly_ij += (coef_vector_alpha @ Fij) * alpha_b[None, :]
                Hpoly_ji += (coef_vector_alpha @ Fji) * alpha_b[None, :]

                tmp = coef_vector_alpha2 @ Fij
                Hpoly_ij += tmp * (alpha_b[None, :] ** 2)
                Hpoly_ji += tmp * (alpha_b[None, :] ** 2)

                if use_beta:
                    Hpoly_ij += (coef_vector_beta @ Fij) * beta_b[None, :]
                    Hpoly_ji += (coef_vector_beta @ Fji) * beta_b[None, :]

                    tmp = coef_vector_beta2 @ Fij
                    Hpoly_ij += tmp * (beta_b[None, :] ** 2)
                    Hpoly_ji += tmp * (beta_b[None, :] ** 2)

                    tmp = coef_vector_alphabeta @ Fij
                    Hpoly_ij += tmp * (alpha_b[None, :] * beta_b[None, :])
                    Hpoly_ji += tmp * (alpha_b[None, :] * beta_b[None, :])

                if use_delta:
                    Hpoly_ij += (coef_vector_delta @ Fij) * delta_b[None, :]
                    Hpoly_ji += (coef_vector_delta @ Fji) * delta_b[None, :]

                    tmp = coef_vector_delta2 @ Fij
                    Hpoly_ij += tmp * (delta_b[None, :] ** 2)
                    Hpoly_ji += tmp * (delta_b[None, :] ** 2)

                    tmp = coef_vector_alphadelta @ Fij
                    Hpoly_ij += tmp * (alpha_b[None, :] * delta_b[None, :])
                    Hpoly_ji += tmp * (alpha_b[None, :] * delta_b[None, :])

                # Test vs. derivation file
                # print(f"r12={r12[5]}, rAB={rAB}, rA1={rA1}, rA2={rA2[5]}, rB1={rB1}, "
                #       f"rB2={rB2[5]}, {[float(z[5]) for z in Mplus1overMPart]}, {( -M1M  + cosA1B )*inv_mu1_2*inv_rAB_2  + c11*inv_mu1*inv_rAB_2- M_inv*inv_rAB_2}*(i^2-i)"
                #       f" + {( -c11*inv_mu1 + 2.0*M_inv ) * inv_rAB_2} *i*h  {-M_inv*inv_rAB_2}*h^2")

                # Symmetry requirements: m even, i+j even, symmetrize to phi_i_j+phi_j_i.

                r12_p = power_table(r12, k_max)
                s_p = power_table(s, n_max, n_min)
                t_p = power_table(t, m_max)
                mu1_p = power_table(mu1, ij_max)

                nr_points = r12_p.shape[0]
                nr_fncts = basis_idx.shape[0]

                scale = np.exp(n_idx * np.log(2.0 * alpha_b) - 0.5 * gammaln(2*n_idx + 1))[None, :]  # (2*alpha_b)**n_idx / np.sqrt(math.factorial(2*n_idx))

                # Assemble basis functions from power matrices
                B = np.ones((nr_points, nr_fncts), dtype=np.float64) * rAB_p[h_idx][None, :]
                B *= r12_p[:, k_idx]
                B *= s_p[:, n_idx]
                B *= t_p[:, m_idx]
                B *= np.exp(-alpha_b[None, :]*s[:, None] - beta_b[None, :]*rAB - delta_b[None, :]*r12[:, None]) * scale
                # B *= np.exp(-alpha*s - beta*rAB - delta*r12)[:, None]

                potential = potential_ri(rAB, rA1, rB1, rA2, rB2, r12)

                mu_part_ij = mu2_p[:, j_idx] * mu1_p[i_idx][None, :]  # symmetrized mu part
                mu_part_ji = mu2_p[:, i_idx] * mu1_p[j_idx][None, :]
                # H_ij_part = mu_part_ij * (coef_vector @ Fij) + mu_part_ji * (coef_vector @ Fji)  # Including the multiplyer for applying H
                H_ij_part = mu_part_ij * Hpoly_ij + mu_part_ji * Hpoly_ji

                A = B * H_ij_part
                B *= (mu_part_ij + mu_part_ji)
                A += B * potential[:, None]

                if abs(rA1-2) < 0.6 and abs(theta-1*np.pi/4) < 0.2 and (BO or abs(rAB-1.4) < 0.0001):  # change to 1/4*Pi to see electron
                    print(theta, x1, y1, rA1, rB1)
                    test_A = A
                    test_B = B  # Vectors of rows HP and P for later comparison of local energy deviation

                S += ( B.T @ B ) * weight
                H += ( B.T @ A ) * weight

                nrP += 1
    print(nrP)
    print(time.time() - start)
    return S, H, test_A, test_B, x2_vals, y2_vals, z2_vals, basis_idx


# TODO
#  * 2. test exp(-beta(r-X)**2) factor; optimize beta & X
#  * 3. try changing coordinates & using larger basis sets
#  * 4. find more efficient eigenvalue solver for the large basis


def potential_ri(rAB, rA1, rB1, rA2, rB2, r12):
    return 1/rAB + 1/r12 - 1/rA1 - 1/rB1 - 1/rA2 - 1/rB2


S, H, test_A, test_B, x2_vals, y2_vals, z2_vals, basis_idx = build_SH_xyz_separate_V_fast()

eigvals = np.linalg.eigvalsh(S)

# optional diagnostics
print((eigvals))
print("min eig:", np.abs(eigvals).min())
print("max eig:", np.abs(eigvals).max())
print("cond:", eigvals.max() / eigvals.min())

# inspect_small_overlap_eigenvectors(S)

H, S = diag_rescale_generalized(H, S)

print("After Diag rescaling:")
eigvals = np.linalg.eigvalsh(S)
print((eigvals))
print("min eig:", np.abs(eigvals).min())
print("max eig:", np.abs(eigvals).max())
print("cond:", eigvals.max() / eigvals.min())

# inspect_small_overlap_eigenvectors(S)

h_idx = basis_idx[:, 0]
k_idx = basis_idx[:, 1]
n_idx = basis_idx[:, 2]
m_idx = basis_idx[:, 3]
i_idx = basis_idx[:, 4]
j_idx = basis_idx[:, 5]

E, C = eig(H, S)  # Todo: test finer grid on 60-element basis set. If worse - what do the new functions (vs. 54) do?
idx = np.argsort(E)
E = np.real(E[idx])
C = np.real(C[:, idx])


i = 0
while i < len(E) and E[i] < 0:
    ci = C[:, i]
    ci = ci/ci[0]
    hp = test_A @ ci
    p = test_B @ ci

    eps = np.sum((hp - p * E[i]) ** 2)
    print(f"E[{i}] := {E[i]}: epsilon[{i}] := {eps}: "
          # f"C[{i}] := {[f' + ({float(x)}) * rAB^{h_idx[ii]}*r12^{k_idx[ii]}*s^{n_idx[ii]}*t^{m_idx[ii]}*mu1^{i_idx[ii]}*mu2^{j_idx[ii]}' for ii, x in enumerate(ci)]}")
          f"C[{i}] := " + "".join( f" + ({float(x)})*rAB^{h_idx[ii]}*r12^{k_idx[ii]}*s^{n_idx[ii]}*t^{m_idx[ii]}*mu1^{i_idx[ii]}*mu2^{j_idx[ii]}" for ii, x in enumerate(ci)) )
    i += 1

# # estimated energy
# sigma = -1.174475
#
# # Start vector from top-left 100×100
# x0, lam0 = make_start_vector_from_topleft(H, S, k=100, sigma=sigma, which="closest")
#
# # Refine
# lam, x, info = shift_invert_target_eigpair(
#     H, S, sigma=sigma, x0=x0,
#     max_iter=30,
#     tol=1e-9,
#     update_sigma=False,     # start conservative
#     regularize_mu=0.0,      # try 1e-12 * np.linalg.norm(H, ord=np.inf) if LU gets cranky
#     normalize="S",
#     verbose=True
# )
#
# print("Result:", lam, info)
# # print(np.real(x))
# print(f"C[{i}] := " + "".join( f" + ({float(x)})*rAB^{h_idx[ii]}*r12^{k_idx[ii]}*s^{n_idx[ii]}*t^{m_idx[ii]}*mu1^{i_idx[ii]}*mu2^{j_idx[ii]}" for ii, x in enumerate(np.real(x))) )
#
#
# plot_wavefn_and_local_energy(
#     test_A, test_B, np.real(x)[None, :], np.array([lam]),
#     x2_vals, y2_vals, z2_vals,
#     eps=1e-12,
#     clip_percentiles = (1, 99)
# )


# for rc in [1e-10, 1e-12, 1e-14, 1e-15]:
#     E, C, info = solve_gen_eig_deflated(H, S, rcond=rc, assume_H_symmetric=False)
#     print(rc, info["K"], np.real(E[:5]), np.max(info["residual_rel"][:5]))
# for rc in [1e-10, 1e-12, 1e-14, 1e-15]:
#     E, C, info = solve_gen_eig_deflated(H, S, rcond=rc, assume_H_symmetric=True)
#     print(rc, info["K"], np.real(E[:5]), np.max(info["residual_rel"][:5]))

# lam, x = residual_minimize_generalized(H, S, -1.17445, C[:, 0], iters=8, rcond=1e-12, damping=0.5, ridge=0.0)
# print(lam)
#
plot_wavefn_and_local_energy(
    test_A, test_B, C, E,
    x2_vals, y2_vals, z2_vals,
    eps=1e-12,
    clip_percentiles = (1, 99)
)
#
#     # TODO: TestL
#     #  * Try exp(rIj/rAB...) exponent - is it actually worse?
#     #  * Is ((rA2-rB2)/rAB)^(2*dB2); correct in the ansatz, or are the ((rA1-rB1)/rAB)^1*((rA2-rB2)/rAB)^1 type terms missing?
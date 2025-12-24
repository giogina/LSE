import time
from distutils.command.sdist import sdist

import numpy as np
from scipy.linalg import eig
from scipy.linalg import eigh
from plot import plot_wavefn_and_local_energy
from util import inspect_small_overlap_eigenvectors, diag_rescale_generalized

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


def build_R2(rA2_vals, rB2_vals, r2_dependencies):
    """
    rA2_vals: shape (N,)
    rB2_vals: shape (N,)
    r2_dependencies: shape (3, M)
        row 1 ignored for now (r12 exponent)
        row 2 = exponent for rA2
        row 3 = exponent for rB2

    returns R2: shape (N, M)
    """

    # Extract exponent rows
    expA = r2_dependencies[1]  # shape (M,)
    expB = r2_dependencies[2]  # shape (M,)

    # Compute powers using broadcasting
    # rA2_vals[:,None] gives shape (N,1)
    # expA[None,:]   gives shape (1,M)
    RA = rA2_vals[:, None] ** expA[None, :]
    RB = rB2_vals[:, None] ** expB[None, :]

    # Elementwise multiplication produces (N, M)
    return RA * RB


# Todo: Ideas:
#  * treat rAB as the scaling length (exponent saved separately) - then use Kronecker products to quickly assemble enhanced S&H
#  * Also save HP components of alpha^2, alpha, 1 separately, AND then - assemble H with different alpha values directly to test for optimal alpha!
#  * Make grid tighter near nuclei (maybe just one of them) for better sampling?

# Todo: Summarize current workings in a PDF (so I don't forget all the stuff that's already implemented)

# maybe it optimizes outside? Try reducing R1max, or the minimum values.
def build_SH_xyz_separate_V_fast(
    # alpha = 0.95/1.4,
    alpha = 0.75,
    beta = 0.1,  # 0.1: quite alright (epsilon[0] := 0.191, no longer duplicated); 0.2 (really well behaving functions; epsilon[0] := 0.1443, all solution functions have about the same shape)
    delta = 0.0,  # 0.5 worse than 0.1  # Careful - not currently implemented in maple
    Rmax=6,
    rStep=0.2,
    R1max=12,  # Maximal radius for radial scanning of rA1
    sigma=3  # Exponent of the rA1 sampling distribution: u1 in [0.1, sqrt(R1max)], rA1=u1^sigma (higher sigma => more points near 0)
):
    matSize = len(P_r1_matrix(1.0, 1.0, 1.0)[1]) #* len(alphas)
    print(f"MatSize = {matSize}")

    Sdict = {}
    Hdict = {}

    nrP = 0

    # Non-zero start values necessary to avoid division by 0 (if mu2=0 or rB2=0)
    x2_vals = np.linspace(0.001, np.floor(Rmax / rStep) * rStep, int(np.floor(Rmax / rStep)) + 1)
    y2_vals = np.linspace(0, np.floor(Rmax / rStep) * rStep, int(np.floor(Rmax / rStep)) + 1)
    z2_vals = np.linspace(0.01, np.floor(Rmax / rStep) * rStep, int(np.floor(Rmax / rStep)) + 1)  # slight offset prevents rX2=0 (all other points are in (x,y)-plane)
    X2s, Y2s, Z2s = np.meshgrid(x2_vals, y2_vals, z2_vals, indexing="ij")
    x2s = X2s.ravel()
    y2s = Y2s.ravel()
    z2s = Z2s.ravel()

    M = 1836.153
    M1M = (M+1) / M
    M_inv = 1 / M

    start = time.time()

    test_A = None
    test_B = None

    for rAB in frange(0.8, 2.2, 0.2):
        xB = rAB/2

        S = np.zeros((matSize, matSize), dtype=np.float64)
        H = np.zeros((matSize, matSize), dtype=np.float64)

        rA2_vals = np.sqrt((x2s + rAB / 2) ** 2 + y2s ** 2 + z2s ** 2)
        rB2_vals = np.sqrt((x2s - rAB / 2) ** 2 + y2s ** 2 + z2s ** 2)

        r2_dependencies = np.asarray(r2deps)
        R2 = build_R2(rA2_vals, rB2_vals, r2_dependencies)  # requires: MInfTest2.mw codeGeneration (bottom of file)

        for u1 in frange(rStep/2, np.sqrt(R1max), rStep):
            rA1 = u1**sigma
            weight=rA1**(2-1/sigma)
            print("rAB =", rAB, "rA1 =", rA1, nrP)
            for theta in np.linspace(0, 2*np.pi, 16, endpoint=False): # Slight offset to avoid hitting nucleus B
                x1 = rA1*np.cos(theta)-xB
                y1 = rA1*np.sin(theta)
                rB1 = np.sqrt(((x1-xB)**2+y1**2))
                if abs(rA1)<0.01 or abs(rB1)<0.01: continue

                # Compute primitives
                rA2=rA2_vals
                rB2=rB2_vals
                r12 = np.sqrt((x1 - x2s) ** 2 + (y1 - y2s) ** 2 + z2s ** 2)  # vector of r12 values for all e2 positions

                inv_r12 = 1.0 / r12
                inv_rAB = 1.0 / rAB
                inv_rA1 = 1.0 / rA1
                inv_rA2 = 1.0 / rA2
                inv_rB1 = 1.0 / rB1
                inv_rB2 = 1.0 / rB2

                r12_2 = r12 * r12
                rA1_2 = rA1 * rA1
                rA2_2 = rA2 * rA2
                rB1_2 = rB1 * rB1
                rB2_2 = rB2 * rB2
                rAB_2 = rAB * rAB

                s1 = rA1+rB1
                s2 = rA2+rB2
                s = s1 + s2
                t = (s1-s2) * inv_rAB * 0.5
                mu1 = (rA1-rB1) * inv_rAB
                mu2 = (rA2-rB2) * inv_rAB

                if abs(mu1) < 0.0001: continue

                inv_s = 1/s
                inv_t = 1/t
                inv_mu1 = 1/mu1
                inv_mu2 = 1/mu2

                inv_rAB_2 = inv_rAB * inv_rAB
                inv_r12_2 = inv_r12 * inv_r12
                inv_s_2 = inv_s * inv_s
                inv_t_2 = inv_t * inv_t
                inv_mu1_2 = inv_mu1 * inv_mu1
                inv_mu2_2 = inv_mu2 * inv_mu2

                v1 = inv_rA1 + inv_rB1
                v2 = inv_rA2 + inv_rB2
                v12 = v1 + v2

                cos12A = (r12_2 - rA1_2 + rA2_2) * inv_r12 * inv_rA2 * 0.5
                cos12B = (r12_2 - rB1_2 + rB2_2) * inv_r12 * inv_rB2 * 0.5
                cos1A2 = (rA1_2 + rA2_2 - r12_2) * inv_rA1 * inv_rA2 * 0.5
                cos1AB = (rA1_2 + rAB_2 - rB1_2) * inv_rA1 * inv_rAB * 0.5
                cos1B2 = (rB1_2 + rB2_2 - r12_2) * inv_rB1 * inv_rB2 * 0.5
                cos1BA = (rAB_2 + rB1_2 - rA1_2) * inv_rAB * inv_rB1 * 0.5
                cos21A = (r12_2 + rA1_2 - rA2_2) * inv_r12 * inv_rA1 * 0.5
                cos21B = (r12_2 + rB1_2 - rB2_2) * inv_r12 * inv_rB1 * 0.5
                cos2AB = (rA2_2 + rAB_2 - rB2_2) * inv_rA2 * inv_rAB * 0.5
                cos2BA = (rAB_2 + rB2_2 - rA2_2) * inv_rAB * inv_rB2 * 0.5
                cosA1B = (rA1_2 - rAB_2 + rB1_2) * inv_rA1 * inv_rB1 * 0.5
                cosA2B = (rA2_2 - rAB_2 + rB2_2) * inv_rA2 * inv_rB2 * 0.5

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
                c12 = M_inv*(cos2AB-cos2BA)



                # Coefficients of: [n^2, n, k*n, n*m, m^2, m, m*k, i^2-i, i, i*k, j^2-j, j, j*k, k^2 + k, k, 1]
                Mplus1overMPart = [
                                        -( 2.0*M1M + c2 + c7 ) * inv_s_2,  # n^2 - n
                                       ( 2.0*(M1M*2.0 + c2 + c7)*alpha - M1M*v12 + c1*delta + c8*beta ) * inv_s,
                                        -c1 * inv_r12 * inv_s,  # k*n
                                        c6 * inv_t  * inv_rAB * inv_s,  # n*m
                                        c8 * inv_rAB * inv_s,  # n*(m-h)
                                        (c8 - c10 * inv_mu1) * inv_rAB * inv_s,  # n*i
                                        (c8 - c10 * inv_mu2) * inv_rAB * inv_s, # n*j
                                       -0.25*( 2*M1M + c2 - c7)*inv_t_2*inv_rAB_2 - M_inv*inv_rAB_2 + 0.5*c9*inv_rAB_2*inv_t,  # m^2


                                   (0.5*( M1M*(v2-v1) - c3*delta - 2*c6*alpha + c9*beta)*inv_t
                                    - c8*alpha - 2.0*beta*M_inv )*inv_rAB
                                    + 0.25*( M1M*2.0 + c2 - c7 )*inv_t_2*inv_rAB_2
                                    + M_inv*inv_rAB_2, # m


                                        0.5 * c3 *inv_r12*inv_t*inv_rAB, # m*k
                                       (c11 + 0.5*c10*inv_t) * inv_mu1 * inv_rAB_2, # m*i
                                       (c12 - 0.5*c10*inv_t) * inv_mu2 * inv_rAB_2, # m*j
                                       ( 0.5 * c9 * inv_t - 2.0*M_inv ) * inv_rAB_2, # m*(i+j-h)
                                       # ( -M1M  + cosA1B )*inv_mu1_2*inv_rAB_2  + c11*inv_mu1*inv_rAB_2- M_inv*inv_rAB_2,  # i^2-i
                                       ( -M1M*(inv_rA1-inv_rB1) + delta + c10*alpha + c11*beta + c11*inv_rAB ) *inv_mu1*inv_rAB - (c8*alpha + 2.*beta*M_inv) * inv_rAB, #i
                                       ( -c7*inv_mu1*inv_mu2 + c11*inv_mu1 + c12*inv_mu2 - 2.0*M_inv ) * inv_rAB_2,  # i*j
                                        -c4 * inv_r12 * inv_mu1 * inv_rAB,  # i*k
                                       # ( -c11*inv_mu1 + 2.0*M_inv ) * inv_rAB_2,  # i*h
                                       ( -M1M + cosA2B)*inv_mu2_2*inv_rAB_2  + c12*inv_mu2*inv_rAB_2 - M_inv*inv_rAB_2, # j^2-j
                                       ( -M1M*(inv_rA2-inv_rB2) + delta*c5 + c10*alpha + c12*beta + c12*inv_rAB)*inv_mu2*inv_rAB - (c8*alpha + 2.*beta*M_inv) * inv_rAB, # j
                                        -c5*inv_r12*inv_mu2*inv_rAB,  # j*k
                                       ( -c12*inv_mu2 + 2.0*M_inv ) * inv_rAB_2,  # j*h
                                       # -M_inv*inv_rAB_2, # h^2
                                       -M_inv*inv_rAB_2 + (M_inv*2.*beta + c8*alpha) * inv_rAB, # h
                                        - inv_r12_2, # k^2 + k
                                        (c1*alpha+2*delta) * inv_r12,  # k
                                       M1M*v12*alpha - ( 2*M1M + c2 + c7 )*alpha**2 + 2*delta*inv_r12 - (c1*alpha+2.0*delta)*delta - (c8*alpha + beta*M_inv)*beta + M_inv*2.*beta*inv_rAB
                                   ]
                print(f"r12={r12[5]}, rAB={rAB}, rA1={rA1}, rA2={rA2[5]}, rB1={rB1}, "
                      f"rB2={rB2[5]}, {[float(z[5]) for z in Mplus1overMPart]}, {( -M1M  + cosA1B )*inv_mu1_2*inv_rAB_2  + c11*inv_mu1*inv_rAB_2- M_inv*inv_rAB_2}*(i^2-i)"
                      f" + {( -c11*inv_mu1 + 2.0*M_inv ) * inv_rAB_2} *i*h  {-M_inv*inv_rAB_2}*h^2")

                nrP += 1
        #
        #         A_blocks = []
        #         B_blocks = []  # Parts of A and B for chosen values of the exponential paramenters
        #
        #         for alpha in alphas:
        #             PR1 = np.asarray(P_r1_matrix(rAB, rA1, rB1), dtype=np.float64)  # Matrix of all the r1-dependent coefficients evaluated at given e1 position
        #             HR1 = np.asarray(HP_r1_matrix(rAB, rA1, rB1, alpha, beta, 0, 1836.153), dtype=np.float64)  # Matrix of all the r1-dependent coefficients evaluated at given e1 position
        #
        #             r12_vals = np.sqrt((x1-x2s)**2+(y1-y2s)**2+z2s**2)   # vector of r12 values for all e2 positions
        #             exp_factors = np.exp(-alpha*(rA1+rB1+rA2_vals+rB2_vals)-beta*rAB-delta*r12_vals)  # vector of exponential factor for all e2 positions
        #             potential = potential_ri(rAB, rA1, rB1, rA2_vals, rB2_vals, r12_vals)
        #
        #             r12_pows = r12_vals[:, None] ** r2_dependencies[0][None, :]  # Multiply columns of R2 contributions with corresponding r12 values
        #             R2_r12 = R2 * r12_pows
        #             A_alpha = (R2_r12 @ HR1) * exp_factors[:, None]  # Use inner product of r1 and r2 dependent monomial pieces to assemble HP elements
        #             B_alpha = (R2_r12 @ PR1) * exp_factors[:, None]
        #             A_alpha += B_alpha * potential[:, None]
        #
        #             A_blocks.append(A_alpha)  # (N2, ncol_alpha)
        #             B_blocks.append(B_alpha)
        #
        #         A = np.concatenate(A_blocks, axis=1)
        #         B = np.concatenate(B_blocks, axis=1)
        #
        #         if abs(rAB-1.4) < 0.0001 and (rA1-4) < 0.6 and (theta-2*np.pi/3) < 0.2:
        #             test_A = A
        #             test_B = B  # Vectors of rows HP and P for later comparison of local energy deviation
        #
        #         S += ( B.T @ B ) * weight
        #         H += ( B.T @ A ) * weight
        #         nrP += 1
        #
        # Sdict[rAB] = S
        # Hdict[rAB] = H

    print(nrP)
    print(time.time() - start)
    # return Sdict, Hdict, test_A, test_B, x2_vals, y2_vals, z2_vals


# TODO
#  * 2. test exp(-beta(r-X)**2) factor; optimize beta & X
#  * 3. try changing coordinates & using larger basis sets
#  * 4. find more efficient eigenvalue solver for the large basis


def potential_ri(rAB, rA1, rB1, rA2, rB2, r12):
    return 1/rAB + 1/r12 - 1/rA1 - 1/rB1 - 1/rA2 - 1/rB2

def assemble_SH(Sdict, Hdict, beta):
    r_vals = np.array(list(Sdict.keys()))
    weights = np.exp(-beta * (r_vals-1.4)**2)

    S_sum = sum(Sdict[r] * w for r, w in zip(r_vals, weights))
    H_sum = sum(Hdict[r] * w for r, w in zip(r_vals, weights))

    return S_sum, H_sum

# Sdict, Hdict, test_A, test_B, x2_vals, y2_vals, z2_vals = build_SH_xyz_separate_V_fast()
build_SH_xyz_separate_V_fast()

# # Sdict, Hdict = build_SH_xyz_fast()
# for beta in frange(0, 0, 0.1):
#     # print(f"beta = {beta};")
#     S, H = assemble_SH(Sdict, Hdict, beta)
#
#     eigvals = np.linalg.eigvalsh(S)
#
#     # optional diagnostics
#     print((eigvals))
#     print("min eig:", np.abs(eigvals).min())
#     print("max eig:", np.abs(eigvals).max())
#     print("cond:", eigvals.max() / eigvals.min())
#
#     H, S = diag_rescale_generalized(H, S)
#
#     print("After Diag rescaling:")
#
#     eigvals = np.linalg.eigvalsh(S)
#
#     # optional diagnostics
#     print((eigvals))
#     print("min eig:", np.abs(eigvals).min())
#     print("max eig:", np.abs(eigvals).max())
#     print("cond:", eigvals.max() / eigvals.min())
#
#
#     inspect_small_overlap_eigenvectors(S)
#
#
#     E, C = eig(H, S)  # Todo: test finer grid on 60-element basis set. If worse - what do the new functions (vs. 54) do?
#     # E, C = solve_gen_eig_project_only(H, S, rcond=1e-9) # Todo: somhow worse? Why?
#
#     idx = np.argsort(E)
#     E = np.real(E[idx])
#     C = np.real(C[:, idx])
#
#     i = 0
#     while i < len(E) and E[i] < 0:
#         ci = C[:, i]
#         ci = ci/ci[0]
#         hp = test_A @ ci
#         p = test_B @ ci
#
#         eps = np.sum((hp - p * E[i]) ** 2)
#         print(f"E[{i}] := {E[i]}: epsilon[{i}] := {eps}: "
#               f"C[{i}] := {[float(x) for x in ci]}")
#         i += 1
#
#     plot_wavefn_and_local_energy(
#         test_A, test_B, C, E,
#         x2_vals, y2_vals, z2_vals,
#         eps=1e-12,
#         clip_percentiles = (1, 99)
#     )
#
#     # TODO: TestL
#     #  * Try exp(rIj/rAB...) exponent - is it actually worse?
#     #  * Is ((rA2-rB2)/rAB)^(2*dB2); correct in the ansatz, or are the ((rA1-rB1)/rAB)^1*((rA2-rB2)/rAB)^1 type terms missing?
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
# R2, each row: [f(rA2, rB2) = rA2^i*rB2^j for each n=1..N; i,j depend on ansatz function index m]

# Todo: Ideas:
#  * treat rAB as the scaling length (exponent saved separately) - then use Kronecker products to quickly assemble enhanced S&H
#  * Also save HP components of alpha^2, alpha, 1 separately, AND then - assemble H with different alpha values directly to test for optimal alpha!
#  * Make grid tighter near nuclei (maybe just one of them) for better sampling?

# Todo: Summarize current workings in a PDF (so I don't forget all the stuff that's already implemented)

# maybe it optimizes outside? Try reducing R1max, or the minimum values.
def build_SH_xyz_separate_V_fast(
    # alpha = 0.95/1.4,
    # alpha = 1,
    alphas = np.array([0.75], dtype=np.float64),  # best: 1.0; 0.8: horrible; 2.0: selecting different state far outside?
    beta = 0.1,  # 0.1: quite alright (epsilon[0] := 0.191, no longer duplicated); 0.2 (really well behaving functions; epsilon[0] := 0.1443, all solution functions have about the same shape)
    delta = 0.0,  # 0.5 worse than 0.1  # Careful - not currently implemented in maple
    Rmax=6,
    rStep=0.2,
    R1max=12,  # Maximal radius for radial scanning of rA1
    sigma=3  # Exponent of the rA1 sampling distribution: u1 in [0.1, sqrt(R1max)], rA1=u1^sigma (higher sigma => more points near 0)
):
    matSize = len(P_r1_matrix(1.0, 1.0, 1.0)[1]) * len(alphas)
    print(f"MatSize = {matSize}")

    Sdict = {}
    Hdict = {}

    nrP = 0

    x2_vals = np.linspace(0, np.floor(Rmax / rStep) * rStep, int(np.floor(Rmax / rStep)) + 1)
    y2_vals = np.linspace(0, np.floor(Rmax / rStep) * rStep, int(np.floor(Rmax / rStep)) + 1)
    z2_vals = np.linspace(0.01, np.floor(Rmax / rStep) * rStep, int(np.floor(Rmax / rStep)) + 1)  # slight offset prevents rX2=0 (all other points are in (x,y)-plane)
    X2s, Y2s, Z2s = np.meshgrid(x2_vals, y2_vals, z2_vals, indexing="ij")
    x2s = X2s.ravel()
    y2s = Y2s.ravel()
    z2s = Z2s.ravel()

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
            print("rAB =", rAB, "rA1 =", rA1, nrP, time.time() - start)
            for theta in np.linspace(0, 2*np.pi, 16, endpoint=False): # Slight offset to avoid hitting nucleus B
                x1 = rA1*np.cos(theta)-xB
                y1 = rA1*np.sin(theta)
                rB1 = np.sqrt(((x1-xB)**2+y1**2))
                if abs(rA1)<0.01 or abs(rB1)<0.01: continue

                A_blocks = []
                B_blocks = []  # Parts of A and B for chosen values of the exponential paramenters

                for alpha in alphas:
                    PR1 = np.asarray(P_r1_matrix(rAB, rA1, rB1), dtype=np.float64)  # Matrix of all the r1-dependent coefficients evaluated at given e1 position
                    HR1 = np.asarray(HP_r1_matrix(rAB, rA1, rB1, alpha, beta, 0, 1836.153), dtype=np.float64)  # Matrix of all the r1-dependent coefficients evaluated at given e1 position

                    r12_vals = np.sqrt((x1-x2s)**2+(y1-y2s)**2+z2s**2)   # vector of r12 values for all e2 positions
                    exp_factors = np.exp(-alpha*(rA1+rB1+rA2_vals+rB2_vals)-beta*rAB-delta*r12_vals)  # vector of exponential factor for all e2 positions
                    potential = potential_ri(rAB, rA1, rB1, rA2_vals, rB2_vals, r12_vals)

                    r12_pows = r12_vals[:, None] ** r2_dependencies[0][None, :]  # Multiply columns of R2 contributions with corresponding r12 values
                    R2_r12 = R2 * r12_pows
                    A_alpha = (R2_r12 @ HR1) * exp_factors[:, None]  # Use inner product of r1 and r2 dependent monomial pieces to assemble HP elements
                    B_alpha = (R2_r12 @ PR1) * exp_factors[:, None]
                    A_alpha += B_alpha * potential[:, None]

                    A_blocks.append(A_alpha)  # (N2, ncol_alpha)
                    B_blocks.append(B_alpha)

                A = np.concatenate(A_blocks, axis=1)
                B = np.concatenate(B_blocks, axis=1)

                if abs(rAB-1.4) < 0.0001 and (rA1-4) < 0.6 and (theta-2*np.pi/3) < 0.2:
                    test_A = A
                    test_B = B  # Vectors of rows HP and P for later comparison of local energy deviation

                S += ( B.T @ B ) * weight
                H += ( B.T @ A ) * weight
                nrP += 1

        Sdict[rAB] = S
        Hdict[rAB] = H

    print(nrP)
    print(time.time() - start)
    return Sdict, Hdict, test_A, test_B, x2_vals, y2_vals, z2_vals


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

Sdict, Hdict, test_A, test_B, x2_vals, y2_vals, z2_vals = build_SH_xyz_separate_V_fast()

# Sdict, Hdict = build_SH_xyz_fast()
for beta in frange(0, 0, 0.1):
    # print(f"beta = {beta};")
    S, H = assemble_SH(Sdict, Hdict, beta)

    eigvals = np.linalg.eigvalsh(S)

    # optional diagnostics
    print((eigvals))
    print("min eig:", np.abs(eigvals).min())
    print("max eig:", np.abs(eigvals).max())
    print("cond:", eigvals.max() / eigvals.min())

    H, S = diag_rescale_generalized(H, S)

    print("After Diag rescaling:")

    eigvals = np.linalg.eigvalsh(S)

    # optional diagnostics
    print((eigvals))
    print("min eig:", np.abs(eigvals).min())
    print("max eig:", np.abs(eigvals).max())
    print("cond:", eigvals.max() / eigvals.min())


    inspect_small_overlap_eigenvectors(S)


    E, C = eig(H, S)  # Todo: test finer grid on 60-element basis set. If worse - what do the new functions (vs. 54) do?
    # E, C = solve_gen_eig_project_only(H, S, rcond=1e-9) # Todo: somhow worse? Why?

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
              f"C[{i}] := {[float(x) for x in ci]}")
        i += 1

    plot_wavefn_and_local_energy(
        test_A, test_B, C, E,
        x2_vals, y2_vals, z2_vals,
        eps=1e-12,
        clip_percentiles = (1, 99)
    )

    # TODO: TestL
    #  * Try exp(rIj/rAB...) exponent - is it actually worse?
    #  * Is ((rA2-rB2)/rAB)^(2*dB2); correct in the ansatz, or are the ((rA1-rB1)/rAB)^1*((rA2-rB2)/rAB)^1 type terms missing?
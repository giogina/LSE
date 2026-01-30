import time

from calc import *
from sampling import *
from coords import *
from layers import *

BO = True
M = 1836.1526738  #Previously used: 1836.153

# Basis set maximum powers (rAB^h * r12^k * s^n * t^m * (mu1^i*mu2^j + mu1^j*mu2^i) * exp( - alpha*s - beta*rAB - gamma*r12 )
h_max = 3 # Even 2 should be pretty accurate
k_max = 5
nm_min = -1  # setting this to negative values makes the r12 polynomial less convergent-looking (larger r12^5 coeff)
n_max = 8
m_max = 8 # smu only  # todo: test negative powers of s1, s2
ij_max = 8 # smu only  # todo: according to bo-scan-coeffs.ods, this needs to go higher than n,m (converges more slowly in mu)
ab_max = 0 # rij only
total_max = 8
delta = 0.1  # TODO: Try delta-sequence instead of r12-poly

# todo: Test h -> h+i+j (essentially not dividing mu1, mu2 by rAB) - is that better w.r.t. beta dependence?
for phiGammaMax in [1.0]:
    nMu = 12
    nrPhi = 12
    nrS = 24  # 30-60 are optimal according to numerical tests (any more, and accumulation of numerical errors starts taking over)
    nrS12 = 15  # Odd -> s1=s2 allowed
    sMax = 24

    abRange, wAB = [1.4], [1.0]
    # abRange, wAB = build_rAB_grid(KR = 8, R_min=0.7, R_max=2.2, gamma = 1.0)
    startFromrAB = 0 # Use when resuming a calculation

    # label = "hij-h3k5"
    # label = f"sampling-test-{nMu}-{nrPhi}-{nrS}-{nrS12}-{sMax}"
    label = f"basis-test-{h_max}-{k_max}-{nm_min}..{n_max}-{ij_max}-{total_max}_sampling-{nMu}-{nrPhi}-{nrS}-{nrS12}-{sMax}"

    # coords = "rij"
    # coords = "stmu"
    # coords = "s12mu"
    coords = "s12mu_morse"  # Best performance & accuracy
    # coords = "s12uv_morse"

    if BO:
        M1M = 1
        M_inv = 0
        h_max = 0  # avoid rAB dependence
    else:
        M1M = (M + 1) / M
        M_inv = 1 / M

    basis_idx, bSize, X = build_basis_idx(coords, h_max, k_max, nm_min, n_max, m_max, ij_max, total_max, ab_max)
    Fij, Fji = calc_Fij(basis_idx, coords)
    print(f"MatSize = {bSize}, Coords: {coords}")

    start = time.time()
    nrP = 0

    for kab, rAB in enumerate(abRange):
        if rAB < startFromrAB: continue
        s_shells, sW = build_s_shells(rAB, Ks=nrS, s_max=sMax, gamma = 3.0)
        init_layers(coords, basis_idx, delta, M1M, M_inv, Fij, Fji, X, total_max, BO)

        for ks, s in enumerate(s_shells):
            s1_vals, s2_vals, splitW = split_s(s, rAB, Ku=nrS12, gamma = 4.0)  # high gamma: a lot more s1 ~ s2
            init_rAB_layers(bSize)

            for j, (s1, s2) in enumerate(zip(s1_vals, s2_vals)):
                x1, y1,  _, mu1, _, w1 = sample_s_shell(rAB, s1, Nphi=2, nMu=nMu)  # sample electron 1 on its s1-shell (x-y plane only)
                x2, y2, z2, mu2, _, w2 = sample_s_shell(rAB, s2, octant=True, nMu=nMu, Nphi=nrPhi, s1=s1, gamma_phi = 1.0)   # sample electron 2 on its s2-shell (octant x,y,z>0)
                B, A_1, A_alpha, A_beta, A_alpha2, A_alphabeta, c_beta2, P = calc_AB(x1, y1, x2, y2, z2, rAB, s, s1, s2, mu1, mu2, w1, w2, wAB[kab] * sW[ks] * splitW[j], coords, basis_idx, delta, M1M, M_inv, Fij, Fji, X)
                accumulate_rAB_layers(B, A_1, A_alpha, A_beta, A_alpha2, A_alphabeta, c_beta2)
                nrP += P

            accumulate_layers(rAB, s)
            print(ks, rAB, s, nrP, time.time() - start)

        save_layers(label, rAB)
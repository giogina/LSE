import time

from calc import *
from sampling import *
from coords import *
from layers import *

BO = True
M = 1836.1526738  #Previously used: 1836.153

# Basis set maximum powers (rAB^h * r12^k * s^n * t^m * (mu1^i*mu2^j + mu1^j*mu2^i) * exp( - alpha*s - beta*rAB - gamma*r12 )
h_max = 2 # Even 2 should be pretty accurate
k_max = 6
nm_min = -1  # improves cusps; little effect on energy
nm_max = 2
ij_max = 8
total_max = 8 # todo: Why beta=8 instead of 13, anyway?
# deltas = [0.0, 0.3]
delta = 0.0  # TODO: Try delta-sequence instead of r12-poly.

# todo: split-s still inaccurate (integral-wise) for gamma > 1; but ugly results in Eloc for gamma = 1. How to solve that?

for nMu in [16]:
    # nMu = 12
    nrPhi = 32
    nrS = 24  # 30-60 are optimal according to numerical tests (any more, and accumulation of numerical errors starts taking over)
    nrS12 = 15  # Odd -> s1=s2 allowed
    nrrAB = 24
    sMax = 20

    startFromrAB = 0.0 # Use when resuming a calculation
    upTorAB = 12.0
    abRange, wAB = build_rAB_grid(KR = nrrAB, R_min=0.4, R_max=3.0, Re=1.4, gamma = 2.0)
    print(abRange)

    label = f"nMu_test_{nMu}"
    # h+i+j is a bit better at alpha=1.0, beta=8.0
    # label = "delta-test"
    # label = f"basis-test-{h_max}-{k_max}-{nm_min}..{nm_max}-{ij_max}-{total_max}_sampling-{nMu}-{nrPhi}-{nrS}-{nrS12}-{sMax}"

    coords = "s12mu_morse"  # Best performance & accuracy
    # coords = "rij"
    # coords = "stmu"
    # coords = "s12mu"
    # coords = "s12uv_morse"

    separateFiles = False

    if BO:
        M1M = 1
        M_inv = 0
        h_max = 0  # avoid rAB dependence
        abRange, wAB = [1.4], [1.0]
    else:
        M1M = (M + 1) / M
        M_inv = 1 / M

    basis_idx, bSize, X = build_basis_idx(coords, h_max, k_max, nm_min, nm_max, nm_max, ij_max, total_max) # nrDeltas=len(deltas)
    Fij, Fji = calc_Fij(basis_idx, coords)
    print(f"MatSize = {bSize}, Coords: {coords}")

    start = time.time()
    nrP = 0

    if not separateFiles: init_layers(coords, basis_idx, delta, M1M, M_inv, Fij, Fji, X, total_max, BO)

    for kab, rAB in enumerate(abRange):
        if rAB < startFromrAB or rAB > upTorAB: continue
        s_shells, sW = build_s_shells(rAB, Ks=nrS, s_max=sMax, gamma = 3.0)
        if separateFiles: init_layers(coords, basis_idx, delta, M1M, M_inv, Fij, Fji, X, total_max, BO)
        for ks, s in enumerate(s_shells):
            s1_vals, s2_vals, splitW = split_s(s, rAB, Ku=nrS12, gamma = 4.0)  # high gamma: a lot more s1 ~ s2
            init_rAB_layers(bSize)

            for j, (s1, s2) in enumerate(zip(s1_vals, s2_vals)):
                x1, y1,  _, mu1, _, w1 = sample_s_shell(rAB, s1, Nphi=2, nMu=nMu)  # sample electron 1 on its s1-shell (x-y plane only)
                x2, y2, z2, mu2, _, w2 = sample_s_shell(rAB, s2, octant=True, nMu=nMu+1, Nphi=nrPhi, s1=s1, gamma_phi = 1.0)   # sample electron 2 on its s2-shell (octant x,y,z>0)
                B, A_1, A_alpha, A_beta, A_alpha2, A_alphabeta, c_beta2, P = calc_AB(x1, y1, x2, y2, z2, rAB, s, s1, s2, mu1, mu2, w1, w2, wAB[kab] * sW[ks] * splitW[j], coords, basis_idx, delta, M1M, M_inv, Fij, Fji, X)
                accumulate_rAB_layers(B, A_1, A_alpha, A_beta, A_alpha2, A_alphabeta, c_beta2)
                nrP += P

            print(ks, rAB, s, nrP, time.time() - start)
            accumulate_layers(rAB, s)

        if separateFiles: save_layers(label, rAB)
    if not separateFiles: save_layers(label, "all")
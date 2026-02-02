import pickle
from glob import glob

import numpy as np
from numpy.ma.core import arange

from plot import plot_Psi_Eloc_by_alpha_beta, plot_nonBO_from_files

# file = "SHlayers_t8_n-i_m-j_delta0_BO.pkl" # pretty good!
# file = "SHlayers_t8_nm-khalf_delta0_BO_363.pkl"
file = "SHlayers_t8_delta0.0_BO_many-low-k_810.pkl" # better ee cusp than high-k with t = k+...
# file = "SHlayers_10.pkl"
# file = "SHlayers_t9_delta0.0_BO_many-low-k_1140.pkl"
# file = "SHlayers_t6_delta0.1_BO_many-low-k-nonneg-nm-COUPLED-DIMER-R12.0_360.pkl" # worse ee, better ne as above
# file = "SHlayers_t7_delta0.1_k5h7_1368_*.pkl"
# file = "SHlayers_t7_delta0.1_BO_k5h7-BOscan_540_1.9856485906844448.pkl"
# file = "SHlayers_t4_delta0.1_k4h4-morse_210_*.pkl"
# plotrAB = 1.368119113560103
plotrAB = 1.4
# file = "SHlayers_t8_delta0.1_BO_s12uv-ij3_805_1.4.pkl" # worse even than t8_nm-khalf_delta0_BO_363 (similar # of basis functions)... Why does that have so much fewer fcts though? (t = k+..., at least)
# file = "SHlayers_t7_delta0.1_BO_k5h7-BOscan_540_1.368119113560103.pkl"
# file = "SHlayers_t5_delta0.1_BO_s12uv-ij3_320_1.4.pkl"
# file = "SHlayers_t5_delta0.1_BO_s12mu-ij3_180_1.4.pkl"

# file = "SHlayers_t8_delta0.1_h3k5_1926_*.pkl"  # Best at: beta = 8, alpha = 0.9, with E = -1.16404
# file = "SHlayers_t8_delta0.1_BO_sampling-test-16-12-20-23-30_810_1.4.pkl"
# file = "SHlayers_t8_delta0.1_BO_sampling-test-12-12-20-15-30-1.0_810_1.4.pkl"
file = "SHlayers_t8_delta0.1_BO_basis-test-3-6-8-8-8_sampling-12-12-20-15-20_945_1.4.pkl"
# file = "SHlayers_t8_delta0.1_BO_basis-test-3-4-8-8-8_sampling-12-12-20-15-20_675_1.4.pkl"
# file = "SHlayers_t8_delta0.1_BO_basis-test-3-4-8-7-8_sampling-12-12-20-15-20_670_1.4.pkl"
# file = "SHlayers_t6_delta0.1_BO_basis-test-3-5--1..8-8-6_sampling-12-12-24-15-24_810_1.4.pkl" # n, m =-1..6
# file = "SHlayers_t8_delta0.1_BO_sampling-test-12-12-20-15-20-no_dens_810_1.4.pkl" # n, m = 0..8
# file = "SHlayers_t8_delta0.1_hij-h3k5_2850_*.pkl"

file = "SHlayers_t5_delta0.1_BO_multialpha_162_1.4.pkl"

# delta = -1.0: horrible.
# delta = -0.3: not great
# delta = -0.1: cusp function still a bit offset (to below, as a whole)
# delta =  0.0: offset as a whole, but flatter
# delta =  0.1: Almost the same, Fee really flat though.
# delta =  0.3: Still good, but Fee bent downwards



# Basis test: (12-12-20-15-20-1.0), delta=0.1
# kmax = 6:       -1.1744757880727803
# kmax = 5:       -1.1744758250054517
# kmax = 4:       -1.1744760111147405  # higher kmax -> less negative
# nmmax = 7:      -1.1744759404572032 # worse local energy

# Below:
# gamma_phi = gamma_phi_from_ds(ds, gamma_max=8.)
# return sample_s_shell_phi_bias_octant(rAB, s, nMu + 1, Nphi, gamma_phi)  # * int(np.sqrt(gamma_phi))

# Exact:      −1.174475931
#   nrS = 10: -1.1752641004439177 (16-12-10-21-30)
#   nrS = 15: -1.1744481424394024 (16-12-15-21-30)
#   nrS = 20: -1.1744756704550880 (16-12-20-21-30)
#   nrS = 25: -1.1744756902015085 (16-12-25-21-30) *
#   nrS = 30: -1.1744756942647818 (16-12-30-21-30)
#   nMu =  8: -1.1744758387977365 ( 8-12-20-21-30)
#   nMu = 12: -1.1744757273010578 (12-12-20-21-30) *
#   nMu = 16: -1.1744756704550880 (16-12-20-21-30)
#   nMu = 20: -1.1744756987908325 (20-12-20-21-30)
#   nMu = 24: -1.1744757008532654 (24-12-20-21-30)
# nrPhi =  8: -1.1744753866153097 (16- 8-20-21-30)
# nrPhi = 12: -1.1744756704550880 (16-12-20-21-30) *
# nrPhi = 16: -1.1744756867794202 (16-16-20-21-30)
# nrPhi = 20: -1.1744756917496630 (16-20-20-21-30)
# nrS12 = 11: -1.1744744085387566 (16-12-20-11-30)
# nrS12 = 15: -1.1744757144797389 (16-12-20-15-30) *
# nrS12 = 19: -1.1744756690241691 (16-12-20-19-30)
# nrS12 = 21: -1.1744756704550880 (16-12-20-21-30)
# nrS12 = 23: -1.1744756721520062 (16-12-20-23-30)

# (s1-s2)/(s-2*rAB) mu densifying: -1.174475756616024 (12-12-20-15-30-3.0)
# (s1-s2)/(2*rAB) mu densifying:   -1.1744757563373653 (12-12-20-15-30-3.0)
# gamma = 1.0 (no densifying):     -1.1744758597236769
# Same, but nrS = 25:              -1.1744758544951772
# Same, but nrS=20, sMax=20:       -1.1744758250054517 (12-12-20-15-20-3.0)
# Same, but nrS=20, sMax=40:       -1.174476601551813  # less density at tiny s -> more negative energy.
# nMu + 1:                         -1.174475841238099
# no mu densifying, only phi w. 6: -1.174475779439825

# todo: plot at rAB = 1 - does it dip negative? Why is beta=8 better than beta=13, when for the 1D problem beta=13 is the clear optimum? (non-rectangular function collection?)
# todo: plot out f(rAB).
#  Would this be different (8 vs 13) if using h-independent tmax check?
#  How "product-like" does the wave function behave in the first place?

if "*" in file:
    for alpha in arange(1.00, 1.01, 0.1):
        for beta in arange(8.8, 9.01, 0.1):
            print(alpha, beta)
            rAB = 1.4

            plot_nonBO_from_files(
                file,
                alpha=alpha,
                beta=beta,
                plot_rAB_target=rAB,
                nx1=50, ny1=50,
                x1_min=-8, x1_max=8,
                y1_min=-8, y1_max=8,
                zlim_eloc=(-1.22, -1.1),
                rcond = 1e-16,
                savepic=f"~/Pictures/{file.replace('SHlayers_', '').replace('*.pkl', '')}_alpha{alpha:.2f}_beta{beta:.2f}_rAB{rAB}_Eeee.png"
            )

else:
    if file == "SHlayers_10.pkl":
        with open("SHlayers_10.pkl", "rb") as f:
            layers = pickle.load(f)
        with open("meta_10.pkl", "rb") as f:
            meta = pickle.load(f)["meta"]
    else:
        with open(file, "rb") as f:
            layers = pickle.load(f)
        meta = layers["meta"]

    plot_Psi_Eloc_by_alpha_beta(
        x1_min=-8.0,
        x1_max= 8.0,
        y1_min=-8.0,
        y1_max= 8.0,
        nx1=50,
        ny1=50,
        meta = meta,
        SH_layers=layers,
        plot_rAB_target=plotrAB,
        alpha_values=np.arange(0.4, 1.3, 0.005),
        beta_values=np.arange(-3.0, 20.0, 0.01),
        zlim_eloc=(-1.2, -1.1)
    )

# for bb in meta["basis_idx"]:
#     print(bb)

# E, C, cond = solve_HS(layers, 0.75, 0.0, 1e-15)
# print(E.min())
# −1.174 475 931

# TODO: Why is the ne cusp function asymmetric? (Which nucleus am I checking anyway?)

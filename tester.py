import pickle
from glob import glob

import numpy as np
from numpy.ma.core import arange

from plot import plot_nonBO_from_files, plot_Psi_Eloc_multi_params_from_files

# file = "SHlayers_t8_n-i_m-j_delta0_BO.pkl" # pretty good!
# file = "SHlayers_t8_nm-khalf_delta0_BO_363.pkl"
# file = "SHlayers_t8_delta0.0_BO_many-low-k_810.pkl" # better ee cusp than high-k with t = k+...
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

# file = "SHlayers_t8_delta0.1_hij-h3k5_2850_*.pkl"

# file = "SHlayers_t5_delta0.1_BO_multialpha_162_1.4.pkl"
# file = "SHlayers_t8_delta0.0_BO_many-low-k_810.pkl"

# file = "SHlayers_t6_delta0.1_multialpha_522_all.pkl"
# file = "SHlayers_t6_delta0.1_BO_multialpha_246_all.pkl"

# really good energy (-1.1744759166422694) with alpha=0.5, 1.0, 1.1

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

# h+i+j is a bit better at alpha=1.0, beta=8.0

# todo: Why is beta=8 better than beta=13, when for the 1D problem beta=13 is the clear optimum? (non-rectangular function collection?)
# todo: plot out f(rAB).
#  Would this be different (8 vs 13) if using h-independent tmax check?
#  How "product-like" does the wave function behave in the first place?

# todo: Check assemble_HS_multi_alpha for cancellations.

# file = "SHlayers_t6_delta0.0_multialpha-new_rAB_clustering_775_all.pkl"
# file = "SHlayers_t6_delta0.1_BO_multialpha_426_all.pkl"
# file = "SHlayers_t5_delta0.0_multialpha-new_rAB_clustering_525_all.pkl"  # nrrAB=16

# file = "SHlayers_t6_delta0.0_BO_nMu_test_24_426_all.pkl"
# file = "SHlayers_t6_delta0.0_BO_nMu_test_24_426_gamma1.0.pkl"  # with split_s_gamma = 1.0
# file = "SHlayers_t8_delta0.0_BO_nMu_test_16_875_all.pkl" # new leggauss-phi sampling (16-32-24)
# file = "SHlayers_t8_delta0.0_BO_nMu_16_arange_phi_875_all.pkl" # new leggauss-phi sampling (16-32-24)
# file = "SHlayers_t6_delta0.0_BO_kL_306_all.pkl"
# file = "SHlayers_t6_delta0.0_BO_k6n4_609_all.pkl" # Increasing the k dependence too much gives very strange results?!
#
# file = "SHlayers_t5_delta0.0_BO_no-matmul-dd_342_all.pkl" # -1.1744869542787129
# file = "SHlayers_t5_delta0.0_BO_nomatmul-dd_s24_s12-15_342_all.pkl"  # Higher sampling grid: -1.1744714437495383
# file = "SHlayers_t5_delta0.0_BO_no-matmul-dd_342_all.pkl"  # s=20
# file = "SHlayers_t6_delta0.0_BO_no-matmul-dd_486_all.pkl"
# file = "SHlayers_t7_delta0.0_BO_no-matmul-dd_756_all.pkl"
file = "SHlayers_t7_delta0.0_BO_sMax30_756_all.pkl"

# file = "SHlayers_t6_delta0.0_multialpha-new_rAB_clustering_775_all.pkl"
# file = "SHlayers_t6_delta0.0_BO_rAB3.5_420_all.pkl"
# file = "SHlayers_t6_delta0.0_BO_rAB7_420_all.pkl"
# file = "SHlayers_t8_delta0.0_BO_rAB4_905_all.pkl"  # E = -1.0163902595696197
# file = "SHlayers_t8_delta0.0_BO_rAB4-psinonr12_181_all.pkl"  # Psi is r12-independent; V is normal: E = -1.014473775868411
# file = "SHlayers_t8_delta0.0_BO_rAB4-Vnonr12_543_all.pkl"  # Psi=Psi(r12), but V is r12-independent: E = -1.3421697872330607
# file = "SHlayers_t8_delta0.0_BO_rAB4-allnonr12_181_all.pkl" # Psi and V r12-independent: E = -1.3421697344332832

# file = "SHlayers_t8_delta0.0_BO_rAB7-psinonr12_181_all.pkl"  #  Psi r12-independent; E = -1.0001032352646306

file = "SHlayers_t8_delta0.0_BO_rAB7_905_all.pkl" # E = -1.000197839221467

# todo: alpha=0.1 seems to be numerically unstable (especially now) - since the outer matrices aren't being squished down enough and cancellation happens while assembling H, S from layers?

# -1.1744714437495383 # unsorted
# -1.174470100748761 with sorted s0

file = "SHlayers_t8_delta0.0_BO_rAB14_905_all.pkl"

plot_Psi_Eloc_multi_params_from_files(
    file,
    xy_max=6.0,
    nx1=50,
    plot_rAB_target=14.0,
    zlim_eloc=(-1.2, -0.9),
    rcond=1e-14,
    alpha_text_init="0.5"
)


#ref: -1.1736534008915154
#     -1.173653400891507
#     -1.1736534008915154
    # -1.1736534008915154
    # -1.1736534008915154
# file = "SHlayers_t3_delta0.0_BO_asym-test_10_all.pkl"

# [ 0  0  0  1  1] -> [ 2  0  1  2  0]
# [ 2  0  0  1  1] -> [ 5  0  0  1  1]
# [ 0  0  0  0  0] -> [ 1  0  0  0  0]
# [ 1  0  0  0  0] -> [ 5  0  0  0  0] 1.7e-5
# [ 0  0  0  0  0] -> [ 5  0  0  0  0] 1.5e-5
# [ 0  0  0  0  0] -> [ 0  2  1  0  0] 0 -> 9

# Fij = np.stack([0, 0, k * k + k, k, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ..., 1], axis=0)
# [n * n - n, n, 0, 0, 0, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n * m, 0, 0, 0, 0, 0, 0, 1]

    # dd_add(dd_add(dd_mul(dd_minus(c13), inv_s1_2), dd_mul(dd_minus(inv_s1_s2), c4)), dd_add(dd_minus(inv_rA1_rB1), dd_minus(inv_rA2_rB2)))

# dd_add(dd_add(dd_mul(v12, M1M), dd_mul_exact_scalar(dd_mul(dd_add(dd_mul_exact_scalar(c13, 2.), c4), inv_s1), 2.)), dd_mul(dd_add(dd_mul_exact_scalar(c14, 2.), c4), inv_s2))



# Low asym:
# phi -> same except different i, j:
# high asym: -1 -> 0 in nm; 0->high k
# # testing without -1: Highest k or delta-k, change in nmk by 1 are worst.
# almost perfect: m or k = 0
# alpha =0.8 -> 1.0 changes things a lot - large k no longer so asymmetric??

# k_max=0:
#   Odd delta-i,j: perfect (also even delta-i,j with n, m = 0)
#   Even delta-i,j: worse (even 0-0).
#
# if "*" in file:
#     for alpha in arange(1.00, 1.01, 0.1):
#         for beta in arange(8.9, 8.91, 0.1):
#             print(alpha, beta)
#             rAB = 1.4
#
#             plot_nonBO_from_files(
#                 file,
#                 alphas=[0.3, 1.0, 2.0, 3.0],
#                 betas=[9.0, 9.0, 9.0, 9.0, 9.0],
#                 Rms=[1.2, 1.3, 1.4, 1.5, 1.6],
#                 plot_rAB_target=rAB,
#                 nx1=50, ny1=50,
#                 x1_min=-6, x1_max=6,
#                 y1_min=-6, y1_max=6,
#                 zlim_eloc=(-1.22, -1.1),
#                 rcond = 1e-14,
#                 savepic=f"~/Pictures/{file.replace('SHlayers_', '').replace('*.pkl', '')}_alpha{alpha:.2f}_beta{beta:.2f}_rAB{rAB}_multi-test_Eeee.png"
#             )
#
# else:
#
#     plot_Psi_Eloc_multi_params_from_files(
#         file,
#         xy_max= 6.0,
#         nx1=50,
#         plot_rAB_target=plotrAB,
#         zlim_eloc=(-1.22, -1.1),
#         rcond=1e-17,
#     )
#

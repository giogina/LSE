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

file = "SHlayers_t8_delta0.1_h3k5_1926_*.pkl"  # Best at: beta = 8, alpha = 0.9, with E = -1.16403

if "*" in file:
    for alpha in arange(0.7, 1.1, 0.1):
        for beta in arange(7.8, 8.2, 0.05):

            plot_nonBO_from_files(
                file,
                alpha=alpha,
                beta=beta,
                plot_rAB_target=1.4,
                nx1=50, ny1=50,
                x1_min=-8, x1_max=8,
                y1_min=-8, y1_max=8,
                zlim_eloc=(-1.2, -1.1),
                savepic=f"~/Pictures/nonbo_t8_delta0.1_h3k5_1926_alpha{alpha}_beta{beta}_Eeee.png"
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

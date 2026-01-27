import pickle
from glob import glob

import numpy as np

from plot import plot_Psi_Eloc_by_alpha_beta

# file = "SHlayers_t8_n-i_m-j_delta0_BO.pkl" # pretty good!
# file = "SHlayers_t8_nm-khalf_delta0_BO_363.pkl"
# file = "SHlayers_t7_delta0.1_BO_many-low-k_360.pkl" # wobbly asymmetric ne cusp function (uses n-i and k<=5 with t not capped by k)
# file = "SHlayers_t8_delta0.0_BO_many-low-k_810.pkl" # better ee cusp than high-k with t = k+...
# file = "SHlayers_10.pkl"
# file = "SHlayers_t9_delta0.0_BO_many-low-k_1140.pkl"
# file = "SHlayers_t6_delta0.1_BO_many-low-k-nonneg-nm-COUPLED-DIMER-R12.0_360.pkl" # worse ee, better ne as above
# file = "SHlayers_t7_delta0.1_k5h7_1368_*.pkl"
# file = "SHlayers_t7_delta0.1_BO_k5h7-BOscan_540_1.9856485906844448.pkl"
file = "SHlayers_t4_delta0.1_k4h4-morse_210_*.pkl"
plotrAB = 1.4

if "*" in file:
    layers = {}
    for ff in glob(file):
        with open(ff, "rb") as f:
            layers_new = pickle.load(f)
        if "meta" not in layers.keys():
            layers = layers_new.copy()
            meta = layers["meta"]
        else:
            for k in layers_new.keys():
                if not k=="meta":
                    for l in layers_new[k].keys():
                        layers[k][l] = layers_new[k][l].copy()
                        print(ff, k, l)
elif file == "SHlayers_10.pkl":
    with open("SHlayers_10.pkl", "rb") as f:
        layers = pickle.load(f)
    with open("meta_10.pkl", "rb") as f:
        meta = pickle.load(f)["meta"]
else:
    with open(file, "rb") as f:
        layers = pickle.load(f)
    meta = layers["meta"]

print(meta["coords"])

# for bb in meta["basis_idx"]:
#     print(bb)

# E, C, cond = solve_HS(layers, 0.75, 0.0, 1e-15)
# print(E.min())
# −1.174 475 931

# TODO: Why is the ne cusp function asymmetric? (Which nucleus am I checking anyway?)

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

import pickle

from plot import plot_Psi_Eloc_by_alpha_beta
from solver import solve_HS

file = "SHlayers_10.pkl"

if file == "SHlayers_10.pkl":
    with open("SHlayers_10.pkl", "rb") as f:
        layers = pickle.load(f)
    with open("meta_10.pkl", "rb") as f:  # todo: later, this is in layers.
        meta = pickle.load(f)
else:
    with open("SHlayers_10.pkl", "rb") as f:
        layers = pickle.load(f)
    meta = layers["meta"]

# E, C, cond = solve_HS(layers, 0.75, 0.0, 1e-15)
# print(E.min())
# −1.174 475 931


plot_Psi_Eloc_by_alpha_beta(

    # plot grid (electron 1)
    x1_min=-8.0,
    x1_max= 8.0,
    y1_min=-8.0,
    y1_max= 8.0,
    nx1=50,
    ny1=50,

    # basis data
    meta = meta,

    # matrices from main computation
    SH_layers=layers,

    # geometry labels / exp factor
    plot_rAB_target=1.4,

    # alpha/beta sliders
    alpha_values=np.arange(0.4, 1.3, 0.005),
    beta_values=np.arange(-3.0, 20.0, 0.01),

    # numerics / plot behavior
    only_negative_E=True,
    eps=1e-14,
    zlim_eloc=(-1.5, -0.9)
)

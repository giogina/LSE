import numpy as np
import pickle

layers = {}
Sl = None
Hl_1 = None
Hl_alpha = None
Hl_alpha2 = None
Hl_alphabeta = None
Hl_beta = None
Hl_beta2 = None
bsize = None
tMax = None
BO = False

def init_layers(coords, basis_idx, delta, M1M, M_inv, Fij, Fji, X, tmax, bo):
    global layers, tMax, BO
    tMax = tmax
    BO = bo
    layers = {
        "S": {},
        "H_1": {},
        "H_alpha": {},
        "H_alpha2": {},
        "H_alphabeta": {},
        "H_beta": {},
        "H_beta2": {},
        "meta": {
            "coords": coords,
            "basis_idx": basis_idx,
            "delta": delta,
            "M1M": M1M,
            "M_inv": M_inv,
            "Fij": Fij,
            "Fji": Fji,
            "X": X
        }
    }
    return layers

def init_rAB_layers(bSize):
    global Sl, Hl_1, Hl_alpha, Hl_alpha2, Hl_alphabeta, Hl_beta, Hl_beta2, bsize
    bsize = bSize
    Sl = np.zeros((bSize, bSize), dtype=np.float64)
    Hl_1 = np.zeros((bSize, bSize), dtype=np.float64)
    Hl_alpha = np.zeros((bSize, bSize), dtype=np.float64)
    Hl_alpha2 = np.zeros((bSize, bSize), dtype=np.float64)
    Hl_alphabeta = np.zeros((bSize, bSize), dtype=np.float64)
    Hl_beta = np.zeros((bSize, bSize), dtype=np.float64)
    Hl_beta2 = np.zeros((bSize, bSize), dtype=np.float64)

def accumulate_rAB_layers(B, A_1, A_alpha, A_beta, A_alpha2, A_alphabeta, c_beta2):
    global Sl, Hl_1, Hl_alpha, Hl_alpha2, Hl_alphabeta, Hl_beta, Hl_beta2

    B = np.asfortranarray(B)
    BT = np.asfortranarray(B.T)
    A_1 = np.asfortranarray(A_1)
    A_alpha = np.asfortranarray(A_alpha)
    A_beta = np.asfortranarray(A_beta)

    Sp = BT @ B
    Sl += Sp
    Hl_1 += BT @ A_1
    Hl_alpha += BT @ A_alpha
    Hl_beta += BT @ A_beta
    Hl_alpha2 += BT @ A_alpha2
    Hl_alphabeta += BT @ A_alphabeta
    Hl_beta2 += Sp * c_beta2  # H_beta2 is just a scalar -Minv -> avoid redoing the @

def accumulate_layers(rAB, s):
    global layers
    layers["S"][rAB, s] = Sl
    layers["H_1"][rAB, s] = Hl_1
    layers["H_alpha"][rAB, s] = Hl_alpha
    layers["H_alpha2"][rAB, s] = Hl_alpha2
    layers["H_alphabeta"][rAB, s] = Hl_alphabeta
    layers["H_beta"][rAB, s] = Hl_beta
    layers["H_beta2"][rAB, s] = Hl_beta2

def save_layers(label, rAB):
    global layers
    savefile = f"SHlayers_t{tMax}_delta{layers['meta']['delta']}{'_BO' if BO else ''}{'_' + label if label is not None else ''}_{bsize}_{rAB}.pkl"
    with open(savefile, "wb") as f:
        pickle.dump(layers, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved to: {savefile}")
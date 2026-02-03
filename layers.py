import numpy as np
import pickle

layers = {}
Sl = None
Hl_1 = None
Hl_alpha = None
Hl_alpha2 = None
Hl_alphabeta = None
Hl_beta = None
Hlc_beta2 = None # Just the coeffs c_beta2
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
        "c_beta2": 0.0,
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
    global Sl, Hl_1, Hl_alpha, Hl_alpha2, Hl_alphabeta, Hl_beta, Hlc_beta2, bsize
    bsize = bSize
    Sl = np.zeros((bSize, bSize), dtype=np.float64)
    Hl_1 = np.zeros((bSize, bSize), dtype=np.float64)
    Hl_alpha = np.zeros((bSize, bSize), dtype=np.float64)
    Hl_alpha2 = np.zeros((bSize, bSize), dtype=np.float64)
    Hl_alphabeta = np.zeros((bSize, bSize), dtype=np.float64)
    Hl_beta = np.zeros((bSize, bSize), dtype=np.float64)
    Hlc_beta2 = 0.

def accumulate_rAB_layers(B, A_1, A_alpha, A_beta, A_alpha2, A_alphabeta, c_beta2):
    global Sl, Hl_1, Hl_alpha, Hl_alpha2, Hl_alphabeta, Hl_beta, Hlc_beta2

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
    Hlc_beta2 = c_beta2  # H_beta2 is just a scalar -> avoid redoing the @

def accumulate_layers(rAB, s):
    global layers
    layers["S"][rAB, s] = Sl
    layers["H_1"][rAB, s] = Hl_1
    layers["H_alpha"][rAB, s] = Hl_alpha
    layers["H_alpha2"][rAB, s] = Hl_alpha2
    layers["H_alphabeta"][rAB, s] = Hl_alphabeta
    layers["H_beta"][rAB, s] = Hl_beta
    layers["c_beta2"] = Hlc_beta2

def rel_asym(A):
    nrm = np.linalg.norm(A)
    if nrm == 0.0:
        return 0.0
    return np.linalg.norm(A - A.T) / nrm

# todo: implement call with proper dR range. Look for all 1.4011 that need adapting
def assemble_HS(SH_layers, alpha, beta, Rm=1.4011, H = None, S = None, debug = False, coords="s12mu"):

    if S is None: S = np.zeros_like(next(iter(SH_layers["S"].values())), dtype=np.float64)
    if H is None: H = np.zeros_like(S, dtype=np.float64)

    for (rAB0, s0), Sl in SH_layers["S"].items():
        if coords.endswith("_morse"):
            exps = np.exp(-2 * alpha * s0 - 2 * beta * (Rm - rAB0)**2)
        else:
            exps = np.exp(-2 * alpha * s0 - 2 * beta * rAB0)

        S += Sl * exps
        Hl = np.zeros_like(H)
        Hl += SH_layers["H_1"][rAB0, s0]
        Hl += SH_layers["H_alpha"][rAB0, s0] * alpha
        Hl += SH_layers["H_alpha2"][rAB0, s0] * alpha**2
        rm = (rAB0 - Rm) if coords == "s12mu_morse" else 1. # (For s12mu_morse, the rm factor is removed from c_beta_* and c_alphabeta_*
        Hl += SH_layers["H_beta"][rAB0, s0] * rm * beta
        Hl += SH_layers["H_alphabeta"][rAB0, s0] * rm * alpha*beta

        cf = SH_layers.get("c_beta2", -4. * SH_layers["meta"]["M_inv"]) * rm**2 * beta**2  # constant factor from _beta and _beta2 that can be directly applied to S  (fallback for older calcs)
        if coords == "s12mu_morse":
            cf += 2. * SH_layers["meta"]["M_inv"] * beta  # was left out there from c_beta_1
        Hl += Sl * cf

        H += Hl * exps

    if debug:
        print("H asym rel:", rel_asym(H))
        print("S asym rel:", rel_asym(S))
    return H, S

from dataclasses import dataclass

@dataclass(frozen=True)
class StitchedBasis:
    N: int                      # basis size per tile
    blocks: tuple               # ({"alpha":..., "beta":..., "Rm":..., "coords":...}, ...)

    def block_slice(self, k: int) -> slice:
        return slice(k*self.N, (k+1)*self.N)

    def split_c(self, c: np.ndarray):
        return [c[self.block_slice(k)] for k in range(len(self.blocks))]

    def global_index(self, k: int, i: int) -> int:
        return k*self.N + i

    def iter_blocks(self):
        for k, blk in enumerate(self.blocks):
            sl = slice(k*self.N, (k+1)*self.N)
            yield k, blk, sl


def assemble_HS_multi_alpha(SH_layers, alphas, betas, Rms, coords, H=None, S=None):

    N = next(iter(SH_layers["S"].values())).shape[0]
    n = len(alphas)*len(betas)
    if H is None:
        S = np.zeros((n*N, n*N), dtype=np.float64)
        H = np.zeros((n*N, n*N), dtype=np.float64)

    blocks_list = []
    for alpha in alphas:
        for b, beta in enumerate(betas):
            blocks_list.append({"alpha": alpha, "beta": beta, "Rm": Rms[b]})
    blocks = tuple(blocks_list)
    print(f"({n*N} functions)")

    for (rAB0, s0), Sl in SH_layers["S"].items():
        for i, b1 in enumerate(blocks):
            r = slice(i * N, (i + 1) * N)
            for j, b2 in enumerate(blocks):
                c = slice(j * N, (j + 1) * N)
                S_tile, H_tile = _assemble_HS_alphas_piece(SH_layers, b1["alpha"], b2["alpha"], b1["beta"], b2["beta"], b1["Rm"], b2["Rm"], coords, s0, rAB0)
                S[r, c] += S_tile
                H[r, c] += H_tile
    return H, S, StitchedBasis(N=N, blocks=blocks)

def _assemble_HS_alphas_piece(SH_layers, alpha1, alpha2, beta1, beta2, Rm1, Rm2, coords, s0, rAB0):
    S_tile = np.zeros_like(next(iter(SH_layers["S"].values())), dtype=np.float64)
    H_tile = np.zeros_like(S_tile, dtype=np.float64)

    if coords.endswith("_morse"):
        exps = np.exp(-(alpha1+alpha2) * s0 - beta1 * (Rm1 - rAB0) ** 2 - beta2 * (Rm2 - rAB0) ** 2)
    else:
        exps = np.exp(-(alpha1+alpha2) * s0 - (beta1+beta2) * rAB0)

    S_tile = SH_layers["S"][rAB0, s0] * exps
    Hl = np.zeros_like(H_tile)
    Hl += SH_layers["H_1"][rAB0, s0]
    Hl += SH_layers["H_alpha"][rAB0, s0] * alpha2
    Hl += SH_layers["H_alpha2"][rAB0, s0] * alpha2 ** 2
    rm = (rAB0 - Rm2) if coords == "s12mu_morse" else 1.
    Hl += SH_layers["H_beta"][rAB0, s0] * rm * beta2
    Hl += SH_layers["H_alphabeta"][rAB0, s0] * rm * alpha2 * beta2

    cf = SH_layers["c_beta2"] * rm**2 * beta2**2  # constant factor from _beta and _beta2 that can be directly applied to S
    if coords == "s12mu_morse":
        cf += 2. * SH_layers["meta"]["M_inv"] * beta2  # was left out there since it doesn't multiply rm
    Hl += SH_layers["S"][rAB0, s0] * cf

    H_tile += Hl * exps
    return S_tile, H_tile

def save_layers(label, rAB):
    global layers
    savefile = f"SHlayers_t{tMax}_delta{layers['meta']['delta']}{'_BO' if BO else ''}{'_' + label if label is not None else ''}_{bsize}_{rAB}.pkl"
    with open(savefile, "wb") as f:
        pickle.dump(layers, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved to: {savefile}")
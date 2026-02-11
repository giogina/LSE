import numpy as np
import pickle
from double_precision import *

from calc import cancellation_kappa

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

def print_H_asymmetry_ranked(
    H: np.ndarray,
    # S: np.ndarray,
    basis_idx: np.ndarray,
    *,
    top: int | None = None,
    min_abs: float = 0.0,
    normalize: bool = True,
    eps: float = 1e-300,
    file=None,
):
    """
    Print (i,j) pairs ranked by asymmetry between H[i,j] and H[j,i].

    Ranking score:
      - if normalize=True: |Hij - Hji| / (|Hij| + |Hji| + eps)
      - else:              |Hij - Hji|

    Prints: score, |diff|, Hij, Hji, i, j, basis_idx[i,:], basis_idx[j,:]

    Notes:
      - Only considers i<j (unique pairs).
      - If top is None, prints the full list (can be huge).
    """
    H = np.asarray(H)
    basis_idx = np.asarray(basis_idx)

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be square, got shape {H.shape}")
    n = H.shape[0]
    if basis_idx.shape[0] != n:
        raise ValueError(f"basis_idx must have {n} rows, got {basis_idx.shape[0]}")

    # Upper-triangle indices (unique unordered pairs)
    iu, ju = np.triu_indices(n, k=1)

    Hij = H[iu, ju]
    Hji = H[ju, iu]
    diff = Hij - Hji
    adiff = np.abs(diff)

    if normalize:
        denom = (np.abs(Hij) + np.abs(Hji) + eps)
        # score = adiff / denom
        score = np.abs(Hij)
    else:
        score = adiff

    # Filter
    mask = adiff >= float(min_abs)
    iu, ju = iu[mask], ju[mask]
    Hij, Hji, diff, adiff, score = Hij[mask], Hji[mask], diff[mask], adiff[mask], score[mask]

    # Sort strongest first
    order = np.argsort(score)[::-1]
    if top is not None:
        order = order[:int(top)]

    # Header
    print(
        "rank  score              |Hij-Hji|          Hij                Hji                i      j      basis_i -> basis_j",
        file=file,
    )

    for r, p in enumerate(order, start=1):
        i = int(iu[p]); j = int(ju[p])
        bi = basis_idx[i]
        bj = basis_idx[j]
        bi_str = "[" + ' '.join(f"{v:{2}d}" for v in bi[1:]) + "]"
        bj_str = "[" + ' '.join(f"{v:{2}d}" for v in bj[1:]) + "]"
        bdiff = "[" + ' '.join(f"{bj[n]-bi[n]:{2}d}" for n in [1, 2, 3, 4, 5]) + "]"
        print(
            f"{r:4d}  {score[p]: .9e}  {diff[p]: .9e}   {Hij[p]: .9e}   {Hji[p]: .9e}   {i:5d}  {j:5d}  {bi_str} -> {bj_str}  {bdiff} {bi[1]*bj[1]}",
            file=file,
        )

    print(f"\nTotal pairs printed: {len(order)} (out of {len(score)} passing min_abs={min_abs})", file=file)


def init_layers(coords, basis_idx, delta, M1M, M_inv, Fij, Fji, Fij_smol, Fji_smol, X, tmax, bo):
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
            "Fij_smol": Fij_smol,
            "Fji_smol": Fji_smol,
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
    # cancellation_kappa(BT, B, "BT @ B")
    Sl += Sp
    Hl_1 += BT @ A_1
    # Hl_1 += cancellation_kappa(BT, A_1, "BT @ A_1")
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
import numpy as np

def assemble_HS_multi_alpha(SH_layers, alphas, betas, Rms, coords, H=None, S=None):
    # N: basis size per block
    N = next(iter(SH_layers["S"].values())).shape[0]

    alphas = np.asarray(alphas, dtype=np.float64)
    betas  = np.asarray(betas,  dtype=np.float64)
    Rms    = np.asarray(Rms,    dtype=np.float64)

    na, nb = len(alphas), len(betas)
    n = na * nb
    M = n * N

    A  = np.repeat(alphas, nb)     # (n,)
    B  = np.tile(betas, na)        # (n,)
    RM = np.tile(Rms,   na)        # (n,)

    if S is None:
        S = np.zeros((M, M), dtype=np.float64)
    if H is None:
        H = np.zeros((M, M), dtype=np.float64)

    # 4D block views: (i, j, r, c)
    S4 = S.reshape(n, N, n, N).transpose(0, 2, 1, 3)
    H4 = H.reshape(n, N, n, N).transpose(0, 2, 1, 3)

    # --- cancellation audit (alpha-diagonal only) ---
    tiny = np.finfo(np.float64).tiny

    # abs-sum accumulators for alpha-diagonal stitched submatrices:
    # store as (na, nb, nb, N, N) matching S4/H4 block layout restricted to one alpha
    absS_a = np.zeros((na, nb, nb, N, N), dtype=np.float64)
    absH_a = np.zeros((na, nb, nb, N, N), dtype=np.float64)


    c_beta2 = float(SH_layers["c_beta2"])
    meta = SH_layers.get("meta", {})

    morse_exp = coords.endswith("_morse")
    morse_rm  = (coords == "s12mu_morse")  # only this one uses rm = (rAB0 - Rm)

    items = SH_layers["S"].items()
    # items = sorted(SH_layers["S"].items(), key=lambda kv: kv[0][1]*np.abs(kv[0][0]-1.45), reverse=True)  # decreasing s0, to add up the tiny tail pieces first. TODO: test if this makes a difference.
    for (rAB0, s0), S_layer in items:

        H1   = SH_layers["H_1"][rAB0, s0]
        Ha   = SH_layers["H_alpha"][rAB0, s0]
        Ha2  = SH_layers["H_alpha2"][rAB0, s0]
        Hb   = SH_layers["H_beta"][rAB0, s0]
        Hab  = SH_layers["H_alphabeta"][rAB0, s0]

        # ---- build Hj[j] = Hl(alpha2,beta2,Rm2) for all j (depends only on block-2) ----
        if morse_rm:
            rm = (rAB0 - RM)  # (n,)
        else:
            rm = 1.0

        Hj = np.empty((n, N, N), dtype=np.float64)
        # Build each Hj[j] with in-place axpy-style updates
        for j in range(n):
            aj = A[j]
            bj = B[j]
            rmj = rm[j] if morse_rm else 1.0

            cf = c_beta2 * (rmj * rmj) * (bj * bj)
            if morse_rm:
                cf += 2.0 * float(meta["M_inv"]) * bj

            out = H1.copy()
            out += Ha  * aj
            out += Ha2 * (aj * aj)
            out += Hb  * (rmj * bj)
            out += Hab * (rmj * aj * bj)

            out += S_layer * cf
            Hj[j] = out

        # ---- build exps[i,j] for this layer ----
        # alpha part (outer sum)
        exp_a = np.exp(-s0 * (A[:, None] + A[None, :]))  # (n,n)

        if morse_exp:
            # beta part is separable: exp(-B_i*(RM_i-rAB)^2) * exp(-B_j*(RM_j-rAB)^2)
            d2 = (RM - rAB0) ** 2
            eb = np.exp(-B * d2)  # (n,)
            exp_b = eb[:, None] * eb[None, :]  # (n,n)
            exps = exp_a * exp_b
        else:
            # exp(-(beta1+beta2)*rAB0)
            exp_b = np.exp(-rAB0 * (B[:, None] + B[None, :]))
            exps = exp_a * exp_b

        # ---- accumulate blocks without (i,j) Python loop ----
        # Update by columns j to avoid huge temporary (n,n,N,N)
        # S4[:, j] += exps[:, j][:,None,None] * S_layer
        # H4[:, j] += exps[:, j][:,None,None] * Hj[j]
        for j in range(n):
            fj = exps[:, j].reshape(n, 1, 1)          # (n,1,1)
            S4[:, j] += fj * S_layer                  # broadcast over (N,N)
            H4[:, j] += fj * Hj[j]                    # broadcast over (N,N)


            # --- cancellation audit update (alpha-diagonal only) ---
            ja = j // nb           # alpha index of block-column j
            jb = j % nb            # beta index within that alpha
            i0 = ja * nb
            i1 = i0 + nb           # rows (in block indices) that share this alpha

            # abs contribution for S and H on the alpha-diagonal submatrix
            fj_sub = exps[i0:i1, j].reshape(nb, 1, 1)     # (nb,1,1)
            abs_fj = np.abs(fj_sub)

            absS_a[ja][:, jb] += abs_fj * np.abs(S_layer)
            absH_a[ja][:, jb] += abs_fj * np.abs(Hj[j])

    blocks = tuple({"alpha": float(A[k]), "beta": float(B[k]), "Rm": float(RM[k])} for k in range(n))

    # print_H_asymmetry_ranked(H, SH_layers["meta"]["basis_idx"])

    # --- compute kappa stats per alpha (alpha-diagonal stitched blocks) ---
    kappa_report = []
    top_k = 20
    thresh = 2.0  # "lost digits" threshold: log10(kappa) > 3

    for ia in range(na):
        rs = ia * nb * N
        re = (ia + 1) * nb * N

        S_sub = S[rs:re, rs:re]
        H_sub = H[rs:re, rs:re]

        absS_sub = absS_a[ia].transpose(0, 2, 1, 3).reshape(nb * N, nb * N)
        absH_sub = absH_a[ia].transpose(0, 2, 1, 3).reshape(nb * N, nb * N)

        denomS = np.maximum(np.abs(S_sub), tiny)
        denomH = np.maximum(np.abs(H_sub), tiny)

        kappaS = absS_sub / denomS
        kappaH = absH_sub / denomH

        digS = np.log10(kappaS, where=(absS_sub > 0.0), out=np.full_like(kappaS, np.nan))
        digH = np.log10(kappaH, where=(absH_sub > 0.0), out=np.full_like(kappaH, np.nan))

        maskS = (absS_sub > 0.0) & np.isfinite(digS)
        maskH = (absH_sub > 0.0) & np.isfinite(digH)

        # "bad" entries count
        badS = maskS & (digS > thresh)
        badH = maskH & (digH > thresh)

        nS = int(np.count_nonzero(maskS))
        nH = int(np.count_nonzero(maskH))
        nBadS = int(np.count_nonzero(badS))
        nBadH = int(np.count_nonzero(badH))

        # helper to extract top-k indices
        def topk_entries(dig, mask, abs_sum, sum_mat, label):
            if not np.any(mask):
                return []

            vals = dig.copy()
            vals[~mask] = -np.inf

            k = min(top_k, int(np.count_nonzero(mask)))
            flat = vals.ravel()

            # argpartition for speed, then sort those k
            idx_part = np.argpartition(flat, -k)[-k:]
            idx_sorted = idx_part[np.argsort(flat[idx_part])[::-1]]

            out = []
            for idx in idx_sorted:
                v = float(flat[idx])
                if not np.isfinite(v):
                    continue
                r, c = np.unravel_index(idx, vals.shape)

                # decode within alpha-diagonal stitched block:
                # r = beta_i*N + row_in_block, c = beta_j*N + col_in_block
                bi, ri = divmod(r, N)
                bj, cj = divmod(c, N)

                out.append({
                    "lost_digits": v,
                    "global_rc": (rs + r, rs + c),
                    "local_rc": (r, c),
                    "beta_rc": (int(bi), int(bj), int(ri), int(cj)),
                    "sum_val": float(sum_mat[r, c]),
                    "abs_sum": float(abs_sum[r, c]),
                })
            return out

        topS = topk_entries(digS, maskS, absS_sub, S_sub, "S")
        topH = topk_entries(digH, maskH, absH_sub, H_sub, "H")

        kappa_report.append({
            "alpha": float(alphas[ia]),
            "S": {
                "mean_log10_kappa": float(np.nanmean(digS[maskS])) if np.any(maskS) else float("nan"),
                "worst_log10_kappa": float(np.nanmax(digS[maskS])) if np.any(maskS) else float("nan"),
                "n_entries": nS,
                "n_worse_than_3": nBadS,
                "frac_worse_than_3": (nBadS / nS) if nS else float("nan"),
                "top": topS,
            },
            "H": {
                "mean_log10_kappa": float(np.nanmean(digH[maskH])) if np.any(maskH) else float("nan"),
                "worst_log10_kappa": float(np.nanmax(digH[maskH])) if np.any(maskH) else float("nan"),
                "n_entries": nH,
                "n_worse_than_3": nBadH,
                "frac_worse_than_3": (nBadH / nH) if nH else float("nan"),
                "top": topH,
            },
        })

        # print a compact summary line
        a = float(alphas[ia])
        print(
            f"alpha={a:.6g}  "
            f"S: mean={kappa_report[-1]['S']['mean_log10_kappa']:.3g}, worst={kappa_report[-1]['S']['worst_log10_kappa']:.3g}, "
            f"> {thresh} digits={nBadS}/{nS} ({(nBadS / nS * 100 if nS else 0):.2f}%)   "
            f"H: mean={kappa_report[-1]['H']['mean_log10_kappa']:.3g}, worst={kappa_report[-1]['H']['worst_log10_kappa']:.3g}, "
            f"> {thresh} digits={nBadH}/{nH} ({(nBadH / nH * 100 if nH else 0):.2f}%)"
        )

        # print worst offenders (H only, usually what you care about)
        print("  Worst H entries (lost_digits, global(i,j), beta_i,beta_j,row,col, sum, abs_sum):")
        for t in topH[:top_k]:
            bi, bj, ri, cj = t["beta_rc"]
            gi, gj = t["global_rc"]
            print(
                f"    {t['lost_digits']:.2f}  ({gi},{gj})  "
                f"b=({bi},{bj}) rc=({ri},{cj})  "
                f"sum={t['sum_val']:.6e}  abs_sum={t['abs_sum']:.6e}"
            )


        t = topH[0]
        bi, bj, ri, cj = t["beta_rc"]
        acc = 0.
        acc_dd = dd_from(0.)
        for (rAB0, s0), S_layer in items:
            H1 = SH_layers["H_1"][rAB0, s0]
            Ha = SH_layers["H_alpha"][rAB0, s0]
            Ha2 = SH_layers["H_alpha2"][rAB0, s0]

            hh1 = H1[ri, cj] * np.exp(-2.*alphas[ia]*s0)
            hha = Ha[ri, cj] * alphas[ia] * np.exp(-2.*alphas[ia]*s0)
            hha2 = Ha2[ri, cj] * alphas[ia]**2 * np.exp(-2.*alphas[ia]*s0)
            acc += hh1+hha+hha2
            acc_dd = dd_add(acc_dd, dd_add(dd_add(dd_from(hh1), dd_from(hha)), dd_from(hha2)))
            print(f"{alphas[ia]}, {float(s0):.5f}, H1: {hh1:.3e}, Ha: {hha:.3e}, Ha2: {hha2:.3e}, acc: {acc:.3e}, delta acc_dd: {(acc-acc_dd[0]):.3e}")

    return H, S, StitchedBasis(N=N, blocks=blocks)

#
# def assemble_HS_multi_alpha(SH_layers, alphas, betas, Rms, coords, H=None, S=None):
#
#     N = next(iter(SH_layers["S"].values())).shape[0]
#     n = len(alphas)*len(betas)
#     if H is None:
#         S = np.zeros((n*N, n*N), dtype=np.float64)
#         H = np.zeros((n*N, n*N), dtype=np.float64)
#
#     blocks_list = []
#     for alpha in alphas:
#         for b, beta in enumerate(betas):
#             blocks_list.append({"alpha": alpha, "beta": beta, "Rm": Rms[b]})
#     blocks = tuple(blocks_list)
#     print(f"({n*N} functions)")
#
#     for (rAB0, s0), Sl in SH_layers["S"].items():
#         for i, b1 in enumerate(blocks):
#             r = slice(i * N, (i + 1) * N)
#             for j, b2 in enumerate(blocks):
#                 c = slice(j * N, (j + 1) * N)
#                 S_tile, H_tile = _assemble_HS_alphas_piece(SH_layers, b1["alpha"], b2["alpha"], b1["beta"], b2["beta"], b1["Rm"], b2["Rm"], coords, s0, rAB0)
#                 S[r, c] += S_tile
#                 H[r, c] += H_tile
#     return H, S, StitchedBasis(N=N, blocks=blocks)
#
# def _assemble_HS_alphas_piece(SH_layers, alpha1, alpha2, beta1, beta2, Rm1, Rm2, coords, s0, rAB0):
#     S_tile = np.zeros_like(next(iter(SH_layers["S"].values())), dtype=np.float64)
#     H_tile = np.zeros_like(S_tile, dtype=np.float64)
#
#     if coords.endswith("_morse"):
#         exps = np.exp(-(alpha1+alpha2) * s0 - beta1 * (Rm1 - rAB0) ** 2 - beta2 * (Rm2 - rAB0) ** 2)
#     else:
#         exps = np.exp(-(alpha1+alpha2) * s0 - (beta1+beta2) * rAB0)
#
#     S_tile = SH_layers["S"][rAB0, s0] * exps
#     Hl = np.zeros_like(H_tile)
#     Hl += SH_layers["H_1"][rAB0, s0]
#     Hl += SH_layers["H_alpha"][rAB0, s0] * alpha2
#     Hl += SH_layers["H_alpha2"][rAB0, s0] * alpha2 ** 2
#     rm = (rAB0 - Rm2) if coords == "s12mu_morse" else 1.
#     Hl += SH_layers["H_beta"][rAB0, s0] * rm * beta2
#     Hl += SH_layers["H_alphabeta"][rAB0, s0] * rm * alpha2 * beta2
#
#     cf = SH_layers["c_beta2"] * rm**2 * beta2**2  # constant factor from _beta and _beta2 that can be directly applied to S
#     if coords == "s12mu_morse":
#         cf += 2. * SH_layers["meta"]["M_inv"] * beta2  # was left out there since it doesn't multiply rm
#     Hl += SH_layers["S"][rAB0, s0] * cf
#
#     H_tile += Hl * exps
#     return S_tile, H_tile

def save_layers(label, rAB):
    global layers
    savefile = f"SHlayers_t{tMax}_delta{layers['meta']['delta']}{'_BO' if BO else ''}{'_' + label if label is not None else ''}_{bsize}_{rAB}.pkl"
    with open(savefile, "wb") as f:
        pickle.dump(layers, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved to: {savefile}")